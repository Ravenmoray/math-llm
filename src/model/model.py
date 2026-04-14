"""Llama-style decoder-only transformer.

- RoPE (precomputed)
- RMSNorm
- SwiGLU FFN
- Grouped-Query Attention (GQA)
- Uses torch.nn.functional.scaled_dot_product_attention (FlashAttention-2 kernel
  available automatically on Ampere+ GPUs in PyTorch 2.x).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # compute in fp32 for stability
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, base: float, device=None):
    """Returns (cos, sin) each of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)         # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    if device is not None:
        emb = emb.to(device)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D).  cos/sin: (T, D)."""
    D = x.size(-1)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    cos = cos[: x.size(-2)].to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin[: x.size(-2)].to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.hd = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)    # B,H,T,D
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)      # B,Hk,T,D
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: repeat K/V to match Q heads
        repeat = self.n_head // self.n_kv
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # SDPA (FlashAttention-2 on Ampere+)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_up   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class MathLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        if cfg.tie_embeddings:
            self.lm_head = None  # uses self.embed.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for output projections
        for n, p in self.named_parameters():
            if n.endswith("o_proj.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.gradient_checkpointing = False

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-100)
            return logits, loss
        return logits, None
