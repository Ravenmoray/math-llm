"""Load a checkpoint and run autoregressive generation.

Usage:
  python scripts/sample.py --ckpt checkpoints/v2/step1000.pt \
    --prompt "Problem: What is 47 + 238?\n\nSolution:"

WARNING: while training is running on the same GPU, this will compete for VRAM.
Either stop training first, or pass --device cpu (slow but safe).
"""
import argparse, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.config import ModelConfig
from model.model import MathLM


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., -1, None]] = -float("inf")
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        sorted_logits[mask] = -float("inf")
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int, temperature: float,
             top_k: int, top_p: float, device: str, eos_id: int):
    enc = tok.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    max_ctx = model.cfg.max_seq_len

    for _ in range(max_new_tokens):
        cond = ids if ids.size(1) <= max_ctx else ids[:, -max_ctx:]
        logits, _ = model(cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if temperature == 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == eos_id:
            break
    return tok.decode(ids[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="Problem: What is 47 + 238?\n\nSolution:")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<|eos|>") or -1

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["cfg"].items() if k in ModelConfig.__dataclass_fields__})
    model = MathLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = model.to(device=args.device, dtype=dtype)
    model.train(mode=False)  # inference mode
    print(f"[loaded {cfg.param_count()/1e6:.0f}M params from {args.ckpt}, step={ckpt.get('step','?')}]\n")

    for i in range(args.n):
        out = generate(model, tok, args.prompt, args.max_new_tokens,
                       args.temperature, args.top_k, args.top_p, args.device, eos_id)
        print(f"=== sample {i+1}/{args.n} ===")
        print(out)
        print()


if __name__ == "__main__":
    main()
