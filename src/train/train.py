"""Training loop for MathLM.

- bf16 mixed precision (native dtype, no GradScaler needed)
- AdamW 8-bit (bitsandbytes) for ~4x optimizer memory savings
- Gradient checkpointing
- Gradient accumulation
- Cosine LR with warmup
- Packed-sequence streaming from .bin (uint16 flat)
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.config import ModelConfig
from model.model import MathLM


def get_batch(bin_path: str, batch_size: int, seq_len: int, device: str):
    """Uniformly sample a batch of (x, y) from a flat uint16 file, memmapped."""
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = len(mm)
    ix = np.random.randint(0, n - seq_len - 1, size=batch_size)
    x = np.stack([mm[i : i + seq_len].astype(np.int64) for i in ix])
    y = np.stack([mm[i + 1 : i + 1 + seq_len].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


def cosine_lr(step, warmup, total, peak, minimum):
    if step < warmup:
        return peak * step / warmup
    if step >= total:
        return minimum
    progress = (step - warmup) / (total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def compute_val_loss(model, bin_path, batch_size, seq_len, iters, device):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(bin_path, batch_size, seq_len, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/tokenized/corpus_v1")
    ap.add_argument("--ckpt-dir", default="checkpoints/v1")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--peak-lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-iters", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=False)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = MathLM(cfg).to(device=device, dtype=torch.bfloat16)
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, ctx={args.seq_len}, bf16")

    if args.compile:
        model = torch.compile(model)

    try:
        import bitsandbytes as bnb
        opt_cls = bnb.optim.AdamW8bit
        opt_name = "AdamW8bit"
    except Exception as e:
        print(f"(bitsandbytes unavailable: {e}; falling back to fp32 AdamW)")
        opt_cls = torch.optim.AdamW
        opt_name = "AdamW"

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = opt_cls(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.peak_lr, betas=(0.9, 0.95), eps=1e-8,
    )
    print(f"Optimizer: {opt_name}, peak_lr={args.peak_lr}, wd={args.weight_decay}")

    train_bin = f"{args.data_dir}/train.bin"
    val_bin   = f"{args.data_dir}/val.bin"

    model.train()
    t0 = time.time()
    running_loss = 0.0; running_n = 0
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.peak_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_bin, args.micro_batch, args.seq_len, device)
            _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()
            running_loss += loss.item(); running_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            avg = running_loss / max(running_n, 1)
            dt = time.time() - t0
            tps = step * args.grad_accum * args.micro_batch * args.seq_len / dt
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"step {step:>5}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"vram {vram:.2f}GB  tok/s {tps:.0f}  elapsed {dt:.0f}s", flush=True)
            running_loss = 0.0; running_n = 0

        if step % args.val_every == 0 or step == args.max_steps:
            vl = compute_val_loss(model, val_bin, args.micro_batch, args.seq_len, args.val_iters, device)
            print(f"  >>> VAL loss {vl:.4f}  ppl {math.exp(vl):.2f}", flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}
            torch.save(ckpt, Path(args.ckpt_dir) / f"step{step}.pt")
            print(f"  saved checkpoint step{step}.pt", flush=True)


if __name__ == "__main__":
    main()
