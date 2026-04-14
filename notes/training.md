# Training plan and tradeoffs

## Hardware
- RTX 4060 Ti 16 GB (Ada, sm_89), Ryzen 5 5600X, 76 GB RAM
- Single-GPU only; no distributed training

## Model size vs VRAM (empirically measured)
All measurements: ctx=2048, micro-batch=2, bf16 + AdamW8bit + grad-checkpointing.

| d_model | layers | params | VRAM | tok/s |
|---:|---:|---:|---:|---:|
| 1280 | 24 | 454M | 2.9 GB | ~7,100 |
| 1536 | 28 | 743M | 4.8 GB | ~4,000 |
| 1792 | 30 | **1.06B** ← default | **6.7 GB** | ~2,900 |
| 2048 | 32 | 1.51B | 9.5 GB | ~2,000 |

Even 1.5B fits comfortably. The binding constraint is **training throughput vs corpus size**, not VRAM.

## Tokens-to-train
Current corpus (L1 synthetic + OpenStax + AIM + ProofWiki): **~40M tokens**.
Chinchilla-optimal for each size:

| Params | Chinchilla tokens | Wall-clock @ measured tok/s |
|---:|---:|---:|
| 454M | 9B | ~15 days |
| 743M | 15B | ~43 days |
| 1.06B | 21B | ~84 days |
| 1.5B | 30B | ~175 days |

Given the data we actually have (40M tokens), any of these is severely data-limited
unless we expand the corpus (OpenWebMath + arXiv + synthetic SymPy at scale would
push us to 20–50B tokens easily).

## Recommended runs

**v1 (pipeline validation, already run):** 30 steps, 1 GB tokens seen. Confirms loss
decreases and the pipeline works end-to-end.

**v2 (first real training):** 10k steps @ micro-batch=2, grad-accum=16, ctx=2048 →
batch = 64k tokens/step → 640M tokens total → ~1.8 epochs over current corpus.
Wall-clock ~2.5 days for 1.06B; ~1 day for 454M. Expect meaningful arithmetic ability,
rudimentary algebra, some proof-style pattern matching.

**v3 (full training):** Expand corpus (OpenWebMath + synthetic SymPy @ scale),
target ~10B tokens total, ~2 weeks wall-clock for 1B model.

## Recommended next-data investments
1. **Synthetic SymPy scale-up** — extend the L1 generator to L2 (algebra), L3
   (trig, precalc identities), L4 (derivatives, integrals via SymPy). Generate
   1-5B tokens of curriculum synthetic. Cheap, high-signal, diverse.
2. **OpenWebMath** (~14.7B tokens) — Common Crawl math pages. The standard
   math-LM pretraining corpus.
3. **arXiv math LaTeX sources** (subset of RedPajama or direct) — L6+ graduate
   material.
4. **Proof-Pile-2** — ArXiv + AlgebraicStack (Lean, Isabelle, Coq proofs).

## Launch commands

Smoke test (5 steps):
```
.venv/bin/python src/train/train.py --max-steps 5 --val-every 999 --save-every 999 --log-every 1
```

v2 real run:
```
.venv/bin/python src/train/train.py \
  --max-steps 10000 --val-every 500 --save-every 1000 --log-every 50 \
  --seq-len 2048 --micro-batch 2 --grad-accum 16 \
  --warmup-steps 500 --peak-lr 3e-4 --min-lr 3e-5
```
