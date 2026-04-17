# math-llm

From-scratch decoder-only LM trained exclusively on mathematics. Phi-style methodology: curriculum-ordered textbook-quality data, evaluate, iterate.

## Mark-1 (current)

**1.06B parameter math problem generator.** Trained from scratch on a 40M token
curriculum corpus (synthetic arithmetic → textbooks → proofs). Generates
mathematically correct worked examples (~83% accuracy) but does not solve
given problems — base LM behavior, no instruction tuning yet.

See [notes/mark1-results.md](notes/mark1-results.md) for full results.

### Architecture
- Llama-style: 30 layers, d=1792, 28 heads, GQA (4 KV), SwiGLU, RoPE, RMSNorm
- 1,062M params, ctx=2048, custom 32k BPE tokenizer
- bf16 + AdamW 8-bit + FlashAttention-2 + gradient checkpointing

### Training
- Single RTX 4060 Ti 16 GB — 60 hours, 3,038 tok/s, 6.71 GB VRAM
- Best checkpoint: step2000 (val ppl 1.76)

### Corpus (v1)
7-level curriculum: L1 synthetic arithmetic → L2-L4 OpenStax textbooks →
L5-L6 AIM proof-bridge texts → L7 ProofWiki. ~40M tokens total.

## Layout
```
src/model/          # Llama-style decoder (config.py, model.py)
src/train/          # training loop (bf16, AdamW8bit, cosine LR)
src/data/           # ingest parsers (CNXML, PreTeXt, MediaWiki, synthetic)
scripts/            # generation, evaluation, corpus assembly
artifacts/          # tokenizer
notes/              # results, training plans, data-source docs
data/               # gitignored — raw/clean/tokenized corpus
checkpoints/ logs/  # gitignored
```
