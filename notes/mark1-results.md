# Mark-1: Training Results & Findings

## What is Mark-1?

A 1.06B parameter decoder-only transformer trained from scratch on mathematics,
using a curriculum-ordered corpus of textbooks, proofs, and synthetic worked examples.

## Architecture
- 30 layers, d=1792, 28 heads, GQA (4 KV heads), SwiGLU, RoPE, RMSNorm
- 1,062M parameters, tied embeddings
- Custom 32k BPE tokenizer trained on the math corpus (3.21 bytes/token)
- ctx=2048, bf16

## Training
- **Hardware:** Single RTX 4060 Ti 16 GB, Ryzen 5 5600X, 76 GB RAM
- **Duration:** 60 hours (10k steps)
- **Throughput:** 3,038 tok/s steady
- **VRAM:** 6.71 GB
- **Optimizer:** AdamW 8-bit, peak LR 3e-4, cosine schedule with 500-step warmup
- **Batch:** micro-batch=2, grad-accum=16, effective batch=65k tokens/step
- **Tokens seen:** ~655M (16.7 epochs over 39M token corpus)

## Corpus (v1)
~40M tokens across 7 curriculum levels:

| Level | Source | Text |
|---|---|---|
| L1 | Synthetic arithmetic (700k worked examples) | 130 MB |
| L2 | OpenStax Prealgebra, Elem/Inter/College Algebra | 5 MB |
| L3 | OpenStax Precalc, Alg & Trig, Statistics | 5 MB |
| L4 | OpenStax Calculus 1/2/3 | 4 MB |
| L5 | Beezer Linear Algebra, Levin Discrete Math | 1.8 MB |
| L6 | Judson Abstract Algebra | 1.1 MB |
| L7 | ProofWiki (~37k theorems/proofs/definitions) | 16 MB |

## Training curve

| Step | Train loss | Val loss | Val ppl |
|---:|---:|---:|---:|
| 500 | 0.065 | 0.939 | 2.56 |
| 1000 | 0.049 | 0.709 | 2.03 |
| 2000 | 0.034 | **0.577** | **1.78** |
| 2500 | 0.030 | **0.563** | **1.76** ← best |
| 3000 | 0.026 | 0.590 | 1.80 |
| 5000 | 0.014 | 0.658 | 1.93 |
| 7000 | 0.005 | 0.823 | 2.28 |
| 10000 | 0.002 | 0.859 | 2.36 |

Val loss bottomed at step 2000-2500 then climbed — classic overfitting from
small corpus + too many epochs.

## Key finding: Mark-1 is a math problem GENERATOR, not a solver

### What it does
When given any input, Mark-1 generates new, original math problems with
complete worked solutions. It does NOT solve the problem you give it.

### Quality of generated problems
Tested with free generation (temperature=0.9) from step4000 checkpoint:

| Generated problem | Answer | Correct? |
|---|---|---|
| (9/10) × (7/9) simplify | 7/10 | Yes — correct GCD reduction |
| 9 + 11 × 9 | 108 | Yes — correct PEMDAS |
| 10 + 6 candies word problem | 16 | Yes |
| 8 + 858 column addition | 866 | Yes — correct carries |
| 56 - 45 | 11 | Yes |

**5/6 samples mathematically correct**, including intermediate steps.

### What it cannot do
- Solve a problem you give it (0% accuracy across all checkpoints)
- Follow instructions of any kind
- Answer questions about math concepts
- Write proofs on demand

### Why
Standard causal LM pretraining on a corpus of independent documents. Each
training example was a complete Problem→Solution→Answer unit. The model
learned P(next_token | context) over the document distribution — it learned
to produce documents that look like the training data, not to map
Problem → Solution for a given input.

This is expected and normal for base (non-instruction-tuned) language models.
GPT-3 base exhibited the same behavior before RLHF/SFT.

## What Mark-1 proved
1. The training infrastructure works end-to-end on consumer hardware
2. 1B params fits comfortably in 16 GB with bf16 + AdamW8bit + grad-ckpt
3. The model learned real arithmetic (not just pattern matching)
4. The custom math tokenizer works well (3.21 bytes/token)
5. Curriculum-ordered data (L1→L7) ingestion pipeline is solid
6. 40M tokens is ~250× too small for 1B params (Chinchilla)

## Paths forward for Mark-2
1. **SFT stage** — fine-tune step2000.pt on instruction-following math data
   (MetaMathQA, GSM8K, NuminaMath) to teach problem→solution mapping
2. **More data** — expand corpus to 5-20B tokens (OpenWebMath, synthetic SymPy
   at scale, arXiv math) before pretraining again
3. **Smaller model** — 150-300M params would be better matched to current corpus
4. **Lean into generation** — Mark-1 as a synthetic data generator for Mark-2
