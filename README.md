# math-llm

From-scratch decoder-only LM trained on mathematics only. Phi-style methodology: start with textbook-quality curated data, evaluate, expand only if needed.

## Target
- ~500M params, Llama-style (RoPE, RMSNorm, SwiGLU, GQA)
- 24 layers, d=1280, 20 heads, GQA groups=5, ctx=2048, vocab=32k
- bf16 + AdamW 8-bit + FlashAttention-2 + gradient checkpointing
- Single RTX 4060 Ti 16 GB

## Phases
1. **Books-only corpus** (current) — open-license math textbooks, ProofWiki, Wikipedia math portal.
2. Tokenizer training (BPE 32k, math-tuned).
3. Pretraining on books.
4. Evaluate; decide whether to add OpenWebMath / arXiv / synthetic SymPy.
5. Reasoning SFT on MetaMathQA / OpenMathInstruct / NuminaMath.

## Layout
```
data/{raw,clean,tokenized}/   # gitignored, large
src/                          # model, training, data code
configs/                      # model + training configs
scripts/                      # ingest + eval scripts
checkpoints/ logs/            # gitignored
notes/                        # design notes, data-quality reports
```
