"""Train a math-tuned BPE tokenizer on the assembled corpus.

Design:
  - ByteLevel BPE (Llama-style; robust to any input)
  - 32k vocab
  - Pre-tokenizer keeps LaTeX commands (\frac, \sum, ...) as atomic-ish units
    by splitting on whitespace + punctuation but preserving backslash sequences
  - Reserved special tokens for training/inference
"""
import glob, json, os
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors, Regex

CORPUS_GLOB = "data/tokenized/corpus_v1/train-*.jsonl"
OUT_DIR = Path("artifacts/tokenizer"); OUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_SIZE = 32000

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


def text_iter():
    n = 0
    for f in sorted(glob.glob(CORPUS_GLOB)):
        with open(f) as fh:
            for line in fh:
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                yield r["text"]
                n += 1
                if n % 50000 == 0:
                    print(f"  [{n} docs fed to trainer]", flush=True)


def main():
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Pre-tokenizer: byte-level, but split on whitespace first so LaTeX tokens
    # like \frac, \sum, \int have a fair shot at becoming single merges.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(r"\\[a-zA-Z]+"),
                             behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training tokenizer, vocab={VOCAB_SIZE} ...")
    tok.train_from_iterator(text_iter(), trainer=trainer, length=None)

    # Post-processor: bos + text + eos for training
    bos_id = tok.token_to_id("<|bos|>")
    eos_id = tok.token_to_id("<|eos|>")
    tok.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[("<|bos|>", bos_id), ("<|eos|>", eos_id)],
    )

    tok_path = OUT_DIR / "tokenizer.json"
    tok.save(str(tok_path))
    print(f"Saved {tok_path}")

    # Quick evaluation: bytes/token on a small val sample
    val_files = sorted(glob.glob("data/tokenized/corpus_v1/val-*.jsonl"))
    total_bytes, total_tokens = 0, 0
    samples = []
    for f in val_files[:1]:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= 2000: break
                r = json.loads(line)
                t = r["text"]
                enc = tok.encode(t)
                total_bytes += len(t.encode("utf-8"))
                total_tokens += len(enc.ids)
                if len(samples) < 3 and 200 < len(t) < 500:
                    samples.append((t, enc.tokens[:40]))
    bpt = total_bytes / max(total_tokens, 1)
    print(f"\nVal sample tokenization: bytes/token = {bpt:.2f}  "
          f"(Llama3 typical ~3.9 on English; lower = worse for our corpus)")
    print(f"Total tokens on 2000 val docs: {total_tokens:,}")
    for i, (t, toks) in enumerate(samples):
        print(f"\n--- sample {i} ---\nINPUT: {t[:200]!r}\nFIRST TOKENS: {toks}")


if __name__ == "__main__":
    main()
