"""Tokenize corpus to a flat uint16 binary for efficient training.

Layout: one big file per split (train.bin / val.bin) with document boundaries
marked by bos/eos as defined in the tokenizer post-processor.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--corpus", default="data/tokenized/corpus_v1")
    ap.add_argument("--out",    default="data/tokenized/corpus_v1")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tok)
    vocab_size = tok.get_vocab_size()
    assert vocab_size < 65535, "using uint16 but vocab is too large"

    for split in ("train", "val"):
        files = sorted(glob.glob(f"{args.corpus}/{split}-*.jsonl"))
        total = 0
        out_path = Path(args.out) / f"{split}.bin"
        # First pass: count + write in chunks via a list-of-arrays
        buffers = []
        for f in files:
            with open(f) as fh:
                texts = [json.loads(l)["text"] for l in fh]
            encs = tok.encode_batch(texts)
            for e in encs:
                ids = np.array(e.ids, dtype=np.uint16)
                buffers.append(ids)
                total += len(ids)
            print(f"  {f}: running total = {total:,} tokens", flush=True)
        arr = np.concatenate(buffers) if buffers else np.array([], dtype=np.uint16)
        arr.tofile(out_path)
        print(f"Wrote {out_path}: {arr.shape[0]:,} tokens ({out_path.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
