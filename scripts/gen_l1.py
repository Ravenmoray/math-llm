"""Generate the L1 synthetic arithmetic corpus.

Usage:
    python scripts/gen_l1.py --examples 700000 --out data/clean/L1_synthetic/ --seed 0
"""
import argparse, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.l1_synthetic import arithmetic as A
from data.l1_synthetic import fractions as F
from data.l1_synthetic import percent_order as P
from data.l1_synthetic import word_problems as W

# Sampling weights across problem types. Tune later based on eval.
GENERATORS = [
    (A.gen_addition,       0.15),
    (A.gen_subtraction,    0.12),
    (A.gen_multiplication, 0.13),
    (A.gen_division,       0.10),
    (F.gen_fraction_add,   0.08),
    (F.gen_fraction_mul,   0.07),
    (P.gen_percent,        0.07),
    (P.gen_order_ops,      0.08),
    (W.gen_word_add,       0.05),
    (W.gen_word_sub,       0.05),
    (W.gen_word_mul,       0.05),
    (W.gen_word_div,       0.05),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=700_000)
    ap.add_argument("--out", default="data/clean/L1_synthetic/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=50_000)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    fns, weights = zip(*GENERATORS)

    shard_idx, count_in_shard, total_chars = 0, 0, 0
    fout = open(out_dir / f"shard_{shard_idx:04d}.jsonl", "w")
    try:
        for i in range(args.examples):
            fn = rng.choices(fns, weights=weights, k=1)[0]
            text, meta = fn(rng)
            total_chars += len(text)
            fout.write(json.dumps({"text": text, "level": 1, **meta}) + "\n")
            count_in_shard += 1
            if count_in_shard >= args.shard_size:
                fout.close()
                shard_idx += 1
                count_in_shard = 0
                fout = open(out_dir / f"shard_{shard_idx:04d}.jsonl", "w")
            if (i + 1) % 50_000 == 0:
                print(f"  {i+1:>8}/{args.examples}  ({total_chars/1e6:.1f} MB)", flush=True)
    finally:
        fout.close()
    print(f"Done. {args.examples} examples, ~{total_chars/1e6:.1f} MB across {shard_idx+1} shards.")

if __name__ == "__main__":
    main()
