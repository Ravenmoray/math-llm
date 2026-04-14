"""Assemble, dedup, shuffle, shard the final pretraining corpus.

Strategy:
  - Exact dedup via SHA256 on normalized text (lowercase, collapsed whitespace)
  - Per-source min-length filter
  - 98/2 train/val split (document level)
  - Curriculum-weighted sampling available later; here we just preserve
    per-record 'level' metadata for downstream samplers.
  - Sharded JSONL output.
"""
import argparse, glob, hashlib, json, random, re
from pathlib import Path
from collections import defaultdict

SOURCES = [
    ("data/clean/L1_synthetic/*.jsonl", "synthetic"),
    ("data/clean/openstax/*.jsonl",     "openstax"),
    ("data/clean/aim/*.jsonl",          "aim"),
    ("data/clean/proofwiki/*.jsonl",    "proofwiki"),
]

def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokenized/corpus_v1/")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--shard-docs", type=int, default=50_000)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    seen = set()
    records = []
    per_src = defaultdict(lambda: {"in": 0, "kept": 0, "dedup_drop": 0, "short_drop": 0})

    for pat, src_tag in SOURCES:
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    try: r = json.loads(line)
                    except json.JSONDecodeError: continue
                    per_src[src_tag]["in"] += 1
                    t = r.get("text", "")
                    if len(t) < args.min_chars:
                        per_src[src_tag]["short_drop"] += 1; continue
                    h = hashlib.sha256(norm(t).encode("utf-8", errors="ignore")).hexdigest()
                    if h in seen:
                        per_src[src_tag]["dedup_drop"] += 1; continue
                    seen.add(h)
                    # Slim down record
                    slim = {"text": t, "source": src_tag,
                            "level": r.get("level", 0),
                            "subtype": r.get("subtype") or r.get("type") or r.get("book_slug")}
                    records.append(slim)
                    per_src[src_tag]["kept"] += 1

    rng.shuffle(records)
    n_val = int(len(records) * args.val_frac)
    val = records[:n_val]; train = records[n_val:]

    def write_shards(recs, prefix):
        shard_idx, count = 0, 0
        total_chars = 0
        fout = open(out / f"{prefix}-{shard_idx:04d}.jsonl", "w")
        for r in recs:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1; total_chars += len(r["text"])
            if count >= args.shard_docs:
                fout.close(); shard_idx += 1; count = 0
                fout = open(out / f"{prefix}-{shard_idx:04d}.jsonl", "w")
        fout.close()
        return shard_idx + 1, total_chars

    train_shards, train_chars = write_shards(train, "train")
    val_shards, val_chars = write_shards(val, "val")

    print("\n=== Corpus assembly ===")
    for src, s in per_src.items():
        print(f"  {src:>12}: in={s['in']:>7}  kept={s['kept']:>7}  "
              f"short_drop={s['short_drop']:>6}  dedup_drop={s['dedup_drop']:>6}")
    print(f"\nTotal kept: {len(records):,}")
    print(f"Train: {len(train):,} docs, {train_shards} shards, {train_chars/1e6:.1f} MB")
    print(f"Val:   {len(val):,} docs, {val_shards} shards, {val_chars/1e6:.1f} MB")

    meta = {
        "total_docs": len(records), "train_docs": len(train), "val_docs": len(val),
        "train_chars": train_chars, "val_chars": val_chars,
        "per_source": dict(per_src),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
