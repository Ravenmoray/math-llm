"""Ingest AIM Open Textbook Initiative titles with PreTeXt source on GitHub.

Prioritized titles (L5-L6 bridge to proofs):
  - Judson, Abstract Algebra: Theory and Applications   (L6)
  - Beezer, A First Course in Linear Algebra            (L5)
  - Levin, Discrete Mathematics: An Open Introduction   (L5)

All three are PreTeXt, with math already in LaTeX (in <m>/<me>/<md> tags).
"""
import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.pretext.parser import parse_file

BOOKS = [
    # (github repo, slug, book title, level, tags, source subdir)
    ("twjudson/aata",          "aata-judson",       "Abstract Algebra: Theory and Applications (Judson)",   6, ["abstract-algebra", "proofs"], "src"),
    ("rbeezer/fcla",           "fcla-beezer",       "A First Course in Linear Algebra (Beezer)",            5, ["linear-algebra", "proofs"],  "src"),
    ("oscarlevin/discrete-book","dmoi-levin",       "Discrete Mathematics: An Open Introduction (Levin)",   5, ["discrete-math", "proofs"],   "source"),
]


def clone_or_update(repo, raw_dir):
    path = raw_dir / repo.split("/")[-1]
    if path.exists():
        subprocess.run(["git", "-C", str(path), "pull", "-q"], check=False)
    else:
        url = f"https://github.com/{repo}.git"
        subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(path)], check=True)
    return path


# Files to skip (meta, not content)
SKIP_NAMES = {"bookinfo.xml", "colophon.xml", "copyright.xml", "preamble.xml",
              "frontmatter.xml", "backmatter.xml", "biblio.xml", "main.xml",
              "index.xml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/aim/")
    ap.add_argument("--out", default="data/clean/aim/")
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    raw_dir = Path(args.raw); raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    grand_total_chars = 0
    for repo, slug, title, level, tags, subdir in BOOKS:
        print(f"\n[{slug}] {title}")
        try:
            repo_path = clone_or_update(repo, raw_dir)
        except subprocess.CalledProcessError as e:
            print(f"  clone failed: {e}"); continue

        src = repo_path / subdir
        if not src.exists():
            # try alternatives
            for alt in ("src", "source", "ptx", "src/ptx"):
                if (repo_path / alt).exists():
                    src = repo_path / alt
                    break
        if not src.exists():
            print(f"  no source dir; skipping"); continue

        files = sorted(list(src.glob("*.xml")) + list(src.glob("*.ptx")))
        n_ok = 0; n_chars = 0
        out_path = out_dir / f"{slug}.jsonl"
        with open(out_path, "w") as fout:
            for f in files:
                if f.name in SKIP_NAMES: continue
                try:
                    _, body = parse_file(f)
                except Exception as e:
                    print(f"  parse failed {f.name}: {e}")
                    continue
                if not body or len(body) < args.min_chars:
                    continue
                rec = {
                    "source": "aim",
                    "book": title,
                    "book_slug": slug,
                    "level": level,
                    "tags": tags,
                    "file": f.name,
                    "text": body,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1; n_chars += len(body)
        grand_total_chars += n_chars
        print(f"  {n_ok} files, {n_chars/1e6:.2f} MB")

    print(f"\n=== Total AIM text: {grand_total_chars/1e6:.2f} MB ===")


if __name__ == "__main__":
    main()
