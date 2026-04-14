"""Ingest OpenStax math book source from GitHub.

Clones the osbooks-*-bundle repos, walks their CNXML modules, and emits
JSONL with cleaned markdown + LaTeX per module.

Usage:
    python scripts/ingest_openstax.py --out data/clean/openstax/ --raw data/raw/openstax/
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.openstax.cnxml import cnxml_to_text

# (github_repo, curriculum_level, tags)
REPOS = [
    ("osbooks-prealgebra-bundle",         2, ["prealgebra", "algebra"]),
    ("osbooks-college-algebra-bundle",    2, ["college-algebra"]),
    ("osbooks-precalculo",                3, ["precalculus"]),  # spanish? check below
    ("osbooks-calculus-bundle",           4, ["calculus"]),
    ("osbooks-introductory-statistics-bundle", 3, ["statistics"]),
    ("osbooks-statistics",                3, ["statistics"]),
    # ("osbooks-calculo-bundle",          4, ["calculus-es"]),  # spanish — skip for english-only v1
]

COLL_NS = "http://cnx.rice.edu/collxml"
MD_NS = "http://cnx.rice.edu/mdml"


def clone_or_update(repo, raw_dir):
    path = raw_dir / repo
    if path.exists():
        subprocess.run(["git", "-C", str(path), "pull", "-q"], check=False)
    else:
        url = f"https://github.com/openstax/{repo}.git"
        subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(path)], check=True)
    return path


def parse_collection(coll_path):
    """Return (book_title, list of (module_id, chapter_title) in reading order)."""
    tree = etree.parse(str(coll_path))
    root = tree.getroot()
    title_el = root.find(f".//{{{MD_NS}}}title")
    book_title = title_el.text if title_el is not None else coll_path.stem

    entries = []
    def walk(node, chapter=None):
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "subcollection":
                t = child.find(f".//{{{MD_NS}}}title")
                new_chapter = t.text if t is not None else chapter
                content = child.find(f"{{{COLL_NS}}}content")
                if content is not None:
                    walk(content, new_chapter)
            elif tag == "module":
                mid = child.get("document")
                if mid:
                    entries.append((mid, chapter))
            elif tag == "content":
                walk(child, chapter)
    content_el = root.find(f"{{{COLL_NS}}}content")
    if content_el is not None:
        walk(content_el)
    return book_title, entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/openstax/")
    ap.add_argument("--out", default="data/clean/openstax/")
    ap.add_argument("--min-chars", type=int, default=400, help="skip modules shorter than this")
    args = ap.parse_args()

    raw_dir = Path(args.raw); raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"modules_ok": 0, "modules_skipped_short": 0, "modules_failed": 0,
             "chars_total": 0, "books": {}}

    for repo, level, tags in REPOS:
        print(f"\n[{repo}]")
        try:
            repo_path = clone_or_update(repo, raw_dir)
        except subprocess.CalledProcessError as e:
            print(f"  clone failed: {e}"); continue

        colls = list((repo_path / "collections").glob("*.collection.xml"))
        for coll in colls:
            book_title, entries = parse_collection(coll)
            book_slug = coll.stem.replace(".collection", "")
            book_out = out_dir / f"{book_slug}.jsonl"
            n_ok = 0; n_chars = 0
            with open(book_out, "w") as fout:
                for mid, chapter in entries:
                    mod_path = repo_path / "modules" / mid / "index.cnxml"
                    if not mod_path.exists():
                        stats["modules_failed"] += 1
                        continue
                    try:
                        title, body = cnxml_to_text(mod_path)
                    except Exception as e:
                        stats["modules_failed"] += 1
                        print(f"  parse failed for {mid}: {e}")
                        continue
                    if len(body) < args.min_chars:
                        stats["modules_skipped_short"] += 1
                        continue
                    rec = {
                        "source": "openstax",
                        "book": book_title,
                        "book_slug": book_slug,
                        "level": level,
                        "tags": tags,
                        "chapter": chapter,
                        "section_title": title,
                        "module_id": mid,
                        "text": body,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok += 1; n_chars += len(body)
                    stats["modules_ok"] += 1
                    stats["chars_total"] += len(body)
            print(f"  {book_slug}: {n_ok} modules, {n_chars/1e6:.2f} MB")
            stats["books"][book_slug] = {"modules": n_ok, "chars": n_chars}

    print("\n=== SUMMARY ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
