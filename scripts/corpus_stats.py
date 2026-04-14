"""Corpus quality review. Emits aggregate stats + samples per source."""
import json, glob, re, random, sys
from collections import Counter, defaultdict
from pathlib import Path

random.seed(0)

SRCS = {
    "L1_synthetic": "data/clean/L1_synthetic/*.jsonl",
    "openstax":     "data/clean/openstax/*.jsonl",
    "aim":          "data/clean/aim/*.jsonl",
    "proofwiki":    "data/clean/proofwiki/*.jsonl",
}

def iter_recs(glob_pat):
    for f in sorted(glob.glob(glob_pat)):
        with open(f) as fh:
            for line in fh:
                try: yield json.loads(line)
                except json.JSONDecodeError: pass

MATH_INLINE = re.compile(r"\$[^$]+\$")
MATH_DISPLAY = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)

report_lines = []
def out(s=""): report_lines.append(s); print(s)

out("# Corpus Quality Review\n")
grand = {"docs": 0, "chars": 0}

for src, pat in SRCS.items():
    n_docs = 0; n_chars = 0; n_math_i = 0; n_math_d = 0
    lens = []
    by_level = Counter()
    by_type = Counter()
    samples = []
    for r in iter_recs(pat):
        n_docs += 1
        t = r.get("text", "")
        n_chars += len(t); lens.append(len(t))
        n_math_i += len(MATH_INLINE.findall(t))
        n_math_d += len(MATH_DISPLAY.findall(t))
        by_level[r.get("level", "?")] += 1
        by_type[r.get("type", r.get("subtype", "?"))] += 1
        if len(samples) < 2 and len(t) > 400:
            samples.append(r)

    if n_docs == 0:
        out(f"## {src}: (no data)\n"); continue
    lens.sort()
    p50 = lens[len(lens)//2]; p90 = lens[int(len(lens)*0.9)]
    out(f"## {src}")
    out(f"- docs: {n_docs:,}")
    out(f"- chars: {n_chars/1e6:.1f} MB  (~{n_chars//4:,} tokens est.)")
    out(f"- doc length: median={p50}, p90={p90}, max={max(lens)}")
    out(f"- inline math `$...$` occurrences: {n_math_i:,}  ({n_math_i/n_docs:.1f}/doc)")
    out(f"- display math `$$...$$` occurrences: {n_math_d:,}")
    out(f"- by level: {dict(by_level.most_common())}")
    out(f"- by type/subtype (top 8): {dict(by_type.most_common(8))}")
    out("")
    grand["docs"] += n_docs; grand["chars"] += n_chars

out("## Grand Total")
out(f"- docs: {grand['docs']:,}")
out(f"- chars: {grand['chars']/1e6:.1f} MB")
out(f"- tokens (char/4 est.): ~{grand['chars']//4/1e6:.1f}M")

Path("notes/corpus-review.md").write_text("\n".join(report_lines))
print("\nwrote notes/corpus-review.md")
