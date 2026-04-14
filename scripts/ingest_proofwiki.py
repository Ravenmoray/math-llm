"""Ingest the ProofWiki MediaWiki XML dump.

Keeps:
  - ns=0   (main: theorems, proofs, results)          ~47k pages
  - ns=100 (Axiom:)                                    ~400 pages
  - ns=102 (Definition:)                              ~30k pages

Skips user/talk/file/category/template/book-ref/symbol pages.

Wikitext -> markdown+LaTeX:
  - [[X|y]] -> y ;  [[X]] -> X ;  [[Definition:X|y]] -> y
  - == H == -> ## H
  - leading ':$...$' lines -> '$$...$$' display math
  - {{qed}} -> ∎
  - other templates -> dropped (citations, book-refs, nav)
  - <onlyinclude>, <includeonly> tags -> stripped (content kept)
  - HTML tags -> dropped
"""
import argparse, gzip, json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mwparserfromhell as mw

MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"
KEEP_NAMESPACES = {"0", "100", "102"}  # main, Axiom, Definition


def clean_wikitext(text):
    """Apply structural cleanups before mwparser processing."""
    # Protect display math lines (':$...$' at start of line) -> placeholder
    display_math = []
    def _dm(m):
        display_math.append(m.group(1))
        return f"\x00DM{len(display_math)-1}\x00"
    text = re.sub(r"^[ \t]*:[ \t]*\$([^\n$]+?)\$[ \t]*$", _dm, text, flags=re.MULTILINE)
    # Some display math uses \ds (displaystyle) or \begin{align}
    return text, display_math


def wiki_to_text(wikitext):
    """Convert MediaWiki wikitext to clean markdown + LaTeX."""
    text, display_math = clean_wikitext(wikitext)

    code = mw.parse(text)

    # Replace templates: qed -> ∎, others -> ""
    for tpl in list(code.ifilter_templates(recursive=True)):
        try:
            name = str(tpl.name).strip().lower()
        except Exception:
            name = ""
        if name in ("qed", "qedb", "qed_box"):
            code.replace(tpl, " ∎ ")
        else:
            try:
                code.remove(tpl)
            except Exception:
                pass

    # Replace wikilinks with display text (or title for Definition: links)
    for link in list(code.ifilter_wikilinks(recursive=True)):
        try:
            if link.text:
                replacement = str(link.text)
            else:
                title = str(link.title)
                # strip "Definition:" / "Axiom:" prefixes
                if ":" in title and title.split(":", 1)[0] in ("Definition", "Axiom", "Symbols"):
                    title = title.split(":", 1)[1]
                replacement = title
            code.replace(link, replacement)
        except Exception:
            try: code.remove(link)
            except Exception: pass

    # Strip HTML tags, keep content (strip_code does this)
    result = code.strip_code(normalize=True, collapse=True, keep_template_params=False)

    # Fix headings: == H == -> ## H  (wikitext headings are already cleaned by strip_code
    # but we want markdown-style). Actually strip_code leaves them bare, so restore:
    # We'll detect lines that were "== X ==" style via the original text pattern... simpler:
    # handle in post-processing if needed. Most useful restoration:
    # Actually strip_code converts "== X ==" to "X" (plain). Restore by re-scanning wikitext for headings.
    # Gather original headings and re-inject.
    heading_lines = re.findall(r"^(={2,6})\s*(.+?)\s*\1\s*$", wikitext, re.MULTILINE)
    # For each heading seen, replace first bare occurrence with md heading
    for eq, htext in heading_lines:
        level = len(eq)
        pattern = re.compile(r"^" + re.escape(htext) + r"\s*$", re.MULTILINE)
        result, n = pattern.subn("\n" + "#" * min(level, 6) + " " + htext + "\n", result, count=1)

    # Restore display math placeholders
    def _restore(m):
        idx = int(m.group(1))
        if idx < len(display_math):
            return f"\n\n$${display_math[idx]}$$\n\n"
        return ""
    result = re.sub(r"\x00DM(\d+)\x00", _restore, result)

    # Drop source/ref/nav sections — match with or without '##' prefix
    _cut_headings = ("Sources", "Also see", "Also known as", "Also defined as",
                     "Also presented as", "Also denoted as", "Notation",
                     "Notes", "References", "Bibliography",
                     "Historical Note", "Linguistic Note",
                     "Comment", "Warning")
    pat = r"(?m)^\s*#*\s*(?:" + "|".join(re.escape(h) for h in _cut_headings) + r")\s*$"
    m = re.search(pat, result)
    if m: result = result[:m.start()]

    # Strip Category:... lines
    result = re.sub(r"(?m)^\s*Category:.*$", "", result)

    # Drop empty section headers (heading immediately followed by another heading or EOF)
    result = re.sub(r"(?m)^\s*#+\s*\S.*\n(?=\s*#|\s*$)", "", result)

    # Collapse whitespace
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    return result.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="data/raw/proofwiki/latest.xml")
    ap.add_argument("--out", default="data/clean/proofwiki/")
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--shard-size", type=int, default=10_000)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    kept_by_ns = {"0": 0, "100": 0, "102": 0}
    skipped_short = 0; skipped_ns = 0; parse_err = 0
    total_chars = 0
    shard_idx, count_in_shard = 0, 0
    fout = open(out_dir / f"shard_{shard_idx:04d}.jsonl", "w")

    for ev, el in ET.iterparse(args.xml, events=("end",)):
        if el.tag != MW_NS + "page":
            continue
        ns_el = el.find(MW_NS + "ns")
        nsnum = ns_el.text if ns_el is not None else "?"
        if nsnum not in KEEP_NAMESPACES:
            skipped_ns += 1
            el.clear(); continue

        title_el = el.find(MW_NS + "title")
        text_el = el.find(f"{MW_NS}revision/{MW_NS}text")
        title = title_el.text if title_el is not None else ""
        wiki = text_el.text if (text_el is not None and text_el.text) else ""

        # Skip redirects
        if wiki.lstrip().lower().startswith("#redirect"):
            el.clear(); continue

        try:
            body = wiki_to_text(wiki)
        except Exception as e:
            parse_err += 1
            el.clear(); continue

        if len(body) < args.min_chars:
            skipped_short += 1
            el.clear(); continue

        # Determine type/level
        if nsnum == "102":
            ptype = "definition"; level = 5
        elif nsnum == "100":
            ptype = "axiom"; level = 5
        else:
            ptype = "proof" if "/Proof" in title or "∎" in body else "theorem"
            level = 7

        rec = {
            "source": "proofwiki",
            "type": ptype,
            "level": level,
            "title": title,
            "text": f"# {title}\n\n{body}",
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept_by_ns[nsnum] += 1
        total_chars += len(rec["text"])
        count_in_shard += 1
        if count_in_shard >= args.shard_size:
            fout.close()
            shard_idx += 1; count_in_shard = 0
            fout = open(out_dir / f"shard_{shard_idx:04d}.jsonl", "w")

        if sum(kept_by_ns.values()) % 5000 == 0:
            print(f"  kept={sum(kept_by_ns.values())} chars={total_chars/1e6:.1f}MB", flush=True)
        el.clear()

    fout.close()
    print(f"\n=== ProofWiki ingest ===")
    print(f"  kept by ns: {kept_by_ns}")
    print(f"  skipped (too short): {skipped_short}")
    print(f"  skipped (wrong ns): {skipped_ns}")
    print(f"  parse errors: {parse_err}")
    print(f"  total text: {total_chars/1e6:.1f} MB across {shard_idx+1} shards")


if __name__ == "__main__":
    main()
