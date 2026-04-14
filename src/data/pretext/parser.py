"""PreTeXt (.ptx / .xml) -> markdown-ish text with LaTeX math.

PreTeXt encodes math as raw LaTeX inside <m> (inline) and <me>/<md> (display),
so no math conversion is needed — just structural walk and delimiter injection.
"""
from lxml import etree
import re


def _ln(el):
    t = el.tag
    if not isinstance(t, str): return None
    return t.split("}", 1)[1] if "}" in t else t


# Elements to skip completely
_SKIP = {"figure", "image", "sage", "sageplot", "asymptote", "latex-image",
         "tikzpicture", "fn", "notation", "idx", "index", "solution",
         "biblio", "references", "bibinfo", "program", "console",
         "caption", "tabular"}  # tabular: crude drop for v1 (often complex)

# Inline formatting
_EM = {"em", "emphasis", "term", "alert"}
_CODE_LIKE = {"c", "tag", "tage", "taga"}


def _inline(el):
    parts = [el.text or ""]
    for c in el:
        name = _ln(c)
        if name is None:
            if c.tail: parts.append(c.tail)
            continue

        if name == "m":
            t = (c.text or "").strip()
            parts.append(f" ${t}$ ")
        elif name == "me":
            t = (c.text or "").strip()
            parts.append(f"\n\n$${t}$$\n\n")
        elif name == "md" or name == "mdn":
            # multi-line display
            lines = []
            for mrow in c:
                if _ln(mrow) == "mrow":
                    lines.append((mrow.text or "").strip())
            body = " \\\\ ".join(lines)
            parts.append(f"\n\n$$\\begin{{aligned}}{body}\\end{{aligned}}$$\n\n")
        elif name == "xref":
            pass  # drop cross-refs
        elif name == "url":
            t = (c.text or "").strip() or c.get("href", "")
            parts.append(t)
        elif name in _EM:
            parts.append(f"*{_inline(c).strip()}*")
        elif name == "q":
            parts.append(f"\"{_inline(c).strip()}\"")
        elif name in _CODE_LIKE:
            parts.append(f"`{_inline(c).strip()}`")
        elif name in _SKIP:
            pass  # drop
        elif name == "nbsp":
            parts.append(" ")
        else:
            # Recurse as inline (default)
            parts.append(_inline(c))

        if c.tail:
            parts.append(c.tail)
    return "".join(parts)


_HEADING = {"chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4,
            "appendix": 1, "preface": 1, "introduction": 0, "conclusion": 0}

_ENV_LABELS = {
    "theorem": "Theorem", "lemma": "Lemma", "proposition": "Proposition",
    "corollary": "Corollary", "definition": "Definition", "example": "Example",
    "exercise": "Exercise", "activity": "Activity", "remark": "Remark",
    "note": "Note", "observation": "Observation", "claim": "Claim",
    "proof": "Proof",
}


def _block(el, depth=0):
    name = _ln(el)
    if name is None: return ""

    if name in _SKIP:
        return ""

    if name == "p":
        return _inline(el).strip() + "\n\n"

    if name == "title":
        t = _inline(el).strip()
        return t  # caller decides heading level

    if name in _HEADING:
        level = _HEADING[name]
        out = []
        # title
        title_el = el.find("title") if el.find("title") is not None else None
        if title_el is not None:
            title_text = _inline(title_el).strip()
            if title_text and level > 0:
                out.append("\n\n" + "#" * min(6, level) + " " + title_text + "\n\n")
        for c in el:
            if _ln(c) == "title": continue
            out.append(_block(c, depth + 1))
        return "".join(out)

    if name in _ENV_LABELS:
        label = _ENV_LABELS[name]
        # PreTeXt may wrap content in <statement> and <proof>
        inner = []
        title_el = el.find("title")
        title_text = _inline(title_el).strip() if title_el is not None else ""
        header = f"**{label}"
        if title_text: header += f" ({title_text})"
        header += ".** "
        inner.append(header)
        for c in el:
            if _ln(c) == "title": continue
            inner.append(_block(c, depth + 1))
        body = "".join(inner).strip()
        if name == "proof":
            body += " ∎"
        return "\n\n" + body + "\n\n"

    if name == "statement":
        return _inline_or_block(el, depth)

    if name == "hypothesis":
        return "*Hypothesis.* " + _inline(el).strip() + "\n\n"

    if name in ("ol", "ul"):
        items = []
        enum = name == "ol"
        for i, c in enumerate(el, 1):
            if _ln(c) != "li": continue
            prefix = f"{i}. " if enum else "- "
            items.append(prefix + _inline_or_block(c, depth).strip())
        return "\n" + "\n".join(items) + "\n\n"

    if name == "li":
        return _inline_or_block(el, depth)

    if name == "blockquote":
        body = _inline_or_block(el, depth).strip()
        return "\n> " + body.replace("\n", "\n> ") + "\n\n"

    if name == "sidebyside":
        # layout container — recurse
        return "".join(_block(c, depth) for c in el)

    # Default: recurse
    return "".join(_block(c, depth) for c in el)


def _inline_or_block(el, depth):
    """Element may contain either <p> blocks or inline content. Handle both."""
    has_block = any(_ln(c) in ("p", "ol", "ul") or _ln(c) in _HEADING or _ln(c) in _ENV_LABELS for c in el)
    if has_block:
        return "".join(_block(c, depth) for c in el)
    return _inline(el)


def parse_file(path):
    """Return (title, text). Title comes from root <title> if present."""
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError:
        return None, ""
    root = tree.getroot()
    # Find a title near the root
    title = ""
    t_el = root.find("title")
    if t_el is not None:
        title = _inline(t_el).strip()
    body = _block(root, 0)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r" {2,}", " ", body)
    return title, body.strip()
