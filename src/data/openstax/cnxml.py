"""CNXML -> plain-text (markdown-ish) with LaTeX math inline.

Conservative: preserves content flow, drops media/figures, preserves math.
"""
from lxml import etree
from .mathml import mathml_to_latex, MML

CNX = "http://cnx.rice.edu/cnxml"


def _ln(el):
    t = el.tag
    return t.split("}", 1)[1] if "}" in t else t


# Elements we skip entirely (and their subtree)
_SKIP = {"image", "media", "figure", "note", "iframe", "link",
         "newline", "cite", "cite-title"}

# Elements whose text content we keep (pass-through)
_PASS = {"document", "content", "span", "preformat"}


def _render(el, depth=0):
    tag = _ln(el)

    # Math → LaTeX
    if tag == "math" or (hasattr(el, "tag") and isinstance(el.tag, str) and el.tag.endswith("}math")):
        latex = mathml_to_latex(el)
        if not latex: return ""
        return f" ${latex}$ "

    if tag in _SKIP:
        return ""

    if tag == "title":
        text = _inline(el)
        if not text: return ""
        h = "#" * min(6, depth + 1)
        return f"\n\n{h} {text}\n\n"

    if tag == "section":
        inner = _children(el, depth + 1)
        return inner + "\n"

    if tag == "para":
        text = _inline(el)
        return text.strip() + "\n\n" if text.strip() else ""

    if tag == "emphasis":
        eff = el.get("effect", "italics")
        inner = _inline(el)
        if not inner: return ""
        if eff == "bold": return f"**{inner}**"
        return f"*{inner}*"

    if tag == "list":
        items = []
        style = el.get("list-type", "bulleted")
        for i, child in enumerate(el, 1):
            if _ln(child) == "item":
                prefix = f"{i}. " if style == "enumerated" else "- "
                items.append(prefix + _inline(child).strip())
        return "\n" + "\n".join(items) + "\n\n"

    if tag == "item":
        return _inline(el)

    if tag == "equation":
        inner = _inline(el).strip()
        return f"\n\n$$\n{inner.strip('$').strip()}\n$$\n\n"

    if tag == "example":
        inner = _children(el, depth + 1)
        return f"\n\n**Example.** {inner.strip()}\n\n"

    if tag == "problem":
        inner = _children(el, depth)
        return f"\n**Problem.** {inner.strip()}\n"

    if tag == "solution":
        inner = _children(el, depth)
        return f"\n**Solution.** {inner.strip()}\n"

    if tag == "rule":
        inner = _children(el, depth + 1)
        return f"\n\n{inner.strip()}\n\n"

    if tag == "statement":
        return _children(el, depth)

    if tag == "table":
        # Flatten to a plain readable line-per-row
        rows = []
        for tr in el.iter():
            if _ln(tr) == "row":
                cells = []
                for td in tr:
                    if _ln(td) == "entry":
                        cells.append(_inline(td).strip())
                if cells:
                    rows.append(" | ".join(cells))
        return "\n\n" + "\n".join(rows) + "\n\n" if rows else ""

    # Pass-through / unknown — recurse into children, keep text
    return _children(el, depth)


def _inline(el):
    """Render an element and its descendants as inline text."""
    parts = [el.text or ""]
    for c in el:
        parts.append(_render(c, 0))
        if c.tail:
            parts.append(c.tail)
    return "".join(parts)


def _children(el, depth):
    parts = [el.text or ""]
    for c in el:
        parts.append(_render(c, depth))
        if c.tail:
            parts.append(c.tail)
    return "".join(parts)


def cnxml_to_text(path):
    """Parse a CNXML file and return (title, body_markdown)."""
    tree = etree.parse(str(path))
    root = tree.getroot()
    # Find title
    title_el = root.find(f"{{{CNX}}}title")
    title = _inline(title_el).strip() if title_el is not None else ""
    content_el = root.find(f"{{{CNX}}}content")
    if content_el is None:
        return title, ""
    body = _children(content_el, 1)
    # Normalize whitespace
    import re
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r" {2,}", " ", body)
    return title, body.strip()
