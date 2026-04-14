"""Minimal MathML -> LaTeX converter.

Handles the subset of Presentation MathML used in OpenStax CNXML. Not a general
MathML implementation — goal is faithful reproduction of the math text so the
downstream tokenizer sees canonical LaTeX.
"""
from lxml import etree

MML = "http://www.w3.org/1998/Math/MathML"

# MathML operators that map directly to LaTeX commands
_OP_MAP = {
    "\u00d7": r"\times",     # ×
    "\u00f7": r"\div",       # ÷
    "\u2212": "-",           # −
    "\u2213": r"\mp",
    "\u00b1": r"\pm",
    "\u2264": r"\le",
    "\u2265": r"\ge",
    "\u2260": r"\ne",
    "\u2248": r"\approx",
    "\u2261": r"\equiv",
    "\u2192": r"\to",
    "\u21d2": r"\Rightarrow",
    "\u21d4": r"\Leftrightarrow",
    "\u221e": r"\infty",
    "\u2211": r"\sum",
    "\u220f": r"\prod",
    "\u222b": r"\int",
    "\u2202": r"\partial",
    "\u2207": r"\nabla",
    "\u221a": r"\sqrt{}",
    "\u00b7": r"\cdot",
    "\u2218": r"\circ",
    "\u22c5": r"\cdot",
    "\u2026": r"\ldots",
    "\u2022": r"\bullet",
}

# Greek letters and common symbols that appear in <mi>
_MI_MAP = {
    "\u03b1": r"\alpha", "\u03b2": r"\beta", "\u03b3": r"\gamma", "\u03b4": r"\delta",
    "\u03b5": r"\epsilon", "\u03b6": r"\zeta", "\u03b7": r"\eta", "\u03b8": r"\theta",
    "\u03b9": r"\iota", "\u03ba": r"\kappa", "\u03bb": r"\lambda", "\u03bc": r"\mu",
    "\u03bd": r"\nu", "\u03be": r"\xi", "\u03c0": r"\pi", "\u03c1": r"\rho",
    "\u03c3": r"\sigma", "\u03c4": r"\tau", "\u03c5": r"\upsilon", "\u03c6": r"\phi",
    "\u03c7": r"\chi", "\u03c8": r"\psi", "\u03c9": r"\omega",
    "\u0393": r"\Gamma", "\u0394": r"\Delta", "\u0398": r"\Theta", "\u039b": r"\Lambda",
    "\u039e": r"\Xi", "\u03a0": r"\Pi", "\u03a3": r"\Sigma", "\u03a6": r"\Phi",
    "\u03a8": r"\Psi", "\u03a9": r"\Omega",
}


def _localname(el):
    t = el.tag
    return t.split("}", 1)[1] if "}" in t else t


def _text(el):
    return (el.text or "").strip()


def _convert_children(el):
    return "".join(_convert(c) for c in el)


def _convert(el):
    if not hasattr(el, "tag") or not isinstance(el.tag, str):
        return ""
    tag = _localname(el)

    if tag in ("math", "mrow", "mstyle", "mpadded", "merror", "mphantom"):
        return _convert_children(el)

    if tag == "mi":
        t = _text(el)
        if t in _MI_MAP:
            return _MI_MAP[t]
        return t

    if tag == "mn":
        return _text(el)

    if tag == "mo":
        t = _text(el)
        return _OP_MAP.get(t, t)

    if tag == "mtext":
        t = _text(el)
        if not t: return ""
        return r"\text{" + t + "}"

    if tag == "mspace":
        return " "

    if tag == "mfrac":
        kids = list(el)
        if len(kids) >= 2:
            return r"\frac{" + _convert(kids[0]) + "}{" + _convert(kids[1]) + "}"
        return _convert_children(el)

    if tag == "msup":
        kids = list(el)
        if len(kids) >= 2:
            return "{" + _convert(kids[0]) + "}^{" + _convert(kids[1]) + "}"
        return _convert_children(el)

    if tag == "msub":
        kids = list(el)
        if len(kids) >= 2:
            return "{" + _convert(kids[0]) + "}_{" + _convert(kids[1]) + "}"
        return _convert_children(el)

    if tag == "msubsup":
        kids = list(el)
        if len(kids) >= 3:
            return "{" + _convert(kids[0]) + "}_{" + _convert(kids[1]) + "}^{" + _convert(kids[2]) + "}"
        return _convert_children(el)

    if tag == "msqrt":
        return r"\sqrt{" + _convert_children(el) + "}"

    if tag == "mroot":
        kids = list(el)
        if len(kids) >= 2:
            return r"\sqrt[" + _convert(kids[1]) + "]{" + _convert(kids[0]) + "}"
        return _convert_children(el)

    if tag == "munder":
        kids = list(el)
        if len(kids) >= 2:
            return r"\underset{" + _convert(kids[1]) + "}{" + _convert(kids[0]) + "}"
        return _convert_children(el)

    if tag == "mover":
        kids = list(el)
        if len(kids) >= 2:
            return r"\overset{" + _convert(kids[1]) + "}{" + _convert(kids[0]) + "}"
        return _convert_children(el)

    if tag == "munderover":
        kids = list(el)
        if len(kids) >= 3:
            return r"\underset{" + _convert(kids[1]) + "}{\overset{" + _convert(kids[2]) + "}{" + _convert(kids[0]) + "}}"
        return _convert_children(el)

    if tag == "mfenced":
        op = el.get("open", "(")
        cl = el.get("close", ")")
        sep = el.get("separators", ",")
        parts = [_convert(c) for c in el]
        if not parts:
            return op + cl
        joined = sep.join(parts) if len(parts) > 1 else parts[0]
        return op + joined + cl

    if tag == "mtable":
        rows = []
        for tr in el:
            if _localname(tr) != "mtr": continue
            cells = [_convert(td) for td in tr if _localname(td) == "mtd"]
            rows.append(" & ".join(cells))
        if not rows: return ""
        return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"

    # Fallback: just convert children
    return _convert_children(el)


def mathml_to_latex(el):
    """Convert a <m:math> element to a LaTeX string (no delimiters)."""
    s = _convert(el)
    # collapse runs of whitespace
    s = " ".join(s.split())
    return s
