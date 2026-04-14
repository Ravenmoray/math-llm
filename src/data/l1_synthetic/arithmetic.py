"""Worked-example generators for +, -, ×, ÷ with step-by-step reasoning."""
import random
from . import templates as T


def _digits_range(d):
    if d == 1: return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def _place_value_add(a, b):
    """Generate column-addition explanation with carries."""
    sa, sb = str(a), str(b)
    w = max(len(sa), len(sb))
    sa, sb = sa.zfill(w), sb.zfill(w)
    lines = []
    carry = 0
    result_digits = []
    for i in range(w - 1, -1, -1):
        da, db = int(sa[i]), int(sb[i])
        s = da + db + carry
        place = ["ones", "tens", "hundreds", "thousands", "ten-thousands", "hundred-thousands", "millions"][w - 1 - i]
        if carry:
            lines.append(f"In the {place} place: {da} + {db} + {carry} (carry) = {s}.")
        else:
            lines.append(f"In the {place} place: {da} + {db} = {s}.")
        result_digits.append(s % 10)
        carry = s // 10
    if carry:
        lines.append(f"The final carry of {carry} becomes the leading digit.")
        result_digits.append(carry)
    return " ".join(lines)


def gen_addition(rng, max_digits=4):
    d1 = rng.randint(1, max_digits)
    d2 = rng.randint(1, max_digits)
    lo1, hi1 = _digits_range(d1)
    lo2, hi2 = _digits_range(d2)
    a, b = rng.randint(lo1, hi1), rng.randint(lo2, hi2)
    prompt = rng.choice(T.ADD_PROMPTS).format(a=a, b=b)
    if max(d1, d2) <= 2:
        body = f"We need to add {a} and {b}. Adding directly: {a} + {b} = {a + b}."
    else:
        pv = _place_value_add(a, b)
        body = f"To add {a} and {b}, we work column by column from right to left. {pv} So {a} + {b} = {a + b}."
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a + b}.", {"subtype": "addition", "difficulty": max(d1, d2)}


def gen_subtraction(rng, max_digits=4):
    d1 = rng.randint(2, max_digits)
    d2 = rng.randint(1, d1)
    lo1, hi1 = _digits_range(d1)
    lo2, hi2 = _digits_range(d2)
    a = rng.randint(lo1, hi1)
    b = rng.randint(lo2, min(a, hi2))
    prompt = rng.choice(T.SUB_PROMPTS).format(a=a, b=b)
    if max(d1, d2) <= 2:
        body = f"We subtract {b} from {a}. Directly: {a} - {b} = {a - b}."
    else:
        body = (f"To compute {a} - {b}, we work from right to left, borrowing when needed. "
                f"The result is {a} - {b} = {a - b}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a - b}.", {"subtype": "subtraction", "difficulty": d1}


def gen_multiplication(rng, max_digits=3):
    d1 = rng.randint(1, max_digits)
    d2 = rng.randint(1, max_digits)
    lo1, hi1 = _digits_range(d1)
    lo2, hi2 = _digits_range(d2)
    a, b = rng.randint(lo1, hi1), rng.randint(lo2, hi2)
    prompt = rng.choice(T.MUL_PROMPTS).format(a=a, b=b)
    if max(d1, d2) == 1:
        body = f"This is a single-digit product. {a} × {b} = {a * b}."
    elif min(d1, d2) == 1:
        big, small = (a, b) if a >= b else (b, a)
        body = (f"We multiply {big} by the single digit {small}. "
                f"Distributing, {big} × {small} = {a * b}.")
    else:
        body = (f"We use long multiplication. Expanding, {a} × {b} can be computed by "
                f"partial products. The result is {a} × {b} = {a * b}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a * b}.", {"subtype": "multiplication", "difficulty": d1 + d2}


def gen_division(rng, max_digits=3):
    """Integer division with remainder."""
    d1 = rng.randint(2, max_digits)
    d2 = rng.randint(1, max(1, d1 - 1))
    lo1, hi1 = _digits_range(d1)
    lo2, hi2 = _digits_range(d2)
    b = rng.randint(max(2, lo2), hi2)
    a = rng.randint(lo1, hi1)
    q, r = divmod(a, b)
    prompt = rng.choice(T.DIV_PROMPTS).format(a=a, b=b)
    if r == 0:
        body = f"We divide {a} by {b}. Since {b} × {q} = {a}, the quotient is exactly {q}."
        ans = f"{q}"
    else:
        body = (f"We divide {a} by {b}. The largest multiple of {b} not exceeding {a} is "
                f"{b} × {q} = {b * q}, leaving a remainder of {a} - {b * q} = {r}. "
                f"So {a} = {b} × {q} + {r}.")
        ans = f"{q} remainder {r}"
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {ans}.", {"subtype": "division", "difficulty": d1}
