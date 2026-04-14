"""Fraction arithmetic with reduction steps."""
from math import gcd
from . import templates as T


def _simplify(n, d):
    g = gcd(abs(n), abs(d))
    return n // g, d // g


def _fmt(n, d):
    if d == 1: return str(n)
    return f"{n}/{d}"


def gen_fraction_add(rng):
    b = rng.randint(2, 12)
    d = rng.randint(2, 12)
    a = rng.randint(1, b * 2)
    c = rng.randint(1, d * 2)
    prompt = rng.choice(T.FRAC_ADD_PROMPTS).format(a=a, b=b, c=c, d=d)
    lcm = b * d // gcd(b, d)
    na = a * (lcm // b)
    nc = c * (lcm // d)
    total_num = na + nc
    sn, sd = _simplify(total_num, lcm)
    steps = [
        f"The denominators are {b} and {d}. A common denominator is {lcm} (the LCM of {b} and {d}).",
        f"Rewrite: {a}/{b} = {na}/{lcm} and {c}/{d} = {nc}/{lcm}.",
        f"Add numerators: {na}/{lcm} + {nc}/{lcm} = {total_num}/{lcm}.",
    ]
    if (sn, sd) != (total_num, lcm):
        steps.append(f"Simplify by dividing numerator and denominator by gcd({total_num}, {lcm}) = {gcd(total_num, lcm)}: result is {_fmt(sn, sd)}.")
    body = " ".join(steps)
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {_fmt(sn, sd)}.", {"subtype": "frac_add", "difficulty": 2}


def gen_fraction_mul(rng):
    a = rng.randint(1, 20); b = rng.randint(2, 20)
    c = rng.randint(1, 20); d = rng.randint(2, 20)
    prompt = rng.choice(T.FRAC_MUL_PROMPTS).format(a=a, b=b, c=c, d=d)
    num, den = a * c, b * d
    sn, sd = _simplify(num, den)
    body = (f"Multiply numerators and denominators: ({a}/{b}) × ({c}/{d}) = ({a}·{c})/({b}·{d}) = {num}/{den}. ")
    if (sn, sd) != (num, den):
        body += f"Simplify: gcd({num}, {den}) = {gcd(num, den)}, giving {_fmt(sn, sd)}."
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {_fmt(sn, sd)}.", {"subtype": "frac_mul", "difficulty": 2}
