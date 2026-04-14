"""Word problems for +, -, ×, ÷. Teach translating language to arithmetic."""
from . import templates as T


def _names(rng):
    n1, n2 = rng.sample(T.NAMES, 2)
    return n1, n2


def gen_word_add(rng):
    name, name2 = _names(rng)
    item = rng.choice(T.ITEMS)
    a = rng.randint(3, 80); b = rng.randint(3, 80)
    tmpl = rng.choice(T.WORD_ADD_TEMPLATES)
    prompt = tmpl.format(name=name, name2=name2, item=item, a=a, b=b)
    body = (f"This is an addition problem: we combine {a} and {b}. "
            f"{a} + {b} = {a + b}. So there are {a + b} {item}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a + b}.", {"subtype": "word_add", "difficulty": 1}


def gen_word_sub(rng):
    name, name2 = _names(rng)
    item = rng.choice(T.ITEMS)
    a = rng.randint(10, 100); b = rng.randint(1, a - 1)
    tmpl = rng.choice(T.WORD_SUB_TEMPLATES)
    prompt = tmpl.format(name=name, name2=name2, item=item, a=a, b=b)
    body = (f"This is a subtraction problem: we remove {b} from {a}. "
            f"{a} - {b} = {a - b}. So {a - b} {item} remain.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a - b}.", {"subtype": "word_sub", "difficulty": 1}


def gen_word_mul(rng):
    name, _ = _names(rng)
    item = rng.choice(T.ITEMS)
    a = rng.randint(2, 20); b = rng.randint(2, 30)
    tmpl = rng.choice(T.WORD_MUL_TEMPLATES)
    prompt = tmpl.format(name=name, item=item, a=a, b=b)
    body = (f"Each group contributes {b}, and there are {a} groups. "
            f"Total = {a} × {b} = {a * b}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {a * b}.", {"subtype": "word_mul", "difficulty": 2}


def gen_word_div(rng):
    name, _ = _names(rng)
    item = rng.choice(T.ITEMS)
    b = rng.randint(2, 10)
    q = rng.randint(2, 15)
    a = b * q
    tmpl = rng.choice(T.WORD_DIV_TEMPLATES)
    prompt = tmpl.format(name=name, item=item, a=a, b=b)
    body = (f"We divide {a} evenly into {b} parts. "
            f"{a} ÷ {b} = {q}. So each part gets {q}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {q}.", {"subtype": "word_div", "difficulty": 2}
