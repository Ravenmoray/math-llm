"""Percentages and order-of-operations (PEMDAS) problems."""
from . import templates as T


def gen_percent(rng):
    p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100])
    n = rng.choice([20, 40, 50, 80, 120, 150, 200, 240, 300, 500, 800, 1000])
    prompt = rng.choice(T.PERCENT_OF_PROMPTS).format(p=p, n=n)
    ans = p * n / 100
    ans_str = f"{int(ans)}" if ans == int(ans) else f"{ans}"
    body = (f"Recall that {p}% means {p}/100. So {p}% of {n} is ({p}/100) × {n} = "
            f"{p * n}/100 = {ans_str}.")
    return f"Problem: {prompt}\n\nSolution: {body}\n\nAnswer: {ans_str}.", {"subtype": "percent", "difficulty": 1}


def gen_order_ops(rng):
    """Generate a PEMDAS expression with 3-4 operators."""
    ops = ['+', '-', '*']
    nums = [rng.randint(2, 12) for _ in range(4)]
    a, b, c, d = nums
    templates = [
        (f"{a} + {b} × {c}", a + b * c, f"Multiplication before addition: {b} × {c} = {b*c}, then {a} + {b*c} = {a + b*c}."),
        (f"({a} + {b}) × {c}", (a + b) * c, f"Parentheses first: {a} + {b} = {a+b}, then ({a+b}) × {c} = {(a+b)*c}."),
        (f"{a} × {b} + {c} × {d}", a*b + c*d, f"Do both multiplications first: {a} × {b} = {a*b} and {c} × {d} = {c*d}. Then add: {a*b} + {c*d} = {a*b + c*d}."),
        (f"{a} + {b} - {c} + {d}", a+b-c+d, f"Left to right for addition and subtraction: {a} + {b} = {a+b}, then {a+b} - {c} = {a+b-c}, then {a+b-c} + {d} = {a+b-c+d}."),
        (f"{a*c} ÷ {c} + {b}", a + b, f"Division before addition: {a*c} ÷ {c} = {a}, then {a} + {b} = {a+b}."),
    ]
    expr, ans, explanation = rng.choice(templates)
    prompt = rng.choice(T.ORDER_OPS_PROMPTS).format(expr=expr)
    return f"Problem: {prompt}\n\nSolution: {explanation}\n\nAnswer: {ans}.", {"subtype": "order_ops", "difficulty": 2}
