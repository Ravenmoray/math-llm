"""Evaluate all checkpoints on a mix of in-distribution and novel problems.

For each checkpoint: generate completions, check correctness, report.
Uses CPU to avoid VRAM issues; slow but thorough.
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.config import ModelConfig
from model.model import MathLM


# ── Test problems ──────────────────────────────────────────────────
# Each: (prompt, expected_answer, category)
# "in-dist" = looks like training data templates
# "novel"   = same math, different phrasing or unseen numbers
# "hard"    = requires generalization beyond templates

PROBLEMS = [
    # L1 in-distribution (matches synthetic templates)
    ("Problem: What is 47 + 238?\n\nSolution:", "285", "L1-add-indist"),
    ("Problem: Compute 156 × 7.\n\nSolution:", "1092", "L1-mul-indist"),
    ("Problem: Add the fractions 3/4 and 2/5.\n\nSolution:", "23/20", "L1-frac-indist"),
    ("Problem: What is 25% of 800?\n\nSolution:", "200", "L1-pct-indist"),
    ("Problem: Divide 145 by 6.\n\nSolution:", "24 remainder 1", "L1-div-indist"),

    # L1 novel phrasing (same difficulty, never-seen wording)
    ("Question: Can you add 389 and 5217 for me?\n\nAnswer:", "5606", "L1-add-novel"),
    ("Calculate: 48 times 53\n\nResult:", "2544", "L1-mul-novel"),
    ("How much is three-quarters plus one-half?", "5/4", "L1-frac-novel"),
    ("If I have $450 and spend 30%, how much did I spend?", "135", "L1-pct-novel"),

    # L2 algebra (OpenStax-level)
    ("Solve for x: 3x + 7 = 22\n\nSolution:", "5", "L2-algebra"),
    ("Solve for x: 2x - 5 = 11\n\nSolution:", "8", "L2-algebra"),
    ("What is the slope of the line y = 4x - 3?", "4", "L2-algebra"),

    # L4 calculus
    ("What is the derivative of x^3 with respect to x?", "3x^2", "L4-calculus"),
    ("Find the integral of 2x dx.", "x^2", "L4-calculus"),

    # L5-L7 proof-style
    ("State the definition of a group in abstract algebra.", None, "L6-definition"),
    ("Prove that the identity element of a group is unique.", None, "L7-proof"),
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["cfg"].items() if k in ModelConfig.__dataclass_fields__})
    model = MathLM(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.train(mode=False)
    return model, cfg, ckpt.get("step", "?")


@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temperature=0.0, device="cpu"):
    enc = tok.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    max_ctx = model.cfg.max_seq_len
    eos_id = tok.token_to_id("<|eos|>") or -1
    for _ in range(max_new):
        cond = ids if ids.size(1) <= max_ctx else ids[:, -max_ctx:]
        logits, _ = model(cond)
        logits = logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == eos_id:
            break
    full = tok.decode(ids[0].tolist(), skip_special_tokens=True)
    # Return only the generated part (after the prompt)
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()


def check_answer(generated, expected):
    """Fuzzy check: does the expected answer appear in the generation?"""
    if expected is None:
        return None  # qualitative, can't auto-check
    gen_clean = generated.lower().replace(",", "").replace(" ", "")
    exp_clean = expected.lower().replace(",", "").replace(" ", "")
    # Check if expected appears in generated
    if exp_clean in gen_clean:
        return True
    # Also check just the "Answer:" line if present
    ans_match = re.search(r"answer:\s*(.+?)[\.\n]", generated, re.IGNORECASE)
    if ans_match:
        ans_text = ans_match.group(1).strip().replace(",", "").replace(" ", "").lower()
        if exp_clean in ans_text:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v2")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    help="specific steps to test (default: all)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    ckpt_dir = Path(args.ckpt_dir)

    if args.steps:
        ckpt_files = [ckpt_dir / f"step{s}.pt" for s in args.steps]
    else:
        ckpt_files = sorted(ckpt_dir.glob("step*.pt"),
                            key=lambda p: int(re.search(r"(\d+)", p.stem).group()))

    all_results = {}

    for ckpt_path in ckpt_files:
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt_path} not found"); continue
        step_name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"  Loading {step_name}...")
        t0 = time.time()
        model, cfg, step = load_model(str(ckpt_path), args.device)
        print(f"  Loaded in {time.time()-t0:.0f}s")

        results = {"correct": 0, "wrong": 0, "qualitative": 0, "details": []}
        for prompt, expected, category in PROBLEMS:
            gen = generate(model, tok, prompt, device=args.device)
            correct = check_answer(gen, expected)
            status = "CORRECT" if correct is True else ("WRONG" if correct is False else "QUAL")
            if correct is True: results["correct"] += 1
            elif correct is False: results["wrong"] += 1
            else: results["qualitative"] += 1
            results["details"].append({
                "category": category, "expected": expected,
                "status": status, "generated": gen[:300]
            })
            tag = f"[{status:>7}]"
            print(f"  {tag} {category:20s}  expected={expected!s:>20s}  got: {gen[:80]!r}")

        total_checkable = results["correct"] + results["wrong"]
        pct = results["correct"] / total_checkable * 100 if total_checkable else 0
        print(f"\n  SCORE: {results['correct']}/{total_checkable} checkable correct ({pct:.0f}%)")
        all_results[step_name] = results
        del model; torch.cuda.empty_cache() if args.device == "cuda" else None

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Checkpoint':>12}  {'Correct':>8}  {'Wrong':>8}  {'Pct':>6}")
    print(f"{'-'*42}")
    for step_name, r in all_results.items():
        total = r["correct"] + r["wrong"]
        pct = r["correct"] / total * 100 if total else 0
        print(f"{step_name:>12}  {r['correct']:>8}  {r['wrong']:>8}  {pct:>5.0f}%")


if __name__ == "__main__":
    main()
