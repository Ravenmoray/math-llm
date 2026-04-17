"""Eval v2: prompts match training data format exactly.

The model trained on documents like:
  Problem: What is 47 + 238?

  Solution: We need to add 47 and 238. ...

  Answer: 285.

So we feed partial documents and check if the completion contains the right answer.
Three prompt styles tested:
  A) Full format: "Problem: ...\n\nSolution:"  (model completes solution+answer)
  B) Skip to answer: "Problem: ...\n\nAnswer:"  (model gives answer directly)
  C) Mid-solution: "Problem: ...\n\nSolution: We need to add X and Y." (model continues)
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.config import ModelConfig
from model.model import MathLM

PROBLEMS = [
    # (prompt_A, prompt_B, expected, category)
    # --- L1 addition ---
    ("Problem: What is 47 + 238?\n\nSolution:",
     "Problem: What is 47 + 238?\n\nAnswer:",
     "285", "L1-add-1"),
    ("Problem: Find the sum of 1234 and 567.\n\nSolution:",
     "Problem: Find the sum of 1234 and 567.\n\nAnswer:",
     "1801", "L1-add-2"),
    ("Problem: Add 89 and 47.\n\nSolution:",
     "Problem: Add 89 and 47.\n\nAnswer:",
     "136", "L1-add-3"),

    # --- L1 subtraction ---
    ("Problem: What is 500 - 237?\n\nSolution:",
     "Problem: What is 500 - 237?\n\nAnswer:",
     "263", "L1-sub-1"),
    ("Problem: Subtract 45 from 312.\n\nSolution:",
     "Problem: Subtract 45 from 312.\n\nAnswer:",
     "267", "L1-sub-2"),

    # --- L1 multiplication ---
    ("Problem: Compute 156 × 7.\n\nSolution:",
     "Problem: Compute 156 × 7.\n\nAnswer:",
     "1092", "L1-mul-1"),
    ("Problem: Multiply 23 by 15.\n\nSolution:",
     "Problem: Multiply 23 by 15.\n\nAnswer:",
     "345", "L1-mul-2"),

    # --- L1 division ---
    ("Problem: What is 145 ÷ 6?\n\nSolution:",
     "Problem: What is 145 ÷ 6?\n\nAnswer:",
     "24", "L1-div-1"),  # 24 remainder 1
    ("Problem: Divide 84 by 7.\n\nSolution:",
     "Problem: Divide 84 by 7.\n\nAnswer:",
     "12", "L1-div-2"),

    # --- L1 fractions ---
    ("Problem: What is 3/4 + 2/5?\n\nSolution:",
     "Problem: What is 3/4 + 2/5?\n\nAnswer:",
     "23/20", "L1-frac-add"),
    ("Problem: Add the fractions 1/3 and 1/6.\n\nSolution:",
     "Problem: Add the fractions 1/3 and 1/6.\n\nAnswer:",
     "1/2", "L1-frac-add-2"),
    ("Problem: What is (3/5) × (2/7)?\n\nSolution:",
     "Problem: What is (3/5) × (2/7)?\n\nAnswer:",
     "6/35", "L1-frac-mul"),

    # --- L1 percent ---
    ("Problem: What is 25% of 800?\n\nSolution:",
     "Problem: What is 25% of 800?\n\nAnswer:",
     "200", "L1-pct-1"),
    ("Problem: Find 10% of 350.\n\nSolution:",
     "Problem: Find 10% of 350.\n\nAnswer:",
     "35", "L1-pct-2"),

    # --- L1 order of operations ---
    ("Problem: Evaluate 3 + 4 × 5.\n\nSolution:",
     "Problem: Evaluate 3 + 4 × 5.\n\nAnswer:",
     "23", "L1-pemdas"),

    # --- L1 word problems ---
    ("Problem: Sam has 45 apples. Taylor gives them 28 more apples. How many apples does Sam have now?\n\nSolution:",
     "Problem: Sam has 45 apples. Taylor gives them 28 more apples. How many apples does Sam have now?\n\nAnswer:",
     "73", "L1-word-add"),
    ("Problem: A teacher divides 72 pencils evenly among 8 students. How many does each student receive?\n\nSolution:",
     "Problem: A teacher divides 72 pencils evenly among 8 students. How many does each student receive?\n\nAnswer:",
     "9", "L1-word-div"),

    # --- Novel (unseen exact numbers but same format) ---
    ("Problem: What is 9999 + 1?\n\nSolution:",
     "Problem: What is 9999 + 1?\n\nAnswer:",
     "10000", "novel-add"),
    ("Problem: Compute 100 × 100.\n\nSolution:",
     "Problem: Compute 100 × 100.\n\nAnswer:",
     "10000", "novel-mul"),
    ("Problem: What is 50% of 50?\n\nSolution:",
     "Problem: What is 50% of 50?\n\nAnswer:",
     "25", "novel-pct"),
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
def generate(model, tok, prompt, max_new=300, temperature=0.0, device="cuda"):
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
        tok_str = tok.decode([nxt.item()], skip_special_tokens=False)
        if nxt.item() == eos_id:
            break
        # Stop at next "Problem:" (new document boundary)
        full_so_far = tok.decode(ids[0].tolist(), skip_special_tokens=True)
        after_prompt = full_so_far[len(prompt):]
        if "\nProblem:" in after_prompt:
            break
    full = tok.decode(ids[0].tolist(), skip_special_tokens=True)
    return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()


def check_answer(generated, expected):
    gen_flat = generated.lower().replace(",", "").replace(" ", "")
    exp_flat = expected.lower().replace(",", "").replace(" ", "")
    if exp_flat in gen_flat:
        return True
    # Check "Answer:" line specifically
    for pat in [r"answer:\s*([^\.\n]+)", r"=\s*([^\.\n]+)"]:
        m = re.search(pat, generated, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().replace(",", "").replace(" ", "").lower().rstrip(".")
            if exp_flat == ans or exp_flat in ans:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v2")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    ckpt_dir = Path(args.ckpt_dir)

    if args.steps:
        ckpt_files = [ckpt_dir / f"step{s}.pt" for s in args.steps]
    else:
        ckpt_files = sorted(ckpt_dir.glob("step*.pt"),
                            key=lambda p: int(re.search(r"(\d+)", p.stem).group()))

    summary_rows = []

    for ckpt_path in ckpt_files:
        if not ckpt_path.exists(): continue
        step_name = ckpt_path.stem
        print(f"\n{'='*70}")
        print(f"  {step_name}")
        print(f"{'='*70}")
        model, cfg, step = load_model(str(ckpt_path), args.device)

        scores = {"A_correct": 0, "A_wrong": 0, "B_correct": 0, "B_wrong": 0}
        for prompt_a, prompt_b, expected, cat in PROBLEMS:
            # Style A: "Problem: ...\n\nSolution:"
            gen_a = generate(model, tok, prompt_a, device=args.device)
            ok_a = check_answer(gen_a, expected)
            scores["A_correct" if ok_a else "A_wrong"] += 1

            # Style B: "Problem: ...\n\nAnswer:"
            gen_b = generate(model, tok, prompt_b, device=args.device)
            ok_b = check_answer(gen_b, expected)
            scores["B_correct" if ok_b else "B_wrong"] += 1

            tag_a = "OK" if ok_a else "X "
            tag_b = "OK" if ok_b else "X "
            if args.verbose:
                print(f"  [{tag_a}|{tag_b}] {cat:18s}  want={expected:>10s}")
                if ok_a:
                    print(f"         A: {gen_a[:120]!r}")
                else:
                    print(f"         A: {gen_a[:120]!r}")
                if not ok_b:
                    print(f"         B: {gen_b[:120]!r}")

        n = len(PROBLEMS)
        pct_a = scores["A_correct"] / n * 100
        pct_b = scores["B_correct"] / n * 100
        print(f"\n  Style A (Solution:): {scores['A_correct']}/{n} = {pct_a:.0f}%")
        print(f"  Style B (Answer:):   {scores['B_correct']}/{n} = {pct_b:.0f}%")
        summary_rows.append((step_name, scores["A_correct"], scores["B_correct"], n))

        del model; torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"{'Checkpoint':>12}  {'Solution':>10}  {'Answer':>10}  {'Total':>6}")
    print(f"{'-'*50}")
    for name, a, b, n in summary_rows:
        print(f"{name:>12}  {a:>5}/{n:<4}  {b:>5}/{n:<4}  {a+b:>3}/{2*n}")


if __name__ == "__main__":
    main()
