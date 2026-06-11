#!/usr/bin/env python3
"""Quick smoke test for a released LoRA adapter.

Loads the adapter on its base model and checks it solves a handful of toy MATH
problems with the canonical prompt (draft slot = ``N/A``). This confirms the
adapter loads and produces sane ``\\boxed{}`` answers — it is NOT a benchmark.

    python scripts/verify_adapter.py --adapter <path-or-hf-id>
    python scripts/verify_adapter.py --adapter ./hf_models/mismatched-wrong

Requires: torch, transformers, peft (no vLLM). A GPU is strongly recommended;
the base model is ~7B parameters.
"""
from __future__ import annotations

import argparse
import re

PROMPT = (
    "Problem: {problem}\n\n"
    "Thinking: N/A\n\n"
    "The thinking section may contain errors. "
    "Solve the math problem step by step. "
    "Write your own correct solution. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Correct Solution:"
)

# Easy problems a strong MATH model should always get — good for a load/sanity check.
PROBLEMS = [
    ("If $x+y=6$ and $xy=5$, find $x^2+y^2$.", "26"),
    ("What is the value of $2^{10}$?", "1024"),
    ("Compute $\\gcd(48, 36)$.", "12"),
    ("If $f(x)=3x-7$, what is $f(5)$?", "8"),
    ("How many positive divisors does $12$ have?", "6"),
    ("Evaluate $\\sum_{k=1}^{10} k$.", "55"),
]


def extract_boxed(text: str):
    """Return the content of the last ``\\boxed{...}`` (brace-balanced), or None."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    j = i + len("\\boxed{")
    depth, out = 1, []
    while j < len(text) and depth:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        j += 1
    return "".join(out).strip()


def norm(s):
    return re.sub(r"[\s$]", "", s) if s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter path or HF id.")
    ap.add_argument("--base-model", default="mistralai/Mathstral-7B-v0.1")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Base:    {args.base_model}")
    print(f"Adapter: {args.adapter}\nLoading...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("Loaded. Running smoke problems...\n", flush=True)

    n_ok = 0
    for prob, gold in PROBLEMS:
        ids = tok(PROMPT.format(problem=prob), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(gen)
        ok = pred is not None and norm(pred) == norm(gold)
        n_ok += ok
        print(f"[{'OK ' if ok else 'XX '}] gold={gold:>6}  pred={str(pred):>10}   {prob[:48]}")
        if pred is None:
            print(f"        no \\boxed found; {len(gen)} chars; head: {gen[:120]!r}")

    print(f"\n{n_ok}/{len(PROBLEMS)} correct.")
    if n_ok == 0:
        print("WARNING: 0/N — adapter likely mis-loaded or prompt format wrong.")
    elif n_ok < len(PROBLEMS):
        print("Note: a strong checkpoint should get all of these; investigate misses.")


if __name__ == "__main__":
    main()
