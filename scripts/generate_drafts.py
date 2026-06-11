"""
Generate N draft solutions per MATH problem using a specified model.

Loads the full MATH dataset (12,500 problems), assigns split labels
(train/math500/test), generates completions via vLLM, and labels each
completion for correctness.

Usage:
    python scripts/generate_drafts.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --n-samples 32 \
        --temperature 0.8 \
        --top-p 0.95 \
        --max-tokens 2560 \
        --output-dir outputs/drafts_math_1.5b

Chain multiple models:
    python scripts/generate_drafts.py --model Qwen/Qwen2.5-Math-1.5B ...
    python scripts/generate_drafts.py --model mistralai/Mathstral-7B-v0.1 ...
    python scripts/generate_drafts.py --model Qwen/Qwen2.5-Math-7B ...
"""

from __future__ import annotations

import os

# HuggingFace caches default to ~/.cache/huggingface. Set HF_HOME to relocate.

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.grpo.rewards import extract_boxed, math_verify_match, mathematically_quasi_correct


# ── Prompt ──────────────────────────────────────────────────────────────────

def build_prompt(problem: str, mode: str = "default") -> str:
    if mode == "nodraft":
        # Matches eval_math500.py's training_nodraft mode
        return (
            f"Problem: {problem}\n\n"
            f"Thinking: N/A\n\n"
            f"The thinking section may contain errors. "
            f"Solve the math problem step by step. "
            f"Write your own correct solution. "
            f"Put your final answer within \\boxed{{}}.\n\n"
            f"Correct Solution:"
        )
    # default: concise instruction
    return (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{}.\n\n"
        f"Problem:\n{problem}\n\nSolution:"
    )


# ── Load MATH dataset with split labels ─────────────────────────────────────

def load_math_with_splits() -> list[dict]:
    """Load full MATH (12,500 problems) with split labels: train/math500/test.

    Note: Hendrycks dataset on HF doesn't expose the original problem number,
    so we match against MATH-500 by problem-text content (which is exact and
    immutable) rather than synthetic unique_ids.
    """
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    # Load MATH-500 to identify those problems by problem-text + carry their
    # canonical unique_id ("test/<subject>/<orig_num>.json") through.
    math500 = load_dataset("HuggingFaceH4/MATH-500")
    math500_text_to_uid: dict[str, str] = {}
    for row in math500["test"]:
        math500_text_to_uid[row["problem"]] = row["unique_id"]

    problems = []

    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject)

        # Train split — synthetic unique_id (Hendrycks has no original number)
        for i, row in enumerate(ds["train"]):
            problems.append({
                "problem": row["problem"],
                "answer": row["solution"].split("boxed{")[-1].rstrip("}") if "boxed{" in row["solution"] else "",
                "solution": row["solution"],
                "level": row["level"],
                "subject": subject,
                "split": "train",
                "unique_id": f"train/{subject}/{i}.json",
            })

        # Test split — match by problem text against MATH-500
        for i, row in enumerate(ds["test"]):
            uid_for_math500 = math500_text_to_uid.get(row["problem"])
            is_math500 = uid_for_math500 is not None
            problems.append({
                "problem": row["problem"],
                "answer": row["solution"].split("boxed{")[-1].rstrip("}") if "boxed{" in row["solution"] else "",
                "solution": row["solution"],
                "level": row["level"],
                "subject": subject,
                "split": "math500" if is_math500 else "test",
                # When in MATH-500, prefer its canonical unique_id (preserves orig number).
                "unique_id": uid_for_math500 if is_math500 else f"test/{subject}/{i}.json",
            })

    return problems


def extract_gold_answer(solution: str) -> str:
    """Extract the gold answer from MATH solution string using extract_boxed."""
    ans = extract_boxed(solution)
    return ans if ans is not None else ""


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2560)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=512,
                   help="Number of problems to process per vLLM batch")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max_model_len (must be >= prompt + max_tokens). Default 8192.")
    p.add_argument("--output-filename", type=str, default="drafts.json",
                   help="Filename to write under --output-dir. Default drafts.json (kept for back-compat).")
    p.add_argument("--prompt-mode", type=str, default="default",
                   choices=["default", "nodraft"],
                   help="default = concise instruction; nodraft = 'Thinking: N/A' training_nodraft template")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)

    print(f"Model:       {args.model}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Samples:     {args.n_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p:       {args.top_p}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Max model len: {args.max_model_len}")
    print(f"Output:      {output_path}")
    print()

    # ── Load problems ───────────────────────────────────────────────────
    print("Loading MATH dataset...")
    problems = load_math_with_splits()

    # Use extract_boxed on the gold solution for a cleaner answer
    for prob in problems:
        gold = extract_gold_answer(prob["solution"])
        if gold:
            prob["answer"] = gold

    split_counts = {}
    for p in problems:
        split_counts[p["split"]] = split_counts.get(p["split"], 0) + 1
    print(f"Loaded {len(problems)} problems: {split_counts}")

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\nLoading model {args.model}...")
    llm = LLM(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_samples,
    )

    # ── Generate ────────────────────────────────────────────────────────
    print(f"\nGenerating {len(problems)} x {args.n_samples} = {len(problems) * args.n_samples} completions...")
    t0 = time.time()

    results = []
    batch_size = args.batch_size

    for batch_start in range(0, len(problems), batch_size):
        batch_end = min(batch_start + batch_size, len(problems))
        batch = problems[batch_start:batch_end]
        prompts = [build_prompt(p["problem"], mode=args.prompt_mode) for p in batch]

        print(f"  Batch {batch_start}-{batch_end} / {len(problems)}...")
        outputs = llm.generate(prompts, sampling_params)

        for prob, output in zip(batch, outputs):
            gold = prob["answer"]
            samples = []
            n_correct_strict = 0
            n_correct_quasi = 0

            for completion in output.outputs:
                text = completion.text
                pred = extract_boxed(text)
                strict = math_verify_match(pred, gold) if pred else False
                quasi = mathematically_quasi_correct(text, gold) if gold else False

                if strict:
                    n_correct_strict += 1
                if quasi:
                    n_correct_quasi += 1

                samples.append({
                    "text": text,
                    "pred": pred,
                    "correct_strict": strict,
                    "correct_quasi": quasi,
                    "length": len(text),
                })

            results.append({
                "problem": prob["problem"],
                "answer": gold,
                "solution": prob["solution"],
                "level": prob["level"],
                "subject": prob["subject"],
                "split": prob["split"],
                "unique_id": prob["unique_id"],
                "n_samples": args.n_samples,
                "n_correct_strict": n_correct_strict,
                "n_correct_quasi": n_correct_quasi,
                "samples": samples,
            })

        # Save intermediate results after each batch
        with open(output_path, "w") as f:
            json.dump({
                "meta": {
                    "model": args.model,
                    "prompt_mode": args.prompt_mode,
                    "n_samples": args.n_samples,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "n_problems": len(problems),
                    "n_completed": len(results),
                },
                "records": results,
            }, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        rate = len(results) / elapsed * 60
        print(f"    {len(results)}/{len(problems)} done, {elapsed:.0f}s elapsed, {rate:.0f} problems/min")

    elapsed = time.time() - t0
    print(f"\nDone! {len(results)} problems, {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Summary stats ───────────────────────────────────────────────────
    total_strict = sum(r["n_correct_strict"] for r in results)
    total_quasi = sum(r["n_correct_quasi"] for r in results)
    total_samples = len(results) * args.n_samples
    all_correct = sum(1 for r in results if r["n_correct_strict"] == args.n_samples)
    all_wrong = sum(1 for r in results if r["n_correct_strict"] == 0)

    print(f"\n=== Summary ===")
    print(f"Strict correct:  {total_strict}/{total_samples} ({total_strict/total_samples*100:.1f}%)")
    print(f"Quasi correct:   {total_quasi}/{total_samples} ({total_quasi/total_samples*100:.1f}%)")
    print(f"All-correct problems: {all_correct}/{len(results)}")
    print(f"All-wrong problems:   {all_wrong}/{len(results)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
