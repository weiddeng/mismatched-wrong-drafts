"""
Evaluate a trained LoRA checkpoint (or base model) on AIME 2024-2026 with sampling.

Generates N completions per problem and reports:
  - pass@k:  fraction of problems where at least 1 of k samples is correct
  - avg@k:   average per-problem accuracy
  - maj@k:   majority vote accuracy (most common answer wins)

Usage:
    # 32-sample eval on a checkpoint
    python scripts/eval_aime_sampling.py \\
        --base-model mistralai/Mathstral-7B-v0.1 \\
        --lora outputs/mismatched_wrong/checkpoint-2000 \\
        --n-samples 32 --temperature 0.7

    # Sweep all checkpoints
    python scripts/eval_aime_sampling.py \\
        --sweep outputs/mismatched_wrong \\
        --base-model mistralai/Mathstral-7B-v0.1 \\
        --n-samples 32 --temperature 0.7
"""

from __future__ import annotations

import os

# HF caches default to ~/.cache/huggingface; set HF_HOME to relocate.

import argparse
import json
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.grpo.rewards import extract_boxed, extract_mathematical_answer, math_verify_match


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on AIME with sampling (pass@k, avg@k, maj@k).")
    p.add_argument("--base-model", type=str, default="mistralai/Mathstral-7B-v0.1")
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--sweep", type=str, default=None,
                   help="Directory containing multiple checkpoint-N subdirs.")
    p.add_argument("--mode", type=str, default="training_nodraft",
                   choices=["training_nodraft", "instruct"],
                   help="Prompt template. 'training_nodraft' matches the GRPO training format "
                        "the released models were trained on (Thinking: N/A ... Correct Solution:); "
                        "'instruct' is the Mistral-style [INST] format for evaluating the "
                        "instruction-tuned base model (Mathstral-7B-v0.1).")
    p.add_argument("--year", type=int, nargs="+", required=True,
                   help="AIME year(s) to evaluate on (2024, 2025, 2026).")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--n-samples", type=int, required=True,
                   help="Number of completions per problem.")
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Dataset loading ───────────────────────────────────────────────────────

AIME_DATASETS = {
    2024: ["MathArena/aime_2024_I", "MathArena/aime_2024_II"],
    2025: ["MathArena/aime_2025"],
    2026: ["MathArena/aime_2026"],
}


def load_aime_problems(year: int) -> list[dict]:
    dataset_names = AIME_DATASETS.get(year)
    if not dataset_names:
        raise ValueError(f"No AIME dataset for year {year}. Available: {sorted(AIME_DATASETS.keys())}")

    problems = []
    for ds_name in dataset_names:
        ds = load_dataset(ds_name, split="train")
        for row in ds:
            pt = row.get("problem_type", "unknown")
            if isinstance(pt, list):
                pt = ", ".join(pt) if pt else "unknown"
            problems.append({
                "problem": row["problem"],
                "answer": str(row["answer"]),
                "year": year,
                "source": ds_name,
                "problem_idx": row.get("problem_idx", len(problems)),
                "problem_type": pt,
            })
    return problems


# ── Prompt building ───────────────────────────────────────────────────────

def build_aime_prompt(problem: str, mode: str = "training_nodraft") -> str:
    if mode == "training_nodraft":
        # Matches the GRPO training format the released models were trained on, so the
        # eval prompt matches the training prompt format.
        return (
            f"Problem: {problem}\n\n"
            f"Thinking: N/A\n\n"
            f"The thinking section may contain errors. "
            f"Solve the math problem step by step. "
            f"Write your own correct solution. "
            f"Put your final answer within \\boxed{{}}.\n\n"
            f"Correct Solution:"
        )
    if mode == "instruct":
        # Mistral-style [INST] format — evaluates the instruction-tuned base model
        # (Mathstral-7B-v0.1) in its native chat format. No system message by design;
        # vLLM's tokenizer auto-prepends the BOS <s>.
        return (
            f"[INST] {problem}\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}. [/INST]"
        )
    raise ValueError(f"unknown mode: {mode}")


# ── Scoring ───────────────────────────────────────────────────────────────

def majority_vote(predictions: list[str | None], gold: str) -> bool:
    valid = [p for p in predictions if p is not None]
    if not valid:
        return False
    most_common = Counter(valid).most_common(1)[0][0]
    return math_verify_match(most_common, gold)


def majority_vote_k(predictions: list[str | None], gold: str, k: int, rng) -> bool:
    if k >= len(predictions):
        return majority_vote(predictions, gold)
    sampled = rng.sample(predictions, k)
    return majority_vote(sampled, gold)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def get_k_values(n_samples: int) -> list[int]:
    ks = []
    k = 1
    while k <= n_samples:
        ks.append(k)
        k *= 2
    if ks[-1] != n_samples:
        ks.append(n_samples)
    return ks


def score_sampling(records: list[dict], n_maj_trials: int = 100) -> dict:
    import random
    rng = random.Random(42)

    n = len(records)
    n_samples = records[0]["n_samples"] if records else 0
    ks = get_k_values(n_samples)

    metrics = {}
    for k in ks:
        pass_k = sum(pass_at_k(r["n_samples"], r["n_correct"], k) for r in records) / n if n else 0
        avg_k = sum(r["n_correct"] / r["n_samples"] for r in records) / n if n else 0

        if k == n_samples:
            maj_k = sum(1 for r in records if r["maj_correct"]) / n if n else 0
        elif k == 1:
            maj_k = pass_k
        else:
            maj_sum = 0
            for r in records:
                preds = [s["pred"] for s in r["samples"]]
                wins = sum(1 for _ in range(n_maj_trials) if majority_vote_k(preds, r["gold"], k, rng))
                maj_sum += wins / n_maj_trials
            maj_k = maj_sum / n if n else 0

        metrics[k] = {"pass": pass_k, "avg": avg_k, "maj": maj_k}

    # By year.
    from collections import defaultdict
    by_year = defaultdict(lambda: {k: {"total": 0, "pass_sum": 0.0, "maj_sum": 0.0} for k in ks})
    for r in records:
        yr = r["year"]
        preds = [s["pred"] for s in r["samples"]]
        for k in ks:
            by_year[yr][k]["total"] += 1
            by_year[yr][k]["pass_sum"] += pass_at_k(r["n_samples"], r["n_correct"], k)
            if k == n_samples:
                maj_val = float(r["maj_correct"])
            elif k == 1:
                maj_val = pass_at_k(r["n_samples"], r["n_correct"], 1)
            else:
                wins = sum(1 for _ in range(n_maj_trials) if majority_vote_k(preds, r["gold"], k, rng))
                maj_val = wins / n_maj_trials
            by_year[yr][k]["maj_sum"] += maj_val

    def _fmt_group(group_data):
        result = {}
        for k in ks:
            d = group_data[k]
            t = d["total"]
            result[f"pass@{k}"] = d["pass_sum"] / t if t else 0
            result[f"maj@{k}"] = d["maj_sum"] / t if t else 0
        result["total"] = group_data[ks[0]]["total"]
        return result

    return {
        "n": n,
        "n_samples": n_samples,
        "ks": ks,
        "metrics": {k: metrics[k] for k in ks},
        "by_year": {k: _fmt_group(v) for k, v in sorted(by_year.items())},
    }


def print_scores(scores: dict, label: str = "") -> None:
    N = scores["n_samples"]
    ks = scores["ks"]
    print(f"\n=== AIME Sampling Eval{' — ' + label if label else ''} (N={N}) ===")
    print(f"  Problems:   {scores['n']}")

    header = "  " + "".join(f"{'@'+str(k):>10}" for k in ks)
    print(header)
    pass_row = "  pass" + "".join(f"{scores['metrics'][k]['pass']:10.3f}" for k in ks)
    maj_row = "  maj " + "".join(f"{scores['metrics'][k]['maj']:10.3f}" for k in ks)
    avg_row = "  avg " + f"{scores['metrics'][ks[0]]['avg']:10.3f}" + " (same for all k)"
    print(pass_row)
    print(maj_row)
    print(avg_row)

    print(f"\n  By year:")
    for yr, v in scores["by_year"].items():
        parts = "  ".join(f"p@{k}={v[f'pass@{k}']:.3f} m@{k}={v[f'maj@{k}']:.3f}" for k in ks)
        print(f"    {yr}: {parts}  (n={v['total']})")


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(base_model: str, lora: str | None, args) -> LLM:
    kwargs = dict(
        model=base_model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enforce_eager=True,
    )
    if lora is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = 64
    return LLM(**kwargs)


# ── Eval loop ─────────────────────────────────────────────────────────────

def run_eval(llm: LLM, lora_path: str | None, args, prompts: list[str],
             problems: list[dict], lora_id: int = 1) -> dict:
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    if lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(
            lora_name=Path(lora_path).name,
            lora_int_id=lora_id,
            lora_path=lora_path,
        )
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    records = []
    for prob, out in zip(problems, outputs):
        sample_results = []
        predictions = []
        for completion in out.outputs:
            text = completion.text
            pred = extract_boxed(text)
            correct = math_verify_match(pred, prob["answer"]) if pred else False
            sample_results.append({
                "text": text,
                "pred": pred,
                "correct": correct,
                "length": len(text),
            })
            predictions.append(pred)

        n_correct = sum(1 for s in sample_results if s["correct"])
        maj_correct = majority_vote(predictions, prob["answer"])

        records.append({
            "problem_idx": prob.get("problem_idx"),
            "problem": prob["problem"],
            "gold": prob["answer"],
            "year": prob["year"],
            "source": prob.get("source", "unknown"),
            "problem_type": prob.get("problem_type", "unknown"),
            "n_samples": args.n_samples,
            "n_correct": n_correct,
            "maj_correct": maj_correct,
            "samples": sample_results,
        })

    return {"records": records, "scores": score_sampling(records)}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    all_problems = []
    for year in args.year:
        problems = load_aime_problems(year)
        print(f"Loaded {len(problems)} AIME {year} problems.")
        all_problems.extend(problems)

    print(f"Total: {len(all_problems)} problems across years {args.year}")

    prompts = [build_aime_prompt(p["problem"], mode=args.mode) for p in all_problems]
    print(f"Built {len(prompts)} prompts.")
    print(f"Sampling: N={args.n_samples}, temp={args.temperature}, top_p={args.top_p}")
    print(f"Sample prompt ({len(prompts[0])} chars):")
    print(prompts[0][:400] + "...\n")

    # Resolve checkpoints.
    years_str = "_".join(str(y) for y in args.year)
    if args.sweep:
        sweep_dir = Path(args.sweep)
        checkpoints = sorted(sweep_dir.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[1]))
        if not checkpoints:
            print(f"No checkpoints found under {sweep_dir}")
            sys.exit(1)
        suffix = f"_aime{years_str}_sampling{args.n_samples}"
        targets = [(str(ckpt), sweep_dir / f"eval_{ckpt.name}{suffix}.json") for ckpt in checkpoints]
    else:
        suffix = f"_aime{years_str}_sampling{args.n_samples}"
        if args.output:
            output = Path(args.output)
        else:
            # Include run name (parent dir of checkpoint) to avoid filename collisions
            lora_path = Path(args.lora) if args.lora else None
            if lora_path and lora_path.parent.name != "outputs":
                run_name = lora_path.parent.name
                output = Path(f"outputs/{run_name}/eval_{lora_path.name}{suffix}.json")
            else:
                output = Path(f"outputs/eval_{lora_path.name if lora_path else 'base'}{suffix}.json")
        targets = [(args.lora, output)]

    print(f"Loading base model: {args.base_model}")
    llm = load_model(args.base_model, args.lora or targets[0][0], args)

    for target_idx, (lora_path, output_path) in enumerate(targets, start=1):
        label = Path(lora_path).name if lora_path else "base"
        print(f"\n{'='*60}\nEvaluating: {label} (N={args.n_samples})\n{'='*60}")
        t_start = time.time()
        result = run_eval(llm, lora_path, args, prompts, all_problems, lora_id=target_idx)
        elapsed = time.time() - t_start

        result["meta"] = {
            "base_model": args.base_model,
            "lora": lora_path,
            "years": args.year,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "elapsed_seconds": round(elapsed, 1),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print_scores(result["scores"], label=label)
        print(f"\n  Elapsed: {elapsed:.0f}s")
        print(f"  Saved: {output_path}")

        # Per-problem summary.
        print(f"\n  Per problem:")
        for r in result["records"]:
            mark = "✓" if r["maj_correct"] else "✗"
            print(f"    {mark} #{r['problem_idx']:2d} ({r['year']}) "
                  f"correct={r['n_correct']}/{r['n_samples']}  "
                  f"gold={r['gold']}  type={r['problem_type']}")


if __name__ == "__main__":
    main()
