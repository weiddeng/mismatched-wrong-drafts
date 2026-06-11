"""
Evaluate a trained LoRA checkpoint (or base model) on MATH-500 with greedy
decoding, scoring boxed answers against gold via math_verify. Supports two
prompt modes (``training_nodraft``, ``draftgen``), with these conveniences:

  1. Adds the ``draftgen`` prompt mode — the same prompt template used by
     scripts/generate_drafts.py to sample drafts from Qwen2.5-Math-1.5B.
  2. Always appends the prompt mode as a suffix in output filenames (so two
     sweeps with different modes don't overwrite each other):
        eval_<ckpt>_<mode>.json
     and meta now records the mode.
  3. Adds --max-checkpoint and --min-checkpoint range filters for sweeps,
     so we can e.g. evaluate only ckpts ≤ 2200 to skip 2nd-epoch overshoot.

A short self-test of both prompt-mode templates runs on every invocation
(unless --skip-self-test is passed) so any prompt-template regression is caught
before launching a multi-hour sweep.

Usage examples:

    # Single ckpt:
    python scripts/eval_math500.py \
        --base-model mistralai/Mathstral-7B-v0.1 \
        --lora outputs/mismatched_wrong/checkpoint-2000 \
        --mode training_nodraft \
        --max-tokens 4096 --max-model-len 8192

    # Sweep, mode = draftgen, ckpts ≤ 2200:
    python scripts/eval_math500.py \
        --base-model mistralai/Mathstral-7B-v0.1 \
        --sweep outputs/mismatched_wrong \
        --mode draftgen \
        --max-checkpoint 2200 \
        --max-tokens 4096 --max-model-len 8192
"""

from __future__ import annotations

import os

# HF caches default to ~/.cache/huggingface; set HF_HOME to relocate.

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.grpo.rewards import extract_boxed, math_verify_match, mathematically_quasi_correct


# ── CLI ────────────────────────────────────────────────────────────────────

PROMPT_MODES = ["training_nodraft", "draftgen", "instruct"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate model on MATH-500.")
    p.add_argument("--base-model", type=str, default="mistralai/Mathstral-7B-v0.1")
    p.add_argument("--lora", type=str, default=None,
                   help="Path to LoRA checkpoint dir. None = base model.")
    p.add_argument("--sweep", type=str, default=None,
                   help="Directory containing multiple checkpoint-N subdirs.")
    p.add_argument("--mode", type=str, default="training_nodraft",
                   choices=PROMPT_MODES,
                   help="Prompt template to render before each problem.")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path. Ignored with --sweep.")
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-checkpoint", type=int, default=None,
                   help="Sweep filter: include only checkpoints with step ≤ this value.")
    p.add_argument("--min-checkpoint", type=int, default=None,
                   help="Sweep filter: include only checkpoints with step ≥ this value.")
    p.add_argument("--skip-self-test", action="store_true",
                   help="Skip the prompt-mode self-test (not recommended).")
    return p.parse_args()


# ── Prompt building ────────────────────────────────────────────────────────

def build_eval_prompt(problem: str, mode: str) -> str:
    """Build an evaluation prompt with no external draft."""
    if mode == "training_nodraft":
        return (
            f"Problem: {problem}\n\n"
            f"Thinking: N/A\n\n"
            f"The thinking section may contain errors. "
            f"Solve the math problem step by step. "
            f"Write your own correct solution. "
            f"Put your final answer within \\boxed{{}}.\n\n"
            f"Correct Solution:"
        )
    if mode == "draftgen":
        # Verbatim copy of build_prompt() in scripts/generate_drafts.py.
        return (
            "Solve the following math problem step by step. "
            "Put your final answer within \\boxed{}.\n\n"
            f"Problem:\n{problem}\n\nSolution:"
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


# ── Self-test (prompt mode regression check) ───────────────────────────────

def _self_test() -> None:
    """Verify each prompt mode renders as expected for a sample problem.

    This catches regressions in prompt-template wording before we burn
    multiple hours of GPU time on a sweep.
    """
    P = "What is 2+2?"

    # training_nodraft — exact equality
    tn = build_eval_prompt(P, "training_nodraft")
    expected_tn = (
        "Problem: What is 2+2?\n\n"
        "Thinking: N/A\n\n"
        "The thinking section may contain errors. "
        "Solve the math problem step by step. "
        "Write your own correct solution. "
        "Put your final answer within \\boxed{}.\n\n"
        "Correct Solution:"
    )
    assert tn == expected_tn, f"training_nodraft drift:\nexpected:\n{expected_tn!r}\n\ngot:\n{tn!r}"

    # draftgen — must be byte-identical to generate_drafts.py's build_prompt().
    dg = build_eval_prompt(P, "draftgen")
    expected_dg = (
        "Solve the following math problem step by step. "
        "Put your final answer within \\boxed{}.\n\n"
        "Problem:\nWhat is 2+2?\n\nSolution:"
    )
    assert dg == expected_dg, f"draftgen drift:\nexpected:\n{expected_dg!r}\n\ngot:\n{dg!r}"

    # instruct — Mistral [INST] format
    inst = build_eval_prompt(P, "instruct")
    expected_inst = (
        "[INST] What is 2+2?\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}. [/INST]"
    )
    assert inst == expected_inst, f"instruct drift:\nexpected:\n{expected_inst!r}\n\ngot:\n{inst!r}"

    # Bad mode raises
    try:
        build_eval_prompt(P, "garbage")
        raise AssertionError("expected ValueError on unknown mode")
    except ValueError:
        pass

    print("[self_test] All prompt modes verified ✓")


# ── Scoring ────────────────────────────────────────────────────────────────

def score_outputs(records: list[dict]) -> dict:
    n = len(records)
    n_boxed = sum(1 for r in records if r["pred"] is not None)
    n_correct = sum(1 for r in records if r["correct"])
    n_correct_lenient = sum(1 for r in records if r.get("correct_lenient", r["correct"]))
    lengths = [len(r["output"]) for r in records]
    avg_len = sum(lengths) / n if n else 0

    by_subject = defaultdict(lambda: [0, 0, 0])
    by_level = defaultdict(lambda: [0, 0, 0])
    for r in records:
        by_subject[r["subject"]][0] += int(r["correct"])
        by_subject[r["subject"]][1] += 1
        by_subject[r["subject"]][2] += int(r.get("correct_lenient", r["correct"]))
        by_level[r["level"]][0] += int(r["correct"])
        by_level[r["level"]][1] += 1
        by_level[r["level"]][2] += int(r.get("correct_lenient", r["correct"]))

    return {
        "n": n,
        "n_has_boxed": n_boxed,
        "n_correct": n_correct,
        "n_correct_lenient": n_correct_lenient,
        "accuracy": n_correct / n if n else 0,
        "accuracy_lenient": n_correct_lenient / n if n else 0,
        "boxed_rate": n_boxed / n if n else 0,
        "avg_output_length": avg_len,
        "by_subject": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1],
                            "correct_lenient": v[2], "acc_lenient": v[2] / v[1]}
                       for k, v in sorted(by_subject.items())},
        "by_level": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1],
                          "correct_lenient": v[2], "acc_lenient": v[2] / v[1]}
                     for k, v in sorted(by_level.items())},
    }


def print_scores(scores: dict, label: str = "") -> None:
    print(f"\n=== Eval Results{' — ' + label if label else ''} ===")
    print(f"  N:              {scores['n']}")
    print(f"  Accuracy:         {scores['accuracy']:.3f}  ({scores['n_correct']}/{scores['n']})")
    if 'accuracy_lenient' in scores:
        print(f"  Accuracy(lenient):{scores['accuracy_lenient']:.3f}  ({scores['n_correct_lenient']}/{scores['n']})")
    print(f"  Boxed rate:       {scores['boxed_rate']:.3f}")
    print(f"  Avg output len:   {scores['avg_output_length']:.0f} chars")
    print("  By level:")
    for lv, v in scores["by_level"].items():
        lenient_str = f"  lenient={v['acc_lenient']:.3f}" if 'acc_lenient' in v else ""
        print(f"    {lv}: {v['acc']:.3f}  ({v['correct']}/{v['total']}){lenient_str}")
    print("  By subject:")
    for s, v in scores["by_subject"].items():
        lenient_str = f"  lenient={v['acc_lenient']:.3f}" if 'acc_lenient' in v else ""
        print(f"    {s:<20} {v['acc']:.3f}  ({v['correct']}/{v['total']}){lenient_str}")


# ── Model loading ──────────────────────────────────────────────────────────

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


# ── Eval loop ──────────────────────────────────────────────────────────────

def run_eval(llm: LLM, lora_path: str | None, args, prompts: list[str],
             problems: list[dict], lora_id: int = 1) -> dict:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\n\nProblem:", "\nProblem:"],
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
        text = out.outputs[0].text
        pred = extract_boxed(text)
        correct = math_verify_match(pred, prob["answer"]) if pred else False
        correct_lenient = mathematically_quasi_correct(text, prob["answer"])
        records.append({
            "unique_id": prob.get("unique_id"),
            "problem": prob["problem"],
            "gold": prob["answer"],
            "pred": pred,
            "correct": correct,
            "correct_lenient": correct_lenient,
            "output": text,
            "subject": prob.get("subject", "unknown"),
            "level": prob.get("level", "unknown"),
        })

    return {"records": records, "scores": score_outputs(records)}


# ── Main ───────────────────────────────────────────────────────────────────

def _checkpoint_step(p: Path) -> int:
    return int(p.name.split("-")[1])


def main() -> None:
    args = parse_args()

    if not args.skip_self_test:
        _self_test()

    print("Loading MATH-500...")
    eval_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(eval_ds)} problems.")

    problems = [dict(row) for row in eval_ds]
    prompts = [build_eval_prompt(p["problem"], args.mode) for p in problems]
    print(f"Built {len(prompts)} prompts (mode={args.mode}).")
    print(f"Sample prompt ({len(prompts[0])} chars):")
    print(prompts[0][:400] + "...\n")

    # Resolve checkpoints to eval — output filename always includes mode suffix.
    if args.sweep:
        sweep_dir = Path(args.sweep)
        checkpoints = sorted(sweep_dir.glob("checkpoint-*"), key=_checkpoint_step)
        if not checkpoints:
            print(f"No checkpoints found under {sweep_dir}")
            sys.exit(1)
        # Apply step-range filters.
        if args.min_checkpoint is not None:
            checkpoints = [c for c in checkpoints if _checkpoint_step(c) >= args.min_checkpoint]
        if args.max_checkpoint is not None:
            checkpoints = [c for c in checkpoints if _checkpoint_step(c) <= args.max_checkpoint]
        if not checkpoints:
            print(f"All checkpoints filtered out by --min/--max-checkpoint")
            sys.exit(1)
        print(f"Sweeping {len(checkpoints)} checkpoints "
              f"(steps {_checkpoint_step(checkpoints[0])}..{_checkpoint_step(checkpoints[-1])}).")
        targets = [(str(ckpt), sweep_dir / f"eval_{ckpt.name}_{args.mode}.json") for ckpt in checkpoints]
    else:
        if args.output:
            output = Path(args.output)
        else:
            base = Path(args.lora).name if args.lora else "base"
            output = Path(f"outputs/eval_{base}_{args.mode}.json")
        targets = [(args.lora, output)]

    print(f"Loading base model: {args.base_model}")
    llm = load_model(args.base_model, args.lora or targets[0][0], args)

    for target_idx, (lora_path, output_path) in enumerate(targets, start=1):
        label = Path(lora_path).name if lora_path else "base"
        print(f"\n{'='*60}\nEvaluating: {label}  (mode={args.mode})\n{'='*60}")
        t_start = time.time()
        result = run_eval(llm, lora_path, args, prompts, problems, lora_id=target_idx)
        elapsed = time.time() - t_start

        result["meta"] = {
            "base_model": args.base_model,
            "lora": lora_path,
            "mode": args.mode,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
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


if __name__ == "__main__":
    main()
