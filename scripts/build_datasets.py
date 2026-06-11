"""Build the matched (non-shuffled) training datasets from a drafts.json.

Produces three Hugging Face ``save_to_disk`` directories:
  - matched_wrong   (priority: quasi=False > strict=False&quasi=True > random)
  - no_draft           (no draft, ``Thinking: N/A``)
  - matched_correct (priority: strict=True > quasi=True&strict=False > random)

Universe: 8,888 clean Level 3-5 problems in (MATH \\ MATH-500). Per problem we
pick one draft from the 32 Qwen2.5-Math-1.5B samples per the priority chain
above. Reproducible (seed=0).

The mismatched-wrong dataset (the paper's headline) is built separately by
``build_mismatch.py``.

Usage:
    python scripts/build_datasets.py \\
        --drafts outputs/drafts_qwen25_math_1.5b/drafts.json \\
        --out-dir data

To reproduce the *exact* row order of the published datasets, pass
``--canonical-order <path>`` pointing at any released dataset (they share one
canonical order). Omitting it builds an equivalent dataset in drafts.json order.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset

PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Thinking: {draft}\n\n"
    "The thinking section may contain errors. "
    "Solve the math problem step by step. "
    "Write your own correct solution. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Correct Solution:"
)


def select_with_priority(samples, priorities, rng):
    """Try each priority predicate in order; return (sample, tier)."""
    for tier_idx, pred in enumerate(priorities, start=1):
        candidates = [s for s in samples if pred(s)]
        if candidates:
            return rng.choice(candidates), tier_idx
    # Fallback: random across all
    return rng.choice(samples), len(priorities) + 1


WRONG_PRIORITIES = [
    lambda s: not s["correct_quasi"],                          # P1: definitively wrong
    lambda s: not s["correct_strict"] and s["correct_quasi"],  # P2: right answer, missing \boxed{}
    # P3 fallback handled by select_with_priority
]
CORRECT_PRIORITIES = [
    lambda s: s["correct_strict"],                             # P1: correct + boxed
    lambda s: s["correct_quasi"] and not s["correct_strict"],  # P2: right answer, no \boxed{}
]


def build_row(rec, mode, rng):
    base = {
        "problem": rec["problem"],
        "level": rec["level"],
        "solution": rec["solution"],
        "answer": rec["answer"],
        "subject": rec["subject"],
        "split": rec["split"],
        "unique_id": rec["unique_id"],
    }
    if mode == "nodraft":
        base["prompt"] = PROMPT_TEMPLATE.format(problem=rec["problem"], draft="N/A")
        base["has_draft"] = False
        base["draft_correct_strict"] = None
        base["draft_correct_quasi"] = None
        base["draft_priority"] = 0
        return base

    if mode == "wrongdraft":
        priorities = WRONG_PRIORITIES
    elif mode == "correctdraft":
        priorities = CORRECT_PRIORITIES
    else:
        raise ValueError(mode)

    sample, tier = select_with_priority(rec["samples"], priorities, rng)
    base["prompt"] = PROMPT_TEMPLATE.format(problem=rec["problem"], draft=sample["text"].strip())
    base["has_draft"] = True
    base["draft_correct_strict"] = bool(sample["correct_strict"])
    base["draft_correct_quasi"] = bool(sample["correct_quasi"])
    base["draft_priority"] = tier
    return base


def parse_args():
    p = argparse.ArgumentParser(description="Build matched training datasets.")
    p.add_argument("--drafts", type=str,
                   default="outputs/drafts_qwen25_math_1.5b/drafts.json",
                   help="Path to the drafts.json produced by generate_drafts.py.")
    p.add_argument("--out-dir", type=str, default="data",
                   help="Directory to write the dataset folders into.")
    p.add_argument("--canonical-order", type=str, default=None,
                   help="Optional path to a released dataset; rows are reordered to "
                        "match its problem order for an exact reproduction. If omitted, "
                        "drafts.json order is used.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for per-problem draft selection (default 0).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"Loading {args.drafts} ...")
    d = json.load(open(args.drafts))
    records = d["records"]
    print(f"Total records: {len(records)}")

    # Universe filter: clean L3-5, excluding MATH-500.
    clean = [r for r in records
             if r["split"] != "math500"
             and r["level"] in ("Level 3", "Level 4", "Level 5")]
    print(f"Clean L3-5 in (MATH \\ MATH-500): {len(clean)}")
    assert len(clean) == 8888, f"expected 8888, got {len(clean)}"

    # Optional: reorder rows to match a released dataset's canonical order, so
    # the produced datasets are row-aligned with the published ones.
    if args.canonical_order:
        from datasets import load_from_disk
        ref = load_from_disk(args.canonical_order)
        ref_order = list(ref["problem"])
        by_problem = {r["problem"]: r for r in clean}
        assert len(by_problem) == len(clean), "duplicate problem texts in clean records!"
        missing = [p for p in ref_order if p not in by_problem]
        extra = [p for p in by_problem if p not in ref_order]
        assert not missing and not extra, \
            f"canonical-order mismatch: missing={len(missing)} extra={len(extra)}"
        clean = [by_problem[p] for p in ref_order]
        print(f"Reordered {len(clean)} rows to match {args.canonical_order}.")

    for mode, name in [
        ("wrongdraft",   "matched_wrong"),
        ("nodraft",      "no_draft"),
        ("correctdraft", "matched_correct"),
    ]:
        rng = random.Random(args.seed)
        rows = [build_row(r, mode, rng) for r in clean]

        if mode != "nodraft":
            tier_counts = {1: 0, 2: 0, 3: 0}
            strict_count = sum(1 for r in rows if r["draft_correct_strict"])
            quasi_count = sum(1 for r in rows if r["draft_correct_quasi"])
            for r in rows:
                tier_counts[r["draft_priority"]] = tier_counts.get(r["draft_priority"], 0) + 1
            print(f"\n=== {name} ===")
            print(f"  rows: {len(rows)}")
            print(f"  draft_priority tiers (1/2/3): {tier_counts}")
            print(f"  draft_correct_strict=True:   {strict_count} ({strict_count/len(rows)*100:.1f}%)")
            print(f"  draft_correct_quasi=True:    {quasi_count} ({quasi_count/len(rows)*100:.1f}%)")
        else:
            print(f"\n=== {name} ===")
            print(f"  rows: {len(rows)}")

        ds = Dataset.from_list(rows)
        out_path = out_dir / name
        if out_path.exists():
            import shutil
            shutil.rmtree(out_path)
        ds.save_to_disk(str(out_path))
        print(f"  saved to {out_path}")
        print(f"  columns: {ds.column_names}")

    # Sample prompts.
    print("\n\n=== SAMPLE PROMPTS (row 0 of each) ===")
    from datasets import load_from_disk
    for name in ["matched_wrong", "no_draft", "matched_correct"]:
        ds = load_from_disk(str(out_dir / name))
        r = ds[0]
        print(f"\n--- {name} (row 0, level={r['level']}, priority={r.get('draft_priority')}) ---")
        print(r["prompt"][:1200])
        print("..." if len(r["prompt"]) > 1200 else "")


if __name__ == "__main__":
    main()
