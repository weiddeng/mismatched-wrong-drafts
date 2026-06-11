"""Build a mismatched-draft training set (the paper's headline construction).

Same per-problem draft selection as build_datasets.py, but the draft text
attached to each problem is taken from a DIFFERENT problem: a random permutation
of the selected drafts, repaired to a derangement (zero fixed points). This is
the "mismatch" step.

``--draft-type wrong``   (default) reproduces the headline mismatched-WRONG set
                         (`mismatched_wrong`, the paper's headline).
``--draft-type correct`` builds the mismatched-CORRECT set (the mismatched-correct config) — correct drafts, shuffled to other problems.

Each row carries:
  - problem/answer/solution/level/subject/split/unique_id from THIS problem
  - prompt with "Thinking: <draft from a different problem>"
  - draft_correct_strict / draft_correct_quasi: correctness of the *injected*
    draft on its own source problem (usually wrong — it was wrong-selected)
  - draft_priority: tier of the injected draft on its source problem
  - mismatched_from: unique_id of the source problem (for analysis)

Usage:
    python scripts/build_mismatch.py \\
        --drafts outputs/drafts_qwen25_math_1.5b/drafts.json \\
        --out data/mismatched_wrong

Reproducibility note: the derangement is a seeded shuffle over the rows in their
current order, so the exact mismatched pairings depend on row order. To reproduce
the *published* dataset's pairings, pass ``--canonical-order <path>`` pointing at
a released dataset. Without it, a fresh valid derangement (same construction,
same distribution) is built.
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

WRONG_PRIORITIES = [
    lambda s: not s["correct_quasi"],                          # definitively wrong
    lambda s: not s["correct_strict"] and s["correct_quasi"],  # right answer, missing \boxed{}
]
CORRECT_PRIORITIES = [
    lambda s: s["correct_strict"],                             # correct + boxed
    lambda s: s["correct_quasi"] and not s["correct_strict"],  # right answer, no \boxed{}
]


def select_with_priority(samples, priorities, rng):
    for tier_idx, pred in enumerate(priorities, start=1):
        cands = [s for s in samples if pred(s)]
        if cands:
            return rng.choice(cands), tier_idx
    return rng.choice(samples), len(priorities) + 1


def parse_args():
    p = argparse.ArgumentParser(description="Build the mismatched-wrong-draft training set.")
    p.add_argument("--drafts", type=str,
                   default="outputs/drafts_qwen25_math_1.5b/drafts.json",
                   help="Path to the drafts.json produced by generate_drafts.py.")
    p.add_argument("--draft-type", choices=["wrong", "correct"], default="wrong",
                   help="'wrong' = mismatched-wrong set (the headline); "
                        "'correct' = mismatched-correct set (the ablation).")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory. Defaults by --draft-type to "
                        "data/mismatched_wrong (wrong) or "
                        "data/mismatched_correct (correct).")
    p.add_argument("--canonical-order", type=str, default=None,
                   help="Optional path to a released dataset; rows are reordered to "
                        "match its problem order so the derangement reproduces the "
                        "published pairings exactly. If omitted, drafts.json order is used.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for draft selection and the derangement shuffle (default 0).")
    return p.parse_args()


def main():
    args = parse_args()
    priorities = WRONG_PRIORITIES if args.draft_type == "wrong" else CORRECT_PRIORITIES
    if args.out is None:
        args.out = ("data/mismatched_wrong" if args.draft_type == "wrong"
                    else "data/mismatched_correct")

    print(f"Loading {args.drafts} ...")
    d = json.load(open(args.drafts))
    records = d["records"]

    clean = [r for r in records
             if r["split"] != "math500"
             and r["level"] in ("Level 3", "Level 4", "Level 5")]
    assert len(clean) == 8888

    # Optional: reorder rows to match a released dataset for exact pairings.
    if args.canonical_order:
        from datasets import load_from_disk
        ref = load_from_disk(args.canonical_order)
        ref_order = list(ref["problem"])
        by_problem = {r["problem"]: r for r in clean}
        clean = [by_problem[p] for p in ref_order]
        print(f"Reordered to match {args.canonical_order}; len={len(clean)}")

    # Step 1: pick one draft per problem by the chosen priority (wrong or correct).
    rng = random.Random(args.seed)
    selections = []  # list of (record, sample, tier)
    for r in clean:
        sample, tier = select_with_priority(r["samples"], priorities, rng)
        selections.append((r, sample, tier))

    # Step 2: random permutation, then repair fixed points into a derangement.
    # Expected ~1 fixed point in a random shuffle of n (independent of n).
    n = len(selections)
    perm = list(range(n))
    rng2 = random.Random(args.seed)
    rng2.shuffle(perm)

    fixed = [i for i in range(n) if perm[i] == i]
    print(f"Fixed points before fixup: {len(fixed)} (expected ~1)")
    for i in fixed:
        for j in range(n):
            if j == i:
                continue
            if perm[j] != i and perm[i] != j:
                perm[i], perm[j] = perm[j], perm[i]
                break
    remaining_fixed = sum(1 for i in range(n) if perm[i] == i)
    print(f"Fixed points after fixup: {remaining_fixed} (must be 0)")
    assert remaining_fixed == 0

    rows = []
    for i, (rec, _, _) in enumerate(selections):
        src_rec, src_sample, src_tier = selections[perm[i]]
        assert src_rec["unique_id"] != rec["unique_id"], "fixed point!"
        rows.append({
            "problem":  rec["problem"],
            "level":    rec["level"],
            "solution": rec["solution"],
            "answer":   rec["answer"],
            "subject":  rec["subject"],
            "split":    rec["split"],
            "unique_id": rec["unique_id"],
            "prompt":   PROMPT_TEMPLATE.format(problem=rec["problem"], draft=src_sample["text"].strip()),
            "has_draft": True,
            "draft_correct_strict": bool(src_sample["correct_strict"]),
            "draft_correct_quasi":  bool(src_sample["correct_quasi"]),
            "draft_priority":       src_tier,
            "mismatched_from":      src_rec["unique_id"],
        })

    print(f"Built {len(rows)} rows")
    same_subject = sum(1 for i, (rec, _, _) in enumerate(selections)
                       if rec["subject"] == selections[perm[i]][0]["subject"])
    same_level = sum(1 for i, (rec, _, _) in enumerate(selections)
                     if rec["level"] == selections[perm[i]][0]["level"])
    print(f"  same-subject pairs: {same_subject}/{len(rows)} ({same_subject/len(rows)*100:.1f}%)")
    print(f"  same-level pairs:   {same_level}/{len(rows)} ({same_level/len(rows)*100:.1f}%)")
    tier_counts = {}
    for r in rows:
        tier_counts[r["draft_priority"]] = tier_counts.get(r["draft_priority"], 0) + 1
    print(f"  injected draft tier distribution: {dict(sorted(tier_counts.items()))}")
    print(f"  draft_correct_strict=True (injected): {sum(1 for r in rows if r['draft_correct_strict'])}")
    print(f"  draft_correct_quasi=True (injected):  {sum(1 for r in rows if r['draft_correct_quasi'])}")

    out = Path(args.out)
    ds = Dataset.from_list(rows)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    ds.save_to_disk(str(out))
    print(f"Saved to {out}")
    print(f"Columns: {ds.column_names}")

    r = ds[0]
    print(f"\n--- row 0 (problem from {r['unique_id']}, draft from {r['mismatched_from']}) ---")
    print(r["prompt"][:1200])
    print("..." if len(r["prompt"]) > 1200 else "")


if __name__ == "__main__":
    main()
