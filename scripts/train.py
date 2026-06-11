"""
Dr.GRPO training on MATH — Path-B (draft-augmented) only.

The training dataset has the draft already baked into the ``prompt`` column.
Standard ``GRPOTrainer`` generates 16 completions per prompt directly —
no custom ``rollout_func`` needed.

Usage:
    python scripts/train.py --model mistralai/Mathstral-7B-v0.1 \\
        --dataset-path data/mismatched_wrong --smoke      # quick 2-step sanity run
    python scripts/train.py \\
        --model mistralai/Mathstral-7B-v0.1 \\
        --dataset-path data/mismatched_wrong \\
        --output-dir outputs/mismatched_wrong \\
        --max-steps 2222 \\
        --max-completion-length 4096
"""

from __future__ import annotations

import os

# HuggingFace caches default to ~/.cache/huggingface. To relocate (e.g. to a
# large scratch disk), set HF_HOME in your environment before running.

import argparse
import logging
import sys
import time
from pathlib import Path

# Append project root so `src.grpo` is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("train")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dr.GRPO Path-B training on MATH.")
    p.add_argument("--model", type=str, required=True,
                   help="Base learner model (e.g. mistralai/Mathstral-7B-v0.1).")
    p.add_argument("--dataset-path", type=str, default="data/math_path_b")
    p.add_argument("--output-dir", type=str, default="outputs/dry_run")
    p.add_argument("--max-seq-length", type=int, default=7168)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--num-generations", type=int, default=16,
                   help="Number of rollouts per prompt (default 16).")
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--lr-scheduler-type", type=str, default="cosine")
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--correction-bonus", type=float, default=1.0,
                   help="Bonus reward when model corrects a wrong/null draft.")
    p.add_argument("--copy-penalty", type=float, default=0.0,
                   help="Penalty when model copies a wrong draft's boxed answer.")
    p.add_argument("--corrupt-penalty", type=float, default=0.0,
                   help="Penalty when model gets it wrong despite a correct draft.")
    p.add_argument("--adam-beta2", type=float, default=0.999,
                   help="Adam beta2 parameter (default 0.999).")
    p.add_argument("--save-steps", type=int, default=100,
                   help="Save checkpoint every N steps (default 100).")
    p.add_argument("--fadeout-schedule", type=str, default=None,
                   choices=[None, "cosine", "linear"],
                   help="Draft fadeout schedule. None=no fadeout (all drafts kept).")
    p.add_argument("--max-prompt-tokens", type=int, default=0,
                   help="Truncate prompts to this many tokens (0=no truncation). "
                        "Useful when the training model's tokenizer produces more "
                        "tokens than the dataset-prep tokenizer. Truncates the draft "
                        "portion of the prompt to fit within budget.")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume training from.")
    p.add_argument("--qlora", action="store_true",
                   help="Use 4-bit quantized base model (QLoRA) to save VRAM.")
    p.add_argument("--smoke", action="store_true",
                   help="Override to max_steps=2, max_completion_length=512, 20 rows.")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Smoke overrides.
    if args.smoke:
        args.max_steps = 2
        args.max_completion_length = 512

    # ── 1. Load dataset ──────────────────────────────────────────────────
    from datasets import load_from_disk

    logger.info("Loading dataset from: %s", args.dataset_path)
    train_dataset = load_from_disk(args.dataset_path)
    if args.smoke:
        train_dataset = train_dataset.select(range(min(20, len(train_dataset))))

    print(f"\n=== Dataset ===")
    print(f"  Path:           {args.dataset_path}")
    print(f"  Rows:           {len(train_dataset)}")
    print(f"  Columns:        {train_dataset.column_names}")
    sample = train_dataset[0]
    print(f"  Sample prompt:  {sample['prompt'][:300]}...")
    print(f"  Sample answer:  {sample['answer']}")

    # ── 1a. Truncate prompts if --max-prompt-tokens is set ──────────────
    if args.max_prompt_tokens > 0:
        from transformers import AutoTokenizer

        _trunc_tokenizer = AutoTokenizer.from_pretrained(args.model)
        _max_prompt_tokens = args.max_prompt_tokens

        def _truncate_prompt(prompt: str, max_tokens: int) -> str:
            """Truncate the draft portion of a prompt to fit within max_tokens.

            The prompt format is:
                Problem: {problem}

                Thinking: {draft}

                The thinking section may contain errors. ...

                Correct Solution:

            We only truncate the draft text between "Thinking: " and
            "\\n\\nThe thinking section", preserving the problem and
            instruction parts.
            """
            token_ids = _trunc_tokenizer.encode(prompt)
            if len(token_ids) <= max_tokens:
                return prompt

            draft_start_marker = "Thinking: "
            draft_end_marker = "\n\nThe thinking section"
            ds = prompt.find(draft_start_marker)
            de = prompt.find(draft_end_marker)

            if ds == -1 or de == -1:
                # No draft markers — fall back to hard truncation
                return _trunc_tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

            # Measure tokens for the non-draft parts (problem + instruction)
            prefix = prompt[:ds + len(draft_start_marker)]
            suffix = prompt[de:]
            non_draft_tokens = len(_trunc_tokenizer.encode(prefix)) + len(_trunc_tokenizer.encode(suffix))
            draft_budget = max_tokens - non_draft_tokens

            if draft_budget <= 0:
                # Even without draft, prompt exceeds budget — use N/A
                return prefix + "N/A" + suffix

            # Truncate the draft text to fit the budget
            draft_text = prompt[ds + len(draft_start_marker):de]
            draft_token_ids = _trunc_tokenizer.encode(draft_text)

            if len(draft_token_ids) <= draft_budget:
                return prompt  # draft fits, no truncation needed

            truncated_draft = _trunc_tokenizer.decode(
                draft_token_ids[:draft_budget], skip_special_tokens=True
            )
            return prefix + truncated_draft + suffix

        n_truncated = 0
        def _truncate_row(row):
            nonlocal n_truncated
            new_prompt = _truncate_prompt(row["prompt"], _max_prompt_tokens)
            if new_prompt != row["prompt"]:
                n_truncated += 1
            return {"prompt": new_prompt}

        train_dataset = train_dataset.map(_truncate_row)
        logger.info(
            "Prompt truncation: max_prompt_tokens=%d, truncated %d/%d prompts",
            _max_prompt_tokens, n_truncated, len(train_dataset),
        )

    # ── 1b. Wrap dataset with draft fadeout if requested ─────────────────
    from transformers import TrainerCallback
    if args.fadeout_schedule:
        import math
        import random as _random
        import torch.utils.data

        def _replace_draft_with_na(prompt: str) -> str:
            """Replace the draft in a prompt with N/A."""
            marker = "Thinking: "
            end_marker = "\n\nThe thinking section"
            s = prompt.find(marker)
            e = prompt.find(end_marker)
            if s == -1 or e == -1:
                return prompt
            return prompt[:s + len(marker)] + "N/A" + prompt[e:]

        class FadeoutDataset(torch.utils.data.Dataset):
            """Wraps a dataset and randomly replaces drafts with N/A."""
            def __init__(self, base_dataset):
                self.base = base_dataset
                self.draft_prob = 1.0  # updated by FadeoutCallback
                self.column_names = base_dataset.column_names

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                item = dict(self.base[idx])
                if _random.random() > self.draft_prob:
                    item["prompt"] = _replace_draft_with_na(item["prompt"])
                return item

        class FadeoutCallback(TrainerCallback):
            def __init__(self, dataset, max_steps, schedule="cosine"):
                self.dataset = dataset
                self.max_steps = max_steps
                self.schedule = schedule

            def on_step_begin(self, args, state, control, **kwargs):
                progress = state.global_step / self.max_steps
                if self.schedule == "cosine":
                    self.dataset.draft_prob = 0.5 * (1 + math.cos(math.pi * progress))
                else:  # linear
                    self.dataset.draft_prob = 1.0 - progress
                if state.global_step % 50 == 0:
                    logger.info(
                        "Fadeout: step=%d, draft_prob=%.3f (%s)",
                        state.global_step, self.dataset.draft_prob, self.schedule,
                    )

        train_dataset = FadeoutDataset(train_dataset)
        fadeout_cb = FadeoutCallback(train_dataset, args.max_steps, args.fadeout_schedule)
        logger.info("Draft fadeout enabled: schedule=%s", args.fadeout_schedule)
    else:
        fadeout_cb = None

    # ── 2. Load model with Unsloth ───────────────────────────────────────
    from unsloth import FastLanguageModel

    logger.info("Loading model: %s", args.model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.qlora,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Parameter counts.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable_params / total_params if total_params else 0

    print(f"\n=== Model ===")
    print(f"  Name:           {args.model}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Trainable:      {trainable_params:,} / {total_params:,} params ({pct:.2f}%)")

    # ── 3. Training config ───────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        loss_type="dr_grpo",
        scale_rewards=False,
        beta=args.beta,
        adam_beta2=args.adam_beta2,
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
    )

    effective_batch = args.gradient_accumulation_steps  # per_device_train_batch_size=1

    print(f"\n=== Command ===")
    print(f"  {' '.join(sys.argv)}")
    print(f"\n=== Training Config ===")
    print(f"  max_steps:              {training_args.max_steps}")
    print(f"  num_generations:        {training_args.num_generations}")
    print(f"  gradient_accumulation:  {training_args.gradient_accumulation_steps}")
    print(f"  max_completion_length:  {training_args.max_completion_length}")
    print(f"  learning_rate:          {training_args.learning_rate}")
    print(f"  loss_type:              {training_args.loss_type}")
    print(f"  scale_rewards:          {training_args.scale_rewards}")
    print(f"  beta:                   {training_args.beta}")
    print(f"  effective_batch:        {effective_batch} problems/step")
    print(f"  correction_bonus:      {args.correction_bonus}")
    print(f"  copy_penalty:          {args.copy_penalty}")
    print(f"  corrupt_penalty:       {args.corrupt_penalty}")
    print(f"  output_dir:             {args.output_dir}")
    print(f"  fadeout_schedule:      {args.fadeout_schedule or 'none'}")
    print(f"  qlora (4-bit):        {args.qlora}")
    print(f"  lora_rank:            {args.lora_rank}")
    print(f"  gpu_mem_util:         {args.gpu_mem_util}")
    print(f"  max_seq_length:       {args.max_seq_length}")
    print(f"  lr_scheduler_type:    {args.lr_scheduler_type}")
    print(f"  adam_beta2:           {args.adam_beta2}")
    print(f"  model:                {args.model}")
    print(f"  dataset_path:         {args.dataset_path}")
    print(f"  max_prompt_tokens:    {args.max_prompt_tokens or 'disabled'}")
    if args.resume_from_checkpoint:
        print(f"  resume_from_checkpoint: {args.resume_from_checkpoint}")
    print()

    # ── 4. Reward wrapper with logging ───────────────────────────────────
    step_counter = [0]
    rollout_path = Path(args.output_dir) / "rollout_samples.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    ROLLOUT_EVERY = 1   # save samples every reward call
    ROLLOUT_N_PER_GROUP = 1  # samples to save per problem group

    def _detect_repetition(text: str, window: int = 100, threshold: int = 3) -> bool:
        """Detect repetition by checking if any window-sized substring repeats."""
        if len(text) < window * 2:
            return False
        for w in [50, 100, 150]:
            if len(text) < w * 2:
                continue
            chunks = [text[i:i+w] for i in range(0, len(text) - w, w)]
            from collections import Counter as _Counter
            if _Counter(chunks).most_common(1)[0][1] >= threshold:
                return True
        return False

    def _detect_pathologies(text: str) -> dict:
        """Detect pathologies in a draft or completion."""
        is_empty = len(text.strip()) == 0
        has_boxed = "\\boxed{" in text
        has_repetition = _detect_repetition(text)
        has_python = "```python" in text or "```py" in text
        return {
            "empty": is_empty,
            "has_boxed": has_boxed,
            "repetition": has_repetition,
            "has_python": has_python,
        }

    def _ngram_overlap(text_a: str, text_b: str, n: int = 4) -> float:
        """Fraction of text_b's word n-grams that appear in text_a (containment)."""
        words_a = text_a.lower().split()
        words_b = text_b.lower().split()
        if len(words_b) < n:
            return 0.0
        ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
        ngrams_b = [tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1)]
        if not ngrams_b:
            return 0.0
        return sum(1 for ng in ngrams_b if ng in ngrams_a) / len(ngrams_b)

    short_completion_log_count = [0]  # mutable counter for short completion logging

    def logged_math_reward(completions, **kwargs):
        import json as _json
        from src.grpo.rewards import (
            extract_boxed, extract_mathematical_answer,
            math_verify_match, mathematically_quasi_correct,
        )

        step_counter[0] += 1

        n = len(completions)

        # Log short completions (first 20 occurrences only)
        if short_completion_log_count[0] < 25:
            for i, c in enumerate(completions):
                if len(c) < 10 and short_completion_log_count[0] < 25:
                    logger.warning("Short completion (step %d, idx %d, %d chars): %r",
                                   step_counter[0], i, len(c), c[:100])
                    short_completion_log_count[0] += 1

        prompts = kwargs.get("prompts", [])
        gold = kwargs.get("answer") or kwargs.get("ground_truth") or []
        if isinstance(gold, str):
            gold = [gold] * n
        levels = kwargs.get("level", [None] * n)
        subjects = kwargs.get("subject", [None] * n)
        if isinstance(levels, str):
            levels = [levels] * n
        if isinstance(subjects, str):
            subjects = [subjects] * n

        # Compute base rewards using lenient extraction (extract_mathematical_answer).
        rewards = []
        for i in range(n):
            gold_i = gold[i] if i < len(gold) else None
            correct = mathematically_quasi_correct(completions[i], gold_i) if gold_i else False
            rewards.append(1.0 if correct else 0.0)

        # Per-completion draft correctness, correction bonus, copy penalty, corrupt penalty.
        # Use precomputed draft_correct from dataset if available (e.g., when drafts are masked).
        precomputed_dc = kwargs.get("draft_correct", None)
        if isinstance(precomputed_dc, bool) or precomputed_dc is None:
            precomputed_dc = [precomputed_dc] * n
        elif isinstance(precomputed_dc, list) and len(precomputed_dc) == n:
            pass  # already per-completion

        draft_correct_flags = []  # True/False/None per completion
        n_bonus = 0
        n_penalty = 0
        n_corrupt = 0
        for i in range(n):
            prompt_i = prompts[i] if i < len(prompts) else ""
            gold_i = gold[i] if i < len(gold) else None

            # Extract draft text from prompt (needed for copy detection regardless).
            marker = "Thinking: "
            end_marker = "\n\nThe thinking section"
            s, e = prompt_i.find(marker), prompt_i.find(end_marker)
            if s != -1 and e != -1:
                draft_text_i = prompt_i[s + len(marker):e]
            else:
                draft_text_i = ""

            # Use precomputed draft_correct if available (needed when drafts have masked \boxed).
            if precomputed_dc is not None and isinstance(precomputed_dc, list):
                dc_val = precomputed_dc[i]
                if dc_val is None:
                    dc = None  # no-draft problem
                else:
                    dc = bool(dc_val)
            else:
                # Fallback: extract draft correctness from prompt text.
                if draft_text_i.strip() in ("N/A", ""):
                    dc = None
                else:
                    dc = mathematically_quasi_correct(draft_text_i, gold_i) if gold_i else None
            draft_correct_flags.append(dc)

            # Draft's extracted answer (for copy detection).
            draft_pred = extract_mathematical_answer(draft_text_i) if draft_text_i else None

            # Correction bonus: draft wrong/null but model correct.
            if rewards[i] > 0 and dc is not True and args.correction_bonus > 0:
                rewards[i] += args.correction_bonus
                n_bonus += 1

            # Copy penalty: draft wrong and model copies draft's answer.
            if dc is False and args.copy_penalty > 0:
                comp_pred = extract_mathematical_answer(completions[i])
                if comp_pred and draft_pred and math_verify_match(comp_pred, draft_pred):
                    rewards[i] -= args.copy_penalty
                    n_penalty += 1

            # Corrupt penalty: draft was correct but model got it wrong.
            if dc is True and rewards[i] <= 0 and args.corrupt_penalty > 0:
                rewards[i] -= args.corrupt_penalty
                n_corrupt += 1

        # Aggregate stats for logging.
        n_correct = sum(1 for r in rewards if r > 0)
        has_boxed = sum(1 for c in completions if "\\boxed{" in c)
        avg_len = sum(len(c) for c in completions) / n if n else 0

        # Per-group (problem) stats: all-correct vs all-wrong vs mixed.
        num_gen = args.num_generations
        n_groups = n // num_gen
        n_all_correct = 0
        n_all_wrong = 0
        n_mixed = 0
        group_categories = []  # 'all_correct', 'all_wrong', 'mixed' per group
        for g in range(n_groups):
            group_rewards = rewards[g * num_gen : (g + 1) * num_gen]
            if all(r > 0 for r in group_rewards):
                n_all_correct += 1
                group_categories.append('all_correct')
            elif all(r <= 0 for r in group_rewards):
                n_all_wrong += 1
                group_categories.append('all_wrong')
            else:
                n_mixed += 1
                group_categories.append('mixed')

        # Summarize draft status across all problems in this batch.
        problem_flags = draft_correct_flags[::num_gen] if draft_correct_flags else []
        n_prob_correct = sum(1 for f in problem_flags if f is True)
        n_prob_wrong = sum(1 for f in problem_flags if f is False)
        n_prob_none = sum(1 for f in problem_flags if f is None)
        tag = f"drafts[C={n_prob_correct},W={n_prob_wrong},N={n_prob_none}]"
        n_mc = sum(1 for r in rewards if r > 0)
        transition = f"{tag}"

        # Per-level breakdown.
        from collections import defaultdict as _defaultdict
        level_counts = _defaultdict(int)  # level -> n_problems
        level_correct = _defaultdict(int)  # level -> n_correct_completions
        level_total = _defaultdict(int)    # level -> n_total_completions
        level_groups = _defaultdict(lambda: _defaultdict(int))  # level -> {all_correct/all_wrong/mixed -> count}
        for g in range(n_groups):
            lv = levels[g * num_gen] if g * num_gen < len(levels) else "?"
            lv_short = str(lv).replace("Level ", "L") if lv else "?"
            level_counts[lv_short] += 1
            group_rewards = rewards[g * num_gen : (g + 1) * num_gen]
            gc = sum(1 for r in group_rewards if r > 0)
            level_correct[lv_short] += gc
            level_total[lv_short] += num_gen
            level_groups[lv_short][group_categories[g]] += 1

        levels_tag = "levels[" + ",".join(f"{lv}:{level_counts[lv]}" for lv in sorted(level_counts)) + "]"
        transition += f" {levels_tag} → correct {n_mc}/{n}"

        # Per-level correct rates.
        level_parts = []
        for lv in sorted(level_correct):
            c, t = level_correct[lv], level_total[lv]
            pct_lv = 100 * c / t if t else 0
            level_parts.append(f"{lv}={pct_lv:.0f}%({c}/{t})")
        level_acc_tag = " [" + ",".join(level_parts) + "]"

        if n_bonus > 0:
            transition += f" (bonus:{n_bonus})"
        if n_penalty > 0:
            transition += f" (copy-pen:{n_penalty})"
        if n_corrupt > 0:
            transition += f" (corrupt-pen:{n_corrupt})"

        # Model groups with level info.
        def _group_level_str(cat):
            parts = [f"{lv}:{cnt}" for lv, cnt in sorted(level_groups.items()) if cnt.get(cat, 0) > 0]
            return "(" + ",".join(f"{lv}" for lv in sorted(lv for lv in level_groups if level_groups[lv].get(cat, 0) > 0)) + ")" if parts else ""

        groups_tag = (
            f"model_groups[all✓={n_all_correct}{_group_level_str('all_correct')},"
            f" all✗={n_all_wrong}{_group_level_str('all_wrong')},"
            f" mixed={n_mixed}{_group_level_str('mixed')}]"
        )

        logger.info(
            "Reward call %d: %d completions, %d correct (%.1f%%), "
            "%d with \\boxed (%.1f%%), avg_len=%.0f chars | %s%s"
            " | %s",
            step_counter[0], n, n_correct, 100 * n_correct / n,
            has_boxed, 100 * has_boxed / n, avg_len, transition, level_acc_tag,
            groups_tag,
        )
        if step_counter[0] <= 3:
            logger.info("Sample completion: %.300s...", completions[0])

        # Save rollout samples periodically.
        if step_counter[0] % ROLLOUT_EVERY == 0 or step_counter[0] <= 3:
            with open(rollout_path, "a") as f:
                # Sample ROLLOUT_N_PER_GROUP from each problem group (every 16th)
                sample_indices = []
                for g in range(0, n, num_gen):
                    sample_indices.extend(range(g, min(g + ROLLOUT_N_PER_GROUP, n)))
                for i in sample_indices:
                    model_correct = bool(rewards[i] > 0)
                    dc = draft_correct_flags[i]
                    # Draft metadata (extract first so we can use for transition).
                    prompt_i = prompts[i] if i < len(prompts) else ""
                    marker = "Thinking: "
                    end_marker = "\n\nThe thinking section"
                    s, e = prompt_i.find(marker), prompt_i.find(end_marker)
                    if s != -1 and e != -1:
                        draft_text = prompt_i[s + len(marker):e]
                    else:
                        draft_text = ""
                    draft_pathologies = _detect_pathologies(draft_text)

                    # Determine transition tag (distinguishing N/A vs no_boxed).
                    if dc is True:
                        transition_tag = "correct→correct" if model_correct else "correct→wrong"
                    elif dc is False:
                        transition_tag = "wrong→correct" if model_correct else "wrong→wrong"
                    else:
                        # dc is None: either "N/A" draft or draft exists but has no \boxed
                        stripped = draft_text.strip()
                        if stripped == "N/A" or stripped == "":
                            none_kind = "N/A"
                        else:
                            none_kind = "no_boxed"
                        transition_tag = f"{none_kind}→correct" if model_correct else f"{none_kind}→wrong"

                    # Completion metadata.
                    comp = completions[i]
                    comp_pathologies = _detect_pathologies(comp)

                    # Check if completion copies the draft answer.
                    draft_answer = extract_boxed(draft_text) if draft_text else None
                    comp_answer = extract_boxed(comp)
                    if draft_answer and comp_answer:
                        copies_draft = bool(math_verify_match(draft_answer, comp_answer))
                    else:
                        copies_draft = None

                    sample = {
                        "reward_call": step_counter[0],
                        "index": i,
                        "level": levels[i] if i < len(levels) else None,
                        "subject": subjects[i] if i < len(subjects) else None,
                        "reward": rewards[i],
                        "draft_correct": dc,
                        "model_correct": model_correct,
                        "transition": transition_tag,
                        "copies_draft": copies_draft,
                        "draft_overlap": round(_ngram_overlap(draft_text, comp), 3) if draft_text else None,
                        # Draft metadata.
                        "draft_len": len(draft_text),
                        "draft_n_boxed": draft_text.count("\\boxed{"),
                        "draft_has_python": draft_pathologies["has_python"],
                        "draft_repetition": draft_pathologies["repetition"],
                        "draft_empty": draft_pathologies["empty"],
                        # Completion metadata.
                        "completion_len": len(comp),
                        "comp_n_boxed": comp.count("\\boxed{"),
                        "comp_has_python": comp_pathologies["has_python"],
                        "comp_repetition": comp_pathologies["repetition"],
                        "comp_has_boxed": "\\boxed{" in comp,
                        # Full text at the end for inspection.
                        "gold": gold[i] if i < len(gold) else None,
                        "pred": comp_answer,
                        "prompt": prompt_i,
                        "draft": draft_text,
                        "completion": comp,
                    }
                    f.write(_json.dumps(sample) + "\n")

        return rewards

    # ── 4b. Colored log callback ────────────────────────────────────────
    from transformers import TrainerCallback

    class ColorLogCallback(TrainerCallback):
        """Reprint the TRL log dict with color highlights on key metrics."""

        G = "\033[92m"  # green
        R = "\033[91m"  # red
        Y = "\033[93m"  # yellow
        E = "\033[0m"   # reset

        def _color_val(self, key, val):
            """Return colored string for highlighted keys, plain otherwise."""
            s = f"{val:.4f}" if isinstance(val, float) else str(val)
            if key == "reward":
                c = self.G if val >= 0.5 else self.R if val < 0.3 else self.Y
                return f"{c}{s}{self.E}"
            if key == "frac_reward_zero_std":
                c = self.R if val >= 0.5 else self.Y if val >= 0.25 else self.G
                return f"{c}{s}{self.E}"
            if key == "kl":
                c = self.R if val > 1.0 else self.Y if val > 0.1 else self.E
                return f"{c}{s}{self.E}"
            if key in ("completions/mean_length", "completions/min_length"):
                c = self.R if val <= 10 else self.Y if val <= 100 else self.G
                return f"{c}{s}{self.E}"
            if key == "entropy":
                c = self.R if val < 0.5 else self.Y if val < 1.0 else self.G
                return f"{c}{s}{self.E}"
            return s

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or "reward" not in logs:
                return
            parts = []
            for k, v in logs.items():
                parts.append(f"'{k}': {self._color_val(k, v)}")
            print("{" + ", ".join(parts) + "}", flush=True)

    # ── 5. Entropy callback ────────────────────────────────────────────
    class EntropyCallback(TrainerCallback):
        """Compute per-token entropy every N steps on a small sample."""

        def __init__(self, every_n_steps: int = 10, n_samples: int = 4, max_len: int = 512):
            self.every_n_steps = every_n_steps
            self.n_samples = n_samples
            self.max_len = max_len
            self.history = []

        _first_run = True

        def on_step_end(self, args, state, control, model=None, **kwargs):
            step = state.global_step
            # Run on first step (for verification), then every N steps.
            run_now = self._first_run or (step % self.every_n_steps == 0)
            if not run_now:
                return
            try:
                self._compute_entropy(model, step)
                self._first_run = False
            except Exception as e:
                logger.warning("EntropyCallback failed at step %d: %s", step, e, exc_info=True)

        def _compute_entropy(self, model, step):
            import torch
            from torch.distributions import Categorical

            # Grab a few prompts from the training dataset
            indices = list(range(min(self.n_samples, len(train_dataset))))
            prompts_text = [train_dataset[i]["prompt"] for i in indices]

            # Tokenize prompts
            inputs = tokenizer(
                prompts_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_len,
                padding=True,
            ).to(next(model.parameters()).device)

            was_training = model.training
            model.eval()

            with torch.no_grad():
                # Generate short completions and capture per-step logits
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    output_logits=True,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # outputs.logits is a tuple of (batch, vocab) per generated step
            stacked_logits = torch.stack(outputs.logits, dim=1)  # (batch, gen_len, vocab)

            # Numerically stable entropy via Categorical
            token_entropy = Categorical(logits=stacked_logits).entropy()  # (batch, gen_len)

            # Mask out post-EOS padding in generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            generated_sequences = outputs.sequences[:, prompt_len:]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            mask = (generated_sequences != pad_id).float()
            seq_len = stacked_logits.shape[1]
            mask = mask[:, :seq_len]

            mean_entropy = (token_entropy * mask).sum() / (mask.sum() + 1e-8)
            entropy_val = mean_entropy.item()

            # Per-token stats for diagnostics
            valid_entropies = token_entropy[mask.bool()]
            min_ent = valid_entropies.min().item()
            max_ent = valid_entropies.max().item()

            if was_training:
                model.train()

            self.history.append({"step": step, "entropy": entropy_val,
                                 "min": round(min_ent, 4), "max": round(max_ent, 4)})

            # Diagnostic info on first run
            diag = ""
            if self._first_run:
                diag = (
                    f"\n  [DIAG] generated logits shape: {list(stacked_logits.shape)}"
                    f"\n  [DIAG] prompt tokens: {prompt_len}"
                    f"\n  [DIAG] vocab size: {stacked_logits.shape[-1]}"
                    f"\n  [DIAG] method: HF model.generate(output_logits=True)"
                )

            logger.info(
                f"\n{'='*50}\n"
                f"COMPLETION ENTROPY — Step {step}\n"
                f"  mean: {entropy_val:.4f} nats\n"
                f"  min:  {min_ent:.4f}  max: {max_ent:.4f}{diag}\n"
                f"{'='*50}"
            )

        def on_train_end(self, args, state, control, **kwargs):
            if self.history:
                import json
                from pathlib import Path
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "entropy_history.json"
                with open(path, "w") as f:
                    json.dump(self.history, f, indent=2)
                logger.info(f"Entropy history ({len(self.history)} entries) saved to {path}")

    entropy_cb = EntropyCallback(every_n_steps=10, n_samples=4)

    # ── 6. Create trainer and run ────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=logged_math_reward,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[cb for cb in [entropy_cb, ColorLogCallback(), fadeout_cb] if cb is not None],
    )

    t_start = time.time()

    if args.resume_from_checkpoint:
        import os
        ckpt_path = args.resume_from_checkpoint
        has_optimizer = os.path.exists(os.path.join(ckpt_path, "optimizer.pt"))
        has_scheduler = os.path.exists(os.path.join(ckpt_path, "scheduler.pt"))
        has_state = os.path.exists(os.path.join(ckpt_path, "trainer_state.json"))
        has_adapter = os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")) or \
                      os.path.exists(os.path.join(ckpt_path, "adapter_model.bin"))
        print(f"\n=== Resume Checkpoint ===")
        print(f"  Path:      {ckpt_path}")
        print(f"  Adapter:   {'found' if has_adapter else 'MISSING'}")
        print(f"  Optimizer: {'found' if has_optimizer else 'MISSING'}")
        print(f"  Scheduler: {'found' if has_scheduler else 'MISSING'}")
        print(f"  State:     {'found' if has_state else 'MISSING'}")
        if has_state:
            import json as _json
            with open(os.path.join(ckpt_path, "trainer_state.json")) as f:
                state = _json.load(f)
            print(f"  Last step: {state.get('global_step', '?')}")
            print(f"  Last epoch: {state.get('epoch', '?')}")
        print()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except KeyError as e:
        logger.error("KeyError during training: %s", e)
        logger.error("Check that the dataset has the expected columns (prompt, answer).")
        sys.exit(1)
    except ValueError as e:
        logger.error("ValueError during training: %s", e)
        logger.error("Check num_generations, reward function inputs, or dataset format.")
        sys.exit(1)
    except TypeError as e:
        logger.error("TypeError during training: %s", e)
        logger.error("TRL may expect different argument types. Inspect the traceback.")
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA OOM: %s", e)
            logger.error("Try reducing --gpu-mem-util or --max-completion-length.")
        else:
            logger.error("RuntimeError during training: %s", e)
        sys.exit(1)

    elapsed = time.time() - t_start

    print(f"\n=== Training Complete ===")
    print(f"  Total steps:    {args.max_steps}")
    print(f"  Runtime:        {elapsed / 60:.1f} minutes")
    print(f"  Output:         {args.output_dir}")


if __name__ == "__main__":
    main()
