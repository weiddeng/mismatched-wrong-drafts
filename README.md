# Weak-to-Strong Elicitation via Mismatched Wrong Drafts

Code, data, and model for the paper **["Weak-to-Strong Elicitation via Mismatched Wrong Drafts"](https://arxiv.org/abs/2605.17314)** (Wei Deng, 2026).

## The idea in one paragraph

For each training problem we sample 32 candidate solutions from a weaker draft
model (Qwen2.5-Math-1.5B), keep one that is **nontrivially wrong**, and then **shuffle the
wrong drafts across problems** (a derangement) so every problem is paired with a
wrong draft written *for a different problem*. The strong learner (Mathstral-7B)
is RL-fine-tuned with Dr. GRPO to produce a correct solution given this
mismatched, wrong "thinking" context. At test time no draft is injected (the
draft slot is the literal string `N/A`).

## Released artifacts

- 📄 **Paper:** https://arxiv.org/abs/2605.17314
- 💻 **Code:** this repo
- 🤗 **Models** — the four trained variants of the 2×2 ablation<sup>†</sup> (rank-16 LoRA on `mistralai/Mathstral-7B-v0.1`):

  | Variant | HF repo |
  |---|---|
  | **Mismatched-wrong** ⭐ headline | [`hugruby/mathstral-7b-mismatched-wrong-drafts`](https://huggingface.co/hugruby/mathstral-7b-mismatched-wrong-drafts) |
  | No-draft (standard GRPO) | [`hugruby/mathstral-7b-grpo-no-draft`](https://huggingface.co/hugruby/mathstral-7b-grpo-no-draft) |
  | Matched-wrong | [`hugruby/mathstral-7b-matched-wrong-drafts`](https://huggingface.co/hugruby/mathstral-7b-matched-wrong-drafts) |
  | Mismatched-correct | [`hugruby/mathstral-7b-mismatched-correct-drafts`](https://huggingface.co/hugruby/mathstral-7b-mismatched-correct-drafts) |

  <sub>† The remaining 2×2 cell — *matched-correct* — is uninteresting and omitted.</sub>

- 🤗 **Dataset:** [`hugruby/mismatched-wrong-drafts`](https://huggingface.co/datasets/hugruby/mismatched-wrong-drafts): four configs (`mismatched_wrong`, `no_draft`, `matched_wrong`, `mismatched_correct`), 8,888 Level 3–5 MATH problems each, MATH-500 held out. Also ships `drafts_qwen25_math_1.5b.json`, the raw 32-draft-per-problem source the configs are built from.

## Repository layout

```
src/grpo/
  rewards.py          # answer extraction + math_verify scoring (shared by the training reward & evals)
scripts/
  generate_drafts.py         # 1. sample 32 drafts/problem from the draft model
  build_datasets.py          # 2. build matched variants (wrong / no / correct draft)
  build_mismatch.py          # 2. build mismatched variants (wrong / correct draft) — the derangement
  train.py                   # 3. Dr. GRPO LoRA training
  eval_math500.py            # 4. greedy pass@1 on MATH-500
  eval_math500_sampling.py   # 4. sampling pass@k on MATH-500
  eval_aime_sampling.py      # 4. sampling pass@k on AIME 2024–2026
  verify_adapter.py          # standalone smoke test: load adapter + generate
  runpod_setup.sh            # fresh-GPU-pod setup (pins vllm 0.18.x, drops bad flashinfer-cubin)
```

Run scripts from the **repository root** (`python scripts/train.py ...`).

## Installation

A CUDA GPU is required. The tested, reproducible installer is `scripts/runpod_setup.sh`:

```bash
bash scripts/runpod_setup.sh
```

Or install the pinned dependencies directly:

```bash
pip install -r requirements.txt
```

> **vLLM must be 0.18.x** — 0.19+ has a `torch.compile` regression with Unsloth.
> 0.18.1 is the tested version on B200/H200.

## Quickstart — run the trained model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mathstral-7B-v0.1"
ADAPTER = "hugruby/mathstral-7b-mismatched-wrong-drafts"

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)

PROMPT = (
    "Problem: {problem}\n\n"
    "Thinking: N/A\n\n"
    "The thinking section may contain errors. "
    "Solve the math problem step by step. "
    "Write your own correct solution. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Correct Solution:"
)

text = PROMPT.format(problem="If $x+y=6$ and $xy=5$, find $x^2+y^2$.")
ids = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=4096, do_sample=False)  # greedy
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Reproduce from scratch

### 1. Generate drafts (draft model)

```bash
python scripts/generate_drafts.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --n-samples 32 --temperature 0.8 --top-p 0.95 --max-tokens 2560 \
    --output-dir outputs/drafts_qwen25_math_1.5b
```

This loads the full MATH dataset (12,500 problems), labels the MATH-500 split by
problem-text match, and writes `drafts.json` with per-sample correctness flags.

### 2. Build the training datasets

```bash
# Generate the no-draft, matched-wrong, and matched-correct datasets from drafts.json.
# Output: data/no_draft, data/matched_wrong, data/matched_correct
python scripts/build_datasets.py \
    --drafts outputs/drafts_qwen25_math_1.5b/drafts.json --out-dir data

# Generate the mismatched-wrong dataset from drafts.json — wrong drafts deranged across problems.
# Output: data/mismatched_wrong
python scripts/build_mismatch.py --draft-type wrong \
    --drafts outputs/drafts_qwen25_math_1.5b/drafts.json

# Generate the mismatched-correct dataset from drafts.json — correct drafts deranged across problems.
# Output: data/mismatched_correct
python scripts/build_mismatch.py --draft-type correct \
    --drafts outputs/drafts_qwen25_math_1.5b/drafts.json
```

The universe is the 8,888 Level 3–5 problems in MATH minus MATH-500.

To reproduce the **exact** published row order and mismatch pairings, download the prepared dataset directly from [`hugruby/mismatched-wrong-drafts`](https://huggingface.co/datasets/hugruby/mismatched-wrong-drafts) — every config already has them baked in, so no rebuilding is needed.

### 3. Train (Dr. GRPO + LoRA, single GPU)

```bash
python scripts/train.py \
    --model mistralai/Mathstral-7B-v0.1 \
    --dataset-path data/mismatched_wrong \
    --output-dir outputs/mismatched_wrong \
    --max-steps 2222 \
    --gradient-accumulation-steps 4 \
    --max-completion-length 4096 \
    --max-seq-length 7168 \
    --max-prompt-tokens 3072 \
    --learning-rate 5e-6 --lr-scheduler-type constant \
    --beta 0 \
    --correction-bonus 0.0 --copy-penalty 0.0 --corrupt-penalty 0.0 \
    --adam-beta2 0.99 \
    --save-steps 50 --gpu-mem-util 0.5
```

The headline checkpoint is **`checkpoint-2000`** (the released adapter); use
`--smoke` for a 2-step sanity run first. `--lora-rank` (16) and `--num-generations`
(16) are the defaults. For the other variants, swap `--dataset-path` /
`--output-dir` to the matching config.

### 4. Evaluate

```bash
# MATH-500 greedy pass@1 (headline 71.98%)
# Output: outputs/eval_math500_greedy.json — scores (pass@1 + by-level/subject) and per-problem records (incl. the generated solution)
python scripts/eval_math500.py \
    --base-model mistralai/Mathstral-7B-v0.1 \
    --lora outputs/mismatched_wrong/checkpoint-2000 \
    --mode training_nodraft --temperature 0.0 --max-tokens 4096 \
    --output outputs/eval_math500_greedy.json

# MATH-500 sampling pass@k
# Output (auto-named): outputs/mismatched_wrong/eval_checkpoint-2000_math500_sampling256_training_nodraft.json — scores (pass@k/maj@k/avg@k) and per-problem records with all 256 raw samples
python scripts/eval_math500_sampling.py \
    --base-model mistralai/Mathstral-7B-v0.1 \
    --lora outputs/mismatched_wrong/checkpoint-2000 \
    --mode training_nodraft --n-samples 256 \
    --temperature 0.6 --top-p 0.95 --max-tokens 4096

# AIME 2025/2026 pass@k  (NOTE: pass --temperature 0.6 explicitly; the script default is 0.7)
# Output (auto-named): outputs/mismatched_wrong/eval_checkpoint-2000_aime2025_2026_sampling2048.json — scores (pass@k/maj@k/avg@k by year) and per-problem records with all 2048 raw samples
python scripts/eval_aime_sampling.py \
    --base-model mistralai/Mathstral-7B-v0.1 \
    --lora outputs/mismatched_wrong/checkpoint-2000 \
    --mode training_nodraft --year 2025 2026 \
    --n-samples 2048 --temperature 0.6 --top-p 0.95 --max-tokens 4096
```

Trained variants use `--mode training_nodraft` (the draft slot is `N/A`). The
Mathstral-7B base model is evaluated in its native `[INST]` chat format, with `--mode instruct` instead.

## Hyperparameters

All four share one Dr. GRPO + rank-16 LoRA recipe; each variant's full training command and hyperparameters live in its model card: [mismatched-wrong](https://huggingface.co/hugruby/mathstral-7b-mismatched-wrong-drafts), [no-draft](https://huggingface.co/hugruby/mathstral-7b-grpo-no-draft), [matched-wrong](https://huggingface.co/hugruby/mathstral-7b-matched-wrong-drafts), [mismatched-correct](https://huggingface.co/hugruby/mathstral-7b-mismatched-correct-drafts).

## Citation

```bibtex
@article{deng2026mismatched,
  title  = {Weak-to-Strong Elicitation via Mismatched Wrong Drafts},
  author = {Deng, Wei},
  year   = {2026},
  eprint = {2605.17314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2605.17314}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The released LoRA adapter is a derivative of
`mistralai/Mathstral-7B-v0.1`, which is itself Apache-2.0.

## Acknowledgements

Built on [Unsloth](https://github.com/unslothai/unsloth),
[TRL](https://github.com/huggingface/trl),
[vLLM](https://github.com/vllm-project/vllm), and
[math-verify](https://github.com/huggingface/math-verify). Problems are from the
[MATH](https://github.com/hendrycks/math) dataset; the held-out test set is
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).
