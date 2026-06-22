# APO — Architecture Policy Optimization

A policy *head* learns a task-conditioned distribution over multi-agent LLM architectures. Given a task, the head samples an architecture `A`: active agents, roles, worker-model tiers, execution order, and directed communication edges. The worker pool is fixed; only the architecture head is trained with a shared SFT warm-start followed by GRPO.

This repository contains the cleaned code, frozen datasets, full telemetry, and measured artifacts for six training runs: 3 task categories × 2 reward variants.

## Runs

All runs use the same SFT warm-start, role prompts, model pool, budgets, hyperparameters, and training split. The two reward variants are:

- `nobonus`: `cost_bonus_scale=0`; correctness only (plus the all-wrong mild push).
- `bonus_fixed`: `cost_bonus_scale=0.5`; cost bonus applies only within mixed groups, and all-correct groups emit zero advantage.

Values below are means over the final 20 training steps. They are on-policy training statistics, not held-out evaluation results.

| Category | no-bonus correct | no-bonus calls | bonus-fixed correct | bonus-fixed calls |
|---|---:|---:|---:|---:|
| code | 0.821 | 9.02 | 0.822 | 7.51 |
| math | 0.725 | 8.30 | 0.708 | 7.46 |
| reasoning | 0.704 | 6.63 | 0.733 | 4.55 |

Final-30 model share (flash / plus / max):

| Run | model share | top architecture family |
|---|---|---|
| `code_nobonus` | 0.12 / 0.18 / 0.70 | `Refiner[max] + Solver[max] + Solver[max] + Solver[max]` (edges=10, count=29) |
| `code_bonus_fixed` | 0.08 / 0.18 / 0.74 | `Solver[max] + Solver[max] + Solver[max]` (edges=4, count=151) |
| `math_nobonus` | 0.19 / 0.43 / 0.38 | `Solver[max] + Solver[max] + Solver[plus]` (edges=4, count=8) |
| `math_bonus_fixed` | 0.19 / 0.40 / 0.41 | `Solver[max] + Solver[plus]` (edges=2, count=28) |
| `reasoning_nobonus` | 0.33 / 0.51 / 0.16 | `Solver[flash] + Solver[plus] + Verifier[plus]` (edges=1, count=17) |
| `reasoning_bonus_fixed` | 0.16 / 0.72 / 0.12 | `Solver[plus] + Solver[plus]` (edges=1, count=405) |

## Repository layout

```
src/arch_policy/        core library
  architecture/         arch spec, sampler, library, encoder
  head/                 policy head (Qwen backbone + LoRA + typed output heads)
  training/             sft.py, grpo.py, advantage.py, entropy.py
  executor/             multi-agent executor, agents, synth, tools, Qwen worker
  reward/               grading (exec code, symbolic/numeric math, MCQ, short-answer)
  bench/                cat_code / cat_math / cat_reasoning / cat_mixed plugins
scripts/                pipeline scripts + run_cat_rl.sh
tests/                  core unit tests
data/categories/        frozen corpora: code/math/reasoning + mixed SFT pool
results/
  *_nobonus/            history.json + details.jsonl
  *_bonus_fixed/        history.json + details.jsonl
  summary/              apo_six_experiment_data.json
  figures/              analysis figures
docs/                   RESULTS.md, related_work.md
```

Trained checkpoints (~259 MB each) are not included. The repository ships code, datasets, and full run telemetry.

## Datasets

| Category | n | train/test | probe mean pass-rate | sources |
|---|---:|---:|---:|---|
| code | 500 | 399/101 | 0.627 | livecodebench, mbpp, humaneval |
| math | 500 | 397/103 | 0.618 | omni_math, olympiad, aime, phybench |
| reasoning | 500 | 398/102 | 0.536 | musr, reclor, arc, BBH families |
| mixed (SFT) | 1592 | — | — | union of category train splits |

`mixed.jsonl` is the shared SFT pool. SFT pairs each task with a random valid architecture (no task-to-architecture grounding) and pushes the model-selection head toward a uniform 1/3 prior.

## Reward shaping

`src/arch_policy/training/advantage.py` implements the fixed two-tier advantage used for the released bonus-fixed runs:

- wrong → `-1`; correct → `1 + bonus`, with `bonus ∈ [0, cost_bonus_scale]` based on lower `n_calls` among correct samples;
- all-correct group → `0`; all-wrong group → `-0.1`; mixed group → `raw / std`;
- engineering-invalid traces are masked out of the gradient.

For `nobonus`, `cost_bonus_scale=0`, so correct samples have raw value `+1`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill DASHSCOPE_API_KEY in .env
```

Credentials are read from environment variables. No credentials are hardcoded.

## Reproduce

```bash
PYTHONPATH=src python scripts/02_train_sft.py --bench cat_mixed --out_dir checkpoints/sft_shared
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_code_nobonus      bash scripts/run_cat_rl.sh cat_code      0 8192 4096 18 0
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_code_bonus_fixed  bash scripts/run_cat_rl.sh cat_code      0 8192 4096 18 0.5
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_math_nobonus      bash scripts/run_cat_rl.sh cat_math      1 8192 4096 18 0
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_math_bonus_fixed  bash scripts/run_cat_rl.sh cat_math      1 8192 4096 18 0.5
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_reasoning_nobonus bash scripts/run_cat_rl.sh cat_reasoning 2 4096 4096 18 0
SFT_CKPT=checkpoints/sft_shared/head_final RUN_TAG=grpo_cat_reasoning_bonus_fixed bash scripts/run_cat_rl.sh cat_reasoning 2 4096 4096 18 0.5
PYTHONPATH=src python scripts/04_evaluate.py --help
```

## Notes

- The shipped `history.json` and `details.jsonl` are on-policy training-batch records.
- No held-out evaluation or fixed-architecture baseline grid is included in this repository.
- `details.jsonl` records `edges_count`, not the full edge matrix.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## License

MIT (see `LICENSE`).
