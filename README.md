# APO — Architecture Policy Optimization

A policy *head* learns a task-conditioned distribution over multi-agent LLM
architectures. Given a task, the head does not answer it directly; it samples an
architecture `A`:

- the number of active agents,
- each agent's role (Planner / Expert / Solver / Critic / Verifier / Refiner / Researcher / Tester),
- each agent's worker model (a tier from a fixed Qwen pool),
- the execution order, and
- the directed communication graph (which agents' outputs are visible to which).

The worker pool is fixed; only the head is trained — an SFT warm-start followed
by GRPO. This repository contains the code, the frozen datasets, the full run
telemetry, and the analysis artifacts for three task categories: **code**,
**math**, **reasoning**.

## Final runs

All three runs start from one shared SFT head and use one reward-shaping
algorithm; only the per-category task reward differs. Each ran 18 epochs
(450 GRPO steps) and produced a final checkpoint (not included here, see below).

Values below are means over the final 20 training steps (`results/summary/apo_three_experiment_data.json`).

| Category | correct rate | n_active | n_calls | model share (flash / plus / max) |
|---|---:|---:|---:|---|
| code | 0.773 | 2.26 | 4.65 | 0.30 / 0.18 / 0.52 |
| math | 0.721 | 2.09 | 5.82 | 0.23 / 0.50 / 0.28 |
| reasoning | 0.733 | 2.91 | 4.55 | 0.16 / 0.72 / 0.12 |

Most frequent architecture family in the final 30 steps:

| Category | top family | edges | count |
|---|---|---:|---:|
| code | `Planner[flash] + Solver[max]` | 2 | 400 |
| math | `Solver[plus] + Solver[plus]` | 0 | 80 |
| reasoning | `Solver[plus] + Solver[plus]` | 1 | 405 |

## Repository layout

```
src/arch_policy/        core library
  architecture/         arch spec, sampler (active/role/edge/sequence/model), library, encoder
  head/                 policy head (Qwen backbone + LoRA + typed output heads)
  training/             sft.py, grpo.py, advantage.py (reward shaping), entropy.py
  executor/             multi-agent executor, agents, synth, tools, Qwen worker
  reward/               grading (exec code, symbolic/numeric math, MCQ, short-answer)
  bench/                plugin registry: cat_code / cat_math / cat_reasoning / cat_mixed
  data/                 task loaders + SFT pairing dataset
  baselines.py          fixed-topology baselines
scripts/                01..06 pipeline, data-curation scripts, run_cat_rl.sh
tests/                  unit tests (algorithm, executor, sampler, resume, ...)
data/categories/        frozen datasets: code/math/reasoning (500 each) + mixed (SFT pool)
results/
  {code,math,reasoning}/  history.json (per-step metrics) + details.jsonl (per-trace logs)
  summary/              apo_three_experiment_data.json (re-parsed final stats)
  figures/              analysis plots
docs/                   RESULTS.md (measured results + figure captions), related_work.md
```

Trained checkpoints (~259 MB each) are not included. The code, datasets, and
telemetry are sufficient to inspect the results and to retrain from scratch.

## Datasets (`data/categories/`)

Each category is a frozen 500-problem corpus loaded by the `CategoryBench`
plugin (data and algorithm are separated: a new category is one `<cat>.jsonl`).
Per-problem pass rates were measured with a 3-tier-Qwen probe
(`scripts/difficulty_probe.py`).

| Category | n | train/test | probe mean pass-rate | sources |
|---|---:|---:|---:|---|
| code | 500 | 399/101 | 0.627 | livecodebench, mbpp, humaneval |
| math | 500 | 397/103 | 0.618 | omni_math, olympiad, aime, phybench |
| reasoning | 500 | 398/102 | 0.536 | musr, reclor, arc, BBH families |
| mixed (SFT) | 1592 | — | — | union of category train splits (includes a knowledge set used only for SFT) |

`mixed.jsonl` is the shared SFT pool. SFT pairs each task with a random valid
architecture (no task→architecture grounding); the model-selection dimension is
pushed toward a uniform 1/3 prior at SFT. Per-category model and role
preferences are therefore learned during GRPO.

## Reward shaping (`src/arch_policy/training/advantage.py`)

Two tiers, identical for all three categories:

- Tier 1 (raw): wrong → `-1`; correct → `1 + bonus`, with `bonus ∈ [0, 0.5]` a
  min-max on `n_calls` within the correct sub-group (cheaper correct → larger).
- Tier 2 (per task, by group composition): all-correct → `0`; all-wrong →
  `-0.1`; mixed → `raw / std`. Cost only differentiates samples inside mixed
  groups; it never overrides the correctness signal.

Engineering-invalid traces (API errors, key truncations, judge-side infra
errors) are excluded from the gradient by an eng-valid mask.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .
cp .env.example .env                      # then set DASHSCOPE_API_KEY
```

Credentials are read from environment variables; none are hardcoded. The runs
use the Qwen 3-tier pool (`qwen3.6-flash`, `qwen3.6-plus`, `qwen3.7-max`) with
thinking mode off for every worker and the judge.

## Reproduce

```bash
# 1) SFT the shared warm-start head on the mixed pool
PYTHONPATH=src python scripts/02_train_sft.py --bench cat_mixed \
  --out_dir checkpoints/sft_shared          # -> checkpoints/sft_shared/head_final

# 2) GRPO per category
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_code      0 8192 4096 18
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_math      1 8192 4096 18
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_reasoning 2 4096 4096 18

# 3) Evaluation entry point (held-out eval / baselines)
PYTHONPATH=src python scripts/04_evaluate.py --help
```

## Notes on the shipped results

- `history.json` and `details.jsonl` are on-policy training-batch records. No
  held-out evaluation set or fixed-architecture baseline grid is included; the
  baseline definitions are in `src/arch_policy/baselines.py` and the eval entry
  point is `scripts/04_evaluate.py`.
- `details.jsonl` records `edges_count` per sampled architecture but not the
  full edge matrix; communication analysis below the family level uses edge
  counts only.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## License

MIT (see `LICENSE`).
