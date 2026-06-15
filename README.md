# APO — Architecture Policy Optimization

Learn a **task-conditioned distribution over multi-agent LLM architectures**.

Given a task, a small policy *head* does not answer the task directly. Instead it
samples a multi-agent architecture `A`:

- **how many** agents to activate,
- each agent's **role** (Planner / Expert / Solver / Critic / Verifier / Refiner / Researcher / Tester),
- each agent's **worker model** (a tier from the Qwen pool),
- the **execution order**, and
- the directed **communication graph** (which agents' outputs are visible to which).

The underlying worker pool is fixed; only the architecture head is trained (SFT
warm-start, then GRPO). The research question: *do different task families induce
different optimal multi-agent organizations?*

This repository is the cleaned, **desensitized** release of the three final task
categories: **code**, **math**, **reasoning**. It contains the code, the frozen
datasets, the full run telemetry, and the analysis figures/report.

---

## TL;DR results

All three runs share one SFT warm-start and one reward-shaping algorithm; only
the per-category GRPO reward differs. Each completed 18 epochs (450 GRPO steps).

| Category | Final acc | Final `n_active` | Final `n_calls` | Learned architecture `A` |
|---|---:|---:|---:|---|
| code | 77.3% | 2.26 | 4.65 | 2-agent `Planner[flash] + Solver[max]`, bidirectional |
| math | 72.1% | 2.09 | 5.82 | 1–2 `Solver`, **plus**-heavy, mostly **0 comm edges** |
| reasoning | 73.3% | 2.91 | 4.55 | `Solver[plus]` core + `Verifier`/`Refiner`, denser comm |

Key finding: APO does **not** collapse to one universal template. It learns
distinct, interpretable, cost-aware architectures per task family — different
team sizes, different model tiers, different communication density. Full analysis
in [`docs/分析报告.md`](docs/分析报告.md) (Chinese) with figures in
[`results/figures/`](results/figures).

> Scope note: this is an empirical pilot suitable for a workshop. It is **not**
> yet a held-out-eval + baseline-grid paper. The shipped telemetry is on-policy
> training-batch statistics; a fixed held-out evaluation and fixed-architecture
> baselines (see `src/arch_policy/baselines.py`, `scripts/04_evaluate.py`) are
> the recommended next experiments.

---

## Repository layout

```
src/arch_policy/        core library
  architecture/         arch spec, sampler (active/role/edge/sequence/model), library, head encoder
  head/                 the policy head (Qwen backbone + LoRA + typed output heads)
  training/             sft.py, grpo.py, advantage.py (reward shaping), entropy.py
  executor/             multi-agent executor, agents, synth, tools, Qwen worker
  reward/               grading (exec-based code, symbolic/numeric math, MCQ, short-answer)
  bench/                plugin registry: cat_code / cat_math / cat_reasoning / cat_mixed
  data/                 task loaders + SFT pairing dataset
  baselines.py          12 fixed-topology baselines (for comparison experiments)
scripts/                01..06 pipeline + data-curation provenance + run_cat_rl.sh
tests/                  core unit tests (algorithm, executor, sampler, resume, ...)
data/categories/        FROZEN datasets: code/math/reasoning (500 each) + mixed (SFT pool)
results/                run telemetry + analysis artifacts
  {code,math,reasoning}/  history.json (per-step metrics) + details.jsonl (per-trace logs)
  summary/              apo_three_experiment_data.json (re-parsed final stats)
  figures/              analysis plots used in the report
docs/                   分析报告.md (full Chinese analysis), related_work.md
```

> Trained model checkpoints (~259 MB each) are **not** included in this repo by
> design. The code, datasets, and full telemetry are sufficient to inspect the
> results and to re-run training from scratch.

---

## Datasets (`data/categories/`)

Each category is a frozen 500-problem corpus loaded via the `CategoryBench`
plugin; data and algorithm are fully separated (add a category = drop in one
`<cat>.jsonl`, no algorithm change). Difficulty was calibrated with a real
3-tier-Qwen probe (`scripts/difficulty_probe.py`) so step-0 has architectural
discrimination.

| Category | n | train/test | probe mean pass-rate | main sources |
|---|---:|---:|---:|---|
| code | 500 | 399/101 | 0.627 | livecodebench, mbpp, humaneval |
| math | 500 | 397/103 | 0.618 | omni_math, olympiad, aime, phybench |
| reasoning | 500 | 398/102 | 0.536 | musr, reclor, arc, BBH families |
| mixed (SFT) | 1592 | — | — | union of category train splits (incl. a knowledge set used only for SFT) |

`mixed.jsonl` is the **shared SFT pool**: a single category-agnostic task set
used to train ONE warm-start head. SFT pairs each task with a *random* valid
architecture (no task→architecture grounding); it only teaches the head to emit
sensible, diverse architectures. The model-selection dimension is pushed toward
uniform 1/3 at SFT, so all per-category model preferences are learned by GRPO.

---

## Reward shaping (unified across all three categories)

`src/arch_policy/training/advantage.py`. Two tiers:

- **Tier 1 (raw):** wrong → `-1`; correct → `1 + bonus`, where `bonus ∈ [0, 0.5]`
  is a min-max on `n_calls` *within the correct sub-group* (cheaper correct → larger).
- **Tier 2 (per-task, gated by group composition):**
  - all correct → advantage `0` (architecture didn't change correctness → no signal, and no cost gradient that could erode accuracy),
  - all wrong → advantage `-0.1` (mild push to explore elsewhere),
  - mixed → `raw / std` (sign + ordering preserved; cost is a tie-break inside correct samples only).

Engineering-invalid traces (API errors, key truncations, judge-side infra
errors) are excluded from the gradient via an eng-valid mask.

> All three released runs and this code use this single unified shaping. (During
> development the reasoning run's shaping was finalized slightly later than
> code/math; the repository ships one consistent version for all three.)

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .
cp .env.example .env                      # then fill in DASHSCOPE_API_KEY
```

All credentials are read from environment variables — none are hardcoded.
The released runs use the Qwen 3-tier pool (`qwen3.6-flash`, `qwen3.6-plus`,
`qwen3.7-max`) with thinking mode OFF for every worker and the judge.

## Reproduce

```bash
# 1) SFT the shared warm-start head on the mixed pool
PYTHONPATH=src python scripts/02_train_sft.py --bench cat_mixed \
  --out_dir checkpoints/sft_shared   # produces checkpoints/sft_shared/head_final

# 2) GRPO per category (one GPU each; two fit in the 128 in-flight API budget)
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_code      0 8192 4096 18
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_math      1 8192 4096 18
SFT_CKPT=checkpoints/sft_shared/head_final bash scripts/run_cat_rl.sh cat_reasoning 2 4096 4096 18

# 3) (optional) held-out eval against fixed-topology baselines
PYTHONPATH=src python scripts/04_evaluate.py --help
```

## Inspect the shipped results without re-running

```bash
PYTHONPATH=src python - <<'PY'
import json
d = json.load(open("results/summary/apo_three_experiment_data.json"))
for cat, e in d["experiments"].items():
    h = e["history"]
    print(cat, "acc", round(h["correct_rate"]["last20"], 3),
          "n_active", round(h["n_active_mean"]["last20"], 2),
          "n_calls", round(h["n_calls_mean"]["last20"], 2),
          "model_share", e["details"]["model_share"])
PY
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q     # core algorithm/executor/resume tests
```

## License

MIT (see `LICENSE`).
