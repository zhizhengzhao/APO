"""Difficulty probe (stage 2 of the 0.5-difficulty corpus flow).

Reads the oversized candidate pools written by `collect_categories.py
--mode candidates`, runs the REAL 3-tier Qwen worker pool over each candidate
under an architecture distribution that mirrors GRPO step-0 (uniform over the
shared `full_library` SFT prior + uniform 1/3 model assignment per agent),
grades with the SAME reward path training uses (`bench.make_reward_fn`, judge
for reasoning), and records a per-problem `pass_rate`.

It then selects the final `--n` (500) problems per category so the mean
pass_rate ≈ `--target` (0.5) — i.e. the band where architecture choice is most
discriminating — and writes the frozen corpus `data/categories/<cat>.jsonl`
(80/20 train/test, stratified by pass_rate bin) that the `category` bench reads.

Fidelity to step-0:
  * arch ~ Uniform(full_library())   (SFT pool == full_library for every cat)
  * model[slot] ~ Uniform(n_models)  (SFT trains the model head to uniform 1/3)
  * grading + eng_valid mask          identical to training.grpo
Because the SFT prior carries no task->arch grounding, this task-independent
arch distribution is the faithful proxy for the (not-yet-trained) step-0 head.

Two phases (so re-selection never needs a re-probe):
  probe : run the workers, write <cand_dir>/<cat>_probed.jsonl  (the $$$ step)
  select: read *_probed.jsonl, pick 500 at target, write <cat>.jsonl  (free)
Default runs both; pass --select_only to redo just selection.
"""
from __future__ import annotations

import argparse
import functools
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

print = functools.partial(print, flush=True)  # unbuffered for live monitoring

from arch_policy import ARCH, bench
from arch_policy.architecture.library import (
    canonical_library, imperfect_library, named_arch_to_concrete, random_archs,
)
from arch_policy.data.tasks import TaskSample
from arch_policy.executor.multi_agent import MultiAgentExecutor
from _common import build_worker_pool

MODELS = ["qwen3.6-flash", "qwen3.6-plus", "qwen3.7-max"]
JUDGE_MODEL = "qwen3.7-max"          # strongest tier, thinking off (user spec)
# Per-category worker output cap (mirrors the tuned budget probe: truncation<5%).
MAX_NEW_TOKENS = {
    "cat_code": 4096, "cat_math": 8192,
    "cat_knowledge": 8192, "cat_reasoning": 8192,
}


# ---------------------------------------------------------------------------
# Arch sampling: mirror GRPO step-0 = the SFT 2-tier prior exactly.
#   pool_ratio (0.85): uniform over canonical(82)+imperfect(15) = 97 entries
#   1-pool_ratio (0.15): fresh on-demand random arch
#   model[slot] ~ Uniform(n_models)  (SFT model head -> uniform 1/3)
# See data/sft_data.py::_draw_2tier — this matches it draw-for-draw.
# ---------------------------------------------------------------------------

SFT_POOL_RATIO = 0.85  # keep in sync with sft_data.SFTArchDataset default


def realize_step0_arch(sft_pool, n_models: int, rng: random.Random,
                       pool_ratio: float = SFT_POOL_RATIO):
    """Draw a step-0 arch from the SFT 2-tier prior + uniform model stamps."""
    if rng.random() < pool_ratio:
        named = sft_pool[rng.randrange(len(sft_pool))]
    else:
        drawn = random_archs(random.Random(rng.random()), n=1)
        named = drawn[0] if drawn else sft_pool[rng.randrange(len(sft_pool))]
    concrete = named_arch_to_concrete(named)
    active_idx = torch.nonzero(concrete.active_mask, as_tuple=True)[0]
    model = torch.zeros(ARCH.n_max, dtype=torch.long)
    for s in active_idx.tolist():
        model[s] = rng.randrange(n_models)
    return replace(concrete, model=model), named.name


def _eng_valid(tr) -> bool:
    """Identical to training.grpo's eng_valid: drop infra-corrupted traces."""
    if tr is None:
        return False
    n_infra = (int(getattr(tr, "n_api_errors", 0))
               + int(getattr(tr, "n_worker_truncations", 0)))
    return not (n_infra > 0 and not getattr(tr, "final_via_synth", False))


# ---------------------------------------------------------------------------
# Probe one category
# ---------------------------------------------------------------------------

def load_candidates(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def rec_to_sample(r: dict) -> TaskSample:
    return TaskSample(
        task=r["task"], gold_answer=r["gold_answer"], family=r["family"],
        task_id=r["task_id"], metadata=dict(r.get("metadata", {})),
    )


def _short(cat: str) -> str:
    return cat.replace("cat_", "")


def _bench_name(cat: str) -> str:
    return cat if cat.startswith("cat_") else f"cat_{cat}"


def probe_category(cat: str, args, pool, judge, sft_pool, spec) -> list[dict]:
    cands = load_candidates(Path(args.cand_dir) / f"{_short(cat)}.jsonl")
    if args.limit:
        cands = cands[: args.limit]
    adapter = bench.get(_bench_name(cat))
    reward_fn = bench.make_reward_fn(
        adapter, judge=judge if adapter.needs_judge() else None)
    mnt = MAX_NEW_TOKENS.get(cat, 8192)
    ex = MultiAgentExecutor(
        worker=pool[JUDGE_MODEL], spec=spec, worker_pool=pool,
        max_new_tokens_per_call=mnt, wall_clock_timeout_s=args.wall,
        max_llm_calls_per_trace=args.max_llm_calls,
    )
    samples = [rec_to_sample(r) for r in cands]
    n_valid = [0] * len(cands)
    n_correct = [0] * len(cands)
    arch_rng = random.Random(args.seed)
    # Pre-draw archs so the run is deterministic regardless of completion order.
    arch_jobs = [
        (ci, *realize_step0_arch(sft_pool, spec.n_models, arch_rng))
        for ci in range(len(cands)) for _ in range(args.K)
    ]
    print(f"\n=== probe [{cat}] : {len(cands)} cands x K={args.K} "
          f"= {len(arch_jobs)} traces (mnt={mnt}, "
          f"judge={'on' if adapter.needs_judge() else 'off'}) ===")

    def run_one(job):
        ci, arch, _name = job
        try:
            tr = ex.run(task=samples[ci].task, arch=arch)
            r = reward_fn(tr, samples[ci].gold_answer, spec, task_sample=samples[ci])
            return ci, _eng_valid(tr), float(r.correctness)
        except Exception as e:  # noqa: BLE001
            print(f"  [{cat}] run error ci={ci}: {type(e).__name__}: {e}")
            return ci, False, 0.0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as tex:
        for f in as_completed([tex.submit(run_one, j) for j in arch_jobs]):
            ci, valid, corr = f.result()
            if valid:
                n_valid[ci] += 1
                n_correct[ci] += int(round(corr))
            done += 1
            step = max(25, len(arch_jobs) // 12)
            if done % step == 0 or done == len(arch_jobs):
                dt = time.time() - t0
                print(f"  [{cat}] {done}/{len(arch_jobs)} traces "
                      f"({dt:.0f}s, {done/max(dt,1):.2f}/s, "
                      f"eta {(len(arch_jobs)-done)/max(done/max(dt,1),1e-6):.0f}s)")
    dt = time.time() - t0
    out = []
    for ci, r in enumerate(cands):
        nv, nc = n_valid[ci], n_correct[ci]
        pr = (nc / nv) if nv > 0 else None
        out.append({**r, "pass_rate": pr, "n_valid": nv,
                    "n_correct": nc, "K": args.K})
    n_scored = sum(1 for o in out if o["pass_rate"] is not None)
    band = sum(1 for o in out if o["pass_rate"] is not None
               and 0.25 <= o["pass_rate"] <= 0.75)
    print(f"  [{cat}] probed {len(out)} cands in {dt:.0f}s | "
          f"scored={n_scored} in-band[.25,.75]={band}")
    probed_path = Path(args.cand_dir) / f"{_short(cat)}_probed.jsonl"
    with open(probed_path, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"  [{cat}] -> wrote {probed_path}")
    return out


# ---------------------------------------------------------------------------
# Selection: greedy pick N so mean(pass_rate) ≈ target
# ---------------------------------------------------------------------------

def select_target(probed: list[dict], n: int, target: float,
                  min_valid: int, *, seed: int = 42,
                  max_mean: float = 0.65) -> list[dict]:
    """Select `n` problems that PRESERVE the pool's natural difficulty.

    Philosophy (the difficulty falls out of the data, it is never forced to a
    round 0.5):
      1. keep EVERY discriminating task (0 < pass_rate < 1) — these are the only
         ones that carry a GRPO signal (architecture choice flips the outcome);
      2. backfill to `n` with saturated tasks so the corpus mean equals the raw
         POOL mean (all usable candidates) — i.e. the selected 500 are as hard
         as the pool the user signed off on, with maximal signal density;
      3. EXCEPT when the pool is too easy (mean > `max_mean`): then make the
         corpus as hard as reachable — spend every all-wrong (pass_rate 0) task
         first, top up with the fewest all-right (pass_rate 1) tasks. This is
         the reasoning case (0.77 pool → ~0.54, well under the cap).

    `target` is accepted for call-site compatibility but ignored; the pool mean
    (capped at `max_mean`) is the real target."""
    _ = target
    rng = random.Random(seed)
    usable = [p for p in probed
              if p["pass_rate"] is not None and p["n_valid"] >= min_valid]
    in_band = [p for p in usable if 0.0 < p["pass_rate"] < 1.0]
    sat0 = [p for p in usable if p["pass_rate"] == 0.0]   # all-wrong (hard)
    sat1 = [p for p in usable if p["pass_rate"] == 1.0]   # all-right (easy)
    rng.shuffle(in_band); rng.shuffle(sat0); rng.shuffle(sat1)
    pool_mean = sum(p["pass_rate"] for p in usable) / max(len(usable), 1)

    if len(in_band) >= n:
        return in_band[:n]

    chosen = list(in_band)
    in_sum = sum(p["pass_rate"] for p in in_band)
    need = n - len(chosen)

    if pool_mean <= max_mean:
        # Match the pool's natural mean: pick n1 all-right tasks (each +1) so
        # (in_sum + n1) / n ≈ pool_mean; the rest are all-wrong.
        n1 = round(pool_mean * n - in_sum)
        n1 = max(0, min(n1, need, len(sat1)))
        n0 = need - n1
        if n0 > len(sat0):                 # rare: not enough all-wrong
            n0 = len(sat0); n1 = min(need - n0, len(sat1))
    else:
        # Pool too easy: hardest reachable — every all-wrong first.
        n0 = min(need, len(sat0))
        n1 = min(need - n0, len(sat1))

    chosen += sat0[:n0] + sat1[:n1]
    if len(chosen) < n:                    # pool smaller than n overall (rare)
        rest = [p for p in usable if p not in chosen]
        chosen += rest[: n - len(chosen)]
    return chosen[:n]


def write_corpus(cat: str, chosen: list[dict], args) -> dict:
    rng = random.Random(args.seed)
    # Stratify the 80/20 split by pass_rate bin so test mirrors train difficulty.
    by_bin = {}
    for p in chosen:
        b = round(p["pass_rate"] * args.K) / args.K
        by_bin.setdefault(b, []).append(p)
    recs = []
    for b, group in by_bin.items():
        rng.shuffle(group)
        cut = int(len(group) * args.train_ratio)
        for i, p in enumerate(group):
            recs.append({
                "task": p["task"], "gold_answer": p["gold_answer"],
                "family": p["family"], "task_id": p["task_id"],
                "metadata": p["metadata"], "category": cat,
                "difficulty": p.get("difficulty", "unknown"),
                "pass_rate": p["pass_rate"],
                "split": "train" if i < cut else "test",
            })
    rng.shuffle(recs)
    out_path = Path(args.out_dir) / f"{cat.replace('cat_', '')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prs = [r["pass_rate"] for r in recs]
    summ = {
        "n": len(recs),
        "train": sum(r["split"] == "train" for r in recs),
        "test": sum(r["split"] == "test" for r in recs),
        "mean_pass_rate": round(sum(prs) / max(len(prs), 1), 3),
        "pass_rate_hist": dict(sorted(Counter(round(p, 2) for p in prs).items())),
        "sources": dict(Counter(r["metadata"].get("source", r["family"]) for r in recs)),
    }
    print(f"  [{cat}] -> wrote {out_path}: {json.dumps(summ, ensure_ascii=False)}")
    return summ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="cat_code,cat_math,cat_knowledge,cat_reasoning")
    ap.add_argument("--cand_dir", default="data/categories/_candidates")
    ap.add_argument("--out_dir", default="data/categories")
    ap.add_argument("--K", type=int, default=4,
                    help="archs sampled per candidate (resolution of pass_rate)")
    ap.add_argument("--n", type=int, default=500, help="final problems per cat")
    ap.add_argument("--target", type=float, default=0.5,
                    help="compat only; selection targets the natural pool mean")
    ap.add_argument("--max_mean", type=float, default=0.65,
                    help="cap on corpus mean (easy pools are pulled down to it)")
    ap.add_argument("--min_valid", type=int, default=2,
                    help="drop candidates with fewer valid (non-infra) runs")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_concurrent", type=int, default=64)
    ap.add_argument("--max_llm_calls", type=int, default=24)
    ap.add_argument("--wall", type=float, default=240.0)
    ap.add_argument("--inflight", type=int, default=0,
                    help="total in-flight API cap across the 3 tiers (solo "
                         "probe can use the full ~110 measured band; 0=keep the "
                         "training-shared 64 default in _common.MODEL_CONCURRENCY)")
    ap.add_argument("--worker_timeout", type=float, default=600.0)
    ap.add_argument("--judge_timeout", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap candidates/cat")
    ap.add_argument("--select_only", action="store_true",
                    help="skip probing; re-select from existing *_probed.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cats = args.cats.split(",")
    spec = replace(ARCH, model_names=tuple(MODELS))
    summary = {}

    if args.select_only:
        for cat in cats:
            probed = load_candidates(Path(args.cand_dir) / f"{_short(cat)}_probed.jsonl")
            chosen = select_target(probed, args.n, args.target, args.min_valid,
                                   seed=args.seed, max_mean=args.max_mean)
            summary[cat] = write_corpus(cat, chosen, args)
        print("\n=== SELECT SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    from arch_policy import QwenWorker
    if args.inflight and args.inflight > 0:
        # Solo probe: no training is sharing the key, so widen the in-flight
        # band. Split ~ flash:plus:max = 3:2:2 (flash is fastest/cheapest).
        import _common
        unit = args.inflight / 7.0
        _common.MODEL_CONCURRENCY = {
            "qwen3.6-flash": max(1, round(unit * 3)),
            "qwen3.6-plus": max(1, round(unit * 2)),
            "qwen3.7-max": max(1, round(unit * 2)),
        }
        print(f"[probe] in-flight caps -> {_common.MODEL_CONCURRENCY}")
    pool = build_worker_pool(MODELS, timeout=args.worker_timeout,
                             temperature=0.6, thinking=False)
    # Judge = strongest tier, deterministic, thinking off, generous timeout.
    judge = QwenWorker(model=JUDGE_MODEL, timeout=args.judge_timeout,
                       temperature=0.0, thinking=False)
    sft_pool = canonical_library() + imperfect_library()  # 97-entry SFT prior
    print(f"[probe] sft_pool={len(sft_pool)} (canonical+imperfect) + 15% random, "
          f"models={MODELS}, K={args.K}, target={args.target}, "
          f"concurrency={args.max_concurrent}")

    for cat in cats:
        probed = probe_category(cat, args, pool, judge, sft_pool, spec)
        chosen = select_target(probed, args.n, args.target, args.min_valid,
                               seed=args.seed, max_mean=args.max_mean)
        summary[cat] = write_corpus(cat, chosen, args)

    print("\n=== PROBE+SELECT SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
