"""Budget probe: measure a bench's output-token budget health at a given MNT.

Runs a sample of a FROZEN bench corpus through the real 3-tier Qwen pool under
the GRPO step-0 arch distribution (same as difficulty_probe), grades with the
training reward path, and reports the three numbers that decide a budget:

  * truncation_rate : fraction of traces that hit max_new_tokens (verbosity cap)
  * eng_invalid_rate: fraction dropped by the GRPO eng_valid mask (infra/no-Synth)
  * correct_rate    : correctness over eng-valid traces (so we can see whether a
                      bigger budget shifts difficulty out of the 0.4-0.6 band)

Use it BEFORE committing to a new --max_new_tokens for a category's GRPO run:
a budget is "OK" when truncation_rate < ~0.05 and correct_rate stays in band.

  python scripts/test_budget.py --bench cat_code --mnt 8192 --n_tasks 48
"""
from __future__ import annotations

import argparse
import functools
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

print = functools.partial(print, flush=True)  # unbuffered for live monitoring

from arch_policy import ARCH, bench, QwenWorker
from arch_policy.architecture.library import canonical_library, imperfect_library
from arch_policy.executor.multi_agent import MultiAgentExecutor
from _common import build_worker_pool
from difficulty_probe import realize_step0_arch, _eng_valid, MODELS, JUDGE_MODEL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--mnt", type=int, required=True, help="max_new_tokens to test")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_tasks", type=int, default=48, help="tasks sampled from split")
    ap.add_argument("--G", type=int, default=4, help="archs per task (mirrors GRPO G)")
    ap.add_argument("--max_concurrent", type=int, default=64)
    ap.add_argument("--max_llm_calls", type=int, default=32, help="match GRPO trace cap")
    ap.add_argument("--wall", type=float, default=400.0, help="match GRPO wall timeout")
    ap.add_argument("--worker_timeout", type=float, default=600.0)
    ap.add_argument("--judge_timeout", type=float, default=180.0)
    ap.add_argument("--inflight", type=int, default=64,
                    help="total in-flight API cap (leave headroom for a co-running "
                         "job: two jobs must sum to <=128)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = replace(ARCH, model_names=tuple(MODELS))

    # Cap in-flight so a co-running training job stays inside the shared 128 band.
    if args.inflight and args.inflight > 0:
        import _common
        unit = args.inflight / 7.0  # flash:plus:max = 3:2:2
        _common.MODEL_CONCURRENCY = {
            "qwen3.6-flash": max(1, round(unit * 3)),
            "qwen3.6-plus": max(1, round(unit * 2)),
            "qwen3.7-max": max(1, round(unit * 2)),
        }
        print(f"[budget] in-flight caps -> {_common.MODEL_CONCURRENCY}")

    pool = build_worker_pool(MODELS, timeout=args.worker_timeout,
                             temperature=0.6, thinking=False)
    adapter = bench.get(args.bench)
    judge = (QwenWorker(model=JUDGE_MODEL, timeout=args.judge_timeout,
                        temperature=0.0, thinking=False)
             if adapter.needs_judge() else None)
    reward_fn = bench.make_reward_fn(adapter, judge=judge)
    sft_pool = canonical_library() + imperfect_library()

    tasks = adapter.load_split(args.split, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(tasks)
    tasks = tasks[: args.n_tasks]

    ex = MultiAgentExecutor(
        worker=pool[JUDGE_MODEL], spec=spec, worker_pool=pool,
        max_new_tokens_per_call=args.mnt, wall_clock_timeout_s=args.wall,
        max_llm_calls_per_trace=args.max_llm_calls,
    )
    arch_rng = random.Random(args.seed)
    jobs = [(ti, *realize_step0_arch(sft_pool, spec.n_models, arch_rng))
            for ti in range(len(tasks)) for _ in range(args.G)]
    print(f"[budget] bench={args.bench} mnt={args.mnt} tasks={len(tasks)} "
          f"G={args.G} -> {len(jobs)} traces | judge={'on' if judge else 'off'} "
          f"max_llm_calls={args.max_llm_calls} wall={args.wall}")

    def run_one(job):
        ti, arch, _name = job
        try:
            tr = ex.run(task=tasks[ti].task, arch=arch)
            r = reward_fn(tr, tasks[ti].gold_answer, spec, task_sample=tasks[ti])
            return (int(getattr(tr, "n_worker_truncations", 0)),
                    int(getattr(tr, "n_api_errors", 0)),
                    _eng_valid(tr), float(r.correctness))
        except Exception as e:  # noqa: BLE001
            print(f"  run error ti={ti}: {type(e).__name__}: {e}")
            return (0, 1, False, 0.0)

    n = len(jobs)
    n_trunc = n_apierr = n_valid = n_correct = 0
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as tex:
        for f in as_completed([tex.submit(run_one, j) for j in jobs]):
            tr_n, api_n, valid, corr = f.result()
            n_trunc += (tr_n > 0)
            n_apierr += (api_n > 0)
            if valid:
                n_valid += 1
                n_correct += int(round(corr))
            done += 1
            if done % max(20, n // 10) == 0 or done == n:
                dt = time.time() - t0
                print(f"  {done}/{n} ({dt:.0f}s, {done/max(dt,1):.2f}/s)")

    print("\n=== BUDGET REPORT ===")
    print(f"bench={args.bench} mnt={args.mnt} traces={n}")
    print(f"truncation_rate  = {n_trunc/n:.3f}  ({n_trunc}/{n})   <- target < 0.05")
    print(f"api_error_rate   = {n_apierr/n:.3f}  ({n_apierr}/{n})")
    print(f"eng_invalid_rate = {(n - n_valid)/n:.3f}  ({n - n_valid}/{n})")
    cr = (n_correct / n_valid) if n_valid else float("nan")
    print(f"correct_rate     = {cr:.3f}  ({n_correct}/{n_valid} valid)  "
          f"<- want in 0.4-0.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
