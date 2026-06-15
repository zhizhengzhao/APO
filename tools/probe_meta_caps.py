"""Empirical sanity check for ALL architecture-can't-control caps.

The architecture (head's gate / role / edge / seq logits) controls
WHICH agents run + the comm graph + the speaking order. It does NOT
control any of:
  - per-call worker output token cap            (max_new_tokens)
  - judge call output token cap                 (hardcoded in bench)
  - per-task input window the head sees         (max_seq_len)
  - per-tool wall-clock timeouts                (PYTHON_TIMEOUT_S, ...)
  - per-trace wall_clock_timeout_s              (semi-arch-controllable)

If any of these are too tight, EVERY architecture is silently
truncated on hard tasks → reward signal becomes a function of the
infra cap, not the architecture. Smoke runs can hide this if the
median task is short.

This probe fires realistic worker/judge calls under each role's actual
system prompt + a generous output budget, then reports:
  - p25 / p50 / p75 / p90 / p95 / p99 / max output tokens
  - the would-be-truncated rate at each candidate cap

Usage:
    PYTHONPATH=src python3 tools/probe_meta_caps.py \\
        [--n_tasks 20] [--roles Solver,Critic,Researcher] \\
        [--include_judge]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arch_policy.bench import get as get_bench  # noqa: E402
from arch_policy.executor.qwen_worker import QwenWorker  # noqa: E402
from arch_policy.executor.prompts import (  # noqa: E402
    build_system_prompt, ROLE_SYSTEM_PROMPTS,
)


GENEROUS_CAP = 8192   # bumped from 4096 (intern audit: probe ceiling
                       # was itself the cap; need to set HIGHER than the
                       # model's expected output to find true truncation)


def _pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]


def _report_dist(label: str, lens: list[int], cap_candidates: list[int]):
    """Pretty-print percentile distribution + would-be-truncated rate."""
    if not lens:
        print(f"\n{label}: no successful calls")
        return
    n = len(lens)
    print(f"\n{'=' * 70}")
    print(f"  {label}  (n={n})")
    print(f"{'=' * 70}")
    print(f"  min={min(lens):>4}  mean={sum(lens)//n:>4}  max={max(lens):>5}")
    print(f"  p25={_pct(lens,25):>4}  p50={_pct(lens,50):>4}  "
          f"p75={_pct(lens,75):>4}  p90={_pct(lens,90):>4}  "
          f"p95={_pct(lens,95):>5}  p99={_pct(lens,99):>5}")
    print(f"  would-be-truncated at each candidate cap:")
    for cap in cap_candidates:
        n_over = sum(1 for L in lens if L > cap)
        pct = 100 * n_over / n
        flag = "✅" if pct < 2 else ("⚠️" if pct < 10 else "🔴")
        print(f"    > {cap:>5}  : {n_over:>3} / {n} = {pct:>5.1f}%  {flag}")


def _build_user_for_role(role: str, task_text: str) -> str:
    """Realistic user prompt the executor would assemble for each role
    on the first cycle (no prior turns yet)."""
    if role == "Solver":
        return (f"TASK:\n{task_text}\n\nProduce your answer using "
                f"ACTION: submit {{ ... }} or ACTION: tool(...) to use "
                f"a tool first.")
    if role == "Critic":
        return (f"TASK:\n{task_text}\n\nA candidate answer was proposed.\n"
                f"CANDIDATE: 42\n\nReview rigorously. Identify weaknesses "
                f"or missing steps. ACTION: submit {{ ... }} with your critique.")
    if role == "Researcher":
        return (f"TASK:\n{task_text}\n\nFind authoritative sources. "
                f"ACTION: tool(...) to search, then ACTION: submit {{ ... }} "
                f"with what you learned.")
    if role == "Verifier":
        return (f"TASK:\n{task_text}\n\nCANDIDATE: 42\n\nVerify step-by-step. "
                f"ACTION: submit {{ ... }} with `verified` or `incorrect`.")
    # Fallback: generic.
    return f"TASK:\n{task_text}\n\nACTION: submit {{ ... }}"


def probe_worker(worker: QwenWorker, tasks, roles: list[str],
                 max_parallel: int = 16) -> dict[str, list[int]]:
    """Fire (n_tasks × n_roles) calls in parallel. Returns role → list
    of output-token counts (one per successful call)."""
    jobs = [(role, idx, t) for role in roles for idx, t in enumerate(tasks)]
    out_lens: dict[str, list[int]] = {r: [] for r in roles}

    def fire(role, idx, task):
        sys_p = build_system_prompt(role)
        user_p = _build_user_for_role(role, task.task)
        t = time.time()
        try:
            out = worker.chat(sys_p, user_p, max_new_tokens=GENEROUS_CAP)
            return role, idx, out.n_output_tokens, time.time() - t, None
        except Exception as e:  # noqa: BLE001
            return role, idx, -1, time.time() - t, f"{type(e).__name__}: {e}"

    t0 = time.time()
    print(f"[probe-worker] firing {len(jobs)} calls "
          f"({len(tasks)} tasks × {len(roles)} roles) "
          f"max_parallel={max_parallel}", flush=True)
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futs = [pool.submit(fire, *j) for j in jobs]
        for f in as_completed(futs):
            role, idx, n_tok, dur, err = f.result()
            if err is None:
                out_lens[role].append(n_tok)
                n_ok += 1
            else:
                n_err += 1
    print(f"[probe-worker] done: {n_ok} ok / {n_err} err "
          f"in {time.time()-t0:.0f}s", flush=True)
    return out_lens


def probe_judge(tasks, judge_model: str, max_parallel: int = 8) -> list[int]:
    """Fire judge calls using the actual bench/hle.py:_JUDGE_PROMPT."""
    from openai import OpenAI
    from arch_policy.bench.hle import _JUDGE_PROMPT

    client = OpenAI(
        api_key=os.environ["GPUGEEK_API_KEY"],
        base_url="https://api.gpugeek.com/v1",
        timeout=180.0,
    )

    def fire(task):
        # Realistic: judge is asked to compare a plausible (often wrong)
        # candidate answer against the gold. The full prompt is what
        # bench/hle.py:_judge fires.
        prompt = _JUDGE_PROMPT.format(
            question=task.task,
            response="42",  # plausible wrong baseline candidate
            correct_answer=task.gold_answer,
        )
        t = time.time()
        try:
            r = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GENEROUS_CAP,
                temperature=0.0,
            )
            usage = r.usage
            n_out = int(getattr(usage, "completion_tokens", 0) or 0)
            return n_out, time.time() - t, None
        except Exception as e:  # noqa: BLE001
            return -1, time.time() - t, f"{type(e).__name__}: {e}"

    t0 = time.time()
    print(f"[probe-judge] firing {len(tasks)} judge calls on "
          f"{judge_model} max_parallel={max_parallel}", flush=True)
    out_lens: list[int] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futs = [pool.submit(fire, t) for t in tasks]
        for f in as_completed(futs):
            n_out, dur, err = f.result()
            if err is None and n_out > 0:
                out_lens.append(n_out)
    print(f"[probe-judge] done: {len(out_lens)} ok in {time.time()-t0:.0f}s",
          flush=True)
    return out_lens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hle")
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--roles", default="Solver,Critic,Researcher,Verifier")
    ap.add_argument("--worker_model", default="qwen3.7-max")
    ap.add_argument("--judge_model", default="Vendor2/GPT-5.1")
    ap.add_argument("--worker_parallel", type=int, default=16)
    ap.add_argument("--judge_parallel", type=int, default=8)
    ap.add_argument("--no_judge", action="store_true")
    args = ap.parse_args()

    print(f"[probe] loading bench={args.bench} train split ...", flush=True)
    tasks = get_bench(args.bench).load_split("train")
    random.Random(42).shuffle(tasks)
    sample = tasks[: args.n_tasks]
    subjects = sorted(set(t.metadata.get("subject", "?") for t in sample))
    print(f"[probe] {len(sample)} tasks, subjects = {subjects}", flush=True)

    roles = [r.strip() for r in args.roles.split(",")]
    bad = [r for r in roles if r not in ROLE_SYSTEM_PROMPTS]
    if bad:
        print(f"[probe] unknown roles: {bad}", file=sys.stderr)
        return 2

    worker = QwenWorker(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model=args.worker_model,
        timeout=180.0, max_retries=1,
    )

    out_per_role = probe_worker(worker, sample, roles, args.worker_parallel)

    CAP_CANDIDATES = [256, 512, 1024, 1536, 2048, 3072]

    for role in roles:
        _report_dist(f"WORKER  role={role}  model={args.worker_model}",
                     out_per_role[role], CAP_CANDIDATES)

    if not args.no_judge:
        out_judge = probe_judge(sample, args.judge_model, args.judge_parallel)
        _report_dist(f"JUDGE   model={args.judge_model}",
                     out_judge, CAP_CANDIDATES)

    return 0


if __name__ == "__main__":
    sys.exit(main())
