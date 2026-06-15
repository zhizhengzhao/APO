"""Concurrency probe for Qwen-Max (Aliyun Bailian) and GpuGeek GPT-5.1.

Sweep concurrency = 1, 4, 8, 16, 32, 64 (configurable). At each level fire
N short identical requests in a ThreadPoolExecutor and record:

  - success rate         (HTTP 200 + non-empty content)
  - p50 / p95 / p99 latency
  - error breakdown      (rate-limit vs 5xx vs timeout vs other)
  - effective throughput (req/s sustained)

Picks the largest concurrency that satisfies the SLA:
  success_rate >= 0.95  AND  p95_latency_s <= sla_p95_s

Output is a markdown table + a one-line recommended setting, suitable to
paste into `memory.md`.

Usage:
    PYTHONPATH=src python3 tools/probe_concurrency.py --provider qwen
    PYTHONPATH=src python3 tools/probe_concurrency.py --provider gpugeek \
        --gpugeek_model "Vendor2/GPT-5.1"
    PYTHONPATH=src python3 tools/probe_concurrency.py --provider both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arch_policy.executor.qwen_worker import QwenWorker  # noqa: E402


# ---------------------------------------------------------------------------
# Provider plumbing
# ---------------------------------------------------------------------------

PROBE_SYS = "You are a helpful assistant."
# Realistic prompt size — mimics the rough shape of an HLE judge call
# (question + candidate answer + reference). Smaller wouldn't stress
# anything; bigger would only inflate cost without changing the
# bottleneck (we already know per-request latency is mostly model-side).
PROBE_USER = (
    "Judge whether the [response] is correct given the [question] and "
    "the [correct_answer] below. Reply with EXACTLY: correct: yes  OR  "
    "correct: no — nothing else.\n\n"
    "[question]: What is 2 + 2?\n"
    "[response]: 4\n"
    "[correct_answer]: 4"
)


def _qwen_call(model: str, timeout: float, max_tokens: int) -> tuple[bool, float, str]:
    """One Qwen API call. Returns (ok, latency_s, error_kind)."""
    t0 = time.time()
    try:
        w = QwenWorker(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model=model,
            timeout=timeout,
            max_retries=1,  # don't mask transient errors during a probe
        )
        out = w.chat(PROBE_SYS, PROBE_USER, max_new_tokens=max_tokens)
    except Exception as e:
        return (False, time.time() - t0, _classify_err(repr(e)))
    dur = time.time() - t0
    text = (out.text or "").strip()
    if text.startswith("[QwenWorker error"):
        return (False, dur, _classify_err(text))
    if not text:
        return (False, dur, "empty_response")
    return (True, dur, "")


def _gpugeek_call(model: str, timeout: float, max_tokens: int) -> tuple[bool, float, str]:
    """One GpuGeek (OpenAI-compatible) API call. Reasoning models like
    GPT-5.x can consume the entire `max_tokens` budget for hidden
    reasoning, leaving content empty — `max_tokens >= 64` is the safe
    floor."""
    try:
        from openai import OpenAI
    except ImportError:
        return (False, 0.0, "missing_openai_sdk")
    t0 = time.time()
    try:
        client = OpenAI(
            api_key=os.environ["GPUGEEK_API_KEY"],
            base_url="https://api.gpugeek.com/v1",
            timeout=timeout,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROBE_SYS},
                {"role": "user", "content": PROBE_USER},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        return (False, time.time() - t0, _classify_err(repr(e)))
    dur = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        return (False, dur, "empty_response")
    return (True, dur, "")


def _classify_err(s: str) -> str:
    s = s.lower()
    if "ratelimit" in s or "rate limit" in s or "429" in s or "too many" in s:
        return "rate_limit"
    if "timeout" in s or "timed out" in s:
        return "timeout"
    if "503" in s or "502" in s or "504" in s or "overloaded" in s:
        return "5xx"
    if "501" in s or "500" in s:
        return "5xx"
    if "connection" in s or "ssl" in s or "eof" in s:
        return "network"
    if "401" in s or "403" in s or "unauthorized" in s:
        return "auth"
    if "400" in s or "invalid" in s:
        return "client_4xx"
    return "other"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

@dataclass
class LevelResult:
    concurrency: int
    n_total: int
    n_ok: int
    latencies_ok: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.n_ok / max(1, self.n_total)

    @property
    def throughput_req_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.n_ok / self.wall_s

    def latency(self, q: float) -> float:
        if not self.latencies_ok:
            return 0.0
        xs = sorted(self.latencies_ok)
        idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[idx]


def run_one_level(
    call_fn,
    concurrency: int,
    n_per_level: int,
    timeout: float,
) -> LevelResult:
    """Fire `n_per_level` requests with `concurrency` workers."""
    res = LevelResult(concurrency=concurrency, n_total=n_per_level, n_ok=0)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_fn, timeout) for _ in range(n_per_level)]
        for f in as_completed(futs):
            try:
                ok, dur, err_kind = f.result()
            except Exception as e:
                ok, dur, err_kind = False, 0.0, _classify_err(repr(e))
            if ok:
                res.n_ok += 1
                res.latencies_ok.append(dur)
            else:
                res.errors[err_kind] = res.errors.get(err_kind, 0) + 1
    res.wall_s = time.time() - t0
    return res


def sweep(
    call_fn,
    levels: list[int],
    n_per_level: int,
    timeout: float,
    cooldown_s: float = 5.0,
) -> list[LevelResult]:
    out = []
    for c in levels:
        print(f"  [sweep] concurrency={c:>3} firing {n_per_level} requests...",
              flush=True)
        r = run_one_level(call_fn, c, n_per_level, timeout)
        print(
            f"  [sweep] concurrency={c:>3} → "
            f"ok={r.n_ok}/{r.n_total} ({100*r.success_rate:>5.1f}%) "
            f"p50={r.latency(0.5):.2f}s p95={r.latency(0.95):.2f}s "
            f"thr={r.throughput_req_per_s:.2f}req/s "
            f"errs={r.errors}",
            flush=True,
        )
        out.append(r)
        if cooldown_s > 0:
            time.sleep(cooldown_s)
    return out


def pick_recommended(
    results: list[LevelResult],
    sla_success: float,
    sla_p95_s: float,
) -> LevelResult | None:
    """Largest concurrency satisfying SLA."""
    passing = [r for r in results
               if r.success_rate >= sla_success and r.latency(0.95) <= sla_p95_s]
    if not passing:
        return None
    return max(passing, key=lambda r: r.concurrency)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def md_table(provider: str, model: str, results: list[LevelResult],
             sla_success: float, sla_p95_s: float) -> str:
    rec = pick_recommended(results, sla_success, sla_p95_s)
    lines = [
        f"### {provider} — `{model}`",
        "",
        f"SLA: success ≥ {100*sla_success:.0f}% AND p95 ≤ {sla_p95_s:.1f}s.",
        "",
        "| concurrency | ok/N | success | p50 | p95 | p99 | thr (req/s) | error breakdown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        errs = ", ".join(f"{k}={v}" for k, v in sorted(r.errors.items())) or "—"
        lines.append(
            f"| {r.concurrency} | {r.n_ok}/{r.n_total} | "
            f"{100*r.success_rate:.1f}% | "
            f"{r.latency(0.5):.2f}s | {r.latency(0.95):.2f}s | "
            f"{r.latency(0.99):.2f}s | {r.throughput_req_per_s:.2f} | "
            f"{errs} |"
        )
    lines.append("")
    if rec is None:
        lines.append("**Recommended**: NONE — no level satisfied the SLA. "
                     "Reduce concurrency below 1 (i.e. serialize), or relax SLA.")
    else:
        lines.append(
            f"**Recommended**: `concurrency = {rec.concurrency}` "
            f"(throughput ≈ {rec.throughput_req_per_s:.2f} req/s, "
            f"p95 ≈ {rec.latency(0.95):.2f}s)."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["qwen", "gpugeek", "both"],
                   default="both")
    p.add_argument("--qwen_model", default="qwen3.7-max")
    p.add_argument("--gpugeek_model", default="Vendor2/GPT-5.1")
    p.add_argument("--levels", default="1,4,8,16,32,64",
                   help="Comma-separated concurrency levels to test.")
    p.add_argument("--n_per_level", type=int, default=16,
                   help="Requests to fire at each concurrency level.")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-request timeout in seconds.")
    p.add_argument("--cooldown_s", type=float, default=5.0,
                   help="Seconds to sleep between levels (let upstream cool).")
    p.add_argument("--sla_success", type=float, default=0.95)
    p.add_argument("--sla_p95_s", type=float, default=20.0)
    p.add_argument("--max_tokens", type=int, default=128,
                   help="Per-request max_tokens. Reasoning models (GPT-5.x) "
                        "need ≥ 64 to leave content room after hidden "
                        "reasoning. Default 128 (judge-realistic).")
    p.add_argument("--out_md", default=None,
                   help="If set, also write the markdown report to this file.")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    sections: list[str] = []

    if args.provider in ("qwen", "both"):
        print(f"\n=== Qwen sweep ({args.qwen_model}, "
              f"max_tokens={args.max_tokens}) ===", flush=True)
        results = sweep(
            lambda t: _qwen_call(args.qwen_model, t, args.max_tokens),
            levels, args.n_per_level, args.timeout, args.cooldown_s,
        )
        sections.append(md_table("Qwen (Aliyun Bailian)", args.qwen_model,
                                 results, args.sla_success, args.sla_p95_s))

    if args.provider in ("gpugeek", "both"):
        print(f"\n=== GpuGeek sweep ({args.gpugeek_model}, "
              f"max_tokens={args.max_tokens}) ===", flush=True)
        results = sweep(
            lambda t: _gpugeek_call(args.gpugeek_model, t, args.max_tokens),
            levels, args.n_per_level, args.timeout, args.cooldown_s,
        )
        sections.append(md_table("GpuGeek (OpenAI-compatible)",
                                 args.gpugeek_model,
                                 results, args.sla_success, args.sla_p95_s))

    md = "\n\n---\n\n".join(sections)
    print("\n" + "=" * 70)
    print(md)
    print("=" * 70 + "\n")
    if args.out_md:
        Path(args.out_md).write_text(md + "\n")
        print(f"  wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
