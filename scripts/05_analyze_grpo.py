"""Offline analysis of a GRPO run.

Reads `<out_dir>/history.json` and `<out_dir>/details.jsonl` and writes
to `<out_dir>/report/`:
  - summary.md      — human-readable overview (run health, tool quality,
                      protocol compliance, training dynamics)
  - per_step.csv    — one row per GRPO step
  - tool_rates.csv  — per-tool call/error counts

Usage::

    python scripts/05_analyze_grpo.py --out_dir <grpo_out_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _load(out_dir: Path) -> tuple[list[dict], list[dict]]:
    h = (out_dir / "history.json")
    d = (out_dir / "details.jsonl")
    history = json.loads(h.read_text()) if h.exists() else []
    details = [json.loads(line) for line in d.read_text().splitlines() if line.strip()] if d.exists() else []
    return history, details


def summarize(history: list[dict], details: list[dict]) -> dict:
    n_steps = len(details)
    if n_steps == 0:
        return {"error": "no details.jsonl found"}

    per_step = []
    tool_calls: Counter = Counter()
    tool_errs:  Counter = Counter()
    tool_err_kinds: Counter = Counter()
    term_total = Counter()
    role_uses: Counter = Counter()  # how many times each role was active
    n_active_dist: list[int] = []
    n_edges_dist: list[int] = []
    n_calls_correct: list[float] = []   # n_calls within correct samples
    n_unanimous_correct = 0
    n_unanimous_wrong   = 0
    n_mixed             = 0

    for d in details:
        step = d["step"]
        elapsed = d["elapsed"]
        all_samples = [s for pt in d["per_task_details"] for s in pt["samples"]]
        n = len(all_samples)
        n_correct = sum(s["correct"] for s in all_samples)
        n_hit_cyc = sum(s["hit_cycle_cap"] for s in all_samples)
        n_hit_wc  = sum(int(s.get("hit_wall_clock", False)) for s in all_samples)
        n_hit_cc  = sum(int(s.get("hit_call_cap",   False)) for s in all_samples)
        # n_api_errors = INFRA failures (masked from advantage);
        # n_arch_caps_hit = arch's choice to over-iterate (NOT masked).
        n_api       = sum(int(s.get("n_api_errors",    0)) for s in all_samples)
        n_arch_caps = sum(int(s.get("n_arch_caps_hit", 0)) for s in all_samples)
        mean_calls = sum(s["n_calls"] for s in all_samples) / n
        for s in all_samples:
            for k, v in (s.get("tool_call_counts") or {}).items():
                tool_calls[k] += v
            for k, v in (s.get("tool_error_counts") or {}).items():
                tool_errs[k] += v
            for k, v in (s.get("tool_error_kinds") or {}).items():
                tool_err_kinds[k] += v
            for k, v in (s.get("termination_breakdown") or {}).items():
                term_total[k] += int(v)
            for r in s.get("active_roles", []):
                role_uses[r] += 1
            n_active_dist.append(int(s["n_active"]))
            n_edges_dist.append(int(s.get("edges_count", 0)))
            if int(s["correct"]) == 1:
                n_calls_correct.append(float(s["n_calls"]))
        # Per-task unanimity check (within each batch task across G samples)
        for pt in d["per_task_details"]:
            ss = pt["samples"]
            cs = [int(x["correct"]) for x in ss]
            if all(c == 1 for c in cs): n_unanimous_correct += 1
            elif all(c == 0 for c in cs): n_unanimous_wrong += 1
            else: n_mixed += 1
        hist = next((h for h in history if h.get("step") == step), {})
        per_step.append({
            "step": step,
            "elapsed_s": round(elapsed, 1),
            "n_samples": n,
            "correct_pct": round(n_correct / n * 100, 1),
            "mean_calls": round(mean_calls, 2),
            "n_hit_cyc": n_hit_cyc,
            "n_hit_wc":  n_hit_wc,
            "n_hit_cc":  n_hit_cc,
            "n_api":     n_api,           # INFRA failures (mask gradient)
            "n_arch_caps": n_arch_caps,   # arch-attributable cap firings
            "loss": round(hist.get("loss", float("nan")), 3),
            "reward_mean": round(hist.get("reward_mean", float("nan")), 3),
            "entropy": round(hist.get("entropy", float("nan")), 3),
        })

    # Step-deltas
    last_t = 0.0
    for r in per_step:
        r["step_dt_s"] = round(r["elapsed_s"] - last_t, 1)
        last_t = r["elapsed_s"]

    total_turns = sum(term_total.values())
    # skip_wall_clock and skip_hit_cap are architecture-attributable
    # (arch's chosen path overran). Engineering noise = real INFRA failures
    # only (API error sentinels, empty-text returns).
    healthy_turns = term_total["submit_implicit"] + term_total["skipped_explicit"]
    arch_turns    = term_total["skip_hit_cap"] + term_total["skip_wall_clock"]
    eng_turns     = term_total["skip_worker_error"] + term_total["skip_empty_text"]

    total_traces = sum(r["n_samples"] for r in per_step)
    total_correct = sum(int(r["correct_pct"] * r["n_samples"] / 100) for r in per_step)

    return {
        "per_step": per_step,
        "tool_calls": dict(tool_calls),
        "tool_errs":  dict(tool_errs),
        "tool_err_kinds": dict(tool_err_kinds),
        "term_total":  dict(term_total),
        "n_steps":     n_steps,
        "n_traces":    total_traces,
        "n_turns":     total_turns,
        "healthy_pct": round(healthy_turns / max(1, total_turns) * 100, 1),
        "arch_pct":    round(arch_turns    / max(1, total_turns) * 100, 1),
        "eng_pct":     round(eng_turns     / max(1, total_turns) * 100, 1),
        "correct_pct": round(total_correct / max(1, total_traces) * 100, 1),
        "role_uses":   dict(role_uses),
        "n_active_mean": round(statistics.mean(n_active_dist), 2) if n_active_dist else 0,
        "n_edges_mean":  round(statistics.mean(n_edges_dist), 2) if n_edges_dist else 0,
        "n_calls_correct_mean":   round(statistics.mean(n_calls_correct), 2) if n_calls_correct else 0,
        "n_calls_correct_median": round(statistics.median(n_calls_correct), 2) if n_calls_correct else 0,
        "unanimity": {
            "all_correct": n_unanimous_correct,
            "all_wrong":   n_unanimous_wrong,
            "mixed":       n_mixed,
        },
    }


def write_report(out_dir: Path, s: dict) -> None:
    rep = out_dir / "report"
    rep.mkdir(parents=True, exist_ok=True)

    # per_step.csv
    if s["per_step"]:
        with open(rep / "per_step.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s["per_step"][0].keys()))
            w.writeheader()
            for row in s["per_step"]:
                w.writerow(row)

    # tool_rates.csv
    with open(rep / "tool_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "calls", "errors", "err_pct"])
        for k in sorted(set(s["tool_calls"]) | set(s["tool_errs"])):
            c = s["tool_calls"].get(k, 0); e = s["tool_errs"].get(k, 0)
            w.writerow([k, c, e, round(e / max(1, c) * 100, 1)])

    # summary.md
    md = []
    md.append(f"# GRPO Run Report\n\n")
    md.append(f"- **Steps**: {s['n_steps']}\n")
    md.append(f"- **Traces**: {s['n_traces']}\n")
    md.append(f"- **Turns**:  {s['n_turns']}\n")
    md.append(f"- **Correctness**: {s['correct_pct']}%\n\n")

    md.append(f"## Turn-level health\n\n")
    md.append(f"| Category | % |\n|---|---|\n")
    md.append(f"| healthy (submit + skip_explicit) | {s['healthy_pct']}% |\n")
    md.append(f"| arch-attributable (skip_hit_cap + skip_wall_clock) | {s['arch_pct']}% |\n")
    md.append(f"| engineering noise (worker_err + empty_text) | {s['eng_pct']}% |\n\n")

    md.append(f"## Tool quality\n\n")
    md.append(f"| Tool | Calls | Errors | Err % |\n|---|---|---|---|\n")
    for k in sorted(set(s["tool_calls"]) | set(s["tool_errs"])):
        c = s["tool_calls"].get(k, 0); e = s["tool_errs"].get(k, 0)
        md.append(f"| {k} | {c} | {e} | {e/max(1,c)*100:.1f}% |\n")
    if s["tool_err_kinds"]:
        md.append(f"\nError breakdown by kind:\n\n")
        md.append(f"| tool:kind | n |\n|---|---|\n")
        for k in sorted(s["tool_err_kinds"]):
            md.append(f"| {k} | {s['tool_err_kinds'][k]} |\n")
    md.append("\n")

    md.append(f"## Architecture dynamics\n\n")
    md.append(f"- Mean active agents: **{s['n_active_mean']}**\n")
    md.append(f"- Mean edges per arch: **{s['n_edges_mean']}**\n")
    md.append(f"- Within-correct n_calls: mean={s['n_calls_correct_mean']}, median={s['n_calls_correct_median']}\n\n")
    md.append(f"Role activations (sum over all sampled archs):\n\n")
    md.append(f"| Role | Times active |\n|---|---|\n")
    for r in sorted(s["role_uses"], key=lambda k: -s["role_uses"][k]):
        md.append(f"| {r} | {s['role_uses'][r]} |\n")
    md.append("\n")

    u = s["unanimity"]; ut = u["all_correct"] + u["all_wrong"] + u["mixed"]
    md.append(f"## Per-task unanimity ({ut} task instances)\n\n")
    md.append(f"| Outcome | n | % |\n|---|---|---|\n")
    md.append(f"| all G correct | {u['all_correct']} | {u['all_correct']/max(1,ut)*100:.1f}% |\n")
    md.append(f"| all G wrong   | {u['all_wrong']}   | {u['all_wrong']/max(1,ut)*100:.1f}% |\n")
    md.append(f"| mixed (signal)| {u['mixed']}       | {u['mixed']/max(1,ut)*100:.1f}% |\n\n")

    md.append(f"## Last 10 steps\n\n")
    md.append(f"| step | dt(s) | correct% | mean_calls | reward | loss |\n")
    md.append(f"|---|---|---|---|---|---|\n")
    for r in s["per_step"][-10:]:
        md.append(f"| {r['step']} | {r['step_dt_s']} | {r['correct_pct']}% | "
                  f"{r['mean_calls']} | {r['reward_mean']} | {r['loss']} |\n")

    (rep / "summary.md").write_text("".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="GRPO output directory containing history.json + details.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        print(f"[analyze] {out_dir} not found", file=sys.stderr)
        return 1

    history, details = _load(out_dir)
    s = summarize(history, details)
    if "error" in s:
        print(json.dumps(s, indent=2), file=sys.stderr)
        return 1
    write_report(out_dir, s)
    # Print top-line summary to stdout for quick eyeballing.
    print(json.dumps({k: v for k, v in s.items() if k != "per_step"}, indent=2))
    print(f"\n[analyze] report written to {out_dir/'report'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
