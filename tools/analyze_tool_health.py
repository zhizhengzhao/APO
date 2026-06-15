"""Per-tool health analyzer for GRPO `details.jsonl`.

Reads the per-step per-trace tool telemetry already saved by GRPO
training and aggregates:

  - tool_call_counts        (how often each tool is invoked)
  - tool_error_counts       (raised exceptions: python TIMEOUT, etc.)
  - search_stub_counts      (Serper-stub fallback: search/scrape failed)
  - tool_error_kinds        (per-exception-class histogram)

Outputs a markdown report with per-tool:
  - total calls
  - failure breakdown (error% + stub%)
  - top error / stub kinds

Why this matters: a tool with high stub rate may indicate a server-side
issue (Serper /scrape on certain URLs, rate-limit), a fragile
implementation, or a wrong choice of upstream — all of which the head
will struggle to learn around if the signal is "soft" (stub returned)
rather than "hard" (architecture penalised).

Usage:
    PYTHONPATH=src python3 tools/analyze_tool_health.py \\
        --details checkpoints/grpo_hle_v1/details.jsonl \\
        [--out_md reports/tool_health_grpo_hle_v1.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _bump(d: dict, k, v: int = 1) -> None:
    d[k] = d.get(k, 0) + v


def analyze(details_path: Path) -> dict:
    """Aggregate per-tool stats over all per-arch samples in `details.jsonl`."""
    n_steps = 0
    n_archs = 0
    per_tool_calls: dict[str, int] = defaultdict(int)
    per_tool_errs: dict[str, int] = defaultdict(int)
    per_tool_stubs: dict[str, int] = defaultdict(int)
    per_tool_err_kinds: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    with open(details_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_steps += 1
            for task in rec.get("per_task_details", []) or []:
                for sample in task.get("samples", []) or []:
                    n_archs += 1
                    for k, v in (sample.get("tool_call_counts") or {}).items():
                        per_tool_calls[k] += int(v)
                    for k, v in (sample.get("tool_error_counts") or {}).items():
                        per_tool_errs[k] += int(v)
                    for k, v in (sample.get("search_stub_counts") or {}).items():
                        per_tool_stubs[k] += int(v)
                    for k, v in (sample.get("tool_error_kinds") or {}).items():
                        # e.g. "python_exec:TIMEOUT" -> tool, kind
                        tool, _, kind = k.partition(":")
                        per_tool_err_kinds[tool][kind or "_"] += int(v)
    return {
        "n_steps": n_steps,
        "n_archs": n_archs,
        "calls": dict(per_tool_calls),
        "errs": dict(per_tool_errs),
        "stubs": dict(per_tool_stubs),
        "err_kinds": {t: dict(k) for t, k in per_tool_err_kinds.items()},
    }


def md_report(stats: dict, source: str) -> str:
    lines = [
        f"# Tool health report — `{source}`",
        "",
        f"Aggregated over **{stats['n_steps']} GRPO steps** "
        f"({stats['n_archs']} per-arch trace samples).",
        "",
        "## Per-tool summary",
        "",
        "| tool | calls | err | stub | err% | stub% | combined-fail% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_tools = sorted(set(stats["calls"]) | set(stats["errs"])
                       | set(stats["stubs"]))
    for t in all_tools:
        c = stats["calls"].get(t, 0)
        e = stats["errs"].get(t, 0)
        s = stats["stubs"].get(t, 0)
        if c == 0:
            err_pct = stub_pct = combined = "—"
        else:
            err_pct = f"{100*e/c:.1f}%"
            stub_pct = f"{100*s/c:.1f}%"
            combined = f"{100*(e+s)/c:.1f}%"
        # Visual flag on combined fail
        flag = ""
        if c > 0:
            combined_v = 100 * (e + s) / c
            if combined_v >= 25:
                flag = " 🔴"
            elif combined_v >= 10:
                flag = " ⚠️"
            elif combined_v < 1:
                flag = " ✅"
        lines.append(f"| `{t}` | {c} | {e} | {s} | {err_pct} | "
                     f"{stub_pct} | {combined}{flag} |")

    lines += [
        "",
        "**Definitions**",
        "",
        "- **calls** — total invocations across all GRPO steps and all "
        "per-arch traces.",
        "- **err** — tool raised a structured error (e.g. `python_exec` "
        "TIMEOUT, `pytest_runner` failure). Returned as `[<tool>] TIMEOUT` "
        "string; the agent sees the failure and can react.",
        "- **stub** — Serper-backed search tool failed (no key / 5xx / "
        "timeout / parse error). Returned as `[<tool>: <ExcClass>: <msg>]` "
        "stub string. NOT counted as `err` — Stub is the soft-failure "
        "fallback explicitly enabled by `_serper_post` when "
        "`ARCH_POLICY_STRICT_TOOLS != 1`.",
        "- **combined-fail%** — `(err + stub) / calls`. Tracks how often "
        "an architecture invokes the tool and gets nothing useful back. "
        "High values point to **fragile tools** (server-side issue, "
        "wrong upstream choice, brittle implementation) that the head "
        "will struggle to route around effectively.",
        "",
    ]

    if stats["err_kinds"]:
        lines += [
            "## Error class breakdown (per tool)",
            "",
            "| tool | error kind | count |",
            "|---|---|---:|",
        ]
        for t in sorted(stats["err_kinds"]):
            for k, v in sorted(stats["err_kinds"][t].items(),
                               key=lambda kv: -kv[1]):
                lines.append(f"| `{t}` | `{k}` | {v} |")
        lines.append("")
    else:
        lines.append("_(no `tool_error_kinds` data — older details.jsonl)_\n")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--details", required=True,
                   help="Path to details.jsonl from a GRPO training run.")
    p.add_argument("--out_md", default=None,
                   help="If set, also write the report to this file.")
    args = p.parse_args()

    details = Path(args.details).resolve()
    if not details.exists():
        print(f"ERROR: {details} not found", file=sys.stderr)
        return 2

    stats = analyze(details)
    md = md_report(stats, source=str(details.relative_to(Path.cwd())
                                     if details.is_relative_to(Path.cwd())
                                     else details))
    print(md)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
        print(f"  wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
