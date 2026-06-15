"""Per-bench evaluation analyzer.

Reads the JSONL from `scripts/04_evaluate.py --bench <name>` and emits
a markdown report: overall accuracy + 95% CI, per-subdomain accuracy,
per-subdomain top-K role multisets with mean reward, cross-subdomain
role usage, cost/depth distributions, and engineering-health counters.

Bench-agnostic: subdomain order comes from `bench.get(name).subdomains`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — well-behaved at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _per_subject_acc(records: list[dict]) -> dict[str, tuple[int, int]]:
    """{subject: (n_correct, n_total)}."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        acc[r.get("subject") or "Unknown"][0] += int(round(r["score"]))
        acc[r.get("subject") or "Unknown"][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def _top_arch_compositions(
    records: list[dict], subject: str | None = None, k: int = 5,
) -> list[tuple[tuple[str, ...], int, float]]:
    """Top-k (role_multiset, count, mean_score) for records in `subject`."""
    acc: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        if subject is not None and (r.get("subject") or "Unknown") != subject:
            continue
        for mset in r.get("role_multisets") or []:
            acc[tuple(mset)][0] += 1
            acc[tuple(mset)][1] += int(round(r["score"]))
    top = sorted(acc.items(), key=lambda kv: -kv[1][0])[:k]
    return [(ms, c, (s / c) if c else 0.0) for ms, (c, s) in top]


def _role_usage(records: list[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for mset in r.get("role_multisets") or []:
            c.update(mset)
    return c


def _stat_distribution(records: list[dict], field: str) -> dict[str, float]:
    """mean / p50 / p90 / p99 of a per-arch list (or scalar) field."""
    xs: list[float] = []
    for r in records:
        v = r.get(field)
        if isinstance(v, list):
            xs.extend(float(x) for x in v if x is not None)
        elif v is not None:
            xs.append(float(v))
    if not xs:
        return {"n": 0}
    xs.sort()
    pct = lambda p: xs[max(0, min(len(xs) - 1, int(p * (len(xs) - 1))))]
    return {"n": len(xs), "mean": sum(xs) / len(xs),
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}


def _format_role_multiset(ms: tuple[str, ...]) -> str:
    """e.g. ('Solver', 'Solver', 'Verifier') → '2×Solver + Verifier'."""
    if not ms:
        return "(empty)"
    counts = Counter(ms)
    parts = [f"{n}×{role}" if n > 1 else role
             for role, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return " + ".join(parts)


def render_report(
    records: list[dict], bench_name: str, subdomains: tuple[str, ...],
    judge_label: str = "(unknown)",
) -> str:
    n_total = len(records)
    n_correct = int(sum(round(r["score"]) for r in records))
    lo, hi = _wilson_ci(n_correct, n_total)
    per_subj = _per_subject_acc(records)
    # Declared subdomains first, then any unexpected leftovers.
    ordered = list(subdomains) + sorted(s for s in per_subj if s not in subdomains)

    out: list[str] = [
        f"# {bench_name.upper()} evaluation summary",
        "",
        f"- Records:  **{n_total}**",
        f"- Accuracy: **{n_correct / max(1, n_total):.4f}** "
        f"(95% CI {lo:.3f}–{hi:.3f}; {n_correct}/{n_total} correct)",
        f"- Judge:    {judge_label}",
        "",
        "## Per-subdomain accuracy",
        "",
        "| Subject | n | accuracy | 95% CI |",
        "|---|---:|---:|:---|",
    ]
    for subj in ordered:
        if subj not in per_subj:
            continue
        c, n = per_subj[subj]
        slo, shi = _wilson_ci(c, n)
        out.append(f"| {subj} | {n} | {c / max(1, n):.3f} | {slo:.2f}–{shi:.2f} |")

    out += ["", "## Per-subdomain top architectures (top-5 role multisets)", ""]
    for subj in ordered:
        if subj not in per_subj:
            continue
        n = per_subj[subj][1]
        out.append(f"### {subj} (n={n})")
        top = _top_arch_compositions(records, subject=subj, k=5)
        if not top:
            out += ["  _(no architectures sampled)_", ""]
            continue
        out += ["| Architecture | uses | mean reward |", "|---|---:|---:|"]
        for ms, count, mean_r in top:
            out.append(f"| {_format_role_multiset(ms)} | {count} | {mean_r:.3f} |")
        out.append("")

    role_c = _role_usage(records)
    total_role = sum(role_c.values()) or 1
    out += ["## Cross-subdomain role usage", "", "| Role | uses | share |", "|---|---:|---:|"]
    for role, c in role_c.most_common():
        out.append(f"| {role} | {c} | {c / total_role:.1%} |")

    out += ["", "## Cost / depth distribution", ""]
    for field in ("llm_calls", "n_cycles", "n_active_per_arch", "n_edges_per_arch"):
        s = _stat_distribution(records, field)
        if s.get("n", 0) == 0:
            continue
        out.append(f"- `{field}`: n={s['n']}, mean={s['mean']:.2f}, "
                   f"p50={s['p50']:.1f}, p90={s['p90']:.1f}, p99={s['p99']:.1f}")

    n_stub = sum(r.get("search_stub_total", 0) for r in records)
    n_runerr = sum(r.get("run_errors_total", 0) for r in records)
    out += [
        "",
        "## Engineering health (lower is better)",
        "",
        f"- Search-tool stub returns: **{n_stub}** "
        f"({n_stub / max(1, n_total):.2f} per task)",
        f"- Run-level errors:         **{n_runerr}** "
        f"({n_runerr / max(1, n_total):.2f} per task)",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    help="Bench id, e.g. 'hle' — used for the subdomain order.")
    ap.add_argument("--eval_jsonl", required=True,
                    help="Output of `04_evaluate.py --bench <name>`.")
    ap.add_argument("--out_md", default=None,
                    help="Write markdown report here; stdout if unset.")
    ap.add_argument("--judge_label", default="(unknown)")
    args = ap.parse_args()

    from arch_policy import bench as bench_mod
    adapter = bench_mod.get(args.bench)
    records = [json.loads(line) for line in Path(args.eval_jsonl).open() if line.strip()]
    if not records:
        print(f"[analyze] no records in {args.eval_jsonl}", file=sys.stderr)
        return 1
    print(f"[analyze] {len(records)} records from {args.eval_jsonl}", file=sys.stderr)

    md = render_report(records, args.bench, adapter.subdomains, args.judge_label)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md)
        print(f"[analyze] report → {Path(args.out_md).resolve()}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
