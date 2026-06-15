"""End-to-end health probe for the agent tool pool.

Run BEFORE any prod GRPO / eval run — especially after a fresh server,
env change, or tool refactor. Hits every tool with a small batch of
realistic queries and reports per-tool:
  - OK / FAIL counts,
  - p50 / p99 latency (ms),
  - sample output snippets,
  - any silent-degradation signals (stubs, Serper-fallback usage, etc.).

This catches:
  - missing Python deps (sympy / numpy / scipy / pint)
  - missing API keys (SERPER_API_KEY)
  - server firewall blocking Wikipedia direct API
  - Serper /scrape failing on specific URLs (the old 24% stub bug)
  - pytest not installed in the worker env

Usage:
    PYTHONPATH=src python3 tools/probe_tool_health.py
    PYTHONPATH=src python3 tools/probe_tool_health.py --serper-only
    PYTHONPATH=src python3 tools/probe_tool_health.py --free-only

Exits 0 iff every required tool passes its full case-list (failures on
optional / network-dependent paths still print but don't fail the script
unless --strict is given).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make `arch_policy.*` importable regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arch_policy.executor import tools as T  # noqa: E402


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    ok: bool
    latency_s: float
    output_snippet: str
    error: str = ""


@dataclass
class ToolReport:
    tool: str
    cases: list[CaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.cases if c.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.cases if not c.ok)

    @property
    def healthy(self) -> bool:
        return self.n_fail == 0

    def latency_stats(self) -> tuple[float, float]:
        xs = [c.latency_s for c in self.cases]
        if not xs:
            return (0.0, 0.0)
        p50 = statistics.median(xs)
        p99 = sorted(xs)[max(0, int(len(xs) * 0.99) - 1)] if len(xs) > 1 else xs[0]
        return (p50, p99)


def _run_case(name: str, fn, *args, success_check=lambda _o: True, **kwargs) -> CaseResult:
    """Time `fn(*args, **kwargs)` and judge OK via `success_check(output)`.
    Any raised exception is captured as a FAIL with the type+message."""
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception as e:
        return CaseResult(
            name=name, ok=False, latency_s=time.time() - t0,
            output_snippet="", error=f"{type(e).__name__}: {e}",
        )
    dur = time.time() - t0
    try:
        ok = bool(success_check(out))
    except Exception as e:
        ok = False
        return CaseResult(
            name=name, ok=False, latency_s=dur,
            output_snippet=str(out)[:160],
            error=f"check raised {type(e).__name__}: {e}",
        )
    return CaseResult(
        name=name, ok=ok, latency_s=dur,
        output_snippet=str(out)[:160],
    )


# ---------------------------------------------------------------------------
# Per-tool probes
# ---------------------------------------------------------------------------

def probe_python_exec() -> ToolReport:
    """Free, local. Exercises every dep we expect agents to `import`."""
    report = ToolReport(tool="python_exec")
    cases = [
        ("arith", "print(2 + 2 * 3)",
         lambda o: "8" in o and "STDOUT" in o),
        ("sympy_integrate",
         "from sympy import symbols, integrate; x = symbols('x'); "
         "print(integrate(x**2, x))",
         lambda o: "x**3/3" in o),
        ("sympy_solve",
         "from sympy import symbols, solve, Eq; "
         "x = symbols('x'); print(solve(Eq(x**2 - 4, 0), x))",
         lambda o: "[-2, 2]" in o),
        ("numpy_array",
         "import numpy as np; print(np.array([1,2,3]).sum())",
         lambda o: "6" in o),
        ("scipy_special",
         "from scipy.special import erf; print(round(erf(1), 4))",
         lambda o: "0.8427" in o),
        ("pint_units",
         "import pint; u = pint.UnitRegistry(); "
         "print((100 * u.meter).to(u.kilometer))",
         lambda o: "0.1" in o and "kilometer" in o),
        ("timeout_safe_sleep",
         "import time; time.sleep(0.5); print('done')",
         lambda o: "done" in o),
        ("stdout_stderr_split",
         "import sys; print('OUT'); print('ERR', file=sys.stderr)",
         lambda o: "STDOUT" in o and "STDERR" in o and "OUT" in o and "ERR" in o),
        ("markdown_fence_stripped",
         "```python\nprint('fence_ok')\n```",
         lambda o: "fence_ok" in o),
    ]
    for name, code, check in cases:
        report.cases.append(_run_case(name, T.python_exec, code, success_check=check))

    # Note any missing-dep failures explicitly — those are install fixes.
    for c in report.cases:
        if not c.ok and "ModuleNotFoundError" in c.output_snippet:
            report.notes.append(
                f"  → {c.name}: dependency missing. Fix: "
                f"`pip install {c.name.split('_')[0]}` on this host."
            )
    return report


def probe_pytest_runner() -> ToolReport:
    """Free, local. Pytest must be on PATH for the subprocess."""
    report = ToolReport(tool="pytest_runner")
    cases = [
        ("pass_basic",
         "def add(a, b): return a + b\n"
         "---TESTS---\n"
         "def test_add():\n"
         "    assert add(2, 3) == 5\n",
         lambda o: "1 passed" in o or "passed" in o),
        ("fail_assertion_visible",
         "def sub(a, b): return a - b\n"
         "---TESTS---\n"
         "def test_sub():\n"
         "    assert sub(5, 3) == 99, 'expected 99'\n",
         lambda o: "failed" in o.lower() or "assert" in o.lower()),
        ("split_on_def_test",  # no delimiter — uses heuristic split
         "def mul(a, b): return a * b\n"
         "def test_mul():\n"
         "    assert mul(2, 3) == 6\n",
         lambda o: "passed" in o or "1 passed" in o),
        ("no_test_at_all",
         "print('just a statement')\n",
         lambda o: "no tests found" in o.lower()),
    ]
    for name, spec, check in cases:
        report.cases.append(_run_case(name, T.pytest_runner, spec, success_check=check))
    return report


# ---------------------------------------------------------------------------
# Serper-backed tools (require SERPER_API_KEY)
# ---------------------------------------------------------------------------

def _serper_key_present() -> bool:
    return bool(os.environ.get("SERPER_API_KEY"))


def probe_web_search() -> ToolReport:
    report = ToolReport(tool="web_search")
    if not _serper_key_present():
        report.notes.append("  SKIPPED: SERPER_API_KEY not set.")
        return report
    queries = [
        ("entity_factual", "Albert Einstein date of birth"),
        ("technical_doc", "python multiprocessing.Pool docs"),
        ("recent_news", "GPT-5 release date"),
        ("non_english_safe", "中国 高考 数学"),
    ]
    for name, q in queries:
        report.cases.append(_run_case(
            name, T.web_search, q,
            success_check=lambda o: not _looks_like_stub(o) and len(o) > 200,
        ))
    return report


def probe_arxiv_search() -> ToolReport:
    report = ToolReport(tool="arxiv_search")
    if not _serper_key_present():
        report.notes.append("  SKIPPED: SERPER_API_KEY not set.")
        return report
    queries = [
        ("classic_paper", "Attention is all you need"),
        ("recent_topic", "deep research agent multi-agent reasoning"),
        ("math_paper", "Plackett-Luce neural ranking"),
    ]
    for name, q in queries:
        report.cases.append(_run_case(
            name, T.arxiv_search, q,
            success_check=lambda o: not _looks_like_stub(o) and len(o) > 100,
        ))
    return report


# ---------------------------------------------------------------------------
# wikipedia_search — direct + Serper-fallback
# ---------------------------------------------------------------------------

def probe_wikipedia_search() -> ToolReport:
    """Tests both the direct Wikipedia REST path AND notices if it
    silently fell back to Serper (firewall / 5xx surface)."""
    report = ToolReport(tool="wikipedia_search")
    queries = [
        ("classic_entity", "Albert Einstein"),
        ("technical_concept", "Watterson estimator"),
        ("historical_event", "Cuban Missile Crisis"),
        ("nonexistent", "zzzz_totally_invented_topic_xyzzy_qq"),
    ]
    # First reset stub counter — we use it to detect Serper-fallback usage.
    T.reset_search_stub_counts()
    n_fallback_used = 0
    for name, q in queries:
        # Functional success criterion: the agent got SOMETHING useful
        # back. Accept either the direct path (has `## ` markdown
        # headers for each title) or the Serper fallback (has numbered
        # list + the "(Serper fallback, ...)" tag) or the legitimate
        # empty-result sentinel. We deliberately do NOT mark fallback
        # usage as FAIL — fallback IS the success path on a degraded
        # network; we surface it via a separate `note` so production
        # operators can see the degradation cost.
        case = _run_case(
            name, T.wikipedia_search, q,
            success_check=lambda o: (
                "no results" in o.lower()
                or "## " in o                          # direct path
                or ("serper fallback" in o.lower()     # fallback path
                    and len(o) > 200)
            ),
        )
        if "serper fallback" in case.output_snippet.lower():
            n_fallback_used += 1
        report.cases.append(case)

    snap = T.snapshot_search_stub_counts()
    if n_fallback_used > 0:
        report.notes.append(
            f"  ⚠️  {n_fallback_used}/{len(queries)} direct Wikipedia calls "
            f"fell through to Serper fallback (server may be firewalled "
            f"from wikipedia.org). Fallback worked, so functionality is "
            f"intact, but cost is higher (paid Serper vs free direct)."
        )
    if snap.get("wikipedia_search", 0) > 0:
        report.notes.append(
            f"  ❌  wikipedia_search returned {snap['wikipedia_search']} "
            f"offline-stubs — direct AND Serper both failed. Check "
            f"SERPER_API_KEY and network reach."
        )
    return report


# ---------------------------------------------------------------------------
# Stub / error detection heuristics
# ---------------------------------------------------------------------------

def _looks_like_stub(out: str) -> bool:
    """A tool's response indicates it FAILED to retrieve real content."""
    out = out.lower()
    return any(
        s in out for s in (
            "stub", "api key missing",
            "urlerror", "httperror", "timeouterror",
            "no results for", "no text extracted",
        )
    )


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _color_ok(b: bool) -> str:
    return f"{GREEN}OK{RESET}" if b else f"{RED}FAIL{RESET}"


def print_report(reports: list[ToolReport]) -> None:
    print()
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  Tool health probe report{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    serper_set = _serper_key_present()
    print(f"  SERPER_API_KEY:        {'SET' if serper_set else 'NOT SET'}")
    print(f"  ARCH_POLICY_STRICT_TOOLS: "
          f"{os.environ.get('ARCH_POLICY_STRICT_TOOLS', '(default — off)')}")
    print()

    for r in reports:
        p50, p99 = r.latency_stats()
        if not r.cases:
            print(f"  {BOLD}{r.tool:<20}{RESET}  "
                  f"{YELLOW}SKIPPED{RESET}")
            for n in r.notes:
                print(n)
            continue
        status_color = GREEN if r.healthy else RED
        print(f"  {BOLD}{r.tool:<20}{RESET}  "
              f"{status_color}{r.n_ok}/{len(r.cases)} OK{RESET}  "
              f"p50={p50 * 1000:.0f}ms  p99={p99 * 1000:.0f}ms")
        for c in r.cases:
            indent = "      "
            line = (f"{indent}{c.name:<28}  {_color_ok(c.ok)}  "
                    f"{c.latency_s * 1000:>5.0f}ms")
            print(line)
            if not c.ok:
                if c.error:
                    print(f"{indent}    EXC: {c.error}")
                if c.output_snippet:
                    print(f"{indent}    OUT: {c.output_snippet}")
        for n in r.notes:
            print(n)
        print()

    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    n_total_fail = sum(r.n_fail for r in reports if r.cases)
    n_required_fail = sum(
        r.n_fail for r in reports
        if r.cases and r.tool in ("python_exec", "pytest_runner")
    )
    if n_required_fail > 0:
        print(f"  {RED}❌  {n_required_fail} REQUIRED-tool failures.{RESET}")
        print(f"     python_exec / pytest_runner MUST pass before any prod run.")
    elif n_total_fail > 0:
        print(f"  {YELLOW}⚠️  {n_total_fail} optional failures "
              f"(network/key-dependent). Review notes above.{RESET}")
    else:
        print(f"  {GREEN}✓  All tools healthy.{RESET}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--free-only", action="store_true",
                   help="Skip Serper-backed tools (no API spend).")
    p.add_argument("--serper-only", action="store_true",
                   help="Skip local-only probes; only test Serper-dependent.")
    p.add_argument("--strict", action="store_true",
                   help="Non-zero exit code if ANY case fails (default: only "
                        "required-tool failures cause non-zero exit).")
    args = p.parse_args()

    reports: list[ToolReport] = []
    if not args.serper_only:
        reports.append(probe_python_exec())
        reports.append(probe_pytest_runner())
    if not args.free_only:
        reports.append(probe_web_search())
        reports.append(probe_arxiv_search())
        reports.append(probe_wikipedia_search())

    print_report(reports)

    n_required_fail = sum(
        r.n_fail for r in reports
        if r.cases and r.tool in ("python_exec", "pytest_runner")
    )
    n_total_fail = sum(r.n_fail for r in reports if r.cases)
    if args.strict and n_total_fail > 0:
        return 1
    return 0 if n_required_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
