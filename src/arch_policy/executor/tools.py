"""Tool implementations available to agents (v3).

A tool is a function `callable[[str], str]` taking a single arg-string and
returning a (truncated) result string. Tools are deliberately simple: they
let any role experiment, and the executor's reward / cost accounting takes
care of "did using the tool actually help".

The minimal set is:

  - python_exec(code)   — sandboxed Python via subprocess, 5-second timeout
  - sympy_check(expr)   — symbolic check / simplification with sympy
  - web_search(query)   — currently a stub (returns "no results"); plug in
                          a real API later if/when needed.

Every tool's output is capped at MAX_OUTPUT_CHARS to keep transcripts
manageable.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap
from typing import Callable

MAX_OUTPUT_CHARS = 1500
PYTHON_TIMEOUT_S = 5.0


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    return s[: MAX_OUTPUT_CHARS - 32] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# python_exec
# ---------------------------------------------------------------------------

def python_exec(code: str) -> str:
    """Run `code` in a fresh `python` subprocess. Capture stdout+stderr.

    No filesystem isolation — relies on a 5-second timeout to prevent
    pathological loops. For production you'd want a real sandbox.
    """
    code = code.strip()
    if not code:
        return "[python_exec] no code provided"
    # Strip enclosing markdown fences if the LLM adds them
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=PYTHON_TIMEOUT_S,
            text=True,
        )
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired:
        return f"[python_exec] TIMEOUT after {PYTHON_TIMEOUT_S}s"
    except Exception as e:  # pragma: no cover (subprocess failure)
        return f"[python_exec] ERROR: {type(e).__name__}: {e}"

    parts = []
    if out.strip():
        parts.append(f"STDOUT:\n{out.rstrip()}")
    if err.strip():
        parts.append(f"STDERR:\n{err.rstrip()}")
    if not parts:
        parts.append("[python_exec] (no output)")
    return _truncate("\n\n".join(parts))


# ---------------------------------------------------------------------------
# sympy_check
# ---------------------------------------------------------------------------

def sympy_check(expr: str) -> str:
    """Try to simplify or evaluate `expr` with SymPy.

    Accepts:
      - a plain expression like '2+3*4'
      - an equation like 'x**2 - 5*x + 6 = 0'  → solves for symbols
      - 'simplify <expr>' → simplifies
    """
    expr = expr.strip()
    if not expr:
        return "[sympy_check] no expression provided"
    try:
        import sympy
    except ImportError:
        return "[sympy_check] sympy not installed"

    try:
        if expr.lower().startswith("simplify "):
            e = sympy.sympify(expr[len("simplify "):], evaluate=False)
            return f"[sympy_check] simplify: {sympy.simplify(e)}"
        if "=" in expr and "==" not in expr:
            lhs, rhs = expr.split("=", 1)
            eq = sympy.Eq(sympy.sympify(lhs), sympy.sympify(rhs))
            sols = sympy.solve(eq)
            return f"[sympy_check] solutions: {sols}"
        e = sympy.sympify(expr)
        val = sympy.simplify(e)
        try:
            num = float(val.evalf())
            return f"[sympy_check] {val}  (numerical: {num})"
        except (TypeError, ValueError):
            return f"[sympy_check] {val}"
    except Exception as e:
        return f"[sympy_check] ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """Stub web search — returns a "no results" placeholder.

    Plug in a real backend (e.g. Tavily, Serper, Brave) by replacing this
    function. Kept as a stub so the codebase has zero external auth deps.
    """
    query = query.strip()
    if not query:
        return "[web_search] empty query"
    return (
        f"[web_search] query={query!r}\n"
        "  No results returned (this is a stub. Plug in a real backend if needed)."
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ToolFn = Callable[[str], str]

TOOLS: dict[str, ToolFn] = {
    "python_exec": python_exec,
    "sympy_check": sympy_check,
    "web_search": web_search,
}


def call_tool(name: str, args: str) -> str:
    if name not in TOOLS:
        return f"[unknown tool {name!r}]; available: {sorted(TOOLS)}"
    return TOOLS[name](args)


__all__ = ["TOOLS", "call_tool", "python_exec", "sympy_check", "web_search"]
