"""Per-task graders.

Different benchmarks need different grading logic:
  - GSM8K / synthetic   : numeric exact match
  - MATH                : sympy-based answer comparison
  - HumanEval / MBPP    : execute generated code against unit tests

Each grader takes a (prediction_text, gold, metadata?) and returns a float
in [0, 1] (we mostly use 0/1 but keep it continuous-friendly).
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# Numeric / string normalization
# ---------------------------------------------------------------------------

def _norm_str(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^[\"'`]+|[\"'`]+$", "", s)
    s = re.sub(r"[.,!?;:]+$", "", s)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s


def _extract_first_number(s: str) -> str | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else None


def _extract_last_boxed(s: str) -> str | None:
    """Find content of the last \\boxed{...} or 'Final answer: X' line."""
    # boxed{...}
    last = None
    for m in re.finditer(r"\\boxed\s*\{", s):
        start = m.end()
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = s[start : i - 1].strip()
    if last is not None:
        return last
    # "Final answer: X" / "Verified answer: X"
    m = re.search(
        r"(?:final\s*answer|verified\s*answer|my\s*answer)\s*[:=]\s*(.+?)(?:\n|$)",
        s,
        re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Numeric (GSM8K / synthetic)
# ---------------------------------------------------------------------------

def grade_numeric(prediction: str, gold: str) -> float:
    if not prediction or not gold:
        return 0.0
    p = _extract_first_number(prediction) or _norm_str(prediction)
    g = _extract_first_number(gold) or _norm_str(gold)
    if p is None or g is None:
        return 0.0
    try:
        return 1.0 if abs(float(p) - float(g)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if p == g else 0.0


# ---------------------------------------------------------------------------
# MATH (boxed answers, sympy-aware)
# ---------------------------------------------------------------------------

def _try_sympy_eq(a: str, b: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return False
    try:
        ea = parse_latex(a) if "\\" in a else parse_latex(f"({a})")
        eb = parse_latex(b) if "\\" in b else parse_latex(f"({b})")
        return bool(simplify(ea - eb) == 0)
    except Exception:
        return False


def grade_math(prediction: str, gold: str) -> float:
    if not prediction or not gold:
        return 0.0
    p_box = _extract_last_boxed(prediction) or prediction
    g = gold.strip()
    p = p_box.strip()

    # Strip whitespace + outer parens for direct compare.
    def _normalize(x: str) -> str:
        x = x.strip().rstrip(".").rstrip(",")
        x = x.replace(" ", "").replace("\\,", "").replace("\\!", "")
        x = re.sub(r"\\left|\\right", "", x)
        return x

    if _normalize(p) == _normalize(g):
        return 1.0
    # Numeric path
    pn = _extract_first_number(p)
    gn = _extract_first_number(g)
    if pn is not None and gn is not None:
        try:
            if abs(float(pn) - float(gn)) < 1e-4:
                return 1.0
        except ValueError:
            pass
    # Sympy path (latex equality)
    if _try_sympy_eq(p, g):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# HumanEval / MBPP — exec-based
# ---------------------------------------------------------------------------

def _exec_with_timeout(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Exec `code` in a fresh subprocess with a wall-clock timeout.

    Uses subprocess.Popen so we do NOT re-import the calling module (which
    multiprocessing.spawn does, and which is catastrophic when the caller
    is a training script that loads multi-GB models at import time).
    Returns (passed, message). `passed` is True iff the subprocess exits 0.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    last = err[-1] if err else "non-zero exit"
    return False, last[:200]


def _extract_python_code(prediction: str) -> str:
    """Pull the first ```python ... ``` block, or the whole text if none."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", prediction, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prediction.strip()


def grade_humaneval(prediction: str, metadata: dict) -> float:
    """Exec the prediction's code + unit tests; return 1.0 iff all tests pass.

    metadata must contain `prompt`, `test`, `entry_point` (HF humaneval format).
    """
    code = _extract_python_code(prediction)
    # Compose the runnable script: prediction + tests + check call.
    full = (
        code
        + "\n\n"
        + metadata.get("test", "")
        + f"\n\ncheck({metadata.get('entry_point', 'solution')})\n"
    )
    passed, _msg = _exec_with_timeout(full, timeout=8.0)
    return 1.0 if passed else 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade(prediction: str, sample) -> float:
    """Dispatch to the correct grader based on `sample.family`."""
    family = sample.family
    if family in ("gsm8k", "synthetic_arith", "synthetic_wordprob", "synthetic_logic"):
        return grade_numeric(prediction, sample.gold_answer)
    if family == "math":
        return grade_math(prediction, sample.gold_answer)
    if family in ("humaneval", "mbpp"):
        return grade_humaneval(prediction, sample.metadata)
    # Fallback to numeric/string match.
    return grade_numeric(prediction, sample.gold_answer)


__all__ = [
    "grade",
    "grade_humaneval",
    "grade_math",
    "grade_numeric",
]
