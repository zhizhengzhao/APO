"""Per-task graders. Each returns a score in [0, 1] (mostly 0/1 today;
continuous-friendly for future partial-credit graders).

Dispatch by `sample.family` lives in `grade()` at the bottom.
"""

from __future__ import annotations

import re
import os
import subprocess
import sys

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
# Multiple-choice letter (MMLU / GPQA / ARC)
# ---------------------------------------------------------------------------

# Explicit "answer" patterns + our role markers (Candidate/Verified/Refined/
# Endorsed/ANSWER: X), so heuristic_extract fallbacks hit the explicit path
# more often and the noisy bare-letter fallback fires less.
_MC_LETTER_RE = re.compile(
    r"(?:^|[^a-zA-Z])"
    r"(?:answer|the\s+answer\s+is|final\s+answer|"
    r"candidate|verified|refined|endorsed)"
    r"\s*[:=]?\s*"
    r"\(?([A-Da-d])\)?\b",
    re.IGNORECASE,
)
_BARE_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def grade_multiple_choice(prediction: str, gold: str) -> float:
    """Grade a single-letter (A/B/C/D) MC answer.

    (1) Last explicit marker (Answer/Final/Candidate/Verified/Refined/
        Endorsed/"the answer is" X) — last wins so a recap beats an earlier
        exploratory mention. (2) else last bare A-D letter — imperfect
        ("pick C, then A is also possible" mis-attributes), so (1) carries
        well-behaved Synth/heuristic_extract output.
    """
    if not prediction or not gold:
        return 0.0
    g = gold.strip().upper()
    if g not in {"A", "B", "C", "D"}:
        return 0.0
    # Take the LAST explicit-marker match (a recap at the end wins over
    # an exploratory "Answer A is wrong" earlier).
    explicit = list(_MC_LETTER_RE.finditer(prediction))
    if explicit:
        return 1.0 if explicit[-1].group(1).upper() == g else 0.0
    matches = _BARE_LETTER_RE.findall(prediction)
    if matches:
        return 1.0 if matches[-1].upper() == g else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Short-answer string grading (BrowseComp / HLE)
# ---------------------------------------------------------------------------

def _normalize_short_answer(s: str) -> str:
    """Aggressively normalize short answers so format mismatches between
    LLM output and gold don't cause false negatives.

    Strips: surrounding quotes, leading "answer:" preamble, trailing period,
    a/an/the article, extra whitespace, case.
    """
    s = s.strip().lower()
    # heuristic_extract wraps code-block answers as ```python\n<code>```.
    # Strip the optional language tag explicitly before the backtick
    # strip below, otherwise "python\n<code>" leaks into the comparison.
    s = re.sub(r"^```(?:python|py|cpp|c\+\+|js|ts|sql|rust|go|java|bash|sh)?\s*\n",
               "", s, count=1)
    s = re.sub(r"\n```\s*$", "", s, count=1)
    s = re.sub(r"^\s*(?:python|py|cpp|c\+\+|js|ts|sql|rust|go|java|bash|sh)\s*\n",
               "", s, count=1)
    # Strip "answer:" / "the answer is" preambles
    s = re.sub(r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:=]?\s*", "", s)
    # Strip quotes & wrapping markdown stars
    s = s.strip("\"'`*_ ")
    # Strip articles
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    # Trailing punctuation
    s = s.rstrip(".,;:!?")
    return s.strip()


def grade_short_answer(prediction: str, gold: str) -> float:
    """Grade BrowseComp / HLE / similar short-text answers.

    Two candidate tiers with different rules, EXPLICIT tried first:
      EXPLICIT (boxed + "Final answer: X"): exact / word-substring / numeric.
      FALLBACK (last paragraph, whole text): exact / numeric ONLY — no
        substring, else "not Newton, but Einstein" scores +1 for gold=Newton.
    """
    if not prediction or not gold:
        return 0.0
    gold_n = _normalize_short_answer(gold)
    if not gold_n:
        return 0.0

    explicit: list[str] = []
    boxed = _extract_last_boxed(prediction)
    if boxed:
        explicit.append(boxed)
    m = re.search(
        r"(?:final\s*answer|the\s*answer\s*(?:is)?|my\s*answer)\s*[:=]\s*"
        r"(.+?)(?:\n|$)",
        prediction, re.IGNORECASE,
    )
    if m:
        explicit.append(m.group(1).strip())

    fallback: list[str] = []
    last_para = prediction.strip().split("\n\n")[-1].strip()
    if last_para:
        fallback.append(last_para)
    fallback.append(prediction)  # whole prediction, very last resort

    gn = _extract_first_number(gold_n)

    def _numeric_match(cand_n: str) -> bool:
        if gn is None:
            return False
        pn = _extract_first_number(cand_n)
        if pn is None:
            return False
        try:
            return abs(float(pn) - float(gn)) < 1e-4
        except ValueError:
            return False

    for cand in explicit:
        cand_n = _normalize_short_answer(cand)
        if cand_n == gold_n:
            return 1.0
        if re.search(rf"\b{re.escape(gold_n)}\b", cand_n):
            return 1.0
        if _numeric_match(cand_n):
            return 1.0

    for cand in fallback:
        cand_n = _normalize_short_answer(cand)
        if cand_n == gold_n:
            return 1.0
        if _numeric_match(cand_n):
            return 1.0
        # NB: NO substring match here — too prone to false positives in
        # negation / contrast / discussion contexts.
    return 0.0


# ---------------------------------------------------------------------------
# LiveCodeBench grading (exec-based, against public test cases)
# ---------------------------------------------------------------------------

def _grade_functional(code: str, test_cases: list) -> float:
    """LeetCode-style functional grading: call Solution().<func>(*args).

    Each functional case carries `metadata.func_name`, an `input` of
    newline-separated literal args (ast.literal_eval each line), and an
    `output` literal to compare the return value against. All-or-nothing
    + short-circuit on first failure, mirroring the stdin path.
    """
    import json as _json
    # func_name: from any case's metadata.
    func_name = None
    for tc in test_cases:
        md = tc.get("metadata") if isinstance(tc, dict) else None
        if isinstance(md, str):
            try:
                md = _json.loads(md)
            except (_json.JSONDecodeError, ValueError):
                md = None
        if isinstance(md, dict) and md.get("func_name"):
            func_name = md["func_name"]
            break
    if not func_name:
        return 0.0

    fcases = [tc for tc in test_cases
              if isinstance(tc, dict) and tc.get("testtype") == "functional"
              and tc.get("output", "") != ""]
    if not fcases:
        return 0.0

    # Driver reads the JSON test list from stdin, calls the method, and
    # prints ALL_PASS / FAIL. Args + expected are parsed with
    # ast.literal_eval (LeetCode inputs are Python literals).
    driver = (
        code + "\n\n"
        "import sys as _sys, json as _json, ast as _ast\n"
        "def _p(s):\n"  # LCB args/outputs are JSON (true/false/null); fall
        "    try: return _json.loads(s)\n"  # back to Python literals if not.
        "    except Exception: return _ast.literal_eval(s)\n"
        "def _norm(v):\n"
        "    return _json.dumps(v, sort_keys=True) if isinstance(v,(list,dict)) else v\n"
        "_tests = _json.loads(_sys.stdin.read())\n"
        "_sol = Solution()\n"
        "for _tc in _tests:\n"
        "    _lines = [l for l in _tc['input'].split('\\n') if l.strip() != '']\n"
        "    try:\n"
        "        _args = [_p(l) for l in _lines]\n"
        "        _res = _sol." + func_name + "(*_args)\n"
        "        _exp = _p(_tc['output'])\n"
        "    except Exception:\n"
        "        print('FAIL'); _sys.exit(0)\n"
        "    if _norm(_res) != _norm(_exp) and str(_res).strip() != str(_exp).strip():\n"
        "        print('FAIL'); _sys.exit(0)\n"
        "print('ALL_PASS')\n"
    )
    import tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                          prefix="apo_lcbfn_") as f:
            f.write(driver)
            path = f.name
        proc = subprocess.run(
            [sys.executable, path],
            input=_json.dumps(fcases), capture_output=True,
            timeout=15.0, text=True,
        )
        return 1.0 if (proc.stdout or "").strip().endswith("ALL_PASS") else 0.0
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0.0
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass


def grade_livecodebench(prediction: str, metadata: dict) -> float:
    """Run the predicted code against LiveCodeBench's test cases.

    `metadata["tests"]` may be a JSON-string list of {"input": "...",
    "output": "...", "testtype": "stdin"|"functional"} or a raw
    assertion block. stdin cases pipe input→stdout; functional cases
    call Solution().<func_name>(*args).
    """
    if not metadata:
        return 0.0
    tests = metadata.get("tests", "")
    if not tests:
        return 0.0
    code = _extract_python_code(prediction)
    if not code:
        return 0.0

    # Parse tests: try JSON list of {"input", "output"} first, fall back
    # to treating tests as a raw assertion block.
    import json as _json
    test_cases = None
    if isinstance(tests, str):
        try:
            tlist = _json.loads(tests)
            if isinstance(tlist, list):
                test_cases = tlist
        except (_json.JSONDecodeError, ValueError):
            test_cases = None
    elif isinstance(tests, list):
        test_cases = tests

    if test_cases is None:
        # treat as raw assertion block (humaneval-style)
        full = code + "\n\n" + str(tests)
        passed, _ = _exec_with_timeout(full, timeout=10.0)
        return 1.0 if passed else 0.0

    # LiveCodeBench has TWO test types. ~37% of v6 rows are `functional`
    # (LeetCode-style: call Solution().<func>(*args), compare return) —
    # these have NO stdin, so the stdin harness below would feed them
    # nothing → no output → always 0 (silently flooring 37% of the code
    # axis). Route functional cases through a call-the-method harness.
    if any(isinstance(tc, dict) and tc.get("testtype") == "functional"
           for tc in test_cases):
        return _grade_functional(code, test_cases)

    # Write the user code ONCE to a tempfile, then pipe each test case's
    # stdin through a fresh subprocess. All test cases run (no cap, else
    # later-failing cases score false positives); tempfile freed in finally.
    n_pass, n_total = 0, 0
    import tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                          prefix="apo_lcb_") as f:
            f.write(code)
            path = f.name
        for tc in test_cases:
            inp = tc.get("input", "") if isinstance(tc, dict) else ""
            exp = (tc.get("output", "") if isinstance(tc, dict) else "").strip()
            if not exp:
                continue
            n_total += 1
            try:
                proc = subprocess.run(
                    [sys.executable, path],
                    input=inp, capture_output=True, timeout=8.0, text=True,
                )
                actual = (proc.stdout or "").strip()
                if actual == exp or actual.replace(" ", "") == exp.replace(" ", ""):
                    n_pass += 1
                else:
                    # Short-circuit: once one test fails the whole task is
                    # already wrong under all-or-nothing grading; skip the
                    # rest of the (potentially many) test cases for speed.
                    break
            except subprocess.TimeoutExpired:
                # Timed out → counts as failure; short-circuit too.
                break
            except (OSError, ValueError):
                # Subprocess startup or arg error → record as failure and
                # bail; rerunning won't help.
                break
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
    if n_total == 0:
        return 0.0
    return 1.0 if n_pass == n_total else 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade(prediction: str, sample) -> float:
    """Dispatch to the correct grader based on `sample.family`."""
    family = sample.family
    if family in ("gsm8k", "synthetic_arith", "synthetic_wordprob", "synthetic_logic"):
        return grade_numeric(prediction, sample.gold_answer)
    if family in ("math", "olympiad", "phybench", "aime", "omni_math"):
        # All have LaTeX/symbolic/numeric answers; grade_math handles
        # boxed extraction + sympy equivalence + numeric fallback.
        return grade_math(prediction, sample.gold_answer)
    if family in ("humaneval", "mbpp"):
        return grade_humaneval(prediction, sample.metadata)
    if family == "livecodebench":
        return grade_livecodebench(prediction, sample.metadata)
    # Multiple-choice families.
    if (family in ("mmlu", "mmlu_pro", "gpqa_diamond", "musr", "reclor")
            or family.startswith("arc_")):
        return grade_multiple_choice(prediction, sample.gold_answer)
    # Short free-form answer families (BrowseComp / HLE / bbh some).
    if family in ("browsecomp", "hle") or family.startswith("bbh_"):
        return grade_short_answer(prediction, sample.gold_answer)
    # Fallback to numeric/string match.
    return grade_numeric(prediction, sample.gold_answer)


# ---------------------------------------------------------------------------
# Reward computation (consumed by GRPO's shaped_advantage)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    """Reward + metadata consumed by GRPO's shaped_advantage."""
    correctness: float    # {0.0, 1.0} — primary signal
    n_active: int         # active agent slots (metadata)
    n_edges: int          # directed edges (metadata)
    n_calls: int          # LLM calls (ReAct + Synth) — used by shaped_advantage
    total: float          # == correctness; cost is shaped in advantage


def grade_answer(prediction: str, gold: str) -> float:
    """Back-compat numeric/string grader. Prefer `grade()` for full dispatch."""
    return grade_numeric(prediction, gold)


def compute_reward(
    trace,                          # ExecutionTrace (avoid circular import)
    gold_answer: str,
    train_spec=None,                # accepted for backward compat
    task_sample=None,
) -> RewardBreakdown:
    """Reward = correctness only. Cost (n_calls) is reported as metadata
    and consumed by `training.grpo.shaped_advantage`, not here.
    """
    del train_spec
    if task_sample is not None:
        correct = grade(trace.final_answer, task_sample)
    else:
        correct = grade_numeric(trace.final_answer, gold_answer)
    return RewardBreakdown(
        correctness=float(correct),
        n_active=int(trace.arch.active_mask.sum().item()),
        n_edges=int(trace.arch.edges.sum().item()),
        n_calls=trace.n_llm_calls,
        total=float(correct),
    )


__all__ = [
    "RewardBreakdown",
    "compute_reward",
    "grade",
    "grade_answer",
    "grade_humaneval",
    "grade_livecodebench",
    "grade_math",
    "grade_multiple_choice",
    "grade_numeric",
    "grade_short_answer",
]
