"""Tool implementations available to agents.

Each tool is `Callable[[str], str]` taking a single arg-string and returning
a (truncated) result string. Tools are registered in `TOOLS` and dispatched
via `call_tool(name, args, allowed=...)`.

Tool set (5 tools):

  COMPUTE (local, free)
    python_exec      — Python subprocess; agents do symbolic math (sympy),
                       unit checks (pint), numerics (numpy/scipy) from here.
    pytest_runner    — write + run pytest tests in a tempdir.

  EXTERNAL INFO (Serper API, ~$0.001/req — except wikipedia_search)
    web_search       — Google web search.
    arxiv_search     — academic papers via Google Scholar.
    wikipedia_search — official Wikipedia REST API (no key, no Serper);
                       returns top-3 page summaries. Falls back to Serper
                       /search with `site:en.wikipedia.org` on network
                       failure (e.g. wikipedia.org blocked by firewall).

Per-role tool whitelists are enforced by `call_tool`'s `allowed` arg; the
role → tools mapping lives in `role_tools.py`. Output is capped at
`MAX_OUTPUT_CHARS` to keep transcripts manageable.
"""

from __future__ import annotations

import json
import os
import re as _re
import subprocess
import sys
import threading as _threading
import time as _time
import urllib.parse
import urllib.request
import urllib.error
from typing import Callable

# Generous caps: tool calls are meant to prevent runaway loops / hangs,
# not to push models toward shorter outputs.
MAX_OUTPUT_CHARS = 4000
# Tool timeouts. Data-driven values — `python_exec_log` on a 128-trace
# smoke (May 2026) showed:
#   78% of legitimate python_exec calls finish in <1s
#   92% finish in <30s
#   the rest are chess-engine / brute-force scripts that intrinsically
#   take >180s (LiveCodeBench hard), where TIMEOUT is the correct outcome.
# Cutting from 180s to 30s costs only ~2% extra successful runs while
# freeing the trace from waiting on doomed code. The eng-valid mask treats
# tool TIMEOUT as architecture-attributable (chose a slow path), so the
# reward signal pushes the head toward cheaper variants.
# Subprocess wall for python_exec. Default 90s covers the bulk of math
# (sympy / scipy / pint). Per-category override via env var
# ARCH_PYTHON_TIMEOUT_S (e.g. cat_math sets 120 in run_cat_rl.sh) so a slow
# symbolic category gets headroom without lengthening code's worst-case wall.
# Bona-fide infinite loops are still killed and surfaced as
# `[python_exec] TIMEOUT ...`, which the agent sees in the next ReAct step
# and can react to (change algorithm / smaller input / give up).
PYTHON_TIMEOUT_S = float(os.environ.get("ARCH_PYTHON_TIMEOUT_S", "90.0"))
PYTEST_TIMEOUT_S = 60.0
SERPER_BASE = "https://google.serper.dev"
SERPER_TIMEOUT_S = 60.0

# Wikipedia direct-API endpoint. No API key required, but Wikipedia's
# policy requires a descriptive User-Agent (else they may return 403).
WIKI_BASE = "https://en.wikipedia.org"
WIKI_USER_AGENT = "arch_policy/1.0 (research; https://github.com/arch-policy/repo)"
WIKI_TIMEOUT_S = 15.0

# Canonical list of search-style tools whose offline-stub returns are
# bookkept in `_stub_counts` / `trace.search_stub_counts`. Keep
# `trace.py`'s `search_stub_counts` default keys in sync with this tuple.
SEARCH_TOOL_NAMES: tuple[str, ...] = (
    "web_search", "arxiv_search", "wikipedia_search",
)


# Set ARCH_POLICY_STRICT_TOOLS=1 to RAISE instead of returning offline-stub
# strings when SERPER_API_KEY is missing or the request fails. Training
# scripts call `preflight_tools()` at startup so this is the loud path;
# unit tests / dev shells get the lenient path. Without strict mode, the
# wrapper still emits a ONE-TIME stderr warning per missing key + bumps
# `SEARCH_STUB_COUNTS` so trace telemetry surfaces silent degradation.
STRICT_TOOLS = os.environ.get("ARCH_POLICY_STRICT_TOOLS", "").lower() in {"1", "true", "yes"}

# Per-call telemetry: number of times each tool returned the offline stub
# instead of a real Serper response. Read by `multi_agent.py` into the
# per-trace ExecutionTrace so analyzers (and `05_analyze_grpo.py`) flag
# runs that silently degraded.
#
# threading.local so each worker thread (GRPO runs B*G traces in parallel
# via ThreadPoolExecutor) accumulates ONLY its own trace's stubs.
_STUB_TLS = _threading.local()


def _stub_counts() -> dict[str, int]:
    if not hasattr(_STUB_TLS, "counts"):
        _STUB_TLS.counts = {name: 0 for name in SEARCH_TOOL_NAMES}
    return _STUB_TLS.counts

# Print the "key missing" warning at most ONCE per process to avoid
# spamming logs (a 1000-task run would otherwise print millions of lines).
_KEY_WARN_PRINTED = False


def _warn_serper_key_missing() -> None:
    """Print a one-time stderr warning if SERPER_API_KEY is not set."""
    global _KEY_WARN_PRINTED
    if _KEY_WARN_PRINTED:
        return
    _KEY_WARN_PRINTED = True
    print(
        "[tools] WARNING: SERPER_API_KEY not set. "
        "web_search / arxiv_search (and Wikipedia's Serper fallback) "
        "will return offline-stub strings. Set "
        "ARCH_POLICY_STRICT_TOOLS=1 to RAISE instead. "
        "Get a key at https://serper.dev .",
        file=sys.stderr, flush=True,
    )


def preflight_tools() -> None:
    """Loud preflight check called at the top of training / eval scripts.

    Asserts that the search tools have a working API key. If
    SERPER_API_KEY is missing this raises so the user notices BEFORE
    paying for hours of GRPO traces that silently used offline stubs.
    (wikipedia_search has its own no-key direct path but ALSO falls back
    to Serper on network failure, so the Serper key check still matters.)
    """
    if not os.environ.get("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY env var is not set. web_search / arxiv_search "
            "would silently return offline-stub strings, which would "
            "corrupt the reward signal for any Researcher / search-using "
            "architecture. Either:\n"
            "  (a) export SERPER_API_KEY=... (get from https://serper.dev), or\n"
            "  (b) export ARCH_POLICY_STRICT_TOOLS=0 to acknowledge the risk "
            "explicitly (a stub-degraded run will still print a one-time "
            "warning to stderr)."
        )


def reset_search_stub_counts() -> dict[str, int]:
    """Return THIS THREAD's snapshot + reset. Called by `multi_agent.run()`
    at trace start; the corresponding `snapshot_search_stub_counts()` is
    called at trace end. Thread-local so concurrent traces never mix."""
    snap = dict(_stub_counts())
    for k in _stub_counts():
        _stub_counts()[k] = 0
    return snap


def snapshot_search_stub_counts() -> dict[str, int]:
    """Return THIS THREAD's current counts without resetting."""
    return dict(_stub_counts())


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    return s[: MAX_OUTPUT_CHARS - 32] + "\n... [truncated]"


def _retry_urlopen_read(
    req: urllib.request.Request,
    timeout: float,
    *,
    retries: int = 1,
    backoff_s: float = 0.3,
) -> bytes:
    """`urlopen(req).read()` with `retries` automatic retries on transient
    errors. SSL handshake EOF, "Connection reset by peer", DNS hiccup —
    all transient enough that one extra attempt at +300ms costs little
    and saves a lot of false-positive failures. 4xx HTTPErrors are NOT
    retried (client-side, no point); 5xx + URLError + OSError +
    TimeoutError ARE retried.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise
            last_exc = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
        if attempt < retries:
            # N29 fix: per-thread jitter so 32 concurrent traces hitting
            # the same upstream 5xx don't retry in lockstep (mirrors the
            # worker-layer N26 fix). Jitter ∈ [0.5, 1.5) keeps the mean
            # backoff intact while spreading the retry window.
            jitter = 0.5 + (hash((_threading.get_ident(),
                                   _time.time_ns(), attempt)) % 1000) / 1000.0
            _time.sleep(backoff_s * (2 ** attempt) * jitter)
    assert last_exc is not None
    raise last_exc


def _strip_markdown_fence(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


# ===========================================================================
# COMPUTE TOOLS
# ===========================================================================

def python_exec(code: str) -> str:
    """Run `code` in a fresh `python` subprocess. Capture stdout+stderr.

    Timeout = `PYTHON_TIMEOUT_S`. No filesystem isolation — relies on the
    timeout to prevent pathological loops. For production-grade use wrap in
    a proper sandbox (e.g. E2B, firejail).
    """
    code = _strip_markdown_fence(code)
    if not code:
        return "[python_exec] no code provided"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            stdin=subprocess.DEVNULL,  # else stdin-reading code (e.g. LCB
            # competition solutions doing sys.stdin.read()) BLOCKS on the
            # unfed parent stdin until the timeout. DEVNULL → instant EOF,
            # so the agent gets fast feedback instead of a 90s hang.
            timeout=PYTHON_TIMEOUT_S,
            text=True,
        )
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired as e:
        # subprocess captured stdout up to the kill — surface the tail
        # (last 1000 chars) so the model can see how far its code got
        # without re-running. Purely factual; no prescriptive hint.
        partial = (getattr(e, "stdout", "") or "")[-1000:]
        head = f"[python_exec] TIMEOUT after {PYTHON_TIMEOUT_S}s (code too slow)"
        if partial.strip():
            return _truncate(f"{head}\nPARTIAL STDOUT:\n{partial.rstrip()}")
        return head
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


def pytest_runner(spec: str) -> str:
    """Run pytest against a code + tests bundle.

    Preferred input format:

        <module code under test>
        ---TESTS---
        <pytest test code>

    Forgiving fallbacks:
      - markdown fences inside either section are stripped
      - `---TESTS---` missing → split on first `def test_`
      - empty module section → run tests standalone (no import)
    """
    spec = _strip_markdown_fence(spec)
    if "---TESTS---" in spec:
        code_part, test_part = spec.split("---TESTS---", 1)
    else:
        # Heuristic split: first `def test_` line starts the test block.
        m = _re.search(r"(?m)^def\s+test_", spec)
        if m:
            code_part, test_part = spec[:m.start()], spec[m.start():]
        else:
            # No tests at all → cannot do anything useful.
            return ("[pytest_runner] no tests found. Include at least one "
                    "`def test_*` function, or use the `---TESTS---` "
                    "delimiter to mark the test section explicitly.")

    code_part = _strip_markdown_fence(code_part).strip()
    test_part = _strip_markdown_fence(test_part).strip()
    if not test_part:
        return "[pytest_runner] no test code provided"

    import tempfile
    with tempfile.TemporaryDirectory(prefix="apo_pytest_") as td:
        test_path = os.path.join(td, "test_mod.py")
        if code_part:
            mod_path = os.path.join(td, "mod.py")
            with open(mod_path, "w") as f:
                f.write(code_part)
            preamble = (
                "import sys, os\n"
                "sys.path.insert(0, os.path.dirname(__file__))\n"
                "from mod import *  # noqa: F401, F403\n\n"
            )
        else:
            # Empty module → tests run standalone (no import).
            preamble = ""
        with open(test_path, "w") as f:
            f.write(preamble + test_part)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=line", "-q",
                 test_path, "--rootdir", td],
                capture_output=True, stdin=subprocess.DEVNULL,
                timeout=PYTEST_TIMEOUT_S, text=True,
            )
        except subprocess.TimeoutExpired as e:
            partial = (getattr(e, "stdout", "") or "")[-1000:]
            head = f"[pytest_runner] TIMEOUT after {PYTEST_TIMEOUT_S}s (tests too slow)"
            if partial.strip():
                return _truncate(f"{head}\nPARTIAL STDOUT:\n{partial.rstrip()}")
            return head
        except FileNotFoundError:
            return "[pytest_runner] ERROR pytest not installed"
        out = (proc.stdout + "\n" + proc.stderr).strip()
        return _truncate(f"[pytest_runner] exit={proc.returncode}\n{out}")


# ===========================================================================
# EXTERNAL INFO TOOLS (Serper API)
# ===========================================================================

def _serper_post(endpoint: str, payload: dict, *, tool_name: str) -> dict | str:
    """POST helper. Returns parsed JSON dict on success, an `[error: …]`
    string on failure.

    Reads SERPER_API_KEY from env. Unset →
      - STRICT_TOOLS: raise RuntimeError (loud)
      - otherwise: print one-time WARN to stderr + bump SEARCH_STUB_COUNTS
        and return a `[<tool>: api key missing — offline stub]` string so
        the agent sees the failure mode in its scratchpad (vs the old
        "no results for QUERY" string which looked like a real-but-empty
        search hit).

    Network / HTTP errors:
      - STRICT_TOOLS: raise the underlying exception
      - otherwise: return `[<tool>: <error class>: <msg>]` so the agent
        can react and trace telemetry captures the failure kind.
    """
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        if STRICT_TOOLS:
            raise RuntimeError(
                f"{tool_name}: SERPER_API_KEY env var not set "
                "(ARCH_POLICY_STRICT_TOOLS=1 forbids offline-stub fallback)."
            )
        _warn_serper_key_missing()
        c = _stub_counts()
        c[tool_name] = c.get(tool_name, 0) + 1
        return f"[{tool_name}: api key missing — offline stub]"
    url = f"{SERPER_BASE}/{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        # retries=3 + jittered backoff. Worst-case wait ≈ 2.1s, well
        # under the trace wall-clock.
        raw = _retry_urlopen_read(req, SERPER_TIMEOUT_S, retries=3)
        return json.loads(raw.decode())
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError, json.JSONDecodeError) as e:
        if STRICT_TOOLS:
            raise
        # Failed call → stub (same bucket as no-key) so the agent sees
        # the failure and trace telemetry captures the kind.
        c = _stub_counts()
        c[tool_name] = c.get(tool_name, 0) + 1
        return f"[{tool_name}: {type(e).__name__}: {str(e)[:80]}]"


def web_search(query: str) -> str:
    """Google web search via Serper /search. Top-5 organic results."""
    query = query.strip()
    if not query:
        return "[web_search] empty query"
    data = _serper_post("search", {"q": query, "num": 5, "gl": "us", "hl": "en"},
                       tool_name="web_search")
    if isinstance(data, str):
        return data  # already an `[error: …]` / `[stub]` string
    organic = data.get("organic", [])[:5]
    if not organic:
        return f"[web_search] no results for {query!r}"
    lines = []
    for i, r in enumerate(organic, 1):
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        link = r.get("link", "").strip()
        lines.append(f"{i}. {title}\n   {snippet}\n   {link}")
    return _truncate(f"[web_search] {query!r}\n" + "\n".join(lines))


def arxiv_search(query: str) -> str:
    """Academic literature search via Serper /scholar. Top-5 papers."""
    query = query.strip()
    if not query:
        return "[arxiv_search] empty query"
    data = _serper_post("scholar", {"q": query, "num": 5}, tool_name="arxiv_search")
    if isinstance(data, str):
        return data
    organic = data.get("organic", [])[:5]
    if not organic:
        return f"[arxiv_search] no results for {query!r}"
    lines = []
    for i, r in enumerate(organic, 1):
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()[:200]
        year = r.get("year", "")
        cited = r.get("citedBy", 0)
        pdf = r.get("pdfUrl") or r.get("link", "")
        lines.append(f"{i}. ({year}, cited {cited}) {title}\n   {snippet}\n   {pdf}")
    return _truncate(f"[arxiv_search] {query!r}\n" + "\n".join(lines))


def wikipedia_search(query: str) -> str:
    """Search English Wikipedia and return top-3 page summaries.

    Two-stage call against Wikipedia's official APIs (no API key):
      1. `/w/api.php?action=opensearch` → matching titles for the query.
      2. `/api/rest_v1/page/summary/{title}` → 1-paragraph extract + URL
         for each of the top-3 titles.

    On any direct-Wikipedia failure (network, firewall, 403/5xx, empty
    summaries) falls back to Serper `/search` with `site:en.wikipedia.org`
    so the agent still gets snippets. The fallback uses the Serper key
    and counts toward `search_stub_counts['wikipedia_search']` on its
    own failure.

    Why a dedicated tool: Wikipedia is a high-leverage source for HLE
    Humanities / Bio Easy / GPQA edges, and the direct REST path is
    ~10× more reliable than Serper /search AND free.
    """
    query = query.strip()
    if not query:
        return "[wikipedia_search] empty query"

    # ---- Stage 1: opensearch ----
    try:
        qs = urllib.parse.urlencode({
            "action": "opensearch", "format": "json",
            "search": query, "limit": 3,
        })
        req = urllib.request.Request(
            f"{WIKI_BASE}/w/api.php?{qs}",
            headers={"User-Agent": WIKI_USER_AGENT},
        )
        raw = _retry_urlopen_read(req, WIKI_TIMEOUT_S, retries=2)
        data = json.loads(raw.decode())
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError, json.JSONDecodeError) as e:
        return _wikipedia_serper_fallback(
            query, reason=f"opensearch:{type(e).__name__}: {str(e)[:60]}"
        )

    if not titles:
        # opensearch is title-prefix matching → misses on hyphen / typo
        # / abbrev. Retry via full-text srsearch before giving up.
        try:
            qs = urllib.parse.urlencode({
                "action": "query", "format": "json",
                "list": "search", "srsearch": query,
                "srlimit": 3, "srprop": "",
            })
            req = urllib.request.Request(
                f"{WIKI_BASE}/w/api.php?{qs}",
                headers={"User-Agent": WIKI_USER_AGENT},
            )
            raw = _retry_urlopen_read(req, WIKI_TIMEOUT_S, retries=2)
            data = json.loads(raw.decode())
            # srsearch returns a dict; defensive isinstance for upstream
            # shape drift / test mocks that returned the opensearch list.
            hits = []
            if isinstance(data, dict):
                hits = (data.get("query") or {}).get("search") or []
            titles = [h["title"] for h in hits if h.get("title")]
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError, json.JSONDecodeError):
            titles = []
    if not titles:
        return f"[wikipedia_search] no results for {query!r}"

    # ---- Stage 2: per-title summary ----
    results: list[str] = []
    for title in titles[:3]:
        slug = urllib.parse.quote(title.replace(" ", "_"))
        try:
            req = urllib.request.Request(
                f"{WIKI_BASE}/api/rest_v1/page/summary/{slug}",
                headers={"User-Agent": WIKI_USER_AGENT},
            )
            raw = _retry_urlopen_read(req, WIKI_TIMEOUT_S, retries=2)
            page = json.loads(raw.decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError, json.JSONDecodeError):
            continue
        extract = (page.get("extract") or "").strip()
        link = (
            page.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
        )
        if not extract:
            continue
        results.append(f"## {title}\n{extract}\n{link}")

    if not results:
        return _wikipedia_serper_fallback(query, reason="no_summaries")
    return _truncate(
        f"[wikipedia_search] {query!r}\n\n" + "\n\n".join(results)
    )


def _wikipedia_serper_fallback(query: str, *, reason: str) -> str:
    """Serper `/search` with `site:en.wikipedia.org` when direct
    Wikipedia API failed. Routes through `_serper_post` so all the
    existing stub-counting / STRICT_TOOLS behavior applies — counted
    under `wikipedia_search`."""
    if STRICT_TOOLS:
        raise RuntimeError(
            f"wikipedia_search direct API failed ({reason}); "
            "STRICT_TOOLS=1 forbids the silent Serper fallback."
        )
    data = _serper_post(
        "search",
        {"q": f"site:en.wikipedia.org {query}",
         "num": 5, "gl": "us", "hl": "en"},
        tool_name="wikipedia_search",
    )
    if isinstance(data, str):
        return data
    organic = data.get("organic", [])[:5]
    if not organic:
        return (
            f"[wikipedia_search] direct API failed ({reason}); "
            f"Serper fallback also found no results for {query!r}"
        )
    lines = []
    for i, r in enumerate(organic, 1):
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        link = r.get("link", "").strip()
        lines.append(f"{i}. {title}\n   {snippet}\n   {link}")
    return _truncate(
        f"[wikipedia_search] (Serper fallback, direct failed: {reason}) "
        f"{query!r}\n" + "\n".join(lines)
    )


# ===========================================================================
# Registry + dispatcher
# ===========================================================================

ToolFn = Callable[[str], str]

TOOLS: dict[str, ToolFn] = {
    # compute
    "python_exec":      python_exec,
    "pytest_runner":    pytest_runner,
    # external info
    "web_search":       web_search,
    "arxiv_search":     arxiv_search,
    "wikipedia_search": wikipedia_search,
}


def call_tool(name: str, args: str, allowed: set[str] | None = None) -> str:
    """Dispatch a tool call.

    `allowed`: optional whitelist of tool names this caller can use. If the
    requested tool isn't in `allowed`, return a refusal string (the agent
    can retry with a different tool). Pass None for unrestricted access.
    """
    if name not in TOOLS:
        return f"[unknown tool {name!r}]; available: {sorted(TOOLS)}"
    if allowed is not None and name not in allowed:
        return (f"[tool {name!r} not available for this role]; "
                f"you may use: {sorted(allowed)}")
    return TOOLS[name](args)


__all__ = [
    "TOOLS",
    "SEARCH_TOOL_NAMES",
    "call_tool",
    "preflight_tools",
    "reset_search_stub_counts",
    "snapshot_search_stub_counts",
    # compute
    "python_exec", "pytest_runner",
    # external info
    "web_search", "arxiv_search", "wikipedia_search",
]
