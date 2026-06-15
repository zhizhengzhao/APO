"""Regression tests for the tool pool: wikipedia_search + role
whitelists + search-stub telemetry consistency.

`browse_url` was retired (May 2026) after empirical probes showed a
40-46% failure rate against the kinds of URLs HLE Researcher agents
naturally produce; the slot is now covered by web_search snippets +
wikipedia_search direct summaries. These tests pin the *remaining*
tools so future refactors can't silently break parsing, fallback
routing, or telemetry plumbing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest


class _MockResp:
    """Mimic the context-manager returned by `urllib.request.urlopen`."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


@contextmanager
def patch_urlopen(monkeypatch, fake):
    """Patch urllib.request.urlopen with `fake(request, timeout=...)`."""
    monkeypatch.setattr("urllib.request.urlopen", fake)
    yield


# ---------------------------------------------------------------------------
# SEARCH_TOOL_NAMES + trace defaults consistency
# ---------------------------------------------------------------------------

def test_search_tool_names_is_the_canonical_list():
    """The list of search-style tools that bump stub counters must
    exactly match what `trace.search_stub_counts` defaults to. If you
    add a tool that routes through `_serper_post` you MUST update both
    places — this test catches the drift."""
    from arch_policy.executor.tools import SEARCH_TOOL_NAMES
    from arch_policy.executor.trace import ExecutionTrace
    from arch_policy.architecture.sampler import ConcreteArch
    import torch

    expected = {"web_search", "arxiv_search", "wikipedia_search"}
    assert set(SEARCH_TOOL_NAMES) == expected, (
        f"SEARCH_TOOL_NAMES drift: {set(SEARCH_TOOL_NAMES)} != {expected}. "
        "Add new tool names here AND in trace.search_stub_counts default."
    )

    arch = ConcreteArch(
        active_mask=torch.zeros(6, dtype=torch.bool),
        roles=torch.zeros(6, dtype=torch.long),
        edges=torch.zeros(6, 6, dtype=torch.bool),
        sequence=torch.zeros(0, dtype=torch.long),
    )
    trace = ExecutionTrace(task="x", arch=arch)
    assert set(trace.search_stub_counts.keys()) == expected, (
        f"trace.search_stub_counts default keys {set(trace.search_stub_counts.keys())} "
        f"!= {expected}. Update trace.py default_factory."
    )


def test_stub_counts_dict_initialized_for_all_search_tools():
    """`_stub_counts()` lazy-init must include every name in
    SEARCH_TOOL_NAMES (otherwise the first stub on a fresh thread
    silently creates a new key via .get/.assign which violates the
    "default-zero for every search tool" contract analyzers rely on)."""
    from arch_policy.executor import tools as t

    if hasattr(t._STUB_TLS, "counts"):
        delattr(t._STUB_TLS, "counts")

    counts = t._stub_counts()
    assert set(counts.keys()) == set(t.SEARCH_TOOL_NAMES), counts


def test_tools_registry_has_expected_five_tools_and_no_browse_url():
    """Explicit pin: `browse_url` is retired. Re-adding it requires
    deleting this assert deliberately so we don't regress."""
    from arch_policy.executor.tools import TOOLS

    assert set(TOOLS.keys()) == {
        "python_exec", "pytest_runner",
        "web_search", "arxiv_search", "wikipedia_search",
    }, TOOLS.keys()
    assert "browse_url" not in TOOLS, (
        "browse_url was retired after a 40-46% empirical failure rate; "
        "remove this assert if you deliberately want it back."
    )
    assert "pdf_reader" not in TOOLS


# ---------------------------------------------------------------------------
# Role × tool whitelist updates
# ---------------------------------------------------------------------------

def test_role_tools_researcher_has_wikipedia_and_search():
    from arch_policy.executor.role_tools import allowed_tools_for

    rs = allowed_tools_for("Researcher")
    assert rs == frozenset({"web_search", "arxiv_search", "wikipedia_search"}), rs
    assert "browse_url" not in rs
    assert "pdf_reader" not in rs


def test_role_tools_expert_has_all_tools():
    from arch_policy.executor.role_tools import allowed_tools_for
    from arch_policy.executor.tools import TOOLS

    expert = allowed_tools_for("Expert")
    assert expert == frozenset(TOOLS.keys()), (
        f"Expert whitelist {expert} should equal the full TOOLS keys "
        f"{set(TOOLS.keys())} (single-agent generalist baseline)."
    )


def test_role_tools_compute_only_roles_get_python_exec_only():
    """Solver / Critic / Verifier get python_exec only — for math
    verification, claim checking, re-derivation. No search, no tests."""
    from arch_policy.executor.role_tools import allowed_tools_for

    for role in ("Solver", "Critic", "Verifier"):
        assert allowed_tools_for(role) == frozenset({"python_exec"}), role


def test_role_tools_planner_refiner_have_no_tools():
    """Per the minimum-tool principle: Planner outputs a plan, Refiner
    integrates candidates — both pure text shaping, no compute needed."""
    from arch_policy.executor.role_tools import allowed_tools_for

    for role in ("Planner", "Refiner"):
        assert allowed_tools_for(role) == frozenset(), role


# ---------------------------------------------------------------------------
# wikipedia_search: direct two-stage success
# ---------------------------------------------------------------------------

def test_wikipedia_search_two_stage_success(monkeypatch):
    """Happy path: opensearch returns titles, summary endpoint returns
    extract+URL for each. Result is the formatted joined string."""
    from arch_policy.executor import tools as t

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", "") or str(req)
        if "opensearch" in url:
            return _MockResp(json.dumps([
                "einstein",
                ["Albert Einstein", "Mileva Marić"],
                ["", ""],
                ["https://en.wikipedia.org/wiki/Albert_Einstein",
                 "https://en.wikipedia.org/wiki/Mileva_Mari%C4%87"],
            ]).encode())
        if "summary/Albert_Einstein" in url:
            return _MockResp(json.dumps({
                "extract": "German-born theoretical physicist.",
                "content_urls": {"desktop": {
                    "page": "https://en.wikipedia.org/wiki/Albert_Einstein"
                }},
            }).encode())
        if "summary/" in url:
            return _MockResp(json.dumps({
                "extract": "Serbian physicist and mathematician.",
                "content_urls": {"desktop": {
                    "page": "https://en.wikipedia.org/wiki/Mileva_Mari%C4%87"
                }},
            }).encode())
        raise AssertionError(f"unexpected URL: {url}")

    with patch_urlopen(monkeypatch, fake_urlopen):
        out = t.wikipedia_search("einstein")
    assert "Albert Einstein" in out
    assert "theoretical physicist" in out
    snap = t.snapshot_search_stub_counts()
    assert snap.get("wikipedia_search", 0) == 0, snap


def test_wikipedia_search_empty_opensearch_returns_no_results(monkeypatch):
    from arch_policy.executor import tools as t

    def fake_urlopen(req, timeout=None):
        return _MockResp(json.dumps([
            "totally_unknown", [], [], [],
        ]).encode())

    with patch_urlopen(monkeypatch, fake_urlopen):
        out = t.wikipedia_search("totally_unknown_topic_xyzzy")
    assert "no results" in out.lower()


def test_wikipedia_search_empty_query():
    from arch_policy.executor.tools import wikipedia_search

    assert "empty" in wikipedia_search("").lower()
    assert "empty" in wikipedia_search("   ").lower()


# ---------------------------------------------------------------------------
# wikipedia_search: Serper fallback path
# ---------------------------------------------------------------------------

def test_wikipedia_search_falls_back_to_serper_on_direct_network_failure(monkeypatch):
    """When direct Wikipedia opensearch raises (firewall, DNS, 5xx),
    we must transparently fall back to Serper `/search` with a
    `site:en.wikipedia.org` filter so the agent still gets snippets."""
    from arch_policy.executor import tools as t

    monkeypatch.setenv("SERPER_API_KEY", "fake-key")
    monkeypatch.setattr(t, "STRICT_TOOLS", False)

    call_log: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = getattr(req, "full_url", "") or str(req)
        call_log.append(url)
        if "wikipedia.org" in url:
            import urllib.error as _ue
            raise _ue.URLError("Network unreachable")
        if "google.serper.dev" in url:
            return _MockResp(json.dumps({
                "organic": [
                    {"title": "Albert Einstein - Wikipedia",
                     "snippet": "German-born theoretical physicist...",
                     "link": "https://en.wikipedia.org/wiki/Albert_Einstein"},
                ],
            }).encode())
        raise AssertionError(f"unexpected URL: {url}")

    with patch_urlopen(monkeypatch, fake_urlopen):
        out = t.wikipedia_search("einstein")
    assert "Serper fallback" in out, out
    assert "Albert Einstein" in out
    assert any("wikipedia.org" in u for u in call_log)
    assert any("google.serper.dev" in u for u in call_log)


def test_wikipedia_search_strict_mode_refuses_fallback(monkeypatch):
    """`ARCH_POLICY_STRICT_TOOLS=1` semantics: we'd rather hard-fail
    than silently degrade. The Serper fallback would silently degrade
    (different data source) so STRICT mode must raise instead."""
    from arch_policy.executor import tools as t

    monkeypatch.setenv("SERPER_API_KEY", "fake-key")
    monkeypatch.setattr(t, "STRICT_TOOLS", True)

    def fake_urlopen(req, timeout=None):
        import urllib.error as _ue
        raise _ue.URLError("Network unreachable")

    with patch_urlopen(monkeypatch, fake_urlopen):
        with pytest.raises(RuntimeError, match="STRICT_TOOLS"):
            t.wikipedia_search("einstein")


def test_retry_urlopen_read_retries_on_transient_error(monkeypatch):
    """`_retry_urlopen_read` must retry on URLError / OSError / 5xx, so
    a one-off SSL handshake EOF doesn't kill an otherwise healthy call.
    Pin: 1 retry by default, fast-fail on 4xx."""
    from arch_policy.executor import tools as t

    n_calls = [0]
    def fake_a(req, timeout=None):
        n_calls[0] += 1
        if n_calls[0] == 1:
            import urllib.error as _ue
            raise _ue.URLError("SSL EOF")
        return _MockResp(b'{"hello": "ok"}')

    with patch_urlopen(monkeypatch, fake_a):
        out = t._retry_urlopen_read(
            urllib.request.Request("https://example.com"),
            timeout=5.0,
        )
    assert json.loads(out.decode()) == {"hello": "ok"}
    assert n_calls[0] == 2, f"expected 1 retry (2 total calls), got {n_calls[0]}"

    n_calls_b = [0]
    def fake_b(req, timeout=None):
        n_calls_b[0] += 1
        import urllib.error as _ue
        raise _ue.HTTPError(
            url="https://example.com", code=404, msg="Not Found",
            hdrs=None, fp=None,
        )

    import urllib.request as _ur
    with patch_urlopen(monkeypatch, fake_b):
        with pytest.raises(urllib.error.HTTPError):
            t._retry_urlopen_read(
                _ur.Request("https://example.com"), timeout=5.0,
            )
    assert n_calls_b[0] == 1, (
        f"4xx must NOT be retried; got {n_calls_b[0]} calls (should be 1)."
    )


def test_wikipedia_search_serper_fallback_counts_stub_on_serper_failure(monkeypatch):
    """If direct fails AND Serper fallback also fails (key missing),
    the stub counter for wikipedia_search must bump so analyzers see
    the degraded run."""
    from arch_policy.executor import tools as t

    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setattr(t, "STRICT_TOOLS", False)
    t.reset_search_stub_counts()

    def fake_urlopen(req, timeout=None):
        import urllib.error as _ue
        raise _ue.URLError("Network unreachable")

    with patch_urlopen(monkeypatch, fake_urlopen):
        out = t.wikipedia_search("einstein")
    assert "stub" in out.lower() or "api key missing" in out.lower(), out
    snap = t.snapshot_search_stub_counts()
    assert snap["wikipedia_search"] == 1, snap
