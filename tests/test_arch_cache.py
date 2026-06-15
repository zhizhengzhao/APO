"""Regression tests for `arch_cache.ArchCache`.

Cache invariants:
  - `arch_key` is deterministic and tensor-content-driven (slot-numbering
    matters; we don't claim graph isomorphism here).
  - `put` then `get` returns the same entry.
  - JSONL persistence round-trips: write, reload, same dict.
  - `reuse_prob=1.0` always returns cached; `reuse_prob=0.0` never does.
  - Concurrent put/get from threads is safe.
  - Per-step stats reset on `pop_step_stats`.
"""

from __future__ import annotations

import json
import threading

import pytest
import torch

from arch_policy.training.arch_cache import (
    ArchCache, CachedEntry, arch_key, state_to_trace_view, trace_to_state,
)


# Build a tiny ConcreteArch-like for the hash function — duck-typed.
class _MiniArch:
    def __init__(self, active, roles, edges, sequence):
        self.active_mask = torch.tensor(active, dtype=torch.bool)
        self.roles       = torch.tensor(roles, dtype=torch.long)
        self.edges       = torch.tensor(edges, dtype=torch.bool)
        self.sequence    = torch.tensor(sequence, dtype=torch.long)


def _arch_a():
    return _MiniArch(
        active=[True, True, False, False],
        roles=[2, 4, 0, 0],
        edges=[[False, True, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]],
        sequence=[0, 1],
    )


def _arch_b():
    # Differs only in roles: 4→5
    return _MiniArch(
        active=[True, True, False, False],
        roles=[2, 5, 0, 0],
        edges=[[False, True, False, False],
               [False, False, False, False],
               [False, False, False, False],
               [False, False, False, False]],
        sequence=[0, 1],
    )


# ---------------------------------------------------------------------------
# arch_key
# ---------------------------------------------------------------------------

def test_arch_key_is_deterministic():
    a, a2 = _arch_a(), _arch_a()
    assert arch_key(a) == arch_key(a2)


def test_arch_key_changes_on_role_swap():
    assert arch_key(_arch_a()) != arch_key(_arch_b())


def test_arch_key_short_hex():
    k = arch_key(_arch_a())
    assert isinstance(k, str)
    assert len(k) == 16
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# put / get round-trip
# ---------------------------------------------------------------------------

def test_put_get_roundtrip(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=1.0)
    arch = _arch_a()
    entry = CachedEntry(
        total=0.7, correct=1, n_calls=5,
        n_active=2, n_edges=1,
        trace_state={"final_answer": "42", "n_api_errors": 0},
        stored_at_step=3,
    )
    assert c.get("task_X", arch) is None
    c.put("task_X", arch, entry)
    got = c.get("task_X", arch)
    assert got is not None
    assert got.total == 0.7
    assert got.correct == 1
    assert got.n_calls == 5
    assert got.trace_state["final_answer"] == "42"


def test_jsonl_persistence_reload(tmp_path):
    path = tmp_path / "cache.jsonl"
    c1 = ArchCache(path, reuse_prob=1.0)
    arch = _arch_a()
    c1.put("t1", arch, CachedEntry(
        total=1.0, correct=1, n_calls=8,
        n_active=2, n_edges=1, trace_state={"x": 1}, stored_at_step=0,
    ))

    # New cache instance — should hydrate from disk.
    c2 = ArchCache(path, reuse_prob=1.0)
    assert len(c2) == 1
    got = c2.get("t1", arch)
    assert got is not None
    assert got.total == 1.0
    assert got.trace_state == {"x": 1}


def test_jsonl_last_write_wins(tmp_path):
    """If the same key is put() twice, the in-memory state reflects the
    most recent value, and reload from disk picks the LAST line."""
    path = tmp_path / "cache.jsonl"
    c = ArchCache(path, reuse_prob=1.0)
    arch = _arch_a()
    c.put("t1", arch, CachedEntry(total=0.1, correct=0, n_calls=1,
                                   n_active=2, n_edges=1,
                                   trace_state={}, stored_at_step=0))
    c.put("t1", arch, CachedEntry(total=0.9, correct=1, n_calls=12,
                                   n_active=2, n_edges=1,
                                   trace_state={"v": 2}, stored_at_step=5))
    assert c.get("t1", arch).total == 0.9
    c2 = ArchCache(path, reuse_prob=1.0)
    assert c2.get("t1", arch).total == 0.9


# ---------------------------------------------------------------------------
# reuse_prob
# ---------------------------------------------------------------------------

def test_reuse_prob_0_never_hits(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=0.0, seed=0)
    arch = _arch_a()
    c.put("t", arch, CachedEntry(total=1.0, correct=1, n_calls=1,
                                  n_active=2, n_edges=1,
                                  trace_state={}, stored_at_step=0))
    for _ in range(50):
        assert c.get("t", arch) is None
    stats = c.pop_step_stats()
    assert stats["cache_hits"] == 0
    assert stats["cache_skipped_reuse"] == 50


def test_reuse_prob_1_always_hits(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=1.0)
    arch = _arch_a()
    c.put("t", arch, CachedEntry(total=1.0, correct=1, n_calls=1,
                                  n_active=2, n_edges=1,
                                  trace_state={}, stored_at_step=0))
    for _ in range(50):
        assert c.get("t", arch) is not None
    stats = c.pop_step_stats()
    assert stats["cache_hits"] == 50


def test_reuse_prob_half_is_roughly_split(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=0.5, seed=42)
    arch = _arch_a()
    c.put("t", arch, CachedEntry(total=1.0, correct=1, n_calls=1,
                                  n_active=2, n_edges=1,
                                  trace_state={}, stored_at_step=0))
    n_hit = sum(1 for _ in range(200) if c.get("t", arch) is not None)
    # Bernoulli(0.5) × 200 ~ N(100, 50). 4σ envelope.
    assert 70 <= n_hit <= 130, n_hit


def test_pop_step_stats_resets(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=1.0)
    arch = _arch_a()
    c.put("t", arch, CachedEntry(total=1.0, correct=1, n_calls=1,
                                  n_active=2, n_edges=1,
                                  trace_state={}, stored_at_step=0))
    c.get("t", arch); c.get("t", arch)
    s1 = c.pop_step_stats()
    assert s1["cache_hits"] == 2
    s2 = c.pop_step_stats()
    assert s2["cache_hits"] == 0
    assert s2["cache_misses"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_put_get_is_safe(tmp_path):
    c = ArchCache(tmp_path / "cache.jsonl", reuse_prob=1.0)
    archs = [_arch_a(), _arch_b()]
    errors = []

    def worker(tid):
        try:
            for i in range(50):
                arch = archs[(tid + i) % 2]
                key = f"task_{tid}_{i}"
                c.put(key, arch, CachedEntry(
                    total=float(i), correct=i % 2, n_calls=i,
                    n_active=2, n_edges=1,
                    trace_state={"tid": tid, "i": i}, stored_at_step=i,
                ))
                got = c.get(key, arch)
                assert got is not None
                assert got.total == float(i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors, errors
    # 8 threads × 50 unique keys = 400 entries.
    assert len(c) == 8 * 50

    # JSONL file must be parseable (no torn writes).
    with open(c.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            json.loads(line)  # should not raise


# ---------------------------------------------------------------------------
# trace_to_state / state_to_trace_view
# ---------------------------------------------------------------------------

def test_trace_to_state_handles_none():
    assert trace_to_state(None) == {}


def test_state_to_trace_view_attrs():
    state = {
        "final_answer": "42",
        "n_api_errors": 0,
        "hit_wall_clock": False,
        "tool_call_counts": {"python_exec": 3},
    }
    view = state_to_trace_view(state)
    assert view.final_answer == "42"
    assert view.n_api_errors == 0
    assert view.hit_wall_clock is False
    assert view.tool_call_counts == {"python_exec": 3}


def test_validation_rejects_bad_reuse_prob(tmp_path):
    with pytest.raises(ValueError):
        ArchCache(tmp_path / "c.jsonl", reuse_prob=-0.1)
    with pytest.raises(ValueError):
        ArchCache(tmp_path / "c.jsonl", reuse_prob=1.5)
