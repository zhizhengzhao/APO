"""Per-run cache for (task, architecture) → outcome.

GRPO samples G architectures per task per step. Because the policy is
narrow + the architecture space is discrete + small (≤ 72 named archs,
realistically ≤ 200 distinct concrete samples after warmup), the same
`(task, arch)` pair gets re-sampled frequently — both within a single
group (the head's typed distribution often produces duplicates at
G=12-16) and across steps. Each duplicate today costs another ~10
worker LLM calls + 1 judge call. The cache lets us reuse the prior
trace + reward and skip the API calls.

Scope:
  - **Within one GRPO run only.** Different runs may have different
    code, judge model, worker model, or seed — bit-stale results would
    silently corrupt the reward signal. Cache file lives next to the
    training checkpoints (`<out_dir>/arch_cache.jsonl`) and is loaded
    only by the same run on resume.
  - Append-only JSONL on disk for crash-recoverable persistence; backed
    by an in-memory dict for O(1) lookup.

Reuse policy:
  - `reuse_prob = 1.0` (default) → every cache hit returns the cached
    result, no fresh sample.
  - `reuse_prob < 1.0` → bernoulli draw per hit; if it returns False we
    pretend the cache missed, run fresh, and overwrite (preserves some
    stochastic exploration at the cost of API spend).

Thread safety: `get` and `put` are guarded by an internal `Lock`. The
JSONL append is also under the same lock (one writer at a time).
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


# ===========================================================================
# Canonical key
# ===========================================================================

def arch_key(arch) -> str:
    """Stable bytes-hash of a ConcreteArch.

    Hash covers the FULL tensor state (active_mask + roles + edges +
    sequence + per-slot model when present). Two arches with identical
    tensors → identical key.

    The model dimension is hashed ONLY when `arch.model is not None`
    (multi-model runs). For single-model runs (model=None) the hash is
    byte-identical to the pre-model-dim version, so an existing
    arch_cache stays valid on resume. Without this, two multi-model
    archs differing ONLY in model assignment would collide → the cache
    returns the first model's reward → the model dimension trains on
    identity/noise gradient.

    NOTE: exact-match, not graph-isomorphic — slot renumbering hashes
    differently, but the sampler picks canonical slot order so
    duplicates dominate.
    """
    parts = [
        arch.active_mask.detach().cpu().to(torch.bool).contiguous().numpy().tobytes(),
        arch.roles.detach().cpu().to(torch.long).contiguous().numpy().tobytes(),
        arch.edges.detach().cpu().to(torch.bool).contiguous().numpy().tobytes(),
        arch.sequence.detach().cpu().to(torch.long).contiguous().numpy().tobytes(),
    ]
    model = getattr(arch, "model", None)
    if model is not None:
        parts.append(model.detach().cpu().to(torch.long).contiguous().numpy().tobytes())
    return hashlib.sha1(b"".join(parts)).hexdigest()[:16]


def cache_key(task_id: str, arch) -> tuple[str, str]:
    return (task_id, arch_key(arch))


# ===========================================================================
# Cache
# ===========================================================================

@dataclass
class CachedEntry:
    """Everything `_run_one` produces that downstream GRPO needs.

    `correct` is `float` (not int) so partial-credit graders (e.g. code
    passes 7/10 = 0.7) round-trip without flipping the advantage tier.
    """
    total: float
    correct: float
    n_calls: int
    n_active: int
    n_edges: int
    trace_state: dict     # flat dict, rehydrated via state_to_trace_view
    stored_at_step: int


def trace_to_state(tr) -> dict:
    """Snapshot every ExecutionTrace field that `grpo.py` reads
    downstream. Anything not listed here is dropped (large transcripts,
    raw model outputs, etc.)."""
    if tr is None:
        return {}
    return {
        "final_answer":           tr.final_answer or "",
        "n_api_errors":           int(getattr(tr, "n_api_errors", 0)),
        "n_worker_truncations":   int(getattr(tr, "n_worker_truncations", 0)),
        "final_via_synth":        bool(getattr(tr, "final_via_synth", False)),
        "hit_cycle_cap":          bool(getattr(tr, "hit_cycle_cap", False)),
        "hit_wall_clock":         bool(getattr(tr, "hit_wall_clock", False)),
        "hit_call_cap":           bool(getattr(tr, "hit_call_cap", False)),
        "n_arch_caps_hit":        int(getattr(tr, "n_arch_caps_hit", 0)),
        "n_agents_hit_step_cap":  int(getattr(tr, "n_agents_hit_step_cap", 0)),
        "n_tool_truncations":     int(getattr(tr, "n_tool_truncations", 0)),
        "total_input_tokens":     int(getattr(tr, "total_input_tokens", 0)),
        "total_output_tokens":    int(getattr(tr, "total_output_tokens", 0)),
        "n_cycles_run":           int(getattr(tr, "n_cycles_run", 0)),
        "n_synth_calls":          int(getattr(tr, "n_synth_calls", 0)),
        "n_skipped_turns":        int(getattr(tr, "n_skipped_turns", 0)),
        "n_protocol_fail_turns":  int(getattr(tr, "n_protocol_fail_turns", 0)),
        "n_llm_calls":            int(getattr(tr, "n_llm_calls", 0)),
        "tool_call_counts":       dict(getattr(tr, "tool_call_counts", {}) or {}),
        "tool_error_counts":      dict(getattr(tr, "tool_error_counts", {}) or {}),
        "tool_error_kinds":       dict(getattr(tr, "tool_error_kinds", {}) or {}),
        "search_stub_counts":     dict(getattr(tr, "search_stub_counts", {}) or {}),
        "run_errors":             list(getattr(tr, "run_errors", []) or []),
        "protocol_compliance":    {k: dict(v) for k, v in
                                   (getattr(tr, "protocol_compliance", {}) or {}).items()},
        "termination_breakdown":  dict(getattr(tr, "termination_breakdown", {}) or {}),
        "python_exec_log":        list(getattr(tr, "python_exec_log", []) or []),
        # judge_audit lives on tr.extra (set by bench.make_reward_fn)
        "extra":                  dict(getattr(tr, "extra", {}) or {}),
    }


def state_to_trace_view(state: dict):
    """Inflate a saved trace_state dict into an object that quacks like
    ExecutionTrace for the fields grpo.py reads."""
    return SimpleNamespace(**state)


class ArchCache:
    """In-memory dict + append-only JSONL persistence.

    Threading: one Lock guards both the dict and the file.
    Loading: re-reads the full JSONL on construction. Last entry for a
    given key wins (so reuse_prob<1 overwrites are reflected).
    """

    def __init__(self, path: str | Path, reuse_prob: float = 1.0,
                 seed: int | None = None, rng_state=None):
        """`rng_state`: optional `_rng.getstate()` snapshot for
        bit-identical resume (overrides `seed` when given)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not (0.0 <= reuse_prob <= 1.0):
            raise ValueError(
                f"reuse_prob must be in [0, 1], got {reuse_prob}"
            )
        self.reuse_prob = float(reuse_prob)
        self._lock = threading.Lock()
        self._mem: dict[tuple[str, str], CachedEntry] = {}
        self._rng = random.Random(seed)
        if rng_state is not None:
            try:
                self._rng.setstate(tuple(rng_state)
                                   if not isinstance(rng_state, tuple)
                                   else rng_state)
            except Exception as e:  # noqa: BLE001
                print(f"[arch_cache] WARN failed to restore rng_state "
                      f"({type(e).__name__}: {e}); falling back to seed",
                      flush=True)
        # Stats — reset per step by `pop_step_stats`.
        self._step_hits = 0
        self._step_misses = 0
        self._step_skipped_reuse = 0   # cache had it, RNG said "fresh"
        self._load_from_disk()

    def rng_state(self):
        """Snapshot the bernoulli RNG state for persistence in resume.pt."""
        with self._lock:
            return self._rng.getstate()

    # ---- persistence ------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Hydrate the in-memory dict from the JSONL on disk.

        Defensive: a corrupt / schema-drifted line never aborts the
        whole load; bad lines are counted with a sample kept for the
        post-load WARN.
        """
        if not self.path.exists():
            return
        n_ok = 0
        n_bad = 0
        bad_sample = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["task_id"], rec["arch_hash"])
                    self._mem[key] = CachedEntry(
                        total=float(rec["total"]),
                        correct=float(rec["correct"]),
                        n_calls=int(rec["n_calls"]),
                        n_active=int(rec["n_active"]),
                        n_edges=int(rec["n_edges"]),
                        trace_state=dict(rec.get("trace_state", {})),
                        stored_at_step=int(rec.get("stored_at_step", -1)),
                    )
                    n_ok += 1
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    n_bad += 1
                    if bad_sample is None:
                        bad_sample = f"{type(e).__name__}: {e}"
                    continue
        msg = (f"[arch_cache] loaded {n_ok} entries "
               f"({len(self._mem)} unique) from {self.path}")
        if n_bad > 0:
            msg += (f"  · skipped {n_bad} corrupt/incompatible lines "
                    f"(first: {bad_sample})")
        print(msg, flush=True)

    def _append_jsonl(self, key: tuple[str, str], entry: CachedEntry) -> None:
        # Caller holds the lock.
        rec = {
            "task_id":   key[0],
            "arch_hash": key[1],
            "total":     entry.total,
            "correct":   entry.correct,
            "n_calls":   entry.n_calls,
            "n_active":  entry.n_active,
            "n_edges":   entry.n_edges,
            "trace_state": entry.trace_state,
            "stored_at_step": entry.stored_at_step,
            "ts": time.time(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---- API --------------------------------------------------------------

    def get(self, task_id: str, arch) -> CachedEntry | None:
        """O(1) lookup. Returns None on miss OR on RNG-driven skip."""
        key = (task_id, arch_key(arch))
        with self._lock:
            entry = self._mem.get(key)
            if entry is None:
                self._step_misses += 1
                return None
            # Bernoulli on reuse_prob.
            if self.reuse_prob >= 1.0 or self._rng.random() < self.reuse_prob:
                self._step_hits += 1
                return entry
            # Cached but RNG said "fresh" — caller will run + overwrite.
            self._step_skipped_reuse += 1
            return None

    def put(self, task_id: str, arch, entry: CachedEntry) -> None:
        key = (task_id, arch_key(arch))
        with self._lock:
            self._mem[key] = entry
            self._append_jsonl(key, entry)

    # ---- stats ------------------------------------------------------------

    def pop_step_stats(self) -> dict:
        """Read + reset the per-step counters."""
        with self._lock:
            stats = {
                "cache_hits":          self._step_hits,
                "cache_misses":        self._step_misses,
                "cache_skipped_reuse": self._step_skipped_reuse,
                "cache_size":          len(self._mem),
            }
            total = stats["cache_hits"] + stats["cache_misses"] + stats["cache_skipped_reuse"]
            stats["cache_hit_rate"] = (stats["cache_hits"] / total) if total else 0.0
            self._step_hits = 0
            self._step_misses = 0
            self._step_skipped_reuse = 0
            return stats

    def __len__(self) -> int:
        return len(self._mem)


__all__ = [
    "ArchCache",
    "CachedEntry",
    "arch_key",
    "cache_key",
    "state_to_trace_view",
    "trace_to_state",
]
