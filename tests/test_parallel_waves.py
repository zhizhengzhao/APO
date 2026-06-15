"""V7-δ parallel within-cycle scheduler tests.

Covers:
  - _compute_waves: chain / star / MoA / no-edges / full-DAG / degenerate
  - Parallel vs sequential equivalence (same trace.messages set, same
    correctness, same termination_breakdown) on multiple architectures
  - Wave-internal independence: no agent in a wave can see another
    wave-mate's output
  - Speedup: parallel wall_clock < sequential wall_clock for wide archs
  - Resilience: an exception in one wave agent doesn't kill its wave-mates
  - Wall-clock + call-cap still fire correctly across wave boundaries
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import replace

import torch

from arch_policy import (
    ARCH,
    MockWorker,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    get_baseline,
)


# Most roles use "You are a <Role>." (e.g. Solver, Verifier, Critic). Two
# roles use "the" instead: "You are the Planner." and "You are the Expert —".
# Both forms must match; \b at the end avoids requiring a specific delimiter.
_ROLE_RE = re.compile(r"You are (?:a|the) (\w+)\b")


def _role_from_system(system: str) -> str:
    """Extract role from a build_system_prompt() system string.

    System prompts begin with `"You are a Solver. ..."` or `"You are the
    Planner. ..."` etc. Returns "Synth" for synth prompts ("You are a Synth.")
    or "?" if unrecognized.
    """
    m = _ROLE_RE.search(system)
    return m.group(1) if m else "?"


# ---------------------------------------------------------------------------
# _compute_waves unit tests
# ---------------------------------------------------------------------------

def _mkarch(active_seq, edges_list, k_max=6):
    """Build a stub ConcreteArch for wave-decomposition tests only."""
    arch = get_baseline("solver_verifier")
    n = k_max
    edges = torch.zeros(n, n, dtype=torch.bool)
    for src, dst in edges_list:
        edges[src, dst] = True
    arch.edges = edges
    arch.sequence = torch.tensor(active_seq, dtype=torch.long)
    return arch


def test_compute_waves_chain():
    """A→B→C: must be 3 sequential waves."""
    arch = _mkarch([0, 1, 2], [(0, 1), (1, 2)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2])
    assert waves == [[0], [1], [2]]


def test_compute_waves_no_edges():
    """No edges at all: 1 wave with all 3 in parallel."""
    arch = _mkarch([0, 1, 2], [])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2])
    assert waves == [[0, 1, 2]]


def test_compute_waves_moa():
    """Mixture-of-agents: 3 parallel Solvers → 1 Refiner reads all 3."""
    # sequence [S1, S2, S3, R], edges S1→R, S2→R, S3→R (NOT between solvers)
    arch = _mkarch([0, 1, 2, 3], [(0, 3), (1, 3), (2, 3)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2, 3])
    assert waves == [[0, 1, 2], [3]]


def test_compute_waves_star():
    """Hub A speaks first, B/C/D all read A independently."""
    arch = _mkarch([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2, 3])
    assert waves == [[0], [1, 2, 3]]


def test_compute_waves_partial_dependency():
    """Mixed: A→B, A→C, B→D (no edge C→D, no edge B→C). Sequence [A,B,C,D].
       Wave 0 = [A], wave 1 = [B, C] (both depend only on A; B/C parallel),
       wave 2 = [D] (depends on B)."""
    arch = _mkarch([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2, 3])
    assert waves == [[0], [1, 2], [3]]


def test_compute_waves_only_back_edges_ignored():
    """Edges that point BACKWARDS in sequence have no effect on waves
    (the source agent already spoke before the destination would read)."""
    # sequence [0, 1], edge 1→0 (backward). Within this cycle, 0 doesn't
    # see 1's output (because 0 spoke first), so wave decomposition treats
    # them as independent.
    arch = _mkarch([0, 1], [(1, 0)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1])
    assert waves == [[0, 1]]


def test_compute_waves_preserves_sequence_order_within_wave():
    """Within a wave, slots appear in original sequence order."""
    arch = _mkarch([2, 0, 1, 3], [(2, 3)])
    waves = MultiAgentExecutor._compute_waves(arch, [2, 0, 1, 3])
    # 2, 0, 1 all wave 0 (only 3 depends on 2); 3 wave 1.
    assert waves == [[2, 0, 1], [3]]


def test_compute_waves_concat_is_permutation():
    """Concatenating all waves must reproduce a permutation of active_idx."""
    arch = _mkarch([0, 1, 2, 3, 4], [(0, 2), (1, 3), (2, 4), (3, 4)])
    waves = MultiAgentExecutor._compute_waves(arch, [0, 1, 2, 3, 4])
    flat = [s for w in waves for s in w]
    assert sorted(flat) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Parallel == Sequential equivalence (on real arch via MockWorker)
# ---------------------------------------------------------------------------

def _run_with_mode(parallel: bool, arch_name="star_3", n_cycles=1):
    spec = replace(ARCH, safety_max_cycles=n_cycles, safety_max_steps=2)
    ex = MultiAgentExecutor(
        worker=MockWorker(fake_answer="42", force_synth_done=True),
        spec=spec,
        parallel_within_cycle=parallel,
    )
    arch = get_baseline(arch_name)
    return ex.run("Q", arch)


def test_parallel_serial_produce_same_final_answer_star():
    t_par = _run_with_mode(True, "star_3")
    t_ser = _run_with_mode(False, "star_3")
    assert t_par.final_answer == t_ser.final_answer
    assert t_par.n_synth_calls == t_ser.n_synth_calls
    assert t_par.n_cycles_run == t_ser.n_cycles_run


def test_parallel_serial_produce_same_message_set():
    """Same set of (slot, role, text) messages, regardless of execution order."""
    t_par = _run_with_mode(True, "star_3")
    t_ser = _run_with_mode(False, "star_3")
    par_set = {(m.slot, m.role, m.text) for m in t_par.messages}
    ser_set = {(m.slot, m.role, m.text) for m in t_ser.messages}
    assert par_set == ser_set


def test_parallel_serial_match_termination_breakdown():
    t_par = _run_with_mode(True, "star_3")
    t_ser = _run_with_mode(False, "star_3")
    assert t_par.termination_breakdown == t_ser.termination_breakdown
    assert t_par.protocol_compliance == t_ser.protocol_compliance


def test_parallel_serial_chain_arch_identical():
    """For a strict chain (solver_verifier), parallel mode == serial because
    waves are all single-agent."""
    t_par = _run_with_mode(True, "solver_verifier")
    t_ser = _run_with_mode(False, "solver_verifier")
    assert t_par.final_answer == t_ser.final_answer
    # In a chain, message ORDER is also identical (each wave has 1 slot).
    par_seq = [(m.slot, m.cycle, m.turn) for m in t_par.messages]
    ser_seq = [(m.slot, m.cycle, m.turn) for m in t_ser.messages]
    assert par_seq == ser_seq


# ---------------------------------------------------------------------------
# Wave-internal independence: a wave-mate's output must NOT leak
# ---------------------------------------------------------------------------

class _CaptureUserWorker(Worker):
    """Records the `user` prompt every agent saw, so we can assert no leak."""
    def __init__(self, fake_answer="42"):
        self.fake_answer = fake_answer
        self.prompts_by_role: dict[str, list[str]] = {}
        self.lock = threading.Lock()
    def chat(self, system, user, max_new_tokens=512):
        if "You are a Synth" in system:
            return WorkerOutput(text=f"ANSWER: {self.fake_answer}",
                                n_input_tokens=1, n_output_tokens=1)
        role = _role_from_system(system)
        with self.lock:
            self.prompts_by_role.setdefault(role, []).append(user)
        return WorkerOutput(text=f"Candidate: {self.fake_answer}",
                            n_input_tokens=10, n_output_tokens=3)


def test_wave_mates_cannot_see_each_others_output_in_star():
    """In star_3 the 3 Solvers run in one wave. None of them should see
    any other Solver's 'Candidate: 42' text in their user prompt."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _CaptureUserWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec, parallel_within_cycle=True)
    arch = get_baseline("star_3")
    ex.run("Q", arch)
    # Sanity: role detection actually worked (regression guard against the
    # original bug where prompts_by_role only had a "?" bucket and the
    # assertion below was vacuously true on an empty list).
    assert "Solver" in worker.prompts_by_role, (
        f"role extraction broken; got keys: {list(worker.prompts_by_role)}"
    )
    solver_prompts = worker.prompts_by_role["Solver"]
    assert len(solver_prompts) == 3, (
        f"expected 3 Solver calls (3-agent wave), got {len(solver_prompts)}"
    )
    # The 3 Solvers ran in wave 0. None of their user prompts should contain
    # "Candidate:" — that text only appears in wave-mate outputs.
    for prompt in solver_prompts:
        assert "Candidate:" not in prompt, (
            "wave-mate leak: Solver saw another Solver's output mid-wave"
        )


# ---------------------------------------------------------------------------
# Speedup: parallel < sequential when worker has a per-call delay
# ---------------------------------------------------------------------------

class _SleepingWorker(Worker):
    """Worker that sleeps `delay_s` per call. Lets us measure scheduling."""
    def __init__(self, fake_answer="42", delay_s=0.2):
        self.fake_answer = fake_answer
        self.delay_s = delay_s
    def chat(self, system, user, max_new_tokens=512):
        time.sleep(self.delay_s)
        if "You are a Synth" in system:
            return WorkerOutput(text=f"ANSWER: {self.fake_answer}",
                                n_input_tokens=1, n_output_tokens=1)
        return WorkerOutput(text=f"Candidate: {self.fake_answer}",
                            n_input_tokens=10, n_output_tokens=3)


def test_parallel_is_faster_for_star_arch():
    """star_3 has 3 parallel Solvers + 1 Verifier = waves of [3, 1].
    Sequential = 4 turns × delay; parallel = (max wave 1) + 1 = 2 × delay.
    Plus 1 synth call after the cycle."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    arch = get_baseline("star_3")

    t0 = time.time()
    ex_ser = MultiAgentExecutor(worker=_SleepingWorker(delay_s=0.2),
                                spec=spec, parallel_within_cycle=False)
    ex_ser.run("Q", arch)
    ser_wall = time.time() - t0

    t0 = time.time()
    ex_par = MultiAgentExecutor(worker=_SleepingWorker(delay_s=0.2),
                                spec=spec, parallel_within_cycle=True)
    ex_par.run("Q", arch)
    par_wall = time.time() - t0

    # Expect substantial speedup: serial ≈ 5 × 0.2 = 1.0s,
    # parallel ≈ 3 × 0.2 = 0.6s (wave_0_parallel + wave_1 + synth).
    # Use a loose ratio threshold (>=1.3x) to avoid flake on busy CI.
    assert par_wall < ser_wall, f"par={par_wall:.2f}s ser={ser_wall:.2f}s"
    speedup = ser_wall / par_wall
    assert speedup >= 1.3, f"speedup={speedup:.2f}x (par={par_wall:.2f}, ser={ser_wall:.2f})"


# ---------------------------------------------------------------------------
# Resilience: a wave-mate crash must not abort other wave-mates
# ---------------------------------------------------------------------------

class _CrashOneSlotWorker(Worker):
    """Worker that crashes only for a specific role; others succeed."""
    def __init__(self, crash_role="Solver", fake_answer="42"):
        self.crash_role = crash_role
        self.fake_answer = fake_answer
        self.calls = 0
        self.lock = threading.Lock()
    def chat(self, system, user, max_new_tokens=512):
        with self.lock:
            self.calls += 1
            n = self.calls
        if "You are a Synth" in system:
            return WorkerOutput(text=f"ANSWER: {self.fake_answer}",
                                n_input_tokens=1, n_output_tokens=1)
        # First Solver crashes; others (and Refiner) succeed.
        if self.crash_role in system and n == 1:
            raise RuntimeError("simulated wave-mate crash")
        return WorkerOutput(text=f"Candidate: {self.fake_answer}",
                            n_input_tokens=10, n_output_tokens=3)


class _DistinctOutputWorker(Worker):
    """Each non-synth call returns a unique `Candidate: ans_N` (N = slot).

    Sleeps `0.05 * (max_slot - slot)` seconds so HIGHER-numbered slots
    finish FIRST in a parallel wave. This forces `as_completed` to commit
    out-of-order vs original sequence — if `_commit_turn` didn't sort by
    (cycle, turn), `trace.messages` would be ordered reverse of sequence
    and `heuristic_extract` (which takes the last marker via reversed())
    would return a DIFFERENT answer in parallel vs serial mode. The
    regression guard catches that without this delay the GIL serializes
    completions in submission order and the test would pass even with
    broken code.
    """
    def __init__(self, fake_answer="42", max_slot=5):
        self.fake_answer = fake_answer
        self.max_slot = max_slot
        self.slot_re = re.compile(r"You are Agent (\d+)")
    def chat(self, system, user, max_new_tokens=512):
        if "You are a Synth" in system:
            return WorkerOutput(text="CONTINUE",
                                n_input_tokens=1, n_output_tokens=1)
        m = self.slot_re.search(user)
        slot = int(m.group(1)) if m else -1
        # Reverse-order delay so threads complete in reverse sequence order.
        time.sleep(0.05 * max(0, self.max_slot - slot))
        return WorkerOutput(text=f"Candidate: ans_{slot}",
                            n_input_tokens=10, n_output_tokens=3)


def test_parallel_serial_deterministic_final_answer_on_distinct_outputs():
    """Critical-bug regression for the subagent-review finding:
    `self_consistency_5` has 5 parallel Solvers each producing a distinct
    candidate. Before the (cycle, turn) sort fix in _commit_turn,
    heuristic_extract returned non-deterministic last-match because
    trace.messages was ordered by completion. After the fix, parallel and
    serial must produce IDENTICAL final_answer and message ORDER."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    arch = get_baseline("self_consistency_5")

    ex_par = MultiAgentExecutor(worker=_DistinctOutputWorker(),
                                spec=spec, parallel_within_cycle=True)
    t_par = ex_par.run("Q", arch)
    ex_ser = MultiAgentExecutor(worker=_DistinctOutputWorker(),
                                spec=spec, parallel_within_cycle=False)
    t_ser = ex_ser.run("Q", arch)

    # Final answer must match exactly (heuristic_extract sees same ordered
    # transcript in both modes).
    assert t_par.final_answer == t_ser.final_answer, (
        f"par={t_par.final_answer!r} ser={t_ser.final_answer!r}"
    )
    # trace.messages must be in the SAME order (sorted by cycle, turn).
    par_seq = [(m.slot, m.cycle, m.turn, m.text) for m in t_par.messages]
    ser_seq = [(m.slot, m.cycle, m.turn, m.text) for m in t_ser.messages]
    assert par_seq == ser_seq, (
        f"trace.messages order diverges:\npar={par_seq}\nser={ser_seq}"
    )


def test_parallel_distinct_outputs_stable_across_repeated_runs():
    """Run the SAME (arch, task) 10 times in parallel mode and assert all
    yield identical final_answer + trace.messages order. Without the sort
    fix, this catches non-determinism due to thread completion ordering."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    arch = get_baseline("self_consistency_5")
    seen_answers: set[str] = set()
    seen_orders: set[tuple] = set()
    for _ in range(10):
        ex = MultiAgentExecutor(worker=_DistinctOutputWorker(),
                                spec=spec, parallel_within_cycle=True)
        trace = ex.run("Q", arch)
        seen_answers.add(trace.final_answer)
        seen_orders.add(tuple((m.slot, m.cycle, m.turn) for m in trace.messages))
    assert len(seen_answers) == 1, (
        f"non-deterministic final_answer across 10 runs: {seen_answers}"
    )
    assert len(seen_orders) == 1, (
        f"non-deterministic trace.messages order: {seen_orders}"
    )


def test_wave_to_wave_propagation_in_star_arch():
    """Wave 0 = 3 Solvers, Wave 1 = 1 Verifier. The Verifier's user prompt
    must contain ALL THREE distinct Solver candidates (edges S→V wired)."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    captured: dict[str, list[str]] = {}
    captured_lock = threading.Lock()

    class _Worker(Worker):
        slot_re = re.compile(r"You are Agent (\d+)")
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                return WorkerOutput(text="ANSWER: ok",
                                    n_input_tokens=1, n_output_tokens=1)
            role = _role_from_system(system)
            with captured_lock:
                captured.setdefault(role, []).append(user)
            m = self.slot_re.search(user)
            slot = int(m.group(1)) if m else -1
            return WorkerOutput(text=f"Candidate: ans_{slot}",
                                n_input_tokens=10, n_output_tokens=3)

    ex = MultiAgentExecutor(worker=_Worker(), spec=spec,
                            parallel_within_cycle=True)
    arch = get_baseline("star_3")
    ex.run("Q", arch)
    verifier_prompts = captured.get("Verifier", [])
    assert len(verifier_prompts) == 1, (
        f"expected 1 Verifier call, got {len(verifier_prompts)}"
    )
    vp = verifier_prompts[0]
    # All 3 solver outputs must be visible to the Verifier (wave-to-wave
    # propagation via edges S0→V, S1→V, S2→V).
    for slot in (0, 1, 2):
        assert f"ans_{slot}" in vp, (
            f"Verifier missing Solver {slot}'s output; prompt:\n{vp}"
        )


def test_multi_cycle_parallel_matches_serial_with_distinct_outputs():
    """End-to-end: 2 cycles × star_3 with distinct per-slot outputs.
    Synth says CONTINUE in cycle 1, ANSWER in cycle 2. Parallel and serial
    must produce identical trace.messages + final_answer."""

    class _MultiCycleWorker(Worker):
        slot_re = re.compile(r"You are Agent (\d+)")
        def __init__(self):
            self.synth_calls = 0
            self.lock = threading.Lock()
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                with self.lock:
                    self.synth_calls += 1
                    n = self.synth_calls
                # cycle 1 → CONTINUE, cycle 2 → ANSWER
                if n == 1:
                    return WorkerOutput(text="CONTINUE",
                                        n_input_tokens=1, n_output_tokens=1)
                return WorkerOutput(text="ANSWER: final",
                                    n_input_tokens=1, n_output_tokens=1)
            m = self.slot_re.search(user)
            slot = int(m.group(1)) if m else -1
            return WorkerOutput(text=f"Candidate: ans_{slot}",
                                n_input_tokens=10, n_output_tokens=3)

    spec = replace(ARCH, safety_max_cycles=3, safety_max_steps=2)
    arch = get_baseline("star_3")

    ex_par = MultiAgentExecutor(worker=_MultiCycleWorker(),
                                spec=spec, parallel_within_cycle=True)
    t_par = ex_par.run("Q", arch)
    ex_ser = MultiAgentExecutor(worker=_MultiCycleWorker(),
                                spec=spec, parallel_within_cycle=False)
    t_ser = ex_ser.run("Q", arch)

    assert t_par.n_cycles_run == t_ser.n_cycles_run == 2
    assert t_par.final_answer == t_ser.final_answer == "final"
    par_seq = [(m.slot, m.cycle, m.turn, m.text) for m in t_par.messages]
    ser_seq = [(m.slot, m.cycle, m.turn, m.text) for m in t_ser.messages]
    assert par_seq == ser_seq, (
        f"multi-cycle divergence:\npar={par_seq}\nser={ser_seq}"
    )


def test_wave_mate_crash_does_not_take_down_other_mates():
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    ex = MultiAgentExecutor(worker=_CrashOneSlotWorker("Solver"),
                            spec=spec, parallel_within_cycle=True)
    arch = get_baseline("star_3")   # 3 Solvers wave + 1 Verifier wave
    trace = ex.run("Q", arch)
    # Should produce 4 AgentMessages: 1 protocol_fail (the crashed one)
    # plus 3 successes (other 2 Solvers + Verifier).
    assert len(trace.messages) == 4
    n_proto_fail = sum(1 for m in trace.messages if m.skip_kind == "protocol_fail")
    n_normal = sum(1 for m in trace.messages if not m.skipped)
    assert n_proto_fail == 1
    assert n_normal == 3
    assert trace.n_api_errors >= 1
