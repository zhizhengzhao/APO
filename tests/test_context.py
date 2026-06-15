"""V7-γ context-management tests.

Covers the new incoming-construction logic in `multi_agent.py`:
  - full chronological history (not just latest_message)
  - time-ordered rendering via format_full_transcript
  - self always visible without self-edge
  - other agents filtered by communication graph
  - skipped messages excluded
  - history_cycles truncation
"""

from __future__ import annotations

from dataclasses import replace

from arch_policy import (
    ARCH,
    MockWorker,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    get_baseline,
)
from arch_policy.executor.prompts import format_full_transcript


# ---------------------------------------------------------------------------
# format_full_transcript: self marker + chronological order
# ---------------------------------------------------------------------------

def test_format_full_transcript_chronological():
    items = [
        (0, "Solver",   0, "A1"),
        (1, "Critic",   0, "B1"),
        (0, "Solver",   1, "A2"),
    ]
    out = format_full_transcript(items)
    # Order must match input order (chronological)
    idx_a1 = out.index("A1")
    idx_b1 = out.index("B1")
    idx_a2 = out.index("A2")
    assert idx_a1 < idx_b1 < idx_a2


def test_format_full_transcript_self_marker():
    items = [
        (0, "Solver", 0, "A1"),
        (1, "Critic", 0, "B1"),
    ]
    out_with = format_full_transcript(items, mark_self_slot=1)
    out_without = format_full_transcript(items)
    # The self-marker variant must contain the suffix on the Critic line
    assert "[your previous reply]" in out_with
    # And NOT on the Solver line — the suffix only attaches to the marked slot
    pre_critic = out_with.split("Critic")[0]
    assert "[your previous reply]" not in pre_critic
    # The unmarked variant has no suffix anywhere
    assert "[your previous reply]" not in out_without


def test_format_full_transcript_empty_returns_empty():
    assert format_full_transcript([]) == ""


def test_format_full_transcript_skipped_text_shows_placeholder():
    # Caller may have left text="" for skipped messages; formatter renders a
    # placeholder rather than a blank line so the model sees something.
    items = [(0, "Solver", 0, "")]
    out = format_full_transcript(items)
    assert "[empty / skipped]" in out


# ---------------------------------------------------------------------------
# Executor incoming: full history, edge filtering, self auto-visible
# ---------------------------------------------------------------------------

class _CapturingWorker(Worker):
    """Captures every non-Synth (system, user) call. Always submits a stub."""

    def __init__(self, reply: str = "ACTION: submit\nARGS: stub"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []   # (system, user) per agent call
        self.synth_calls = 0

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        if "You are a Synth" in system:
            self.synth_calls += 1
            # cycle 1 → CONTINUE, cycle 2 → ANSWER
            if self.synth_calls == 1:
                return WorkerOutput(text="CONTINUE", n_input_tokens=1, n_output_tokens=1)
            return WorkerOutput(text="ANSWER: ok", n_input_tokens=1, n_output_tokens=1)
        self.calls.append((system, user))
        return WorkerOutput(text=self.reply, n_input_tokens=10, n_output_tokens=4)


def _run_two_cycles_with_capturing_worker(arch_name: str = "solver_verifier"):
    """Helper: run a 2-cycle episode with CapturingWorker and return
    (executor, trace, captured calls per agent turn)."""
    spec = replace(ARCH, safety_max_cycles=3, safety_max_steps=2)
    worker = _CapturingWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline(arch_name)
    trace = ex.run("Q", arch)
    return ex, trace, worker.calls


def test_incoming_includes_self_history_without_self_edge():
    """`solver_verifier` baseline has edges [(0,1)] only — no self-loops.
    After cycle 1, in cycle 2 the Solver (slot 0) should still see its
    OWN cycle-1 reply in incoming (self always visible)."""
    _, trace, calls = _run_two_cycles_with_capturing_worker("solver_verifier")
    # Calls layout for sequence [0, 1] over 2 cycles:
    #   calls[0] = c1 Solver (no prior incoming → 'A1' not in user)
    #   calls[1] = c1 Verifier  (sees Solver via edge 0→1)
    #   calls[2] = c2 Solver    (sees its own c1 reply via self auto-visible)
    #   calls[3] = c2 Verifier  (sees both)
    assert len(calls) >= 3
    _, c2_solver_user = calls[2]
    # The Solver's c1 reply was the stub "stub"; the c2 Solver must see it
    # marked as its own previous reply.
    assert "stub" in c2_solver_user
    assert "[your previous reply]" in c2_solver_user


def test_incoming_filtered_by_edge():
    """Verifier in `solver_verifier` (edges=[(0,1)]) sees Solver. If we
    swap to `single` baseline (no edges, 1 agent), the loner has no
    incoming except its own history."""
    _, _, calls = _run_two_cycles_with_capturing_worker("single")
    # `single` has only Solver (slot 0), edges=[]. So cycle 1 has no prior;
    # cycle 2 sees its own cycle-1 reply (self auto-visible).
    assert len(calls) == 2
    c1_user = calls[0][1]
    c2_user = calls[1][1]
    # c1: no discussion section, just task + agent header
    assert "[Discussion so far]" not in c1_user
    # c2: discussion section appears, contains the self-marked c1 reply
    assert "[Discussion so far]" in c2_user
    assert "[your previous reply]" in c2_user


def test_incoming_time_ordered_in_chain_of_three():
    """`chain_3` baseline = Solver→Critic→Verifier, 3 agents in sequence.
    After cycle 1, the cycle-2 Verifier should see all 3 cycle-1 replies
    in chronological order (Solver first, Critic second, Verifier self
    third). Plus the cycle-2 turns that have already happened.
    """
    _, trace, calls = _run_two_cycles_with_capturing_worker("chain_3")
    # 6 agent calls over 2 cycles × 3 actives
    assert len(calls) == 6
    _, c2_verifier_user = calls[5]
    # The transcript section in the user prompt should list cycles in
    # chronological order. We check the Cycle 1 lines precede Cycle 2.
    assert "Cycle 1" in c2_verifier_user
    assert "Cycle 2" in c2_verifier_user
    idx_c1 = c2_verifier_user.index("Cycle 1")
    idx_c2 = c2_verifier_user.index("Cycle 2")
    assert idx_c1 < idx_c2
    # And the Verifier (self) marker should appear at least once for the
    # cycle-1 self reply
    assert "[your previous reply]" in c2_verifier_user


def test_skipped_messages_excluded_from_incoming():
    """If an agent issues ACTION: skip, downstream agents must NOT see
    that empty 'turn' in their incoming."""

    class SkipFirstWorker(Worker):
        def __init__(self):
            self.synth_calls = 0
            self.agent_calls = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                self.synth_calls += 1
                return WorkerOutput(
                    text=("ANSWER: ok" if self.synth_calls >= 1 else "CONTINUE"),
                    n_input_tokens=1, n_output_tokens=1,
                )
            self.agent_calls += 1
            # cycle 1 Solver: skip
            # cycle 1 Verifier: submit, capture user for inspection
            if self.agent_calls == 1:
                return WorkerOutput(
                    text="ACTION: skip\nARGS: nothing yet",
                    n_input_tokens=10, n_output_tokens=3,
                )
            self._verifier_user = user
            return WorkerOutput(
                text="ACTION: submit\nARGS: Verified: ok",
                n_input_tokens=10, n_output_tokens=3,
            )

    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = SkipFirstWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    ex.run("Q", arch)

    # The Verifier's user prompt must not contain the skip reason
    # nor an "[empty / skipped]" placeholder (skipped messages are
    # filtered out before format_full_transcript is called).
    assert "nothing yet" not in worker._verifier_user
    assert "[empty / skipped]" not in worker._verifier_user
    # Should also not contain any "[Discussion so far]" (since the only
    # prior turn was skipped)
    assert "[Discussion so far]" not in worker._verifier_user


def test_history_cycles_truncation():
    """With history_cycles=1, only the current cycle (not earlier ones)
    should appear in incoming. With history_cycles=0, all cycles."""

    class ManyCyclesWorker(Worker):
        def __init__(self):
            self.synth_calls = 0
            self.agent_calls = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                self.synth_calls += 1
                # CONTINUE for cycles 1 and 2; ANSWER on cycle 3
                if self.synth_calls < 3:
                    return WorkerOutput(text="CONTINUE", n_input_tokens=1, n_output_tokens=1)
                return WorkerOutput(text="ANSWER: ok", n_input_tokens=1, n_output_tokens=1)
            self.agent_calls += 1
            # Tag each submit with the call count so we can spot which
            # cycle's content appears in later prompts.
            tag = f"TURN{self.agent_calls}"
            # Always capture latest user for inspection
            self._last_user = user
            return WorkerOutput(
                text=f"ACTION: submit\nARGS: {tag}",
                n_input_tokens=10, n_output_tokens=3,
            )

    # Cycle 3 Solver (call #5) — its user prompt should contain history.
    # arch = solver_verifier (2 actives), sequence [0,1], 3 cycles.
    # Without truncation: Solver in c3 sees [c1 Solver=TURN1, c1 Verifier=TURN2,
    #                                        c2 Solver=TURN3, c2 Verifier=TURN4]
    # With history_cycles=1: only c3 messages should appear → empty incoming
    # (since c3 Solver is the FIRST turn of c3)
    spec = replace(ARCH, safety_max_cycles=5, safety_max_steps=2)

    # --- variant A: history_cycles = 0 (unlimited) ---
    worker_a = ManyCyclesWorker()
    ex_a = MultiAgentExecutor(worker=worker_a, spec=spec, history_cycles=0)
    arch = get_baseline("solver_verifier")
    ex_a.run("Q", arch)
    # find the call right at the start of cycle 3 = 5th non-synth call
    user_c3_solver_a = ex_a  # placeholder; better path: re-derive from worker_a
    # ManyCyclesWorker stored _last_user only for the LAST call (which was
    # cycle 3 Verifier). To get cycle 3 Solver's prompt, instrument differently:
    # Re-run with explicit capture.

    class CaptureAllWorker(ManyCyclesWorker):
        def __init__(self):
            super().__init__()
            self.user_per_call: list[str] = []
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" not in system:
                self.user_per_call.append(user)
            return super().chat(system, user, max_new_tokens)

    worker_a2 = CaptureAllWorker()
    ex_a2 = MultiAgentExecutor(worker=worker_a2, spec=spec, history_cycles=0)
    ex_a2.run("Q", arch)
    # Sequence: c1 Solver(1), c1 Verifier(2), c2 Solver(3), c2 Verifier(4),
    #           c3 Solver(5), c3 Verifier(6) — final via Synth on c3.
    assert len(worker_a2.user_per_call) >= 5
    c3_solver_user_a = worker_a2.user_per_call[4]   # 0-indexed: 5th call
    # Unlimited: should contain TURN1 (c1 Solver, self auto-visible) AND TURN3
    assert "TURN1" in c3_solver_user_a
    assert "TURN3" in c3_solver_user_a

    # --- variant B: history_cycles = 1 (only current cycle) ---
    worker_b = CaptureAllWorker()
    ex_b = MultiAgentExecutor(worker=worker_b, spec=spec, history_cycles=1)
    ex_b.run("Q", arch)
    c3_solver_user_b = worker_b.user_per_call[4]
    # Truncated to cycle 3 only: c3 Solver hasn't said anything in c3 yet,
    # and edges[Verifier→Solver] = False in solver_verifier baseline, so
    # incoming is EMPTY.
    assert "TURN1" not in c3_solver_user_b
    assert "TURN3" not in c3_solver_user_b
    assert "[Discussion so far]" not in c3_solver_user_b


def test_history_cycles_negative_raises():
    import pytest
    with pytest.raises(ValueError):
        MultiAgentExecutor(worker=MockWorker(), history_cycles=-1)


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    print("\nall context tests pass.")
