"""V7-γ termination protocol tests.

Covers:
  - parse_action: submit / skip / tool / garbage / case-insensitivity
  - Agent.run terminal flag invariants:
      submit_explicit XOR skipped_explicit XOR skipped_protocol_fail
  - submit ARGS becomes the turn's outgoing text
  - skip yields empty text + no propagation to latest_message
  - protocol_fail path: first violation → retry warning,
                       second violation → treated as skip
  - Executor: skipped turns do NOT update latest_message; downstream
    incoming continues to see the prior non-skipped reply
  - ExecutionTrace.protocol_compliance accumulates per role
"""

from __future__ import annotations

from dataclasses import replace

from arch_policy import (
    ARCH,
    BASELINE_REGISTRY,
    MockWorker,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    get_baseline,
)
from arch_policy.executor.agent import Agent, parse_action


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------

def test_parse_action_submit():
    out = parse_action("ACTION: submit\nARGS: hello world")
    assert out == ("submit", "hello world")


def test_parse_action_submit_case_insensitive():
    for variant in ("submit", "Submit", "SUBMIT", "sUbMiT"):
        out = parse_action(f"ACTION: {variant}\nARGS: x")
        assert out == ("submit", "x"), f"variant {variant!r} failed"


def test_parse_action_skip_with_reason():
    out = parse_action("ACTION: skip\nARGS: waiting for verifier")
    assert out == ("skip", "waiting for verifier")


def test_parse_action_skip_no_reason():
    # ARGS missing entirely → empty string
    out = parse_action("ACTION: skip")
    assert out == ("skip", "")


def test_parse_action_real_tool():
    out = parse_action(
        "THOUGHT: compute\nACTION: python_exec\nARGS: print(1+2)"
    )
    assert out == ("python_exec", "print(1+2)")


def test_parse_action_garbage_returns_none():
    # Common LLM typos that the V7-α whitelist caught — still must return None
    for text in (
        "no action here at all",
        "ACTION: thermal\nARGS: x",
        "ACTION: python_execARGS: x",   # missing newline before ARGS
        "ACTION: ketones",
    ):
        assert parse_action(text) is None, f"{text!r} should be None"


def test_parse_action_args_multiline():
    text = (
        "ACTION: submit\n"
        "ARGS: line one\n"
        "line two\n"
        "line three"
    )
    out = parse_action(text)
    assert out[0] == "submit"
    assert "line one" in out[1] and "line three" in out[1]


# ---------------------------------------------------------------------------
# Agent.run terminal flags
# ---------------------------------------------------------------------------

class _StubWorker(Worker):
    """Deterministic worker: returns a list of canned replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.n_calls = 0

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        # Synth uses this stub too in some tests; just keep cycling.
        if "You are a Synth" in system:
            return WorkerOutput(text="ANSWER: stub", n_input_tokens=1, n_output_tokens=1)
        idx = min(self.n_calls, len(self.replies) - 1)
        self.n_calls += 1
        return WorkerOutput(
            text=self.replies[idx], n_input_tokens=10, n_output_tokens=5,
        )


def test_explicit_submit_legacy_sets_only_submit_flag():
    """Legacy `ACTION: submit\\nARGS: x` is still accepted and treated
    identically to an implicit submit (V7-γ removed submit from the
    advertised protocol but kept the parser path for back-compat)."""
    agent = Agent(slot=0, role="Solver",
                  worker=_StubWorker(["ACTION: submit\nARGS: 42"]))
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.submit_implicit and not out.skipped_explicit
    assert not out.skipped_protocol_fail
    assert out.text == "42"
    assert out.n_steps == 1
    assert out.n_real_tool_calls == 0
    assert not out.skipped


def test_explicit_skip_sets_only_skip_flag():
    agent = Agent(slot=0, role="Critic",
                  worker=_StubWorker(["ACTION: skip\nARGS: nothing to add"]))
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.skipped_explicit and not out.submit_implicit
    assert not out.skipped_protocol_fail
    assert out.text == ""
    assert out.skip_reason == "nothing to add"
    assert out.skipped


def test_real_tool_then_implicit_submit_two_steps():
    """Step 1 is a real tool call; step 2 is plain text with no ACTION
    block — V7-γ takes that text as the turn's implicit submit reply."""
    agent = Agent(
        slot=0, role="Solver",
        worker=_StubWorker([
            "THOUGHT: compute\nACTION: python_exec\nARGS: print(1+2)",
            "The result is 3.",   # plain text → implicit submit
        ]),
        max_steps=4,
    )
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.submit_implicit
    assert "3" in out.text
    assert out.n_steps == 2
    assert out.n_tool_calls == 1
    assert out.n_real_tool_calls == 1
    assert out.tool_log[0][0] == "python_exec"


def test_plain_text_response_is_implicit_submit():
    """Bare text with no ACTION block is the default healthy path under
    V7-γ — no retry, no protocol failure, text becomes the contribution."""
    agent = Agent(
        slot=0, role="Solver",
        worker=_StubWorker(["The answer is 5 (no ACTION line)."]),
        max_steps=4,
    )
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.submit_implicit
    assert "5" in out.text
    assert out.n_steps == 1
    assert not out.skipped_protocol_fail


def test_empty_response_is_protocol_fail():
    """API returns blank string → skipped_protocol_fail (engineering noise,
    not architecture signal)."""
    agent = Agent(
        slot=0, role="Solver",
        worker=_StubWorker(["", "   ", "\n\n"]),
        max_steps=4,
    )
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.skipped_protocol_fail and out.skipped
    assert not out.submit_implicit and not out.skipped_explicit
    assert out.skip_cause == "empty_text"
    assert out.text == ""


def test_max_steps_without_submit_treated_as_skip():
    # Worker keeps issuing tool calls forever — eventually hits max_steps cap.
    agent = Agent(
        slot=0, role="Solver",
        worker=_StubWorker([
            "ACTION: python_exec\nARGS: print(1)",
        ]),
        max_steps=3,
    )
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.hit_cap and out.skipped_protocol_fail
    assert out.text == ""
    assert out.n_real_tool_calls == 3


def test_worker_error_treated_as_skip():
    agent = Agent(
        slot=0, role="Solver",
        worker=_StubWorker([
            "[GpuGeekWorker error: ConnectionError: ...]",
        ]),
        max_steps=4,
    )
    out = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert out.worker_error and out.skipped_protocol_fail
    assert out.text == ""


# ---------------------------------------------------------------------------
# Executor: skipped turns don't update latest_message; protocol_compliance accumulates
# ---------------------------------------------------------------------------

def test_executor_skipped_turn_excluded_from_protocol_compliance_submits():
    """Run a 1-cycle 2-agent arch where slot 0 skips. Verify
    protocol_compliance reflects 1 explicit_skip, 1 explicit_submit, and
    the skipped slot did NOT pollute latest_message."""

    class SkipFirstSubmitSecondWorker(Worker):
        def __init__(self):
            self.n = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                return WorkerOutput(text="ANSWER: ok", n_input_tokens=1, n_output_tokens=1)
            # First non-synth call (slot 0) → skip; second (slot 1) → submit
            self.n += 1
            if self.n == 1:
                return WorkerOutput(
                    text="ACTION: skip\nARGS: nothing yet",
                    n_input_tokens=10, n_output_tokens=3,
                )
            return WorkerOutput(
                text="ACTION: submit\nARGS: Verified: ok",
                n_input_tokens=10, n_output_tokens=3,
            )

    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = SkipFirstSubmitSecondWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")   # 2 actives, sequence [0, 1]
    trace = ex.run("Q", arch)

    # Two AgentMessage rows recorded
    assert len(trace.messages) == 2
    am_solver, am_verifier = trace.messages
    assert am_solver.skipped and am_solver.skip_kind == "explicit"
    assert am_solver.text == ""
    assert not am_verifier.skipped
    assert am_verifier.text == "Verified: ok"

    # protocol_compliance keyed by role only now (per-model dimension dropped
    # together with multi-vendor worker_pool).
    pc = trace.protocol_compliance
    assert "Solver" in pc and "Verifier" in pc
    assert pc["Solver"]["skipped_explicit"] == 1
    assert pc["Solver"]["submit_implicit"] == 0
    assert pc["Verifier"]["submit_implicit"] == 1
    assert trace.n_skipped_turns == 1
    assert trace.n_protocol_fail_turns == 0


def test_executor_skipped_turn_does_not_overwrite_latest_message():
    """Cycle 1: Solver submits "A". Cycle 2: Solver skips. Verifier in
    cycle 2 must still see "A" (the last non-skipped Solver reply), not
    empty / not the skip marker."""

    class TwoCycleWorker(Worker):
        def __init__(self):
            self.n = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                # CONTINUE after cycle 1 so we get a cycle 2; then ANSWER
                self.n += 0
                # use a count-by-synth approach
                pass
            self.n += 1
            # call sequence for a 2-agent arch (solver, verifier) with 2 cycles:
            #   c1 turn0 (Solver), c1 turn1 (Verifier), c1 Synth,
            #   c2 turn0 (Solver), c2 turn1 (Verifier), c2 Synth
            # but Synth gets its own branch above; we ignore it via n_real
            # Simpler: route by system content.
            if "You are a Synth" in system:
                return WorkerOutput(text="CONTINUE", n_input_tokens=1, n_output_tokens=1)
            return WorkerOutput(text="", n_input_tokens=0, n_output_tokens=0)

    # Override the above lazy stub with a proper count-driven worker.
    class TwoCycleWorker2(Worker):
        def __init__(self):
            self.synth_calls = 0
            self.agent_calls = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                self.synth_calls += 1
                # Cycle 1 → CONTINUE; cycle 2 → ANSWER
                if self.synth_calls == 1:
                    return WorkerOutput(text="CONTINUE", n_input_tokens=1, n_output_tokens=1)
                return WorkerOutput(text="ANSWER: A", n_input_tokens=1, n_output_tokens=1)
            self.agent_calls += 1
            # Sequence within an episode of solver_verifier (2 actives, 2 cycles):
            #   1: c1 Solver  → submit "A"
            #   2: c1 Verifier → submit "Verified: A"
            #   3: c2 Solver  → skip
            #   4: c2 Verifier → submit referring to whatever it saw
            if self.agent_calls == 1:
                return WorkerOutput(text="ACTION: submit\nARGS: A",
                                    n_input_tokens=10, n_output_tokens=3)
            if self.agent_calls == 2:
                return WorkerOutput(text="ACTION: submit\nARGS: Verified: A",
                                    n_input_tokens=10, n_output_tokens=3)
            if self.agent_calls == 3:
                return WorkerOutput(text="ACTION: skip\nARGS: no new info",
                                    n_input_tokens=10, n_output_tokens=3)
            # cycle 2 Verifier: capture what user prompt contained for later check
            self._cycle2_verifier_user = user
            return WorkerOutput(text="ACTION: submit\nARGS: Verified: A again",
                                n_input_tokens=10, n_output_tokens=3)

    spec = replace(ARCH, safety_max_cycles=3, safety_max_steps=2)
    worker = TwoCycleWorker2()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)

    # 4 turns (2 cycles × 2 agents), final via Synth on cycle 2
    assert trace.n_cycles_run == 2
    assert len(trace.messages) == 4
    c2_solver = trace.messages[2]
    c2_verifier = trace.messages[3]
    assert c2_solver.skipped and c2_solver.skip_kind == "explicit"
    assert c2_solver.text == ""
    assert not c2_verifier.skipped
    # Cycle 2 Verifier's user prompt MUST contain Solver's c1 "A"
    # (because latest_message[0] still holds c1's submission, not c2's skip).
    assert "A" in worker._cycle2_verifier_user
    # And it must NOT contain "no new info" (the skip reason should not leak)
    assert "no new info" not in worker._cycle2_verifier_user


def test_baselines_still_run_after_protocol_change():
    """All BASELINE_REGISTRY archs should still complete end-to-end with
    MockWorker (which now emits ACTION: submit wrappers)."""
    worker = MockWorker(fake_answer="42", force_synth_done=True)
    ex = MultiAgentExecutor(worker=worker)
    for name in BASELINE_REGISTRY:
        arch = get_baseline(name)
        trace = ex.run(task="6 * 7?", arch=arch)
        # Every active slot speaks once per cycle; Synth says DONE → 1 cycle
        assert trace.n_cycles_run == 1
        assert trace.final_via_synth
        assert trace.final_answer == "42"
        # Every recorded message should be an explicit submit (MockWorker
        # always wraps in ACTION: submit) → none skipped, none protocol_fail
        for m in trace.messages:
            assert not m.skipped, f"{name}: {m.role} skipped unexpectedly"


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
    print("\nall protocol tests pass.")
