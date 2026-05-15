"""Executor smoke tests using the MockWorker (no LLM)."""

from __future__ import annotations

from arch_policy import (
    ARCH,
    BASELINE_REGISTRY,
    MockWorker,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    encode_named_arch,
    full_library,
    get_baseline,
    sample_arch,
)
from arch_policy.architecture.spec import ArchLogits
from arch_policy.executor.agent import Agent, parse_tool_call
from arch_policy.executor.synth import Synth, heuristic_extract
from arch_policy.executor.tools import call_tool
from arch_policy.reward import compute_reward


def test_each_baseline_runs_with_mock():
    worker = MockWorker(fake_answer="42", force_synth_done=True)
    ex = MultiAgentExecutor(worker=worker)
    for name in BASELINE_REGISTRY.keys():
        arch = get_baseline(name)
        trace = ex.run(task="What is 6 times 7?", arch=arch)
        assert trace.messages, f"{name}: no messages"
        # Sequence length == #active and all messages come from active slots
        assert len(trace.messages) == arch.n_active * trace.n_cycles_run
        for m in trace.messages:
            assert arch.active_mask[m.slot].item(), f"{name}: speaker {m.slot} not active"
        # Synth-DONE means we ran exactly 1 cycle
        assert trace.n_cycles_run == 1
        assert trace.final_via_synth
        assert trace.final_answer == "42"


def test_continue_then_done_runs_two_cycles():
    """If MockWorker returns CONTINUE first then ANSWER, executor should loop."""
    class ToggleSynthMock(Worker):
        def __init__(self):
            self.synth_calls = 0
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                self.synth_calls += 1
                if self.synth_calls == 1:
                    text = "CONTINUE"
                else:
                    text = "ANSWER: 42"
            else:
                text = "Candidate: 42"
            return WorkerOutput(text=text, n_input_tokens=10, n_output_tokens=4)

    worker = ToggleSynthMock()
    ex = MultiAgentExecutor(worker=worker)
    arch = get_baseline("solver_verifier")  # 2 actives → 2 turns per cycle
    trace = ex.run("Q", arch)
    assert trace.n_cycles_run == 2
    assert trace.final_via_synth
    assert trace.final_answer == "42"
    assert worker.synth_calls == 2


def test_safety_cap_uses_heuristic_fallback():
    """If Synth never says ANSWER, we hit safety cap and use heuristic extract."""
    class NeverDoneMock(Worker):
        def chat(self, system, user, max_new_tokens=512):
            if "You are a Synth" in system:
                return WorkerOutput(text="CONTINUE", n_input_tokens=4, n_output_tokens=2)
            return WorkerOutput(
                text="Candidate: 99", n_input_tokens=10, n_output_tokens=3
            )

    worker = NeverDoneMock()
    # Lower the safety cap to keep test fast (dataclasses.replace works on frozen)
    from dataclasses import replace
    spec = replace(ARCH, safety_max_cycles=2, safety_max_steps=2)
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    assert trace.n_cycles_run == 2
    assert not trace.final_via_synth
    # Heuristic picks the latest 'Candidate: 99'
    assert "99" in trace.final_answer


def test_react_inner_loop():
    """An agent that asks for python_exec then finalizes runs 2 ReAct steps."""
    class ToolingMock(Worker):
        def __init__(self):
            self.n = 0
        def chat(self, system, user, max_new_tokens=512):
            self.n += 1
            if self.n == 1:
                t = "THOUGHT: compute.\nACTION: python_exec\nARGS: print(2+3)"
            else:
                t = "Tested: pass; result 5"
            return WorkerOutput(text=t, n_input_tokens=20, n_output_tokens=8)

    worker = ToolingMock()
    agent = Agent(slot=0, role="Tester", worker=worker, max_steps=4)
    turn = agent.run("compute 2+3", incoming=[], cycle=0, turn=0, n_turns=1)
    assert turn.n_steps == 2
    assert turn.n_tool_calls == 1
    assert "5" in turn.text
    assert not turn.hit_cap


def test_parse_tool_call():
    assert parse_tool_call("ACTION: python_exec\nARGS: 1+1") == ("python_exec", "1+1")
    assert parse_tool_call("no tool here") is None
    multi = (
        "THOUGHT: think\n"
        "ACTION: sympy_check\n"
        "ARGS: x**2 - 4 = 0\n"
        "more notes"
    )
    name, args = parse_tool_call(multi)
    assert name == "sympy_check"
    assert "x**2" in args


def test_python_exec_basic():
    out = call_tool("python_exec", "print(1+2)")
    assert "3" in out


def test_synth_parse_done():
    class JudgeMock(Worker):
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text="ANSWER: 42", n_input_tokens=10, n_output_tokens=2)
    s = Synth(JudgeMock())
    v = s.judge("Q", [(0, "Solver", 0, "Candidate: 42")])
    assert v.is_done
    assert v.answer == "42"


def test_synth_parse_continue():
    class JudgeMock(Worker):
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text="CONTINUE", n_input_tokens=10, n_output_tokens=1)
    s = Synth(JudgeMock())
    v = s.judge("Q", [(0, "Solver", 0, "Candidate: ?")])
    assert not v.is_done


def test_synth_malformed_falls_back_to_continue():
    class JudgeMock(Worker):
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text="hmm let me think", n_input_tokens=10, n_output_tokens=4)
    s = Synth(JudgeMock())
    v = s.judge("Q", [])
    assert not v.is_done
    assert v.malformed


def test_heuristic_extract_priority():
    items = [
        (0, "Solver", 0, "Candidate: 1"),
        (1, "Verifier", 0, "Verified: 2"),
        (2, "Refiner", 0, "Refined: 3"),
    ]
    assert heuristic_extract(items) == "3"


def test_reward_correctness_dispatch():
    worker = MockWorker(fake_answer="42", force_synth_done=True)
    ex = MultiAgentExecutor(worker=worker)
    arch = get_baseline("single")
    trace = ex.run("Q", arch)
    rb = compute_reward(trace, gold_answer="42")
    assert rb.correctness == 1.0
    rb_wrong = compute_reward(trace, gold_answer="100")
    assert rb_wrong.correctness == 0.0


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
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
    print("\nall executor tests pass.")
