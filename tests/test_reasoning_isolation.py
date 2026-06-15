"""V7-γ reasoning-isolation contract tests.

Pins the design invariant: WorkerOutput.reasoning (the model's private
chain-of-thought) must NEVER appear in:
  - the agent's scratchpad (would feed back into the agent's own next step
    AND would leak into the agent's downstream messages via Synth/incoming)
  - the AgentTurnOutput.text (would propagate via submit ARGS)
  - the AgentMessage.text (the public propagation channel)
  - any downstream agent's incoming prompt
  - the Synth judge's transcript prompt
  - trace.final_answer

If you change worker output handling or context construction, the tests
in this file should catch any accidental reasoning leak.
"""

from __future__ import annotations

from dataclasses import replace

from arch_policy import (
    ARCH,
    MultiAgentExecutor,
    Worker,
    WorkerOutput,
    get_baseline,
)
from arch_policy.executor.agent import Agent


# Sentinel that is extremely unlikely to occur naturally; we plant it in
# `reasoning` and check no downstream surface contains it.
_SECRET = "ZZZ_SECRET_REASONING_TRACE_XYZ987"
_PUBLIC = "PUBLIC_REPLY_VALUE_42"


class _LeakProbeWorker(Worker):
    """Always returns text="ACTION: submit ARGS: PUBLIC..." plus a planted
    SECRET in `reasoning`. Captures every (system, user) call so we can
    assert SECRET does not appear in any downstream prompt."""

    def __init__(self, public: str = _PUBLIC, secret: str = _SECRET):
        self.public = public
        self.secret = secret
        self.calls: list[tuple[str, str]] = []   # (system, user)
        self.synth_calls = 0

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        self.calls.append((system, user))
        if "You are a Synth" in system:
            self.synth_calls += 1
            return WorkerOutput(
                text=f"ANSWER: {self.public}",
                n_input_tokens=1, n_output_tokens=4,
                reasoning=self.secret,
            )
        return WorkerOutput(
            text=f"ACTION: submit\nARGS: {self.public}",
            n_input_tokens=10, n_output_tokens=4,
            reasoning=self.secret,
        )


def test_reasoning_never_leaks_into_scratchpad_or_turn_text():
    """Single-agent turn: even though worker returned reasoning=SECRET,
    AgentTurnOutput.text must contain only the public submit ARGS."""
    worker = _LeakProbeWorker()
    agent = Agent(slot=0, role="Solver", worker=worker, max_steps=2)
    turn = agent.run(task="?", incoming=[], cycle=0, turn=0, n_turns=1)
    assert _SECRET not in turn.text
    assert _PUBLIC in turn.text
    # And the very next call's user prompt would have been built from
    # scratchpad → there's no next call here (single step), but if there
    # were, scratchpad has never written reasoning into it (parse_action
    # path returned on submit; tool path writes text not reasoning).


def test_reasoning_never_leaks_to_downstream_agents():
    """2-agent episode: Solver returns SECRET in reasoning; Verifier's
    user prompt must NOT contain SECRET."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _LeakProbeWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    # Verify the test setup actually ran both agents
    assert len(trace.messages) == 2
    # Every (system, user) ever sent must NOT contain SECRET.
    for sys_text, usr_text in worker.calls:
        assert _SECRET not in sys_text, (
            "reasoning leaked into a system prompt"
        )
        assert _SECRET not in usr_text, (
            "reasoning leaked into a user prompt — broken isolation!"
        )


def test_reasoning_never_leaks_to_synth_transcript():
    """Synth's transcript (which it sees in `user`) must NOT contain
    SECRET planted in worker reasoning."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _LeakProbeWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    ex.run("Q", arch)
    # Find the synth call
    synth_calls = [c for c in worker.calls if "You are a Synth" in c[0]]
    assert len(synth_calls) >= 1
    for sys_text, usr_text in synth_calls:
        assert _SECRET not in usr_text, (
            "Synth saw reasoning trace in its transcript — broken isolation!"
        )


def test_reasoning_never_leaks_into_trace_messages_or_final_answer():
    """trace.messages[*].text and trace.final_answer are the public
    record. Neither should ever contain SECRET."""
    spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=2)
    worker = _LeakProbeWorker()
    ex = MultiAgentExecutor(worker=worker, spec=spec)
    arch = get_baseline("solver_verifier")
    trace = ex.run("Q", arch)
    for m in trace.messages:
        assert _SECRET not in (m.text or ""), (
            f"slot {m.slot} message contains reasoning trace"
        )
    assert _SECRET not in trace.final_answer


def test_worker_output_reasoning_field_round_trips():
    """Direct round-trip: WorkerOutput now carries the reasoning field;
    consumers that want it can read it (e.g. for opt-in debug logs)."""
    out = WorkerOutput(
        text="public", n_input_tokens=1, n_output_tokens=1,
        reasoning="private CoT",
    )
    assert out.text == "public"
    assert out.reasoning == "private CoT"
    # Default is None
    out2 = WorkerOutput(text="x")
    assert out2.reasoning is None


# ---------------------------------------------------------------------------
# Real-worker leak regression tests
#
# `_LeakProbeWorker` above directly constructs WorkerOutput, which bypasses
# the actual *_worker.chat() implementations. The two tests below exercise
# the REAL chat() path with a stubbed OpenAI client and verify that an
# API response carrying (content="", reasoning_content="<secret>") does
# NOT smuggle the reasoning into WorkerOutput.text — which would propagate
# downstream via the agent's implicit-submit path.
# ---------------------------------------------------------------------------

class _StubMessage:
    def __init__(self, content: str, reasoning: str):
        self.content = content
        self.reasoning_content = reasoning


class _StubChoice:
    def __init__(self, content: str, reasoning: str):
        self.message = _StubMessage(content, reasoning)


class _StubUsage:
    def __init__(self, in_tok: int = 5, out_tok: int = 0):
        self.prompt_tokens = in_tok
        self.completion_tokens = out_tok


class _StubResponse:
    def __init__(self, content: str, reasoning: str):
        self.choices = [_StubChoice(content, reasoning)]
        self.usage = _StubUsage()


class _StubCompletions:
    def __init__(self, content: str, reasoning: str):
        self._content = content
        self._reasoning = reasoning

    def create(self, **_kw):
        return _StubResponse(self._content, self._reasoning)


class _StubChat:
    def __init__(self, content: str, reasoning: str):
        self.completions = _StubCompletions(content, reasoning)


class _StubOpenAIClient:
    def __init__(self, content: str, reasoning: str):
        self.chat = _StubChat(content, reasoning)


def test_deepseek_worker_does_not_leak_reasoning_on_empty_content():
    """API returns content="" + reasoning_content=SECRET (budget exhausted
    by reasoning). Worker MUST surface text="" (so agent.py's empty_text
    path fires) and keep SECRET only in the .reasoning field — never let
    it slide into .text where it would propagate to other agents / Synth."""
    from arch_policy.executor.deepseek_worker import DeepSeekWorker
    SECRET = "ZZZ_DS_SECRET_DO_NOT_LEAK_INTO_TEXT_42"
    w = DeepSeekWorker(api_key="dummy")
    w._client = _StubOpenAIClient(content="", reasoning=SECRET)

    out = w.chat("sys", "user", max_new_tokens=128)
    assert out.text == "", (
        f"reasoning_content leaked into WorkerOutput.text "
        f"(would propagate via implicit submit): {out.text!r}"
    )
    assert out.reasoning == SECRET, (
        "reasoning field should still expose the CoT for telemetry"
    )


def test_gpugeek_worker_does_not_leak_reasoning_on_empty_content():
    """Same contract test against the GpuGeek (OpenAI-compat) path."""
    from arch_policy.executor.gpugeek_worker import GpuGeekWorker
    SECRET = "ZZZ_GG_SECRET_DO_NOT_LEAK_INTO_TEXT_42"
    w = GpuGeekWorker(model="Vendor3/DeepSeek-V4-Flash", api_key="dummy")
    # The OpenAI-compat path caches per-key clients; populate the cache so
    # _openai(key) returns our stub.
    for key in w._key_pool:
        w._openai_clients[key] = _StubOpenAIClient(content="", reasoning=SECRET)

    out = w.chat("sys", "user", max_new_tokens=128)
    assert out.text == "", (
        f"reasoning_content leaked into WorkerOutput.text "
        f"(would propagate via implicit submit): {out.text!r}"
    )
    assert out.reasoning == SECRET


def test_qwen_worker_does_not_leak_reasoning_on_empty_content():
    """Same contract test against the Qwen / DashScope OpenAI-compat path."""
    from arch_policy.executor.qwen_worker import QwenWorker
    SECRET = "ZZZ_QW_SECRET_DO_NOT_LEAK_INTO_TEXT_42"
    w = QwenWorker(model="qwen3.7-max", api_key="dummy")
    w._client = _StubOpenAIClient(content="", reasoning=SECRET)

    out = w.chat("sys", "user", max_new_tokens=128)
    assert out.text == "", (
        f"reasoning_content leaked into WorkerOutput.text "
        f"(would propagate via implicit submit): {out.text!r}"
    )
    assert out.reasoning == SECRET


def test_workers_preserve_reasoning_when_content_also_present():
    """Sanity: the leak-fix doesn't drop the reasoning field when content
    is non-empty. Both fields should round-trip through chat()."""
    from arch_policy.executor.deepseek_worker import DeepSeekWorker
    PUBLIC = "PUBLIC_REPLY_OK"
    PRIVATE = "PRIVATE_COT_FOR_TELEMETRY"
    w = DeepSeekWorker(api_key="dummy")
    w._client = _StubOpenAIClient(content=PUBLIC, reasoning=PRIVATE)

    out = w.chat("sys", "user", max_new_tokens=128)
    assert out.text == PUBLIC
    assert out.reasoning == PRIVATE


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
    print("\nall reasoning-isolation tests pass.")
