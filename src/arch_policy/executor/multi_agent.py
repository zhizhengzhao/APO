"""Multi-agent executor — main loop.

Naming convention:
  episode  — one full run of a task (one MultiAgentExecutor.run call)
  cycle    — one full pass through the PL-sampled sequence
  turn     — one agent's slot in a cycle (one Agent.run invocation)
  step     — one LLM call inside a turn's ReAct loop

Main loop:

  for cycle in 1 .. safety_max_cycles:
      for slot in arch.sequence:           # PL permutation of active slots
          incoming = msgs from slots with edge → slot
          turn_out = agent[slot].run(task, incoming, cycle, turn_idx)
          record turn; latest_message[slot] = turn_out.text
      verdict = synth.judge(task, transcript)
      if verdict.is_done: final_answer = verdict.answer; break

  # safety cap reached → heuristic_extract from transcript

Worker abstractions live here too: `Worker` (abstract), `MockWorker`
(for tests). The OpenAI-compatible worker for DeepSeek lives in
`openai_worker.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..architecture.sampler import ConcreteArch
from ..config import ARCH, ArchSpec


# ---------------------------------------------------------------------------
# Worker abstraction
# ---------------------------------------------------------------------------

@dataclass
class WorkerOutput:
    text: str
    n_input_tokens: int = 0
    n_output_tokens: int = 0


class Worker(ABC):
    """Anything that turns (system, user) into (text, token counts)."""

    @abstractmethod
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput: ...


# ---------------------------------------------------------------------------
# Mock worker (deterministic, no LLM) — for tests
# ---------------------------------------------------------------------------

class MockWorker(Worker):
    """Deterministic mock that mirrors role-suffix conventions.

    By default it returns a Solver-style 'Candidate: <fake_answer>' suffix.
    Pass `force_synth_done=True` to make it always reply 'ANSWER: <fake>' so
    the executor terminates after one cycle (great for fast smoke tests).
    """

    def __init__(
        self,
        fake_answer: str = "42",
        force_synth_done: bool = True,
    ) -> None:
        self.fake_answer = fake_answer
        self.force_synth_done = force_synth_done
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        self.calls.append((system, user))
        if self.force_synth_done and "You are a Synth" in system:
            text = f"ANSWER: {self.fake_answer}"
        elif "You are a Synth" in system:
            text = "CONTINUE"
        else:
            # Pick a suffix matching the role (best-effort — not strict)
            if "Verifier" in system:
                suffix = f"Verified: {self.fake_answer}"
            elif "Refiner" in system:
                suffix = f"Refined: {self.fake_answer}"
            elif "Tester" in system:
                suffix = f"TESTS RESULT: pass (mock); answer {self.fake_answer}"
            elif "Researcher" in system:
                suffix = f"Findings: known facts say {self.fake_answer}"
            elif "Planner" in system:
                suffix = "PLAN:\n  1. compute the answer\n  2. verify"
            elif "Decomposer" in system:
                suffix = "ACTIONS:\n  1. write down the formula\n  2. compute"
            elif "Critic" in system:
                suffix = "Critique:\n  - looks fine"
            else:
                suffix = f"Candidate: {self.fake_answer}"
            text = (
                f"[mock reply] (system head: {system[:60]!r}; "
                f"user head: {user[:60]!r})\n{suffix}"
            )
        return WorkerOutput(
            text=text,
            n_input_tokens=len((system + user).split()),
            n_output_tokens=len(text.split()),
        )


# ---------------------------------------------------------------------------
# Trace + executor
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """One agent's final reply at one (cycle, turn) position."""
    slot: int
    role: str
    cycle: int
    turn: int
    text: str
    n_steps: int = 0              # how many ReAct steps were used in this turn
    n_tool_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    hit_step_cap: bool = False


@dataclass
class ExecutionTrace:
    task: str
    arch: ConcreteArch
    messages: list[AgentMessage] = field(default_factory=list)
    final_answer: str = ""
    n_llm_calls: int = 0           # all worker.chat invocations (incl. ReAct steps + synth)
    n_cycles_run: int = 0          # how many cycles actually executed
    n_synth_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    synth_log: list[str] = field(default_factory=list)
    final_via_synth: bool = False  # True if final answer came from a DONE verdict
    extra: dict = field(default_factory=dict)


class MultiAgentExecutor:
    """Run a `ConcreteArch` against a task using a pluggable Worker."""

    def __init__(
        self,
        worker: "Worker",
        spec: ArchSpec | None = None,
        max_new_tokens_per_call: int | None = None,
        synth_max_new_tokens: int = 64,
    ) -> None:
        self.worker = worker
        self.spec = spec or ARCH
        # Default to ArchSpec.safety_max_tokens_per_call so a single source of
        # truth governs both the budget annotation and the actual call cap.
        self.max_new_tokens = (
            max_new_tokens_per_call
            if max_new_tokens_per_call is not None
            else self.spec.safety_max_tokens_per_call
        )
        self.synth_max_new_tokens = synth_max_new_tokens

    # ------------------------------------------------------------------
    def run(self, task: str, arch: ConcreteArch) -> ExecutionTrace:
        # Deferred imports to avoid circular import at module load time
        from .agent import Agent
        from .synth import Synth, heuristic_extract

        spec = self.spec
        trace = ExecutionTrace(task=task, arch=arch)

        # Build one Agent per active slot
        active_idx = arch.sequence.tolist()  # already a permutation of actives
        agents: dict[int, Agent] = {}
        for slot in active_idx:
            role = spec.role_names[int(arch.roles[slot].item())]
            agents[slot] = Agent(
                slot=slot,
                role=role,
                worker=self.worker,
                max_steps=spec.safety_max_steps,
                max_new_tokens=self.max_new_tokens,
            )

        synth = Synth(self.worker, max_new_tokens=self.synth_max_new_tokens)
        latest_message: dict[int, AgentMessage] = {}

        for cycle in range(spec.safety_max_cycles):
            for turn_idx, speaker_slot in enumerate(active_idx):
                agent = agents[speaker_slot]
                # Gather incoming messages from slots with an edge → speaker
                incoming: list[tuple[int, str, str]] = []
                for src in range(spec.n_max):
                    if src == speaker_slot:
                        continue
                    if not arch.edges[src, speaker_slot].item():
                        continue
                    if src not in latest_message:
                        continue
                    src_role = spec.role_names[int(arch.roles[src].item())]
                    incoming.append((src, src_role, latest_message[src].text))

                turn_out = agent.run(
                    task=task,
                    incoming=incoming,
                    cycle=cycle,
                    turn=turn_idx,
                    n_turns=len(active_idx),
                )

                am = AgentMessage(
                    slot=speaker_slot,
                    role=agent.role,
                    cycle=cycle,
                    turn=turn_idx,
                    text=turn_out.text,
                    n_steps=turn_out.n_steps,
                    n_tool_calls=turn_out.n_tool_calls,
                    n_input_tokens=turn_out.n_input_tokens,
                    n_output_tokens=turn_out.n_output_tokens,
                    hit_step_cap=turn_out.hit_cap,
                )
                trace.messages.append(am)
                latest_message[speaker_slot] = am

                trace.n_llm_calls += turn_out.n_steps
                trace.total_input_tokens += turn_out.n_input_tokens
                trace.total_output_tokens += turn_out.n_output_tokens

            trace.n_cycles_run = cycle + 1

            # Synth check after each cycle
            transcript_items = [
                (m.slot, m.role, m.cycle, m.text) for m in trace.messages
            ]
            verdict = synth.judge(task, transcript_items)
            trace.n_llm_calls += 1
            trace.n_synth_calls += 1
            trace.total_input_tokens += verdict.n_input_tokens
            trace.total_output_tokens += verdict.n_output_tokens
            trace.synth_log.append(verdict.raw_output)
            if verdict.is_done:
                # For code answers, Synth's terse "ANSWER: X" loses the body.
                # Check the transcript for a code-block Candidate/Verified/Refined
                # marker; if found, prefer it over Synth's short string.
                items = [(m.slot, m.role, m.cycle, m.text) for m in trace.messages]
                code_answer = heuristic_extract(items)
                if code_answer.startswith("```python"):
                    trace.final_answer = code_answer
                else:
                    trace.final_answer = verdict.answer
                trace.final_via_synth = True
                return trace

        # Hit safety cap → heuristic fallback
        transcript_items = [(m.slot, m.role, m.cycle, m.text) for m in trace.messages]
        trace.final_answer = heuristic_extract(transcript_items)
        trace.final_via_synth = False
        return trace


__all__ = [
    "AgentMessage",
    "ExecutionTrace",
    "MockWorker",
    "MultiAgentExecutor",
    "Worker",
    "WorkerOutput",
]
