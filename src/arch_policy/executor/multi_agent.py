"""Multi-agent executor (v3 main loop):

  - Build one Agent per active slot (each is a ReAct sub-loop).
  - Big-round loop:
      for speaker_slot in arch.sequence:    # Plackett-Luce permutation of actives
          incoming = msgs from slots with edge → speaker_slot
          turn = agent.run(task, incoming, big_round, mini_step)
          record turn; latest_message[speaker_slot] = turn.text
      verdict = synth.judge(task, transcript)
      if verdict.is_done: final_answer = verdict.answer; break
  - If we hit the safety cap with no DONE, fall back to heuristic extraction.

Worker abstractions live here too: `Worker` (abstract), `MockWorker`,
`HFWorker` (kept for offline/no-API testing). The OpenAI-compatible
worker for DeepSeek lives in `openai_worker.py`.
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
    the executor terminates after one big round (great for fast smoke tests).
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
            elif "ToolUser" in system:
                suffix = f"Computed: {self.fake_answer}"
            elif "Researcher" in system:
                suffix = f"Findings: known facts say {self.fake_answer}"
            elif "Planner" in system:
                suffix = "PLAN:\n  1. compute the answer\n  2. verify"
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
# HF transformers worker (kept for offline/local-only experiments)
# ---------------------------------------------------------------------------

class HFWorker(Worker):
    """Generate via a HuggingFace transformers model. Slow, no extra deps."""

    def __init__(self, model, tokenizer, *, device: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> WorkerOutput:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n_in = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return WorkerOutput(
            text=text,
            n_input_tokens=n_in,
            n_output_tokens=int(new_tokens.shape[-1]),
        )


# ---------------------------------------------------------------------------
# Trace + executor
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    slot: int
    role: str
    big_round: int
    mini_step: int
    text: str
    n_inner_rounds: int = 0
    n_tool_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    hit_inner_cap: bool = False


@dataclass
class ExecutionTrace:
    task: str
    arch: ConcreteArch
    messages: list[AgentMessage] = field(default_factory=list)
    final_answer: str = ""
    n_llm_calls: int = 0           # all worker.chat invocations (incl. inner + synth)
    n_big_rounds_run: int = 0      # how many big rounds actually executed
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
        max_new_tokens_per_call: int = 1024,
        synth_max_new_tokens: int = 64,
    ) -> None:
        self.worker = worker
        self.spec = spec or ARCH
        self.max_new_tokens = max_new_tokens_per_call
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
                max_inner_rounds=spec.safety_max_inner_rounds,
                max_new_tokens=self.max_new_tokens,
            )

        synth = Synth(self.worker, max_new_tokens=self.synth_max_new_tokens)
        latest_message: dict[int, AgentMessage] = {}

        for big_round in range(spec.safety_max_big_rounds):
            for mini_step, speaker_slot in enumerate(active_idx):
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

                turn = agent.run(
                    task=task,
                    incoming=incoming,
                    big_round=big_round,
                    mini_step=mini_step,
                    n_mini=len(active_idx),
                )

                am = AgentMessage(
                    slot=speaker_slot,
                    role=agent.role,
                    big_round=big_round,
                    mini_step=mini_step,
                    text=turn.text,
                    n_inner_rounds=turn.n_inner_rounds,
                    n_tool_calls=turn.n_tool_calls,
                    n_input_tokens=turn.n_input_tokens,
                    n_output_tokens=turn.n_output_tokens,
                    hit_inner_cap=turn.hit_cap,
                )
                trace.messages.append(am)
                latest_message[speaker_slot] = am

                trace.n_llm_calls += turn.n_inner_rounds
                trace.total_input_tokens += turn.n_input_tokens
                trace.total_output_tokens += turn.n_output_tokens

            trace.n_big_rounds_run = big_round + 1

            # Synth check after each big round
            transcript_items = [
                (m.slot, m.role, m.big_round, m.text) for m in trace.messages
            ]
            verdict = synth.judge(task, transcript_items)
            trace.n_llm_calls += 1
            trace.n_synth_calls += 1
            trace.total_input_tokens += verdict.n_input_tokens
            trace.total_output_tokens += verdict.n_output_tokens
            trace.synth_log.append(verdict.raw_output)
            if verdict.is_done:
                trace.final_answer = verdict.answer
                trace.final_via_synth = True
                return trace

        # Hit safety cap → heuristic fallback
        transcript_items = [(m.slot, m.role, m.big_round, m.text) for m in trace.messages]
        trace.final_answer = heuristic_extract(transcript_items)
        trace.final_via_synth = False
        return trace


__all__ = [
    "AgentMessage",
    "ExecutionTrace",
    "HFWorker",
    "MockWorker",
    "MultiAgentExecutor",
    "Worker",
    "WorkerOutput",
]
