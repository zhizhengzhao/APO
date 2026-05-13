"""Agent: a single role+slot wrapped around a worker LLM with a ReAct loop.

For each mini-step in the big-round sequence, the executor calls
`Agent.run(task, incoming, big_round, mini_step)`. The agent then runs an
inner ReAct loop:

  for inner_round in 1 .. safety_max_inner_rounds:
    1. accumulate scratchpad (THOUGHT / ACTION / ARGS / OBSERVATION lines)
    2. call worker LLM with [system_prompt, initial_user + scratchpad]
    3. parse response for ACTION
       - if ACTION found: run tool, append OBSERVATION, continue
       - else: this is the agent's final reply; return it

The safety cap is an engineering safeguard, NOT a policy parameter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .multi_agent import Worker, WorkerOutput
from .prompts import build_system_prompt, format_incoming_messages
from .tools import call_tool


# ---------------------------------------------------------------------------
# ReAct parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_ARGS_RE = re.compile(
    # ARGS: ... up to either next directive (THOUGHT/ACTION/OBSERVATION/ARGS)
    # or end of string.
    r"ARGS:\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION|OBSERVATION|ARGS):|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Look for an ACTION line; if present, return (tool_name, args).

    Returns None if no tool call detected (i.e., the response is final).
    """
    a = _ACTION_RE.search(text)
    if not a:
        return None
    tool_name = a.group(1)
    args_m = _ARGS_RE.search(text, a.end())
    args = args_m.group(1).strip() if args_m else ""
    return tool_name, args


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class AgentTurnOutput:
    """Result of one Agent.run invocation (one mini-step)."""
    text: str                     # final reply (last LLM output)
    n_inner_rounds: int = 0       # how many ReAct iterations were used
    n_tool_calls: int = 0         # how many tools were actually run
    n_input_tokens: int = 0       # cumulative across inner rounds
    n_output_tokens: int = 0
    hit_cap: bool = False         # True if safety_max_inner_rounds was hit
    tool_log: list[tuple[str, str, str]] = field(default_factory=list)
    # tool_log entries: (tool_name, args, output_truncated)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """A role-typed agent with a ReAct inner loop using one shared worker."""

    def __init__(
        self,
        slot: int,
        role: str,
        worker: Worker,
        max_inner_rounds: int = 8,
        max_new_tokens: int = 1024,
    ) -> None:
        self.slot = slot
        self.role = role
        self.worker = worker
        self.system_prompt = build_system_prompt(role)
        self.max_inner_rounds = max_inner_rounds
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    def _build_initial_user(
        self,
        task: str,
        incoming: list[tuple[int, str, str]],
        big_round: int,
        mini_step: int,
        n_mini: int,
    ) -> str:
        parts = [f"[Task]\n{task}"]
        ctx = format_incoming_messages(incoming)
        if ctx:
            parts.append("")
            parts.append(ctx)
        parts.append("")
        parts.append(
            f"[You are Agent {self.slot} ({self.role}) | "
            f"Big round {big_round + 1} | Mini step {mini_step + 1}/{n_mini}]"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        incoming: list[tuple[int, str, str]],
        big_round: int,
        mini_step: int,
        n_mini: int,
    ) -> AgentTurnOutput:
        """Run the ReAct inner loop. Returns the final (non-tool-call) reply."""
        initial_user = self._build_initial_user(
            task, incoming, big_round, mini_step, n_mini
        )
        out = AgentTurnOutput(text="")
        scratchpad = ""
        last_response: WorkerOutput | None = None

        for inner in range(self.max_inner_rounds):
            full_user = (
                initial_user + "\n\n[Scratchpad]\n" + scratchpad
                if scratchpad else initial_user
            )
            last_response = self.worker.chat(
                system=self.system_prompt,
                user=full_user,
                max_new_tokens=self.max_new_tokens,
            )
            out.n_inner_rounds = inner + 1
            out.n_input_tokens += last_response.n_input_tokens
            out.n_output_tokens += last_response.n_output_tokens

            tool_call = parse_tool_call(last_response.text)
            if tool_call is None:
                # Final reply
                out.text = last_response.text
                return out

            tool_name, args = tool_call
            tool_out = call_tool(tool_name, args)
            out.n_tool_calls += 1
            out.tool_log.append((tool_name, args, tool_out))

            # Append the agent's reply + observation to the scratchpad for next iter
            scratchpad += last_response.text.rstrip() + "\n"
            scratchpad += f"OBSERVATION:\n{tool_out}\n\n"

        # Hit cap: the agent kept asking for tools without finalizing.
        out.text = (
            last_response.text if last_response is not None
            else "[agent reached safety cap with no final reply]"
        )
        out.hit_cap = True
        return out


__all__ = ["Agent", "AgentTurnOutput", "parse_tool_call"]
