"""Agent: a single role+slot wrapped around a worker LLM with a ReAct loop.

Naming convention used throughout the executor:
  episode  — one full run of a task
  cycle    — one full pass through the PL-sampled sequence
  turn     — one agent's slot in a cycle (one Agent.run invocation)
  step     — one LLM call inside a turn's ReAct loop

For each `turn`, the executor calls `Agent.run(task, incoming, cycle, turn)`.
The agent then runs an inner ReAct loop over `step`s:

  for step in 1 .. safety_max_steps:
    1. accumulate scratchpad (THOUGHT / ACTION / ARGS / OBSERVATION lines)
    2. call worker LLM with [system_prompt, initial_user + scratchpad]
    3. parse response:
         - ACTION: <tool>   → run tool, append OBSERVATION, continue
         - ACTION: skip     → explicit skip (no outgoing message)
         - anything else    → IMPLICIT SUBMIT: the whole response text
                              becomes the turn's outgoing message

Submit is *implicit*: as long as the model writes a non-tool, non-skip
reply, it's taken as the agent's contribution. This avoids penalizing
the architecture for the LLM's formatting quirks (e.g. forgetting an
`ACTION: submit` wrapper around a perfectly valid answer).

Skip is *explicit* — the model must say `ACTION: skip` to bow out.

The safety cap is an engineering safeguard, NOT a policy parameter.

Note: the prompt format here (THOUGHT / ACTION / ARGS / OBSERVATION) is a
provisional ReAct convention. When wiring up a real worker API, switch to
its native tool-calling protocol if available (e.g. OpenAI tool_calls,
Anthropic tool_use blocks); the executor's outer loop is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .multi_agent import Worker, WorkerOutput
from .prompts import build_system_prompt, format_full_transcript
from .role_tools import allowed_tools_for
from .tools import TOOLS, call_tool


# ---------------------------------------------------------------------------
# ReAct / termination parsing
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

# Termination action names. `skip` is the real protocol primitive (model
# must opt-in explicitly to bow out). `submit` is recognized so old prompts
# producing `ACTION: submit\nARGS: <text>` are treated as implicit submits
# with text=args (same outcome as plain text).
_TERMINATION_ACTIONS = frozenset({"submit", "skip"})


def parse_action(text: str) -> tuple[str, str] | None:
    """Parse the first ACTION block in `text`.

    Returns one of:
      ("skip",   args)       — explicit skip   (terminates the turn, no message)
      ("submit", args)       — legacy explicit submit; caller treats as implicit
                               submit with text=args
      (tool_name, args)      — tool call, where tool_name in TOOLS
      None                   — no ACTION block; caller treats whole `text` as
                               an implicit submit (the default healthy path)

    Whitelisting protects against the common LLM typo of jamming the next
    section onto the action name (`ACTION: python_execARGS: ...`) — the
    regex captures "python_execARGS" which is neither a tool nor a known
    primitive, so we return None and the caller treats the text as a reply.
    """
    a = _ACTION_RE.search(text)
    if not a:
        return None
    name = a.group(1)
    args_m = _ARGS_RE.search(text, a.end())
    args = args_m.group(1).strip() if args_m else ""
    if name.lower() in _TERMINATION_ACTIONS:
        return (name.lower(), args)
    if name in TOOLS:
        return (name, args)
    return None



# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class AgentTurnOutput:
    """Result of one Agent.run invocation (one turn).

    Termination outcome is captured by EXACTLY ONE of these booleans being
    True:
      submit_implicit         — model's non-tool reply taken as the agent's
                                contribution; `text` = reply (or ARGS if the
                                model used a legacy `ACTION: submit` wrapper)
      skipped_explicit        — model said `ACTION: skip`; `text` = ""
      skipped_protocol_fail   — engineering failure (worker error / wall_clock
                                / max_steps exhausted while still mid-tool /
                                truly empty response); `text` = ""

    The `skipped_*` paths produce NO outgoing message — the executor must
    NOT propagate them to other agents (this prevents engineering
    failures from polluting the architecture's correctness signal).
    """
    text: str                     # implicit submit body (or "" for skip / protocol_fail)
    n_steps: int = 0
    n_tool_calls: int = 0         # total tool dispatcher invocations
    n_real_tool_calls: int = 0    # subset excluding skip primitives (same as
                                  # n_tool_calls today; symmetry for future)
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    hit_cap: bool = False         # safety_max_steps exhausted while still using tools
    hit_wall_clock: bool = False  # wall_deadline fired mid-turn
    worker_error: bool = False    # API worker returned error sentinel
    # Termination flags (mutually exclusive among the three outcomes):
    submit_implicit: bool = False
    skipped_explicit: bool = False
    skipped_protocol_fail: bool = False
    # If skipped_protocol_fail: which cause fired. Must stay in sync
    # with the keys ExecutionTrace.termination_breakdown registers
    # (`skip_<cause>`); a new value here without a matching key there
    # silently falls back to skip_empty_text in _commit_turn → telemetry
    # lies. Current set:
    # ("worker_error" | "wall_clock" | "hit_cap" | "truncated" | "empty_text" | "")
    skip_cause: str = ""
    skip_reason: str = ""         # ARGS of explicit skip (telemetry only)
    tool_log: list[tuple[str, str, str, float]] = field(default_factory=list)
    # tool_log entries: (tool_name, args, tool_output_text, elapsed_s)

    # Telemetry passed back through the inner thread pool boundary —
    # _run_one_turn writes these so _commit_turn (which holds the trace
    # lock) can merge them. Direct trace mutation from inside the inner
    # pool would otherwise race; passing through turn_out is the clean
    # serialization point.
    search_stub_snapshot: dict | None = None   # {tool_name: n_stubs}
    run_error: dict | None = None              # {kind, type, message, traceback}

    @property
    def skipped(self) -> bool:
        """True if this turn produced NO outgoing message (either skip path)."""
        return self.skipped_explicit or self.skipped_protocol_fail


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
        max_steps: int = 20,
        max_new_tokens: int = 2048,
        allowed_tools: frozenset[str] | None = None,
    ) -> None:
        self.slot = slot
        self.role = role
        self.worker = worker
        self.system_prompt = build_system_prompt(role)
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        # If `allowed_tools` is None, fall back to the role's default pool
        # from role_tools.ROLE_TOOL_POOLS. Empty set ⇒ no tools.
        self.allowed_tools = (
            allowed_tools if allowed_tools is not None
            else allowed_tools_for(role)
        )

    # ------------------------------------------------------------------
    def _build_initial_user(
        self,
        task: str,
        incoming: list[tuple[int, str, int, str]],
        cycle: int,
        turn: int,
        n_turns: int,
    ) -> str:
        """Compose the per-turn user prompt.

        `incoming` is a chronological list of (slot, role, cycle_idx, text)
        — the agent's visible slice of trace.messages (filtered by edges +
        self + history_cycles + skipped exclusion). Rendered via
        `format_full_transcript` so the wire format matches what Synth
        sees, with the speaker's own past replies clearly marked.
        """
        parts = [f"[Task]\n{task}"]
        if incoming:
            parts.append("")
            parts.append(
                format_full_transcript(
                    incoming,
                    header="[Discussion so far]",
                    mark_self_slot=self.slot,
                )
            )
        parts.append("")
        parts.append(
            f"[You are Agent {self.slot} ({self.role}) | "
            f"Cycle {cycle + 1} | Turn {turn + 1}/{n_turns}]"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        incoming: list[tuple[int, str, int, str]],
        cycle: int,
        turn: int,
        n_turns: int,
        wall_deadline: float | None = None,
    ) -> AgentTurnOutput:
        """Run the ReAct inner loop until submit / skip / protocol_fail / cap.

        `wall_deadline` (unix epoch seconds) bounds the loop — if any step
        starts after the deadline, we abort and treat the turn as a
        protocol failure (no message propagated). Without this the loop
        can block max_steps × worker_timeout seconds inside a single turn.

        See AgentTurnOutput docstring for terminal flags.
        """
        import time as _time
        initial_user = self._build_initial_user(
            task, incoming, cycle, turn, n_turns
        )
        out = AgentTurnOutput(text="")
        scratchpad = ""
        last_response: WorkerOutput | None = None

        for step in range(self.max_steps):
            if wall_deadline is not None and _time.time() > wall_deadline:
                out.text = ""
                out.hit_wall_clock = True
                out.skipped_protocol_fail = True
                out.skip_cause = "wall_clock"
                return out

            full_user = (
                initial_user + "\n\n[Scratchpad]\n" + scratchpad
                if scratchpad else initial_user
            )
            last_response = self.worker.chat(
                system=self.system_prompt,
                user=full_user,
                max_new_tokens=self.max_new_tokens,
            )
            out.n_steps = step + 1
            out.n_input_tokens += last_response.n_input_tokens
            out.n_output_tokens += last_response.n_output_tokens

            text = last_response.text or ""

            # Worker error sentinel: parse out class+msg into structured
            # run_error BEFORE we wipe text="", so trace.run_errors keeps
            # the WHY (RateLimit / 500 / Timeout / ...) for forensics.
            # re.DOTALL is required — OpenAI APIError messages are often
            # multi-line.
            import re as _re
            sentinel_match = _re.match(
                r"\[(GpuGeek|DeepSeek|Qwen)Worker error: ([^:]+): (.+?)\]\s*\Z",
                text.strip(),
                _re.DOTALL,
            )
            sentinel_present = (
                "[GpuGeekWorker error" in text
                or "[DeepSeekWorker error" in text
                or "[QwenWorker error" in text
            )
            if sentinel_present:
                if sentinel_match:
                    worker_name, err_type, err_msg = sentinel_match.groups()
                else:
                    # Substring present but shape unexpected (truncation,
                    # etc.). Tag "Unknown" so we record something.
                    worker_name = err_type = "Unknown"
                    err_msg = text[:200]
                out.run_error = {
                    "kind": "worker_chat_sentinel",
                    "type": err_type,
                    "message": err_msg[:300],
                    "worker": worker_name + "Worker",
                }
                out.text = ""
                out.worker_error = True
                out.skipped_protocol_fail = True
                out.skip_cause = "worker_error"
                return out

            # Worker hit max_tokens mid-token (finish_reason=='length'):
            # parsing the half-sentence as `submit_implicit` would
            # corrupt final_answer. max_new_tokens is a meta-confound
            # the architecture can't control → mark as worker_error so
            # eng_valid masks the sample from the gradient.
            if getattr(last_response, "truncated", False):
                out.run_error = {
                    "kind": "worker_chat_truncated",
                    "type": "TruncatedAtMaxTokens",
                    "message": (f"reply hit max_tokens cap "
                                f"(n_output={last_response.n_output_tokens}); "
                                f"text first 80 chars: {text[:80]!r}"),
                    "worker": "(any)",
                }
                out.text = ""
                out.worker_error = True
                out.skipped_protocol_fail = True
                out.skip_cause = "truncated"
                return out

            # Empty response: engineering issue, not an architecture signal.
            if not text.strip():
                out.text = ""
                out.skipped_protocol_fail = True
                out.skip_cause = "empty_text"
                return out

            action = parse_action(text)

            # No ACTION block at all → implicit submit (the model's reply
            # IS the contribution; no protocol burden).
            if action is None:
                out.text = text
                out.submit_implicit = True
                return out

            kind, args = action

            if kind == "submit":
                # Legacy explicit submit — treat as implicit (same outcome).
                out.text = args if args else text
                out.submit_implicit = True
                return out

            if kind == "skip":
                out.text = ""
                out.skip_reason = args
                out.skipped_explicit = True
                return out

            # Real tool call — execute and continue ReAct.
            tool_name = kind
            t_tool_start = _time.time()
            tool_out = call_tool(tool_name, args, allowed=self.allowed_tools)
            t_tool_elapsed = _time.time() - t_tool_start
            out.n_tool_calls += 1
            out.n_real_tool_calls += 1
            out.tool_log.append((tool_name, args, tool_out, t_tool_elapsed))

            scratchpad += text.rstrip() + "\n"
            scratchpad += f"OBSERVATION:\n{tool_out}\n\n"

        # Exhausted max_steps without submit/skip — treat as protocol
        # failure rather than smuggling a half-baked response downstream.
        out.text = ""
        out.hit_cap = True
        out.skipped_protocol_fail = True
        out.skip_cause = "hit_cap"
        return out


__all__ = ["Agent", "AgentTurnOutput", "parse_action"]
