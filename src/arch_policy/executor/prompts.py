"""Role prompts + Synth prompt + ReAct tool-call protocol.

DESIGN PRINCIPLE — prompts are LUBRICANT, not POLICY.
======================================================
Each role gets a minimal identity sentence + an output-marker format. We
intentionally avoid prompt-engineering role behaviors (e.g. "Critic should
only raise concrete errors") because that bakes a fixed policy into the
prompt and defeats the project goal: the head learns *via RL* which roles
to activate for which task. Prompts should just keep the message stream
parseable and the agent generally on-task.

Markers (kept structural, not behavioral):
  Solver     -> Candidate: X         |       (or `Candidate:` + ```python``` for code)
  Critic     -> Critique: ...
  Verifier   -> Verified: X          |       (or `Verified:` + ```python``` for code)
  Refiner    -> Refined: X           |       (or `Refined:` + ```python``` for code)
  Planner    -> PLAN: 1. ...
  Decomposer -> ACTIONS: 1. ...
  Researcher -> Findings: ...
  Tester     -> TESTS RESULT: ...

ReAct tool-call protocol (parsed by `executor/agent.py`):
    THOUGHT: <reasoning>
    ACTION: <tool_name>
    ARGS: <single-line or multi-line args>

If no ACTION line is present, the response is treated as the agent's final
output for this turn.
"""

from __future__ import annotations


REACT_INSTRUCTION = (
    "\n\nYou may use a tool. To do so, write EXACTLY:\n"
    "  THOUGHT: <one-sentence reason>\n"
    "  ACTION: <tool name>\n"
    "  ARGS: <input to the tool>\n"
    "Available tools:\n"
    "  - python_exec : run python (5 s timeout)\n"
    "  - sympy_check : check a math identity or simplify\n"
    "  - web_search  : look up a fact (currently mocked)\n"
    "When no more tools are needed, write your final reply with the marker\n"
    "your role requires. Do not mix a tool call and a final reply."
)


# Each role: one identity sentence + output marker. Nothing more.
ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    "Planner": (
        "You are a Planner. Read the task and any context from other agents,"
        " then produce a short plan (2-4 sub-steps).\n"
        "End your reply with:\n"
        "  PLAN:\n  1. ...\n  2. ..."
        + REACT_INSTRUCTION
    ),
    "Decomposer": (
        "You are a Decomposer. Break the next thing the team needs to do into"
        " a short list of atomic actions.\n"
        "End your reply with:\n"
        "  ACTIONS:\n  1. ...\n  2. ..."
        + REACT_INSTRUCTION
    ),
    "Solver": (
        "You are a Solver. Produce a concrete candidate answer.\n"
        "End your reply with ONE of:\n"
        "  Candidate: <your answer>\n"
        "OR (for multi-line code answers):\n"
        "  Candidate:\n"
        "  ```python\n  <full code>\n  ```"
        + REACT_INSTRUCTION
    ),
    "Critic": (
        "You are a Critic. Read the team's discussion and add your perspective.\n"
        "End your reply with:\n"
        "  Critique: <your remarks>"
        + REACT_INSTRUCTION
    ),
    "Verifier": (
        "You are a Verifier. Independently check the team's answer and report\n"
        "the answer you endorse.\n"
        "End your reply with ONE of:\n"
        "  Verified: <your endorsed answer>\n"
        "OR (for multi-line code endorsements):\n"
        "  Verified:\n"
        "  ```python\n  <full code>\n  ```"
        + REACT_INSTRUCTION
    ),
    "Refiner": (
        "You are a Refiner. Integrate the team's candidates and critiques into\n"
        "a polished final answer.\n"
        "End your reply with ONE of:\n"
        "  Refined: <your refined answer>\n"
        "OR (for multi-line code refinements):\n"
        "  Refined:\n"
        "  ```python\n  <full code>\n  ```"
        + REACT_INSTRUCTION
    ),
    "Researcher": (
        "You are a Researcher. Gather facts / equations / definitions the team\n"
        "needs (use web_search if useful; else write what you know).\n"
        "End your reply with:\n"
        "  Findings: <one-paragraph summary>"
        + REACT_INSTRUCTION
    ),
    "Tester": (
        "You are a Tester. Write and run tests against the team's candidates\n"
        "(use python_exec).\n"
        "End your reply with:\n"
        "  TESTS RESULT: <pass/fail summary>"
        + REACT_INSTRUCTION
    ),
}


SYNTH_SYSTEM_PROMPT = (
    "You are a Synth. You read the original task plus the team's transcript\n"
    "and decide whether a final answer has emerged.\n"
    "Output EXACTLY one of:\n"
    "  ANSWER: <the concise final answer extracted from the transcript>\n"
    "  CONTINUE\n"
    "Do NOT reason. Do NOT explain. If unclear or unfinished, output CONTINUE."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_system_prompt(role: str) -> str:
    if role not in ROLE_SYSTEM_PROMPTS:
        raise KeyError(f"unknown role {role!r} (have {sorted(ROLE_SYSTEM_PROMPTS)})")
    return ROLE_SYSTEM_PROMPTS[role]


def build_synth_prompt() -> str:
    return SYNTH_SYSTEM_PROMPT


def format_incoming_messages(items: list[tuple[int, str, str]]) -> str:
    """Render messages from other agents into a context block.

    Each item is (sender_slot, sender_role, message_text).
    """
    if not items:
        return ""
    lines = ["[Messages from other agents]"]
    for slot, role, msg in items:
        lines.append(f"- Agent {slot} ({role}): {msg.strip()}")
    return "\n".join(lines)


def format_full_transcript(messages: list[tuple[int, str, int, str]]) -> str:
    """Render the full discussion transcript for the Synth.

    Each item is (slot, role, cycle_index, text).
    """
    lines = ["[Discussion transcript]"]
    for slot, role, cycle_idx, text in messages:
        lines.append(f"-- Cycle {cycle_idx + 1}, Agent {slot} ({role}):")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "REACT_INSTRUCTION",
    "ROLE_SYSTEM_PROMPTS",
    "SYNTH_SYSTEM_PROMPT",
    "build_synth_prompt",
    "build_system_prompt",
    "format_full_transcript",
    "format_incoming_messages",
]
