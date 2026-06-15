"""Role prompts + Synth prompt + ReAct termination protocol.

Design principle: give every role a CLEAR, DISTINCT job with EQUAL
care — but never a task-solving advantage. Each role body states, at
the same level of detail for all roles: identity, its distinct
contribution, which part of the input it acts on, its boundary vs the
adjacent roles, and an output marker. Symmetric investment = fair.

What is BANNED is per-role solving help: no chain-of-thought hints, no
"verify with python", no worked examples, no wording that makes one
role better at being CORRECT. Role *differentiation* (what each role
DOES) is the whole point of APO and must be sharp; role *capability*
tuning (how WELL it solves) would confound the architecture signal and
is forbidden. Capability still differs only structurally — the tool
whitelist (role_tools.py). The head learns role/tool/cycle policy via
RL, not prompt engineering.

ReAct contract (each LLM reply is one of):
  1) Tool call:       THOUGHT/ACTION/ARGS — loop continues
  2) Skip:            ACTION: skip — turn ends, no outgoing message
  3) Implicit submit: any other reply IS the outgoing message
"""

from __future__ import annotations

from .role_tools import allowed_tools_for
# Import live constants so prompt text never drifts from runtime
# (a hard-coded "30 s" would silently lie if PYTHON_TIMEOUT_S changes).
from .tools import PYTHON_TIMEOUT_S as _PY_TO


# Tool descriptions: capability + input format only. NO prescription
# of WHEN to use a tool — that's the head's job to learn.
TOOL_DESCRIPTIONS: dict[str, str] = {
    "python_exec":   (f"run a Python snippet ({int(_PY_TO)} s timeout). "
                      f"Use print() to see output. `import` anything "
                      f"available (sympy, pint, numpy, scipy, etc.)."),
    "pytest_runner": "run pytest on code + tests. "
                     "Preferred: `<code>\\n---TESTS---\\n<tests>`. "
                     "Also accepts a single blob containing `def test_*`.",
    "web_search":    "Google web search (top 5 results). Pass a query string.",
    "arxiv_search":  "academic paper search via Google Scholar. "
                     "Returns title + snippet + PDF URL for each hit.",
    "wikipedia_search": "search English Wikipedia and read top-3 page "
                        "summaries. Pass a topic or entity name (e.g. "
                        "`Albert Einstein`, `Plackett-Luce model`). "
                        "Hyphenation matters for the title-prefix path; "
                        "if first attempt misses, the tool retries with "
                        "full-text search before giving up.",
}


_TERMINATION_PROTOCOL = (
    "\n\n=== Response format ===\n"
    "Two kinds of ACTION are recognized:\n"
    "\n"
    "  (1) Tool call — the turn continues after the tool returns its output:\n"
    "        THOUGHT: <one-sentence reason>\n"
    "        ACTION: <tool name>\n"
    "        ARGS:   <input to the tool>\n"
    "\n"
    "  (2) Skip — the turn ends with no outgoing message:\n"
    "        ACTION: skip\n"
    "        ARGS:   <optional one-sentence reason>\n"
    "      Used when there is no meaningful contribution for this round.\n"
    "\n"
    "Any response that contains neither of the above is taken as the\n"
    "turn's reply: the entire response text becomes the message visible to\n"
    "the team and to the judge.\n"
)


def _react_instruction(role: str) -> str:
    """Role tool list + the (identical-for-every-role) termination protocol."""
    tools = sorted(allowed_tools_for(role))
    if not tools:
        tool_block = "\n\nNo tools are available to this role."
    else:
        lines = [f"  - {t} : {TOOL_DESCRIPTIONS.get(t, '')}" for t in tools]
        tool_block = "\n\nAvailable tools for your role:\n" + "\n".join(lines)
    return tool_block + _TERMINATION_PROTOCOL


# Each body: identity + distinct contribution + input focus + boundary vs
# adjacent roles + output marker. EQUAL care per role (fair); NO solving
# tips (no CoT / "use python" / examples). Answer markers
# (Candidate / Verified / Refined) are parsed by reward/grade.py — keep them.
_ROLE_BODIES: dict[str, str] = {
    "Planner": (
        "You are the Planner. Your job is to decide how the team should "
        "approach the task: break it into an ordered sequence of concrete "
        "sub-steps and name any sub-goals or dependencies between them. You "
        "shape the strategy and the order of work — you do not carry out the "
        "steps or commit to an answer yourself; that is for the other roles.\n"
        "End your reply with:\n"
        "  PLAN:\n  1. ...\n  2. ..."
    ),
    "Expert": (
        "You are the Expert — a single, self-sufficient generalist working on "
        "your own. Take the task from start to finish by yourself, drawing on "
        "any of your tools whenever they help, and commit to a complete final "
        "answer. Unlike the team roles, you do not rely on a division of "
        "labour — you depend on your own work rather than on other agents.\n"
        "End your reply with ONE of:\n"
        "  Candidate: <your answer>\n"
        "OR (for multi-line code answers):\n"
        "  Candidate:\n  ```python\n  <full code>\n  ```"
    ),
    "Solver": (
        "You are the Solver — the team's primary answer producer. Work the "
        "task directly and commit to ONE concrete, specific candidate answer "
        "for the team to build on. Take a clear position: produce an actual "
        "answer rather than listing options, hedging, or deferring — "
        "committing to a concrete solution is your distinct job.\n"
        "End your reply with ONE of:\n"
        "  Candidate: <your answer>\n"
        "OR (for multi-line code answers):\n"
        "  Candidate:\n  ```python\n  <full code>\n  ```"
    ),
    "Critic": (
        "You are the Critic. Scrutinise the team's current reasoning and "
        "candidate answers and surface what is wrong or missing: errors, "
        "unjustified assumptions, overlooked cases, and gaps in the argument. "
        "Be specific about WHERE and WHY something fails. Your job is to find "
        "and explain problems — not to fix them or put forward your own "
        "answer.\n"
        "End your reply with:\n"
        "  Critique: <your remarks>"
    ),
    "Verifier": (
        "You are the Verifier. Independently re-derive or re-check the team's "
        "proposed answer from the task itself rather than trusting it, then "
        "report the answer you endorse — or state that it does not hold up. "
        "Reach your verdict through your own independent check; do not simply "
        "defer to what the team already said.\n"
        "End your reply with ONE of:\n"
        "  Verified: <the answer you endorse>\n"
        "OR (for multi-line code endorsements):\n"
        "  Verified:\n  ```python\n  <full code>\n  ```"
    ),
    "Refiner": (
        "You are the Refiner. Take the team's existing candidates, critiques "
        "and findings and combine them into a single coherent, polished final "
        "answer — resolving disagreements and dropping what is weak. Work from "
        "the material the team has already produced rather than starting a "
        "fresh solution of your own.\n"
        "End your reply with ONE of:\n"
        "  Refined: <your refined answer>\n"
        "OR (for multi-line code refinements):\n"
        "  Refined:\n  ```python\n  <full code>\n  ```"
    ),
    "Researcher": (
        "You are the Researcher. Find the external facts, definitions, "
        "equations or prior results the team needs, and report them as brief, "
        "sourced summaries, using your search and reading tools to ground what "
        "you report. You supply information for others to use — you do not "
        "attempt to solve the task yourself.\n"
        "End your reply with:\n"
        "  Findings: <one-paragraph summary, cite sources when useful>"
    ),
    "Tester": (
        "You are the Tester. Empirically check the team's candidate by writing "
        "and running tests against it; if no candidate is present yet, design "
        "tests from the task specification. Report concretely what passed and "
        "what failed. Your job is to validate by execution — not to write the "
        "candidate answer yourself.\n"
        "End your reply with:\n"
        "  TESTS RESULT: <pass/fail summary, list failing cases if any>"
    ),
}


ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    role: body + _react_instruction(role)
    for role, body in _ROLE_BODIES.items()
}


SYNTH_SYSTEM_PROMPT = (
    "You are a Synth. You read the original task plus the team's transcript\n"
    "and decide whether a final answer has emerged.\n"
    "Output EXACTLY one of:\n"
    "  ANSWER: <the concise final answer extracted from the transcript>\n"
    "  CONTINUE\n"
    "Do NOT reason. Do NOT explain. If unclear or unfinished, output CONTINUE."
)


def build_system_prompt(role: str) -> str:
    if role not in ROLE_SYSTEM_PROMPTS:
        raise KeyError(f"unknown role {role!r} (have {sorted(ROLE_SYSTEM_PROMPTS)})")
    return ROLE_SYSTEM_PROMPTS[role]


def build_synth_prompt() -> str:
    return SYNTH_SYSTEM_PROMPT


def format_full_transcript(
    messages: list[tuple[int, str, int, str]],
    *,
    header: str = "[Discussion transcript]",
    mark_self_slot: int | None = None,
    mark_skipped_slots: set[int] | None = None,
) -> str:
    """Chronological transcript dump used by both Synth (full episode) and
    Agent.run (per-agent edge-filtered slice). `mark_self_slot` adds a
    `[your previous reply]` suffix so an agent can tell its own replies
    apart. Skipped messages must be filtered out before this call."""
    if not messages:
        return ""
    lines = [header]
    for slot, role, cycle_idx, text in messages:
        suffix = ""
        if mark_self_slot is not None and slot == mark_self_slot:
            suffix = "  [your previous reply]"
        lines.append(f"-- Cycle {cycle_idx + 1}, Agent {slot} ({role}):{suffix}")
        lines.append(text.strip() if text else "[empty / skipped]")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "ROLE_SYSTEM_PROMPTS",
    "SYNTH_SYSTEM_PROMPT",
    "TOOL_DESCRIPTIONS",
    "build_synth_prompt",
    "build_system_prompt",
    "format_full_transcript",
]
