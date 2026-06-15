"""Synth: lightweight DONE/CONTINUE + answer-extraction judge.

Synth must NOT reason. Each call outputs exactly one of:
  ANSWER: <final answer>
  CONTINUE

One LLM call per cycle (counts toward `n_llm_calls`, so calling CONTINUE
forever indirectly costs the policy via shaped_advantage). Malformed
output → fallback to CONTINUE; the reward signal trains Synth's accuracy
indirectly (early ANSWER on a wrong final → correctness 0 → -ε).

Truncation handling: if the underlying worker returns
`WorkerOutput.truncated = True` (finish_reason='length'), the regex
parser would silently capture the cut-off string (`ANSWER: <partial>`
still matches `^\\s*ANSWER\\s*:\\s*(.+?)\\s*$` because `.+?` is lazy
but pinned by the `$` end anchor). The captured partial would then
flow through to the grader as a "complete" answer, scoring 0 for any
HLE task whose answer is longer than the cap. So we explicitly reject
truncated outputs upstream of the regex and treat them as malformed
→ CONTINUE, giving the trace another cycle to produce a clean,
non-truncated verdict. (TODO-14, fixed 2026-05-28.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .multi_agent import Worker, WorkerOutput
from .prompts import build_synth_prompt, format_full_transcript


_ANSWER_PREFIX = re.compile(r"^\s*ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_CONTINUE_RE = re.compile(r"^\s*CONTINUE\s*$", re.IGNORECASE)


@dataclass
class SynthVerdict:
    is_done: bool
    answer: str = ""
    raw_output: str = ""
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    malformed: bool = False


class Synth:
    """A wrapper around a worker that issues the synth prompt and parses output."""

    def __init__(self, worker: Worker, max_new_tokens: int = 1024) -> None:
        # Default bumped 64 → 1024 (2026-05-28) after live data showed
        # ~1.5% of synth answers landed at the 64-tok cap on HLE
        # list/proof-style tasks. p99.9 of real synth answers is ~65
        # tokens, so 1024 gives ~16× headroom; cost is negligible
        # since synth is called only 1.4× per trace on average.
        self.worker = worker
        self.system_prompt = build_synth_prompt()
        self.max_new_tokens = max_new_tokens

    def judge(
        self,
        task: str,
        transcript_items: list[tuple[int, str, int, str]],
    ) -> SynthVerdict:
        """Return verdict given task + transcript.

        `transcript_items` follows `format_full_transcript`'s schema:
        list of (slot, role, cycle_idx, text).
        """
        user = (
            f"[Task]\n{task}\n\n"
            + format_full_transcript(transcript_items)
            + "\n[Your decision]"
        )
        out: WorkerOutput = self.worker.chat(
            system=self.system_prompt,
            user=user,
            max_new_tokens=self.max_new_tokens,
        )

        # Reject truncated outputs BEFORE regex match. See module
        # docstring for why: the ANSWER regex absorbs cut-off text
        # silently, causing partial answers to be graded as if
        # complete. Forcing malformed → CONTINUE gives the trace
        # another cycle to retry.
        if getattr(out, "truncated", False):
            return SynthVerdict(
                is_done=False, raw_output=(out.text or ""),
                malformed=True,
                n_input_tokens=out.n_input_tokens,
                n_output_tokens=out.n_output_tokens,
            )

        text = out.text.strip()

        # Try ANSWER first
        m = _ANSWER_PREFIX.match(text)
        if m:
            answer = m.group(1).strip()
            # Trim long single-line outputs (Synth should be terse, but defensive)
            if "\n" in answer:
                answer = answer.split("\n", 1)[0].strip()
            return SynthVerdict(
                is_done=True, answer=answer, raw_output=text,
                n_input_tokens=out.n_input_tokens,
                n_output_tokens=out.n_output_tokens,
            )
        if _CONTINUE_RE.match(text):
            return SynthVerdict(
                is_done=False, raw_output=text,
                n_input_tokens=out.n_input_tokens,
                n_output_tokens=out.n_output_tokens,
            )

        # Malformed — assume CONTINUE (the safer default; if all cycles
        # exhaust we'll fall back to a heuristic extractor at the executor level)
        return SynthVerdict(
            is_done=False, raw_output=text, malformed=True,
            n_input_tokens=out.n_input_tokens,
            n_output_tokens=out.n_output_tokens,
        )


# ---------------------------------------------------------------------------
# Heuristic fallback extractor used when we hit the safety cap with no DONE
# ---------------------------------------------------------------------------

# Inline (single-line) markers
_VERIFIED_RE = re.compile(r"Verified\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_REFINED_RE = re.compile(r"Refined\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CANDIDATE_RE = re.compile(r"Candidate\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Multi-line code-block variant: "Candidate:\n```python\n...\n```"
_VERIFIED_CODE_RE = re.compile(
    r"Verified\s*:\s*\n?\s*```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_REFINED_CODE_RE = re.compile(
    r"Refined\s*:\s*\n?\s*```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_CANDIDATE_CODE_RE = re.compile(
    r"Candidate\s*:\s*\n?\s*```(?:python|py)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def heuristic_extract(transcript_items: list[tuple[int, str, int, str]]) -> str:
    """Fallback answer extraction when Synth never produces a DONE verdict.

    Iterates by MESSAGE (latest first) → tries all patterns inside each
    message. Matches the executor's sequence-of-agents contract: the
    last writer in speaking order wins (an earlier message's marker
    cannot override a later one's). Within a single message, code-block
    patterns beat inline patterns (the model is "showing work" and the
    code block is the canonical form).

    Code blocks keep their ```python fences so exec-based graders
    (grade_humaneval etc.) recognise them; the rule grader's
    `_norm_short_answer` strips the fence + language tag for text
    benches like HLE.
    """
    code_patterns = (_REFINED_CODE_RE, _VERIFIED_CODE_RE, _CANDIDATE_CODE_RE)
    inline_patterns = (_REFINED_RE, _VERIFIED_RE, _CANDIDATE_RE)
    for _slot, _role, _cycle, text in reversed(transcript_items):
        # Code block on THIS message wins first within the message.
        for re_pat in code_patterns:
            m = re_pat.search(text)
            if m:
                code = m.group(1).rstrip()
                return f"```python\n{code}\n```"
        # Then inline `Refined:` / `Verified:` / `Candidate:` markers.
        for re_pat in inline_patterns:
            m = re_pat.search(text)
            if m:
                return m.group(1).strip().rstrip(".")
    # Final fallback: last non-empty line of the most recent message.
    for _slot, _role, _cycle, text in reversed(transcript_items):
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line:
                return line[:200]
    return ""


__all__ = ["Synth", "SynthVerdict", "heuristic_extract"]
