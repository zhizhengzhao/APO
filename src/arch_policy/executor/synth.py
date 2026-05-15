"""Synth: the lightweight DONE/CONTINUE + answer-extraction judge.

Design:
  - Synth must NOT do reasoning. It only outputs one of:
      ANSWER: <the final answer>
      CONTINUE
  - It is implemented as one LLM call to the same worker as the agents.
  - Its cost (one call per cycle) is included in the reward calculation.

If Synth's output is malformed (neither ANSWER: nor CONTINUE), we fall back
to "CONTINUE" + log a warning. This is a safety behavior, not a learned
policy — Synth's accuracy is partially trained-in by the reward signal:
calling answer too early gives wrong answer → reward 0 → policy avoids
that pattern. Saying CONTINUE forever costs more API → reward shrinks.
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

    def __init__(self, worker: Worker, max_new_tokens: int = 64) -> None:
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

_VERIFIED_RE = re.compile(r"Verified\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_REFINED_RE = re.compile(r"Refined\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_CANDIDATE_RE = re.compile(r"Candidate\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_COMPUTED_RE = re.compile(r"Computed\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def heuristic_extract(transcript_items: list[tuple[int, str, int, str]]) -> str:
    """Best-effort answer extraction when Synth never said DONE.

    Priority: latest Refined > latest Verified > latest Computed > latest Candidate
              > last non-empty line.
    """
    # Reverse-scan for each prefix.
    for re_pat in (_REFINED_RE, _VERIFIED_RE, _COMPUTED_RE, _CANDIDATE_RE):
        for _slot, _role, _cycle, text in reversed(transcript_items):
            m = re_pat.search(text)
            if m:
                return m.group(1).strip().rstrip(".")
    # Fallback: the last non-empty line of the last message
    for _slot, _role, _cycle, text in reversed(transcript_items):
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line:
                return line[:200]
    return ""


__all__ = ["Synth", "SynthVerdict", "heuristic_extract"]
