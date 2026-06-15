"""LLM-judge helper for open-ended short-answer grading.

A single judge call returns a `(score, audit)` pair. Worker exceptions map to
`GRADE_ERROR`; a missing `correct:` verdict falls back to the rule grader.

Only invoked for judge families (open short-text answers). The three released
RL categories (code / math / reasoning) all grade by rule, so this path is
inert for them; it is kept so the generic `CategoryBench` stays self-contained
for any future judge-graded corpus.

The judge prompt is the CAIS verbatim format (arXiv:2501.14249 §C.1.1).
"""

from __future__ import annotations

import re

from ..executor.multi_agent import Worker
from ..reward.grade import grade_short_answer
from .base import GRADE_ERROR


_JUDGE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


# Tolerant `correct: yes/no` matcher. Accepts the canonical CAIS format plus
# markdown-bold (`**correct:** yes`), JSON wrappers (`"correct": "yes"`), and
# mixed quoting, so a strict regex never silently falls back on wrappers.
_CORRECT_RE = re.compile(
    r"""(?:\*\*)?[\"']?correct[\"']?(?:\*\*)?      # name w/ optional ** and quotes
        \s*[:=]                                     # separator
        (?:\*\*)?\s*[\"']?\s*                       # optional ** + quote between
        (yes|no)\b                                  # the verdict
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXTRACTED_RE = re.compile(
    r"extracted_final_answer\s*[:=]\s*(.+?)(?=\n\s*(?:reasoning|correct|confidence)\s*[:=]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_RE = re.compile(
    r"reasoning\s*[:=]\s*(.+?)(?=\n\s*(?:correct|confidence)\s*[:=]|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def judge(
    question: str, prediction: str, gold: str, worker: Worker,
) -> tuple[float, dict]:
    """One judge call -> (score, audit).

    Worker exception -> GRADE_ERROR; missing `correct:` field -> rule grader.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question, response=prediction, correct_answer=gold,
    )
    try:
        # Judge is the single most important call for a judge-graded sample (a
        # truncated verdict silently mis-grades it), so give it a large safety
        # budget. Observed judge output is short (p99.9 ~316 tok), so 8192 is
        # pure headroom that guarantees the verdict line is never cut off.
        out = worker.chat(system="", user=prompt, max_new_tokens=8192)
    except Exception as e:  # noqa: BLE001
        return GRADE_ERROR[0], {
            **GRADE_ERROR[1],
            "judge_error": f"{type(e).__name__}: {str(e)[:160]}",
        }
    raw = (getattr(out, "text", "") or "").strip()
    judge_in = int(getattr(out, "n_input_tokens", 0) or 0)
    judge_out = int(getattr(out, "n_output_tokens", 0) or 0)
    truncated = bool(getattr(out, "truncated", False))
    m_correct = _CORRECT_RE.search(raw)
    if m_correct is None:
        err_label = "truncated_no_verdict" if truncated else "no_correct_field"
        print(f"[llm_judge] WARN: no `correct: yes/no` field in judge output "
              f"(len={len(raw)}, truncated={truncated}); falling back to rule "
              f"grader. first 160 chars: {raw[:160]!r}", flush=True)
        return grade_short_answer(prediction, gold), {
            "judge_path": "fallback",
            "judge_raw": raw[:300],
            "judge_error": err_label,
            "judge_truncated": truncated,
            "judge_in_tokens": judge_in,
            "judge_out_tokens": judge_out,
        }
    score = 1.0 if m_correct.group(1).lower() == "yes" else 0.0
    m_extr = _EXTRACTED_RE.search(raw)
    m_reas = _REASONING_RE.search(raw)
    return score, {
        "judge_path": "judge",
        "judge_extracted_answer": m_extr.group(1).strip() if m_extr else None,
        "judge_reasoning": m_reas.group(1).strip()[:500] if m_reas else None,
        "judge_truncated": truncated,
        "judge_in_tokens": judge_in,
        "judge_out_tokens": judge_out,
    }


__all__ = ["judge"]
