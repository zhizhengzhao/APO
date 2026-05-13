"""Task pool builders + grading.

Sources supported:
  - synthetic       : zero-network bootstrap (3 toy task families)
  - gsm8k           : math word problems, # ANSWER convention
  - humaneval       : code completion, exec-based grading
  - math            : MATH benchmark, boxed-answer grading
  - hotpotqa (TODO) : multi-hop QA, F1 grading

Each loaded sample carries (`task`, `gold_answer`, `family`, `task_id`,
optional `metadata`). The metadata field is used by complex graders
(HumanEval needs the test code, MATH the original boxed answer).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal


@dataclass
class TaskSample:
    task: str
    gold_answer: str
    family: str
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic tasks (no internet needed)
# ---------------------------------------------------------------------------

def _make_arithmetic(rng: random.Random, idx: int) -> TaskSample:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    op = rng.choice(["+", "-", "*"])
    if op == "+":
        ans = a + b
        text = f"What is {a} plus {b}?"
    elif op == "-":
        ans = a - b
        text = f"What is {a} minus {b}?"
    else:
        ans = a * b
        text = f"What is {a} times {b}?"
    return TaskSample(text, str(ans), family="synthetic_arith", task_id=f"arith_{idx}")


def _make_word_problem(rng: random.Random, idx: int) -> TaskSample:
    a = rng.randint(3, 30)
    b = rng.randint(1, a - 1)
    c = rng.randint(1, 20)
    ans = a - b + c
    text = (
        f"Mary has {a} apples. She gives {b} to John, then buys {c} more. "
        "How many apples does Mary have?"
    )
    return TaskSample(text, str(ans), family="synthetic_wordprob", task_id=f"wp_{idx}")


def _make_logic(rng: random.Random, idx: int) -> TaskSample:
    n = rng.randint(2, 4)
    vals = [rng.choice([True, False]) for _ in range(n)]
    label = sum(vals)
    text = "How many of these are True? " + ", ".join(str(v) for v in vals) + "."
    return TaskSample(text, str(label), family="synthetic_logic", task_id=f"logic_{idx}")


def load_local_synthetic(n_per_family: int = 50, seed: int = 7) -> list[TaskSample]:
    """Small dev set with no network requirement."""
    rng = random.Random(seed)
    out: list[TaskSample] = []
    for i in range(n_per_family):
        out.append(_make_arithmetic(rng, i))
    for i in range(n_per_family):
        out.append(_make_word_problem(rng, i))
    for i in range(n_per_family):
        out.append(_make_logic(rng, i))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Real tasks via HF datasets (optional; only if user has internet)
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K answers are in the form '... reasoning ... #### 18'."""
    if "####" in answer_field:
        return answer_field.split("####")[-1].strip().replace(",", "")
    return answer_field.strip()


def _extract_math_boxed(s: str) -> str:
    """Find the LAST \\boxed{...} content; nesting handled with brace counter."""
    # Find rightmost \boxed{
    m = re.search(r"\\boxed\s*\{", s)
    if not m:
        return s.strip()
    # Walk forward counting braces.
    start = m.end()
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return s[start : i - 1].strip()
    return s.strip()


def load_huggingface(
    dataset: Literal["gsm8k", "humaneval", "math"],
    split: str = "train",
    n: int | None = 200,
    seed: int = 0,
) -> list[TaskSample]:
    """Load a benchmark via HuggingFace `datasets`. Requires internet (or HF cache)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("`datasets` not installed. Run `pip install datasets`.") from e

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [
            TaskSample(
                task=row["question"],
                gold_answer=_extract_gsm8k_answer(row["answer"]),
                family="gsm8k",
                task_id=f"gsm8k_{i}",
                metadata={"raw_answer": row["answer"]},
            )
            for i, row in enumerate(ds)
        ]

    if dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            # The "task" is the prompt; the "gold_answer" is canonical_solution.
            # For grading we exec the model output against `test` + `entry_point`.
            out.append(
                TaskSample(
                    task=row["prompt"],
                    gold_answer=row["canonical_solution"],
                    family="humaneval",
                    task_id=row["task_id"],
                    metadata={
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                        "prompt": row["prompt"],
                    },
                )
            )
        return out

    if dataset == "math":
        # The lighteval split layout — try a couple of common dataset ids.
        for ds_name in ("HuggingFaceH4/MATH-500", "lighteval/MATH"):
            try:
                ds = load_dataset(ds_name, split=split if ds_name != "HuggingFaceH4/MATH-500" else "test")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load MATH dataset from any known mirror.")

        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            problem = row.get("problem") or row.get("question")
            solution = row.get("solution") or row.get("answer", "")
            gold = _extract_math_boxed(solution) if solution else row.get("answer", "")
            out.append(
                TaskSample(
                    task=problem,
                    gold_answer=gold,
                    family="math",
                    task_id=f"math_{i}",
                    metadata={"solution": solution, "level": row.get("level")},
                )
            )
        return out

    raise NotImplementedError(f"dataset={dataset} not wired yet")


def split_pools(samples: list[TaskSample], n_sft: int, n_rl: int, seed: int = 11):
    """Disjoint split for the SFT pool and the RL pool."""
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    if n_sft + n_rl > len(pool):
        raise ValueError(f"requested {n_sft}+{n_rl}={n_sft+n_rl} > available {len(pool)}")
    return pool[:n_sft], pool[n_sft : n_sft + n_rl]


__all__ = ["TaskSample", "load_huggingface", "load_local_synthetic", "split_pools"]
