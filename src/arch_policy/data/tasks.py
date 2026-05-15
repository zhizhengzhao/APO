"""Task pool builders + grading.

Sources supported:
  - synthetic       : zero-network bootstrap (3 toy task families)
  - gsm8k           : math word problems, # ANSWER convention
  - humaneval       : code completion, exec-based grading
  - mbpp            : code completion, exec-based grading
  - math            : MATH benchmark, boxed-answer grading
  - mmlu            : 4-choice general knowledge
  - bbh             : Big-Bench Hard, mixed reasoning
  - arc             : ARC science reasoning, 4-choice
  - mixed           : 6-source mix for SFT (gsm8k+math+humaneval+mmlu+bbh+arc)

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


def _format_mc_question(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    """Render a multiple-choice question as a single text prompt.

    `choices` are the option strings; `labels` are the answer letters
    (defaults to A/B/C/...).
    """
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [question.strip(), ""]
    for lab, ch in zip(labels, choices):
        lines.append(f"{lab}. {ch}")
    lines.append("")
    lines.append("Answer with the single letter.")
    return "\n".join(lines)


def load_huggingface(
    dataset: Literal["gsm8k", "humaneval", "mbpp", "math", "mmlu", "bbh", "arc"],
    split: str = "train",
    n: int | None = 200,
    seed: int = 0,
    subset: str | None = None,
) -> list[TaskSample]:
    """Load a benchmark via HuggingFace `datasets`. Requires internet (or HF cache).

    `subset` (optional) lets you pick a sub-config for datasets that have many
    (MMLU has 57 subjects, BBH has 27 tasks, ARC has 'ARC-Easy'/'ARC-Challenge').
    Defaults pick a reasonable fallback.
    """
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

    if dataset == "mbpp":
        ds = load_dataset("mbpp", subset or "sanitized", split=split if split != "train" else "train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            test_list = row.get("test_list") or row.get("test") or []
            tests = "\n".join(test_list) if isinstance(test_list, list) else str(test_list)
            out.append(
                TaskSample(
                    task=row.get("text") or row.get("prompt") or "",
                    gold_answer=row.get("code", ""),
                    family="mbpp",
                    task_id=f"mbpp_{row.get('task_id', i)}",
                    metadata={
                        "test": tests,
                        "code": row.get("code", ""),
                    },
                )
            )
        return out

    if dataset == "math":
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

    if dataset == "mmlu":
        # MMLU has 57 subjects. Use the aggregated `cais/mmlu` "all" config and
        # pick `n` random rows across subjects (more diverse than one subject).
        config = subset or "all"
        ds = load_dataset("cais/mmlu", config, split=split if split != "train" else "auxiliary_train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            choices = row["choices"]
            answer_idx = int(row["answer"])
            gold_letter = chr(ord("A") + answer_idx)
            out.append(
                TaskSample(
                    task=_format_mc_question(row["question"], choices),
                    gold_answer=gold_letter,
                    family="mmlu",
                    task_id=f"mmlu_{row.get('subject', 'all')}_{i}",
                    metadata={
                        "subject": row.get("subject"),
                        "choices": choices,
                        "answer_idx": answer_idx,
                    },
                )
            )
        return out

    if dataset == "bbh":
        # Big-Bench Hard has 27 tasks. We pick `n` rows from the chosen subset
        # (default: "logical_deduction_three_objects" — one of the smaller, well-defined ones).
        config = subset or "logical_deduction_three_objects"
        ds = load_dataset("lukaemon/bbh", config, split="test")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            out.append(
                TaskSample(
                    task=row["input"],
                    gold_answer=row["target"].strip(),
                    family=f"bbh_{config}",
                    task_id=f"bbh_{config}_{i}",
                    metadata={"subset": config},
                )
            )
        return out

    if dataset == "arc":
        # ARC has Easy + Challenge; default Challenge (harder, ~1.2k test).
        config = subset or "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", config, split=split if split != "train" else "train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            out.append(
                TaskSample(
                    task=_format_mc_question(row["question"], choices, labels),
                    gold_answer=row["answerKey"].strip(),
                    family=f"arc_{config}",
                    task_id=f"arc_{config}_{row['id']}",
                    metadata={"subset": config, "choices": choices, "labels": labels},
                )
            )
        return out

    raise NotImplementedError(f"dataset={dataset!r} not wired yet")


# ---------------------------------------------------------------------------
# Mixed pool for SFT (6-source diversity)
# ---------------------------------------------------------------------------

# Default mix: ratio targets for an SFT task pool.
# Tweak via `load_mixed`'s `ratios` arg if you want a different composition.
DEFAULT_SFT_MIX = {
    "gsm8k": 1500,
    "math": 1000,
    "humaneval": 800,    # actually capped at HumanEval's 164 unless you sample with replacement
    "mmlu": 700,
    "bbh": 500,
    "arc": 500,
}


def load_mixed(
    ratios: dict[str, int] | None = None,
    *,
    seed: int = 0,
    sft_split: str = "train",
) -> list[TaskSample]:
    """Build a mixed SFT task pool from multiple sources.

    `ratios`: maps dataset name → desired count. Defaults to DEFAULT_SFT_MIX
    (5000 total). Each source is loaded independently then concatenated and
    shuffled. If a source has fewer rows than requested, we just return all of
    them (no oversampling).
    """
    if ratios is None:
        ratios = DEFAULT_SFT_MIX
    rng = random.Random(seed)
    out: list[TaskSample] = []
    for source, count in ratios.items():
        try:
            samples = load_huggingface(source, split=sft_split, n=count, seed=seed)
        except Exception as e:
            print(f"[load_mixed] WARNING skipping {source}: {e}")
            continue
        out.extend(samples)
    rng.shuffle(out)
    return out


def split_pools(samples: list[TaskSample], n_sft: int, n_rl: int, seed: int = 11):
    """Disjoint split for the SFT pool and the RL pool."""
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    if n_sft + n_rl > len(pool):
        raise ValueError(f"requested {n_sft}+{n_rl}={n_sft+n_rl} > available {len(pool)}")
    return pool[:n_sft], pool[n_sft : n_sft + n_rl]


__all__ = [
    "TaskSample",
    "DEFAULT_SFT_MIX",
    "load_huggingface",
    "load_local_synthetic",
    "load_mixed",
    "split_pools",
]
