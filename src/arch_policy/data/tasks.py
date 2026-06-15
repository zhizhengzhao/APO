"""Task pool builders + grading.

Sources supported:
  - synthetic       : zero-network bootstrap (3 toy task families)
  - gsm8k           : math word problems, # ANSWER convention
  - humaneval       : code completion, exec-based grading
  - mbpp            : code completion, exec-based grading
  - math            : MATH benchmark, boxed-answer grading
                      Supports `level_filter=(4, 5)` to keep hardest only.
  - mmlu            : 4-choice general knowledge
  - bbh             : Big-Bench Hard, mixed reasoning
  - arc             : ARC science reasoning, 4-choice
  - gpqa            : GPQA-Diamond (graduate-level STEM, 198 multi-choice,
                      ~34% non-expert / ~65% expert accuracy)
  - olympiad        : OlympiadBench math text-only English (581 IMO/IMC-level)
  - browsecomp      : OpenAI BrowseComp (1266 browsing-agent Q, ~30% SOTA;
                      tests Researcher + web_search heavily)
  - hle             : Humanity's Last Exam (2500 hard mixed-domain Q;
                      <50% SOTA. Auto-filtered to text-only; needs HF_TOKEN)
  - phybench        : PHYBench (500 physics olympiad-level Q; ~37% SOTA;
                      symbolic LaTeX answers, graded with sympy equivalence)
  - livecodebench   : LiveCodeBench code-generation (hard subset, ~30-50% SOTA;
                      exec-based grading via public/private test cases)
  - aime            : AIME 1983-2024 (~933 competition problems, integer
                      answers 0-999, very hard; graded as math)
  - omni_math       : Omni-MATH (~4.4k olympiad problems with a numeric
                      `difficulty`; `min_difficulty=` keeps the hard tail)
  - mmlu_pro        : MMLU-Pro (12k harder MC, up to 10 options, reasoning-
                      heavy; `subset=` filters by category, e.g. "physics")
  - mixed           : 11-source mix for SFT (classical 7 + agent-system 4;
                      see DEFAULT_SFT_MIX for the exact per-source counts)

Each loaded sample carries (`task`, `gold_answer`, `family`, `task_id`,
optional `metadata`). The metadata field is used by complex graders
(HumanEval needs the test code, MATH the original boxed answer).
"""

from __future__ import annotations

import ast
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal


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
    """Find the LAST `\\boxed{...}` content; handles nested braces.

    MATH solutions often enumerate intermediate \\boxed{} (case analysis,
    earlier-result recall); the LAST one is the final answer.
    """
    last_extracted = None
    for m in re.finditer(r"\\boxed\s*\{", s):
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
            last_extracted = s[start : i - 1].strip()
    return last_extracted if last_extracted is not None else s.strip()


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
    dataset: Literal["gsm8k", "humaneval", "mbpp", "math", "mmlu", "bbh", "arc",
                     "gpqa", "olympiad",
                     "browsecomp", "hle", "phybench", "livecodebench",
                     "aime", "omni_math", "mmlu_pro", "musr", "reclor"],
    split: str = "train",
    n: int | None = 200,
    seed: int = 0,
    subset: str | None = None,
    level_filter: tuple[int, ...] | None = None,
    min_difficulty: float | None = None,
) -> list[TaskSample]:
    """Load a benchmark via HuggingFace `datasets`. Requires internet (or HF cache).

    `subset` (optional) lets you pick a sub-config for datasets that have many
    (MMLU has 57 subjects, BBH has 27 tasks, ARC has 'ARC-Easy'/'ARC-Challenge').
    Defaults pick a reasonable fallback.

    `level_filter` (optional, MATH only): keep only rows whose `level` field
    matches. E.g. `level_filter=(4, 5)` keeps the two hardest tiers.

    `min_difficulty` (optional, Omni-MATH only): keep only rows whose numeric
    `difficulty` field is >= this threshold (Omni-MATH grades ≈1-10).
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
        ds = load_dataset("mbpp", subset or "sanitized", split=split)
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

        # Optional level filter (MATH levels are integers 1..5; some mirrors
        # store as "Level 5" string. Accept both.).
        if level_filter is not None:
            allowed = set()
            for lv in level_filter:
                allowed.add(int(lv))
                allowed.add(f"Level {lv}")
                allowed.add(str(lv))

            def _keep(row):
                lvl = row.get("level")
                if lvl is None:
                    return False
                if isinstance(lvl, str) and lvl.startswith("Level "):
                    try:
                        return int(lvl.split()[1]) in {int(x) for x in level_filter}
                    except (IndexError, ValueError):
                        return False
                try:
                    return int(lvl) in {int(x) for x in level_filter}
                except (TypeError, ValueError):
                    return str(lvl) in allowed

            ds = ds.filter(_keep)

        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            problem = row.get("problem") or row.get("question")
            solution = row.get("solution") or ""
            # HuggingFaceH4/MATH-500 already stores the canonical final
            # answer in `answer`; only fall back to boxed-extraction from
            # the solution if the dataset doesn't include it.
            gold = row.get("answer")
            if not gold:
                gold = _extract_math_boxed(solution) if solution else ""
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

    if dataset == "gpqa":
        # Public mirror of GPQA-Diamond (198 graduate-level STEM MC questions).
        # The original `Idavidrein/gpqa` is gated; this mirror packages the
        # `gpqa_diamond` config into a single CSV with the same content.
        # Columns: `problem` (question + embedded Choices), `answer` (A/B/C/D).
        ds = load_dataset("aradhye/gpqa_diamond", split="train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            out.append(
                TaskSample(
                    task=row["problem"].strip() + "\n\nAnswer with the single letter.",
                    gold_answer=row["answer"].strip(),
                    family="gpqa_diamond",
                    task_id=f"gpqa_d_{i}",
                    metadata={"subset": "diamond"},
                )
            )
        return out

    if dataset == "browsecomp":
        # OpenAI BrowseComp — 1266 browsing-agent questions (~30% SOTA 2026).
        # OpenAI publishes the test set as a CSV at openaipublic.blob with
        # 4 columns: `problem` (encrypted), `answer` (encrypted), `query_id`,
        # `canary` (per-row key). The HF mirror `OpenResearcher/web-bench`
        # drops the `canary` column, so we fetch the raw CSV directly.
        # Decryption: XOR(base64-decode(ciphertext), sha256(canary)*).
        import base64, csv, hashlib, io, urllib.request
        url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                csv_text = r.read().decode()
        except Exception as e:
            raise RuntimeError(f"failed to fetch BrowseComp CSV: {e}") from e

        def _bc_decrypt(ct_b64: str, password: str) -> str:
            ct = base64.b64decode(ct_b64)
            h = hashlib.sha256(password.encode()).digest()
            key = (h * ((len(ct) // len(h)) + 1))[: len(ct)]
            return bytes(a ^ b for a, b in zip(ct, key)).decode("utf-8")

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if n is not None:
            rng = random.Random(seed)
            rng.shuffle(rows)
            rows = rows[:n]
        out = []
        for i, row in enumerate(rows):
            canary = row.get("canary", "")
            try:
                q = _bc_decrypt(row.get("problem", ""), canary)
                ans = _bc_decrypt(row.get("answer", ""), canary)
            except Exception:
                continue  # skip rows that fail to decrypt
            out.append(
                TaskSample(
                    task=q.strip(),
                    gold_answer=ans.strip(),
                    family="browsecomp",
                    task_id=f"bc_{row.get('query_id', i)}",
                    metadata={"canary": canary},
                )
            )
        return out

    if dataset == "hle":
        # Humanity's Last Exam — 2500 questions across math/sciences/humanities.
        # SOTA <50% in 2026; ~half have images we cannot handle (filter out).
        # Loader requires HF_TOKEN env var (dataset is gated). Returns the
        # text-only subset by default; pass `subset="all"` to include image rows
        # (they'll be loaded but the executor will see a text-only prompt).
        ds = load_dataset("cais/hle", split="test",
                          token=os.environ.get("HF_TOKEN"))
        # Filter to text-only rows (no `image` / `image_url` / has_image false).
        if subset != "all":
            def _is_text_only(row):
                im = row.get("image") or row.get("image_url") or ""
                return not im
            ds = ds.filter(_is_text_only)
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            q = row.get("question") or row.get("problem") or ""
            ans = row.get("answer") or row.get("Answer") or ""
            cat = row.get("category") or row.get("subject") or "unknown"
            out.append(
                TaskSample(
                    task=q.strip(),
                    gold_answer=str(ans).strip(),
                    family="hle",
                    task_id=f"hle_{row.get('id', i)}",
                    metadata={
                        "category": cat,
                        "answer_type": row.get("answer_type"),
                        "author": row.get("author_name") or row.get("author"),
                    },
                )
            )
        return out

    if dataset == "phybench":
        # PHYBench — 500 physics problems (mechanics / EM / thermo / optics /
        # modern / advanced), high-school to olympiad level. Best 2026 model
        # ~37% accuracy. Answers are SYMBOLIC LaTeX (e.g. `\sqrt{2g/3R}`),
        # graded via sympy equivalence after LaTeX→Python preprocessing.
        # HF: Eureka-Lab/PHYBench, files served as JSON (not parquet), so
        # we use snapshot_download + manual json read to bypass dataset
        # loading-script issues.
        import json as _json
        from huggingface_hub import snapshot_download
        repo = snapshot_download(repo_id="Eureka-Lab/PHYBench",
                                 repo_type="dataset",
                                 allow_patterns=["*.json"])
        # File names include `_v1` suffix on current upload.
        for fname in ("PHYBench-fullques_v1.json", "PHYBench-questions_v1.json",
                      "PHYBench-onlyques_v1.json",
                      "PHYBench-fullques.json", "PHYBench-questions.json",
                      "PHYBench-onlyques.json"):
            fpath = os.path.join(repo, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    rows = _json.load(f)
                if rows:
                    break
        else:
            raise RuntimeError(
                f"No PHYBench JSON file found in snapshot at {repo}. "
                f"Files: {os.listdir(repo)}"
            )

        if n is not None:
            # Reproducible shuffle on the raw list (no HF Dataset wrap).
            rng = random.Random(seed)
            rows = rows.copy()
            rng.shuffle(rows)
            rows = rows[:n]
        ds = rows  # list[dict]
        out = []
        for i, row in enumerate(ds):
            q = row.get("content") or row.get("question") or row.get("problem", "")
            # `answer` may be the LaTeX symbolic gold; some splits use
            # `reference_answer` or `final_answer`.
            ans = (row.get("answer") or row.get("reference_answer")
                   or row.get("final_answer") or "")
            out.append(
                TaskSample(
                    task=(q.strip() + "\n\nWrite your final answer inside \\boxed{}."),
                    gold_answer=str(ans).strip(),
                    family="phybench",
                    task_id=f"phy_{row.get('id', i)}",
                    metadata={
                        "tag": row.get("tag") or row.get("tags"),
                        "solution": row.get("solution"),
                    },
                )
            )
        return out

    if dataset == "livecodebench":
        # LiveCodeBench (code generation) — rotating LeetCode/AtCoder/CodeForces
        # problems. JameSand/livecodebench-v6 mirror packs the data in
        # Verl/RL-training schema (131 rows in v6):
        #   prompt:       JSON-string list of {role, content} chat messages
        #   reward_model: JSON-string {style, ground_truth=test_cases_json}
        #   extra_info:   JSON-string {split, index, reference}
        # No difficulty field — all 131 are kept regardless.
        import json as _json
        ds = load_dataset("JameSand/livecodebench-v6", split="train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            # Parse JSON-string fields.
            prompt_raw = row.get("prompt", "[]")
            try:
                prompt = _json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
            except (_json.JSONDecodeError, TypeError):
                prompt = []
            if isinstance(prompt, list) and prompt:
                q = "\n\n".join(
                    str(m.get("content", "")) for m in prompt
                    if isinstance(m, dict) and m.get("role") in ("user", "system")
                )
            else:
                q = ""

            rm_raw = row.get("reward_model", "{}")
            try:
                rm = _json.loads(rm_raw) if isinstance(rm_raw, str) else rm_raw
            except (_json.JSONDecodeError, TypeError):
                rm = {}
            ground_truth = rm.get("ground_truth") if isinstance(rm, dict) else None

            extra_raw = row.get("extra_info", "{}")
            try:
                extra = _json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
            except (_json.JSONDecodeError, TypeError):
                extra = {}

            out.append(
                TaskSample(
                    task=q.strip() + "\n\nWrite the complete solution in a Python code block.",
                    gold_answer="",   # exec-based grading via tests
                    family="livecodebench",
                    task_id=f"lcb_{i}",
                    metadata={
                        "tests": ground_truth,
                        "split": (extra.get("split") if isinstance(extra, dict) else None),
                    },
                )
            )
        return out

    if dataset == "olympiad":
        # OlympiadBench math text-only English subset (581 IMO/IMC-level
        # problems). Schema: `question`, `final_answer` (list[str], 1 element),
        # `answer_type` ("Numerical" / "Expression"), `subfield`
        # (Algebra/Combinatorics/Geometry/Number Theory).
        ds = load_dataset("afraamn/olympiadbench_math_textonly", split="test_en")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            fa = row.get("final_answer")
            # final_answer is a list; gold = first element (canonical form).
            gold = fa[0] if isinstance(fa, list) and fa else str(fa or "")
            out.append(
                TaskSample(
                    task=(row["question"].strip()
                          + "\n\nWrite your final answer inside \\boxed{}."),
                    gold_answer=gold,
                    family="olympiad",
                    task_id=f"olymp_{row.get('question_id', i)}",
                    metadata={
                        "subfield": row.get("subfield"),
                        "answer_type": row.get("answer_type"),
                        "unit": row.get("unit"),
                        "is_multiple_answer": row.get("is_multiple_answer"),
                    },
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
        ds = load_dataset("allenai/ai2_arc", config, split=split)
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

    if dataset == "aime":
        # AIME 1983-2024 (di-zhang-fdu/AIME_1983_2024) — ~933 competition
        # problems, integer answers 0-999. Very hard for non-thinking models;
        # graded as math (boxed extraction + numeric equivalence).
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            q = (row.get("Question") or "").strip()
            ans = str(row.get("Answer") or "").strip()
            if not q or not ans:
                continue
            out.append(
                TaskSample(
                    task=q + "\n\nWrite your final answer inside \\boxed{}.",
                    gold_answer=ans,
                    family="aime",
                    task_id=f"aime_{row.get('ID', i)}",
                    metadata={"year": row.get("Year"), "difficulty": "hard"},
                )
            )
        return out

    if dataset == "omni_math":
        # Omni-MATH (KbsdJames/Omni-MATH) — ~4.4k olympiad problems carrying a
        # numeric `difficulty` (≈1-10). `min_difficulty` keeps only the hard
        # tail. Symbolic/boxed answers → grade_math.
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
        rows = list(ds)
        if min_difficulty is not None:
            def _omni_diff(r):
                try:
                    return float(r.get("difficulty"))
                except (TypeError, ValueError):
                    return -1.0
            rows = [r for r in rows if _omni_diff(r) >= min_difficulty]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if n is not None:
            rows = rows[:n]
        out = []
        for i, row in enumerate(rows):
            q = (row.get("problem") or "").strip()
            ans = str(row.get("answer") or "").strip()
            if not q or not ans:
                continue
            out.append(
                TaskSample(
                    task=q + "\n\nWrite your final answer inside \\boxed{}.",
                    gold_answer=ans,
                    family="omni_math",
                    task_id=f"omni_{i}",
                    metadata={
                        "difficulty": row.get("difficulty"),
                        "domain": row.get("domain"),
                        "omni_source": row.get("source"),
                    },
                )
            )
        return out

    if dataset == "mmlu_pro":
        # MMLU-Pro (TIGER-Lab/MMLU-Pro) — ~12k harder MC (up to 10 options,
        # reasoning-heavy, ~15% lower accuracy than MMLU). `subset` (comma list)
        # filters by `category` (e.g. "physics,chemistry,math"). Trailing "N/A"
        # padding options are stripped; gold from `answer_index` (safe since
        # padding is tail-only and the answer is never N/A).
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        if subset:
            wanted = {s.strip().lower() for s in subset.split(",")}
            ds = ds.filter(lambda r: (r.get("category") or "").lower() in wanted)
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            opts = list(row.get("options") or [])
            while opts and opts[-1] in ("N/A", "", None):
                opts.pop()
            ai = int(row.get("answer_index", -1))
            if not opts or ai < 0 or ai >= len(opts):
                continue
            gold_letter = chr(ord("A") + ai)
            out.append(
                TaskSample(
                    task=_format_mc_question(row["question"], opts),
                    gold_answer=gold_letter,
                    family="mmlu_pro",
                    task_id=f"mmlupro_{row.get('question_id', i)}",
                    metadata={
                        "category": row.get("category"),
                        "choices": opts,
                        "answer_idx": ai,
                        "difficulty": "hard",
                    },
                )
            )
        return out

    if dataset == "musr":
        # MuSR (TAUR-Lab/MuSR) — multistep soft reasoning over a narrative
        # (murder mysteries / object placements / team allocation). `subset`
        # picks the split; rows carry a narrative+question, a `choices` list,
        # and `answer_index`. Moderate-hard, exact MC grading (no judge).
        config = subset or "murder_mysteries"
        ds = load_dataset("TAUR-Lab/MuSR", split=config)
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            choices = row.get("choices")
            # `choices` is stored as a stringified python list in some revisions.
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except (ValueError, SyntaxError):
                    choices = [c.strip() for c in choices.strip("[]").split(",")]
            ai = int(row.get("answer_index", -1))
            if not choices or ai < 0 or ai >= len(choices):
                continue
            narrative = (row.get("narrative") or "").strip()
            question = (row.get("question") or "").strip()
            stem = f"{narrative}\n\n{question}" if narrative else question
            out.append(
                TaskSample(
                    task=_format_mc_question(stem, list(choices)),
                    gold_answer=chr(ord("A") + ai),
                    family="musr",
                    task_id=f"musr_{config}_{i}",
                    metadata={"subset": config, "choices": list(choices),
                              "answer_idx": ai, "difficulty": "hard"},
                )
            )
        return out

    if dataset == "reclor":
        # ReClor (metaeval/reclor) — LSAT/GMAT logical-reasoning MC. Use the
        # `validation` split (the `test` split ships hidden labels = -1). Each
        # row: context + question + `answers` (4 options) + `label` (gold idx).
        ds = load_dataset("metaeval/reclor", split=split if split != "train" else "validation")
        if n is not None:
            ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out = []
        for i, row in enumerate(ds):
            answers = list(row.get("answers") or [])
            lab = int(row.get("label", -1))
            if not answers or lab < 0 or lab >= len(answers):
                continue
            stem = f"{(row.get('context') or '').strip()}\n\n{(row.get('question') or '').strip()}"
            out.append(
                TaskSample(
                    task=_format_mc_question(stem, answers),
                    gold_answer=chr(ord("A") + lab),
                    family="reclor",
                    task_id=f"reclor_{row.get('id_string', i)}",
                    metadata={"choices": answers, "answer_idx": lab,
                              "difficulty": "hard"},
                )
            )
        return out

    raise NotImplementedError(f"dataset={dataset!r} not wired yet")


# ---------------------------------------------------------------------------
# Mixed pool for SFT (11-source diversity — classical 7 + agent-system 4)
# ---------------------------------------------------------------------------

# Default mix: per-source target counts for the SFT task pool.
# Tweak via `load_mixed`'s `ratios` arg if you want a different composition.
#
# Note on caps: `load_huggingface(..., n=N)` returns up to min(N, dataset_size).
# HumanEval has only 164 rows total, so it always returns 164 here.
# MBPP sanitized has ~974 rows; ARC-Challenge train ~1.1k; BBH per-task ~250.
DEFAULT_SFT_MIX = {
    # ~11.5K diverse task wordings. Balance is not the goal — each task is
    # randomly paired with a NamedArch so what matters is wording variety,
    # not per-source counts. Includes agent-system benches (browsecomp /
    # hle / phybench / livecodebench) so the head's prior covers
    # research-heavy and web-aware-code task styles before GRPO.
    "gsm8k":          5000,  # math word problems (full train ~7473)
    "mmlu":           2967,  # broad-knowledge MC (full ~99k; 3k is plenty)
    "arc":            1119,  # ARC-Challenge train, max
    "math":            500,  # MATH-500 (max under our loader)
    "phybench":        400,  # physics olympiad (symbolic LaTeX answers)
    "hle":             500,  # Humanity's Last Exam (frontier reasoning)
    "browsecomp":      200,  # multi-step web research
    "livecodebench":   300,  # live competitive coding
    "bbh":             250,  # BBH single subset, max under our loader
    "humaneval":       164,  # full HumanEval, max
    "mbpp":            120,  # MBPP sanitized, max under our loader
}


# Curated subset of BBH tasks that are pure-text (no diagrams), reasoning-
# heavy, and gradeable with the multi-choice / numeric / string matchers.
# Excludes diagram-dependent ones (dyck_languages, geometric_shapes etc.)
# and ones with non-standard grading (movie_recommendation, word_sorting).
BBH_DIVERSE_SUBSETS = (
    "causal_judgement",
    "disambiguation_qa",
    "formal_fallacies",
    "logical_deduction_seven_objects",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
)


def load_bbh_mixed(
    n: int,
    subsets: tuple[str, ...] = BBH_DIVERSE_SUBSETS,
    seed: int = 0,
) -> list[TaskSample]:
    """Sample `n` tasks evenly across multiple BBH subsets.

    BBH has 27 subtasks of varying flavor — `subsets` defaults to a curated
    pure-text reasoning slice (no diagrams, gradeable by string match).
    Per-subtask cap is roughly `n / len(subsets)`. Returns a shuffled list.
    """
    per = max(1, n // len(subsets))
    out: list[TaskSample] = []
    for sub in subsets:
        try:
            ts = load_huggingface("bbh", subset=sub, n=per, seed=seed)
            out.extend(ts)
        except Exception as e:
            print(f"[load_bbh_mixed] WARN skipping {sub}: {e}")
            continue
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


def load_mixed(
    ratios: dict[str, int] | None = None,
    *,
    seed: int = 0,
    sft_split: str = "train",
) -> list[TaskSample]:
    """Build a mixed SFT task pool from multiple sources.

    `ratios`: maps dataset name → desired count. Defaults to DEFAULT_SFT_MIX
    (~11.5K rows across 11 sources). Each source is loaded independently
    then concatenated and shuffled. If a source has fewer rows than
    requested, we just return all of them (no oversampling).

    NOTE: `bbh` and `arc` are loaded with their default sub-config
    (BBH=logical_deduction_three_objects, ARC=ARC-Challenge). For
    BBH-internal diversity across the 27 subsets, call `load_bbh_mixed`
    directly instead of routing through `load_mixed`.
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
    "BBH_DIVERSE_SUBSETS",
    "load_huggingface",
    "load_local_synthetic",
    "load_mixed",
    "load_bbh_mixed",
    "split_pools",
]
