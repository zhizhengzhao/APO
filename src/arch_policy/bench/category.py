"""Generic task-CATEGORY adapter (the measurement-paper task axis).

One `CategoryBench` class, instantiated once per category and registered in
`bench/__init__.py`. Each reads a frozen `data/categories/<cat>.jsonl`
(produced by `scripts/collect_categories.py`: a balanced 500-problem corpus
pooled from several sources so no single benchmark identity dominates).

Verification is per-problem, routed by each sample's original `family`:
  code (livecodebench/humaneval/mbpp) → execution
  math                                 → symbolic/numeric (grade_math)
  knowledge (gpqa/mmlu)                → multiple-choice letter
  reasoning (bbh/musr/reclor/arc)      → exact-match / multiple-choice letter

The SFT architecture prior is the shared `full_library` for every category
(controls the prior so cross-category architecture differences are the
learned signal, not a category-specific pool).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..architecture.library import full_library
from ..data.tasks import TaskSample
from ..executor.multi_agent import Worker
from ..reward.grade import grade as _grade_dispatch
from .base import GRADE_ERROR
from .llm_judge import judge as _llm_judge

# Families whose answers are open short-text → graded by an LLM judge.
_JUDGE_FAMILIES = {"hle", "browsecomp"}

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "categories"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"category corpus not found: {path}. Run "
            f"`python scripts/collect_categories.py` first."
        )
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


class CategoryBench:
    """A task category backed by a frozen pooled corpus."""

    def __init__(self, name: str, *, needs_judge: bool, data_dir: Optional[Path] = None):
        self.name = name
        self._needs_judge = needs_judge
        self._dir = Path(data_dir) if data_dir else _DATA_DIR
        self.subdomains = ("easy", "medium", "hard", "unknown")

    def load_split(self, split, train_ratio=0.8, seed=42, n_cap=None):
        # train/test split is frozen in the corpus (the `split` field); the
        # train_ratio arg is ignored on purpose so every stage sees the same
        # held-out test set.
        recs = _load_jsonl(self._dir / f"{self.name.replace('cat_', '')}.jsonl")
        out = []
        for r in recs:
            if split in ("train", "test") and r.get("split") != split:
                continue
            out.append(TaskSample(
                task=r["task"], gold_answer=r["gold_answer"],
                family=r["family"], task_id=r["task_id"],
                metadata=dict(r.get("metadata", {}),
                              difficulty=r.get("difficulty", "unknown")),
            ))
        if n_cap is not None:
            out = out[:n_cap]
        return out

    def get_pool(self):
        return full_library()

    def grade(self, prediction, sample, judge=None):
        if sample.family in _JUDGE_FAMILIES and judge is not None:
            return _llm_judge(sample.task, prediction, sample.gold_answer, judge)
        try:
            score = _grade_dispatch(prediction, sample)
        except Exception:  # noqa: BLE001
            return GRADE_ERROR
        return float(score), {"judge_path": f"rule:{sample.family}"}

    def needs_judge(self) -> bool:
        return self._needs_judge


__all__ = ["CategoryBench"]
