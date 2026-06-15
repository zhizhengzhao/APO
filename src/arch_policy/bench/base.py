"""`BenchAdapter` Protocol — one file per benchmark, registered in
`bench/__init__.py`. Protocol-typed (not ABC) to keep adapter files
literal-data heavy and ceremony-light."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..architecture.library import NamedArch
from ..data.tasks import TaskSample
from ..executor.multi_agent import Worker


@runtime_checkable
class BenchAdapter(Protocol):
    name: str
    subdomains: tuple[str, ...]

    def load_split(
        self,
        split: str,                     # "train" | "test"
        train_ratio: float = 0.8,
        seed: int = 42,
        n_cap: Optional[int] = None,
    ) -> list[TaskSample]: ...

    def get_pool(self) -> list[NamedArch]: ...

    def grade(
        self,
        prediction: str,
        sample: TaskSample,
        judge: Optional[Worker] = None,
    ) -> tuple[float, dict]:
        """Returns (score, audit). Audit schema is adapter-defined and
        is stashed on `trace.extra` by `make_reward_fn`. On hard
        failure (e.g. judge worker exception) return `GRADE_ERROR`."""

    def needs_judge(self) -> bool: ...


# Sentinel for graders that hard-fail. `judge_path == "error"` lets
# analyzers separate engineering noise from real wrong answers.
GRADE_ERROR = (0.0, {"judge_path": "error"})
