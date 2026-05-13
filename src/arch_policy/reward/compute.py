"""Reward computation for an architecture run.

Composite reward:
    R = correctness  -  λ_a * #active_agents  -  λ_e * #edges  -  λ_c * #LLM_calls

The `correctness` term is computed by a task-specific grader (see grade.py),
which can be either numeric, MATH-style boxed, or HumanEval exec-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import TRAIN, TrainSpec
from ..executor.multi_agent import ExecutionTrace
from .grade import grade as _grade_dispatch
from .grade import grade_numeric


@dataclass
class RewardBreakdown:
    correctness: float
    cost_agent: float
    cost_edge: float
    cost_call: float
    total: float


def grade_answer(prediction: str, gold: str) -> float:
    """Backwards-compatible numeric/string grader. Use `grade.grade()` for full dispatch."""
    return grade_numeric(prediction, gold)


def compute_reward(
    trace: ExecutionTrace,
    gold_answer: str,
    train_spec: TrainSpec | None = None,
    task_sample: Optional[object] = None,  # `data.tasks.TaskSample` if available
) -> RewardBreakdown:
    """Compute the composite reward.

    If `task_sample` is provided we dispatch to the family-specific grader
    (handles MATH boxed answers, HumanEval exec, etc.). Otherwise we fall
    back to plain numeric/string match against `gold_answer`.
    """
    if train_spec is None:
        train_spec = TRAIN
    if task_sample is not None:
        correct = _grade_dispatch(trace.final_answer, task_sample)
    else:
        correct = grade_numeric(trace.final_answer, gold_answer)
    n_active = int(trace.arch.active_mask.sum().item())
    n_edges = int(trace.arch.edges.sum().item())
    n_calls = trace.n_llm_calls
    cost_agent = train_spec.reward_lambda_agent * n_active
    cost_edge = train_spec.reward_lambda_edge * n_edges
    cost_call = train_spec.reward_lambda_call * n_calls
    total = correct - cost_agent - cost_edge - cost_call
    return RewardBreakdown(
        correctness=correct,
        cost_agent=cost_agent,
        cost_edge=cost_edge,
        cost_call=cost_call,
        total=total,
    )


__all__ = ["RewardBreakdown", "compute_reward", "grade_answer"]
