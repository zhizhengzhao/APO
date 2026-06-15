"""Per-bench plugin registry. Add a new bench = one file + one line."""

from __future__ import annotations

from .base import BenchAdapter, GRADE_ERROR
from .category import CategoryBench


_REGISTRY: dict[str, BenchAdapter] = {
    # The three released task-category corpora (the measurement axis); each is
    # graded by rule (no judge). See bench/category.py.
    "cat_code": CategoryBench("cat_code", needs_judge=False),
    "cat_math": CategoryBench("cat_math", needs_judge=False),
    "cat_reasoning": CategoryBench("cat_reasoning", needs_judge=False),
    # Shared SFT pool: union of all category train splits, used ONLY to train
    # the single shared SFT prior (one entropy-rich warm-start for every RL
    # run). Shipped pre-built as data/categories/mixed.jsonl.
    "cat_mixed": CategoryBench("cat_mixed", needs_judge=False),
}


def get(name: str) -> BenchAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown bench {name!r}. Available: {sorted(_REGISTRY)}. "
            f"Add new benches at `src/arch_policy/bench/<name>.py` "
            f"and register here."
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def make_reward_fn(adapter: BenchAdapter, judge=None):
    """Adapt `BenchAdapter.grade` for `grpo.train_grpo(reward_fn=...)`.
    Stashes the judge audit on `trace.extra[<bench>_audit]`."""
    from ..reward import RewardBreakdown, compute_reward

    def reward_fn(trace, gold, spec, task_sample=None):
        if task_sample is None:
            return compute_reward(trace, gold, spec, task_sample=None)
        score, audit = adapter.grade(trace.final_answer, task_sample, judge=judge)
        if trace.extra is None:
            trace.extra = {}
        trace.extra.setdefault(f"{adapter.name}_audit", []).append(audit)
        # CRITICAL: judge-side failure is INFRA noise, not architecture
        # quality. Without this bump, GRADE_ERROR=(0.0, ...) would slip
        # past eng_valid (which only checks n_api_errors) and silently
        # feed "wrong answer" into the GRPO gradient. The mask treats
        # it as engineering-invalid, so it contributes no PG signal.
        if isinstance(audit, dict) and audit.get("judge_path") == "error":
            trace.n_api_errors = int(getattr(trace, "n_api_errors", 0)) + 1
        return RewardBreakdown(
            correctness=float(score),
            n_active=int(trace.arch.active_mask.sum().item()),
            n_edges=int(trace.arch.edges.sum().item()),
            n_calls=trace.n_llm_calls,
            total=float(score),
        )

    return reward_fn


__all__ = ["BenchAdapter", "GRADE_ERROR", "get", "available", "make_reward_fn"]
