"""Shared helpers for the runner scripts. Kept tiny — anything that
belongs in the library proper lives under `src/arch_policy/…`."""

from __future__ import annotations

from typing import Optional


def _build(
    kind: str, model: str,
    timeout: float, temperature: float, thinking: bool,
):
    """Construct a Worker by vendor kind + concrete model id. Snaps a
    cross-vendor `--model` default to the chosen vendor's flagship."""
    if kind == "mock":
        from arch_policy import MockWorker
        return MockWorker(fake_answer=model or "ok")
    if kind == "gpugeek":
        from arch_policy import GpuGeekWorker
        return GpuGeekWorker(model=model, timeout=timeout,
                             temperature=temperature, thinking=thinking)
    if kind == "deepseek":
        from arch_policy import DeepSeekWorker
        # Vendor mismatch — raise instead of silently snapping, so the
        # caller notices a `--worker deepseek --worker_model Vendor3/...`
        # typo (caller likely meant `--worker gpugeek`).
        if model.startswith("Vendor") or model.startswith("qwen"):
            raise ValueError(
                f"--worker deepseek + --worker_model {model!r} is a "
                f"vendor mismatch. Pick the right --worker for that "
                f"model (e.g. `--worker gpugeek` for Vendor3/..., "
                f"`--worker qwen` for qwen3.7-max)."
            )
        return DeepSeekWorker(model=model, timeout=timeout,
                              temperature=temperature, thinking=thinking)
    if kind == "qwen":
        from arch_policy import QwenWorker
        if model.startswith("Vendor") or model.startswith("deepseek"):
            raise ValueError(
                f"--worker qwen + --worker_model {model!r} is a "
                f"vendor mismatch. Pick the right --worker for that "
                f"model (e.g. `--worker gpugeek` for Vendor*/..., "
                f"`--worker deepseek` for deepseek-v4-*)."
            )
        return QwenWorker(model=model, timeout=timeout,
                          temperature=temperature, thinking=thinking)
    raise ValueError(f"unknown worker kind {kind!r}")


def build_worker(args):
    """Build the per-agent Worker from CLI args."""
    if args.worker == "mock":
        from arch_policy import MockWorker
        return MockWorker(fake_answer=args.mock_answer)
    return _build(
        args.worker, args.worker_model,
        timeout=args.worker_timeout,
        temperature=args.worker_temperature,
        thinking=args.worker_thinking,
    )


# Per-model in-flight caps for the 3-tier single-vendor Qwen pool
# (flash/plus/max, all via one DashScope key). Probed 2026-06-03: each
# tier alone is clean to ~128, but the THREE share ONE account quota whose
# clean band is ~128 total in-flight (192→p50 7s, 384→p50 17s — accepted,
# no 429). So the caps SUM to ~128 to stay in the <1s band; the per-model
# semaphore makes excess calls WAIT, not fail, so reward reflects model
# quality not throttling luck.
MODEL_CONCURRENCY = {
    # Halved (24/20/20 = 64/process) because we run TWO trainings at once
    # sharing one DashScope key whose clean band is ~128 total in-flight;
    # two processes => combined ~48/40/40 = 128, staying in the <1s band.
    "qwen3.6-flash": 24,
    "qwen3.6-plus": 20,
    "qwen3.7-max": 20,
}
_DEFAULT_MODEL_CONCURRENCY = 16   # conservative for any unlisted model


def build_worker_pool(model_names, *, timeout, temperature, thinking):
    """Build {model_name: ConcurrencyLimitedWorker(QwenWorker)} for the
    DashScope-served model pool.

    The 3-tier pool (qwen3.6-flash / qwen3.6-plus / qwen3.7-max) is reached
    through the one DashScope OpenAI-compat endpoint + key, differing only by
    model id — so every pool entry is a QwenWorker, wrapped in a per-model
    semaphore at its share of the account's clean-concurrency budget. Used for
    the per-agent model-selection dimension (ArchSpec.n_models>1); these are
    three capability/cost tiers of one family, so the model dimension is a
    clean "capability allocation" axis.
    """
    from arch_policy import ConcurrencyLimitedWorker, QwenWorker
    pool = {}
    for m in model_names:
        inner = QwenWorker(model=m, timeout=timeout,
                           temperature=temperature, thinking=thinking)
        cap = MODEL_CONCURRENCY.get(m, _DEFAULT_MODEL_CONCURRENCY)
        pool[m] = ConcurrencyLimitedWorker(inner, max_concurrency=cap)
    return pool


def build_judge(args) -> Optional[object]:
    """Build the LLM-judge Worker. None if `--judge_model` is unset —
    the adapter's rule grader is then used as fallback."""
    model = getattr(args, "judge_model", None)
    if not model:
        return None
    return _build(
        getattr(args, "judge", "gpugeek"), model,
        timeout=getattr(args, "judge_timeout", 120.0),
        temperature=0.0, thinking=False,
    )
