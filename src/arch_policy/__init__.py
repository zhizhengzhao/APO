"""arch_policy — learnable multi-agent architecture distributions.

Top-level public API. Heavy dependencies (transformers, datasets) are
imported lazily so that pure-architecture tests can run without them.

Example::

    from arch_policy import ARCH, sample_arch, full_library
    # ↑ no transformers needed

    from arch_policy import ArchitectureHead          # ↓ requires transformers
"""

from __future__ import annotations

from .architecture import (
    ANSWERER_ROLES,
    ArchLogits,
    ArchTargets,
    CANONICAL_FAMILIES,
    ConcreteArch,
    CRITIC,
    EXPERT,
    NamedArch,
    PLANNER,
    REFINER,
    RESEARCHER,
    ROLE_POOL,
    ROLE_TO_ID,
    SOLVER,
    TESTER,
    VERIFIER,
    active_pair_mask,
    canonical_library,
    default_inject_pool,        # 82-arch canonical pool (42 families)
    encode_library,
    encode_named_arch,
    family_of,
    full_library,
    full_mesh,
    imperfect_library,
    log_prob_edges,
    log_prob_gates,
    log_prob_joint,
    log_prob_pl,
    log_prob_roles,
    named_arch_to_concrete,
    random_archs,
    sample_arch,
    sample_inject_family_stratified,
    sample_pl,
)
from .baselines import BASELINE_REGISTRY, get_baseline
from .config import ARCH, MODEL, TRAIN, ArchSpec, ModelSpec, TrainSpec
from .executor import (
    Agent,
    AgentMessage,
    AgentTurnOutput,
    ConcurrencyLimitedWorker,
    DeepSeekWorker,
    ExecutionTrace,
    GpuGeekWorker,
    MockWorker,
    QwenWorker,
    MultiAgentExecutor,
    ROLE_TOOL_POOLS,
    Synth,
    SynthVerdict,
    TOOLS,
    Worker,
    WorkerOutput,
    allowed_tools_for,
    call_tool,
    heuristic_extract,
    parse_action,
)
from .reward import RewardBreakdown, compute_reward, grade_answer, grade_multiple_choice
from . import bench  # plugin registry — `bench.get("cat_code")` etc.


def seed_all(seed: int = 0) -> None:
    """Set python / numpy / torch RNGs in one call."""
    import os
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---- lazy attribute access for heavy deps ---------------------------------

_LAZY_HEAD = ("ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits")
_LAZY_TRAINING = (
    "save_head_checkpoint", "load_head_checkpoint",
    "train_sft", "evaluate_sft",
    "sft_loss_batch", "sft_loss_single",
    "GRPOBatch", "DEFAULT_ENTROPY_WEIGHTS",
    "entropy_typed", "grpo_step", "shaped_advantage", "train_grpo",
)
_LAZY_DATA = (
    "BBH_DIVERSE_SUBSETS", "DEFAULT_SFT_MIX",
    "SFTArchDataset", "SFTSample",
    "TaskSample",
    "load_bbh_mixed", "load_local_synthetic", "load_huggingface", "load_mixed",
    "split_pools",
)


def __getattr__(name: str):
    if name in _LAZY_HEAD:
        from . import head as _head_mod
        return getattr(_head_mod, name)
    if name in _LAZY_TRAINING:
        from . import training as _train_mod
        return getattr(_train_mod, name)
    if name in _LAZY_DATA:
        from . import data as _data_mod
        return getattr(_data_mod, name)
    raise AttributeError(f"module 'arch_policy' has no attribute {name!r}")


__all__ = [
    # config
    "ARCH", "MODEL", "TRAIN", "ArchSpec", "ModelSpec", "TrainSpec",
    # architecture
    "ArchLogits", "ArchTargets", "ConcreteArch", "NamedArch",
    "active_pair_mask",
    "canonical_library",
    "default_inject_pool",
    "encode_library", "encode_named_arch",
    "family_of", "full_library", "full_mesh", "imperfect_library",
    "named_arch_to_concrete", "random_archs",
    "sample_arch", "sample_inject_family_stratified", "sample_pl",
    "log_prob_edges", "log_prob_gates", "log_prob_joint",
    "log_prob_pl", "log_prob_roles",
    # role / family metadata
    "ANSWERER_ROLES", "CANONICAL_FAMILIES",
    "ROLE_TO_ID", "ROLE_POOL",
    "PLANNER", "EXPERT", "SOLVER", "CRITIC",
    "VERIFIER", "REFINER", "RESEARCHER", "TESTER",
    # executor
    "Agent", "AgentMessage", "AgentTurnOutput",
    "ConcurrencyLimitedWorker", "DeepSeekWorker", "ExecutionTrace",
    "GpuGeekWorker", "MockWorker", "MultiAgentExecutor", "QwenWorker",
    "ROLE_TOOL_POOLS",
    "Synth", "SynthVerdict", "TOOLS", "allowed_tools_for",
    "Worker", "WorkerOutput",
    "call_tool", "heuristic_extract", "parse_action",
    # head (lazy)
    "ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits",
    # training (lazy)
    "DEFAULT_ENTROPY_WEIGHTS", "GRPOBatch", "entropy_typed",
    "evaluate_sft", "grpo_step", "shaped_advantage", "train_grpo",
    "load_head_checkpoint", "save_head_checkpoint",
    "sft_loss_batch", "sft_loss_single", "train_sft",
    # data (lazy)
    "BBH_DIVERSE_SUBSETS", "DEFAULT_SFT_MIX",
    "SFTArchDataset", "SFTSample", "TaskSample",
    "load_bbh_mixed", "load_huggingface", "load_local_synthetic",
    "load_mixed", "split_pools",
    # reward
    "RewardBreakdown", "compute_reward",
    "grade_answer", "grade_multiple_choice",
    # baselines
    "BASELINE_REGISTRY", "get_baseline",
    # bench plugin registry
    "bench",
    # utils
    "seed_all",
]
