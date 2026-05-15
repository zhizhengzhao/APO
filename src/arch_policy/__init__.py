"""arch_policy — learnable multi-agent architecture distributions (v3).

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
    DECOMPOSER,
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
    core_library,
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
    random_archs,
    sample_arch,
    sample_pl,
)
from .baselines import BASELINE_REGISTRY, get_baseline
from .config import ARCH, MODEL, TRAIN, ArchSpec, ModelSpec, TrainSpec
from .executor import (
    Agent,
    AgentMessage,
    AgentTurnOutput,
    ExecutionTrace,
    MockWorker,
    MultiAgentExecutor,
    OpenAIWorker,
    Synth,
    SynthVerdict,
    TOOLS,
    Worker,
    WorkerOutput,
    call_tool,
    heuristic_extract,
    parse_tool_call,
)
from .reward import RewardBreakdown, compute_reward, grade_answer
from .utils import seed_all


# ---- lazy attribute access for heavy deps ---------------------------------

_LAZY_HEAD = ("ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits")
_LAZY_TRAINING = (
    "save_head_checkpoint", "load_head_checkpoint",
    "train_sft", "evaluate_sft",
    "sft_loss_batch", "sft_loss_single",
    "GRPOBatch", "entropy_typed", "grpo_step", "train_grpo",
)
_LAZY_EXECUTOR = ("HFWorker",)
_LAZY_DATA = (
    "DEFAULT_SFT_MIX",
    "SFTArchDataset", "SFTSample",
    "TaskSample", "load_local_synthetic", "load_huggingface", "load_mixed", "split_pools",
)


def __getattr__(name: str):
    if name in _LAZY_HEAD:
        from . import head as _head_mod
        return getattr(_head_mod, name)
    if name in _LAZY_TRAINING:
        from . import training as _train_mod
        return getattr(_train_mod, name)
    if name in _LAZY_EXECUTOR:
        from . import executor as _ex_mod
        return getattr(_ex_mod, name)
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
    "canonical_library", "core_library", "encode_library", "encode_named_arch",
    "family_of", "full_library", "full_mesh", "imperfect_library", "random_archs",
    "sample_arch", "sample_pl",
    "log_prob_edges", "log_prob_gates", "log_prob_joint",
    "log_prob_pl", "log_prob_roles",
    # role / family metadata
    "ANSWERER_ROLES", "CANONICAL_FAMILIES",
    "ROLE_TO_ID", "ROLE_POOL",
    "PLANNER", "DECOMPOSER", "SOLVER", "CRITIC",
    "VERIFIER", "REFINER", "RESEARCHER", "TESTER",
    # executor
    "Agent", "AgentMessage", "AgentTurnOutput",
    "ExecutionTrace", "MockWorker", "MultiAgentExecutor",
    "OpenAIWorker", "Synth", "SynthVerdict", "TOOLS",
    "Worker", "WorkerOutput",
    "call_tool", "heuristic_extract", "parse_tool_call",
    # head (lazy)
    "ArchitectureHead", "HeadConfig", "load_tokenizer", "to_arch_logits",
    # training (lazy)
    "GRPOBatch", "entropy_typed",
    "evaluate_sft", "grpo_step", "train_grpo",
    "load_head_checkpoint", "save_head_checkpoint",
    "sft_loss_batch", "sft_loss_single", "train_sft",
    # data (lazy)
    "DEFAULT_SFT_MIX",
    "SFTArchDataset", "SFTSample", "TaskSample",
    "load_huggingface", "load_local_synthetic", "load_mixed", "split_pools",
    # reward
    "RewardBreakdown", "compute_reward", "grade_answer",
    # baselines
    "BASELINE_REGISTRY", "get_baseline",
    # executor extra
    "HFWorker",
    # utils
    "seed_all",
]
