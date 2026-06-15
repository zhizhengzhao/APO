"""Training: typed SFT loss + architecture-level GRPO."""

from .grpo import (
    DEFAULT_ENTROPY_WEIGHTS,
    GRPOBatch,
    entropy_typed,
    grpo_step,
    shaped_advantage,
    train_grpo,
)
from .sft import (
    evaluate_sft,
    load_head_checkpoint,
    save_head_checkpoint,
    sft_loss_batch,
    sft_loss_single,
    train_sft,
)

__all__ = [
    "DEFAULT_ENTROPY_WEIGHTS",
    "GRPOBatch",
    "entropy_typed",
    "evaluate_sft",
    "grpo_step",
    "load_head_checkpoint",
    "save_head_checkpoint",
    "sft_loss_batch",
    "sft_loss_single",
    "shaped_advantage",
    "train_grpo",
    "train_sft",
]
