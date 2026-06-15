"""Global configuration. Full design in METHOD.md.

The head emits 4-5 typed distributions (the 5th is present only with a
multi-model pool): gates (Bernoulli, which slots active), roles
(Categorical), edges (Bernoulli graph from latent + role-pair SBM), seq
(Plackett-Luce order), and model (Categorical per-agent model choice, only
when n_models > 1). Naming: episode > cycle > turn > step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    """Sizing of the architecture distribution (head output shapes)."""

    n_max: int = 6           # max number of agent slots
    k_roles: int = 8         # 8 roles, see role_names
    d_latent: int = 32       # per-agent latent dim used by bilinear edge decoder

    # Per-agent model-selection dimension. len 1 (default) → the head emits
    # no model_logits and every agent runs on the single configured worker:
    # byte-identical to the 4-head setup. len > 1 → the head learns to assign
    # each agent slot a model from this pool (5th typed distribution), GRPO
    # explores it from a uniform prior (SFT does not supervise it).
    model_names: tuple[str, ...] = ("default",)

    # Role names — list index == role id used everywhere. Role identity is
    # responsibility + tool affordance; tool pools live in
    # `executor/role_tools.py`.
    role_names: tuple[str, ...] = (
        "Planner",      # 0 — high-level decomposition (2-4 sub-steps)
        "Expert",       # 1 — all-tools generalist (single-agent baseline)
        "Solver",       # 2 — produce candidate answers via reasoning
        "Critic",       # 3 — challenge candidates, identify flaws
        "Verifier",     # 4 — independent rule/tool-based correctness check
        "Refiner",      # 5 — integrate / synthesize multiple candidates
        "Researcher",   # 6 — gather facts/equations/definitions
        "Tester",       # 7 — write & run test cases against candidates
    )

    # Engineering safety caps — NOT exposed to the head, normally never
    # reached. Efficiency is driven by GRPO shaped_advantage's cost bonus
    # (prefers fewer n_llm_calls within a correct group); these caps only
    # prevent runaway loops if Synth keeps saying CONTINUE.
    safety_max_cycles: int = 20             # outer loop hard stop
    safety_max_steps:  int = 20             # ReAct loop per-turn hard stop
    safety_max_tokens_per_call: int = 2048

    @property
    def n_models(self) -> int:
        return len(self.model_names)


@dataclass(frozen=True)
class ModelSpec:
    """Model selection. Override via env vars or CLI flags as needed."""

    head_model: str = "Qwen/Qwen3-4B"
    head_dtype: str = "bfloat16"
    head_device: str = "cuda:0"

    # Default worker model (informational; override with --worker_model).
    worker_model: str = "deepseek-v4-flash"


@dataclass(frozen=True)
class TrainSpec:
    """Default training hyperparameters."""

    # ---- SFT (Stage 1) -----------------------------------------------------
    # Per-epoch checkpoint is always written (in addition to the
    # every-N-steps cadence below). Epoch ckpts are the canonical
    # selection units for picking which weight to push to GRPO.
    sft_epochs: int = 3            # production: 3 (prefer GRPO depth over SFT)
    # LR is auto-picked by scripts/02_train_sft.py if --lr unset:
    #   LoRA → 1e-4 (recommended), frozen-head → 5e-5, full FT → 1e-5.
    sft_lr: float = 1e-5
    sft_batch_size: int = 8
    sft_grad_accum: int = 2
    sft_warmup_ratio: float = 0.05
    sft_save_every_n_steps: int = 200
    sft_max_steps: int | None = None

    # Per-typed-head loss weights.
    sft_w_gate: float = 1.0
    sft_w_role: float = 1.0
    sft_w_edge: float = 1.0
    sft_w_seq: float = 1.0
    # 5th dim (per-agent model choice). SFT trains head_M toward a UNIFORM
    # prior via a deterministic max-entropy loss (-mean(log_softmax); no
    # task→model grounding) — GRPO learns the real preferences from reward.
    # Only active when n_models > 1.
    sft_w_model: float = 1.0

    # Label smoothing on Bernoulli (gate / edge) and Categorical (role).
    # Plackett-Luce (seq) is NOT smoothed — it is listwise and stochastic
    # sampling already provides diversity.
    sft_label_smoothing: float = 0.05

    # ---- GRPO (Stage 2) — production budget (matches scripts/03 defaults) --
    grpo_group_size: int = 8       # production: 8 (B×G=128 traces/step)
    grpo_lr: float = 2e-5
    grpo_batch_size: int = 16      # production: 16

    # Global scaler over DEFAULT_ENTROPY_WEIGHTS (per-component weights);
    # 1.0 uses them as-is. Tune up/down on entropy collapse / over-explore.
    grpo_entropy_weight: float = 1.0

    # shaped_advantage cost bonus scale: within a task's correct sub-group,
    # the cheapest correct sample gets +scale bonus on top of the +1 base,
    # the most expensive correct gets +0. Wrong samples ignore n_calls.
    # scale=0.5 means correct ∈ [+1, +1.5] vs wrong = -1 (before /σ): cost is
    # a light tiebreak, correctness stays clearly dominant. (Was 1.0; lowered
    # 2026-06-04 to compress n_calls' influence while keeping the wrong-push
    # symmetric — see the /σ-no-mean-subtract design in advantage.py.)
    advantage_cost_bonus_scale: float = 0.5

    # ---- Tokenization ------------------------------------------------------
    # Head's tokenizer cap for task text (both SFT and GRPO truncate to
    # this). HLE probe: p50=140, p90=522, p99=1762, max=13340 →
    # cap=1024 truncates 3.4% of tasks (74 / 2158). HumanEval / GSM8K /
    # MATH all fit comfortably under 1024. CLI flags can override.
    tokenizer_max_len: int = 1024


# Module-level singletons. Tests should construct fresh dataclasses
# (`dataclasses.replace(ARCH, ...)`) rather than mutating these.
ARCH = ArchSpec()
MODEL = ModelSpec()
TRAIN = TrainSpec()
