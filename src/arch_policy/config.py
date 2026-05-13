"""Global configuration for the architecture-policy project.

v3 design (typed heads + ReAct agents + dynamic stopping via Synth):

  - The head outputs 4 *typed* tensors describing 4 independent distributions:
      * gates  g[i]      : Bernoulli  — which agent slots are active
      * roles  Q[i]      : Categorical(R) — role assigned to slot i
      * edges  P[i,j]    : Bernoulli  — directed comm graph (computed from
                                         latent + role-pair SBM)
      * seq    s[i]      : real score — Plackett-Luce permutation over actives

  - There is NO `weights`, NO `m_max`, NO fixed `k_repeat_max` exposed to the
    head. The head learns 4 distributions; the executor handles the rest.

  - Execution: an "Agent" is a ReAct sub-loop with role-specific prompt and
    tools. After each big round (one pass through the PL-sampled sequence),
    a Synth LLM call decides ANSWER:<x> or CONTINUE. We never expose hard
    budgets to the policy — only engineering safety caps catch runaway loops.

  - Loss: SFT uses typed losses (BCE + CE + BCE + PL-NLL). GRPO uses typed
    log_pi (Bern + Cat + Bern + PL) plus an entropy bonus. NO KL.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    """Sizing of the architecture distribution.

    The head produces tensors of these shapes:
      gate_logits  [n_max]            (sigmoid → Bernoulli)
      role_logits  [n_max, k_roles]   (softmax → Categorical)
      edge_logits  [n_max, n_max]     (computed via U^T M U + Q^T B Q + b0)
      seq_scores   [n_max]            (Plackett-Luce score per slot)
    """

    n_max: int = 6           # max number of agent slots
    k_roles: int = 7         # 7 roles, see role_names
    d_latent: int = 32       # per-agent latent dim used by bilinear edge decoder

    # Role names — index in this list == role id used everywhere.
    # Chosen to be a minimal complete cover of the cognitive functions
    # appearing in 17 popular multi-agent harnesses (see README.md).
    role_names: tuple[str, ...] = (
        "Planner",      # 0 — decompose the task into sub-tasks
        "Solver",       # 1 — produce candidate answers
        "Critic",       # 2 — challenge proposed solutions
        "Verifier",     # 3 — rule-based / re-derivation check
        "Refiner",      # 4 — integrate / edit / synthesize candidates
        "Researcher",   # 5 — gather information (multi-step search)
        "ToolUser",     # 6 — call external tools (python, api, ...)
    )

    # Engineering safety caps — NOT exposed to the head, normally never reached.
    # The reward function's cost terms (-λ_c·#calls) are what teaches the system
    # to be efficient. These caps just prevent a runaway bug from burning the
    # API budget.
    safety_max_big_rounds: int = 20      # outer loop hard stop
    safety_max_inner_rounds: int = 8     # ReAct loop per agent hard stop
    safety_max_tokens_per_call: int = 2048


@dataclass(frozen=True)
class ModelSpec:
    """Model selection. Override via env vars or CLI flags as needed."""

    # The HEAD is the only thing we train.
    head_model: str = "Qwen/Qwen3-0.6B"
    head_dtype: str = "bfloat16"
    head_device: str = "cuda:0"

    # Agents and Synth use the SAME worker (DeepSeek API / OpenAI-compatible).
    # No local agent inference is required for the v3 plan.
    worker_model: str = "deepseek-chat"
    worker_base_url: str = "https://api.deepseek.com/v1"


@dataclass(frozen=True)
class TrainSpec:
    """Default training hyperparameters."""

    # ---- SFT (Stage 1) ------------------------------------------------------
    sft_epochs: int = 5
    sft_lr: float = 5e-5
    sft_batch_size: int = 8
    sft_grad_accum: int = 2
    sft_warmup_ratio: float = 0.05
    sft_save_every_n_steps: int = 50
    sft_max_steps: int | None = None

    # Per-head loss weights (typed losses).
    sft_w_gate: float = 1.0
    sft_w_role: float = 1.0
    sft_w_edge: float = 1.0
    sft_w_seq: float = 1.0

    # ---- GRPO (Stage 2) -----------------------------------------------------
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_batch_size: int = 4
    grpo_clip_eps: float = 0.2

    # No KL term. Entropy bonus only.
    grpo_entropy_weight: float = 0.01

    # ---- Reward shaping coefficients ---------------------------------------
    # All in units that make a "correct" answer worth ~1.0 dominantly,
    # and costs land in the [-0.3, 0] region for normal architectures.
    reward_lambda_agent: float = 0.03      # cost per active agent
    reward_lambda_edge: float = 0.005      # cost per active edge
    reward_lambda_call: float = 0.01       # cost per LLM call (incl. inner loop + synth)
    reward_lambda_token: float = 0.0       # off by default; enable for token-cost ablation


# ---------------------------------------------------------------------------
# Singletons used everywhere for convenience.
# Tests should construct fresh dataclasses instead of mutating these.
# ---------------------------------------------------------------------------
ARCH = ArchSpec()
MODEL = ModelSpec()
TRAIN = TrainSpec()
