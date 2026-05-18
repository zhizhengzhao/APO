"""Global configuration for the architecture-policy project.

Design (typed heads + ReAct agents + dynamic stopping via Synth):

  - The head outputs 4 *typed* tensors describing 4 independent distributions:
      * gates  g[i]      : Bernoulli  — which agent slots are active
      * roles  Q[i]      : Categorical(R) — role assigned to slot i
      * edges  P[i,j]    : Bernoulli  — directed comm graph (computed from
                                         latent + role-pair SBM)
      * seq    s[i]      : real score — Plackett-Luce permutation over actives

  - Execution model (using cycle / turn / step naming):
      An "Agent" is a ReAct sub-loop with role-specific prompt and tools.
      One "cycle" = one full pass through the PL-sampled sequence; each agent
      gets one "turn" per cycle. Inside a turn, the agent runs a ReAct loop
      of "steps" (one LLM call each). After every cycle, a Synth LLM call
      decides ANSWER:<x> or CONTINUE.

  - Loss: SFT uses typed losses (BCE + CE + BCE + PL-NLL) with label
    smoothing on the Bernoulli/Categorical heads. GRPO uses typed log_pi
    (Bern + Cat + Bern + PL) plus an entropy bonus. NO KL.
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
    k_roles: int = 8         # 8 roles, see role_names
    d_latent: int = 32       # per-agent latent dim used by bilinear edge decoder

    # Role names — index in this list == role id used everywhere.
    # Each role is defined by *responsibility*, not by *means*: any role can
    # call any of the available tools (python_exec / sympy_check / web_search)
    # as needed during its ReAct turn. Tools are a capability, not a role.
    role_names: tuple[str, ...] = (
        "Planner",      # 0 — high-level decomposition (2-4 sub-steps)
        "Decomposer",   # 1 — sub-step → atomic actions doable in one turn
        "Solver",       # 2 — produce candidate answers via reasoning
        "Critic",       # 3 — challenge candidates, identify flaws
        "Verifier",     # 4 — independent rule/tool-based correctness check
        "Refiner",      # 5 — integrate / synthesize multiple candidates
        "Researcher",   # 6 — gather facts/equations/definitions
        "Tester",       # 7 — write & run test cases against candidates
    )

    # Engineering safety caps — NOT exposed to the head, normally never reached.
    # Reward cost terms (-λ_c · #calls) teach the system to be efficient;
    # these caps just prevent runaway loops if Synth misbehaves.
    safety_max_cycles: int = 20             # outer loop hard stop
    safety_max_steps: int = 8               # ReAct loop per turn hard stop
    safety_max_tokens_per_call: int = 2048


@dataclass(frozen=True)
class ModelSpec:
    """Model selection. Override via env vars or CLI flags as needed.

    The HEAD is the trained model. In V3.5+ the backbone defaults to
    trainable (full FT or LoRA) — see `ArchitectureHead` for the three modes.
    """

    head_model: str = "Qwen/Qwen3-4B"
    head_dtype: str = "bfloat16"
    head_device: str = "cuda:0"

    # Agents and Synth use a remote OpenAI-compatible chat API (DeepSeek V4
    # by default). The head and worker are completely decoupled.
    worker_model: str = "deepseek-v4-pro"
    worker_base_url: str = "https://api.deepseek.com/v1"


@dataclass(frozen=True)
class TrainSpec:
    """Default training hyperparameters."""

    # ---- SFT (Stage 1) ------------------------------------------------------
    # Per-epoch checkpoint is ALWAYS written (in addition to the
    # every-N-steps cadence below). Epoch ckpts are the canonical
    # selection units for picking which weight to push to GRPO.
    sft_epochs: int = 5
    # Default LR is conservative (good for full FT 9B). Override via CLI:
    #   - frozen-backbone head-only      :  5e-5  (was V3 default)
    #   - LoRA backbone + heads          :  1e-4
    #   - full fine-tune (9B all params) :  1e-5  (this default)
    sft_lr: float = 1e-5
    sft_batch_size: int = 8
    sft_grad_accum: int = 2
    sft_warmup_ratio: float = 0.05
    sft_save_every_n_steps: int = 200
    sft_max_steps: int | None = None

    # Per-head loss weights (typed losses).
    sft_w_gate: float = 1.0
    sft_w_role: float = 1.0
    sft_w_edge: float = 1.0
    sft_w_seq: float = 1.0

    # Label smoothing for Bernoulli (gate / edge) and Categorical (role).
    # Plackett-Luce sequence loss is NOT smoothed — it is listwise and the
    # stochastic sampling already provides natural diversity.
    sft_label_smoothing: float = 0.05      # 0/1 → 0.05/0.95 for Bernoulli;
                                           # one-hot → (1-eps)·oh + eps/R for Cat

    # ---- GRPO (Stage 2) -----------------------------------------------------
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_batch_size: int = 4

    # No KL term. Entropy bonus only.
    grpo_entropy_weight: float = 0.01

    # ---- Tokenization --------------------------------------------------------
    # Both SFT and GRPO read the head's task tokenizer with this max length.
    # Tasks from GSM8K / MATH / HumanEval are well under 512; larger settings
    # are rarely useful and slow the head's frozen backbone forward pass.
    tokenizer_max_len: int = 512

    # ---- Reward shaping coefficients ---------------------------------------
    # Calibrated so a "correct" answer is worth ~1.0 dominantly, and the
    # cost terms land in the [-0.3, 0] region for normal architectures.
    reward_lambda_agent: float = 0.03      # per active agent
    reward_lambda_edge: float = 0.005      # per active edge
    reward_lambda_call: float = 0.01       # per LLM call (inner ReAct + synth)
    reward_lambda_token: float = 0.0       # off by default; for token-cost ablation


# ---------------------------------------------------------------------------
# Singletons used everywhere for convenience.
# Tests should construct fresh dataclasses instead of mutating these.
# ---------------------------------------------------------------------------
ARCH = ArchSpec()
MODEL = ModelSpec()
TRAIN = TrainSpec()
