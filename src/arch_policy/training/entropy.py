"""Per-component-weighted typed entropy bonus for GRPO.

The head emits 4 typed distributions (gate / role / edge / seq); each has
a different natural entropy range. Without per-component weights, the
entropy bonus `α·Σ H_c` is ~43% controlled by edge alone and tiny on
seq. `DEFAULT_ENTROPY_WEIGHTS` rescales each so a uniform target
contributes a comparable absolute bonus.

Calibration note (verify after any change to ArchSpec / weights):

  At a uniform-prior init the bonus magnitude works out to:
      gate ≈ 0.0104   role ≈ 0.0052
      edge ≈ 0.0025   seq  ≈ 0.0090
      Σ    ≈ 0.027

  Typical |advantage × log_pi| per sample is order 10-30, so the
  default `grpo_entropy_weight = 1.0` gives an entropy contribution
  of ~0.1% of the policy-gradient loss. This is a *soft* regularizer
  by design, not the main signal. If entropy collapses
  on a real run and want a stronger anti-collapse force, scale
  `--entropy_weight` to 30-100 (or rescale `DEFAULT_ENTROPY_WEIGHTS`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import ARCH, ArchSpec


# Per-component entropy weights. Formula: α_c = base × priority_c / max_H_c
# where base=0.01. Tuned for ArchSpec defaults (n_max=6, k_roles=8).
# Recompute if those change dramatically.
DEFAULT_ENTROPY_WEIGHTS: dict[str, float] = {
    "gate": 0.0025,    # max H ≈ 6·ln2  = 4.16
    "role": 0.00083,   # max H ≈ 6·ln8  = 12.48
    "edge": 0.00048,   # max H ≈ 30·ln2 = 20.79
    "seq":  0.005,     # max H ≈ ln6    = 1.79
    "model": 0.0033,   # max H ≈ 6·ln4  = 8.32 (n_models=4); absent if 1 model
}


def entropy_typed(
    head_out: dict[str, torch.Tensor],
    spec: ArchSpec | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-component-weighted sum of typed entropies, batch-averaged.

    Returns Σ_c (w_c × mean_batch(H_c)) — a single scalar; callers add as
    `loss -= entropy_typed(...)`. With DEFAULT_ENTROPY_WEIGHTS each typed
    head gets a comparable absolute bonus regardless of its max entropy.

    Per-slot entropies (role / edge) are MASKED by the soft-active mask
    sigmoid(gate_logits). Without this mask, >90% of the entropy budget
    would be spent on INACTIVE slots that never run, preventing the head
    from learning sharp preferences on active slots.

    CRITICAL: the soft mask `g_p` is `.detach()`ed everywhere it acts as
    a *weight* on another entropy term. Without detach, the role / edge
    entropy contributions flow gradient back to gate_logits with a
    uniformly positive sign (h_per_slot ≥ 0, σ' ≥ 0 ⇒ ∂(loss = -H)/∂g ≤ 0
    ⇒ SGD monotonically pushes gate UP every step), fighting
    shaped_advantage's "fewer-active is cheaper" signal and biasing the
    policy toward all-active architectures. Numerically verified May 2026.

    For PL we use the marginal entropy of softmax(seq_scores) as a
    surrogate; exact PL entropy is O(N!).
    """
    if spec is None:
        spec = ARCH
    if weights is None:
        weights = DEFAULT_ENTROPY_WEIGHTS
    N = spec.n_max

    # gate: H(Bern(p)) — ALL slots count (the "how many to activate"
    # decision is a global property; no masking).
    g_logits = head_out["gate_logits"]                    # [B, N]
    g_p = torch.sigmoid(g_logits)
    h_g = -(
        g_p * torch.log(g_p.clamp_min(1e-9))
        + (1 - g_p) * torch.log((1 - g_p).clamp_min(1e-9))
    ).sum(dim=-1)                                         # [B]

    # role: per-slot Categorical, MASKED by detached P(slot active).
    log_q = F.log_softmax(head_out["role_logits"], dim=-1)   # [B, N, R]
    q = log_q.exp()
    h_q_per_slot = -(q * log_q).sum(dim=-1)                  # [B, N]
    h_q = (h_q_per_slot * g_p.detach()).sum(dim=-1)          # [B]

    # edge: per-pair Bernoulli, MASKED by detached P(i active) * P(j active).
    e_logits = head_out["edge_logits"]                       # [B, N, N]
    e_p = torch.sigmoid(e_logits)
    h_e_full = -(
        e_p * torch.log(e_p.clamp_min(1e-9))
        + (1 - e_p) * torch.log((1 - e_p).clamp_min(1e-9))
    )                                                        # [B, N, N]
    eye = torch.eye(N, device=h_e_full.device, dtype=torch.bool).unsqueeze(0)
    h_e_full = h_e_full.masked_fill(eye, 0.0)
    pair_p = (g_p.unsqueeze(-1) * g_p.unsqueeze(-2)).detach()
    h_e = (h_e_full * pair_p).sum(dim=(-1, -2))              # [B]

    # seq: marginal entropy of softmax(seq_scores) as a PL surrogate.
    log_s = F.log_softmax(head_out["seq_scores"], dim=-1)
    s = log_s.exp()
    h_s = -(s * log_s).sum(dim=-1)                           # [B]

    total = (
        weights["gate"] * h_g.mean()
        + weights["role"] * h_q.mean()
        + weights["edge"] * h_e.mean()
        + weights["seq"]  * h_s.mean()
    )

    # model: per-slot Categorical, MASKED by detached P(slot active).
    # Present only in multi-model runs (n_models > 1).
    m_logits = head_out.get("model_logits")
    if m_logits is not None and m_logits.shape[-1] > 1:
        log_m = F.log_softmax(m_logits, dim=-1)              # [B, N, n_models]
        m = log_m.exp()
        h_m_per_slot = -(m * log_m).sum(dim=-1)              # [B, N]
        h_m = (h_m_per_slot * g_p.detach()).sum(dim=-1)      # [B]
        total = total + weights["model"] * h_m.mean()

    return total


__all__ = ["DEFAULT_ENTROPY_WEIGHTS", "entropy_typed"]
