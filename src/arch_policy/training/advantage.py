"""Two-tier shaped advantage for GRPO.

See `shaped_advantage` docstring for the full design + properties. Lives
in its own module so `training/grpo.py` reads as orchestration and the
math is reviewable in isolation.
"""

from __future__ import annotations

import torch


# All-wrong fallback: when σ=0 because every sample failed, we still want a
# mild downward push (so the head shifts probability mass away from this
# group of architectures and entropy bonus can lift it elsewhere).
# Magnitude is small so when correct samples eventually appear they
# dominate the gradient.
_ALL_WRONG_PUSH = -0.1


def shaped_advantage(
    correct: torch.Tensor,    # [G, B] 0.0/1.0
    n_calls: torch.Tensor,    # [G, B] int (cost proxy)
    cost_bonus_scale: float = 1.0,
    valid_mask: torch.Tensor | None = None,   # [G, B] bool, default = all True
) -> torch.Tensor:
    """Per-task two-tier advantage with /std normalization (no mean subtract).

    Tier 1 — raw advantage by correctness:
      wrong   → -1
      correct → 1 + bonus, where bonus ∈ [0, cost_bonus_scale]
                via min-max on n_calls within the correct sub-group:
                  cheapest correct → bonus = cost_bonus_scale
                  most expensive    → bonus = 0
                  single correct OR all correct same cost
                                    → bonus = cost_bonus_scale (no rivals)
      With scale=1.0: wrong = -1, correct ∈ [+1, +2].

    Tier 2 — per-task /std normalization (NO mean subtraction), gated by the
    group's correctness composition (NOT by σ — see below):
      all correct (n_correct == n_valid) → adv = 0 uniform
            Architecture choice does not change correctness on this task, so it
            carries NO accuracy signal. Emitting no gradient prevents a cost
            spread inside an all-correct group from driving adv = raw/σ with
            large magnitude; cost shaping lives exclusively inside mixed groups.
      all wrong (n_correct == 0)         → adv = -0.1 uniform  (mild push off
            these architectures so entropy can explore elsewhere)
      mixed (0 < n_correct < n_valid)    → adv = raw / σ
            preserves sign + ordering; amplifies rare-correct groups (small σ);
            cost bonus still differentiates the correct samples here.

    `valid_mask` (optional): per-sample bool mask. Engineering-invalid
    samples (n_api_errors > 0, sentinels) MUST be passed here. Without
    valid_mask the invalid sentinels (treated as wrong) enter σ and
    artificially inflate the /σ amplification on the surviving valid
    samples. Empirically: 3 invalids out of 8 with 1 correct inflates
    adv_correct by ~21%. Numerically verified May 2026.

    Properties:
      - any wrong < 0, any correct > 0 (sign preserved by /σ since σ>0)
      - any correct adv > any wrong adv within the same group
      - within correct: monotonic in n_calls (cheaper → larger adv)
      - bounded magnitude before /σ in [-1, +1+scale]; after /σ controlled
        by grad_clip_norm in the outer training loop
      - cost ONLY enters the correct sub-group; wrong samples ignore n_calls
        (avoids the "fail-fast incentive" of standard cost-shaped rewards)
      - invalid samples (per valid_mask) get adv = 0 regardless
    """
    G, B = correct.shape
    if valid_mask is None:
        valid_mask = torch.ones_like(correct, dtype=torch.bool)
    adv = torch.zeros_like(correct, dtype=torch.float32)
    for b in range(B):
        valid_b = valid_mask[:, b]
        n_valid = int(valid_b.sum().item())
        if n_valid == 0:
            continue  # entire group degenerate → zero gradient

        # `c > 0.5` accommodates future graders returning partial scores.
        # Only valid samples count toward the correctness pool.
        correct_mask = (correct[:, b] > 0.5) & valid_b
        n_correct = int(correct_mask.sum().item())

        # ---- Tier 1: raw advantage by correctness + cost bonus ----
        # Raw stays full length [G] for clean indexing; advantage for
        # invalid slots is zeroed at the end.
        raw = torch.full((G,), -1.0, dtype=torch.float32, device=correct.device)
        if n_correct > 0:
            cn = n_calls[correct_mask, b].float()
            cn_max, cn_min = cn.max(), cn.min()
            if (cn_max - cn_min) < 1e-9:
                bonus = torch.full_like(cn, cost_bonus_scale)
            else:
                bonus = (cn_max - cn) / (cn_max - cn_min) * cost_bonus_scale
            raw[correct_mask] = 1.0 + bonus

        # ---- Tier 2: gate by correctness composition, NOT by σ ----
        # all-correct groups carry no accuracy signal and must NOT emit a cost
        # gradient; only mixed groups get /σ normalization. Cost shaping
        # therefore lives exclusively inside mixed groups.
        if n_correct == n_valid:                      # all correct → no signal
            adv_b = torch.zeros((G,), device=correct.device, dtype=torch.float32)
        elif n_correct == 0:                          # all wrong → mild push
            adv_b = torch.full((G,), _ALL_WRONG_PUSH,
                               device=correct.device, dtype=torch.float32)
        else:                                         # mixed → /σ normalization
            raw_valid = raw[valid_b]
            sigma = raw_valid.std(unbiased=False)
            if sigma < 1e-9:
                # Degenerate mixed (should not occur because correct raw differs
                # from wrong raw), but guard against it.
                adv_b = torch.zeros((G,), device=correct.device,
                                    dtype=torch.float32)
            else:
                adv_b = raw / sigma
        # Zero out invalid-sample advantages (defence in depth: the
        # caller also masks, but doing it here keeps the contract local).
        adv_b = adv_b * valid_b.float()
        adv[:, b] = adv_b
    return adv


__all__ = ["shaped_advantage"]
