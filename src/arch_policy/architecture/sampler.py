"""Sample a concrete architecture from `ArchLogits`.

A concrete architecture is `ConcreteArch`:

  active_mask  : bool [N]      which slots are active
  roles        : long [N]      role id (0 if inactive)
  edges        : bool [N, N]   directed comm graph
  sequence     : long [K]      permutation of active slot ids (K = #active)

Sampling rules:
  - Bernoulli on gates with sigmoid(gate_logits)
  - At least one slot is forced active (the highest-prob slot)
  - Categorical on role_logits per active slot
  - Bernoulli on edges, masked to active pairs, no self-loop
  - Plackett-Luce permutation on seq_scores over active slots

We provide log-probability functions for each typed distribution; GRPO uses
them to assemble the joint log_pi.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import ARCH, ArchSpec
from .spec import ArchLogits, active_pair_mask


@dataclass
class ConcreteArch:
    """A discrete sampled architecture (no batch dim).

    Fields (all length-N where N = n_max):
      active_mask  bool         — which slots are active
      roles        long         — role id per slot (0 if inactive)
      edges        bool [N,N]   — directed comm graph (False on inactive pairs)
      sequence     long [K]     — speaking order (K = #active)
      model        long [N]|None — model id per slot (None ⇒ single-model
                                    setup; executor uses one worker)
    """

    active_mask: torch.Tensor    # bool [N]
    roles: torch.Tensor          # long [N], 0 for inactive (do not interpret)
    edges: torch.Tensor          # bool [N, N], False for inactive pairs
    sequence: torch.Tensor       # long [K]   K = #active
    model: torch.Tensor | None = None   # long [N], 0 for inactive; None if n_models==1

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def n_edges(self) -> int:
        return int(self.edges.sum().item())

    def role_name(self, slot: int, spec: ArchSpec | None = None) -> str:
        if spec is None:
            spec = ARCH
        return spec.role_names[int(self.roles[slot].item())]

    def to_summary(self, spec: ArchSpec | None = None) -> str:
        if spec is None:
            spec = ARCH
        actives = [i for i in range(spec.n_max) if self.active_mask[i].item()]
        roles = [self.role_name(i, spec) for i in actives]
        edges = [
            f"{i}→{j}" for i in actives for j in actives if self.edges[i, j].item()
        ]
        seq = [int(s.item()) for s in self.sequence]
        return f"agents={list(zip(actives, roles))} edges={edges} seq={seq}"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _bernoulli_sample(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """torch.bernoulli with optional generator (the public API doesn't accept one)."""
    u = torch.rand(probs.shape, generator=generator, device=probs.device, dtype=probs.dtype)
    return u < probs


def _categorical_sample(probs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    """Categorical sample per row of `probs` (last dim)."""
    flat = probs.reshape(-1, probs.shape[-1])
    out = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return out.view(*probs.shape[:-1])


def sample_pl(
    seq_scores: torch.Tensor,
    active_idx: torch.Tensor,
    *,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Plackett-Luce sample of a permutation over the *active* slots.

    Args:
        seq_scores: [N] real scores (one per slot, but we only use scores at
                    active positions).
        active_idx: [K] long, slot ids of active slots (K <= N).
        deterministic: if True, produce a deterministic permutation by
                       repeatedly taking argmax over the remaining set.
        generator: optional torch.Generator for reproducibility.

    Returns:
        perm: [K] long — a permutation of `active_idx`.
    """
    if active_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=seq_scores.device)

    remaining = active_idx.tolist()
    perm: list[int] = []
    while remaining:
        sub_scores = seq_scores[remaining]
        if deterministic:
            local = int(torch.argmax(sub_scores).item())
        else:
            probs = F.softmax(sub_scores, dim=-1)
            local = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        perm.append(remaining.pop(local))
    return torch.tensor(perm, dtype=torch.long, device=seq_scores.device)


def sample_arch(
    logits: ArchLogits,
    spec: ArchSpec | None = None,
    *,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> ConcreteArch:
    """Sample one concrete architecture from `logits`.

    `logits` must be unbatched (shapes given by ArchSpec). Caller iterates for
    batches.
    """
    if spec is None:
        spec = ARCH
    logits.validate(spec)

    # Gates: independent Bernoulli per slot, conditioned on "at least one
    # slot is active" (since an empty architecture can't run). We implement
    # the conditional via rejection sampling: keep resampling until ≥1
    # slot is active. `log_prob_gates` mirrors this by subtracting the
    # normalizer log(1 - prod(1 - p_i)).
    g_prob = torch.sigmoid(logits.gate_logits)
    if deterministic:
        active = (g_prob >= 0.5)
        if not active.any():
            active = torch.zeros_like(g_prob, dtype=torch.bool)
            active[int(torch.argmax(g_prob).item())] = True
    else:
        # 64 rejections is enough for any non-pathological g_prob; under
        # typical training the first sample already has ≥1 active. We
        # fall back to argmax only on the (astronomically rare) all-rejected
        # path so the executor never sees an empty arch.
        active = None
        for _ in range(64):
            cand = _bernoulli_sample(g_prob, generator)
            if cand.any():
                active = cand
                break
        if active is None:
            active = torch.zeros_like(g_prob, dtype=torch.bool)
            active[int(torch.argmax(g_prob).item())] = True

    # Roles: Categorical per slot; zero out inactive for clean logs.
    role_probs = F.softmax(logits.role_logits, dim=-1)
    roles = role_probs.argmax(dim=-1) if deterministic else _categorical_sample(role_probs, generator)
    roles = torch.where(active, roles, torch.zeros_like(roles))

    # Edges: Bernoulli, masked to active pairs.
    e_prob = torch.sigmoid(logits.edge_logits)
    edges = (e_prob >= 0.5) if deterministic else _bernoulli_sample(e_prob, generator)
    edges = edges & active_pair_mask(active)

    # Sequence: Plackett-Luce permutation of active slots.
    active_idx = torch.nonzero(active, as_tuple=True)[0]
    sequence = sample_pl(
        logits.seq_scores, active_idx,
        deterministic=deterministic, generator=generator,
    )

    # Model: Categorical per slot — ONLY when the pool has >1 model.
    # n_models==1 draws nothing (model stays None), so single-model runs
    # consume zero extra RNG and are bit-for-bit unchanged.
    model = None
    if logits.model_logits is not None and spec.n_models > 1:
        model_probs = F.softmax(logits.model_logits, dim=-1)
        model = (model_probs.argmax(dim=-1) if deterministic
                 else _categorical_sample(model_probs, generator))
        model = torch.where(active, model, torch.zeros_like(model))

    return ConcreteArch(
        active_mask=active,
        roles=roles,
        edges=edges,
        sequence=sequence,
        model=model,
    )


# ---------------------------------------------------------------------------
# Log-probability helpers (used by GRPO)
# ---------------------------------------------------------------------------

_LOG_EPS = 1e-9


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x ≤ 0.

    For x close to 0 (i.e. exp(x) close to 1) use log(-expm1(x));
    for very negative x use log1p(-exp(x)). Both avoid the catastrophic
    cancellation that direct `torch.log(1 - torch.exp(x))` suffers.
    """
    return torch.where(
        x > -0.6931,                       # ln 2
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def log_prob_gates(gate_logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Bernoulli log-prob over slots, CONDITIONED on "at least one active".

    Sampling rejects all-zero outcomes (see `sample_arch`), so the
    sampling distribution is:
        P(active | at-least-one) = P_indep(active) / (1 - prod(1 - p_i))
    The numerator is the usual sum of per-slot log-probs; the denominator
    correction subtracts log(1 - prod(1 - p_i)).
    """
    # log-sigmoid for numerical stability:
    #   log P(slot=1) = log sigmoid(z) = -softplus(-z)
    #   log P(slot=0) = log sigmoid(-z) = -softplus(z)
    log_p1 = -F.softplus(-gate_logits)
    log_p0 = -F.softplus(gate_logits)
    log_per = torch.where(active, log_p1, log_p0)
    raw_log_prob = log_per.sum()
    # Conditional normalizer: log P(at-least-one) = log(1 - exp(sum log_p0))
    # Clamp the sum strictly below 0 so log1mexp is well-defined even when
    # the head pathologically outputs all gates ≈ 0.
    log_p_none = log_p0.sum().clamp(max=-_LOG_EPS)
    log_p_atleast = _log1mexp(log_p_none)
    return raw_log_prob - log_p_atleast


def log_prob_roles(
    role_logits: torch.Tensor,
    roles: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Sum of Categorical log-probs only over active slots."""
    log_q = F.log_softmax(role_logits, dim=-1)               # [N, R]
    picked = log_q.gather(-1, roles.unsqueeze(-1)).squeeze(-1)  # [N]
    masked = torch.where(active, picked, torch.zeros_like(picked))
    return masked.sum()


def log_prob_models(
    model_logits: torch.Tensor,
    model: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Sum of Categorical log-probs over active slots (mirrors roles)."""
    log_q = F.log_softmax(model_logits, dim=-1)                # [N, n_models]
    picked = log_q.gather(-1, model.unsqueeze(-1)).squeeze(-1)  # [N]
    masked = torch.where(active, picked, torch.zeros_like(picked))
    return masked.sum()


def log_prob_edges(
    edge_logits: torch.Tensor,
    edges: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Sum of Bernoulli log-probs over the active (i,j) pairs (i != j)."""
    log_p1 = -F.softplus(-edge_logits)
    log_p0 = -F.softplus(edge_logits)
    log_per = torch.where(edges, log_p1, log_p0)
    pair = active_pair_mask(active)
    log_per = torch.where(pair, log_per, torch.zeros_like(log_per))
    return log_per.sum()


def log_prob_pl(seq_scores: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    """Plackett-Luce log-prob of a permutation.

    `sequence` is [K] long, listing the slot ids in selected order (a
    permutation of the active slots). `seq_scores` is [N] (we only consume
    entries at indices in `sequence`).
    """
    if sequence.numel() == 0:
        return torch.zeros((), device=seq_scores.device, dtype=seq_scores.dtype)

    seq_list = sequence.tolist()
    remaining = list(seq_list)
    log_p = torch.zeros((), device=seq_scores.device, dtype=seq_scores.dtype)
    for chosen in seq_list:
        denom = torch.logsumexp(seq_scores[remaining], dim=-1)
        log_p = log_p + (seq_scores[chosen] - denom)
        remaining.remove(chosen)
    return log_p


def log_prob_joint(
    logits: ArchLogits,
    arch: ConcreteArch,
) -> torch.Tensor:
    """Sum the typed log-probs into the joint policy log-prob.

    The model term is added only when the architecture carries a model
    assignment (n_models > 1); single-model archs are 4-term as before.
    """
    lp = (
        log_prob_gates(logits.gate_logits, arch.active_mask)
        + log_prob_roles(logits.role_logits, arch.roles, arch.active_mask)
        + log_prob_edges(logits.edge_logits, arch.edges, arch.active_mask)
        + log_prob_pl(logits.seq_scores, arch.sequence)
    )
    if arch.model is not None and logits.model_logits is not None:
        lp = lp + log_prob_models(logits.model_logits, arch.model, arch.active_mask)
    return lp


__all__ = [
    "ConcreteArch",
    "log_prob_edges",
    "log_prob_gates",
    "log_prob_joint",
    "log_prob_models",
    "log_prob_pl",
    "log_prob_roles",
    "sample_arch",
    "sample_pl",
]
