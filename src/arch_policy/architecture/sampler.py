"""Sample a concrete architecture from `ArchLogits` (v3).

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
    """A discrete sampled architecture (no batch dim)."""

    active_mask: torch.Tensor    # bool [N]
    roles: torch.Tensor          # long [N], 0 for inactive (do not interpret)
    edges: torch.Tensor          # bool [N, N], False for inactive pairs
    sequence: torch.Tensor       # long [K]   K = #active

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
        return (
            f"agents={list(zip(actives, roles))} edges={edges} seq={seq}"
        )


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
    N = spec.n_max

    # ----- gates -----------------------------------------------------------
    g_prob = torch.sigmoid(logits.gate_logits)
    if deterministic:
        active = g_prob >= 0.5
    else:
        active = _bernoulli_sample(g_prob, generator)
    if not active.any():
        active = torch.zeros_like(g_prob, dtype=torch.bool)
        active[int(torch.argmax(g_prob).item())] = True

    # ----- roles -----------------------------------------------------------
    role_probs = F.softmax(logits.role_logits, dim=-1)
    if deterministic:
        roles = role_probs.argmax(dim=-1)
    else:
        roles = _categorical_sample(role_probs, generator)
    # zero out role ids on inactive slots so they're unambiguous
    roles = torch.where(active, roles, torch.zeros_like(roles))

    # ----- edges -----------------------------------------------------------
    e_prob = torch.sigmoid(logits.edge_logits)
    if deterministic:
        edges = e_prob >= 0.5
    else:
        edges = _bernoulli_sample(e_prob, generator)
    pair_active = active_pair_mask(active)
    edges = edges & pair_active

    # ----- sequence (PL over active slots) ---------------------------------
    active_idx = torch.nonzero(active, as_tuple=True)[0]
    sequence = sample_pl(
        logits.seq_scores, active_idx,
        deterministic=deterministic, generator=generator,
    )

    return ConcreteArch(
        active_mask=active,
        roles=roles,
        edges=edges,
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Log-probability helpers (used by GRPO)
# ---------------------------------------------------------------------------

_LOG_EPS = 1e-9


def log_prob_gates(gate_logits: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Sum of Bernoulli log-probs over slots, given the sampled `active` (bool [N])."""
    # Use log-sigmoid for numerical stability:
    #   log P(active=1) = log sigmoid(z) = -softplus(-z)
    #   log P(active=0) = log sigmoid(-z) = -softplus(z)
    log_p1 = -F.softplus(-gate_logits)
    log_p0 = -F.softplus(gate_logits)
    log_per = torch.where(active, log_p1, log_p0)
    return log_per.sum()


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


def log_prob_joint(logits: ArchLogits, arch: ConcreteArch) -> torch.Tensor:
    """Sum the 4 typed log-probs into the joint policy log-prob."""
    return (
        log_prob_gates(logits.gate_logits, arch.active_mask)
        + log_prob_roles(logits.role_logits, arch.roles, arch.active_mask)
        + log_prob_edges(logits.edge_logits, arch.edges, arch.active_mask)
        + log_prob_pl(logits.seq_scores, arch.sequence)
    )


__all__ = [
    "ConcreteArch",
    "log_prob_edges",
    "log_prob_gates",
    "log_prob_joint",
    "log_prob_pl",
    "log_prob_roles",
    "sample_arch",
    "sample_pl",
]
