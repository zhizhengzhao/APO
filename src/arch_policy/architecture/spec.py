"""Typed structures for the architecture distribution.

4 independent typed tensors (gate / role / edge / seq); each has its own
natural geometry and loss function. See `config.py`.

This module defines `ArchLogits` (raw head output), `ArchTargets` (SFT
supervised target), and the `active_pair_mask` helper. The discrete
sampled architecture (`ConcreteArch`) lives in `sampler.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import ARCH, ArchSpec


@dataclass
class ArchLogits:
    """Raw head outputs for a single architecture (no batch dim).

    Shapes (N=n_max, R=k_roles):
      gate_logits   [N]    sigmoid → P(slot active)
      role_logits   [N,R]  softmax → P(role | slot)
      edge_logits   [N,N]  sigmoid → P(edge i→j); diagonal pre-masked to −∞
      seq_scores    [N]    Plackett-Luce score; ranking over actives

    Batched head outputs add a leading batch dim; the validators here
    assume single-arch shape.
    """

    gate_logits: torch.Tensor  # [N]
    role_logits: torch.Tensor  # [N, R]
    edge_logits: torch.Tensor  # [N, N]
    seq_scores:  torch.Tensor  # [N]
    # 5th typed distribution — present ONLY when ArchSpec.n_models > 1.
    # None ⇒ single-model setup (no per-agent model choice).
    model_logits: torch.Tensor | None = None  # [N, n_models]

    def validate(self, spec: ArchSpec | None = None) -> None:
        if spec is None:
            spec = ARCH
        N, R = spec.n_max, spec.k_roles
        if self.gate_logits.shape != (N,):
            raise ValueError(f"gate_logits {self.gate_logits.shape} != ({N},)")
        if self.role_logits.shape != (N, R):
            raise ValueError(f"role_logits {self.role_logits.shape} != ({N}, {R})")
        if self.edge_logits.shape != (N, N):
            raise ValueError(f"edge_logits {self.edge_logits.shape} != ({N}, {N})")
        if self.seq_scores.shape != (N,):
            raise ValueError(f"seq_scores {self.seq_scores.shape} != ({N},)")
        if self.model_logits is not None and self.model_logits.shape != (N, spec.n_models):
            raise ValueError(
                f"model_logits {self.model_logits.shape} != ({N}, {spec.n_models})"
            )


@dataclass
class ArchTargets:
    """Supervised SFT targets for a single architecture (no batch dim).

    Shapes (N=n_max, K_active=#active):
      gates   [N]            {0,1} — 1 if slot is in teacher arch
      roles   [N]            long  — role id (only meaningful where gates=1)
      edges   [N,N]          {0,1} — teacher edges (only meaningful on active pairs)
      seq     [K_active]     long  — teacher permutation of active slot ids
    """

    gates: torch.Tensor    # [N]
    roles: torch.Tensor    # [N]
    edges: torch.Tensor    # [N, N]
    seq: torch.Tensor      # [K_active]

    def validate(self, spec: ArchSpec | None = None) -> None:
        if spec is None:
            spec = ARCH
        N = spec.n_max
        if self.gates.shape != (N,):
            raise ValueError(f"gates {self.gates.shape} != ({N},)")
        if self.roles.shape != (N,):
            raise ValueError(f"roles {self.roles.shape} != ({N},)")
        if self.edges.shape != (N, N):
            raise ValueError(f"edges {self.edges.shape} != ({N}, {N})")
        if self.seq.dim() != 1 or self.seq.shape[0] > N:
            raise ValueError(f"seq {self.seq.shape} expects 1d ≤ {N}")
        active = self.gates.bool()
        K = int(active.sum().item())
        if self.seq.shape[0] != K:
            raise ValueError(f"seq length {self.seq.shape[0]} != #active {K}")
        # Each entry of seq must be an active slot, no repeats.
        seq_set = set(int(s.item()) for s in self.seq)
        active_idx = set(int(i) for i in torch.nonzero(active, as_tuple=True)[0])
        if seq_set != active_idx:
            raise ValueError(
                f"seq slots {seq_set} != active slots {active_idx}"
            )

    def active_indices(self) -> torch.Tensor:
        return torch.nonzero(self.gates.bool(), as_tuple=True)[0]


# ---------------------------------------------------------------------------
# Mask helpers (used by losses and sampling)
# ---------------------------------------------------------------------------

def active_pair_mask(active: torch.Tensor) -> torch.Tensor:
    """Given active [N] bool, return [N, N] bool with True only on (i,j) where
    both i and j are active and i != j.
    """
    pair = active.unsqueeze(-1) & active.unsqueeze(-2)
    pair = pair.clone()
    pair.fill_diagonal_(False)
    return pair


__all__ = [
    "ArchLogits",
    "ArchTargets",
    "active_pair_mask",
]
