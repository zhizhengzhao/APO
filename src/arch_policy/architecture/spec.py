"""Typed structures for the architecture distribution (v3).

In v2 we stored everything as a single flat 1311-dim vector. v3 abandons that
in favor of 4 independent typed tensors, each with its own natural geometry
and loss function. See `config.py` for the design rationale.

Three tensor groups live here:

  - `ArchLogits`   — raw outputs of the head (one per architecture per batch).
  - `ArchTargets`  — supervised SFT targets derived from a `NamedArch`.
  - Helper: `active_pair_mask`, `active_role_mask` for masking active slots.

`ConcreteArch` (the discrete sampled architecture) lives in `sampler.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import ARCH, ArchSpec


@dataclass
class ArchLogits:
    """Raw head outputs for a single architecture (no batch dim).

    Shapes (with N=n_max, R=k_roles):
      gate_logits  [N]       — sigmoid → P(slot active)
      role_logits  [N, R]    — softmax → P(role | slot)
      edge_logits  [N, N]    — sigmoid → P(edge i→j); diagonal pre-masked to -inf
      seq_scores   [N]       — Plackett-Luce score; ranking over actives

    For batched outputs (e.g. during training), each tensor has a leading
    batch dim; the validators here assume single-arch shape.
    """

    gate_logits: torch.Tensor   # [N]
    role_logits: torch.Tensor   # [N, R]
    edge_logits: torch.Tensor   # [N, N]
    seq_scores: torch.Tensor    # [N]

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


@dataclass
class ArchTargets:
    """Supervised SFT targets for a single architecture (no batch dim).

    Shapes (with N=n_max, R=k_roles, K=number of active slots):
      gates   [N]      bool/int (in {0,1})  — 1 if slot is active in the teacher
      roles   [N]      long                 — role id (0..R-1); only meaningful
                                              where gates==1
      edges   [N, N]   bool/int (in {0,1})  — 1 if (i,j) is a teacher edge;
                                              only meaningful for active pairs
      seq     [K]      long                 — teacher's permutation of active
                                              slot ids (length depends on K)
    """

    gates: torch.Tensor   # [N]
    roles: torch.Tensor   # [N]
    edges: torch.Tensor   # [N, N]
    seq: torch.Tensor     # [K]

    def validate(self, spec: ArchSpec | None = None) -> None:
        if spec is None:
            spec = ARCH
        N, R = spec.n_max, spec.k_roles
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
