"""Encode a `NamedArch` into typed `ArchTargets` for SFT.

Plain conversion from human-readable NamedArch to the typed tensors the
head's SFT losses consume.
"""

from __future__ import annotations

from typing import Iterable

import torch

from ..config import ARCH, ArchSpec
from .library import NamedArch
from .spec import ArchTargets


def encode_named_arch(arch: NamedArch, spec: ArchSpec | None = None) -> ArchTargets:
    """Convert a NamedArch into typed `ArchTargets` (CPU tensors)."""
    if spec is None:
        spec = ARCH
    arch.validate()
    N = spec.n_max

    gates = torch.zeros(N, dtype=torch.long)
    roles = torch.zeros(N, dtype=torch.long)
    edges = torch.zeros(N, N, dtype=torch.long)

    for slot, role in arch.agents:
        gates[slot] = 1
        roles[slot] = role

    for src, dst in arch.edges:
        edges[src, dst] = 1

    seq = torch.tensor(arch.sequence, dtype=torch.long)

    targets = ArchTargets(gates=gates, roles=roles, edges=edges, seq=seq)
    targets.validate(spec)
    return targets


def encode_library(archs: Iterable[NamedArch], spec: ArchSpec | None = None) -> list[ArchTargets]:
    """Encode each NamedArch in `archs` into a list of `ArchTargets`."""
    return [encode_named_arch(a, spec) for a in archs]


__all__ = ["encode_named_arch", "encode_library"]
