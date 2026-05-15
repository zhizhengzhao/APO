"""Baseline architectures (v3): fixed topologies for fair comparison.

Each `make_*` function returns a `ConcreteArch` ready to be passed to
`MultiAgentExecutor.run`. We use NamedArch as the source of truth and
materialize it into a ConcreteArch deterministically (no head sampling).

The set covers the canonical patterns reported by MaAS / G-Designer /
ARG-Designer / EvoMAC so we can plot apples-to-apples curves.
"""

from __future__ import annotations

import torch

from .architecture.library import (
    CRITIC,
    DECOMPOSER,
    PLANNER,
    REFINER,
    RESEARCHER,
    SOLVER,
    TESTER,
    VERIFIER,
    NamedArch,
    full_mesh,
)
from .architecture.sampler import ConcreteArch
from .config import ARCH


def _arch_from_named(named: NamedArch) -> ConcreteArch:
    """Build a `ConcreteArch` from a `NamedArch` deterministically.

    `named.sequence` must be a permutation of active slots (length = #active).
    """
    named.validate()
    n = ARCH.n_max

    active_mask = torch.zeros(n, dtype=torch.bool)
    roles = torch.zeros(n, dtype=torch.long)
    edges = torch.zeros(n, n, dtype=torch.bool)

    for slot, role in named.agents:
        active_mask[slot] = True
        roles[slot] = role

    for src, dst in named.edges:
        edges[src, dst] = True

    sequence = torch.tensor(named.sequence, dtype=torch.long)

    return ConcreteArch(
        active_mask=active_mask,
        roles=roles,
        edges=edges,
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# 11 fixed-topology baselines covering canonical patterns reported by
# MaAS / G-Designer / ARG-Designer / EvoMAC.
# ---------------------------------------------------------------------------

def make_single_agent() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_single",
        agents=[(0, SOLVER)],
        edges=[],
        sequence=[0],
    ))


def make_solver_verifier() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_solver_verifier",
        agents=[(0, SOLVER), (1, VERIFIER)],
        edges=[(0, 1)],
        sequence=[0, 1],
    ))


def make_chain_3() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_chain_3",
        agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
        edges=[(0, 1), (1, 2)],
        sequence=[0, 1, 2],
    ))


def make_star_solver_verifier(n_solvers: int = 3) -> ConcreteArch:
    """A 'star' with n_solvers fanning into one verifier."""
    n_solvers = min(n_solvers, ARCH.n_max - 1)
    agents = [(i, SOLVER) for i in range(n_solvers)] + [(n_solvers, VERIFIER)]
    edges = [(i, n_solvers) for i in range(n_solvers)]
    seq = list(range(n_solvers + 1))
    return _arch_from_named(NamedArch(
        name=f"bl_star_{n_solvers}solvers",
        agents=agents,
        edges=edges,
        sequence=seq,
    ))


def make_mesh_3() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_mesh_3",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=full_mesh([0, 1, 2]),
        sequence=[0, 1, 2],
    ))


def make_self_consistency_5() -> ConcreteArch:
    """Five independent solvers; Synth + heuristic picks the consensus."""
    n = min(5, ARCH.n_max)
    agents = [(i, SOLVER) for i in range(n)]
    return _arch_from_named(NamedArch(
        name="bl_self_consistency_5",
        agents=agents,
        edges=[],
        sequence=list(range(n)),
    ))


def make_debate_3() -> ConcreteArch:
    """3 solvers fully connected — debate via repeated cycles (Synth decides when to stop)."""
    return _arch_from_named(NamedArch(
        name="bl_debate_3",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=full_mesh([0, 1, 2]),
        sequence=[0, 1, 2],
    ))


def make_solver_critic_verifier() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_solver_critic_verifier",
        agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 2)],
        sequence=[0, 1, 2],
    ))


def make_tester_solver_verifier() -> ConcreteArch:
    """Tester drafts cases / runs code, Solver answers, Verifier confirms."""
    return _arch_from_named(NamedArch(
        name="bl_tester_solver_verifier",
        agents=[(0, TESTER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
    ))


def make_plan_solve_verify() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_plan_solve_verify",
        agents=[(0, PLANNER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
    ))


def make_programmer_tester_refiner() -> ConcreteArch:
    """Solver writes code, Tester runs tests, Refiner integrates."""
    return _arch_from_named(NamedArch(
        name="bl_programmer_tester_refiner",
        agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2)],
        sequence=[0, 1, 2],
    ))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: dict[str, callable] = {
    "single": make_single_agent,
    "self_consistency_5": make_self_consistency_5,
    "solver_verifier": make_solver_verifier,
    "chain_3": make_chain_3,
    "star_3": lambda: make_star_solver_verifier(3),
    "mesh_3": make_mesh_3,
    "debate_3": make_debate_3,
    "solver_critic_verifier": make_solver_critic_verifier,
    "tester_solver_verifier": make_tester_solver_verifier,
    "plan_solve_verify": make_plan_solve_verify,
    "programmer_tester_refiner": make_programmer_tester_refiner,
}


def get_baseline(name: str) -> ConcreteArch:
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name]()


__all__ = ["BASELINE_REGISTRY", "get_baseline"]
