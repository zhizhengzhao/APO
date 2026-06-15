"""Baseline architectures: fixed topologies for apples-to-apples comparison.

Each `make_*` returns a `ConcreteArch` ready for `MultiAgentExecutor.run`.
NamedArch is the source of truth; materialized deterministically (no head
sampling). Covers canonical patterns from MaAS / G-Designer / ARG-Designer
/ EvoMAC.
"""

from __future__ import annotations

from .architecture.library import (
    CRITIC,
    EXPERT,
    PLANNER,
    REFINER,
    SOLVER,
    TESTER,
    VERIFIER,
    NamedArch,
    full_mesh,
    named_arch_to_concrete as _arch_from_named,
)
from .architecture.sampler import ConcreteArch
from .config import ARCH


# 11 fixed-topology baselines.

def make_single_agent() -> ConcreteArch:
    return _arch_from_named(NamedArch(
        name="bl_single",
        agents=[(0, SOLVER)],
        edges=[],
        sequence=[0],
    ))


def make_single_expert() -> ConcreteArch:
    """B1 baseline: one Expert agent with the full tool pool.

    This is the critical 'no-architecture' control. The Expert role
    is the single-agent generalist (gets all 5 tools per role_tools.py).
    Run with the production per-trace budget (safety_max_cycles=8 rounds
    of Synth-gated self-revision, each a safety_max_steps=16 ReAct loop,
    all tools) it's 'a single strong agent with all tools + ample budget'.

    The architecture search space INCLUDES this topology, so the
    trained head must beat 'always pick solo-Expert' to justify the
    APO machinery. Hence this is the lower-bound reference.
    """
    return _arch_from_named(NamedArch(
        name="bl_single_expert",
        agents=[(0, EXPERT)],
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
    "single_expert": make_single_expert,
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
