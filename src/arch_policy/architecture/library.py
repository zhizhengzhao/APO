"""A library of human-recognizable multi-agent architectures (v3).

v3 design changes:
  - Roles: 7 (Planner, Solver, Critic, Verifier, Refiner, Researcher, ToolUser).
  - Sequence: a *permutation* of the active slot ids (length = #active).
    Same speaker repeating is implemented at execution time via either:
      (a) Agent ReAct inner loop (single agent thinks multiple times)
      (b) Big-round repetition (the sequence cycles), terminated by Synth.
    This keeps the representation clean and makes Plackett-Luce learning sound.
  - No `weights` field, no aggregator role (Synth handles termination).

A `NamedArch` is a teacher prototype used as an SFT target. Each one is
randomly paired with a task during SFT to teach the head "live in the manifold
of reasonable architectures, regardless of the specific task".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import ARCH

# Convenience role aliases — index in ARCH.role_names.
ROLE_TO_ID = {name: i for i, name in enumerate(ARCH.role_names)}
PLANNER, SOLVER, CRITIC, VERIFIER, REFINER, RESEARCHER, TOOLUSER = (
    ROLE_TO_ID["Planner"],
    ROLE_TO_ID["Solver"],
    ROLE_TO_ID["Critic"],
    ROLE_TO_ID["Verifier"],
    ROLE_TO_ID["Refiner"],
    ROLE_TO_ID["Researcher"],
    ROLE_TO_ID["ToolUser"],
)


@dataclass
class NamedArch:
    """A reference architecture used as an SFT target.

    Stored data:
      - agents:   list of (slot_idx, role_id). Slot must be < ARCH.n_max,
                  unique. Inactive slots are simply omitted.
      - edges:    list of (src_slot, dst_slot). Both endpoints active, no
                  self-loops.
      - sequence: list of slot ids — a permutation of the active slot set.
                  Length must == #active. No repeats.
    """

    name: str
    agents: list[tuple[int, int]]
    edges: list[tuple[int, int]]
    sequence: list[int]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    # ---- sanity checks --------------------------------------------------
    def validate(self) -> None:
        slots = [s for s, _ in self.agents]
        if len(slots) != len(set(slots)):
            raise ValueError(f"{self.name}: duplicate agent slot ids")
        for s in slots:
            if not (0 <= s < ARCH.n_max):
                raise ValueError(f"{self.name}: slot {s} out of range")
        for _, r in self.agents:
            if not (0 <= r < ARCH.k_roles):
                raise ValueError(f"{self.name}: role id {r} out of range")
        active = set(slots)
        for src, dst in self.edges:
            if src not in active or dst not in active:
                raise ValueError(
                    f"{self.name}: edge {src}->{dst} touches inactive slot"
                )
            if src == dst:
                raise ValueError(f"{self.name}: self-loop {src}->{src}")
        if len(self.sequence) != len(active):
            raise ValueError(
                f"{self.name}: sequence length {len(self.sequence)} != "
                f"#active {len(active)} (must be a permutation of actives)"
            )
        if set(self.sequence) != active:
            raise ValueError(
                f"{self.name}: sequence {self.sequence} not a permutation of "
                f"active slots {sorted(active)}"
            )


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def full_mesh(active: list[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in active for j in active if i != j]


# -----------------------------------------------------------------------
# Library: hand-crafted prototypes
# -----------------------------------------------------------------------

def core_library() -> list[NamedArch]:
    """Curated set of named architectures spanning canonical patterns."""
    archs: list[NamedArch] = []

    # ----- single-agent (5) ------------------------------------------------
    archs.append(NamedArch(
        name="single_solver",
        agents=[(0, SOLVER)], edges=[], sequence=[0],
        description="One Solver answers directly.",
        tags=("baseline", "single"),
    ))
    archs.append(NamedArch(
        name="single_planner",
        agents=[(0, PLANNER)], edges=[], sequence=[0],
        description="One Planner decomposes and answers.",
        tags=("single",),
    ))
    archs.append(NamedArch(
        name="single_tooluser",
        agents=[(0, TOOLUSER)], edges=[], sequence=[0],
        description="One ToolUser computes via tool calls.",
        tags=("baseline", "single"),
    ))
    archs.append(NamedArch(
        name="single_verifier",
        agents=[(0, VERIFIER)], edges=[], sequence=[0],
        description="One Verifier (re-derive from scratch).",
        tags=("single",),
    ))
    archs.append(NamedArch(
        name="single_researcher",
        agents=[(0, RESEARCHER)], edges=[], sequence=[0],
        description="One Researcher gathers info then answers.",
        tags=("single",),
    ))

    # ----- 2-agent chains / loops (8) -------------------------------------
    archs.append(NamedArch(
        name="solver_verifier",
        agents=[(0, SOLVER), (1, VERIFIER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Solver then Verifier (open-loop).",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="solver_critic_loop",
        agents=[(0, SOLVER), (1, CRITIC)],
        edges=[(0, 1), (1, 0)],
        sequence=[0, 1],
        description="Solver and Critic exchange (loop via big-round repeat).",
        tags=("loop",),
    ))
    archs.append(NamedArch(
        name="planner_solver",
        agents=[(0, PLANNER), (1, SOLVER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Planner decomposes; Solver executes.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="researcher_solver",
        agents=[(0, RESEARCHER), (1, SOLVER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Researcher fetches context; Solver answers.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="tool_solver",
        agents=[(0, TOOLUSER), (1, SOLVER)],
        edges=[(0, 1), (1, 0)],
        sequence=[0, 1],
        description="ToolUser computes, Solver reasons; iterate via big rounds.",
        tags=("loop",),
    ))
    archs.append(NamedArch(
        name="solver_refiner",
        agents=[(0, SOLVER), (1, REFINER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Solver proposes; Refiner edits / integrates.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="critic_then_solver",
        agents=[(0, CRITIC), (1, SOLVER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Critic flags pitfalls upfront; Solver avoids them.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="dual_solver_independent",
        agents=[(0, SOLVER), (1, SOLVER)],
        edges=[],
        sequence=[0, 1],
        description="Two parallel Solvers; Synth picks via majority.",
        tags=("ensemble",),
    ))

    # ----- 3-agent (10) ---------------------------------------------------
    archs.append(NamedArch(
        name="plan_solve_verify",
        agents=[(0, PLANNER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="Plan → Solve → Verify, classic chain.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="solver_critic_verifier",
        agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="Solver heard by Critic and Verifier; Verifier last.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="dual_solver_refiner",
        agents=[(0, SOLVER), (1, SOLVER), (2, REFINER)],
        edges=[(0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="Two Solvers fan in to a Refiner (MoA pattern).",
        tags=("moa",),
    ))
    archs.append(NamedArch(
        name="dual_solver_verifier",
        agents=[(0, SOLVER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="Two Solvers checked by Verifier.",
        tags=("ensemble",),
    ))
    archs.append(NamedArch(
        name="tool_solver_verifier",
        agents=[(0, TOOLUSER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="ToolUser computes, Solver reasons, Verifier checks.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="researcher_solver_verifier",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2)],
        sequence=[0, 1, 2],
        description="Researcher → Solver → Verifier.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="planner_solver_refiner",
        agents=[(0, PLANNER), (1, SOLVER), (2, REFINER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="Planner sets goal, Solver attempts, Refiner polishes.",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="peers_3_chain",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=[(0, 1), (1, 2)],
        sequence=[0, 1, 2],
        description="Three Solvers in a chain (each sees only previous).",
        tags=("chain",),
    ))
    archs.append(NamedArch(
        name="peers_3_mesh",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=full_mesh([0, 1, 2]),
        sequence=[0, 1, 2],
        description="Three peer Solvers fully connected (debate).",
        tags=("mesh", "debate"),
    ))
    archs.append(NamedArch(
        name="critic_solver_refiner",
        agents=[(0, CRITIC), (1, SOLVER), (2, REFINER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="Critic frames concerns, Solver attempts, Refiner finalizes.",
        tags=("chain",),
    ))

    # ----- 4-agent (8) ----------------------------------------------------
    archs.append(NamedArch(
        name="plan_research_solve_verify",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (1, 2), (2, 3), (0, 2)],
        sequence=[0, 1, 2, 3],
        description="Plan → Research → Solve → Verify (full pipeline).",
        tags=("pipeline",),
    ))
    archs.append(NamedArch(
        name="plan_solver_critic_verifier",
        agents=[(0, PLANNER), (1, SOLVER), (2, CRITIC), (3, VERIFIER)],
        edges=[(0, 1), (1, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Plan → Solve, then Critic + Verifier review.",
        tags=("pipeline",),
    ))
    archs.append(NamedArch(
        name="quad_solver_mesh",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, SOLVER)],
        edges=full_mesh([0, 1, 2, 3]),
        sequence=[0, 1, 2, 3],
        description="Four peer Solvers fully connected.",
        tags=("mesh",),
    ))
    archs.append(NamedArch(
        name="tool_dual_solver_verifier",
        agents=[(0, TOOLUSER), (1, SOLVER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="ToolUser feeds two Solvers; Verifier reconciles.",
        tags=("ensemble",),
    ))
    archs.append(NamedArch(
        name="dual_solver_critic_refiner",
        agents=[(0, SOLVER), (1, SOLVER), (2, CRITIC), (3, REFINER)],
        edges=[(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Two Solvers reviewed by Critic; Refiner integrates.",
        tags=("ensemble",),
    ))
    archs.append(NamedArch(
        name="research_dual_solver_refiner",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Researcher feeds two Solvers; Refiner finalizes.",
        tags=("ensemble",),
    ))
    archs.append(NamedArch(
        name="plan_critic_solver_verifier",
        agents=[(0, PLANNER), (1, CRITIC), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Plan → Critic flags risks → Solver → Verifier.",
        tags=("pipeline",),
    ))
    archs.append(NamedArch(
        name="solver_dual_critic_verifier",
        agents=[(0, SOLVER), (1, CRITIC), (2, CRITIC), (3, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Solver + 2 independent Critics + Verifier.",
        tags=("ensemble",),
    ))

    # ----- 5+ agent (4) ---------------------------------------------------
    archs.append(NamedArch(
        name="rich_5_full",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Plan + Research → Solve → Verify + Refine (full team).",
        tags=("rich",),
    ))
    archs.append(NamedArch(
        name="rich_5_mesh_solvers",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, REFINER), (4, VERIFIER)],
        edges=full_mesh([0, 1, 2]) + [(0, 3), (1, 3), (2, 3), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="3 Solvers in mesh → Refiner → Verifier.",
        tags=("rich", "mesh"),
    ))
    archs.append(NamedArch(
        name="rich_5_tool_team",
        agents=[(0, TOOLUSER), (1, RESEARCHER), (2, SOLVER), (3, CRITIC), (4, VERIFIER)],
        edges=[(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="ToolUser + Researcher feed Solver; Critic + Verifier check.",
        tags=("rich",),
    ))
    archs.append(NamedArch(
        name="rich_6_full",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, SOLVER), (4, CRITIC), (5, VERIFIER)],
        edges=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="Full 6-agent pipeline.",
        tags=("rich",),
    ))

    for a in archs:
        a.validate()
    return archs


# -----------------------------------------------------------------------
# Random perturbations of the core library (data augmentation for SFT)
# -----------------------------------------------------------------------

def random_perturbations(rng: random.Random, n_per_base: int = 1) -> list[NamedArch]:
    """Generate small perturbations of the core library to diversify SFT."""
    out: list[NamedArch] = []
    base = core_library()
    role_pool = (PLANNER, SOLVER, CRITIC, VERIFIER, REFINER, RESEARCHER, TOOLUSER)

    for arch in base:
        if "single" in arch.tags:
            continue  # singles are already minimal; perturbing is silly
        for k in range(n_per_base):
            agents = list(arch.agents)
            edges = list(arch.edges)
            sequence = list(arch.sequence)

            # role swap (one slot)
            if rng.random() < 0.5 and agents:
                idx = rng.randrange(len(agents))
                slot, role = agents[idx]
                new_role = rng.choice([r for r in role_pool if r != role])
                agents[idx] = (slot, new_role)

            # edge drop
            if edges and rng.random() < 0.5:
                edges.pop(rng.randrange(len(edges)))

            # edge add (pick two distinct active slots)
            actives = [s for s, _ in agents]
            if rng.random() < 0.5 and len(actives) >= 2:
                tries = 0
                while tries < 5:
                    a = rng.choice(actives)
                    b = rng.choice(actives)
                    if a != b and (a, b) not in edges:
                        edges.append((a, b))
                        break
                    tries += 1

            # shuffle sequence (swap two positions)
            if rng.random() < 0.5 and len(sequence) >= 2:
                i, j = rng.sample(range(len(sequence)), 2)
                sequence[i], sequence[j] = sequence[j], sequence[i]

            # repair: sequence must be a permutation of actives
            sequence = list(actives)
            rng.shuffle(sequence)

            try:
                new = NamedArch(
                    name=f"{arch.name}_perturb{k}",
                    agents=agents,
                    edges=edges,
                    sequence=sequence,
                    description=f"Perturbation of {arch.name}",
                    tags=("perturbed",),
                )
                new.validate()
                out.append(new)
            except ValueError:
                continue
    return out


def random_archs(rng: random.Random, n: int = 5) -> list[NamedArch]:
    """A few completely random valid architectures."""
    out: list[NamedArch] = []
    role_pool = (PLANNER, SOLVER, CRITIC, VERIFIER, REFINER, RESEARCHER, TOOLUSER)
    for k in range(n):
        n_agents = rng.randint(2, ARCH.n_max)
        slots = sorted(rng.sample(range(ARCH.n_max), n_agents))
        agents = [(s, rng.choice(role_pool)) for s in slots]

        # Random edges (~40% density), no self-loop, between actives only
        edges: list[tuple[int, int]] = []
        for i in slots:
            for j in slots:
                if i != j and rng.random() < 0.4:
                    edges.append((i, j))

        # Sequence = random permutation of active slots
        sequence = list(slots)
        rng.shuffle(sequence)

        try:
            arch = NamedArch(
                name=f"random_{k}",
                agents=agents,
                edges=edges,
                sequence=sequence,
                description="Random valid architecture",
                tags=("random",),
            )
            arch.validate()
            out.append(arch)
        except ValueError:
            continue
    return out


def full_library(seed: int = 42) -> list[NamedArch]:
    """Core + perturbations + random — full SFT target pool."""
    rng = random.Random(seed)
    return (
        core_library()
        + random_perturbations(rng, n_per_base=1)
        + random_archs(rng, n=8)
    )


__all__ = [
    "NamedArch",
    "PLANNER", "SOLVER", "CRITIC", "VERIFIER", "REFINER", "RESEARCHER", "TOOLUSER",
    "ROLE_TO_ID",
    "core_library",
    "full_library",
    "full_mesh",
    "random_archs",
    "random_perturbations",
]
