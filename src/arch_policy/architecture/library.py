"""Library of human-recognizable multi-agent architectures.

Used as SFT teacher targets. Each task during SFT is paired with a uniformly
random NamedArch from this library; per-epoch reshuffling means the head sees
many (task, arch) combinations and learns the *manifold* of reasonable
architectures rather than memorizing task→arch pairings.

Conventions:
  - 8 roles: Planner / Decomposer / Solver / Critic / Verifier / Refiner /
    Researcher / Tester (defined in `config.py`)
  - sequence is a permutation of active slot ids (length = #active, no repeats);
    repeated speech is implemented at execution time via cycles (Synth-controlled)
    or a single agent's ReAct steps inside its turn
  - no `weights` field, no aggregator role — Synth handles termination
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..config import ARCH

# Convenience role aliases — index in ARCH.role_names.
ROLE_TO_ID = {name: i for i, name in enumerate(ARCH.role_names)}
PLANNER = ROLE_TO_ID["Planner"]
DECOMPOSER = ROLE_TO_ID["Decomposer"]
SOLVER = ROLE_TO_ID["Solver"]
CRITIC = ROLE_TO_ID["Critic"]
VERIFIER = ROLE_TO_ID["Verifier"]
REFINER = ROLE_TO_ID["Refiner"]
RESEARCHER = ROLE_TO_ID["Researcher"]
TESTER = ROLE_TO_ID["Tester"]

ROLE_POOL = (PLANNER, DECOMPOSER, SOLVER, CRITIC, VERIFIER, REFINER, RESEARCHER, TESTER)


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
    """Curated set of named architectures spanning canonical patterns.

    Coverage:
      - 1-6 agent sizes
      - all 8 roles appear in ≥3 archs each
      - patterns: chain / loop / mesh / star / hierarchical / debate /
        MoA / Plan-Solve-Verify / Programmer-Tester / Critic-Refiner-loop /
        MAD-Judge / Researcher-Heavy / Verifier-Council
    """
    archs: list[NamedArch] = []

    # =======================================================================
    # Single agent (5)
    # =======================================================================
    archs.append(NamedArch(
        name="single_solver",
        agents=[(0, SOLVER)], edges=[], sequence=[0],
        description="One Solver answers directly.",
        tags=("baseline", "single"),
    ))
    archs.append(NamedArch(
        name="single_planner",
        agents=[(0, PLANNER)], edges=[], sequence=[0],
        description="One Planner produces the answer (degenerate: plan = answer).",
        tags=("single",),
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
        description="One Researcher gathers info then summarizes.",
        tags=("single",),
    ))
    archs.append(NamedArch(
        name="single_tester",
        agents=[(0, TESTER)], edges=[], sequence=[0],
        description="One Tester writes & runs code to derive the answer.",
        tags=("single",),
    ))

    # =======================================================================
    # 2-agent chains / loops (8)
    # =======================================================================
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
        description="Solver and Critic exchange (loop via cycle repeat).",
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
        name="solver_tester_loop",
        agents=[(0, SOLVER), (1, TESTER)],
        edges=[(0, 1), (1, 0)],
        sequence=[0, 1],
        description="Solver writes; Tester runs tests; iterate via cycle repeat.",
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
        description="Two independent Solvers; Synth picks via majority.",
        tags=("ensemble",),
    ))

    # =======================================================================
    # 3-agent (12)
    # =======================================================================
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
        description="Solver heard by Critic and Verifier.",
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
        name="tester_solver_verifier",
        agents=[(0, TESTER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="Tester drafts checks; Solver answers; Verifier confirms.",
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
        description="Three peer Solvers fully connected (Multi-Agent Debate).",
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
    archs.append(NamedArch(
        name="planner_decomposer_solver",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER)],
        edges=[(0, 1), (1, 2)],
        sequence=[0, 1, 2],
        description="Planner sets sub-steps; Decomposer atomizes; Solver executes.",
        tags=("hierarchical",),
    ))
    archs.append(NamedArch(
        name="programmer_tester_refiner",
        agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="Solver writes code; Tester runs tests; Refiner fixes failures.",
        tags=("programmer_tester",),
    ))

    # =======================================================================
    # 4-agent (10)
    # =======================================================================
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
        name="tester_dual_solver_verifier",
        agents=[(0, TESTER), (1, SOLVER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Tester drafts checks; two Solvers answer; Verifier reconciles.",
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
    archs.append(NamedArch(
        name="mad_judge",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, VERIFIER)],
        edges=full_mesh([0, 1, 2]) + [(0, 3), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Three debater Solvers + Verifier as judge (MAD-Judge, Liang'23).",
        tags=("debate", "judge"),
    ))
    archs.append(NamedArch(
        name="critic_refiner_loop_4",
        agents=[(0, SOLVER), (1, CRITIC), (2, REFINER), (3, VERIFIER)],
        edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Solver / Critic / Refiner iterate via cycles, Verifier checks at end.",
        tags=("loop",),
    ))

    # =======================================================================
    # 5-agent (5)
    # =======================================================================
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
        name="rich_5_tester_team",
        agents=[(0, TESTER), (1, RESEARCHER), (2, SOLVER), (3, CRITIC), (4, VERIFIER)],
        edges=[(0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Tester + Researcher feed Solver; Critic + Verifier check.",
        tags=("rich",),
    ))
    archs.append(NamedArch(
        name="researcher_heavy",
        agents=[(0, RESEARCHER), (1, RESEARCHER), (2, RESEARCHER), (3, SOLVER), (4, VERIFIER)],
        edges=[(0, 3), (1, 3), (2, 3), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Three Researchers gather different aspects → Solver → Verifier.",
        tags=("rich", "ensemble"),
    ))
    archs.append(NamedArch(
        name="verifier_council",
        agents=[(0, SOLVER), (1, VERIFIER), (2, VERIFIER), (3, VERIFIER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="One Solver checked by 3 independent Verifiers; Refiner aggregates.",
        tags=("rich", "ensemble"),
    ))
    archs.append(NamedArch(
        name="hierarchical_5",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER), (3, SOLVER), (4, REFINER)],
        edges=[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Planner → Decomposer → 2 Solver workers → Refiner (hierarchical).",
        tags=("rich", "hierarchical"),
    ))

    # =======================================================================
    # 6-agent (2)
    # =======================================================================
    archs.append(NamedArch(
        name="rich_6_full",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, SOLVER), (4, CRITIC), (5, VERIFIER)],
        edges=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="Full 6-agent pipeline.",
        tags=("rich",),
    ))
    archs.append(NamedArch(
        name="hierarchical_6",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER), (3, SOLVER), (4, TESTER), (5, REFINER)],
        edges=[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="Planner → Decomposer → 2 Solvers → Tester → Refiner.",
        tags=("rich", "hierarchical"),
    ))

    for a in archs:
        a.validate()
    return archs


# -----------------------------------------------------------------------
# Random perturbations of the core library (data augmentation for SFT)
# -----------------------------------------------------------------------

# Roles that can produce a final answer. SFT targets must contain ≥1 of these
# — otherwise the architecture has nobody to actually solve the task, which
# is "fully unreasonable" (vs the desired "a bit unreasonable" for noise).
ANSWERER_ROLES = (SOLVER, REFINER, VERIFIER)


def _has_answerer(agents: list[tuple[int, int]]) -> bool:
    return any(r in ANSWERER_ROLES for _, r in agents)


def _perturb_one(rng: random.Random, base: NamedArch, k: int) -> NamedArch | None:
    """Apply a few random mutations to `base` and return a valid NamedArch.

    Mutations applied with independent probabilities:
      - role swap on one slot (forbidden if it would remove the last answerer)
      - one edge drop
      - one edge add (between two distinct active slots)
      - sequence shuffle (always — sequence is a permutation of actives)

    Returns None if the result is structurally invalid OR has no answerer
    (Solver / Refiner / Verifier).
    """
    agents = list(base.agents)
    edges = list(base.edges)

    # role swap (with answerer guard)
    if rng.random() < 0.6 and agents:
        idx = rng.randrange(len(agents))
        slot, role = agents[idx]
        new_role = rng.choice([r for r in ROLE_POOL if r != role])
        candidate_agents = list(agents)
        candidate_agents[idx] = (slot, new_role)
        if _has_answerer(candidate_agents):
            agents = candidate_agents
        # else: skip the swap — keep the answerer

    # edge drop
    if edges and rng.random() < 0.5:
        edges.pop(rng.randrange(len(edges)))

    # edge add (between two distinct active slots, no self-loop, no dup)
    actives = [s for s, _ in agents]
    if rng.random() < 0.5 and len(actives) >= 2:
        for _ in range(5):
            a = rng.choice(actives)
            b = rng.choice(actives)
            if a != b and (a, b) not in edges:
                edges.append((a, b))
                break

    sequence = list(actives)
    rng.shuffle(sequence)

    if not _has_answerer(agents):
        return None
    try:
        new = NamedArch(
            name=f"{base.name}_perturb{k}",
            agents=agents, edges=edges, sequence=sequence,
            description=f"Perturbation of {base.name}",
            tags=("perturbed",) + tuple(t for t in base.tags if t != "baseline"),
        )
        new.validate()
        return new
    except ValueError:
        return None


def random_perturbations(rng: random.Random, n_per_base: int = 3) -> list[NamedArch]:
    """Generate `n_per_base` perturbations per (non-single) core arch."""
    out: list[NamedArch] = []
    for arch in core_library():
        if "single" in arch.tags:
            continue
        for k in range(n_per_base):
            new = _perturb_one(rng, arch, k)
            if new is not None:
                out.append(new)
    return out


def random_archs(rng: random.Random, n: int = 30) -> list[NamedArch]:
    """Random valid architectures with one minimal reasonability constraint:
    at least one Solver / Refiner / Verifier (an "answerer" role).

    Apart from that: random size (2..n_max), random active slots, random
    roles, ~40% edge density, random permutation as sequence. Good for
    injecting "a bit unreasonable" noise into the SFT pool without
    degenerating to "no agent can give a final answer".
    """
    out: list[NamedArch] = []
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        n_agents = rng.randint(2, ARCH.n_max)
        slots = sorted(rng.sample(range(ARCH.n_max), n_agents))
        agents = [(s, rng.choice(ROLE_POOL)) for s in slots]

        if not _has_answerer(agents):
            # Force an answerer at a random slot
            idx = rng.randrange(len(agents))
            agents[idx] = (agents[idx][0], rng.choice(ANSWERER_ROLES))

        edges: list[tuple[int, int]] = []
        for i in slots:
            for j in slots:
                if i != j and rng.random() < 0.4:
                    edges.append((i, j))

        sequence = list(slots)
        rng.shuffle(sequence)

        try:
            arch = NamedArch(
                name=f"random_{len(out)}",
                agents=agents, edges=edges, sequence=sequence,
                description="Random valid architecture.",
                tags=("random",),
            )
            arch.validate()
            out.append(arch)
        except ValueError:
            continue
    return out


def full_library(seed: int = 42) -> list[NamedArch]:
    """Core + perturbations + random — full SFT target pool (~200 entries)."""
    rng = random.Random(seed)
    return (
        core_library()
        + random_perturbations(rng, n_per_base=3)
        + random_archs(rng, n=30)
    )


__all__ = [
    "NamedArch",
    "ROLE_TO_ID", "ROLE_POOL",
    "PLANNER", "DECOMPOSER", "SOLVER", "CRITIC",
    "VERIFIER", "REFINER", "RESEARCHER", "TESTER",
    "core_library",
    "full_library",
    "full_mesh",
    "random_archs",
    "random_perturbations",
]
