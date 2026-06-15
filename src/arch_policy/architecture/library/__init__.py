"""Library of multi-agent architecture prototypes (family-organized).

Three tiers (sizes locked by
`tests/test_grpo_advantage.py::test_v7beta_library_sizes`):

  TIER 1 — Canonical (82 entries, 42 families). Defined in `_families`.
      Each family generator yields 1-4 variants sharing a `family_*`
      tag so callers can do family-stratified sampling (default in
      `02_train_sft.py`), avoiding the bias of uniform-over-entries
      sampling toward high-variant families.

  TIER 2 — Imperfect (15 entries). Defined inline below.
      Each entry is a *valid* arch with exactly ONE clear flaw (Critic
      before Solver, lonely Researcher, planner-last, role duplication,
      etc.). Sample-level label smoothing — teaches the head that the
      SFT distribution is *not* a manifold of perfect architectures, so
      it doesn't collapse to sharp template attractors during SFT.

  TIER 3 — Random (10 entries, fixed seed). Defined inline below.
      Random size / roles / edges, constrained to have ≥1 answerer
      (Solver / Refiner / Verifier / Expert). Noise floor.

`full_library()` returns 107 = 82 canonical + 15 imperfect + 10 random.
Default SFT tier_ratio (0.75, 0.15, 0.10) roughly matches this.

Conventions:
  - 8 roles (defined in `config.py:ArchSpec.role_names`).
  - sequence is a permutation of active slot ids; repeated speech is
    handled at execution time via cycles, not at the library level.
  - No `weights` field, no aggregator role — termination handled by Synth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from ..sampler import ConcreteArch
from ...config import ARCH


# ----- Role aliases (used by _families.py + callers) -------------------

ROLE_TO_ID = {name: i for i, name in enumerate(ARCH.role_names)}
PLANNER    = ROLE_TO_ID["Planner"]
EXPERT     = ROLE_TO_ID["Expert"]
SOLVER     = ROLE_TO_ID["Solver"]
CRITIC     = ROLE_TO_ID["Critic"]
VERIFIER   = ROLE_TO_ID["Verifier"]
REFINER    = ROLE_TO_ID["Refiner"]
RESEARCHER = ROLE_TO_ID["Researcher"]
TESTER     = ROLE_TO_ID["Tester"]

ROLE_POOL      = (PLANNER, EXPERT, SOLVER, CRITIC, VERIFIER,
                  REFINER, RESEARCHER, TESTER)
ANSWERER_ROLES = (SOLVER, REFINER, VERIFIER, EXPERT)


@dataclass
class NamedArch:
    """A reference architecture used as an SFT target.

    Fields:
      - agents:   list of (slot_idx, role_id). slot < ARCH.n_max, unique.
      - edges:    list of (src_slot, dst_slot). Both endpoints active, no self-loops.
      - sequence: permutation of active slot ids (length == #active).
    """

    name: str
    agents: list[tuple[int, int]]
    edges: list[tuple[int, int]]
    sequence: list[int]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_concrete(self):
        """Materialize as a `ConcreteArch` with n_max-padded tensors."""
        return named_arch_to_concrete(self)

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
                f"#active {len(active)}"
            )
        if set(self.sequence) != active:
            raise ValueError(
                f"{self.name}: sequence {self.sequence} not a permutation of "
                f"active slots {sorted(active)}"
            )


# ----- Helpers (used by _families.py + callers) ------------------------

def full_mesh(active: list[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in active for j in active if i != j]


def named_arch_to_concrete(named: NamedArch) -> ConcreteArch:
    """Materialize a `NamedArch` into a `ConcreteArch` with n_max-padded tensors."""
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
        active_mask=active_mask, roles=roles, edges=edges, sequence=sequence,
    )


def _has_answerer(agents: list[tuple[int, int]]) -> bool:
    return any(r in ANSWERER_ROLES for _, r in agents)


def family_of(arch: NamedArch) -> str:
    """Return the family tag (`family_*`) of `arch`, or '_imperfect' / '_random'.

    Used for family-stratified sampling. Each canonical NamedArch has exactly
    one tag starting with `family_`; imperfect/random entries don't.
    """
    for t in arch.tags:
        if t.startswith("family_"):
            return t
    if "imperfect" in arch.tags:
        return "_imperfect"
    if "random" in arch.tags:
        return "_random"
    return "_other"


# ----- Tier 1 (deferred import; _families.py needs the names defined above) ---
from ._families import CANONICAL_FAMILIES  # noqa: E402


def canonical_library() -> list[NamedArch]:
    """Tier 1: all canonical archetypes from the 42 families (82 entries)."""
    out: list[NamedArch] = []
    for fam in CANONICAL_FAMILIES:
        for arch in fam():
            arch.validate()
            out.append(arch)
    return out


# ----- Tier 2: imperfect (controlled flaws) ----------------------------

def imperfect_library() -> list[NamedArch]:
    """Tier 2: 15 entries with controlled imperfections.

    Each is *valid* (validates) and has an answerer, but exhibits one
    semantic mismatch (Critic before Solver, lonely Researcher, Verifier
    in middle, role duplication, etc.).
    """
    out: list[NamedArch] = []

    # IMP1 — Critic-before-Solver
    out.append(NamedArch(name="imp_critic_first",
        agents=[(0, CRITIC), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 0), (1, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: Critic speaks first with nothing to critique, "
                    "but loop allows it to react in cycle 2+.",
        tags=("imperfect", "imp_critic_first")))

    # IMP2 — Single-source Refiner
    out.append(NamedArch(name="imp_solo_refiner",
        agents=[(0, SOLVER), (1, REFINER)], edges=[(0, 1)], sequence=[0, 1],
        description="IMPERFECT: Refiner with single input (semantically 'editor').",
        tags=("imperfect", "imp_solo_refiner")))

    # IMP3 — Verifier-in-middle
    out.append(NamedArch(name="imp_verifier_middle_4",
        agents=[(0, SOLVER), (1, VERIFIER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 3), (2, 3), (1, 3)], sequence=[0, 1, 2, 3],
        description="IMPERFECT: Verifier speaks before second Solver finishes.",
        tags=("imperfect", "imp_verifier_middle")))

    # IMP4 — Lonely Researcher
    out.append(NamedArch(name="imp_lonely_researcher",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
        edges=[(1, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: Researcher in the team but isolated (no edges out).",
        tags=("imperfect", "imp_lonely_researcher")))

    # IMP5 — Sparse mesh (×2)
    out.append(NamedArch(name="imp_sparse_mesh_3",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=[(0, 1), (1, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: 3 Solvers but only chain (no full mesh).",
        tags=("imperfect", "imp_sparse_mesh")))
    out.append(NamedArch(name="imp_sparse_mesh_4",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 3), (1, 3), (2, 3)], sequence=[0, 1, 2, 3],
        description="IMPERFECT: 3 Solvers + Refiner; Solver 2 isolated from peers.",
        tags=("imperfect", "imp_sparse_mesh")))

    # IMP6 — No Solver
    out.append(NamedArch(name="imp_no_solver",
        agents=[(0, RESEARCHER), (1, CRITIC), (2, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: No Solver — Refiner must produce the answer.",
        tags=("imperfect", "imp_no_solver")))

    # IMP7 — Backward sequence
    out.append(NamedArch(name="imp_backward_sv",
        agents=[(0, VERIFIER), (1, SOLVER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
        description="IMPERFECT: Verifier first (test-driven style; cycle back-edge gives it data).",
        tags=("imperfect", "imp_backward")))

    # IMP8 — Expert + Critic but no Refiner
    out.append(NamedArch(name="imp_expert_critique_only",
        agents=[(0, EXPERT), (1, CRITIC)], edges=[(0, 1)], sequence=[0, 1],
        description="IMPERFECT: Expert + Critic but no Refiner — Critic's feedback has no consumer.",
        tags=("imperfect", "imp_expert_critique_only")))

    # IMP9 — Tester drives, no Solver
    out.append(NamedArch(name="imp_tester_no_solver",
        agents=[(0, TESTER), (1, REFINER)], edges=[(0, 1)], sequence=[0, 1],
        description="IMPERFECT: Tester drives, Refiner consumes test output as answer.",
        tags=("imperfect", "imp_tester_no_solver")))

    # IMP10 — Over-edge
    out.append(NamedArch(name="imp_over_edge_5",
        agents=[(0, PLANNER), (1, SOLVER), (2, CRITIC), (3, VERIFIER), (4, REFINER)],
        edges=full_mesh([0, 1, 2, 3, 4]), sequence=[0, 1, 2, 3, 4],
        description="IMPERFECT: All 5 agents fully connected (over-communication).",
        tags=("imperfect", "imp_over_edge")))

    # IMP11 — Planner-last
    out.append(NamedArch(name="imp_planner_last",
        agents=[(0, SOLVER), (1, VERIFIER), (2, PLANNER)],
        edges=[(0, 1), (2, 0), (2, 1)], sequence=[0, 1, 2],
        description="IMPERFECT: Planner speaks last; first cycle has no plan to follow.",
        tags=("imperfect", "imp_planner_last")))

    # IMP12 — No loop-back
    out.append(NamedArch(name="imp_no_loopback",
        agents=[(0, VERIFIER), (1, SOLVER)], edges=[(0, 1)], sequence=[0, 1],
        description="IMPERFECT: Verifier→Solver but no Solver→Verifier; nothing to verify in cycle 1.",
        tags=("imperfect", "imp_no_loopback")))

    # IMP13 — Role duplication (×2)
    out.append(NamedArch(name="imp_dup_planner",
        agents=[(0, PLANNER), (1, PLANNER), (2, SOLVER)],
        edges=[(0, 2), (1, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: 2 Planners with no coordination both feed Solver.",
        tags=("imperfect", "imp_role_duplication")))
    out.append(NamedArch(name="imp_dup_verifier",
        agents=[(0, SOLVER), (1, VERIFIER), (2, VERIFIER)],
        edges=[(0, 1), (0, 2)], sequence=[0, 1, 2],
        description="IMPERFECT: 2 Verifiers with no consensus mechanism (no Refiner).",
        tags=("imperfect", "imp_role_duplication")))

    for a in out:
        a.validate()
    return out


# ----- Tier 3: random noise --------------------------------------------

def random_archs(rng: random.Random, n: int = 10) -> list[NamedArch]:
    """Tier 3: `n` random valid architectures (with answerer constraint).
    n_agents range [1, n_max] so the random tier covers solo entries
    (matches the support of `hle_pool` / `canonical_library`)."""
    out: list[NamedArch] = []
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        n_agents = rng.randint(1, ARCH.n_max)
        slots = sorted(rng.sample(range(ARCH.n_max), n_agents))
        agents = [(s, rng.choice(ROLE_POOL)) for s in slots]
        if not _has_answerer(agents):
            idx = rng.randrange(len(agents))
            agents[idx] = (agents[idx][0], rng.choice(ANSWERER_ROLES))
        edges = [
            (i, j) for i in slots for j in slots
            if i != j and rng.random() < 0.4
        ]
        sequence = list(slots); rng.shuffle(sequence)
        try:
            arch = NamedArch(
                name=f"random_{len(out)}",
                agents=agents, edges=edges, sequence=sequence,
                description="Random valid architecture (noise floor for SFT).",
                tags=("random",),
            )
            arch.validate()
            out.append(arch)
        except ValueError:
            continue
    return out


# ----- Tier composition + inject samplers ------------------------------

def full_library(seed: int = 42) -> list[NamedArch]:
    """SFT target pool: canonical + imperfect + random (three-tier).

    Tier 1 — canonical (~77%): 82 well-formed archs across 42 families.
    Tier 2 — imperfect (15%): 15 controlled-flaw archs. Sample-level
            label smoothing — prevents head collapse onto exact templates.
    Tier 3 — random (10%): noise floor. Tells the head "the world has a
            low-quality tail; don't put all probability mass on a few
            canonical patterns".

    Combined with the 0.05 token-level label smoothing in the typed
    losses, this gives a regularised SFT objective that GRPO can later
    refine task-conditionally.
    """
    rng = random.Random(seed)
    return canonical_library() + imperfect_library() + random_archs(rng, n=10)


def default_inject_pool() -> list[NamedArch]:
    """All canonical archs as a flat list (no imperfect / random).

    Imperfect / random are deliberately excluded from inject — they teach
    "what's broken", which is the wrong signal for a BC anchor; bad archs
    naturally get -eps via shaped_advantage when sampled on-policy.

    NB: this is uniform-over-entries which biases BC pressure toward
    high-variant families. For uniform-over-family use
    `sample_inject_family_stratified()`.
    """
    return canonical_library()


def sample_inject_family_stratified(
    rng: random.Random,
    k: int,
) -> list[NamedArch]:
    """Principled inject sampler — uniform BC pressure across families.

    Picks `k` DISTINCT canonical families uniformly, then 1 variant from
    each. Every architectural pattern (family) receives the same expected
    BC pressure regardless of variant count.
    """
    n_fams = len(CANONICAL_FAMILIES)
    if k <= 0 or k > n_fams:
        raise ValueError(f"k={k} must be in [1, {n_fams}]")
    chosen_fams = rng.sample(CANONICAL_FAMILIES, k)
    return [rng.choice(fam()) for fam in chosen_fams]


__all__ = [
    "NamedArch",
    "ROLE_TO_ID", "ROLE_POOL", "ANSWERER_ROLES",
    "PLANNER", "EXPERT", "SOLVER", "CRITIC",
    "VERIFIER", "REFINER", "RESEARCHER", "TESTER",
    "CANONICAL_FAMILIES",
    "canonical_library",
    "imperfect_library",
    "random_archs",
    "full_library",
    "full_mesh",
    "family_of",
    "named_arch_to_concrete",
    "default_inject_pool",
    "sample_inject_family_stratified",
]
