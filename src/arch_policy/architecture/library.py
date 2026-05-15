"""Library of multi-agent architecture prototypes (family-organized).

Used as SFT teacher targets. Each task during SFT is paired with a uniformly
random NamedArch from this library; per-epoch reshuffling means the head sees
many (task, arch) combinations and learns the *manifold* of reasonable
architectures rather than memorizing task→arch pairings.

Library is organized in 3 tiers (counts are exact at the time of writing;
see `canonical_library() / imperfect_library() / random_archs()` for live
counts):

  TIER 1 — Canonical (69 entries from 33 families)
    Each family is a generator function producing 1-4 variants by varying
    size / role substitution / edge density. Variants share the same
    `family_*` tag so callers can do family-stratified sampling
    (recommended; otherwise high-variant families like fam_mad_debate get
    sampled disproportionately under uniform-over-entries sampling).

  TIER 2 — Imperfect (15 entries, 13 imperfection patterns)
    Controlled flaws: Critic-before-Solver, single-source Refiner,
    Verifier-in-middle, lonely Researcher, planner-last, no-loopback,
    role-duplication, etc. Each entry has exactly ONE clear flaw; teaches
    the head that "slightly off" architectures still live near the
    canonical manifold so GRPO can interpolate around them. Trimmed from
    22 → 15 in V3 by removing redundant 3-agent vs 4-agent twins.

  TIER 3 — Random noise (10 entries)
    Random size + role + edges, but constrained to have ≥1 answerer
    (Solver / Refiner / Verifier). Generated with fixed seed for
    reproducible noise floor (intentional design — Tier 3 is meant to be
    a stable component of the SFT distribution, not per-epoch novelty).

Total full_library() ≈ 94 entries with tier ratio ≈ 73 / 16 / 11.

Design notes (responding to adversarial review):
  - Mesh + sequence: full_mesh edges combined with strict sequence ordering
    is internally consistent because the executor's multi-cycle loop lets
    every agent eventually see every other agent's latest message
    (cycle 1: forward propagation only; cycle 2+: full bidirectional).
  - Verifier-as-Judge in MAD-judge variants: Verifier's responsibility
    ("re-derive from scratch / use sympy / run python_exec") is consistent
    with picking the correct candidate among debating Solvers for objective
    tasks. For subjective tasks, MoA-style Refiner aggregation is more
    appropriate (covered by fam_moa_*).

Conventions:
  - 8 roles: Planner / Decomposer / Solver / Critic / Verifier / Refiner /
    Researcher / Tester (defined in `config.py`)
  - sequence is a permutation of active slot ids (length = #active);
    repeated speech is implemented at execution time via cycles
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
ANSWERER_ROLES = (SOLVER, REFINER, VERIFIER)


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


def _has_answerer(agents: list[tuple[int, int]]) -> bool:
    return any(r in ANSWERER_ROLES for _, r in agents)


def family_of(arch: NamedArch) -> str:
    """Return the family tag (`family_*`) of `arch`, or 'imperfect' / 'random'.

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


def _safe_append(out: list[NamedArch], arch: NamedArch) -> None:
    """Validate and append; raise informative error if invalid."""
    arch.validate()
    out.append(arch)


# =======================================================================
# TIER 1: CANONICAL FAMILIES (33)
# =======================================================================
# Each function returns a list of NamedArch variants. Family tag is
# `family_<short_name>` so callers can filter by family for diagnostics.

# ----- Singles (2 families) --------------------------------------------

def fam_single_solver() -> list[NamedArch]:
    """Family: Single-Solver baseline. 1 variant."""
    return [NamedArch(
        name="single_solver",
        agents=[(0, SOLVER)], edges=[], sequence=[0],
        description="One Solver answers directly (CoT-style baseline).",
        tags=("single", "baseline", "family_single_solver"),
    )]


def fam_single_researcher() -> list[NamedArch]:
    """Family: Single-Researcher with grounding. 1 variant.

    Note: a single Researcher producing the final answer is semantically
    incoherent (Researcher's responsibility is to gather, not to solve).
    Removed in V3 (was `single_researcher` baseline-exception entry).
    Kept the R→V grounded variant which has a real answerer.
    """
    return [
        NamedArch(
            name="researcher_verified",
            agents=[(0, RESEARCHER), (1, VERIFIER)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Researcher gathers + drafts, Verifier checks (Wikipedia-style RAG).",
            tags=("chain", "family_single_researcher"),
        ),
    ]


# ----- 2-agent (5 families) --------------------------------------------

def fam_solver_verifier() -> list[NamedArch]:
    """Family: Solver-Verifier chain ± back-edge. 2 variants."""
    return [
        NamedArch(
            name="sv_chain",
            agents=[(0, SOLVER), (1, VERIFIER)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Open-loop: Solver answers, Verifier checks.",
            tags=("chain", "family_sv"),
        ),
        NamedArch(
            name="sv_loop",
            agents=[(0, SOLVER), (1, VERIFIER)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="Closed-loop: Verifier feedback informs next cycle's Solver.",
            tags=("loop", "family_sv"),
        ),
    ]


def fam_solver_critic() -> list[NamedArch]:
    """Family: Solver-Critic refinement loop. 2 variants."""
    return [
        NamedArch(
            name="sc_chain",
            agents=[(0, SOLVER), (1, CRITIC)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Solver proposes, Critic flags issues (feedback for cycle 2+).",
            tags=("chain", "family_sc"),
        ),
        NamedArch(
            name="sc_loop",
            agents=[(0, SOLVER), (1, CRITIC)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="Solver/Critic exchange; loops via cycle repeat.",
            tags=("loop", "family_sc"),
        ),
    ]


def fam_planner_solver() -> list[NamedArch]:
    """Family: Planner-Solver (Plan & Execute). 2 variants."""
    return [
        NamedArch(
            name="ps_chain",
            agents=[(0, PLANNER), (1, SOLVER)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Planner sets sub-goals, Solver executes.",
            tags=("chain", "family_ps"),
        ),
        NamedArch(
            name="ps_loop",
            agents=[(0, PLANNER), (1, SOLVER)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="Plan-Execute-Replan loop.",
            tags=("loop", "family_ps"),
        ),
    ]


def fam_self_rag() -> list[NamedArch]:
    """Family: Self-RAG / Researcher↔Solver iterative retrieval. 2 variants."""
    return [
        NamedArch(
            name="self_rag_chain",
            agents=[(0, RESEARCHER), (1, SOLVER)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Researcher fetches context once; Solver answers.",
            tags=("chain", "family_self_rag"),
        ),
        NamedArch(
            name="self_rag_loop",
            agents=[(0, RESEARCHER), (1, SOLVER)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="Iterative retrieval: Solver requests more context, "
                        "Researcher fetches; loops via cycle repeat (Self-RAG).",
            tags=("loop", "family_self_rag"),
        ),
    ]


def fam_solver_tester() -> list[NamedArch]:
    """Family: Solver-Tester (Programmer-Tester loop). 2 variants."""
    return [
        NamedArch(
            name="st_chain",
            agents=[(0, SOLVER), (1, TESTER)],
            edges=[(0, 1)],
            sequence=[0, 1],
            description="Solver writes code, Tester runs tests.",
            tags=("chain", "family_st"),
        ),
        NamedArch(
            name="st_loop",
            agents=[(0, SOLVER), (1, TESTER)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="Test-driven loop; Solver fixes failed tests in next cycle.",
            tags=("loop", "family_st"),
        ),
    ]


# ----- 3-agent (8 families) --------------------------------------------

def fam_plan_solve_verify() -> list[NamedArch]:
    """Family: Plan → Solve → Verify (classic). 1 variant."""
    return [
        NamedArch(
            name="psv",
            agents=[(0, PLANNER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (1, 2), (0, 2)],
            sequence=[0, 1, 2],
            description="Plan → Solve → Verify, with Plan also visible to Verifier.",
            tags=("chain", "family_psv"),
        ),
    ]


def fam_plan_solve_critique() -> list[NamedArch]:
    """Family: Plan → Solve → Critic (no rule-based verifier, subjective review)."""
    return [NamedArch(
        name="psc",
        agents=[(0, PLANNER), (1, SOLVER), (2, CRITIC)],
        edges=[(0, 1), (1, 2), (0, 2)],
        sequence=[0, 1, 2],
        description="Plan → Solve → Critic (subjective-review variant; no rule check). "
                    "Final answer comes from Solver via Synth, with Critic as check.",
        tags=("chain", "family_psc"),
    )]


def fam_plan_decompose_solve() -> list[NamedArch]:
    """Family: Plan → Decompose → Solve. 2 variants."""
    return [
        NamedArch(
            name="pds",
            agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER)],
            edges=[(0, 1), (1, 2)],
            sequence=[0, 1, 2],
            description="Hierarchical: Planner sets goal, Decomposer atomizes, Solver acts.",
            tags=("chain", "family_pds"),
        ),
        NamedArch(
            name="pds_v",
            agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER)],
            edges=[(0, 1), (1, 2), (0, 2)],
            sequence=[0, 1, 2],
            description="PDS with Planner also directly visible to Solver.",
            tags=("chain", "family_pds"),
        ),
    ]


def fam_solver_critic_verifier() -> list[NamedArch]:
    """Family: Solver in triangle with Critic + Verifier. 2 variants."""
    return [
        NamedArch(
            name="scv",
            agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="Solver heard by both Critic and Verifier; Critic also informs Verifier.",
            tags=("triangle", "family_scv"),
        ),
        NamedArch(
            name="scv_loop",
            agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 0)],
            sequence=[0, 1, 2],
            description="SCV with Verifier feedback to Solver (multi-cycle refinement).",
            tags=("triangle", "loop", "family_scv"),
        ),
    ]


def fam_solver_critic_refiner() -> list[NamedArch]:
    """Family: Solver-Critic-Refiner (propose-critique-integrate). 2 variants."""
    return [
        NamedArch(
            name="scr",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="Solver proposes, Critic flags, Refiner integrates.",
            tags=("triangle", "family_scr"),
        ),
        NamedArch(
            name="scr_loop",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 0)],
            sequence=[0, 1, 2],
            description="SCR with Refiner feedback to Solver (iterative improvement).",
            tags=("triangle", "loop", "family_scr"),
        ),
    ]


def fam_programmer_tester_refiner() -> list[NamedArch]:
    """Family: Solver writes, Tester runs, Refiner integrates fixes. 2 variants."""
    return [
        NamedArch(
            name="ptr",
            agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="Solver code → Tester runs → Refiner consolidates.",
            tags=("triangle", "family_ptr"),
        ),
        NamedArch(
            name="ptr_loop",
            agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
            edges=[(0, 1), (1, 0), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="PTR with Tester→Solver feedback (test-fix-retest loop).",
            tags=("triangle", "loop", "family_ptr"),
        ),
    ]


def fam_research_solve_verify() -> list[NamedArch]:
    """Family: Researcher → Solver → Verifier. 2 variants."""
    return [
        NamedArch(
            name="rsv",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (1, 2)],
            sequence=[0, 1, 2],
            description="Researcher → Solver → Verifier (RAG with verification).",
            tags=("chain", "family_rsv"),
        ),
        NamedArch(
            name="rsv_full",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="RSV with Researcher also visible to Verifier (cross-check ground truth).",
            tags=("chain", "family_rsv"),
        ),
    ]


def fam_moa_mini() -> list[NamedArch]:
    """Family: MoA-mini (2 Solvers, ± Refiner). 2 variants."""
    return [
        NamedArch(
            name="moa_2sr",
            agents=[(0, SOLVER), (1, SOLVER), (2, REFINER)],
            edges=[(0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="MoA mini: 2 Solvers fan in to Refiner.",
            tags=("moa", "family_moa_mini"),
        ),
        NamedArch(
            name="moa_2s_indep",
            agents=[(0, SOLVER), (1, SOLVER)],
            edges=[],
            sequence=[0, 1],
            description="MoA mini (no Refiner): 2 independent Solvers, Synth picks.",
            tags=("moa", "family_moa_mini"),
        ),
    ]


def fam_mad_mini() -> list[NamedArch]:
    """Family: MAD-mini (2 Solvers debate, ± Judge). 2 variants."""
    return [
        NamedArch(
            name="mad_2sj",
            agents=[(0, SOLVER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (1, 0), (0, 2), (1, 2)],
            sequence=[0, 1, 2],
            description="MAD mini: 2 Solvers debate, Verifier judges.",
            tags=("debate", "family_mad_mini"),
        ),
        NamedArch(
            name="mad_2s_pure",
            agents=[(0, SOLVER), (1, SOLVER)],
            edges=[(0, 1), (1, 0)],
            sequence=[0, 1],
            description="MAD mini (no Judge): pure 2-Solver debate, Synth selects.",
            tags=("debate", "family_mad_mini"),
        ),
    ]


# ----- 4-5 agent (10 families) -----------------------------------------

def fam_mad_debate() -> list[NamedArch]:
    """Family: N-Solver MAD debate ± Judge. 4 variants (n=3,4 × ±judge)."""
    out = []
    for n in (3, 4):
        for judge in (False, True):
            agents = [(i, SOLVER) for i in range(n)]
            edges = full_mesh(list(range(n)))
            if judge:
                if n + 1 > ARCH.n_max:
                    continue
                agents.append((n, VERIFIER))
                for i in range(n):
                    edges.append((i, n))
            name = f"mad_{n}solvers" + ("_judge" if judge else "")
            out.append(NamedArch(
                name=name,
                agents=agents, edges=edges,
                sequence=list(range(len(agents))),
                description=f"MAD: {n} Solvers full mesh"
                            + (" + Verifier as Judge." if judge else "."),
                tags=("debate", "family_mad"),
            ))
    return out


def fam_moa_fanin() -> list[NamedArch]:
    """Family: N Solvers → Refiner. 3 variants (N=2,3,4)."""
    out = []
    for n in (2, 3, 4):
        if n + 1 > ARCH.n_max:
            continue
        agents = [(i, SOLVER) for i in range(n)] + [(n, REFINER)]
        edges = [(i, n) for i in range(n)]
        out.append(NamedArch(
            name=f"moa_{n}sr",
            agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"Mixture of Agents: {n} Solvers fan in to Refiner.",
            tags=("moa", "family_moa"),
        ))
    return out


def fam_verifier_council() -> list[NamedArch]:
    """Family: 1 Solver + N Verifiers (vote) + Refiner. 3 variants (N=2,3,4)."""
    out = []
    for n in (2, 3, 4):
        if 1 + n + 1 > ARCH.n_max:
            continue
        agents = [(0, SOLVER)] + [(i, VERIFIER) for i in range(1, n + 1)] + [(n + 1, REFINER)]
        edges = [(0, i) for i in range(1, n + 1)] + [(i, n + 1) for i in range(1, n + 1)]
        out.append(NamedArch(
            name=f"vcouncil_{n}v",
            agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"Solver + {n} independent Verifiers + Refiner aggregator.",
            tags=("council", "family_vcouncil"),
        ))
    return out


def fam_critic_council() -> list[NamedArch]:
    """Family: 1 Solver + N Critics + Refiner. 2 variants (N=2,3)."""
    out = []
    for n in (2, 3):
        if 1 + n + 1 > ARCH.n_max:
            continue
        agents = [(0, SOLVER)] + [(i, CRITIC) for i in range(1, n + 1)] + [(n + 1, REFINER)]
        edges = [(0, i) for i in range(1, n + 1)] + [(i, n + 1) for i in range(1, n + 1)]
        out.append(NamedArch(
            name=f"ccouncil_{n}c",
            agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"Solver + {n} Critics with diverse perspectives + Refiner.",
            tags=("council", "family_ccouncil"),
        ))
    return out


def fam_researcher_council() -> list[NamedArch]:
    """Family: N Researchers (different aspects) → Solver + Verifier. 3 variants."""
    out = []
    for n in (2, 3):
        if n + 2 > ARCH.n_max:
            continue
        agents = [(i, RESEARCHER) for i in range(n)] + [(n, SOLVER), (n + 1, VERIFIER)]
        edges = [(i, n) for i in range(n)] + [(n, n + 1)]
        out.append(NamedArch(
            name=f"rcouncil_{n}r",
            agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"{n} Researchers gather aspects → Solver → Verifier.",
            tags=("council", "family_rcouncil"),
        ))
    # Also a 3-Researcher + Solver only (no Verifier)
    if 3 + 1 <= ARCH.n_max:
        agents = [(0, RESEARCHER), (1, RESEARCHER), (2, RESEARCHER), (3, SOLVER)]
        edges = [(0, 3), (1, 3), (2, 3)]
        out.append(NamedArch(
            name="rcouncil_3r_solver",
            agents=agents, edges=edges,
            sequence=[0, 1, 2, 3],
            description="3 Researchers (different aspects) → 1 Solver.",
            tags=("council", "family_rcouncil"),
        ))
    return out


def fam_hierarchical() -> list[NamedArch]:
    """Family: Planner → Decomposer → N Workers (Solver) → Refiner. 3 variants."""
    out = []
    for n_workers in (1, 2, 3):
        total = 2 + n_workers + 1
        if total > ARCH.n_max:
            continue
        agents = [(0, PLANNER), (1, DECOMPOSER)]
        worker_slots = list(range(2, 2 + n_workers))
        agents += [(s, SOLVER) for s in worker_slots]
        refiner_slot = 2 + n_workers
        agents.append((refiner_slot, REFINER))
        edges = [(0, 1)] + [(1, w) for w in worker_slots] + [(w, refiner_slot) for w in worker_slots]
        out.append(NamedArch(
            name=f"hier_{n_workers}w",
            agents=agents, edges=edges,
            sequence=list(range(total)),
            description=f"Hierarchical: Planner → Decomposer → {n_workers} workers → Refiner.",
            tags=("hierarchical", "family_hier"),
        ))
    return out


def fam_plan_research_solve_verify() -> list[NamedArch]:
    """Family: Plan + Research → Solve → Verify. 2 variants."""
    out = []
    # canonical: Plan → Research → Solve → Verify (4-agent linear-ish)
    out.append(NamedArch(
        name="prsv",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Plan → Research → Solve → Verify (full pipeline).",
        tags=("pipeline", "family_prsv"),
    ))
    # variant: Plan + Research in parallel, both feed Solver
    out.append(NamedArch(
        name="prsv_parallel",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 2), (1, 2), (2, 3), (0, 3)],
        sequence=[0, 1, 2, 3],
        description="Plan and Research run independently then both feed Solver → Verify.",
        tags=("pipeline", "family_prsv"),
    ))
    return out


def fam_planner_hub() -> list[NamedArch]:
    """Family: Planner-Hub star (HuggingGPT-style). 3 variants."""
    out = []
    # 4-agent: Planner + Researcher + Solver + Verifier (replaces Tester to ensure
    # there is a primary problem-solver in the smallest hub variant).
    out.append(NamedArch(
        name="hub_4",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3), (0, 3)],
        sequence=[0, 1, 2, 3],
        description="Planner-Hub (small): Planner dispatches Researcher + Solver, "
                    "Verifier checks.",
        tags=("hub", "star", "family_hub"),
    ))
    # 5-agent: Planner + 3 specialists + Refiner
    out.append(NamedArch(
        name="hub_5",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, TESTER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Planner-Hub: 3 specialists each handles one sub-task; Refiner aggregates.",
        tags=("hub", "star", "family_hub"),
    ))
    # 6-agent: Planner + 4 specialists + Refiner
    out.append(NamedArch(
        name="hub_6",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, TESTER), (4, CRITIC), (5, REFINER)],
        edges=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="Planner-Hub (full): 4 specialists in parallel → Refiner.",
        tags=("hub", "star", "family_hub"),
    ))
    return out


def fam_tot_fanout() -> list[NamedArch]:
    """Family: Tree-of-Thoughts fan-out (Planner → N Solvers → Refiner). 3 variants."""
    out = []
    for n_solvers in (2, 3, 4):
        if 1 + n_solvers + 1 > ARCH.n_max:
            continue
        agents = [(0, PLANNER)] + [(i, SOLVER) for i in range(1, n_solvers + 1)]
        refiner_slot = n_solvers + 1
        agents.append((refiner_slot, REFINER))
        edges = [(0, i) for i in range(1, n_solvers + 1)] + [(i, refiner_slot) for i in range(1, n_solvers + 1)]
        out.append(NamedArch(
            name=f"tot_{n_solvers}solvers",
            agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"ToT fan-out: Planner → {n_solvers} parallel Solvers → Refiner.",
            tags=("tot", "fanout", "family_tot"),
        ))
    return out


def fam_critic_refiner_loop() -> list[NamedArch]:
    """Family: Solver / Critic / Refiner with cycle loop (multi-iteration polish)."""
    out = []
    out.append(NamedArch(
        name="crloop_3",
        agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
        edges=[(0, 1), (1, 2), (2, 0)],
        sequence=[0, 1, 2],
        description="3-cycle: Solver → Critic → Refiner → Solver (iterate via cycle).",
        tags=("loop", "family_crloop"),
    ))
    out.append(NamedArch(
        name="crloop_4_with_verifier",
        agents=[(0, SOLVER), (1, CRITIC), (2, REFINER), (3, VERIFIER)],
        edges=[(0, 1), (1, 2), (2, 0), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Critic-Refiner loop with Verifier as final gate.",
        tags=("loop", "family_crloop"),
    ))
    return out


# ----- 5+ agent rich (5 families) ---------------------------------------

def fam_full_team() -> list[NamedArch]:
    """Family: Full multi-role team. 2 variants."""
    out = []
    out.append(NamedArch(
        name="full_5",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="5-role team: Plan + Research → Solve → Verify + Refine.",
        tags=("rich", "family_full"),
    ))
    out.append(NamedArch(
        name="full_6",
        agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, SOLVER), (4, CRITIC), (5, VERIFIER)],
        edges=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="6-role team: Plan + Research → 2 Solvers → Critic → Verifier.",
        tags=("rich", "family_full"),
    ))
    return out


def fam_tester_centric() -> list[NamedArch]:
    """Family: Tester is central (code-heavy task). 2 variants."""
    out = []
    # Reorder: Solver speaks before Tester so Tester has something to test.
    out.append(NamedArch(
        name="tester_5",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, TESTER), (3, CRITIC), (4, VERIFIER)],
        edges=[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Researcher → Solver writes code → Tester runs → Critic + Verifier check.",
        tags=("rich", "family_tester_centric"),
    ))
    out.append(NamedArch(
        name="tester_4",
        agents=[(0, SOLVER), (1, TESTER), (2, TESTER), (3, REFINER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Solver + 2 independent Testers (different test angles) + Refiner.",
        tags=("rich", "family_tester_centric"),
    ))
    return out


def fam_sop_linear() -> list[NamedArch]:
    """Family: SOP-style long linear pipeline (ChatDev / MetaGPT). 2 variants."""
    out = []
    # 5-agent linear: Planner → Decomposer → Solver → Tester → Verifier
    out.append(NamedArch(
        name="sop_5",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER), (3, TESTER), (4, VERIFIER)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="SOP linear pipeline: Plan → Decompose → Solve → Test → Verify "
                    "(ChatDev/MetaGPT style; minimal cross-talk, single artifact handoff).",
        tags=("pipeline", "sop", "family_sop"),
    ))
    # 6-agent: + Refiner at the end for polishing
    out.append(NamedArch(
        name="sop_6",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, RESEARCHER), (3, SOLVER), (4, TESTER), (5, REFINER)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="SOP linear pipeline (full): Plan → Decompose → Research → Solve → Test → Refine.",
        tags=("pipeline", "sop", "family_sop"),
    ))
    return out


def fam_tot_deep() -> list[NamedArch]:
    """Family: ToT-style 2-layer expansion (deeper search). 1 variant."""
    return [NamedArch(
        name="tot_deep_6",
        agents=[(0, PLANNER), (1, SOLVER), (2, SOLVER), (3, CRITIC), (4, CRITIC), (5, REFINER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="ToT deeper: Planner → 2 Solvers → each gets a Critic → Refiner aggregates "
                    "(2-layer expansion vs single-layer fan-out in fam_tot_fanout).",
        tags=("tot", "fanout", "family_tot_deep"),
    )]


def fam_mesh_then_aggregate() -> list[NamedArch]:
    """Family: Multi-Solver mesh → aggregator → final check. 2 variants."""
    out = []
    out.append(NamedArch(
        name="mesh_then_agg_5",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, REFINER), (4, VERIFIER)],
        edges=full_mesh([0, 1, 2]) + [(0, 3), (1, 3), (2, 3), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="3 Solvers debate (mesh) → Refiner integrates → Verifier checks.",
        tags=("rich", "mesh", "family_mesh_then_agg"),
    ))
    out.append(NamedArch(
        name="mesh_then_agg_4",
        agents=[(0, SOLVER), (1, SOLVER), (2, REFINER), (3, VERIFIER)],
        edges=[(0, 1), (1, 0), (0, 2), (1, 2), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="2 Solvers debate → Refiner → Verifier.",
        tags=("rich", "mesh", "family_mesh_then_agg"),
    ))
    return out


def fam_decomposer_heavy() -> list[NamedArch]:
    """Family: Multi-Decomposer (parallel sub-task atomization). 2 variants."""
    out = []
    out.append(NamedArch(
        name="decomp_heavy_4",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
        sequence=[0, 1, 2, 3],
        description="Planner → Decomposer → Solver → Refiner (with Plan also visible to Refiner).",
        tags=("rich", "family_decomp_heavy"),
    ))
    out.append(NamedArch(
        name="decomp_heavy_5",
        agents=[(0, PLANNER), (1, DECOMPOSER), (2, DECOMPOSER), (3, SOLVER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Planner → 2 Decomposers (split sub-tasks) → Solver → Refiner.",
        tags=("rich", "family_decomp_heavy"),
    ))
    return out


def fam_research_then_mesh() -> list[NamedArch]:
    """Family: Research-then-mesh (RAG + debate). 2 variants."""
    out = []
    out.append(NamedArch(
        name="research_mesh_4",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)],
        sequence=[0, 1, 2, 3],
        description="Researcher feeds 2 Solvers; Solvers debate; Refiner aggregates.",
        tags=("rich", "family_research_mesh"),
    ))
    out.append(NamedArch(
        name="research_mesh_5",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, SOLVER), (3, SOLVER), (4, REFINER)],
        edges=[(0, 1), (0, 2), (0, 3)] + full_mesh([1, 2, 3]) + [(1, 4), (2, 4), (3, 4)],
        sequence=[0, 1, 2, 3, 4],
        description="Researcher feeds 3 Solvers; full debate; Refiner aggregates.",
        tags=("rich", "family_research_mesh"),
    ))
    return out


# =======================================================================
# Combine all families → canonical_library
# =======================================================================

CANONICAL_FAMILIES = [
    # Singles (2)
    fam_single_solver, fam_single_researcher,
    # 2-agent (5)
    fam_solver_verifier, fam_solver_critic, fam_planner_solver,
    fam_self_rag, fam_solver_tester,
    # 3-agent (9)
    fam_plan_solve_verify, fam_plan_solve_critique, fam_plan_decompose_solve,
    fam_solver_critic_verifier, fam_solver_critic_refiner,
    fam_programmer_tester_refiner, fam_research_solve_verify,
    fam_moa_mini, fam_mad_mini,
    # 4-5 agent (10)
    fam_mad_debate, fam_moa_fanin, fam_verifier_council, fam_critic_council,
    fam_researcher_council, fam_hierarchical, fam_plan_research_solve_verify,
    fam_planner_hub, fam_tot_fanout, fam_critic_refiner_loop,
    # Rich (7)
    fam_full_team, fam_tester_centric, fam_mesh_then_aggregate,
    fam_decomposer_heavy, fam_research_then_mesh,
    fam_sop_linear, fam_tot_deep,
]


def canonical_library() -> list[NamedArch]:
    """Tier 1: all canonical archetypes from the 33 families (~70 entries)."""
    out: list[NamedArch] = []
    for fam in CANONICAL_FAMILIES:
        for arch in fam():
            arch.validate()
            out.append(arch)
    return out


# =======================================================================
# TIER 2: IMPERFECT entries (controlled flaws)
# =======================================================================

def imperfect_library() -> list[NamedArch]:
    """Tier 2: ~20 entries with controlled imperfections.

    Each one is *valid* (validates) and has an answerer, but exhibits one
    semantic mismatch (e.g. Critic before Solver, single-source Refiner,
    Verifier in middle, lonely Researcher). They teach the head that the
    SFT distribution is *not* a manifold of perfect architectures — it
    has soft edges where slightly off variants live.
    """
    out: list[NamedArch] = []

    # IMP1 — Critic-before-Solver (sequence puts Critic first; Critic has no
    #        candidate to react to in cycle 1, but receives Solver's output
    #        in subsequent cycles via the back-edge from Solver).
    out.append(NamedArch(
        name="imp_critic_first",
        agents=[(0, CRITIC), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 0), (1, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: Critic speaks first with nothing to critique, "
                    "but loop allows it to react in cycle 2+.",
        tags=("imperfect", "imp_critic_first"),
    ))

    # IMP2 — Single-source Refiner (Refiner with only 1 input; "integrate" is
    #        a stretch — it's basically a polishing pass).
    out.append(NamedArch(
        name="imp_solo_refiner",
        agents=[(0, SOLVER), (1, REFINER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="IMPERFECT: Refiner with single input (semantically more like 'editor').",
        tags=("imperfect", "imp_solo_refiner"),
    ))

    # IMP3 — Verifier-in-middle (Verifier speaks before all Solvers/Refiners
    #        have produced their output).
    out.append(NamedArch(
        name="imp_verifier_middle_4",
        agents=[(0, SOLVER), (1, VERIFIER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 3), (2, 3), (1, 3)],
        sequence=[0, 1, 2, 3],
        description="IMPERFECT: Verifier speaks before second Solver finishes.",
        tags=("imperfect", "imp_verifier_middle"),
    ))

    # IMP4 — Lonely Researcher (Researcher in team but no outgoing edge to
    #        any Solver — its findings go to nobody useful).
    out.append(NamedArch(
        name="imp_lonely_researcher",
        agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
        edges=[(1, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: Researcher is in the team but isolated (no edges out).",
        tags=("imperfect", "imp_lonely_researcher"),
    ))

    # IMP5 — Sparse mesh (debate-style mesh but with edges removed; some
    #        Solvers can't see each other).
    out.append(NamedArch(
        name="imp_sparse_mesh_3",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER)],
        edges=[(0, 1), (1, 2)],   # chain instead of mesh
        sequence=[0, 1, 2],
        description="IMPERFECT: 3 Solvers but only chain (no full mesh) — limited debate.",
        tags=("imperfect", "imp_sparse_mesh"),
    ))
    out.append(NamedArch(
        name="imp_sparse_mesh_4",
        agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, REFINER)],
        edges=[(0, 1), (0, 3), (1, 3), (2, 3)],   # solver 2 isolated from peers
        sequence=[0, 1, 2, 3],
        description="IMPERFECT: 3 Solvers + Refiner, but Solver 2 isolated from peer debate.",
        tags=("imperfect", "imp_sparse_mesh"),
    ))

    # IMP6 — Pure-Critic team (no Solver, but Critic reading what?
    #        Replaced by Refiner answer-producer to keep an answerer).
    out.append(NamedArch(
        name="imp_no_solver",
        agents=[(0, RESEARCHER), (1, CRITIC), (2, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: No Solver — Refiner must produce the answer from research + critique.",
        tags=("imperfect", "imp_no_solver"),
    ))

    # IMP7 — Backward sequence (Verifier first, then Solver; valid graph but
    #        weird information flow — makes more sense for "test-driven" but
    #        Verifier can't really verify nothing).
    out.append(NamedArch(
        name="imp_backward_sv",
        agents=[(0, VERIFIER), (1, SOLVER)],
        edges=[(0, 1), (1, 0)],
        sequence=[0, 1],
        description="IMPERFECT: Verifier first (test-driven style; cycle back-edge gives it data).",
        tags=("imperfect", "imp_backward"),
    ))

    # IMP8 — Decomposer without Planner (does atomization without high-level plan).
    out.append(NamedArch(
        name="imp_decomp_no_planner",
        agents=[(0, DECOMPOSER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: Decomposer without an upstream Planner.",
        tags=("imperfect", "imp_decomp_no_planner"),
    ))

    # IMP9 — Tester-no-Solver (Tester as primary, with a Refiner producing answer).
    out.append(NamedArch(
        name="imp_tester_no_solver",
        agents=[(0, TESTER), (1, REFINER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="IMPERFECT: Tester drives, Refiner consumes test output as answer.",
        tags=("imperfect", "imp_tester_no_solver"),
    ))

    # IMP10 — Over-edge (every active pair connected, forming a fully-connected
    #         5-agent system — too dense for the role mix).
    out.append(NamedArch(
        name="imp_over_edge_5",
        agents=[(0, PLANNER), (1, SOLVER), (2, CRITIC), (3, VERIFIER), (4, REFINER)],
        edges=full_mesh([0, 1, 2, 3, 4]),
        sequence=[0, 1, 2, 3, 4],
        description="IMPERFECT: All 5 agents fully connected (over-communication).",
        tags=("imperfect", "imp_over_edge"),
    ))

    # IMP11 — Planner-last (orchestrator at tail; first cycle has no plan).
    out.append(NamedArch(
        name="imp_planner_last",
        agents=[(0, SOLVER), (1, VERIFIER), (2, PLANNER)],
        edges=[(0, 1), (2, 0), (2, 1)],
        sequence=[0, 1, 2],
        description="IMPERFECT: Planner speaks last; first cycle has no plan to follow.",
        tags=("imperfect", "imp_planner_last"),
    ))

    # IMP12 — No-loopback (Verifier→Solver edge exists but no Solver→Verifier;
    #         Verifier has nothing to verify in cycle 1).
    out.append(NamedArch(
        name="imp_no_loopback",
        agents=[(0, VERIFIER), (1, SOLVER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="IMPERFECT: Verifier→Solver but no Solver→Verifier; "
                    "Verifier has nothing to verify in cycle 1.",
        tags=("imperfect", "imp_no_loopback"),
    ))

    # IMP13 — Role duplication (two Planners or two Verifiers with no
    #         coordination — wasted capacity / parallel duplicate channels).
    out.append(NamedArch(
        name="imp_dup_planner",
        agents=[(0, PLANNER), (1, PLANNER), (2, SOLVER)],
        edges=[(0, 2), (1, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: 2 Planners with no coordination both feed Solver "
                    "(duplicated channel).",
        tags=("imperfect", "imp_role_duplication"),
    ))
    out.append(NamedArch(
        name="imp_dup_verifier",
        agents=[(0, SOLVER), (1, VERIFIER), (2, VERIFIER)],
        edges=[(0, 1), (0, 2)],
        sequence=[0, 1, 2],
        description="IMPERFECT: 2 Verifiers with no consensus mechanism (no Refiner).",
        tags=("imperfect", "imp_role_duplication"),
    ))

    for a in out:
        a.validate()
    return out


# =======================================================================
# TIER 3: Random noise
# =======================================================================

def random_archs(rng: random.Random, n: int = 10) -> list[NamedArch]:
    """Tier 3: ~10 random valid architectures (with answerer constraint).

    Random size (2..n_max), random active slots, random roles, ~40% edge
    density, random permutation as sequence. Forces ≥1 answerer
    (Solver/Refiner/Verifier).
    """
    out: list[NamedArch] = []
    attempts = 0
    while len(out) < n and attempts < n * 10:
        attempts += 1
        n_agents = rng.randint(2, ARCH.n_max)
        slots = sorted(rng.sample(range(ARCH.n_max), n_agents))
        agents = [(s, rng.choice(ROLE_POOL)) for s in slots]

        if not _has_answerer(agents):
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
                description="Random valid architecture (noise floor for SFT).",
                tags=("random",),
            )
            arch.validate()
            out.append(arch)
        except ValueError:
            continue
    return out


# =======================================================================
# Public API
# =======================================================================

def core_library() -> list[NamedArch]:
    """Backwards-compatible alias for the canonical (Tier 1) library.

    Used by callers that just want the cleanly-designed archetypes,
    without imperfect or random entries (e.g. for inspection in scripts).
    """
    return canonical_library()


def full_library(seed: int = 42) -> list[NamedArch]:
    """Tier 1 + Tier 2 + Tier 3 = full SFT target pool (~100 entries)."""
    rng = random.Random(seed)
    return canonical_library() + imperfect_library() + random_archs(rng, n=10)


# Backwards-compat: some external scripts might still import this.
def random_perturbations(rng: random.Random, n_per_base: int = 1) -> list[NamedArch]:
    """DEPRECATED: variants now generated by family generators directly.

    Kept as a no-op for backwards compatibility.
    """
    return []


__all__ = [
    "NamedArch",
    "ROLE_TO_ID", "ROLE_POOL", "ANSWERER_ROLES",
    "PLANNER", "DECOMPOSER", "SOLVER", "CRITIC",
    "VERIFIER", "REFINER", "RESEARCHER", "TESTER",
    "CANONICAL_FAMILIES",
    "canonical_library",
    "imperfect_library",
    "random_archs",
    "random_perturbations",   # deprecated, kept for compat
    "core_library",
    "full_library",
    "full_mesh",
    "family_of",
]
