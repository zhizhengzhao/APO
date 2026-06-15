"""42 canonical architecture families (returned as `list[NamedArch]`).

Imported by `architecture.library.__init__` which exposes the public
`canonical_library()` / `CANONICAL_FAMILIES` API. Keeping the family
definitions in their own file so the library package's __init__ stays
focused on the public API + tier composition rules.

Each `fam_*` returns 1-4 variants. Family tag is `family_<short_name>`
so callers can do family-stratified sampling.
"""

from __future__ import annotations

# Import the dataclass + role aliases + helpers from the library package.
# This works because __init__.py defines these names BEFORE importing
# from this module (see end of __init__.py).
from . import (
    ARCH,
    CRITIC,
    EXPERT,
    NamedArch,
    PLANNER,
    REFINER,
    RESEARCHER,
    SOLVER,
    TESTER,
    VERIFIER,
    full_mesh,
)


# ----- Singles (3 families: single_solver, researcher_verified, expert_only) ----

def fam_single_solver() -> list[NamedArch]:
    """Family: Single-Solver baseline. 1 variant."""
    return [NamedArch(
        name="single_solver",
        agents=[(0, SOLVER)], edges=[], sequence=[0],
        description="One Solver answers directly (CoT-style baseline).",
        tags=("single", "baseline", "family_single_solver"),
    )]


def fam_researcher_verified() -> list[NamedArch]:
    """Family: Researcher → Verifier grounded chain. 1 variant."""
    return [NamedArch(
        name="researcher_verified",
        agents=[(0, RESEARCHER), (1, VERIFIER)],
        edges=[(0, 1)],
        sequence=[0, 1],
        description="Researcher gathers + drafts, Verifier checks (Wikipedia-style RAG).",
        tags=("chain", "family_researcher_verified"),
    )]


def fam_expert_only() -> list[NamedArch]:
    """Family: single Expert (all-tools generalist). 1 variant.

    The strongest single-agent baseline — distinct from `fam_single_solver`
    (Solver has only python_exec; Expert has the full 5-tool pool).
    """
    return [NamedArch(
        name="expert_only",
        agents=[(0, EXPERT)], edges=[], sequence=[0],
        description="Single Expert (all tools) — strongest single-agent baseline.",
        tags=("single", "baseline", "family_expert_only"),
    )]


# ----- 2-agent (9 families) --------------------------------------------

def fam_solver_verifier() -> list[NamedArch]:
    """Family: Solver-Verifier chain ± back-edge. 2 variants."""
    return [
        NamedArch(name="sv_chain",
            agents=[(0, SOLVER), (1, VERIFIER)], edges=[(0, 1)], sequence=[0, 1],
            description="Open-loop: Solver answers, Verifier checks.",
            tags=("chain", "family_sv")),
        NamedArch(name="sv_loop",
            agents=[(0, SOLVER), (1, VERIFIER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Closed-loop: Verifier feedback informs next cycle's Solver.",
            tags=("loop", "family_sv")),
    ]


def fam_solver_critic() -> list[NamedArch]:
    """Family: Solver-Critic refinement loop. 2 variants."""
    return [
        NamedArch(name="sc_chain",
            agents=[(0, SOLVER), (1, CRITIC)], edges=[(0, 1)], sequence=[0, 1],
            description="Solver proposes, Critic flags issues (feedback for cycle 2+).",
            tags=("chain", "family_sc")),
        NamedArch(name="sc_loop",
            agents=[(0, SOLVER), (1, CRITIC)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Solver/Critic exchange; loops via cycle repeat.",
            tags=("loop", "family_sc")),
    ]


def fam_planner_solver() -> list[NamedArch]:
    """Family: Planner-Solver (Plan & Execute). 2 variants."""
    return [
        NamedArch(name="ps_chain",
            agents=[(0, PLANNER), (1, SOLVER)], edges=[(0, 1)], sequence=[0, 1],
            description="Planner sets sub-goals, Solver executes.",
            tags=("chain", "family_ps")),
        NamedArch(name="ps_loop",
            agents=[(0, PLANNER), (1, SOLVER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Plan-Execute-Replan loop.",
            tags=("loop", "family_ps")),
    ]


def fam_self_rag() -> list[NamedArch]:
    """Family: Self-RAG / Researcher↔Solver iterative retrieval. 2 variants."""
    return [
        NamedArch(name="self_rag_chain",
            agents=[(0, RESEARCHER), (1, SOLVER)], edges=[(0, 1)], sequence=[0, 1],
            description="Researcher fetches context once; Solver answers.",
            tags=("chain", "family_self_rag")),
        NamedArch(name="self_rag_loop",
            agents=[(0, RESEARCHER), (1, SOLVER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Iterative retrieval: Solver requests more context, "
                        "Researcher fetches; loops via cycle repeat (Self-RAG).",
            tags=("loop", "family_self_rag")),
    ]


def fam_solver_tester() -> list[NamedArch]:
    """Family: Solver-Tester (Programmer-Tester loop). 2 variants."""
    return [
        NamedArch(name="st_chain",
            agents=[(0, SOLVER), (1, TESTER)], edges=[(0, 1)], sequence=[0, 1],
            description="Solver writes code, Tester runs tests.",
            tags=("chain", "family_st")),
        NamedArch(name="st_loop",
            agents=[(0, SOLVER), (1, TESTER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Test-driven loop; Solver fixes failed tests in next cycle.",
            tags=("loop", "family_st")),
    ]


def fam_expert_verified() -> list[NamedArch]:
    """Family: Expert + Verifier. 2 variants."""
    return [
        NamedArch(name="expert_verified",
            agents=[(0, EXPERT), (1, VERIFIER)], edges=[(0, 1)], sequence=[0, 1],
            description="Expert proposes, Verifier independently checks.",
            tags=("chain", "family_expert_verified")),
        NamedArch(name="expert_verified_loop",
            agents=[(0, EXPERT), (1, VERIFIER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Expert ⟷ Verifier closed loop.",
            tags=("loop", "family_expert_verified")),
    ]


def fam_expert_researcher() -> list[NamedArch]:
    """Family: Expert + Researcher (research-augmented domain expert). 2 variants."""
    return [
        NamedArch(name="er_chain",
            agents=[(0, RESEARCHER), (1, EXPERT)], edges=[(0, 1)], sequence=[0, 1],
            description="Researcher gathers facts → Expert reasons with them.",
            tags=("chain", "family_er")),
        NamedArch(name="er_verified",
            agents=[(0, RESEARCHER), (1, EXPERT), (2, VERIFIER)],
            edges=[(0, 1), (1, 2), (0, 2)], sequence=[0, 1, 2],
            description="Researcher → Expert → Verifier (Researcher facts also visible to Verifier).",
            tags=("chain", "family_er")),
    ]


def fam_expert_planned() -> list[NamedArch]:
    """Family: Planner decomposes, then Expert (full-tool) executes. 2 variants."""
    return [
        NamedArch(name="expert_planned",
            agents=[(0, PLANNER), (1, EXPERT)], edges=[(0, 1)], sequence=[0, 1],
            description="Planner decomposes → Expert executes with full tool pool.",
            tags=("chain", "family_expert_planned")),
        NamedArch(name="expert_planned_verified",
            agents=[(0, PLANNER), (1, EXPERT), (2, VERIFIER)],
            edges=[(0, 1), (1, 2), (0, 2)], sequence=[0, 1, 2],
            description="Planner → Expert → Verifier.",
            tags=("chain", "family_expert_planned")),
    ]


def fam_expert_tested() -> list[NamedArch]:
    """Family: Expert as programmer + Tester. 2 variants."""
    return [
        NamedArch(name="expert_tested",
            agents=[(0, EXPERT), (1, TESTER)], edges=[(0, 1)], sequence=[0, 1],
            description="Expert writes code (can consult web/pdf) → Tester runs tests.",
            tags=("chain", "family_expert_tested")),
        NamedArch(name="expert_tested_loop",
            agents=[(0, EXPERT), (1, TESTER)], edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="Expert ⟷ Tester (test failures drive Expert's next-cycle fixes).",
            tags=("loop", "family_expert_tested")),
    ]


# ----- 3-agent (13 families) -------------------------------------------

def fam_plan_solve_verify() -> list[NamedArch]:
    """Family: Plan → Solve → Verify (classic). 1 variant."""
    return [NamedArch(name="psv",
        agents=[(0, PLANNER), (1, SOLVER), (2, VERIFIER)],
        edges=[(0, 1), (1, 2), (0, 2)], sequence=[0, 1, 2],
        description="Plan → Solve → Verify, with Plan also visible to Verifier.",
        tags=("chain", "family_psv"))]


def fam_plan_solve_critique() -> list[NamedArch]:
    """Family: Plan → Solve → Critic (subjective-review variant). 1 variant."""
    return [NamedArch(name="psc",
        agents=[(0, PLANNER), (1, SOLVER), (2, CRITIC)],
        edges=[(0, 1), (1, 2), (0, 2)], sequence=[0, 1, 2],
        description="Plan → Solve → Critic (subjective-review variant; no rule check).",
        tags=("chain", "family_psc"))]


def fam_expert_team() -> list[NamedArch]:
    """Family: Expert + Critic + Refiner (size 3). 1 variant."""
    return [NamedArch(name="expert_critiqued_refined",
        agents=[(0, EXPERT), (1, CRITIC), (2, REFINER)],
        edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
        description="Expert produces, Critic flags issues, Refiner integrates.",
        tags=("triangle", "family_expert_team"))]


def fam_solver_critic_verifier() -> list[NamedArch]:
    """Family: Solver in triangle with Critic + Verifier. 2 variants."""
    return [
        NamedArch(name="scv",
            agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="Solver heard by both Critic and Verifier; Critic also informs Verifier.",
            tags=("triangle", "family_scv")),
        NamedArch(name="scv_loop",
            agents=[(0, SOLVER), (1, CRITIC), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 0)], sequence=[0, 1, 2],
            description="SCV with Verifier feedback to Solver (multi-cycle refinement).",
            tags=("triangle", "loop", "family_scv")),
    ]


def fam_solver_critic_refiner() -> list[NamedArch]:
    """Family: Solver-Critic-Refiner (propose-critique-integrate). 2 variants."""
    return [
        NamedArch(name="scr",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="Solver proposes, Critic flags, Refiner integrates.",
            tags=("triangle", "family_scr")),
        NamedArch(name="scr_loop",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 0)], sequence=[0, 1, 2],
            description="SCR with Refiner feedback to Solver (iterative improvement).",
            tags=("triangle", "loop", "family_scr")),
    ]


def fam_programmer_tester_refiner() -> list[NamedArch]:
    """Family: Solver writes, Tester runs, Refiner integrates. 2 variants."""
    return [
        NamedArch(name="ptr",
            agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="Solver code → Tester runs → Refiner consolidates.",
            tags=("triangle", "family_ptr")),
        NamedArch(name="ptr_loop",
            agents=[(0, SOLVER), (1, TESTER), (2, REFINER)],
            edges=[(0, 1), (1, 0), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="PTR with Tester→Solver feedback (test-fix-retest loop).",
            tags=("triangle", "loop", "family_ptr")),
    ]


def fam_research_solve_verify() -> list[NamedArch]:
    """Family: Researcher → Solver → Verifier. 2 variants."""
    return [
        NamedArch(name="rsv",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (1, 2)], sequence=[0, 1, 2],
            description="Researcher → Solver → Verifier (RAG with verification).",
            tags=("chain", "family_rsv")),
        NamedArch(name="rsv_full",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="RSV with Researcher also visible to Verifier.",
            tags=("chain", "family_rsv")),
    ]


def fam_moa_mini() -> list[NamedArch]:
    """Family: MoA-mini (2 Solvers, no aggregator). 1 variant."""
    return [NamedArch(name="moa_2s_indep",
        agents=[(0, SOLVER), (1, SOLVER)], edges=[], sequence=[0, 1],
        description="MoA mini (no Refiner): 2 independent Solvers, Synth picks.",
        tags=("moa", "family_moa_mini"))]


def fam_mad_mini() -> list[NamedArch]:
    """Family: MAD-mini (2 Solvers debate, ± Judge). 2 variants."""
    return [
        NamedArch(name="mad_2sj",
            agents=[(0, SOLVER), (1, SOLVER), (2, VERIFIER)],
            edges=[(0, 1), (1, 0), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="MAD mini: 2 Solvers debate, Verifier judges.",
            tags=("debate", "family_mad_mini")),
        NamedArch(name="mad_2s_pure",
            agents=[(0, SOLVER), (1, SOLVER)],
            edges=[(0, 1), (1, 0)], sequence=[0, 1],
            description="MAD mini (no Judge): pure 2-Solver debate, Synth selects.",
            tags=("debate", "family_mad_mini")),
    ]


def fam_tdd_iterate() -> list[NamedArch]:
    """Family: TDD (Tester writes tests FIRST, then Solver, then Refiner). 2 variants."""
    return [
        NamedArch(name="tdd_chain",
            agents=[(0, TESTER), (1, SOLVER), (2, REFINER)],
            edges=[(0, 1), (1, 2), (0, 2)], sequence=[0, 1, 2],
            description="Tester writes failing tests → Solver makes them pass → Refiner cleans up.",
            tags=("chain", "family_tdd")),
        NamedArch(name="tdd_loop",
            agents=[(0, TESTER), (1, SOLVER), (2, REFINER)],
            edges=[(0, 1), (1, 2), (1, 0), (2, 0)], sequence=[0, 1, 2],
            description="TDD with tester re-checking after Solver and Refiner edits.",
            tags=("loop", "family_tdd")),
    ]


def fam_spec_first() -> list[NamedArch]:
    """Family: Planner writes spec → Tester writes tests → Solver implements. 2 variants."""
    return [
        NamedArch(name="spec_first",
            agents=[(0, PLANNER), (1, TESTER), (2, SOLVER)],
            edges=[(0, 1), (0, 2), (1, 2)], sequence=[0, 1, 2],
            description="Planner specs → Tester turns spec into tests → Solver implements.",
            tags=("chain", "family_spec_first")),
        NamedArch(name="spec_first_loop",
            agents=[(0, PLANNER), (1, TESTER), (2, SOLVER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 1)], sequence=[0, 1, 2],
            description="Spec-first with Solver→Tester re-verification loop.",
            tags=("loop", "family_spec_first")),
    ]


def fam_research_code() -> list[NamedArch]:
    """Family: Researcher → Solver → Tester (web-aware code). 2 variants."""
    return [
        NamedArch(name="rsct_chain",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, TESTER)],
            edges=[(0, 1), (1, 2)], sequence=[0, 1, 2],
            description="Researcher finds docs/algo → Solver codes → Tester runs unit tests.",
            tags=("chain", "family_rsct")),
        NamedArch(name="rsct_loop",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, TESTER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 1)], sequence=[0, 1, 2],
            description="Research-code-test with Tester→Solver fix loop.",
            tags=("loop", "family_rsct")),
    ]


def fam_expert_diversity() -> list[NamedArch]:
    """Family: heterogeneous attack — tool-using Expert + pure-reasoning Solver. 1 variant."""
    return [NamedArch(name="expert_solver_refiner",
        agents=[(0, EXPERT), (1, SOLVER), (2, REFINER)],
        edges=[(0, 2), (1, 2)], sequence=[0, 1, 2],
        description="Expert (with tools) + Solver (reasoning) in parallel, Refiner synthesizes.",
        tags=("triangle", "fanin", "family_expert_diversity"))]


# ----- 4-5 agent (11 families) -----------------------------------------

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
                name=name, agents=agents, edges=edges,
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
            name=f"moa_{n}sr", agents=agents, edges=edges,
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
            name=f"vcouncil_{n}v", agents=agents, edges=edges,
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
            name=f"ccouncil_{n}c", agents=agents, edges=edges,
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
            name=f"rcouncil_{n}r", agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"{n} Researchers gather aspects → Solver → Verifier.",
            tags=("council", "family_rcouncil"),
        ))
    # Also a 3-Researcher + Solver only (no Verifier)
    if 3 + 1 <= ARCH.n_max:
        out.append(NamedArch(
            name="rcouncil_3r_solver",
            agents=[(0, RESEARCHER), (1, RESEARCHER), (2, RESEARCHER), (3, SOLVER)],
            edges=[(0, 3), (1, 3), (2, 3)], sequence=[0, 1, 2, 3],
            description="3 Researchers (different aspects) → 1 Solver.",
            tags=("council", "family_rcouncil"),
        ))
    return out


def fam_plan_research_solve_verify() -> list[NamedArch]:
    """Family: Plan + Research → Solve → Verify. 2 variants."""
    return [
        NamedArch(name="prsv",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
            edges=[(0, 1), (1, 2), (0, 2), (2, 3)], sequence=[0, 1, 2, 3],
            description="Plan → Research → Solve → Verify (full pipeline).",
            tags=("pipeline", "family_prsv")),
        NamedArch(name="prsv_parallel",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
            edges=[(0, 2), (1, 2), (2, 3), (0, 3)], sequence=[0, 1, 2, 3],
            description="Plan and Research run independently then both feed Solver → Verify.",
            tags=("pipeline", "family_prsv")),
    ]


def fam_planner_hub() -> list[NamedArch]:
    """Family: Planner-Hub star (HuggingGPT-style). 3 variants."""
    return [
        NamedArch(name="hub_4",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 3), (0, 3)], sequence=[0, 1, 2, 3],
            description="Planner-Hub (small): Planner dispatches Researcher + Solver, Verifier checks.",
            tags=("hub", "star", "family_hub")),
        NamedArch(name="hub_5",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, TESTER), (4, REFINER)],
            edges=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)], sequence=[0, 1, 2, 3, 4],
            description="Planner-Hub: 3 specialists each handles one sub-task; Refiner aggregates.",
            tags=("hub", "star", "family_hub")),
        NamedArch(name="hub_6",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, TESTER), (4, CRITIC), (5, REFINER)],
            edges=[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 5), (4, 5)],
            sequence=[0, 1, 2, 3, 4, 5],
            description="Planner-Hub (full): 4 specialists in parallel → Refiner.",
            tags=("hub", "star", "family_hub")),
    ]


def fam_tot_fanout() -> list[NamedArch]:
    """Family: Tree-of-Thoughts fan-out (Planner → N Solvers → Refiner). 3 variants."""
    out = []
    for n_solvers in (2, 3, 4):
        if 1 + n_solvers + 1 > ARCH.n_max:
            continue
        refiner_slot = n_solvers + 1
        agents = ([(0, PLANNER)] + [(i, SOLVER) for i in range(1, n_solvers + 1)]
                  + [(refiner_slot, REFINER)])
        edges = ([(0, i) for i in range(1, n_solvers + 1)]
                 + [(i, refiner_slot) for i in range(1, n_solvers + 1)])
        out.append(NamedArch(
            name=f"tot_{n_solvers}solvers", agents=agents, edges=edges,
            sequence=list(range(len(agents))),
            description=f"ToT fan-out: Planner → {n_solvers} parallel Solvers → Refiner.",
            tags=("tot", "fanout", "family_tot"),
        ))
    return out


def fam_critic_refiner_loop() -> list[NamedArch]:
    """Family: Solver / Critic / Refiner with cycle loop. 2 variants."""
    return [
        NamedArch(name="crloop_3",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER)],
            edges=[(0, 1), (1, 2), (2, 0)], sequence=[0, 1, 2],
            description="3-cycle: Solver → Critic → Refiner → Solver (iterate via cycle).",
            tags=("loop", "family_crloop")),
        NamedArch(name="crloop_4_with_verifier",
            agents=[(0, SOLVER), (1, CRITIC), (2, REFINER), (3, VERIFIER)],
            edges=[(0, 1), (1, 2), (2, 0), (2, 3)], sequence=[0, 1, 2, 3],
            description="Critic-Refiner loop with Verifier as final gate.",
            tags=("loop", "family_crloop")),
    ]


def fam_tester_council() -> list[NamedArch]:
    """Family: 1 Solver + K Testers (parallel multi-angle testing). 2 variants."""
    return [
        NamedArch(name="tcouncil_3t_iter",
            agents=[(0, SOLVER), (1, TESTER), (2, TESTER), (3, TESTER)],
            edges=[(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)],
            sequence=[0, 1, 2, 3],
            description="1 Solver + 3 Testers; Testers all feed back to Solver for iteration.",
            tags=("council", "loop", "family_tcouncil")),
        NamedArch(name="tcouncil_3t_refine",
            agents=[(0, SOLVER), (1, TESTER), (2, TESTER), (3, TESTER), (4, REFINER)],
            edges=[(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
            sequence=[0, 1, 2, 3, 4],
            description="Solver → 3 parallel Testers → Refiner aggregates (5-agent council).",
            tags=("council", "family_tcouncil")),
    ]


def fam_expert_council() -> list[NamedArch]:
    """Family: panel of K Experts (specialist debate / consensus). 2 variants."""
    return [
        NamedArch(name="ecouncil_2e",
            agents=[(0, EXPERT), (1, EXPERT), (2, REFINER)],
            edges=[(0, 2), (1, 2)], sequence=[0, 1, 2],
            description="2 Experts (different domain angles) → Refiner consolidates.",
            tags=("council", "family_ecouncil")),
        NamedArch(name="ecouncil_3e_critic",
            agents=[(0, EXPERT), (1, EXPERT), (2, EXPERT), (3, CRITIC)],
            edges=[(0, 3), (1, 3), (2, 3), (0, 1), (1, 2), (0, 2)],
            sequence=[0, 1, 2, 3],
            description="3 Experts debate (full mesh) + Critic adjudicates.",
            tags=("council", "debate", "family_ecouncil")),
    ]


# ----- Rich (6 families) -----------------------------------------------

def fam_full_team() -> list[NamedArch]:
    """Family: Full multi-role team. 2 variants."""
    return [
        NamedArch(name="full_5",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, VERIFIER), (4, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)],
            sequence=[0, 1, 2, 3, 4],
            description="5-role team: Plan + Research → Solve → Verify + Refine.",
            tags=("rich", "family_full")),
        NamedArch(name="full_6",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, SOLVER), (4, CRITIC), (5, VERIFIER)],
            edges=[(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 5), (4, 5)],
            sequence=[0, 1, 2, 3, 4, 5],
            description="6-role team: Plan + Research → 2 Solvers → Critic → Verifier.",
            tags=("rich", "family_full")),
    ]


def fam_tester_centric() -> list[NamedArch]:
    """Family: Tester is central (code-heavy task). 2 variants."""
    return [
        NamedArch(name="tester_5",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, TESTER), (3, CRITIC), (4, VERIFIER)],
            edges=[(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)], sequence=[0, 1, 2, 3, 4],
            description="Researcher → Solver writes code → Tester runs → Critic + Verifier check.",
            tags=("rich", "family_tester_centric")),
        NamedArch(name="tester_4",
            agents=[(0, SOLVER), (1, TESTER), (2, TESTER), (3, REFINER)],
            edges=[(0, 1), (0, 2), (1, 3), (2, 3)], sequence=[0, 1, 2, 3],
            description="Solver + 2 independent Testers (different test angles) + Refiner.",
            tags=("rich", "family_tester_centric")),
    ]


def fam_mesh_then_aggregate() -> list[NamedArch]:
    """Family: Multi-Solver mesh → aggregator → final check. 2 variants."""
    return [
        NamedArch(name="mesh_then_agg_5",
            agents=[(0, SOLVER), (1, SOLVER), (2, SOLVER), (3, REFINER), (4, VERIFIER)],
            edges=full_mesh([0, 1, 2]) + [(0, 3), (1, 3), (2, 3), (3, 4)],
            sequence=[0, 1, 2, 3, 4],
            description="3 Solvers debate (mesh) → Refiner integrates → Verifier checks.",
            tags=("rich", "mesh", "family_mesh_then_agg")),
        NamedArch(name="mesh_then_agg_4",
            agents=[(0, SOLVER), (1, SOLVER), (2, REFINER), (3, VERIFIER)],
            edges=[(0, 1), (1, 0), (0, 2), (1, 2), (2, 3)], sequence=[0, 1, 2, 3],
            description="2 Solvers debate → Refiner → Verifier.",
            tags=("rich", "mesh", "family_mesh_then_agg")),
    ]


def fam_research_then_mesh() -> list[NamedArch]:
    """Family: Research-then-mesh (RAG + debate). 2 variants."""
    return [
        NamedArch(name="research_mesh_4",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, SOLVER), (3, REFINER)],
            edges=[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3)],
            sequence=[0, 1, 2, 3],
            description="Researcher feeds 2 Solvers; Solvers debate; Refiner aggregates.",
            tags=("rich", "family_research_mesh")),
        NamedArch(name="research_mesh_5",
            agents=[(0, RESEARCHER), (1, SOLVER), (2, SOLVER), (3, SOLVER), (4, REFINER)],
            edges=[(0, 1), (0, 2), (0, 3)] + full_mesh([1, 2, 3])
                  + [(1, 4), (2, 4), (3, 4)],
            sequence=[0, 1, 2, 3, 4],
            description="Researcher feeds 3 Solvers; full debate; Refiner aggregates.",
            tags=("rich", "family_research_mesh")),
    ]


def fam_sop_linear() -> list[NamedArch]:
    """Family: SOP-style long linear pipeline (ChatDev / MetaGPT). 2 variants."""
    return [
        NamedArch(name="sop_4",
            agents=[(0, PLANNER), (1, SOLVER), (2, TESTER), (3, VERIFIER)],
            edges=[(0, 1), (1, 2), (2, 3)], sequence=[0, 1, 2, 3],
            description="SOP linear pipeline: Plan → Solve → Test → Verify "
                        "(ChatDev/MetaGPT style; minimal cross-talk).",
            tags=("pipeline", "sop", "family_sop")),
        NamedArch(name="sop_5",
            agents=[(0, PLANNER), (1, RESEARCHER), (2, SOLVER), (3, TESTER), (4, REFINER)],
            edges=[(0, 1), (1, 2), (2, 3), (3, 4)], sequence=[0, 1, 2, 3, 4],
            description="SOP linear pipeline (full): Plan → Research → Solve → Test → Refine.",
            tags=("pipeline", "sop", "family_sop")),
    ]


def fam_tot_deep() -> list[NamedArch]:
    """Family: ToT-style 2-layer expansion (deeper search). 1 variant."""
    return [NamedArch(name="tot_deep_6",
        agents=[(0, PLANNER), (1, SOLVER), (2, SOLVER), (3, CRITIC), (4, CRITIC), (5, REFINER)],
        edges=[(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
        sequence=[0, 1, 2, 3, 4, 5],
        description="ToT deeper: Planner → 2 Solvers → each gets a Critic → Refiner aggregates "
                    "(2-layer expansion vs single-layer fan-out in fam_tot_fanout).",
        tags=("tot", "fanout", "family_tot_deep"))]


# =======================================================================
# Master list — order matters only for inline group counts (readability).
# Total: 3 + 9 + 13 + 11 + 6 = 42 families.
# =======================================================================

CANONICAL_FAMILIES = [
    # Singles (3)
    fam_single_solver, fam_researcher_verified, fam_expert_only,
    # 2-agent (9)
    fam_solver_verifier, fam_solver_critic, fam_planner_solver,
    fam_self_rag, fam_solver_tester, fam_expert_verified,
    fam_expert_researcher, fam_expert_planned, fam_expert_tested,
    # 3-agent (13)
    fam_plan_solve_verify, fam_plan_solve_critique, fam_expert_team,
    fam_solver_critic_verifier, fam_solver_critic_refiner,
    fam_programmer_tester_refiner, fam_research_solve_verify,
    fam_moa_mini, fam_mad_mini,
    fam_tdd_iterate, fam_spec_first, fam_research_code,
    fam_expert_diversity,
    # 4-5 agent (11)
    fam_mad_debate, fam_moa_fanin, fam_verifier_council, fam_critic_council,
    fam_researcher_council, fam_plan_research_solve_verify,
    fam_planner_hub, fam_tot_fanout, fam_critic_refiner_loop,
    fam_tester_council, fam_expert_council,
    # Rich (6)
    fam_full_team, fam_tester_centric, fam_mesh_then_aggregate,
    fam_research_then_mesh, fam_sop_linear, fam_tot_deep,
]
