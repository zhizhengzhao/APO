"""Per-role tool whitelists. Role differentiation is STRUCTURAL (tools)
and via output marker (prompts.py), never prompt-engineered behaviors.

| Role       | Tools                                       |
|------------|---------------------------------------------|
| Planner    | — (plans in text; no computation)           |
| Expert     | all 5 (single-agent generalist)             |
| Solver     | python_exec                                 |
| Critic     | python_exec                                 |
| Verifier   | python_exec                                 |
| Refiner    | — (integrates in text; no computation)      |
| Researcher | web_search, arxiv_search, wikipedia_search  |
| Tester     | pytest_runner, python_exec                  |

Minimum-tool principle: a role gets only the tools its job requires.
Planner outputs a plan; Refiner integrates candidates — both pure
text-shaping operations, no need for compute. `python_exec` covers
sympy / pint / numpy / scipy via `import`.
"""

from __future__ import annotations


# Role names must match ArchSpec.role_names exactly.
ROLE_TOOL_POOLS: dict[str, frozenset[str]] = {
    "Planner":    frozenset(),
    "Expert":     frozenset({
        "python_exec", "pytest_runner",
        "web_search", "arxiv_search", "wikipedia_search",
    }),
    "Solver":     frozenset({"python_exec"}),
    "Critic":     frozenset({"python_exec"}),
    "Verifier":   frozenset({"python_exec"}),
    "Refiner":    frozenset(),
    "Researcher": frozenset({
        "web_search", "arxiv_search", "wikipedia_search",
    }),
    "Tester":     frozenset({"pytest_runner", "python_exec"}),
}


def allowed_tools_for(role_name: str) -> frozenset[str]:
    """Tool whitelist for `role_name`.

    Unknown role → empty set (defensive: if a new role is added to the
    head but not to this mapping, agents simply can't call tools rather
    than being silently unrestricted).
    """
    return ROLE_TOOL_POOLS.get(role_name, frozenset())


__all__ = ["ROLE_TOOL_POOLS", "allowed_tools_for"]
