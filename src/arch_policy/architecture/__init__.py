"""Architecture: typed logits / typed targets / sampler / library / encoder.

Public API:
    - ArchLogits, ArchTargets, active_pair_mask              (spec.py)
    - ConcreteArch, sample_arch, sample_pl,
      log_prob_gates, log_prob_roles, log_prob_edges,
      log_prob_pl, log_prob_joint                            (sampler.py)
    - NamedArch, family_of, ANSWERER_ROLES, CANONICAL_FAMILIES,
      canonical_library, imperfect_library, random_archs,
      core_library (alias for canonical), full_library, full_mesh,
      role-id constants (PLANNER, DECOMPOSER, SOLVER, ...)   (library.py)
    - encode_named_arch, encode_library                      (encoder.py)
"""

from .encoder import encode_library, encode_named_arch
from .library import (
    ANSWERER_ROLES,
    CANONICAL_FAMILIES,
    CRITIC,
    DECOMPOSER,
    PLANNER,
    REFINER,
    RESEARCHER,
    ROLE_POOL,
    ROLE_TO_ID,
    SOLVER,
    TESTER,
    VERIFIER,
    NamedArch,
    canonical_library,
    core_library,
    family_of,
    full_library,
    full_mesh,
    imperfect_library,
    random_archs,
)
from .sampler import (
    ConcreteArch,
    log_prob_edges,
    log_prob_gates,
    log_prob_joint,
    log_prob_pl,
    log_prob_roles,
    sample_arch,
    sample_pl,
)
from .spec import ArchLogits, ArchTargets, active_pair_mask

__all__ = [
    "ANSWERER_ROLES",
    "ArchLogits",
    "ArchTargets",
    "CANONICAL_FAMILIES",
    "CRITIC",
    "ConcreteArch",
    "DECOMPOSER",
    "NamedArch",
    "PLANNER",
    "REFINER",
    "RESEARCHER",
    "ROLE_POOL",
    "ROLE_TO_ID",
    "SOLVER",
    "TESTER",
    "VERIFIER",
    "active_pair_mask",
    "canonical_library",
    "core_library",
    "encode_library",
    "encode_named_arch",
    "family_of",
    "full_library",
    "full_mesh",
    "imperfect_library",
    "log_prob_edges",
    "log_prob_gates",
    "log_prob_joint",
    "log_prob_pl",
    "log_prob_roles",
    "random_archs",
    "sample_arch",
    "sample_pl",
]
