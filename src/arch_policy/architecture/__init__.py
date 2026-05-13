"""Architecture: typed logits / typed targets / sampler / library / encoder.

Public API:
    - ArchLogits, ArchTargets, active_pair_mask              (spec.py)
    - ConcreteArch, sample_arch, sample_pl,
      log_prob_gates, log_prob_roles, log_prob_edges,
      log_prob_pl, log_prob_joint                            (sampler.py)
    - NamedArch, core_library, full_library, full_mesh,
      random_archs, random_perturbations,
      role-id constants (PLANNER, SOLVER, ...)               (library.py)
    - encode_named_arch, encode_library                      (encoder.py)
"""

from .encoder import encode_library, encode_named_arch
from .library import (
    CRITIC,
    PLANNER,
    REFINER,
    RESEARCHER,
    ROLE_TO_ID,
    SOLVER,
    TOOLUSER,
    VERIFIER,
    NamedArch,
    core_library,
    full_library,
    full_mesh,
    random_archs,
    random_perturbations,
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
    "ArchLogits",
    "ArchTargets",
    "CRITIC",
    "ConcreteArch",
    "NamedArch",
    "PLANNER",
    "REFINER",
    "RESEARCHER",
    "ROLE_TO_ID",
    "SOLVER",
    "TOOLUSER",
    "VERIFIER",
    "active_pair_mask",
    "core_library",
    "encode_library",
    "encode_named_arch",
    "full_library",
    "full_mesh",
    "log_prob_edges",
    "log_prob_gates",
    "log_prob_joint",
    "log_prob_pl",
    "log_prob_roles",
    "random_archs",
    "random_perturbations",
    "sample_arch",
    "sample_pl",
]
