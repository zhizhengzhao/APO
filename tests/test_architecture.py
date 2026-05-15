"""Architecture tests (v3 typed heads + Plackett-Luce sequence)."""

from __future__ import annotations

import math

import torch

from arch_policy import (
    ARCH,
    ArchLogits,
    ArchTargets,
    ConcreteArch,
    encode_library,
    encode_named_arch,
    full_library,
    log_prob_edges,
    log_prob_gates,
    log_prob_joint,
    log_prob_pl,
    log_prob_roles,
    sample_arch,
    sample_pl,
)
from arch_policy.architecture.spec import active_pair_mask


def _make_random_logits(seed: int = 0) -> ArchLogits:
    torch.manual_seed(seed)
    N, R = ARCH.n_max, ARCH.k_roles
    logits = ArchLogits(
        gate_logits=torch.randn(N) * 0.5,
        role_logits=torch.randn(N, R) * 0.5,
        edge_logits=torch.randn(N, N) * 0.5,
        seq_scores=torch.randn(N) * 0.5,
    )
    eye = torch.eye(N, dtype=torch.bool)
    logits.edge_logits = logits.edge_logits.masked_fill(eye, -1e9)
    return logits


def test_logits_shapes_and_validate():
    logits = _make_random_logits()
    logits.validate(ARCH)


def test_sampler_basic():
    logits = _make_random_logits()
    arch = sample_arch(logits, deterministic=False)
    assert isinstance(arch, ConcreteArch)
    assert arch.active_mask.dtype == torch.bool
    assert arch.edges.dtype == torch.bool
    assert arch.sequence.dtype == torch.long
    # No self-loops
    assert not arch.edges.diagonal().any()
    # Edges only between actives
    pair = active_pair_mask(arch.active_mask)
    assert (arch.edges & ~pair).sum() == 0
    # At least one active
    assert arch.active_mask.any()
    # Sequence is a permutation of actives (no repeats, all actives present)
    seq_set = set(arch.sequence.tolist())
    actives = set(i for i in range(ARCH.n_max) if arch.active_mask[i].item())
    assert seq_set == actives, f"seq {seq_set} != actives {actives}"
    assert len(arch.sequence) == len(actives)


def test_sampler_deterministic():
    logits = _make_random_logits()
    a1 = sample_arch(logits, deterministic=True)
    a2 = sample_arch(logits, deterministic=True)
    assert torch.equal(a1.active_mask, a2.active_mask)
    assert torch.equal(a1.roles, a2.roles)
    assert torch.equal(a1.edges, a2.edges)
    assert torch.equal(a1.sequence, a2.sequence)


def test_force_at_least_one_active():
    """If gate_logits are all very low, sampler must still force one active."""
    N, R = ARCH.n_max, ARCH.k_roles
    logits = ArchLogits(
        gate_logits=torch.full((N,), -10.0),
        role_logits=torch.zeros(N, R),
        edge_logits=torch.zeros(N, N),
        seq_scores=torch.zeros(N),
    )
    eye = torch.eye(N, dtype=torch.bool)
    logits.edge_logits = logits.edge_logits.masked_fill(eye, -1e9)
    arch = sample_arch(logits, deterministic=False)
    assert arch.active_mask.sum() == 1


def test_pl_log_prob_known():
    """Hand-checked PL log_prob matches the formula."""
    scores = torch.tensor([3.0, 1.0, 2.0, 0.0, 0.0, 0.0])
    perm = torch.tensor([0, 2, 1])
    expected = (
        3.0 - math.log(math.exp(3) + math.exp(1) + math.exp(2))
        + 2.0 - math.log(math.exp(2) + math.exp(1))
        # last position has remaining = {1}, log P(1) = 0
    )
    lp = log_prob_pl(scores, perm)
    assert abs(lp.item() - expected) < 1e-5, (lp.item(), expected)


def test_pl_sample_argmax_matches_argsort():
    scores = torch.tensor([1.0, 5.0, 3.0, 2.0])
    active = torch.tensor([0, 1, 2, 3])
    perm = sample_pl(scores, active, deterministic=True)
    expected = torch.tensor([1, 2, 3, 0])  # argsort descending
    assert torch.equal(perm, expected), (perm.tolist(), expected.tolist())


def test_log_prob_joint_grads_flow_for_nontrivial_arch():
    """When sampled arch has multiple actives + edges + non-trivial seq,
    grads flow through all 4 typed logits."""
    N, R = ARCH.n_max, ARCH.k_roles
    g = torch.full((N,), 2.0, requires_grad=True)
    Q = torch.randn(N, R, requires_grad=True)
    E = torch.randn(N, N, requires_grad=True)
    S = torch.randn(N, requires_grad=True)
    eye = torch.eye(N, dtype=torch.bool)
    E_masked = E.masked_fill(eye, -1e9)
    logits = ArchLogits(g, Q, E_masked, S)
    with torch.no_grad():
        arch = sample_arch(logits)
    assert arch.n_active >= 2
    lp = log_prob_joint(logits, arch)
    (-lp).backward()
    for name, t in (("g", g), ("Q", Q), ("E", E), ("S", S)):
        assert t.grad is not None and t.grad.norm().item() > 0, f"no grad on {name}"


def test_library_validates():
    lib = full_library(seed=0)
    assert len(lib) >= 30
    for arch in lib:
        arch.validate()
        # Sequence must be a permutation of active slots
        actives = {s for s, _ in arch.agents}
        assert set(arch.sequence) == actives
        assert len(arch.sequence) == len(actives)
        for _, r in arch.agents:
            assert 0 <= r < ARCH.k_roles


def test_library_no_duplicate_names():
    """No two NamedArch entries should share the same `name` (would inflate
    sampling weight and confuse diagnostics)."""
    from collections import Counter
    lib = full_library(seed=0)
    counts = Counter(a.name for a in lib)
    dups = {n: c for n, c in counts.items() if c > 1}
    assert not dups, f"duplicate NamedArch names: {dups}"


def test_library_no_structural_duplicates_in_canonical():
    """Two canonical entries with identical (agents, edges, sequence) would
    train the head twice on the same target; drop one."""
    from collections import Counter
    from arch_policy.architecture.library import canonical_library
    canon = canonical_library()
    keys = Counter(
        (tuple(sorted(a.agents)), tuple(sorted(a.edges)), tuple(a.sequence))
        for a in canon
    )
    struct_dups = {k: c for k, c in keys.items() if c > 1}
    if struct_dups:
        examples = []
        for k in list(struct_dups)[:3]:
            names = [a.name for a in canon
                     if (tuple(sorted(a.agents)), tuple(sorted(a.edges)),
                         tuple(a.sequence)) == k]
            examples.append(names)
        assert False, f"structurally identical canonical entries: {examples}"


def test_encoder_round_trip_to_targets():
    """encode → ArchTargets matches the NamedArch's structure."""
    lib = full_library(seed=0)
    for arch in lib[:8]:
        target = encode_named_arch(arch)
        target.validate(ARCH)
        actives = {s for s, _ in arch.agents}
        # Active mask matches
        for s in range(ARCH.n_max):
            assert (target.gates[s].item() == 1) == (s in actives)
        # Roles match
        for s, r in arch.agents:
            assert int(target.roles[s].item()) == r
        # Edges match
        for s, d in arch.edges:
            assert int(target.edges[s, d].item()) == 1
        # Seq matches
        assert target.seq.tolist() == arch.sequence


def test_encode_library_returns_list():
    lib = full_library(seed=0)
    enc = encode_library(lib)
    assert len(enc) == len(lib)
    for t in enc:
        assert isinstance(t, ArchTargets)


def test_log_prob_pieces_sum_to_joint():
    logits = _make_random_logits(seed=3)
    torch.manual_seed(7)
    arch = sample_arch(logits)
    parts = (
        log_prob_gates(logits.gate_logits, arch.active_mask)
        + log_prob_roles(logits.role_logits, arch.roles, arch.active_mask)
        + log_prob_edges(logits.edge_logits, arch.edges, arch.active_mask)
        + log_prob_pl(logits.seq_scores, arch.sequence)
    )
    joint = log_prob_joint(logits, arch)
    assert abs(parts.item() - joint.item()) < 1e-5


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    print("\nall architecture tests pass.")
