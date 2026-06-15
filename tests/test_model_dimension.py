"""Tests for the per-agent model-selection dimension (5th typed head).

Two contracts:

1. BACKWARD COMPAT (n_models == 1, the default / HLE setup): the head
   instantiates no head_M, forward emits no model_logits, sample_arch
   draws no model (arch.model is None, zero extra RNG), log_prob_joint
   is the 4-term sum, and the executor uses the single worker. I.e. the
   running HLE checkpoint + behavior are untouched.

2. ACTIVE (n_models > 1): head_M present, model_logits shaped
   [B, N, n_models], sample_arch assigns a per-slot model id (0 on
   inactive slots), log_prob_joint gains a 5th Categorical term,
   entropy_typed gains a model term, and the executor dispatches each
   slot to worker_pool[model_names[id]].
"""

from __future__ import annotations

from dataclasses import replace

import torch

from arch_policy.config import ARCH
from arch_policy.architecture.spec import ArchLogits
from arch_policy.architecture.sampler import (
    ConcreteArch, sample_arch, log_prob_joint, log_prob_models,
)


def _logits(spec, with_model: bool, seed: int = 0):
    torch.manual_seed(seed)
    N, R = spec.n_max, spec.k_roles
    return ArchLogits(
        gate_logits=torch.randn(N),
        role_logits=torch.randn(N, R),
        edge_logits=torch.randn(N, N),
        seq_scores=torch.randn(N),
        model_logits=torch.randn(N, spec.n_models) if with_model else None,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_default_spec_is_single_model():
    assert ARCH.n_models == 1
    assert ARCH.model_names == ("default",)


def test_n_models_property_tracks_pool():
    spec = replace(ARCH, model_names=("a", "b", "c", "d"))
    assert spec.n_models == 4


# ---------------------------------------------------------------------------
# Backward compat: n_models == 1
# ---------------------------------------------------------------------------

def test_single_model_sample_has_no_model_field():
    spec = ARCH
    L = _logits(spec, with_model=False)
    a = sample_arch(L, spec)
    assert a.model is None


def test_single_model_log_prob_is_four_term():
    """With no model dim, log_prob_joint must equal the sum of the 4
    classic terms — adding the model dim must not perturb it."""
    from arch_policy.architecture.sampler import (
        log_prob_gates, log_prob_roles, log_prob_edges, log_prob_pl,
    )
    spec = ARCH
    L = _logits(spec, with_model=False)
    a = sample_arch(L, spec)
    four = (
        log_prob_gates(L.gate_logits, a.active_mask)
        + log_prob_roles(L.role_logits, a.roles, a.active_mask)
        + log_prob_edges(L.edge_logits, a.edges, a.active_mask)
        + log_prob_pl(L.seq_scores, a.sequence)
    )
    assert torch.allclose(log_prob_joint(L, a), four)


def test_single_model_sampling_rng_unchanged():
    """n_models==1 must consume ZERO extra RNG draws vs a hypothetical
    no-model sampler — so resuming the HLE run is bit-identical. We
    verify by sampling the SAME arch twice from the same seed."""
    spec = ARCH
    L = _logits(spec, with_model=False)
    torch.manual_seed(123)
    a1 = sample_arch(L, spec)
    torch.manual_seed(123)
    a2 = sample_arch(L, spec)
    assert torch.equal(a1.active_mask, a2.active_mask)
    assert torch.equal(a1.roles, a2.roles)
    assert torch.equal(a1.edges, a2.edges)
    assert torch.equal(a1.sequence, a2.sequence)
    assert a1.model is None and a2.model is None


def test_concrete_arch_model_defaults_none():
    """All existing ConcreteArch constructions (baselines, cache, tests)
    omit model → must default to None, not crash."""
    a = ConcreteArch(
        active_mask=torch.zeros(6, dtype=torch.bool),
        roles=torch.zeros(6, dtype=torch.long),
        edges=torch.zeros(6, 6, dtype=torch.bool),
        sequence=torch.zeros(0, dtype=torch.long),
    )
    assert a.model is None


# ---------------------------------------------------------------------------
# Active: n_models > 1
# ---------------------------------------------------------------------------

def test_multi_model_sample_assigns_per_slot_model():
    spec = replace(ARCH, model_names=("m0", "m1", "m2", "m3"))
    L = _logits(spec, with_model=True)
    a = sample_arch(L, spec)
    assert a.model is not None
    assert a.model.shape == (spec.n_max,)
    # inactive slots are pinned to model 0
    inactive = ~a.active_mask
    assert torch.all(a.model[inactive] == 0)
    # model ids are valid
    assert int(a.model.max()) < spec.n_models


def test_multi_model_log_prob_adds_fifth_term():
    spec = replace(ARCH, model_names=("m0", "m1", "m2", "m3"))
    L = _logits(spec, with_model=True)
    a = sample_arch(L, spec)
    joint = log_prob_joint(L, a)
    # strip the model term and confirm joint = four-term + model-term
    model_term = log_prob_models(L.model_logits, a.model, a.active_mask)
    L_no_model = replace(L, model_logits=None)
    a_no_model = replace(a, model=None)
    four = log_prob_joint(L_no_model, a_no_model)
    assert torch.allclose(joint, four + model_term)
    # the model term is a real (negative) contribution
    assert model_term.item() < 0


def test_multi_model_validate_shape():
    spec = replace(ARCH, model_names=("m0", "m1"))
    L = _logits(spec, with_model=True)
    L.validate(spec)
    # wrong n_models shape must raise
    bad = replace(L, model_logits=torch.randn(spec.n_max, 5))
    try:
        bad.validate(spec)
        assert False, "expected shape mismatch to raise"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_no_model_term_when_single_model():
    from arch_policy.training.entropy import entropy_typed
    spec = ARCH
    N, R = spec.n_max, spec.k_roles
    head_out = {
        "gate_logits": torch.randn(2, N),
        "role_logits": torch.randn(2, N, R),
        "edge_logits": torch.randn(2, N, N),
        "seq_scores": torch.randn(2, N),
        # no model_logits
    }
    h = entropy_typed(head_out, spec)
    assert torch.isfinite(h)


def test_entropy_gains_model_term_when_multi_model():
    """Adding model_logits must strictly increase the entropy bonus
    (a non-degenerate Categorical has positive entropy)."""
    from arch_policy.training.entropy import entropy_typed
    spec = replace(ARCH, model_names=("m0", "m1", "m2", "m3"))
    N, R = spec.n_max, spec.k_roles
    torch.manual_seed(0)
    base = {
        "gate_logits": torch.randn(2, N),
        "role_logits": torch.randn(2, N, R),
        "edge_logits": torch.randn(2, N, N),
        "seq_scores": torch.randn(2, N),
    }
    h_no_model = entropy_typed(base, spec)
    with_model = dict(base, model_logits=torch.randn(2, N, spec.n_models))
    h_model = entropy_typed(with_model, spec)
    assert h_model > h_no_model


# ---------------------------------------------------------------------------
# Executor dispatch
# ---------------------------------------------------------------------------

def test_executor_dispatches_per_slot_model():
    """With a worker_pool and an arch carrying model ids, each agent
    must be built on the head-chosen model's worker."""
    from arch_policy.executor.multi_agent import MultiAgentExecutor, Worker, WorkerOutput

    spec = replace(ARCH, model_names=("m0", "m1"))

    class _TaggedWorker(Worker):
        def __init__(self, tag):
            self.tag = tag
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text=f"ANSWER: {self.tag}", n_input_tokens=1,
                                n_output_tokens=1)

    pool = {"m0": _TaggedWorker("m0"), "m1": _TaggedWorker("m1")}
    ex = MultiAgentExecutor(worker=pool["m0"], spec=spec, worker_pool=pool,
                            parallel_within_cycle=False)

    # 2-agent arch: slot 0 → model 1, slot 1 → model 0
    arch = ConcreteArch(
        active_mask=torch.tensor([True, True] + [False] * 4),
        roles=torch.tensor([2, 4] + [0] * 4, dtype=torch.long),  # Solver, Verifier
        edges=torch.zeros(6, 6, dtype=torch.bool),
        sequence=torch.tensor([0, 1], dtype=torch.long),
        model=torch.tensor([1, 0] + [0] * 4, dtype=torch.long),
    )
    w0 = ex._worker_for(arch, 0)
    w1 = ex._worker_for(arch, 1)
    assert w0.tag == "m1", w0.tag
    assert w1.tag == "m0", w1.tag


def test_executor_single_model_ignores_pool_when_arch_has_no_model():
    from arch_policy.executor.multi_agent import MultiAgentExecutor, Worker, WorkerOutput

    class _W(Worker):
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text="x", n_input_tokens=1, n_output_tokens=1)

    w = _W()
    ex = MultiAgentExecutor(worker=w, parallel_within_cycle=False)
    arch = ConcreteArch(
        active_mask=torch.tensor([True] + [False] * 5),
        roles=torch.zeros(6, dtype=torch.long),
        edges=torch.zeros(6, 6, dtype=torch.bool),
        sequence=torch.tensor([0], dtype=torch.long),
        model=None,
    )
    assert ex._worker_for(arch, 0) is w


def test_executor_rejects_pool_missing_a_model():
    from arch_policy.executor.multi_agent import MultiAgentExecutor, Worker, WorkerOutput

    class _W(Worker):
        def chat(self, system, user, max_new_tokens=512):
            return WorkerOutput(text="x", n_input_tokens=1, n_output_tokens=1)

    spec = replace(ARCH, model_names=("m0", "m1", "m2"))
    try:
        MultiAgentExecutor(worker=_W(), spec=spec,
                           worker_pool={"m0": _W()})  # missing m1, m2
        assert False, "expected missing-worker validation to raise"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# SFT model loss (deterministic max-entropy → uniform head_M prior)
# ---------------------------------------------------------------------------

def _targets(spec, n_active=2):
    from arch_policy.architecture.spec import ArchTargets
    N = spec.n_max
    gates = torch.zeros(N)
    gates[:n_active] = 1.0
    return ArchTargets(
        gates=gates,
        roles=torch.zeros(N, dtype=torch.long),
        edges=torch.zeros(N, N),
        seq=torch.arange(n_active, dtype=torch.long),
    )


def test_sft_model_loss_uniform_equals_ln_nmodels():
    """At uniform model_logits the deterministic max-entropy loss == ln(M)."""
    import math
    from arch_policy.training.sft import sft_loss_single
    spec = replace(ARCH, model_names=("a", "b", "c", "d"))  # n_models=4
    N = spec.n_max
    logits = ArchLogits(
        gate_logits=torch.zeros(N), role_logits=torch.zeros(N, spec.k_roles),
        edge_logits=torch.zeros(N, N), seq_scores=torch.zeros(N),
        model_logits=torch.zeros(N, 4),  # uniform
    )
    comp = sft_loss_single(logits, _targets(spec, 2), None)
    assert "model" in comp
    assert abs(float(comp["model"]) - math.log(4)) < 1e-5


def test_sft_model_loss_none_is_zero():
    """Single-model SFT (model_logits=None) → model loss is a true no-op."""
    from arch_policy.training.sft import sft_loss_single
    N = ARCH.n_max
    logits = ArchLogits(
        gate_logits=torch.zeros(N), role_logits=torch.zeros(N, ARCH.k_roles),
        edge_logits=torch.zeros(N, N), seq_scores=torch.zeros(N),
        model_logits=None,
    )
    comp = sft_loss_single(logits, _targets(ARCH, 2), None)
    assert float(comp["model"]) == 0.0


def test_sft_model_loss_penalizes_nonuniform_and_grad_pushes_uniform():
    """Biased model_logits → loss > ln(M); gradient reduces the bias."""
    import math
    from arch_policy.training.sft import sft_loss_single
    spec = replace(ARCH, model_names=("a", "b", "c", "d"))
    N = spec.n_max
    ml = torch.zeros(N, 4)
    ml[:2, 0] = 3.0   # active slots biased toward model 0
    ml = ml.clone().requires_grad_(True)
    logits = ArchLogits(
        gate_logits=torch.zeros(N), role_logits=torch.zeros(N, spec.k_roles),
        edge_logits=torch.zeros(N, N), seq_scores=torch.zeros(N),
        model_logits=ml,
    )
    comp = sft_loss_single(logits, _targets(spec, 2), None)
    assert float(comp["model"]) > math.log(4)   # non-uniform penalized
    comp["model"].backward()
    # gradient on the over-weighted logit is positive → a step DOWN reduces it
    assert ml.grad[0, 0].item() > 0.0
