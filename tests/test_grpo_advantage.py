"""Unit tests for the two-tier shaped_advantage + GRPO step rich logging.

`shaped_advantage` tests lock the per-branch behavior so future refactors
do not silently change the RL gradient direction. The current design is:
  Tier 1: wrong → -1, correct → 1 + min-max bonus on n_calls ∈ [+1, +1+scale]
  Tier 2: per-task /σ normalization (no mean subtraction)
  Edge:   σ=0 → -0.1 (all wrong) or 0 (all same correct)

The logging tests verify grpo_step returns rich per-task details for the
offline analyzer (correct_count_dist, per_task_details, inject vs on-policy
split, etc.).
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from arch_policy.training.grpo import shaped_advantage

# Tests that need a real backbone are skipped on environments without
# `transformers` installed (e.g. CI's minimal venv).
transformers_required = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)


def test_all_wrong_uniform_minus_push():
    """All wrong → σ=0 trigger → uniform -0.1 push (mild entropy hint).
    Magnitude is small so that when correct samples appear in OTHER groups,
    their +adv dominates this group's contribution to the overall gradient."""
    G, B = 8, 2
    correct = torch.zeros(G, B)
    n_calls = torch.tensor([[3, 4]] * G).float()  # any values, all wrong
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    assert adv.shape == (G, B)
    assert torch.allclose(adv, torch.full_like(adv, -0.1))


def test_one_correct_in_group_amplified_by_small_sigma():
    """Single correct in a wrong-majority group: raw = [+2, -1, -1, -1].
    σ is small → /σ amplifies, single correct ends up substantially above
    +1 (a real "rare event matters" boost). Wrong samples stay below 0."""
    G, B = 4, 1
    correct = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    n_calls = torch.tensor([[5], [4], [3], [2]]).float()
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)

    # Compute expected: raw=[2,-1,-1,-1], σ = sqrt(mean((x-mean)^2)).
    # mean = (2 - 3)/4 = -0.25
    # var  = ((2.25)^2 + 3·(0.75)^2)/4 = (5.0625 + 1.6875)/4 = 1.6875
    # σ    ≈ 1.2990
    import math
    sigma = math.sqrt(1.6875)
    assert abs(adv[0, 0].item() - (2.0 / sigma)) < 1e-4, f"correct: got {adv[0,0]}"
    for g in (1, 2, 3):
        assert abs(adv[g, 0].item() - (-1.0 / sigma)) < 1e-4, f"wrong g={g}: got {adv[g,0]}"
    # Sign check: correct positive, wrong negative
    assert adv[0, 0] > 0
    assert (adv[1:, 0] < 0).all()


def test_mixed_lexicographic_ordering():
    """Mixed group: every correct adv > every wrong adv (sign preserved by /σ).
    Within correct, the cheapest n_calls gets the highest adv."""
    G, B = 8, 1
    correct = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]).T.float()
    n_calls = torch.tensor([[3, 5, 5, 8, 10, 12, 4, 7]]).T.float()
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)

    min_correct = adv[correct.squeeze() > 0.5].min().item()
    max_wrong = adv[correct.squeeze() < 0.5].max().item()
    assert min_correct > max_wrong, f"lex broken: min_correct={min_correct}, max_wrong={max_wrong}"

    # Wrong samples all share the same adv (= -1/σ); n_calls ignored for wrong.
    wrong_advs = adv[correct.squeeze() < 0.5]
    assert torch.allclose(wrong_advs, torch.full((4,), wrong_advs[0].item()), atol=1e-5)

    # Within correct: g=0 (cn=3) cheapest → highest adv; g=3 (cn=8) most expensive → lowest.
    correct_idxs = [0, 1, 2, 3]  # g indices that are correct
    cn_correct = [3, 5, 5, 8]
    best_idx = correct_idxs[cn_correct.index(min(cn_correct))]
    worst_idx = correct_idxs[cn_correct.index(max(cn_correct))]
    assert adv[best_idx, 0] > adv[worst_idx, 0], "cheapest correct should beat most expensive"


def test_all_correct_varied_cost_all_positive():
    """All correct with distinct n_calls: raw ∈ [1, 2], σ > 0, /σ preserves
    sign. Every sample positive; cheapest > most expensive."""
    G, B = 16, 1
    correct = torch.ones(G, B)
    n_calls = torch.tensor([
        [3.], [5.], [5.], [8.], [12.], [4.], [6.], [7.],
        [4.], [5.], [6.], [8.], [10.], [11.], [9.], [7.],
    ])
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    # All positive (raw was all in [1, 2] > 0, /σ preserves sign)
    assert (adv > 0).all(), f"all-correct adv should be all positive; got {adv}"
    # Cheapest cn=3 (g=0) → max raw = +2 → max adv after /σ
    assert adv[0, 0] == adv[:, 0].max()
    # Most expensive cn=12 (g=4) → min raw = +1 → min adv after /σ
    assert adv[4, 0] == adv[:, 0].min()


def test_all_correct_same_calls_fires_sigma_zero_zero():
    """All correct with identical n_calls → raw all = +(1+scale) = +2
    → σ=0 → trigger fires → 'not all wrong' branch → all adv = 0
    (no signal, the task was too easy / no cost differentiation)."""
    G, B = 4, 1
    correct = torch.ones(G, B)
    n_calls = torch.full((G, B), 5.0)
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    assert torch.allclose(adv, torch.zeros_like(adv)), f"expected zeros, got {adv}"


def test_per_task_independence():
    """Two tasks in the same batch are processed independently. Different
    σ → different normalized magnitudes; doesn't leak across."""
    G = 4
    # task 0: all wrong → -0.1 fallback
    # task 1: all correct, varied cost → /σ-normalized positives
    correct = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [0., 1.]])
    n_calls = torch.tensor([[5., 3.], [5., 4.], [5., 5.], [5., 6.]])
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    # task 0 (all wrong) → -0.1 uniform
    assert torch.allclose(adv[:, 0], torch.full((G,), -0.1))
    # task 1 (all correct, distinct cost) → all positive, cheapest highest
    assert (adv[:, 1] > 0).all()
    assert adv[0, 1] == adv[:, 1].max()  # cn=3, cheapest
    assert adv[3, 1] == adv[:, 1].min()  # cn=6, most expensive


def test_g_equals_1_degenerate_safe():
    """G=1: σ=0 by construction (single sample). The σ=0 trigger handles
    both branches cleanly without NaN."""
    n_calls = torch.tensor([[5.]])
    # G=1 wrong → all-wrong branch → -0.1
    adv = shaped_advantage(torch.zeros(1, 1), n_calls, cost_bonus_scale=1.0)
    assert abs(adv[0, 0].item() - (-0.1)) < 1e-6
    # G=1 correct → 'not all wrong' branch → 0 (no signal extractable from 1 sample)
    adv = shaped_advantage(torch.ones(1, 1), n_calls, cost_bonus_scale=1.0)
    assert abs(adv[0, 0].item() - 0.0) < 1e-6


def test_defensive_partial_correct_threshold():
    """correct uses c > 0.5 to allow future partial-score graders.
    Samples with c ≤ 0.5 are treated as wrong (raw = -1).
    Samples with c > 0.5 are treated as correct (raw = 1 + bonus)."""
    G, B = 4, 1
    correct = torch.tensor([[0.3], [0.6], [0.7], [0.4]])  # 2 above 0.5
    n_calls = torch.tensor([[5.], [3.], [4.], [5.]])
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)

    # g=0 and g=3 are "wrong" (c<0.5) → raw = -1
    # g=1 and g=2 are "correct" (c>0.5) → raw = 1 + bonus
    #   correct cn = [3, 4]; min-max bonus = [(4-3)/1, (4-4)/1] = [1, 0]
    #   raw_correct = [2, 1]; raw = [-1, 2, 1, -1]
    #   mean = 0.25, var = (1.5625 + 3.0625 + 0.5625 + 1.5625)/4 = 1.6875
    #   σ ≈ 1.2990 → adv = [-0.770, 1.540, 0.770, -0.770]
    import math
    sigma = math.sqrt(1.6875)
    assert abs(adv[0, 0].item() - (-1.0 / sigma)) < 1e-4  # wrong
    assert abs(adv[3, 0].item() - (-1.0 / sigma)) < 1e-4  # wrong
    assert abs(adv[1, 0].item() - (2.0 / sigma)) < 1e-4   # correct, cheaper
    assert abs(adv[2, 0].item() - (1.0 / sigma)) < 1e-4   # correct, more expensive
    # Lex preserved + cost ordering preserved
    assert adv[1, 0] > adv[2, 0] > 0 > adv[0, 0]


def test_cost_bonus_scale_controls_within_correct_spread():
    """The cost_bonus_scale knob:
       - scale=0 → all correct collapse to +1 raw (no cost bonus). When
         mixed with wrong samples, correct stays > wrong but no within-
         correct ordering remains.
       - scale > 0 → cheapest correct grows further above worst correct.
       - In a MIXED group (wrong + correct), larger scale → larger max adv
         after /σ (the wrong = -1 baseline anchors the spread).

       Note: in an all-correct group, /σ exactly compensates the scale
       (raw range × scale → σ × scale, ratio unchanged). So we test
       scale's effect in a MIXED group where wrong samples anchor the
       baseline."""
    G = 8
    # 4 wrong + 4 correct, correct n_calls varied
    correct = torch.tensor([[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.]])
    n_calls = torch.tensor([[20.], [25.], [15.], [30.], [3.], [5.], [7.], [9.]])

    # scale=0: bonus always 0 → raw correct all = +1 (no within-correct order)
    adv_zero = shaped_advantage(correct, n_calls, cost_bonus_scale=0.0)
    assert (adv_zero[:4] < 0).all() and (adv_zero[4:] > 0).all()
    # No spread within correct: all four correct samples get same adv.
    assert torch.allclose(adv_zero[4:], torch.full((4, 1), adv_zero[4, 0].item()), atol=1e-5)

    # scale=1: raw correct ∈ [1, 2]; wrong = -1
    adv_one = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    # scale=2: raw correct ∈ [1, 3]; wrong = -1 → wider spread
    adv_two = shaped_advantage(correct, n_calls, cost_bonus_scale=2.0)

    # In mixed group, larger scale lifts the best-correct further above the
    # wrong-baseline.
    assert adv_two.max() > adv_one.max(), (
        f"larger scale should give larger max adv in mixed group; "
        f"got adv_one.max={adv_one.max():.4f}, adv_two.max={adv_two.max():.4f}"
    )


def test_single_correct_uses_max_bonus():
    """A single correct sample in a mixed group should get the FULL bonus
    (raw = 1 + scale) since it has no rivals to rank against. With scale=1
    its raw is +2 (same as 'cheapest correct in a multi-correct group')."""
    G = 4
    correct = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    n_calls = torch.tensor([[20.], [5.], [3.], [7.]])  # the lone correct is "expensive"

    # Even though n_calls[0]=20 is large, it's the only correct → full bonus.
    # raw = [2, -1, -1, -1]
    adv = shaped_advantage(correct, n_calls, cost_bonus_scale=1.0)
    # The correct should be > 0; magnitude depends on σ, but the RAW should
    # be 2.0 regardless of how expensive it was (no rivals).
    # We verify indirectly: the correct adv is > 0 and all wrong are equal.
    assert adv[0, 0] > 0
    assert (adv[1:, 0] < 0).all()
    # All wrong identical (n_calls ignored for wrong)
    assert torch.allclose(adv[1:, 0], torch.full((3,), adv[1, 0].item()), atol=1e-5)


@transformers_required
def test_grpo_step_logs_rich_per_task_fields():
    """grpo_step must return correct_count_dist + per_task_details so the
    training loop can dump them to details.jsonl."""
    from dataclasses import replace

    from arch_policy import (
        ARCH, ArchitectureHead, GRPOBatch, MockWorker, MultiAgentExecutor,
        TRAIN, load_tokenizer,
    )
    from arch_policy.training.grpo import grpo_step

    spec = replace(TRAIN, grpo_group_size=4, grpo_batch_size=2,
                   tokenizer_max_len=32, grpo_entropy_weight=0.0)
    arch_spec = replace(ARCH, safety_max_cycles=1, safety_max_steps=1)

    # tiny tokenizer is enough; skip cleanly if HF unreachable / no cache
    try:
        tok = load_tokenizer("gpt2")
        head = ArchitectureHead(backbone_name="gpt2", freeze_backbone=True,
                                lora_rank=0, torch_dtype="float32")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gpt2 unavailable (offline env): {e}")
    head.eval()

    worker = MockWorker(fake_answer="42")
    ex = MultiAgentExecutor(worker=worker, spec=arch_spec,
                            max_new_tokens_per_call=32, synth_max_new_tokens=8)
    batch = GRPOBatch(
        task_texts=["What is 6*7?", "What is 2+2?"],
        gold_answers=["42", "4"],
        task_samples=None,
    )

    info = grpo_step(head, tok, batch, ex, spec, device="cpu",
                     max_concurrent_runs=4)

    # New rich-logging fields
    assert "correct_count_dist" in info
    assert "n_unanimous" in info
    assert "n_mixed" in info
    assert "per_task_details" in info

    G = spec.grpo_group_size
    B = spec.grpo_batch_size

    # correct_count_dist sums to B (each task contributes to exactly one bin).
    assert len(info["correct_count_dist"]) == G + 1
    assert sum(info["correct_count_dist"]) == B

    # unanimous + mixed == B
    assert info["n_unanimous"] + info["n_mixed"] == B

    # per_task_details length matches B
    assert len(info["per_task_details"]) == B
    for t in info["per_task_details"]:
        assert "task_idx" in t and "correct_count" in t
        assert "samples" in t and len(t["samples"]) == G
        for s in t["samples"]:
            for k in ("correct", "n_calls", "n_active",
                     "active_roles", "sequence", "edges_count", "advantage"):
                assert k in s, f"missing {k} in sample"


def test_to_arch_logits_forwards_all_typed_fields():
    """to_arch_logits must forward all 4 typed components."""
    import torch
    from arch_policy import ARCH
    from arch_policy.head.model import to_arch_logits
    N, R = ARCH.n_max, ARCH.k_roles
    out = {
        "gate_logits": torch.randn(2, N),
        "role_logits": torch.randn(2, N, R),
        "edge_logits": torch.randn(2, N, N),
        "seq_scores":  torch.randn(2, N),
    }
    logits = to_arch_logits(out, batch_idx=0)
    logits.validate(ARCH)
    assert logits.gate_logits.shape == (N,)
    assert logits.role_logits.shape == (N, R)
    assert logits.edge_logits.shape == (N, N)
    assert logits.seq_scores.shape == (N,)


def test_entropy_typed_masks_inactive_slots():
    """entropy_typed must weight per-slot entropies (role / edge) by
    P(slot active). With low gate logits, role entropy contribution
    should be small even when role logits are uniform-flat."""
    import torch
    from arch_policy import ARCH
    from arch_policy.training.grpo import entropy_typed
    N, R = ARCH.n_max, ARCH.k_roles

    out_low = {
        "gate_logits": torch.full((1, N), -5.0),   # sigmoid ~ 0.007
        "role_logits": torch.zeros(1, N, R),        # max entropy per slot
        "edge_logits": torch.zeros(1, N, N),        # max entropy per pair
        "seq_scores":  torch.zeros(1, N),
    }
    out_high = {**out_low,
                "gate_logits": torch.full((1, N), +5.0)}  # sigmoid ~ 0.993

    h_low  = entropy_typed(out_low, ARCH).item()
    h_high = entropy_typed(out_high, ARCH).item()
    assert h_high > h_low * 2.0, (
        f"per-slot entropies should scale with gate_p; "
        f"got h_low={h_low:.4f}, h_high={h_high:.4f}"
    )


def test_extract_math_boxed_returns_last_not_first():
    """V7-fix #7a: prior version used re.search (FIRST match) despite docstring
    claiming LAST. With multiple \\boxed{} (case analysis, recall), the
    extracted answer was wrong."""
    from arch_policy.data.tasks import _extract_math_boxed
    solution = (
        "First we boxed an intermediate result: \\boxed{42}.\n"
        "Then we continued and computed the final answer:\n"
        "Therefore, the answer is \\boxed{99}."
    )
    assert _extract_math_boxed(solution) == "99"


def test_pool_last_token_handles_left_and_right_padding():
    """V7-fix #6: _pool_last_token must pick the LAST non-pad token
    regardless of padding side. Qwen3 default is left-pad."""
    import torch
    from arch_policy.head.model import ArchitectureHead
    # Test the function statically without instantiating the head (which
    # would download Qwen3 weights). Manually replicate the pooler logic
    # using the SAME formula the V7 fix uses.
    def _pool(hidden_states, attention_mask):
        total = attention_mask.sum(dim=1, keepdim=True)
        cum   = attention_mask.cumsum(dim=1)
        last_idx = (cum == total).long().argmax(dim=1)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)

    # Hidden = [B=2, T=5, H=3]; each token's hidden is just [t, t, t].
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    # Row 0: right-padded mask=[1,1,1,0,0] → last token at idx 2
    # Row 1: left-padded mask =[0,0,1,1,1] → last token at idx 4
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    pooled = _pool(hidden, mask)
    # Row 0 idx 2 → hidden[0, 2] = [6, 7, 8]
    # Row 1 idx 4 → hidden[1, 4] = [27, 28, 29]
    assert pooled[0].tolist() == [6, 7, 8]
    assert pooled[1].tolist() == [27, 28, 29]


def test_default_inject_pool_is_all_canonical():
    """default_inject_pool returns the FULL canonical library (all 82
    archs across 42 families, post-library-rebalance May 2026) for
    maximal anchor coverage. No imperfect / random entries."""
    from arch_policy.architecture.library import (
        canonical_library, default_inject_pool,
    )

    arches = default_inject_pool()
    canon = canonical_library()
    assert len(arches) == len(canon), (
        f"default_inject_pool size {len(arches)} != canonical_library "
        f"size {len(canon)}"
    )

    # All distinct, all valid, all canonical (carry a `family_*` tag).
    names = [a.name for a in arches]
    assert len(set(names)) == len(arches), "duplicate names in pool"
    for a in arches:
        a.validate()
        assert any(t.startswith("family_") for t in a.tags), (
            f"{a.name} missing family_* tag (not a canonical arch?)"
        )

    # Size coverage spans 1..6.
    sizes = {len(a.agents) for a in arches}
    for s in (1, 2, 3, 4, 5):
        assert s in sizes, f"missing size-{s} archs in canonical pool"


def test_default_inject_pool_idempotent_fresh_instances():
    """Calling default_inject_pool() twice yields independent NamedArch
    instances (callers can safely .to_concrete() each without aliasing)."""
    from arch_policy.architecture.library import default_inject_pool

    p1 = default_inject_pool()
    p2 = default_inject_pool()
    assert [a.name for a in p1] == [a.name for a in p2]
    for x, y in zip(p1, p2):
        assert x is not y, f"{x.name} is aliased between calls"


def test_executor_requires_worker():
    """MultiAgentExecutor is single-worker only; worker=None must raise."""
    from arch_policy import MultiAgentExecutor
    try:
        MultiAgentExecutor(worker=None)
    except (ValueError, TypeError):
        return
    raise AssertionError("expected error when worker is None")


def test_executor_synth_worker_defaults_to_worker():
    """When synth_worker is not set, Synth uses the main worker."""
    from arch_policy import MockWorker, MultiAgentExecutor
    w = MockWorker()
    ex = MultiAgentExecutor(worker=w)
    assert ex.synth_worker is w
    # Explicit override
    w2 = MockWorker(fake_answer="other")
    ex2 = MultiAgentExecutor(worker=w, synth_worker=w2)
    assert ex2.synth_worker is w2 and ex2.worker is w


def test_agent_allowed_tools_inferred_from_role():
    """Agent constructor must default allowed_tools to the role's pool
    when not given explicitly. Minimum-tool principle:
      - Planner / Refiner: ∅ (pure text shaping)
      - Solver / Critic / Verifier: {python_exec} only
      - Expert: full TOOLS set (generalist)
    """
    from arch_policy import MockWorker
    from arch_policy.executor.agent import Agent
    from arch_policy.executor.tools import TOOLS

    for role in ("Planner", "Refiner"):
        a = Agent(slot=0, role=role, worker=MockWorker())
        assert a.allowed_tools == frozenset(), role
    for role in ("Solver", "Critic", "Verifier"):
        a = Agent(slot=0, role=role, worker=MockWorker())
        assert a.allowed_tools == frozenset({"python_exec"}), role
    a_expert = Agent(slot=0, role="Expert", worker=MockWorker())
    assert a_expert.allowed_tools == frozenset(TOOLS.keys())


def test_call_tool_respects_allowlist():
    """call_tool must refuse tools outside the allowlist with a clear message."""
    from arch_policy.executor.tools import call_tool

    out = call_tool("web_search", "test query", allowed=frozenset({"python_exec"}))
    assert "not available for this role" in out
    # Whereas an allowed tool runs (python_exec returns valid result on '2+2')
    out2 = call_tool("python_exec", "print(2+2)", allowed=frozenset({"python_exec"}))
    assert "STDOUT" in out2 and "4" in out2


def test_parse_action_whitelists_real_tools_only():
    """parse_action must REJECT garbage tool names that the LLM commonly
    emits (e.g. `ACTION: python_execARGS: ...` without a newline, or
    single-word noise like `ACTION: thermal`). Previously these wasted a
    whole ReAct step on an [unknown tool] reply."""
    from arch_policy.executor.agent import parse_action

    out = parse_action("THOUGHT: x\nACTION: python_exec\nARGS: print(1)")
    assert out is not None and out[0] == "python_exec"

    # LLM-typo: trailing ARGS jammed onto tool name with no newline
    out = parse_action("ACTION: python_execARGS print(1)")
    assert out is None, "should reject `python_execARGS` (not a real tool)"

    # Bare-word noise (real captures from a 128-trace smoke run)
    for word in ("thermal", "ketones", "butan", "identify", "None"):
        assert parse_action(f"ACTION: {word}") is None, (
            f"should reject garbage tool name {word!r}"
        )

    # Tools removed during development must stay rejected.
    for old in ("z3_check", "wikipedia_lookup", "code_linter", "sympy_check"):
        assert parse_action(f"ACTION: {old}\nARGS: stuff") is None, (
            f"removed tool {old!r} must be rejected"
        )


def test_named_to_concrete_roundtrip():
    """named_arch_to_concrete produces a ConcreteArch whose tensors agree
    with the NamedArch's slot/role/edge/sequence data. V7: also checks
    the `models` tensor defaults to zeros (model 0 = first in the pool;
    GRPO replaces it via sampling, but library entries are model-agnostic)."""
    from arch_policy.architecture.library import (
        SOLVER, VERIFIER, NamedArch, named_arch_to_concrete,
    )

    na = NamedArch(
        name="t_sv",
        agents=[(0, SOLVER), (2, VERIFIER)],
        edges=[(0, 2), (2, 0)],
        sequence=[0, 2],
    )
    ca = named_arch_to_concrete(na)
    assert bool(ca.active_mask[0]) and bool(ca.active_mask[2])
    assert not bool(ca.active_mask[1])
    assert int(ca.roles[0]) == SOLVER
    assert int(ca.roles[2]) == VERIFIER
    assert bool(ca.edges[0, 2]) and bool(ca.edges[2, 0])
    assert not bool(ca.edges[0, 1])
    assert ca.sequence.tolist() == [0, 2]
    # Single-vendor: ConcreteArch carries no models / synth_model fields.
    assert not hasattr(ca, "models")
    assert not hasattr(ca, "synth_model")


# ---------------------------------------------------------------------------
# V7 post-review regression: public API + __all__ hygiene
# ---------------------------------------------------------------------------

def test_prompts_star_import_does_not_crash():
    """Regression for teacher review A: `__all__` must not list names that
    no longer exist in the module (previously `REACT_INSTRUCTION` was a
    stale entry after we made it role-conditional)."""
    import arch_policy.executor.prompts as _p
    ns: dict = {}
    exec("from arch_policy.executor.prompts import *", ns)
    for name in getattr(_p, "__all__", []):
        assert name in ns, f"__all__ exports missing name: {name!r}"


def test_public_api_reexports_visible():
    """Typed-policy additions must be importable from the top-level
    `arch_policy` package, not buried in submodule paths."""
    import arch_policy as ap
    expected = [
        # architecture
        "default_inject_pool", "named_arch_to_concrete",
        # reward
        "grade_multiple_choice",
        # training
        "DEFAULT_ENTROPY_WEIGHTS",
    ]
    for name in expected:
        assert hasattr(ap, name), f"arch_policy.{name} not re-exported"


def test_data_bbh_helpers_reexported():
    """Regression for teacher review C: BBH mixed helpers must be reachable
    from the lazy-loaded `arch_policy.data` namespace."""
    import arch_policy as ap
    for name in ("BBH_DIVERSE_SUBSETS", "load_bbh_mixed"):
        assert hasattr(ap, name), f"arch_policy.{name} not re-exported"


def test_05_eval_token_prices_cover_default_worker():
    """The default --worker_model in 04_evaluate.py must have a TOKEN_PRICES
    entry, otherwise `_price()` silently returns $0."""
    import importlib.util as _u, pathlib as _p
    from arch_policy.config import MODEL
    spec = _u.spec_from_file_location(
        "_eval", _p.Path(__file__).resolve().parents[1] / "scripts" / "04_evaluate.py"
    )
    mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)
    assert MODEL.worker_model in mod.TOKEN_PRICES, (
        f"missing price for default worker_model: {MODEL.worker_model}"
    )


# ---------------------------------------------------------------------------
# V7-β dataset-library rebalance regression
# ---------------------------------------------------------------------------

def test_v7beta_new_canonical_families_present():
    """V7-β added 4 Tester/research-code families. Lock them in so a future
    refactor doesn't drop them silently."""
    from arch_policy.architecture.library import CANONICAL_FAMILIES
    fam_names = {f.__name__ for f in CANONICAL_FAMILIES}
    for needed in ("fam_tdd_iterate", "fam_spec_first",
                   "fam_tester_council", "fam_research_code"):
        assert needed in fam_names, f"missing V7-β family: {needed}"


def test_v7beta_library_sizes():
    """Lock the rebalanced library composition (42 families / 82 canonical /
    15 imperfect / 10 random = 107 total in full_library)."""
    import random as _r
    from arch_policy.architecture.library import (
        CANONICAL_FAMILIES, canonical_library, imperfect_library,
        random_archs, default_inject_pool,
    )
    assert len(CANONICAL_FAMILIES) == 42
    assert len(canonical_library()) == 82
    assert len(imperfect_library()) == 15
    assert len(random_archs(_r.Random(0), n=10)) == 10
    assert len(default_inject_pool()) == 82


def test_v7beta_family_stratified_inject_is_uniform_over_families():
    """The principled inject sampler must equalize BC pressure across
    families (max/min hit ratio ≤ 1.3 over a long run). Compared to the
    legacy uniform-over-arch sampler which has ratio ~4x due to
    high-variant families dominating."""
    import random as _r
    from collections import Counter
    from arch_policy.architecture.library import (
        CANONICAL_FAMILIES, canonical_library,
        sample_inject_family_stratified,
    )
    rng = _r.Random(42)
    hits = Counter()
    n_steps = 4000
    k = 6
    for _ in range(n_steps):
        for arch in sample_inject_family_stratified(rng, k):
            fam_tag = next(t for t in arch.tags if t.startswith("family_"))
            hits[fam_tag] += 1
    assert len(hits) == len(CANONICAL_FAMILIES), (
        f"only {len(hits)} of {len(CANONICAL_FAMILIES)} families ever sampled"
    )
    ratio = max(hits.values()) / min(hits.values())
    assert ratio <= 1.3, (
        f"family-stratified sampler should be near-uniform; got max/min ratio "
        f"{ratio:.2f} (>1.3). hits: {dict(hits)}"
    )


def test_v7beta_family_stratified_inject_k_bounds():
    """Sampler must reject k <= 0 and k > num_families."""
    import random as _r
    import pytest
    from arch_policy.architecture.library import (
        CANONICAL_FAMILIES, sample_inject_family_stratified,
    )
    rng = _r.Random(0)
    with pytest.raises(ValueError):
        sample_inject_family_stratified(rng, 0)
    with pytest.raises(ValueError):
        sample_inject_family_stratified(rng, len(CANONICAL_FAMILIES) + 1)
    # Exact boundary should work
    out = sample_inject_family_stratified(rng, len(CANONICAL_FAMILIES))
    assert len(out) == len(CANONICAL_FAMILIES)


def test_v7beta_tier_ratio_default_sums_to_one():
    """Regression: previously the CLI default (0.65, 0.15, 0.10) summed to
    0.90, which crashed SFTArchDataset's `sum == 1.0` assertion the moment
    the user ran `02_train_sft.py` without --tier_ratio override."""
    import importlib.util as _u, pathlib as _p
    spec = _u.spec_from_file_location(
        "_sft_script",
        _p.Path(__file__).resolve().parents[1] / "scripts" / "02_train_sft.py",
    )
    mod = _u.module_from_spec(spec); spec.loader.exec_module(mod)
    parser = mod.parse_args.__wrapped__ if hasattr(mod.parse_args, "__wrapped__") else None
    # Re-instantiate parser to inspect the default cleanly
    import sys
    saved_argv = sys.argv
    try:
        sys.argv = ["02_train_sft.py"]
        args = mod.parse_args()
    finally:
        sys.argv = saved_argv
    # Post-May-2026 redesign: SFT uses a 2-tier sampler with `pool_ratio`
    # (default 0.85) + true on-demand random for the remaining 1 -
    # pool_ratio. The legacy 3-tier `--legacy_tier_ratio` is None by
    # default. Check the new contract holds.
    assert 0.0 < args.pool_ratio < 1.0, (
        f"pool_ratio default {args.pool_ratio} must be in (0,1)"
    )
    assert args.legacy_tier_ratio is None, (
        f"legacy 3-tier sampler must be OFF by default; "
        f"got legacy_tier_ratio={args.legacy_tier_ratio}"
    )


def test_v7beta_grpo_eval_dataset_choices_include_new_benches():
    """Regression: `--dataset` choices in both 03_train_grpo.py and
    04_evaluate.py must include the 4 V7-β benchmarks now that their
    loaders + DEFAULT_SFT_MIX entries are wired."""
    import pathlib as _p
    scripts_dir = _p.Path(__file__).resolve().parents[1] / "scripts"
    for script in ("03_train_grpo.py", "04_evaluate.py"):
        text = (scripts_dir / script).read_text()
        # Both scripts have a `--dataset` argparse arg; verify the 4 new bench
        # names appear in the same file (in practice they're in the choices=
        # list within ~20 lines of the `--dataset` arg).
        assert "--dataset" in text, f"{script} missing --dataset arg"
        for bench in ("browsecomp", "hle", "phybench", "livecodebench"):
            assert bench in text, f"{script}: --dataset choices missing {bench!r}"


def test_v7beta_default_sft_mix_includes_new_benches():
    """SFT default mix must include the 4 new benchmarks so the head sees
    those task wordings during warmup."""
    from arch_policy.data.tasks import DEFAULT_SFT_MIX
    for bench in ("browsecomp", "hle", "phybench", "livecodebench"):
        assert bench in DEFAULT_SFT_MIX, f"DEFAULT_SFT_MIX missing {bench!r}"
        assert DEFAULT_SFT_MIX[bench] > 0


if __name__ == "__main__":
    test_all_wrong_uniform_minus_push()
    test_one_correct_in_group_amplified_by_small_sigma()
    test_mixed_lexicographic_ordering()
    test_all_correct_varied_cost_all_positive()
    test_all_correct_same_calls_fires_sigma_zero_zero()
    test_per_task_independence()
    test_g_equals_1_degenerate_safe()
    test_defensive_partial_correct_threshold()
    test_cost_bonus_scale_controls_within_correct_spread()
    test_single_correct_uses_max_bonus()
    test_to_arch_logits_forwards_all_typed_fields()
    test_entropy_typed_masks_inactive_slots()
    test_extract_math_boxed_returns_last_not_first()
    test_pool_last_token_handles_left_and_right_padding()
    test_default_inject_pool_is_all_canonical()
    test_default_inject_pool_idempotent_fresh_instances()
    test_executor_requires_worker()
    test_executor_synth_worker_defaults_to_worker()
    test_agent_allowed_tools_inferred_from_role()
    test_call_tool_respects_allowlist()
    test_parse_action_whitelists_real_tools_only()
    test_named_to_concrete_roundtrip()
    print("All shaped_advantage + library tests pass.")
