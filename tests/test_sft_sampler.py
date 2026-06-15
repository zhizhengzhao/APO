"""SFT sampler 2-tier redesign tests (May 2026).

Verifies the contract:
  - default 2-tier: 85% from closed pool (canonical+imperfect, uniform
    over 97), 15% true on-demand random
  - on-demand random produces a FRESH valid arch each draw
  - per-(epoch, task) seeding gives reproducibility within an epoch but
    different randoms across epochs
  - legacy 3-tier mode still works for ablation
"""

from __future__ import annotations

from collections import Counter

import pytest

from arch_policy.architecture.library import (
    NamedArch,
    canonical_library,
    family_of,
    imperfect_library,
    random_archs,
)
from arch_policy.data.sft_data import SFTArchDataset
from arch_policy.data.tasks import TaskSample


class _StubTokenizer:
    """Minimal tokenizer stub so SFTArchDataset doesn't need transformers."""
    def __call__(self, texts, **kw):
        import torch
        ids = torch.zeros(len(texts), 4, dtype=torch.long)
        mask = torch.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


def _make_dataset(n_tasks=2000, **kwargs):
    """Helper: build a deterministic dataset for ratio / behaviour checks."""
    import random as _r
    library = (
        canonical_library()
        + imperfect_library()
        + random_archs(_r.Random(42), n=10)
    )
    tasks = [
        TaskSample(task=f"q{i}", gold_answer="x",
                   family="gsm8k", task_id=f"t{i}")
        for i in range(n_tasks)
    ]
    return SFTArchDataset(
        tasks=tasks, library=library, targets=None,
        tokenizer=_StubTokenizer(), max_len=8, seed=0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ratio tests
# ---------------------------------------------------------------------------

def test_default_pool_ratio_is_85_15():
    """Default 2-tier sampler: ~85% from SFT pool, ~15% true random."""
    ds = _make_dataset(n_tasks=2000)
    kinds = Counter(kind for kind, _ in ds._pairing)
    pool_frac = kinds["lib"] / sum(kinds.values())
    rand_frac = kinds["random"] / sum(kinds.values())
    # ~5% slack on 2000 draws (binomial p=0.85, σ ≈ 0.008 → 3σ ≈ 2.4%)
    assert 0.82 < pool_frac < 0.88, f"pool_ratio drifted: {pool_frac:.3f}"
    assert 0.12 < rand_frac < 0.18, f"random_ratio drifted: {rand_frac:.3f}"


def test_custom_pool_ratio_respected():
    """A non-default pool_ratio should be honoured."""
    ds = _make_dataset(n_tasks=2000, pool_ratio=0.70)
    kinds = Counter(kind for kind, _ in ds._pairing)
    pool_frac = kinds["lib"] / sum(kinds.values())
    assert 0.67 < pool_frac < 0.73, f"custom 0.70 drifted: {pool_frac:.3f}"


# ---------------------------------------------------------------------------
# Pool composition: canonical+imperfect, uniform over 97 entries
# ---------------------------------------------------------------------------

def test_sft_pool_has_97_entries():
    """canonical (82) + imperfect (15) = 97 entries in the pool."""
    ds = _make_dataset(n_tasks=10)
    assert len(ds._sft_pool_idxs) == 97, len(ds._sft_pool_idxs)


def test_pool_draws_never_pick_random_archs():
    """`lib` draws must come from the 97 canonical+imperfect entries
    only, never from the legacy 10 pre-generated random archs."""
    ds = _make_dataset(n_tasks=5000)
    bad = []
    for kind, idx in ds._pairing:
        if kind != "lib":
            continue
        fam = family_of(ds.library[idx])
        if fam == "_random":
            bad.append(ds.library[idx].name)
    assert not bad, (
        f"SFT pool draws leaked random-tier archs: {bad[:5]}"
    )


def test_pool_draws_cover_both_canonical_and_imperfect():
    """Over enough draws, both tiers of the pool should be hit."""
    ds = _make_dataset(n_tasks=5000)
    fams = Counter()
    for kind, idx in ds._pairing:
        if kind != "lib":
            continue
        fams[family_of(ds.library[idx])] += 1
    assert fams.get("_imperfect", 0) > 0, "imperfect never drawn from SFT pool"
    canon_total = sum(c for f, c in fams.items() if f.startswith("family_"))
    assert canon_total > 0, "canonical never drawn from SFT pool"


# ---------------------------------------------------------------------------
# True on-demand random behaviour
# ---------------------------------------------------------------------------

def test_random_draws_resolve_to_unique_archs_across_epochs():
    """The whole point of on-demand random: different epoch → different
    random arch for the same task index. Otherwise it degenerates to
    the legacy "10 fixed archs" behaviour."""
    ds = _make_dataset(n_tasks=200)
    e0_random_seeds = [s for k, s in ds._pairing if k == "random"]
    ds.reshuffle()
    e1_random_seeds = [s for k, s in ds._pairing if k == "random"]
    assert e0_random_seeds and e1_random_seeds
    overlap = set(e0_random_seeds) & set(e1_random_seeds)
    assert len(overlap) == 0, (
        f"random seeds reused across epochs: {len(overlap)}/{len(e0_random_seeds)} "
        "epochs would see the same random archs each pass"
    )


def test_random_draw_returns_valid_arch():
    """Each on-demand random draw must produce a validated NamedArch
    (we never feed an invalid arch into SFT loss)."""
    ds = _make_dataset(n_tasks=200)
    n_random_seen = 0
    for i, (kind, _) in enumerate(ds._pairing):
        if kind != "random":
            continue
        n_random_seen += 1
        sample = ds[i]
        # ArchTargets.validate raises on invalid shapes / mismatches.
        sample.target.validate()
        # arch_name marker shows it came from random tier.
        assert sample.arch_name.startswith("random_e"), sample.arch_name
        if n_random_seen >= 5:
            break
    assert n_random_seen > 0, "no random draws observed in 200 tasks"


def test_random_draw_reproducible_for_same_seed():
    """Same (dataset seed, epoch, task_idx) must produce the SAME random
    arch — bit-reproducible runs are critical for debugging."""
    ds1 = _make_dataset(n_tasks=200)
    ds2 = _make_dataset(n_tasks=200)
    # Find a random-tier draw at the same index in both.
    rand_idx = next(
        i for i, (k, _) in enumerate(ds1._pairing) if k == "random"
    )
    # Both datasets used identical seed=0, so the random tier's
    # per-draw seed for `rand_idx` is also identical → same arch.
    s1 = ds1[rand_idx]
    s2 = ds2[rand_idx]
    assert s1.arch_name == s2.arch_name
    import torch
    assert torch.equal(s1.target.gates, s2.target.gates)
    assert torch.equal(s1.target.roles, s2.target.roles)
    assert torch.equal(s1.target.edges, s2.target.edges)


def test_no_random_on_demand_falls_back_to_legacy_10():
    """`random_on_demand=False` → use the old 10 pre-generated random
    archs as the random tier (for ablation comparison)."""
    ds = _make_dataset(n_tasks=2000, random_on_demand=False)
    # All draws should be 'lib' type now (no 'random' kind).
    kinds = Counter(kind for kind, _ in ds._pairing)
    assert "random" not in kinds, (
        f"on_demand=False but draws still tagged random: {kinds}"
    )
    # The lib draws should include some indices from the legacy random tier.
    rand_lib_hits = sum(
        1 for k, i in ds._pairing
        if k == "lib" and i in set(ds._old_random_idxs)
    )
    assert rand_lib_hits > 0, "no legacy random archs drawn"


# ---------------------------------------------------------------------------
# Legacy 3-tier mode (ablation path)
# ---------------------------------------------------------------------------

def test_legacy_3tier_ratio_still_works():
    """The old (0.75, 0.15, 0.10) family-stratified sampler must remain
    available for ablation comparisons."""
    ds = _make_dataset(
        n_tasks=2000,
        legacy_tier_ratio=(0.75, 0.15, 0.10),
    )
    # In legacy mode all draws are 'lib' type (no on-demand random).
    kinds = Counter(kind for kind, _ in ds._pairing)
    assert kinds.get("random", 0) == 0, kinds

    # Approx tier breakdown
    counts = Counter()
    for k, i in ds._pairing:
        fam = family_of(ds.library[i])
        if fam == "_imperfect":
            counts["imp"] += 1
        elif fam == "_random":
            counts["rand"] += 1
        else:
            counts["canon"] += 1
    total = sum(counts.values())
    # 5% slack
    assert 0.70 < counts["canon"] / total < 0.80, counts
    assert 0.12 < counts["imp"] / total < 0.18, counts
    assert 0.07 < counts["rand"] / total < 0.13, counts


def test_legacy_tier_ratio_validates_sum_to_one():
    """Bad legacy_tier_ratio must raise — silent norm is a foot-gun."""
    with pytest.raises(ValueError, match="sum to 1"):
        _make_dataset(n_tasks=10, legacy_tier_ratio=(0.5, 0.3, 0.3))
