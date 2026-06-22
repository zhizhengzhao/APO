"""SFT dataset: pair each task with a randomly chosen `ArchTargets`.

By design, no task→architecture grounding: the head learns to live in
the manifold of reasonable architectures without locking task→template.

Default sampling:

  - 85% draws from the *closed* SFT pool = canonical (82) + imperfect (15)
    = 97 hand-designed archs. Internal distribution: **uniform over the
    97 entries** (no family stratification). Trade-off: high-variant
    canonical families (e.g. fam_mad_debate, 4 variants) get 4× the BC
    pressure of 1-variant families. Acceptable because high-variant
    families are typically the most-studied patterns and merit more
    exposure.

  - 15% draws are TRUE on-demand random: each draw generates a fresh
    valid arch via `random_archs(rng, n=1)` (subject to ≥1 answerer
    + valid constraints). Reproducible via per-draw seed derived from
    a session-level base seed, so different epochs see different
    randoms (regularization) but the same epoch's draws are bit-
    reproducible. Replaces the old "10 fixed random archs reused
    every epoch" behaviour which was effectively just 10 more anchor
    targets.

`pool_ratio` (default 0.85) controls the 2-tier split. Legacy 3-tier
family-stratified sampling is available via `legacy_tier_ratio` for
ablations.

`ArchTargets.seq` has variable length so we can't stack into a single
tensor; the collate function returns a list of `ArchTargets` and the SFT
loss iterates over them.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from torch.utils.data import Dataset

from ..architecture.encoder import encode_library, encode_named_arch
from ..architecture.library import NamedArch, family_of, random_archs
from ..architecture.spec import ArchTargets
from .tasks import TaskSample


@dataclass
class SFTSample:
    task_id: str
    task: str
    family: str
    target: ArchTargets
    arch_name: str


class SFTArchDataset(Dataset):
    """Each `__getitem__` returns one (task, randomly-paired ArchTargets).

    2-tier sampler:
      - `pool_ratio` (default 0.85) of draws → uniform over the closed
        SFT pool (canonical + imperfect, 97 entries).
      - `1 - pool_ratio` (default 0.15) of draws → true on-demand random
        arch, generated fresh per draw via `random_archs(rng, n=1)`.

    Legacy 3-tier mode (canonical 75% / imperfect 15% / random 10% with
    family-stratified canonical + 10 pre-fixed random archs) is available
    by passing `legacy_tier_ratio=(...)`. Used only for ablation.
    """

    def __init__(
        self,
        tasks: Sequence[TaskSample],
        library: Sequence[NamedArch],
        targets: list[ArchTargets] | None,
        tokenizer,
        max_len: int = 512,
        seed: int = 0,
        # ---- 2-tier default (NEW) -----------------------------------
        pool_ratio: float = 0.85,
        random_on_demand: bool = True,
        # ---- 3-tier legacy fallback (for ablation) ------------------
        stratify_by_family: bool = False,
        legacy_tier_ratio: tuple[float, float, float] | None = None,
    ) -> None:
        if targets is None:
            targets = encode_library(library)
        if len(targets) != len(library):
            raise ValueError(
                f"targets length {len(targets)} != library size {len(library)}"
            )
        self.tasks = list(tasks)
        self.library = list(library)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._seed = seed
        self._rng = random.Random(seed)
        self._epoch = 0    # advanced by reshuffle() for per-draw random seeds

        self._pool_ratio = pool_ratio
        self._random_on_demand = random_on_demand
        self._stratify_by_family = stratify_by_family
        self._legacy_tier_ratio = legacy_tier_ratio

        if legacy_tier_ratio is not None:
            if abs(sum(legacy_tier_ratio) - 1.0) > 1e-6:
                raise ValueError(
                    f"legacy_tier_ratio must sum to 1.0, got {legacy_tier_ratio}"
                )
        elif not (0.0 <= pool_ratio <= 1.0):
            raise ValueError(f"pool_ratio must be in [0,1], got {pool_ratio}")

        # Validate that the library has the canonical/imperfect entries
        # we'll be sampling from when pool_ratio > 0. Without this the
        # first __getitem__ call raises an obscure IndexError on
        # self._sft_pool_idxs being empty.
        # (Build the indexes below first; only enforce after.)

        # Indexes for the 2-tier path: every canonical OR imperfect entry
        # is "sft_pool"; every random entry is in `_old_random_idxs`
        # (used only in legacy mode or as fallback when random_on_demand
        # is False).
        self._sft_pool_idxs: list[int] = []
        self._old_random_idxs: list[int] = []
        # 3-tier legacy indexes (only built when needed)
        self._family_to_idxs: dict[str, list[int]] = defaultdict(list)
        self._imperfect_idxs: list[int] = []
        self._family_keys: list[str] = []
        for i, arch in enumerate(self.library):
            fam = family_of(arch)
            if fam == "_random":
                self._old_random_idxs.append(i)
            else:
                # canonical OR imperfect → both go in the SFT pool
                self._sft_pool_idxs.append(i)
                if fam == "_imperfect":
                    self._imperfect_idxs.append(i)
                else:
                    self._family_to_idxs[fam].append(i)
        self._family_keys = sorted(self._family_to_idxs.keys())

        # N24 guard: if pool_ratio > 0 (default 0.85) but no canonical or
        # imperfect entry exists in `library`, every pool-tier draw would
        # raise IndexError at __getitem__. Raise loudly now with a clear
        # message instead of a delayed crash mid-epoch.
        if pool_ratio > 0 and not self._sft_pool_idxs and legacy_tier_ratio is None:
            raise ValueError(
                f"pool_ratio={pool_ratio} > 0 requires at least one canonical "
                f"or imperfect arch in `library`, got 0. Either pass a "
                f"library containing canonical/imperfect entries (e.g. "
                f"full_library() or canonical_library()) or set "
                f"pool_ratio=0.0 to use only the random tier."
            )

        self._pairing: list[tuple[str, int]] = self._draw_pairing()

    def __len__(self) -> int:
        return len(self.tasks)

    def reshuffle(self) -> None:
        """Re-sample the task→arch pairing for the next epoch.

        Advances an internal `_epoch` counter; per-draw random seeds for
        the on-demand random tier are derived from
        `(seed, epoch, task_idx)` so each epoch sees a different set of
        random archs (regularization) while remaining bit-reproducible.
        """
        self._epoch += 1
        self._pairing = self._draw_pairing()

    def _draw_pairing(self) -> list[tuple[str, int]]:
        """Returns a list of (kind, payload) per task:

          ('lib',    lib_idx)        → use self.targets[lib_idx]
          ('random', random_seed)    → generate fresh arch with this seed
        """
        # Legacy 3-tier path (only when caller opts in via legacy_tier_ratio).
        if self._legacy_tier_ratio is not None:
            return self._draw_legacy_3tier()
        # 2-tier default path.
        return self._draw_2tier()

    def _draw_2tier(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for task_idx in range(len(self.tasks)):
            r = self._rng.random()
            if r < self._pool_ratio:
                # Uniform over the 97 SFT-pool entries (no family stratification).
                out.append(("lib", self._rng.choice(self._sft_pool_idxs)))
            else:
                if self._random_on_demand:
                    # Stable per-draw seed: depends on dataset seed,
                    # epoch, and task index. Two epochs see DIFFERENT
                    # randoms; same (seed, epoch, task) replays
                    # bit-identical.
                    #
                    # Stride choice: each axis must use a stride > the
                    # max possible value of every lower-significance
                    # axis to avoid collisions. With epoch * 10_000 the
                    # old formula collided once `task_idx >= 10_000`
                    # (DEFAULT_SFT_MIX has ~11.5K tasks → 13% of the
                    # epoch-N draws collided with epoch-N+1). Stride
                    # 1_000_003 (prime, > any realistic task count) on
                    # both epoch and seed is collision-free for
                    # task_idx < 1e6, epoch < 1e6.
                    rseed = (self._seed * 1_000_003 * 1_000_003
                             + self._epoch * 1_000_003
                             + task_idx)
                    out.append(("random", rseed))
                else:
                    # Fallback to legacy fixed 10 random archs when caller
                    # wants reproducibility across epochs.
                    if self._old_random_idxs:
                        out.append(("lib", self._rng.choice(self._old_random_idxs)))
                    else:
                        # No random in library and on_demand disabled →
                        # degrade to SFT-pool draw. WARN once so the log
                        # header's "X% random" stops being a lie.
                        if not getattr(self, "_warned_random_degrade", False):
                            self._warned_random_degrade = True
                            print(f"[SFTArchDataset] WARN: --no_random_on_demand "
                                  f"set but library has 0 random archs → "
                                  f"the configured random-tier fraction is "
                                  f"effectively 0%, all draws come from the "
                                  f"SFT pool.", flush=True)
                        out.append(("lib", self._rng.choice(self._sft_pool_idxs)))
        return out

    def _draw_legacy_3tier(self) -> list[tuple[str, int]]:
        """Old 3-tier family-stratified sampler (canonical / imperfect /
        random with separate ratios + canonical family-uniform). Kept for
        ablation studies."""
        canon_p, imp_p, _ = self._legacy_tier_ratio   # type: ignore[misc]
        out: list[tuple[str, int]] = []
        for _ in self.tasks:
            r = self._rng.random()
            if r < canon_p:
                fam = self._rng.choice(self._family_keys)
                out.append(("lib", self._rng.choice(self._family_to_idxs[fam])))
            elif r < canon_p + imp_p and self._imperfect_idxs:
                out.append(("lib", self._rng.choice(self._imperfect_idxs)))
            elif self._old_random_idxs:
                out.append(("lib", self._rng.choice(self._old_random_idxs)))
            else:
                fam = self._rng.choice(self._family_keys)
                out.append(("lib", self._rng.choice(self._family_to_idxs[fam])))
        return out

    def __getitem__(self, idx: int) -> SFTSample:
        task = self.tasks[idx]
        kind, payload = self._pairing[idx]
        if kind == "lib":
            target = self.targets[payload]
            arch_name = self.library[payload].name
        else:  # ("random", seed)
            # Generate ONE fresh random arch with this draw's seed.
            # `random_archs` validates internally + retries until it gets
            # a valid arch with ≥1 answerer.
            arches = random_archs(random.Random(payload), n=1)
            if not arches:
                # Pathological seed never produced a valid arch in 10
                # attempts; fall back to a known-good canonical pick so
                # the batch isn't silently smaller.
                fallback_idx = self._rng.choice(self._sft_pool_idxs)
                target = self.targets[fallback_idx]
                arch_name = f"random_fallback_{self.library[fallback_idx].name}"
            else:
                arch = arches[0]
                target = encode_named_arch(arch)
                arch_name = f"random_e{self._epoch}_i{idx}_s{payload}"
        return SFTSample(
            task_id=task.task_id,
            task=task.task,
            family=task.family,
            target=target,
            arch_name=arch_name,
        )

    def collate(self, batch: list[SFTSample]) -> dict:
        texts = [b.task for b in batch]
        targets = [b.target for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "task_ids": [b.task_id for b in batch],
            "task_texts": texts,
            "arch_names": [b.arch_name for b in batch],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "targets": targets,
        }


__all__ = ["SFTArchDataset", "SFTSample"]
