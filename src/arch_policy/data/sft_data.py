"""SFT dataset: pair each task with a randomly chosen `ArchTargets`.

The user explicitly asked for *no* task→architecture bias: each task is
paired with a randomly-sampled architecture from the library. This makes
the head learn to "live in the manifold of reasonable architectures"
without locking any particular task to a particular template.

Two sampling modes (controlled by `stratify_by_family`):

  - `stratify_by_family=True` (DEFAULT, recommended):
        First sample a tier (canonical / imperfect / random) using the
        configured `tier_ratio`, then:
          * canonical: pick a family uniformly, then variant uniformly
          * imperfect / random: pick uniformly within tier
        This gives each canonical FAMILY equal weight (regardless of variant
        count), AND maintains the canonical/imperfect/random tier ratio.

        Default `tier_ratio = (0.73, 0.16, 0.11)` matches the library's
        natural composition (68 canonical / 15 imperfect / 10 random = 93).
        If you want to deliberately amplify imperfect (e.g. to thicken the
        manifold for GRPO), pass a custom tier_ratio explicitly.

  - `stratify_by_family=False` (legacy, NOT recommended):
        uniform over library entries. Biased — high-variant families like
        `fam_mad_debate` (4 entries) get sampled 4× more often than 1-
        variant families like `fam_psv` (1 entry). Kept for ablation /
        backward compatibility only.

Note: each `ArchTargets` has a variable-length `seq` (length = #active),
so we can't stack into a single tensor. The collate function returns a
*list* of `ArchTargets` (one per batch row); SFT loss iterates over them.

Usage::

    from arch_policy.architecture import full_library, encode_library
    library = full_library()
    targets = encode_library(library)        # list[ArchTargets]
    ds = SFTArchDataset(tasks, library, targets, tokenizer,
                        max_len=512, seed=0, stratify_by_family=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=ds.collate)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..architecture.encoder import encode_library
from ..architecture.library import NamedArch, family_of
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
    """Each `__getitem__` returns one (task, randomly-paired ArchTargets)."""

    def __init__(
        self,
        tasks: Sequence[TaskSample],
        library: Sequence[NamedArch],
        targets: list[ArchTargets] | None,
        tokenizer,
        max_len: int = 512,
        seed: int = 0,
        stratify_by_family: bool = True,
        tier_ratio: tuple[float, float, float] = (0.73, 0.16, 0.11),
    ) -> None:
        if targets is None:
            targets = encode_library(library)
        if len(targets) != len(library):
            raise ValueError(
                f"targets length {len(targets)} != library size {len(library)}"
            )
        if abs(sum(tier_ratio) - 1.0) > 1e-6:
            raise ValueError(f"tier_ratio must sum to 1.0, got {tier_ratio}")
        self.tasks = list(tasks)
        self.library = list(library)
        self.targets = list(targets)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._rng = random.Random(seed)
        self._stratify_by_family = stratify_by_family
        self._tier_ratio = tier_ratio

        # Pre-build family → list[lib_idx] lookup + tier index lists.
        if self._stratify_by_family:
            self._family_to_idxs: dict[str, list[int]] = defaultdict(list)
            self._imperfect_idxs: list[int] = []
            self._random_idxs: list[int] = []
            for i, arch in enumerate(self.library):
                fam = family_of(arch)
                if fam == "_imperfect":
                    self._imperfect_idxs.append(i)
                elif fam == "_random":
                    self._random_idxs.append(i)
                else:
                    self._family_to_idxs[fam].append(i)
            self._family_keys = sorted(self._family_to_idxs.keys())

        self._pairing: list[int] = self._draw_pairing()

    def __len__(self) -> int:
        return len(self.tasks)

    def reshuffle(self) -> None:
        self._pairing = self._draw_pairing()

    def _draw_pairing(self) -> list[int]:
        if not self._stratify_by_family:
            # Uniform over entries (default, backward-compatible).
            return [self._rng.randrange(len(self.library)) for _ in self.tasks]

        # Tier-aware family-stratified sampling:
        #   1. sample a tier (canonical / imperfect / random) using tier_ratio
        #   2. canonical → uniform over families, then uniform over variants
        #      imperfect / random → uniform within tier
        canon_p, imp_p, rand_p = self._tier_ratio
        out = []
        for _ in self.tasks:
            r = self._rng.random()
            if r < canon_p:
                fam = self._rng.choice(self._family_keys)
                out.append(self._rng.choice(self._family_to_idxs[fam]))
            elif r < canon_p + imp_p and self._imperfect_idxs:
                out.append(self._rng.choice(self._imperfect_idxs))
            elif self._random_idxs:
                out.append(self._rng.choice(self._random_idxs))
            else:
                # Fallback: uniform over canonical
                fam = self._rng.choice(self._family_keys)
                out.append(self._rng.choice(self._family_to_idxs[fam]))
        return out

    def __getitem__(self, idx: int) -> SFTSample:
        task = self.tasks[idx]
        lib_idx = self._pairing[idx]
        return SFTSample(
            task_id=task.task_id,
            task=task.task,
            family=task.family,
            target=self.targets[lib_idx],
            arch_name=self.library[lib_idx].name,
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
