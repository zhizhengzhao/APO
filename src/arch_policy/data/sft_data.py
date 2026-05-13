"""SFT dataset (v3): pair each task with a randomly chosen `ArchTargets`.

The user explicitly asked for *no* task→architecture bias: each task is
paired with a uniformly-sampled architecture from the library. This makes
the head learn to "live in the manifold of reasonable architectures"
without locking any particular task to a particular template.

Note: in v3 each `ArchTargets` has a variable-length `seq` (length = #active),
so we can't stack into a single tensor. The collate function returns a
*list* of `ArchTargets` (one per batch row); SFT loss iterates over them.

Usage::

    from arch_policy.architecture import full_library, encode_library
    library = full_library()
    targets = encode_library(library)        # list[ArchTargets]
    ds = SFTArchDataset(tasks, library, targets, tokenizer, max_len=512, seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=ds.collate)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import Dataset

from ..architecture.encoder import encode_library
from ..architecture.library import NamedArch
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
        self._rng = random.Random(seed)
        self._pairing: list[int] = self._draw_pairing()

    def __len__(self) -> int:
        return len(self.tasks)

    def reshuffle(self) -> None:
        self._pairing = self._draw_pairing()

    def _draw_pairing(self) -> list[int]:
        return [self._rng.randrange(len(self.library)) for _ in self.tasks]

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
