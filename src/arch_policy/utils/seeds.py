"""Seeding utility — sets python / numpy / torch RNGs in one call."""

from __future__ import annotations

import os
import random


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


__all__ = ["seed_all"]
