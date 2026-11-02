"""M3's only change: draw raised and canonical training frames 50/50 (with replacement)."""
from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class PoseBalancedSampler:
    """Index stream for a DataLoader. Epoch length = number of frames, as with uniform sampling."""

    def __init__(self, poses: Sequence[str], seed: int = 0, raised_prob: float = 0.5,
                 num_samples: int | None = None):
        if any(p not in ("canonical", "raised") for p in poses):
            raise ValueError("every training frame needs a pose label")
        self.raised = [i for i, p in enumerate(poses) if p == "raised"]
        self.canonical = [i for i, p in enumerate(poses) if p == "canonical"]
        if not self.raised or not self.canonical:
            raise ValueError("need at least one raised and one canonical frame")
        self.p = raised_prob
        self.n = num_samples or len(poses)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed * 100003 + self.epoch)
        for _ in range(self.n):
            pool = self.raised if rng.random() < self.p else self.canonical
            yield pool[rng.randrange(len(pool))]

    def __len__(self) -> int:
        return self.n
