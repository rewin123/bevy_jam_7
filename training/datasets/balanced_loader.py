"""Balanced multi-dataset loader for mixed training.

Provides infinite, balanced sampling across multiple datasets regardless of size.
"""

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from .video_flow import InfiniteSamplerWrapper


class InfiniteLoader:
    """Wraps a Dataset in an infinite DataLoader iterator."""

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int):
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=InfiniteSamplerWrapper(dataset),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self._iter: Iterator = iter(self._loader)

    def next(self) -> dict | torch.Tensor:
        return next(self._iter)


class BalancedMultiLoader:
    """Draws batches from multiple datasets with balanced weights.

    Each dataset gets its own InfiniteLoader. On each call to next(),
    one dataset is chosen randomly according to the provided weights.

    Args:
        datasets: list of (name, dataset) tuples
        batch_size: batch size for all loaders
        num_workers: total workers (divided among loaders)
        weights: optional sampling weights (default: uniform)
    """

    def __init__(
        self,
        datasets: list[tuple[str, Dataset]],
        batch_size: int,
        num_workers: int,
        weights: list[float] | None = None,
    ):
        self.names = [name for name, _ in datasets]

        if weights is None:
            weights = [1.0] * len(datasets)
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights]

        workers_each = max(1, num_workers // len(datasets))
        self._loaders = [
            InfiniteLoader(ds, batch_size, workers_each)
            for _, ds in datasets
        ]

    def next(self) -> tuple[str, dict | torch.Tensor]:
        """Return (dataset_name, batch)."""
        idx = random.choices(range(len(self._loaders)), weights=self.weights, k=1)[0]
        return self.names[idx], self._loaders[idx].next()
