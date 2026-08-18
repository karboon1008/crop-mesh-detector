"""Batch sampler for mixing two datasets (PlantVillage + PlantDoc) at a
fixed per-batch domain ratio, with class-balanced sampling on the smaller
(secondary) side.
"""

from __future__ import annotations
import numpy as np
from torch.utils.data import Dataset, Sampler


class EvalView(Dataset):
    def __init__(self, base, indices: list[int]):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, crop_idx, disease_idx = self.base._lookup(self.indices[i])
        return self.base.eval_transform(image), crop_idx, disease_idx


class MixedDomainBatchSampler(Sampler[list[int]]):
    """Every batch is `round(batch_size * primary_fraction)` primary
    samples (plain shuffle, no replacement within an epoch) plus the
    remainder drawn from secondary — secondary is small and highly
    class-skewed (e.g. PlantDoc's ~28 classes at ~50-100 images each), so
    those picks use inverse-frequency class weights, with replacement, to
    keep rare secondary classes from disappearing under the ones with more
    images.

    Primary's own class balance is untouched here — that's handled
    upstream by the existing per-node inverse-frequency loss weights, which
    reflect that node's local (non-IID) label distribution.
    """

    def __init__(
        self,
        primary_len: int,
        secondary_len: int,
        secondary_labels: list[int],
        batch_size: int,
        primary_fraction: float,
        seed: int,
        num_batches: int | None = None,
    ):
        if not 0.0 < primary_fraction <= 1.0:
            raise ValueError("primary_fraction must be in (0, 1]")
        if secondary_len and len(secondary_labels) != secondary_len:
            raise ValueError("secondary_labels must have one entry per secondary sample")

        self.primary_len = primary_len
        self.secondary_len = secondary_len
        self.n_primary = min(primary_len, max(1, round(batch_size * primary_fraction))) if secondary_len else batch_size
        self.n_secondary = batch_size - self.n_primary
        self.num_batches = num_batches or max(1, -(-primary_len // max(1, self.n_primary)))  # ceil div
        self.rng = np.random.RandomState(seed)

        self.secondary_weights = None
        if secondary_len and self.n_secondary:
            labels = np.asarray(secondary_labels)
            counts = np.bincount(labels)
            class_weight = np.where(counts > 0, 1.0 / counts, 0.0)
            weights = class_weight[labels]
            self.secondary_weights = weights / weights.sum()

    def __iter__(self):
        primary_order = self.rng.permutation(self.primary_len) if self.primary_len else np.empty(0, dtype=int)
        pos = 0
        for _ in range(self.num_batches):
            batch: list[int] = []
            if self.primary_len:
                if pos + self.n_primary > self.primary_len:
                    primary_order = self.rng.permutation(self.primary_len)
                    pos = 0
                batch.extend(primary_order[pos : pos + self.n_primary].tolist())
                pos += self.n_primary
            if self.n_secondary and self.secondary_weights is not None:
                sec = self.rng.choice(self.secondary_len, size=self.n_secondary, replace=True, p=self.secondary_weights)
                batch.extend((sec + self.primary_len).tolist())
            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
