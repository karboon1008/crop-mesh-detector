"""Splitting helpers for the PlantVillage pool:

    PlantVillage (all classes)
    ├── global probe set  (data.probe_set_fraction, default 5%, stratified)
    │     public, identical for every node, FIXED for every continual batch
    └── private pool      (the other 95%)
          └── partition_nodes -> which node each image belongs to (non-IID)

The continual stream (src/data/stream.py) then draws each batch as a
stratified sample of the images not used yet, and each node splits its own
arrivals into private train/test only when the batch arrives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader

from src.data.plantvillage import (
    carve_public_probe_set,
    compute_crop_class_weights,
    compute_disease_class_weights,
    make_subset,
    partition_nodes,
    train_test_split_indices,
)


@dataclass
class BatchSplit:
    train_idx: list[int]
    test_idx: list[int]


def carve_probe_and_partition(cfg, dataset) -> tuple[list[int], list[list[int]]]:
    """(probe_idx, node_shards): the global probe carve, then the non-IID
    partition of everything left over into data.num_nodes private shards.
    """
    probe_idx, remaining_idx = carve_public_probe_set(
        dataset,
        cfg.get("data.probe_set_fraction", 0.05),
        cfg.get("data.seed", 42),
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 6),
        cfg.get("data.non_iid_strategy", "dirichlet"),
        cfg.get("data.dirichlet_alpha", 0.5),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    return probe_idx, shards


def stratified_sample(dataset, pool: list[int], n: int, seed: int) -> list[int]:
    """Exactly min(n, len(pool)) indices from `pool`, with every class
    represented in proportion to its share of `pool` (largest-remainder
    rounding, so the per-class counts always add up to n).
    """
    n = min(n, len(pool))
    rng = random.Random(seed)
    targets = np.asarray(dataset.targets)[pool]
    by_class: dict[int, list[int]] = {}
    for idx, cls in zip(pool, targets.tolist()):
        by_class.setdefault(cls, []).append(idx)
    classes = sorted(by_class)
    quotas = {cls: n * len(by_class[cls]) / len(pool) for cls in classes}
    counts = {cls: int(quotas[cls]) for cls in classes}
    shortfall = n - sum(counts.values())
    for cls in sorted(classes, key=lambda c: (quotas[c] - counts[c], len(by_class[c])), reverse=True)[:shortfall]:
        counts[cls] += 1
    sample: list[int] = []
    for cls in classes:
        sample.extend(rng.sample(by_class[cls], counts[cls]))
    rng.shuffle(sample)
    return sample


def split_node_arrival(dataset, indices: list[int], test_fraction: float, seed: int) -> BatchSplit:
    """A node's own train/test split of the images it just received —
    stratified per class, and with at least one test image whenever it got
    two or more (a small arrival where every class has a single image would
    otherwise put everything in train and leave nothing to evaluate on).
    """
    train_idx, test_idx = train_test_split_indices(dataset, indices, test_fraction, seed)
    if not test_idx and len(train_idx) >= 2:
        rng = random.Random(seed)
        test_idx = [train_idx.pop(rng.randrange(len(train_idx)))]
    return BatchSplit(train_idx, test_idx)


def build_probe_loader(cfg, dataset, probe_idx: list[int]) -> DataLoader:
    # shuffle=False is load-bearing: a probe batch's position is how it's
    # matched back to the consensus logits during distillation.
    return DataLoader(make_subset(dataset, probe_idx), batch_size=cfg.get("training.batch_size", 32), shuffle=False)


def build_batch_loaders(cfg, dataset, split: BatchSplit):
    """(train_loader, test_loader, crop_class_weights, disease_class_weights)
    for one node's one batch.
    """
    batch_size = cfg.get("training.batch_size", 32)
    # a trailing batch of exactly one image crashes BatchNorm in train mode
    train_loader = DataLoader(
        make_subset(dataset, split.train_idx, train=True), batch_size=batch_size, shuffle=True,
        drop_last=len(split.train_idx) % batch_size == 1,
    )
    test_loader = DataLoader(make_subset(dataset, split.test_idx), batch_size=batch_size, shuffle=False)
    crop_weights = (
        compute_crop_class_weights(dataset, split.train_idx)
        if cfg.get("training.crop_class_balanced", False) else None
    )
    disease_weights = (
        compute_disease_class_weights(dataset, split.train_idx)
        if cfg.get("training.disease_class_balanced", True) else None
    )
    return train_loader, test_loader, crop_weights, disease_weights


def build_dataloaders(cfg, dataset):
    """Static (non-continual) split used by the mesh disruption scenarios
    (src/scenarios/): the same probe carve + node partition, but each
    node's whole shard is one train/test split.

    Returns (probe_loader, node_loaders, crop_class_weights, disease_class_weights).
    """
    probe_idx, shards = carve_probe_and_partition(cfg, dataset)
    node_loaders, crop_class_weights, disease_class_weights = [], [], []
    for shard in shards:
        split = BatchSplit(*train_test_split_indices(
            dataset, shard, cfg.get("data.test_fraction", 0.2), cfg.get("data.seed", 42)
        ))
        train_loader, test_loader, crop_weights, disease_weights = build_batch_loaders(cfg, dataset, split)
        node_loaders.append((train_loader, test_loader))
        crop_class_weights.append(crop_weights)
        disease_class_weights.append(disease_weights)
    return build_probe_loader(cfg, dataset, probe_idx), node_loaders, crop_class_weights, disease_class_weights
