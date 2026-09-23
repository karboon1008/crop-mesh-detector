"""Turns the PlantVillage pool into the splits the mesh runs on:

    PlantVillage (all classes)
    ├── global probe set  (data.probe_set_fraction, default 5%, stratified)
    │     public, identical for every node, FIXED for every continual batch
    └── private pool      (the other 95%)
          └── partition_nodes -> 6 non-IID node shards
                └── each shard -> continual.num_batches batches (a stream)
                      └── each batch -> private train / private test

The probe set is carved once and never changes across batches: a node that
stops uploading keeps its last payload in the knowledge store, so its probe
logits have to stay positionally aligned with every later batch's logits.

Two ways to cut a node's shard into a stream (continual.batch_strategy):

  - "stratified":  every batch is a stratified random 1/num_batches slice
                   of the shard — same class mix each batch, just new images
                   (tests "more data of the same kind keeps arriving").
  - "incremental": a node's classes arrive over time — the first
                   `initial_class_fraction` of its classes appear in batch
                   0, the rest are introduced evenly over the later
                   batches, and a class keeps appearing in every batch from
                   the one it was introduced in (tests "new diseases show up
                   on the farm while old ones persist" — the case where
                   peers that already know a class should help).

Each batch is then split into private train/test (data.test_fraction),
stratified per class, and batches never share an image.
"""

from __future__ import annotations

import math
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

BATCH_STRATEGIES = ("stratified", "incremental")


@dataclass
class BatchSplit:
    train_idx: list[int]
    test_idx: list[int]


@dataclass
class ContinualSplits:
    probe_idx: list[int]
    # node_batches[node][batch] -> that node's private train/test for that batch
    node_batches: list[list[BatchSplit]]


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


def split_shard_into_batches(
    dataset,
    shard: list[int],
    num_batches: int,
    strategy: str = "stratified",
    initial_class_fraction: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Cuts one node's shard into `num_batches` disjoint index lists (see
    the module docstring for the two strategies). Every shard index lands
    in exactly one batch.
    """
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if strategy not in BATCH_STRATEGIES:
        raise ValueError(f"Unknown continual.batch_strategy: {strategy!r} (expected one of {BATCH_STRATEGIES})")

    rng = random.Random(seed)
    targets = np.asarray(dataset.targets)[shard]
    by_class: dict[int, list[int]] = {}
    for idx, cls in zip(shard, targets.tolist()):
        by_class.setdefault(cls, []).append(idx)
    classes = sorted(by_class)

    # class -> first batch it appears in
    if strategy == "stratified" or num_batches == 1:
        first_batch = {cls: 0 for cls in classes}
    else:
        order = classes.copy()
        rng.shuffle(order)
        n_initial = min(len(order), max(1, math.ceil(len(order) * initial_class_fraction)))
        first_batch = {cls: 0 for cls in order[:n_initial]}
        later = order[n_initial:]
        for offset, group in enumerate(np.array_split(np.array(later, dtype=int), num_batches - 1)):
            for cls in group.tolist():
                first_batch[cls] = offset + 1

    batches: list[list[int]] = [[] for _ in range(num_batches)]
    for cls in classes:
        cls_indices = by_class[cls].copy()
        rng.shuffle(cls_indices)
        active = list(range(first_batch[cls], num_batches))
        for batch_idx, chunk in zip(active, np.array_split(np.array(cls_indices, dtype=int), len(active))):
            batches[batch_idx].extend(chunk.tolist())
    for batch in batches:
        rng.shuffle(batch)
    return batches


def build_continual_splits(cfg, dataset) -> ContinualSplits:
    probe_idx, shards = carve_probe_and_partition(cfg, dataset)
    seed = cfg.get("data.seed", 42)
    test_fraction = cfg.get("data.test_fraction", 0.2)
    node_batches = []
    for node_i, shard in enumerate(shards):
        batches = split_shard_into_batches(
            dataset,
            shard,
            cfg.get("continual.num_batches", 4),
            cfg.get("continual.batch_strategy", "stratified"),
            cfg.get("continual.initial_class_fraction", 0.5),
            seed=seed + node_i,
        )
        node_batches.append([
            BatchSplit(*train_test_split_indices(dataset, batch, test_fraction, seed)) for batch in batches
        ])
    empty = [
        f"node_{n} batch {b} ({len(s.train_idx)} train / {len(s.test_idx)} test)"
        for n, batches in enumerate(node_batches) for b, s in enumerate(batches)
        if not s.train_idx or not s.test_idx
    ]
    if empty:
        raise ValueError(
            "Continual split left a node batch with no train or no test images: " + ", ".join(empty)
            + " — lower continual.num_batches or data.num_nodes, or raise data.dirichlet_alpha."
        )
    return ContinualSplits(probe_idx=probe_idx, node_batches=node_batches)


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
