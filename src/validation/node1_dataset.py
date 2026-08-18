"""Scopes the full PlantVillage dataset down to node_1's crops (per
config.yaml's manual_node_crops) and splits it into a train/eval pair using
a dedup-aware split, so near-duplicate PlantVillage shots (the same
physical leaf photographed more than once) can't leak across the split.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from src.config import Config
from src.data.plantvillage import PlantVillageDataset, partition_nodes
from src.validation.hashing import average_hash, hamming_distance

# Position of "node_1" in partition_nodes()'s output list — node keys in
# config.yaml's manual_node_crops are literally "node_0", "node_1", "node_2",
# and _manual_partition() (src/data/plantvillage.py) maps them to this same
# integer position regardless of dict iteration order.
NODE1_INDEX = 1


def get_node1_indices(dataset: PlantVillageDataset, cfg: Config) -> list[int]:
    """Node_1's raw sample indices, via the same partition_nodes() the real
    mesh pipeline uses — no probe-set carve-out here (this is a local-only
    training run, not mesh/distillation), so every sample index is passed
    in as "remaining".
    """
    all_indices = list(range(len(dataset)))
    shards = partition_nodes(
        dataset,
        all_indices,
        cfg.get("data.num_nodes", 3),
        "manual",
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    return shards[NODE1_INDEX]


def compute_image_hashes(dataset: PlantVillageDataset, indices: list[int]) -> dict[int, int]:
    """Perceptual hash per sample index (index is the same one used by
    dataset.base.samples / dataset.base.targets everywhere else).
    """
    hashes: dict[int, int] = {}
    for idx in indices:
        path, _ = dataset.base.samples[idx]
        with Image.open(path) as img:
            hashes[idx] = average_hash(img.convert("RGB"))
    return hashes


def group_duplicates(indices: list[int], hashes: dict[int, int], threshold: int = 5) -> dict[int, int]:
    """Union-find over `indices`: any two whose hashes are within Hamming
    distance <= threshold land in the same group. Returns index -> group_id
    (group_id is one representative index per group).

    O(n^2) pairwise comparisons — fine for node_1's few-thousand-image
    scope, not intended for the full dataset.
    """
    parent = {idx: idx for idx in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(indices):
        for b in indices[i + 1 :]:
            if hamming_distance(hashes[a], hashes[b]) <= threshold:
                union(a, b)

    return {idx: find(idx) for idx in indices}


def dedup_aware_split(
    indices: list[int],
    hashes: dict[int, int],
    test_fraction: float,
    seed: int,
    threshold: int = 5,
) -> tuple[list[int], list[int]]:
    """Like the existing train_test_split_indices, but splits at the
    duplicate-group level (see group_duplicates) instead of the raw index
    level.
    """
    groups = group_duplicates(indices, hashes, threshold)
    group_members: dict[int, list[int]] = {}
    for idx, group_id in groups.items():
        group_members.setdefault(group_id, []).append(idx)

    group_ids = list(group_members.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_test_target = max(1, int(len(indices) * test_fraction)) if len(indices) > 1 else 0
    train_idx: list[int] = []
    test_idx: list[int] = []
    for group_id in group_ids:
        members = group_members[group_id]
        if len(test_idx) < n_test_target:
            test_idx.extend(members)
        else:
            train_idx.extend(members)
    return train_idx, test_idx
