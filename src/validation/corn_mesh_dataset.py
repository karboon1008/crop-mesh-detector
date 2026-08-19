"""Scopes the full PlantVillage dataset down to Corn only, and remaps
Corn's 4 classes (healthy + 3 diseases) into a compact, Corn-only label
space -- see docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md
for why the head is scoped this tightly instead of reusing the full
14-crop/~22-disease global label space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.data.plantvillage import PlantVillageDataset
from src.validation.node1_dataset import compute_image_hashes, group_duplicates

CORN_DISEASE_ORDER = [
    "healthy",
    "Common_rust",
    "Cercospora_leaf_spot_Gray_leaf_spot",
    "Northern_Leaf_Blight",
]

NODE_DISEASE_DEFAULT = {
    "node_0": "Common_rust",
    "node_1": "Cercospora_leaf_spot_Gray_leaf_spot",
    "node_2": "Northern_Leaf_Blight",
}


@dataclass
class CornLabelMap:
    crop_classes: list[str]
    disease_classes: list[str]
    name_to_disease_idx: dict[str, int]


def build_corn_label_map(disease_names: list[str] = CORN_DISEASE_ORDER) -> CornLabelMap:
    return CornLabelMap(
        crop_classes=["Corn"],
        disease_classes=list(disease_names),
        name_to_disease_idx={name: i for i, name in enumerate(disease_names)},
    )


def get_corn_indices(dataset: PlantVillageDataset) -> list[int]:
    corn_crop_idx = dataset.labels.crop_classes.index("Corn")
    indices = []
    for idx in range(len(dataset)):
        crop_idx, _ = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        if crop_idx == corn_crop_idx:
            indices.append(idx)
    return indices


def get_corn_disease_indices(
    dataset: PlantVillageDataset, indices: list[int], disease_name: str
) -> list[int]:
    result = []
    for idx in indices:
        _, disease_idx = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        if dataset.labels.disease_classes[disease_idx] == disease_name:
            result.append(idx)
    return result


def _assign_groups_to_shares(
    group_list: list[list[int]], num_nodes: int, seed: int
) -> list[list[int]]:
    """Greedy load-balancing: largest groups first, each assigned to
    whichever share currently has the fewest images -- keeps every
    duplicate-group intact on one share while balancing share sizes.
    """
    rng = random.Random(seed)
    shuffled = list(group_list)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    shares: list[list[int]] = [[] for _ in range(num_nodes)]
    for members in shuffled:
        target = min(range(num_nodes), key=lambda i: len(shares[i]))
        shares[target].extend(members)
    return shares


def split_healthy_3way(
    dataset: PlantVillageDataset,
    healthy_indices: list[int],
    num_nodes: int,
    seed: int,
    threshold: int = 5,
) -> list[list[int]]:
    hashes = compute_image_hashes(dataset, healthy_indices)
    groups = group_duplicates(healthy_indices, hashes, threshold)
    group_members: dict[int, list[int]] = {}
    for idx, group_id in groups.items():
        group_members.setdefault(group_id, []).append(idx)
    return _assign_groups_to_shares(list(group_members.values()), num_nodes, seed)
