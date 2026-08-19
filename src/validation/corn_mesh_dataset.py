"""Scopes the full PlantVillage dataset down to Corn only, and remaps
Corn's 4 classes (healthy + 3 diseases) into a compact, Corn-only label
space -- see docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md
for why the head is scoped this tightly instead of reusing the full
14-crop/~22-disease global label space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from torch.utils.data import Dataset

from src.config import Config
from src.data.plantvillage import PlantVillageDataset, carve_public_probe_set, load_full_dataset
from src.validation.node1_dataset import (
    build_train_eval_datasets,
    compute_image_hashes,
    dedup_aware_split,
    group_duplicates,
)

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


class CornDiseaseView(Dataset):
    """Wraps a PlantVillageDataset + a fixed list of its raw sample
    indices, remapping each sample's disease label into the compact
    Corn-only label space (see build_corn_label_map). Crop label is
    always 0 -- there is only one crop in this experiment.
    """

    def __init__(self, base: PlantVillageDataset, indices: list[int], label_map: CornLabelMap):
        self.base = base
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        image, _, disease_idx = self.base[self.indices[pos]]
        disease_name = self.base.labels.disease_classes[disease_idx]
        compact_idx = self.label_map.name_to_disease_idx[disease_name]
        return image, 0, compact_idx


@dataclass
class CornMeshData:
    train_base: PlantVillageDataset
    eval_base: PlantVillageDataset
    label_map: CornLabelMap
    image_size: int
    probe_idx: list[int]
    per_node: dict[str, dict[str, list[int]]] = field(default_factory=dict)


def prepare_corn_mesh_data(cfg: Config) -> CornMeshData:
    image_size = cfg.get("data.image_size", 160)
    root = cfg.get("data.root", "data/PlantVillage")
    seed = cfg.get("data.seed", 42)
    test_fraction = cfg.get("data.test_fraction", 0.15)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    healthy_dedup_threshold = cfg.get("corn_mesh.healthy_dedup_threshold", 5)
    node_diseases = cfg.get("corn_mesh.node_diseases", NODE_DISEASE_DEFAULT)

    dataset = load_full_dataset(root, image_size)
    label_map = build_corn_label_map()

    corn_indices = get_corn_indices(dataset)
    corn_set = set(corn_indices)

    probe_idx_full, remaining_idx_full = carve_public_probe_set(
        dataset,
        probe_fraction,
        seed,
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    probe_idx = [i for i in probe_idx_full if i in corn_set]
    remaining_corn_idx = [i for i in remaining_idx_full if i in corn_set]

    healthy_remaining = get_corn_disease_indices(dataset, remaining_corn_idx, "healthy")
    healthy_shares = split_healthy_3way(
        dataset, healthy_remaining, num_nodes=3, seed=seed, threshold=healthy_dedup_threshold
    )

    per_node: dict[str, dict[str, list[int]]] = {}
    for node_id, disease_name in node_diseases.items():
        node_idx = int(str(node_id).rsplit("_", 1)[-1])
        disease_idx_list = get_corn_disease_indices(dataset, remaining_corn_idx, disease_name)
        combined = disease_idx_list + healthy_shares[node_idx]
        hashes = compute_image_hashes(dataset, combined)
        train_idx, test_idx = dedup_aware_split(
            combined, hashes, test_fraction, seed, threshold=healthy_dedup_threshold
        )
        per_node[node_id] = {"train_idx": train_idx, "test_idx": test_idx}

    train_base, eval_base = build_train_eval_datasets(root, image_size)
    return CornMeshData(
        train_base=train_base,
        eval_base=eval_base,
        label_map=label_map,
        image_size=image_size,
        probe_idx=probe_idx,
        per_node=per_node,
    )
