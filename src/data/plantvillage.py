"""PlantVillage loading, crop/disease label parsing, and non-IID partitioning
across simulated farm nodes.

Expected layout (the standard PlantVillage release, one folder per class,
crop and disease encoded in the folder name):

    data/PlantVillage/
        Tomato___Bacterial_spot/
        Tomato___healthy/
        Potato___Early_blight/
        Pepper__bell___healthy/
        ...

Each class folder name is parsed into (crop, disease) so every image gets
TWO labels — crop type and disease status — trained with a shared backbone
and two heads (see src/models/factory.py). This is done once, locally,
per node; only derived, non-invertible artefacts ever leave a node
(see src/federated/).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _parse_crop_disease(class_name: str) -> tuple[str, str]:
    """'Tomato___Bacterial_spot' -> ('Tomato', 'Bacterial_spot').

    PlantVillage folder names separate crop and disease with a run of
    underscores (2 or 3 depending on the mirror); we split on the first
    run of 2+ underscores and treat everything after it as the disease
    label (defaulting to 'healthy' when absent).
    """
    import re

    parts = re.split(r"_{2,}", class_name, maxsplit=1)
    crop = parts[0].strip("_")
    disease = parts[1].strip("_") if len(parts) > 1 else "healthy"
    return crop, disease


@dataclass
class LabelMaps:
    crop_classes: list[str]
    disease_classes: list[str]
    # original ImageFolder class index -> (crop_idx, disease_idx)
    class_to_crop_disease: dict[int, tuple[int, int]] = field(default_factory=dict)


class PlantVillageDataset(Dataset):
    """Wraps torchvision's ImageFolder, exposing (image, crop_label, disease_label)."""

    def __init__(self, root: str | Path, image_size: int = 160):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.base = ImageFolder(str(root))
        self.labels = self._build_label_maps(self.base.classes)

    @staticmethod
    def _build_label_maps(class_names: list[str]) -> LabelMaps:
        crops: list[str] = []
        diseases: list[str] = []
        parsed = [_parse_crop_disease(c) for c in class_names]
        for crop, disease in parsed:
            if crop not in crops:
                crops.append(crop)
            if disease not in diseases:
                diseases.append(disease)
        crop_to_idx = {c: i for i, c in enumerate(crops)}
        disease_to_idx = {d: i for i, d in enumerate(diseases)}
        class_to_crop_disease = {
            i: (crop_to_idx[crop], disease_to_idx[disease])
            for i, (crop, disease) in enumerate(parsed)
        }
        return LabelMaps(crops, diseases, class_to_crop_disease)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, class_idx = self.base[idx]
        image = self.transform(image)
        crop_idx, disease_idx = self.labels.class_to_crop_disease[class_idx]
        return image, crop_idx, disease_idx

    @property
    def targets(self) -> list[int]:
        """Original ImageFolder class index per sample — used for partitioning."""
        return self.base.targets


def load_full_dataset(root: str | Path, image_size: int = 160) -> PlantVillageDataset:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"PlantVillage data not found at {root}. Run "
            f"'python scripts/download_plantvillage.py' first, or point "
            f"config.yaml's data.root at your existing copy."
        )
    return PlantVillageDataset(root, image_size=image_size)


def carve_public_probe_set(
    dataset: PlantVillageDataset, probe_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split all indices into (probe_indices, remaining_indices).

    The probe set is PUBLIC and IDENTICAL across every node: it is used
    only to compute soft logits for knowledge exchange (Itahara et al.,
    DS-FL), never for local training, so it carries no per-farm private
    information.
    """
    n = len(dataset)
    rng = random.Random(seed)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    n_probe = max(1, int(n * probe_fraction))
    return all_idx[:n_probe], all_idx[n_probe:]


def partition_nodes(
    dataset: PlantVillageDataset,
    remaining_indices: list[int],
    num_nodes: int,
    strategy: str,
    dirichlet_alpha: float,
    seed: int,
) -> list[list[int]]:
    """Split `remaining_indices` into `num_nodes` non-IID shards.

    strategy:
      - "by_crop":    each node sees a disjoint subset of crop species
                      (mirrors different farms growing different crops).
      - "by_disease": each node sees a disjoint subset of original
                      crop-disease classes (mirrors different regional
                      disease prevalence within similar crops).
      - "dirichlet":  classic label-skew partition via a symmetric
                      Dirichlet distribution over class proportions
                      per node (Zhu et al.'s non-IID survey; Q. Li et al.).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)[remaining_indices]

    if strategy == "by_crop":
        class_to_crop = {c: cd[0] for c, cd in dataset.labels.class_to_crop_disease.items()}
        group_key = np.array([class_to_crop[t] for t in targets])
    elif strategy == "by_disease":
        group_key = targets
    elif strategy == "dirichlet":
        return _dirichlet_partition(remaining_indices, targets, num_nodes, dirichlet_alpha, rng)
    else:
        raise ValueError(f"Unknown non_iid_strategy: {strategy}")

    unique_groups = sorted(set(group_key.tolist()))
    rng.shuffle(unique_groups)
    group_to_node = {g: unique_groups.index(g) % num_nodes for g in unique_groups}

    shards: list[list[int]] = [[] for _ in range(num_nodes)]
    for local_pos, global_idx in enumerate(remaining_indices):
        node_id = group_to_node[group_key[local_pos]]
        shards[node_id].append(global_idx)
    return shards


def _dirichlet_partition(
    indices: list[int],
    targets: np.ndarray,
    num_nodes: int,
    alpha: float,
    rng: np.random.RandomState,
) -> list[list[int]]:
    shards: list[list[int]] = [[] for _ in range(num_nodes)]
    for cls in sorted(set(targets.tolist())):
        cls_indices = [indices[i] for i in range(len(indices)) if targets[i] == cls]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(alpha=[alpha] * num_nodes)
        counts = (proportions * len(cls_indices)).astype(int)
        counts[-1] = len(cls_indices) - counts[:-1].sum()  # fix rounding drift
        start = 0
        for node_id, count in enumerate(counts):
            shards[node_id].extend(cls_indices[start : start + count])
            start += count
    return shards


def train_test_split_indices(
    indices: list[int], test_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_test:], shuffled[:n_test]


def make_subset(dataset: PlantVillageDataset, indices: list[int]) -> Subset:
    return Subset(dataset, indices)
