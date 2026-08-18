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
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# mobilenet & efficient net is trained on imagenet dataset so they share the one normalization convention
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform(image_size: int):
    """Training-only augmentation. Resize/Normalize, random
    crop+zoom, rotation/flip, color jitter, perspective skew, blur, and
    patch erasure. Applied to train splits only.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


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
        self.transform = build_eval_transform(image_size)
        self.train_transform = build_train_transform(image_size)
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


def stratified_carve_by_labels(
    indices: list[int],
    labels: list[int],
    fraction: float,
    seed: int,
    large_class_threshold: int,
    min_samples_small_class: int,
    max_fraction_small_class: float,
) -> tuple[list[int], list[int]]:
    """Stratified carve/remaining split by an arbitrary per-sample label
    (`labels`, aligned 1:1 with `indices`) — the label-agnostic core behind
    `_stratified_carve` below, factored out so non-PlantVillage datasets
    (PlantDoc, PlantWild) can carve their own probe/global-test slices the
    same way, stratified by their own raw class, without needing a
    PlantVillageDataset.
    """
    labels_arr = np.asarray(labels)
    rng = random.Random(seed)
    carved: list[int] = []
    remaining: list[int] = []
    for cls in sorted(set(labels_arr.tolist())):
        cls_indices = [indices[i] for i in range(len(indices)) if labels_arr[i] == cls]
        rng.shuffle(cls_indices)
        count = len(cls_indices)
        if count >= large_class_threshold:
            n_cls = max(1, round(count * fraction))
        else:
            target_n = max(min_samples_small_class, round(count * fraction))
            n_cls = min(target_n, int(count * max_fraction_small_class), count)
        carved.extend(cls_indices[:n_cls])
        remaining.extend(cls_indices[n_cls:])
    rng.shuffle(carved)
    rng.shuffle(remaining)
    return carved, remaining


def _stratified_carve(
    dataset: PlantVillageDataset,
    indices: list[int],
    fraction: float,
    seed: int,
    large_class_threshold: int,
    min_samples_small_class: int,
    max_fraction_small_class: float,
) -> tuple[list[int], list[int]]:
    labels = np.array(dataset.targets)[indices]
    return stratified_carve_by_labels(
        indices, labels, fraction, seed, large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )


def carve_public_probe_set(
    dataset: PlantVillageDataset,
    probe_fraction: float,
    seed: int,
    large_class_threshold: int = 200,
    min_samples_small_class: int = 8,
    max_fraction_small_class: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Split all indices into (probe_indices, remaining_indices), stratified
    by original ImageFolder class.

    The probe set is PUBLIC and IDENTICAL across every node: it is used
    only to compute soft logits for knowledge exchange (Itahara et al.,
    DS-FL), never for local training, so it carries no per-farm private
    information.
    """
    all_indices = list(range(len(dataset)))
    return _stratified_carve(
        dataset, all_indices, probe_fraction, seed,
        large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )


def carve_global_test_set(
    dataset: PlantVillageDataset,
    indices: list[int],
    test_fraction: float,
    seed: int,
    large_class_threshold: int = 200,
    min_samples_small_class: int = 8,
    max_fraction_small_class: float = 0.2,
) -> tuple[list[int], list[int]]:
    return _stratified_carve(
        dataset, indices, test_fraction, seed,
        large_class_threshold, min_samples_small_class, max_fraction_small_class,
    )


def partition_nodes(
    dataset: PlantVillageDataset,
    remaining_indices: list[int],
    num_nodes: int,
    strategy: str,
    dirichlet_alpha: float,
    seed: int,
    manual_node_crops: dict[str, list[str]] | None = None,
) -> list[list[int]]:
    """Split `remaining_indices` into `num_nodes` non-IID shards.

    strategy:
      - "by_crop":    each node sees a disjoint subset of crop species,
                      auto-assigned round-robin (mirrors different farms
                      growing different crops).
      - "by_disease": each node sees a disjoint subset of original
                      crop-disease classes (mirrors different regional
                      disease prevalence within similar crops).
      - "dirichlet":  classic label-skew partition via a symmetric
                      Dirichlet distribution over class proportions
                      per node (Zhu et al.'s non-IID survey; Q. Li et al.).
      - "manual":     each node gets exactly the crop species named for it
                      in `manual_node_crops` (e.g. an "orchard farm" node
                      grows Apple/Cherry/Peach/Blueberry/Raspberry) — same
                      disjoint-by-crop shape as "by_crop", but the farm/crop
                      assignment is explicit instead of round-robin.
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
    elif strategy == "manual":
        return _manual_partition(dataset, remaining_indices, targets, num_nodes, manual_node_crops)
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


def _manual_partition(
    dataset: PlantVillageDataset,
    indices: list[int],
    targets: np.ndarray,
    num_nodes: int,
    manual_node_crops: dict[str, list[str]] | None,
) -> list[list[int]]:
    if not manual_node_crops:
        raise ValueError("non_iid_strategy 'manual' requires data.manual_node_crops in config.yaml")

    crop_to_node: dict[str, int] = {}
    for node_key, crops in manual_node_crops.items():
        node_idx = int(str(node_key).rsplit("_", 1)[-1])
        for crop in crops:
            if crop in crop_to_node:
                raise ValueError(f"Crop '{crop}' assigned to more than one node in manual_node_crops")
            crop_to_node[crop] = node_idx

    all_crops = set(dataset.labels.crop_classes)
    missing = all_crops - set(crop_to_node)
    if missing:
        raise ValueError(f"manual_node_crops is missing an assignment for: {sorted(missing)}")
    unknown = set(crop_to_node) - all_crops
    if unknown:
        raise ValueError(f"manual_node_crops references unknown crop(s): {sorted(unknown)}")

    shards: list[list[int]] = [[] for _ in range(num_nodes)]
    for local_pos, global_idx in enumerate(indices):
        crop_idx, _ = dataset.labels.class_to_crop_disease[int(targets[local_pos])]
        crop_name = dataset.labels.crop_classes[crop_idx]
        shards[crop_to_node[crop_name]].append(global_idx)
    return shards


def train_test_split_indices(
    indices: list[int], test_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
    return shuffled[n_test:], shuffled[:n_test]


class TransformedSubset(Dataset):
    def __init__(self, dataset: PlantVillageDataset, indices: list[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        image, class_idx = self.dataset.base[idx]  # raw PIL image, ImageFolder has no transform of its own
        image = self.transform(image)
        crop_idx, disease_idx = self.dataset.labels.class_to_crop_disease[class_idx]
        return image, crop_idx, disease_idx


def make_subset(dataset: PlantVillageDataset, indices: list[int], train: bool = False) -> Dataset:
    """train=True applies the dataset's augmentation transform (for a node's
    training split); train=False (default) applies the plain eval transform
    (test/probe/global-test splits, so accuracy stays comparable/deterministic).
    """
    transform = dataset.train_transform if train else dataset.transform
    return TransformedSubset(dataset, indices, transform)


def _count_pairs(pairs: list[tuple[int, int]], pair_index: int, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for pair in pairs:
        counts[pair[pair_index]] += 1
    return counts


def _inverse_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights from raw per-class counts, computed on
    whatever subset they were counted from (e.g. a single node's local,
    non-IID shard) — this reweights the loss towards under-represented
    classes in that subset without touching the subset's own label
    distribution.
    """
    num_classes = counts.numel()
    weights = torch.ones(num_classes)
    present = counts > 0
    weights[present] = counts.sum() / (num_classes * counts[present])
    return weights


def _pairs_from_indices(dataset: PlantVillageDataset, indices: list[int]) -> list[tuple[int, int]]:
    targets = np.asarray(dataset.targets)[indices]
    return [dataset.labels.class_to_crop_disease[int(t)] for t in targets]


def compute_disease_class_weights_from_pairs(pairs: list[tuple[int, int]], num_classes: int) -> torch.Tensor:
    """Same inverse-frequency disease-class weighting as
    `compute_disease_class_weights`, for datasets that aren't a
    PlantVillageDataset (PlantDoc, PlantWild) — takes their own
    (crop_idx, disease_idx) pairs directly (see the `.pairs` property on
    PlantDocDataset/PlantWildDataset).
    """
    return _inverse_frequency_weights(_count_pairs(pairs, pair_index=1, num_classes=num_classes))


def compute_crop_class_weights_from_pairs(pairs: list[tuple[int, int]], num_classes: int) -> torch.Tensor:
    """Same inverse-frequency crop-class weighting as
    `compute_crop_class_weights`, for datasets that aren't a
    PlantVillageDataset (PlantDoc, PlantWild) — see
    `compute_disease_class_weights_from_pairs`.
    """
    return _inverse_frequency_weights(_count_pairs(pairs, pair_index=0, num_classes=num_classes))


def compute_disease_class_weights(dataset: PlantVillageDataset, indices: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights for the disease head's loss,
    computed from this subset's own label counts.
    """
    return compute_disease_class_weights_from_pairs(
        _pairs_from_indices(dataset, indices), num_classes=len(dataset.labels.disease_classes)
    )


def compute_crop_class_weights(dataset: PlantVillageDataset, indices: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights for the crop head's loss,
    computed from this subset's own label counts.
    """
    return compute_crop_class_weights_from_pairs(
        _pairs_from_indices(dataset, indices), num_classes=len(dataset.labels.crop_classes)
    )
