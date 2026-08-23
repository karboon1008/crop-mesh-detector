"""Tomato+Apple combined multi-crop Dirichlet mesh dataset preparation --
pools PlantVillage + PlantDoc + PlantWild (v1+v2) for BOTH crops into one
canonical (crop, disease) label space, carves a GLOBAL dedup-aware test
split, then Dirichlet-partitions the remaining training pool across nodes.
Direct mirror of apple_mesh_dataset.py / tomato_mesh_dataset.py, extended
to a second crop dimension.

Unlike the single-crop mesh datasets (where crop_classes=["Apple"] or
["Tomato"] and crop_accuracy is trivially 1.0), this dataset gives the
model's crop head a real two-way decision (Apple vs Tomato) and lets the
Dirichlet draw skew crop MIX per node, not just disease mix within one
crop -- e.g. a node can end up Apple-heavy/Tomato-light or vice versa,
independently of its per-crop disease skew. This mirrors a real
mixed-orchard/market-garden farm more closely than either single-crop
mesh does on its own.

Disease label space: the union of APPLE_DISEASE_ORDER and
TOMATO_DISEASE_ORDER, with the shared "healthy" name collapsed to ONE
disease-head index for both crops (same convention as the original
full-PlantVillage global label map in src/data/plantvillage.py, where
"healthy" is one class shared across all 14 crops) -- an Apple-healthy
and a Tomato-healthy image both count toward the same "healthy" bucket
for the disease head, and are only distinguished via the separate crop
head.

Dirichlet partitioning is done over the (crop, disease) PAIR, not disease
alone -- so "Apple healthy" and "Tomato healthy" get independently-drawn
per-node proportions rather than being shuffled together into one shared
per-node "healthy" bucket. This keeps each node's crop mix legible instead
of accidentally averaging it out.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, _dirichlet_partition, load_full_dataset
from src.validation.hashing import average_hash
from src.validation.node1_dataset import group_duplicates

logger = logging.getLogger(__name__)


def _win_long_path(path: str) -> str:
    """Same `\\\\?\\` extended-length-path fix as apple_mesh_dataset.py --
    needed here too since the Apple portion of this pool includes the same
    PlantDoc stock-photo filename that exceeds Windows' 260-char MAX_PATH.
    No-op on non-Windows platforms and already-prefixed/UNC paths.
    """
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


APPLE_DISEASE_ORDER = [
    "Apple_scab",
    "Black_rot",
    "Cedar_apple_rust",
    "healthy",
]

TOMATO_DISEASE_ORDER = [
    "Bacterial_spot",
    "Early_blight",
    "healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
]

CROP_ORDER = ["Apple", "Tomato"]


def _combined_disease_order() -> list[str]:
    """Apple's 4 names, then any Tomato names not already present (only
    "healthy" overlaps) -- deterministic order so repeated calls / repeated
    runs against the same config always produce the same disease index
    assignment.
    """
    combined = list(APPLE_DISEASE_ORDER)
    for name in TOMATO_DISEASE_ORDER:
        if name not in combined:
            combined.append(name)
    return combined


@dataclass
class TomatoAppleLabelMap:
    crop_classes: list[str]
    disease_classes: list[str]
    name_to_disease_idx: dict[str, int]
    crop_name_to_idx: dict[str, int]


def build_tomato_apple_label_map(
    disease_names: list[str] | None = None, crop_names: list[str] | None = None
) -> TomatoAppleLabelMap:
    disease_names = list(disease_names) if disease_names is not None else _combined_disease_order()
    crop_names = list(crop_names) if crop_names is not None else list(CROP_ORDER)
    return TomatoAppleLabelMap(
        crop_classes=crop_names,
        disease_classes=disease_names,
        name_to_disease_idx={name: i for i, name in enumerate(disease_names)},
        crop_name_to_idx={name: i for i, name in enumerate(crop_names)},
    )


@dataclass
class TomatoAppleRawItem:
    source: str
    path: str
    canonical_crop_idx: int
    canonical_disease_idx: int


class TomatoAppleMergedDataset(Dataset):
    """A flat, canonical-label-space dataset over pooled multi-source,
    multi-crop (Apple + Tomato) items. Exposes the same
    `.base.samples`/`.base.targets`/`.labels.class_to_crop_disease`/
    `.labels.crop_classes`/`.labels.disease_classes` surface as
    PlantVillageDataset so train_mobilenet.run_training,
    export_onnx.export_checkpoint, and evaluate_onnx.run_evaluation all
    work against it unmodified.

    Unlike AppleMergedDataset/TomatoMergedDataset (where crop_idx is
    always 0 and `base.samples`' second element IS the disease index),
    here `base.samples`/`base.targets` store a FLATTENED
    `crop_idx * num_disease_classes + disease_idx` combined index, with
    `labels.class_to_crop_disease` mapping that back to (crop_idx,
    disease_idx) -- the same shape PlantVillageDataset itself uses for the
    full multi-crop label space, needed here because crop_idx is no longer
    constant.
    """

    def __init__(
        self, items: list[TomatoAppleRawItem], label_map: TomatoAppleLabelMap, transform: transforms.Compose
    ):
        self.items = items
        self.transform = transform
        num_diseases = len(label_map.disease_classes)
        self.labels = SimpleNamespace(
            crop_classes=list(label_map.crop_classes),
            disease_classes=list(label_map.disease_classes),
            class_to_crop_disease={
                crop_idx * num_diseases + disease_idx: (crop_idx, disease_idx)
                for crop_idx in range(len(label_map.crop_classes))
                for disease_idx in range(num_diseases)
            },
        )
        self.base = SimpleNamespace(
            samples=[
                (item.path, item.canonical_crop_idx * num_diseases + item.canonical_disease_idx) for item in items
            ],
            targets=[item.canonical_crop_idx * num_diseases + item.canonical_disease_idx for item in items],
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, pos: int):
        item = self.items[pos]
        with Image.open(_win_long_path(item.path)) as img:
            image = self.transform(img.convert("RGB"))
        return image, item.canonical_crop_idx, item.canonical_disease_idx


def build_tomato_apple_train_eval_datasets(
    items: list[TomatoAppleRawItem], label_map: TomatoAppleLabelMap, image_size: int
) -> tuple[TomatoAppleMergedDataset, TomatoAppleMergedDataset]:
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_ds = TomatoAppleMergedDataset(items, label_map, train_transform)
    eval_ds = TomatoAppleMergedDataset(items, label_map, eval_transform)
    return train_ds, eval_ds


def load_plantvillage_tomato_apple_items(root, label_map: TomatoAppleLabelMap) -> list[TomatoAppleRawItem]:
    """Single scan of the full PlantVillage ImageFolder, filtered down to
    Apple + Tomato (instead of one scan per crop) -- avoids re-walking the
    same directory tree twice.
    """
    dataset = load_full_dataset(root, image_size=32)  # image_size unused beyond this scan
    crop_idx_by_pv_name = {
        name: dataset.labels.crop_classes.index(name)
        for name in label_map.crop_classes
        if name in dataset.labels.crop_classes
    }
    pv_crop_idx_to_canonical = {v: label_map.crop_name_to_idx[k] for k, v in crop_idx_by_pv_name.items()}

    items: list[TomatoAppleRawItem] = []
    for idx in range(len(dataset)):
        class_idx = dataset.base.targets[idx]
        pv_crop_idx, disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        if pv_crop_idx not in pv_crop_idx_to_canonical:
            continue
        disease_name = dataset.labels.disease_classes[disease_idx]
        if disease_name not in label_map.name_to_disease_idx:
            continue
        path, _ = dataset.base.samples[idx]
        items.append(
            TomatoAppleRawItem(
                source="plantvillage",
                path=path,
                canonical_crop_idx=pv_crop_idx_to_canonical[pv_crop_idx],
                canonical_disease_idx=label_map.name_to_disease_idx[disease_name],
            )
        )
    return items


def _items_from_paths(
    pairs: list[tuple[str, str]], source: str, crop_name: str, label_map: TomatoAppleLabelMap
) -> list[TomatoAppleRawItem]:
    """Skips any (path, canonical_name) pair whose canonical_name isn't in
    label_map (e.g. a disease excluded via tomato_apple_mesh.exclude_diseases)
    instead of raising KeyError -- mirrors apple_mesh_dataset.py's
    `_items_from_paths` for the same case.
    """
    crop_idx = label_map.crop_name_to_idx[crop_name]
    items = []
    for path, canonical_name in pairs:
        if canonical_name not in label_map.name_to_disease_idx:
            continue
        items.append(
            TomatoAppleRawItem(
                source=source,
                path=path,
                canonical_crop_idx=crop_idx,
                canonical_disease_idx=label_map.name_to_disease_idx[canonical_name],
            )
        )
    return items


def compute_merged_image_hashes(items: list[TomatoAppleRawItem], indices: list[int]) -> dict[int, int]:
    hashes: dict[int, int] = {}
    for idx in indices:
        with Image.open(_win_long_path(items[idx].path)) as img:
            hashes[idx] = average_hash(img.convert("RGB"))
    return hashes


def enforce_max_group_size(
    indices: list[int],
    groups: dict[int, int],
    items: list[TomatoAppleRawItem],
    hashes: dict[int, int],
    max_group_size: int,
    max_group_fraction_of_class: float = 0.05,
) -> dict[int, int]:
    """Same over-cap duplicate-group breakup as apple_mesh_dataset.py /
    tomato_mesh_dataset.py's `enforce_max_group_size`, keyed on the
    (crop, disease) pair rather than disease alone -- an Apple-healthy and
    a Tomato-healthy image should never be treated as belonging to the
    same "dominant class" cap even though they share a disease index.
    """
    num_diseases = max((item.canonical_disease_idx for item in items), default=0) + 1
    class_totals: dict[int, int] = {}
    for idx in indices:
        item = items[idx]
        key = item.canonical_crop_idx * num_diseases + item.canonical_disease_idx
        class_totals[key] = class_totals.get(key, 0) + 1

    members_by_group: dict[int, list[int]] = {}
    for idx in indices:
        members_by_group.setdefault(groups[idx], []).append(idx)

    fixed = dict(groups)
    for members in members_by_group.values():
        dominant_item = items[members[0]]
        dominant_key = dominant_item.canonical_crop_idx * num_diseases + dominant_item.canonical_disease_idx
        cap = min(max_group_size, max(1, int(class_totals[dominant_key] * max_group_fraction_of_class)))
        if len(members) > cap:
            by_exact_hash: dict[int, list[int]] = {}
            for member in members:
                by_exact_hash.setdefault(hashes[member], []).append(member)
            for exact_hash_members in by_exact_hash.values():
                representative = exact_hash_members[0]
                for member in exact_hash_members:
                    fixed[member] = representative
    return fixed


def capped_dedup_split(
    indices: list[int],
    hashes: dict[int, int],
    items: list[TomatoAppleRawItem],
    test_fraction: float,
    seed: int,
    threshold: int,
    max_group_size: int,
) -> tuple[list[int], list[int]]:
    """Same greedy-fill algorithm as apple_mesh_dataset.py's
    `capped_dedup_split`."""
    groups = group_duplicates(indices, hashes, threshold)
    groups = enforce_max_group_size(indices, groups, items, hashes, max_group_size)

    group_members: dict[int, list[int]] = {}
    for idx in indices:
        group_members.setdefault(groups[idx], []).append(idx)

    group_sizes = sorted((len(members) for members in group_members.values()), reverse=True)
    logger.info(
        "capped_dedup_split: %d groups after max-group-size cap enforcement (cap=%d); "
        "top 10 largest group sizes: %s",
        len(group_sizes),
        max_group_size,
        group_sizes[:10],
    )

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


def carve_merged_probe_set(
    items: list[TomatoAppleRawItem],
    indices: list[int],
    probe_fraction: float,
    seed: int,
    large_class_threshold: int = 200,
    min_samples_small_class: int = 8,
    max_fraction_small_class: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Same stratified-by-(crop,disease) algorithm as
    apple_mesh_dataset.py's `carve_merged_probe_set`, stratifying on the
    combined crop/disease key so e.g. Apple-healthy and Tomato-healthy are
    each guaranteed their own minimum probe representation instead of
    competing within one shared "healthy" stratum.
    """
    num_diseases = max((item.canonical_disease_idx for item in items), default=0) + 1
    keys = np.array([items[i].canonical_crop_idx * num_diseases + items[i].canonical_disease_idx for i in indices])
    rng = random.Random(seed)
    probe_idx: list[int] = []
    remaining_idx: list[int] = []
    for key in sorted(set(keys.tolist())):
        key_indices = [indices[i] for i in np.flatnonzero(keys == key).tolist()]
        rng.shuffle(key_indices)
        count = len(key_indices)
        if count >= large_class_threshold:
            n_probe_cls = max(1, round(count * probe_fraction))
        else:
            target_n = max(min_samples_small_class, round(count * probe_fraction))
            n_probe_cls = min(target_n, int(count * max_fraction_small_class), count)
        probe_idx.extend(key_indices[:n_probe_cls])
        remaining_idx.extend(key_indices[n_probe_cls:])
    rng.shuffle(probe_idx)
    rng.shuffle(remaining_idx)
    return probe_idx, remaining_idx


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _summarize_partition(items: list[TomatoAppleRawItem], shard: list[int], label_map: TomatoAppleLabelMap) -> dict:
    num_classes = len(label_map.disease_classes)
    num_crops = len(label_map.crop_classes)
    disease_counts = np.zeros(num_classes, dtype=int)
    crop_counts = np.zeros(num_crops, dtype=int)
    # per-crop disease breakdown, e.g. crop_disease_counts["Apple"]["healthy"]
    crop_disease_counts = {crop: {d: 0 for d in label_map.disease_classes} for crop in label_map.crop_classes}
    for idx in shard:
        item = items[idx]
        disease_counts[item.canonical_disease_idx] += 1
        crop_counts[item.canonical_crop_idx] += 1
        crop_name = label_map.crop_classes[item.canonical_crop_idx]
        disease_name = label_map.disease_classes[item.canonical_disease_idx]
        crop_disease_counts[crop_name][disease_name] += 1

    total = int(disease_counts.sum())
    proportions = disease_counts / total if total > 0 else disease_counts.astype(float)
    uniform = np.full(num_classes, 1.0 / num_classes)
    dominant_idx = int(np.argmax(disease_counts))
    nonzero_counts = disease_counts[disease_counts > 0]
    low_rep_threshold = np.percentile(nonzero_counts, 25) if len(nonzero_counts) > 0 else 0

    dominant_crop_idx = int(np.argmax(crop_counts))
    return {
        "num_samples": total,
        "per_class_counts": {name: int(c) for name, c in zip(label_map.disease_classes, disease_counts)},
        "per_crop_counts": {name: int(c) for name, c in zip(label_map.crop_classes, crop_counts)},
        "per_crop_disease_counts": crop_disease_counts,
        "dominant_class": label_map.disease_classes[dominant_idx],
        "dominant_class_fraction": float(proportions[dominant_idx]) if total > 0 else 0.0,
        "dominant_crop": label_map.crop_classes[dominant_crop_idx],
        "dominant_crop_fraction": float(crop_counts[dominant_crop_idx] / total) if total > 0 else 0.0,
        "js_divergence_from_uniform": _jensen_shannon_divergence(proportions, uniform),
        "num_classes_present": int(np.count_nonzero(disease_counts)),
        "low_representation_classes": [
            label_map.disease_classes[i]
            for i in range(num_classes)
            if disease_counts[i] <= low_rep_threshold
        ],
    }


@dataclass
class TomatoAppleMeshData:
    train_base: TomatoAppleMergedDataset
    eval_base: TomatoAppleMergedDataset
    label_map: TomatoAppleLabelMap
    image_size: int
    probe_idx: list[int]
    test_idx: list[int]
    per_node: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    partition_diagnostics: dict[str, dict] = field(default_factory=dict)


def prepare_tomato_apple_mesh_data(cfg) -> TomatoAppleMeshData:
    from src.data.plantdoc import load_plantdoc_apple_paths, load_plantdoc_tomato_paths
    from src.data.plantwild import (
        load_plantwild_v1_apple_paths,
        load_plantwild_v1_tomato_paths,
        load_plantwild_v2_apple_paths,
        load_plantwild_v2_tomato_paths,
    )

    image_size = cfg.get("data.image_size", 160)
    pv_root = cfg.get("data.root", "data/PlantVillage")
    plantdoc_root = cfg.get("tomato_apple_mesh.plantdoc_root", "data/PlantDoc")
    plantwild_v1_root = cfg.get(
        "tomato_apple_mesh.plantwild_v1_root", "data/PlantWild/plantwild/plantwild/images"
    )
    plantwild_v2_root = cfg.get("tomato_apple_mesh.plantwild_v2_root", "data/PlantWild/plantwild_v2/plantwild_v2")
    seed = cfg.get("data.seed", 42)
    num_nodes = cfg.get("tomato_apple_mesh.num_nodes", 3)
    dirichlet_alpha = cfg.get("tomato_apple_mesh.dirichlet_alpha", 0.3)
    test_fraction = cfg.get("tomato_apple_mesh.test_fraction", 0.20)
    dedup_threshold = cfg.get("tomato_apple_mesh.dedup_threshold", 5)
    dedup_max_group_size = cfg.get("tomato_apple_mesh.dedup_max_group_size", 25)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    exclude_diseases = set(cfg.get("tomato_apple_mesh.exclude_diseases", []) or [])

    disease_names = [name for name in _combined_disease_order() if name not in exclude_diseases]
    label_map = build_tomato_apple_label_map(disease_names)

    pv_items = load_plantvillage_tomato_apple_items(pv_root, label_map)
    pd_items = _items_from_paths(
        load_plantdoc_apple_paths(plantdoc_root), "plantdoc", "Apple", label_map
    ) + _items_from_paths(load_plantdoc_tomato_paths(plantdoc_root), "plantdoc", "Tomato", label_map)
    pw1_items = _items_from_paths(
        load_plantwild_v1_apple_paths(plantwild_v1_root), "plantwild_v1", "Apple", label_map
    ) + _items_from_paths(
        load_plantwild_v1_tomato_paths(plantwild_v1_root), "plantwild_v1", "Tomato", label_map
    )
    pw2_items = _items_from_paths(
        load_plantwild_v2_apple_paths(plantwild_v2_root), "plantwild_v2", "Apple", label_map
    ) + _items_from_paths(
        load_plantwild_v2_tomato_paths(plantwild_v2_root), "plantwild_v2", "Tomato", label_map
    )
    items = pv_items + pd_items + pw1_items + pw2_items
    if not items:
        raise FileNotFoundError(
            "No Apple/Tomato images found across PlantVillage/PlantDoc/PlantWild -- check "
            "data.root/tomato_apple_mesh.plantdoc_root/tomato_apple_mesh.plantwild_v1_root/"
            "tomato_apple_mesh.plantwild_v2_root."
        )

    indices = list(range(len(items)))
    hashes = compute_merged_image_hashes(items, indices)
    train_idx_pool, test_idx = capped_dedup_split(
        indices, hashes, items, test_fraction, seed, threshold=dedup_threshold, max_group_size=dedup_max_group_size
    )

    probe_idx, remaining_idx = carve_merged_probe_set(
        items,
        train_idx_pool,
        probe_fraction,
        seed,
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )

    num_diseases = len(label_map.disease_classes)
    combined_targets = np.array(
        [items[i].canonical_crop_idx * num_diseases + items[i].canonical_disease_idx for i in remaining_idx]
    )
    rng = np.random.RandomState(seed)
    shards = _dirichlet_partition(remaining_idx, combined_targets, num_nodes, dirichlet_alpha, rng)

    per_node: dict[str, dict[str, list[int]]] = {}
    partition_diagnostics: dict[str, dict] = {}
    for node_idx, shard in enumerate(shards):
        node_id = f"node_{node_idx}"
        per_node[node_id] = {"train_idx": shard}
        partition_diagnostics[node_id] = _summarize_partition(items, shard, label_map)

    train_base, eval_base = build_tomato_apple_train_eval_datasets(items, label_map, image_size)
    return TomatoAppleMeshData(
        train_base=train_base,
        eval_base=eval_base,
        label_map=label_map,
        image_size=image_size,
        probe_idx=probe_idx,
        test_idx=test_idx,
        per_node=per_node,
        partition_diagnostics=partition_diagnostics,
    )
