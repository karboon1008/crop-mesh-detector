"""Tomato multi-source Dirichlet mesh dataset preparation -- pools
PlantVillage + PlantDoc + PlantWild (v1+v2) into one canonical label
space, carves a GLOBAL dedup-aware test split, then Dirichlet-partitions
the remaining training pool across 3 nodes. See
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.

Unlike corn_mesh_dataset.py, no CornDiseaseView-style label remap is
needed here: the merged item list is built directly in the canonical
compact label space (crop always index 0, disease already 0-9), so
TomatoMergedDataset's `.labels.class_to_crop_disease` is simply
{i: (0, i) for i in range(len(disease_classes))} -- there is no
global-vs-compact index mismatch to bridge, because there is no single
underlying ImageFolder spanning other crops the way PlantVillage's full
dataset does for corn_mesh_dataset.py.
"""

from __future__ import annotations

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


@dataclass
class TomatoLabelMap:
    crop_classes: list[str]
    disease_classes: list[str]
    name_to_disease_idx: dict[str, int]


def build_tomato_label_map(disease_names: list[str] = TOMATO_DISEASE_ORDER) -> TomatoLabelMap:
    return TomatoLabelMap(
        crop_classes=["Tomato"],
        disease_classes=list(disease_names),
        name_to_disease_idx={name: i for i, name in enumerate(disease_names)},
    )


@dataclass
class TomatoRawItem:
    source: str
    path: str
    canonical_disease_idx: int


class TomatoMergedDataset(Dataset):
    """A flat, canonical-label-space dataset over pooled multi-source
    Tomato items. Exposes the same `.base.samples`/`.base.targets`/
    `.labels.class_to_crop_disease`/`.labels.crop_classes`/
    `.labels.disease_classes` surface as PlantVillageDataset so
    train_mobilenet.run_training, export_onnx.export_checkpoint, and
    evaluate_onnx.run_evaluation all work against it unmodified.
    """

    def __init__(self, items: list[TomatoRawItem], label_map: TomatoLabelMap, transform: transforms.Compose):
        self.items = items
        self.transform = transform
        self.labels = SimpleNamespace(
            crop_classes=list(label_map.crop_classes),
            disease_classes=list(label_map.disease_classes),
            class_to_crop_disease={i: (0, i) for i in range(len(label_map.disease_classes))},
        )
        self.base = SimpleNamespace(
            samples=[(item.path, item.canonical_disease_idx) for item in items],
            targets=[item.canonical_disease_idx for item in items],
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, pos: int):
        item = self.items[pos]
        with Image.open(item.path) as img:
            image = self.transform(img.convert("RGB"))
        return image, 0, item.canonical_disease_idx


def build_tomato_train_eval_datasets(
    items: list[TomatoRawItem], label_map: TomatoLabelMap, image_size: int
) -> tuple[TomatoMergedDataset, TomatoMergedDataset]:
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
    train_ds = TomatoMergedDataset(items, label_map, train_transform)
    eval_ds = TomatoMergedDataset(items, label_map, eval_transform)
    return train_ds, eval_ds


def load_plantvillage_tomato_items(root, label_map: TomatoLabelMap) -> list[TomatoRawItem]:
    dataset = load_full_dataset(root, image_size=32)  # image_size unused beyond this scan
    tomato_crop_idx = dataset.labels.crop_classes.index("Tomato")
    items: list[TomatoRawItem] = []
    for idx in range(len(dataset)):
        class_idx = dataset.base.targets[idx]
        crop_idx, disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        if crop_idx != tomato_crop_idx:
            continue
        disease_name = dataset.labels.disease_classes[disease_idx]
        if disease_name not in label_map.name_to_disease_idx:
            continue
        path, _ = dataset.base.samples[idx]
        items.append(
            TomatoRawItem(
                source="plantvillage",
                path=path,
                canonical_disease_idx=label_map.name_to_disease_idx[disease_name],
            )
        )
    return items


def _items_from_paths(
    pairs: list[tuple[str, str]], source: str, label_map: TomatoLabelMap
) -> list[TomatoRawItem]:
    return [
        TomatoRawItem(source=source, path=path, canonical_disease_idx=label_map.name_to_disease_idx[canonical_name])
        for path, canonical_name in pairs
    ]


def compute_merged_image_hashes(items: list[TomatoRawItem], indices: list[int]) -> dict[int, int]:
    hashes: dict[int, int] = {}
    for idx in indices:
        with Image.open(items[idx].path) as img:
            hashes[idx] = average_hash(img.convert("RGB"))
    return hashes


def enforce_max_group_size(
    indices: list[int],
    groups: dict[int, int],
    items: list[TomatoRawItem],
    max_group_size: int,
    max_group_fraction_of_class: float = 0.05,
) -> dict[int, int]:
    """Any duplicate-group larger than min(max_group_size, fraction * that
    group's dominant class's pooled count) is broken into singleton
    groups -- this is the fix for the exact failure mode that let 97% of
    node_1's Soybean class collapse into one "duplicate" group
    (docs/node1_validation_tuning_and_results.md).
    """
    class_totals: dict[int, int] = {}
    for idx in indices:
        cls = items[idx].canonical_disease_idx
        class_totals[cls] = class_totals.get(cls, 0) + 1

    members_by_group: dict[int, list[int]] = {}
    for idx in indices:
        members_by_group.setdefault(groups[idx], []).append(idx)

    fixed = dict(groups)
    for members in members_by_group.values():
        dominant_cls = items[members[0]].canonical_disease_idx
        cap = min(max_group_size, max(1, int(class_totals[dominant_cls] * max_group_fraction_of_class)))
        if len(members) > cap:
            for member in members:
                fixed[member] = member
    return fixed


def capped_dedup_split(
    indices: list[int],
    hashes: dict[int, int],
    items: list[TomatoRawItem],
    test_fraction: float,
    seed: int,
    threshold: int,
    max_group_size: int,
) -> tuple[list[int], list[int]]:
    """Same greedy-fill algorithm as node1_dataset.dedup_aware_split, plus
    the max-group-size cap above. Reimplemented (not calling
    dedup_aware_split directly) because that function recomputes grouping
    internally and has no cap parameter to inject.
    """
    groups = group_duplicates(indices, hashes, threshold)
    groups = enforce_max_group_size(indices, groups, items, max_group_size)

    group_members: dict[int, list[int]] = {}
    for idx in indices:
        group_members.setdefault(groups[idx], []).append(idx)

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
    items: list[TomatoRawItem],
    indices: list[int],
    probe_fraction: float,
    seed: int,
    large_class_threshold: int = 200,
    min_samples_small_class: int = 8,
    max_fraction_small_class: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Same stratified-by-class algorithm as plantvillage.carve_public_probe_set,
    adapted to operate on a subset of positions (indices) into the merged
    items list rather than an entire PlantVillageDataset -- needed because
    the probe set here is carved from just the post-test-split training
    pool, not the whole dataset.
    """
    targets = np.array([items[i].canonical_disease_idx for i in indices])
    rng = random.Random(seed)
    probe_idx: list[int] = []
    remaining_idx: list[int] = []
    for cls in sorted(set(targets.tolist())):
        cls_indices = [indices[i] for i in np.flatnonzero(targets == cls).tolist()]
        rng.shuffle(cls_indices)
        count = len(cls_indices)
        if count >= large_class_threshold:
            n_probe_cls = max(1, round(count * probe_fraction))
        else:
            target_n = max(min_samples_small_class, round(count * probe_fraction))
            n_probe_cls = min(target_n, int(count * max_fraction_small_class), count)
        probe_idx.extend(cls_indices[:n_probe_cls])
        remaining_idx.extend(cls_indices[n_probe_cls:])
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


def _summarize_partition(items: list[TomatoRawItem], shard: list[int], label_map: TomatoLabelMap) -> dict:
    num_classes = len(label_map.disease_classes)
    counts = np.zeros(num_classes, dtype=int)
    for idx in shard:
        counts[items[idx].canonical_disease_idx] += 1
    total = int(counts.sum())
    proportions = counts / total if total > 0 else counts.astype(float)
    uniform = np.full(num_classes, 1.0 / num_classes)
    dominant_idx = int(np.argmax(counts))
    nonzero_counts = counts[counts > 0]
    low_rep_threshold = np.percentile(nonzero_counts, 25) if len(nonzero_counts) > 0 else 0
    return {
        "num_samples": total,
        "per_class_counts": {name: int(c) for name, c in zip(label_map.disease_classes, counts)},
        "dominant_class": label_map.disease_classes[dominant_idx],
        "dominant_class_fraction": float(proportions[dominant_idx]) if total > 0 else 0.0,
        "js_divergence_from_uniform": _jensen_shannon_divergence(proportions, uniform),
        "num_classes_present": int(np.count_nonzero(counts)),
        "low_representation_classes": [
            label_map.disease_classes[i]
            for i in range(num_classes)
            if 0 < counts[i] <= low_rep_threshold
        ],
    }


@dataclass
class TomatoMeshData:
    train_base: TomatoMergedDataset
    eval_base: TomatoMergedDataset
    label_map: TomatoLabelMap
    image_size: int
    probe_idx: list[int]
    test_idx: list[int]
    per_node: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    partition_diagnostics: dict[str, dict] = field(default_factory=dict)


def prepare_tomato_mesh_data(cfg) -> TomatoMeshData:
    from src.data.plantdoc import load_plantdoc_tomato_paths
    from src.data.plantwild import load_plantwild_v1_tomato_paths, load_plantwild_v2_tomato_paths

    image_size = cfg.get("data.image_size", 160)
    pv_root = cfg.get("data.root", "data/PlantVillage")
    plantdoc_root = cfg.get("tomato_mesh.plantdoc_root", "data/PlantDoc")
    plantwild_v1_root = cfg.get("tomato_mesh.plantwild_v1_root", "data/PlantWild/plantwild/plantwild/images")
    plantwild_v2_root = cfg.get("tomato_mesh.plantwild_v2_root", "data/PlantWild/plantwild_v2/plantwild_v2")
    seed = cfg.get("data.seed", 42)
    num_nodes = cfg.get("tomato_mesh.num_nodes", 3)
    dirichlet_alpha = cfg.get("tomato_mesh.dirichlet_alpha", 0.3)
    test_fraction = cfg.get("tomato_mesh.test_fraction", 0.20)
    dedup_threshold = cfg.get("tomato_mesh.dedup_threshold", 5)
    dedup_max_group_size = cfg.get("tomato_mesh.dedup_max_group_size", 25)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)

    label_map = build_tomato_label_map()

    pv_items = load_plantvillage_tomato_items(pv_root, label_map)
    pd_items = _items_from_paths(load_plantdoc_tomato_paths(plantdoc_root), "plantdoc", label_map)
    pw1_items = _items_from_paths(
        load_plantwild_v1_tomato_paths(plantwild_v1_root), "plantwild_v1", label_map
    )
    pw2_items = _items_from_paths(
        load_plantwild_v2_tomato_paths(plantwild_v2_root), "plantwild_v2", label_map
    )
    items = pv_items + pd_items + pw1_items + pw2_items
    if not items:
        raise FileNotFoundError(
            "No Tomato images found across PlantVillage/PlantDoc/PlantWild -- check "
            "data.root/tomato_mesh.plantdoc_root/tomato_mesh.plantwild_v1_root/"
            "tomato_mesh.plantwild_v2_root."
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

    targets = np.array([items[i].canonical_disease_idx for i in remaining_idx])
    rng = np.random.RandomState(seed)
    shards = _dirichlet_partition(remaining_idx, targets, num_nodes, dirichlet_alpha, rng)

    per_node: dict[str, dict[str, list[int]]] = {}
    partition_diagnostics: dict[str, dict] = {}
    for node_idx, shard in enumerate(shards):
        node_id = f"node_{node_idx}"
        per_node[node_id] = {"train_idx": shard}
        partition_diagnostics[node_id] = _summarize_partition(items, shard, label_map)

    train_base, eval_base = build_tomato_train_eval_datasets(items, label_map, image_size)
    return TomatoMeshData(
        train_base=train_base,
        eval_base=eval_base,
        label_map=label_map,
        image_size=image_size,
        probe_idx=probe_idx,
        test_idx=test_idx,
        per_node=per_node,
        partition_diagnostics=partition_diagnostics,
    )
