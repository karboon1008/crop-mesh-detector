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
from dataclasses import dataclass
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, load_full_dataset
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
