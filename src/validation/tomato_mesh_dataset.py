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

from dataclasses import dataclass
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, load_full_dataset

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
