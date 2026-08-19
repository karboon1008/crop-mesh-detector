"""Scopes the full PlantVillage dataset down to Corn only, and remaps
Corn's 4 classes (healthy + 3 diseases) into a compact, Corn-only label
space -- see docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md
for why the head is scoped this tightly instead of reusing the full
14-crop/~22-disease global label space.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.plantvillage import PlantVillageDataset

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
