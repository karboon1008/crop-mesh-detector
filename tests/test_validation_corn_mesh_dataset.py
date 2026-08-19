from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.corn_mesh_dataset import (
    CORN_DISEASE_ORDER,
    build_corn_label_map,
    get_corn_disease_indices,
    get_corn_indices,
)


def test_build_corn_label_map_has_one_crop_and_four_diseases():
    label_map = build_corn_label_map()

    assert label_map.crop_classes == ["Corn"]
    assert label_map.disease_classes == CORN_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == 0
    assert label_map.name_to_disease_idx["Common_rust"] == 1


def test_get_corn_indices_excludes_other_crops(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    corn_indices = get_corn_indices(dataset)

    assert len(corn_indices) == 12 + 6 + 6 + 6  # healthy + 3 diseases, no Tomato
    for idx in corn_indices:
        crop_idx, _ = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        assert dataset.labels.crop_classes[crop_idx] == "Corn"


def test_get_corn_disease_indices_filters_to_named_disease(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)

    rust_indices = get_corn_disease_indices(dataset, corn_indices, "Common_rust")
    healthy_indices = get_corn_disease_indices(dataset, corn_indices, "healthy")

    assert len(rust_indices) == 6
    assert len(healthy_indices) == 12
    assert set(rust_indices).isdisjoint(set(healthy_indices))
