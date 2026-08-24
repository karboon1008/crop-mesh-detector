from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.plantvillage import (
    GlobalLabelMap,
    PlantVillageDataset,
    build_global_label_map,
    load_global_label_map,
    save_global_label_map,
)

FULL_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


def _make_class_folders(root, class_names, n_per_class=3):
    rng = np.random.RandomState(0)
    for cls in class_names:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


def test_global_label_map_preserves_indices_for_subset_of_classes(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, FULL_CLASSES)
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    node_root = tmp_path / "node_0"
    subset = ["Tomato___Bacterial_spot", "Potato___healthy"]
    _make_class_folders(node_root, subset)
    node_dataset = PlantVillageDataset(node_root, image_size=16, global_label_map=global_map)

    for local_idx, name in enumerate(node_dataset.base.classes):
        full_idx = full_dataset.base.classes.index(name)
        expected = full_dataset.labels.class_to_crop_disease[full_idx]
        assert node_dataset.labels.class_to_crop_disease[local_idx] == expected

    assert node_dataset.labels.crop_classes == full_dataset.labels.crop_classes
    assert node_dataset.labels.disease_classes == full_dataset.labels.disease_classes


def test_global_label_map_raises_on_class_missing_from_map(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, ["Tomato___Bacterial_spot", "Potato___healthy"])
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    node_root = tmp_path / "node_stale"
    _make_class_folders(node_root, ["Tomato___Bacterial_spot", "Corn___healthy"])  # Corn unknown to the map

    with pytest.raises(ValueError, match="Corn___healthy"):
        PlantVillageDataset(node_root, image_size=16, global_label_map=global_map)


def test_global_label_map_json_roundtrip(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, FULL_CLASSES)
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    path = tmp_path / "classes.json"
    save_global_label_map(global_map, path)
    loaded = load_global_label_map(path)

    assert loaded.crop_classes == global_map.crop_classes
    assert loaded.disease_classes == global_map.disease_classes
    assert loaded.name_to_crop_disease == global_map.name_to_crop_disease
