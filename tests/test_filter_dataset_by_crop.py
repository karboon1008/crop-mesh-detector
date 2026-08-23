# tests/test_filter_dataset_by_crop.py
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.plantvillage import PlantVillageDataset, filter_dataset_by_crop

CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


def _build_dataset(tmp_path, image_size=32):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            arr = rng.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return PlantVillageDataset(root, image_size=image_size)


def test_filter_keeps_only_included_crops(tmp_path):
    dataset = _build_dataset(tmp_path)
    filter_dataset_by_crop(dataset, included_crops=["Apple", "Tomato"])

    assert sorted(dataset.labels.crop_classes) == ["Apple", "Tomato"]
    class_names = {name for name in dataset.base.classes}
    assert class_names == {
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
        "Apple___healthy", "Tomato___Bacterial_spot", "Tomato___healthy",
    }
    # Potato images must be gone entirely, not just relabeled.
    assert len(dataset) == 4 * 6


def test_filter_drops_named_disease_for_one_crop(tmp_path):
    dataset = _build_dataset(tmp_path)
    filter_dataset_by_crop(
        dataset, included_crops=["Apple", "Tomato"], excluded_diseases={"Apple": ["Black_rot"]}
    )

    assert "Apple___Black_rot" not in dataset.base.classes
    assert "Apple___Apple_scab" in dataset.base.classes
    assert len(dataset) == 4 * 5  # 3 Apple classes + 2 Tomato classes


def test_filter_remaps_class_indices_so_samples_and_labels_stay_aligned(tmp_path):
    dataset = _build_dataset(tmp_path)
    filter_dataset_by_crop(
        dataset, included_crops=["Apple", "Tomato"], excluded_diseases={"Apple": ["Black_rot"]}
    )

    for idx in range(len(dataset)):
        _, class_idx = dataset.base.samples[idx]
        class_name = dataset.base.classes[class_idx]
        crop_idx, disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        assert dataset.labels.crop_classes[crop_idx] == class_name.split("___")[0]
        # Iterating the dataset must also produce labels consistent with the class name.
        _, returned_crop_idx, returned_disease_idx = dataset[idx]
        assert returned_crop_idx == crop_idx
        assert returned_disease_idx == disease_idx


def test_filter_with_no_included_crops_only_applies_excluded_diseases(tmp_path):
    dataset = _build_dataset(tmp_path)
    filter_dataset_by_crop(dataset, included_crops=None, excluded_diseases={"Apple": ["Black_rot"]})

    assert sorted(dataset.labels.crop_classes) == ["Apple", "Potato", "Tomato"]
    assert "Apple___Black_rot" not in dataset.base.classes
    assert len(dataset) == 4 * 7  # every original class except Apple___Black_rot


def test_filter_raises_if_everything_is_excluded(tmp_path):
    dataset = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="excluded every class"):
        filter_dataset_by_crop(dataset, included_crops=["Grape"])
