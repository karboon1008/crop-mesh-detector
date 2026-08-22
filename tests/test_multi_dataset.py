"""End-to-end smoke test for the "one_node_one_dataset" strategy on tiny
synthetic PlantVillage/PlantDoc/PlantWild trees: one node per dataset, a
cross-dataset stratified probe/global-test set, and augmented train vs.
plain-eval validation splits.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.data.multi_dataset import NODE_ORDER, build_one_node_one_dataset_loaders

PV_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]
PLANTDOC_CLASSES = ["Tomato leaf bacterial spot", "Potato leaf early blight", "Tomato leaf"]
PLANTWILD_CLASSES = ["tomato+bacterial+leaf+spot", "potato+early+blight", "tomato+leaf"]
IMAGES_PER_CLASS = 10
EXPECTED_LEN = {
    "plantvillage": len(PV_CLASSES) * IMAGES_PER_CLASS,
    "plantdoc": len(PLANTDOC_CLASSES) * IMAGES_PER_CLASS,
    "plantwild": len(PLANTWILD_CLASSES) * IMAGES_PER_CLASS,
}


def _make_tree(root, class_names, seed):
    rng = np.random.RandomState(seed)
    for cls in class_names:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(IMAGES_PER_CLASS):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


@pytest.fixture
def cfg(tmp_path):
    pv_root = tmp_path / "PlantVillage"
    plantdoc_root = tmp_path / "PlantDoc"
    plantwild_root = tmp_path / "PlantWild"
    _make_tree(pv_root, PV_CLASSES, seed=0)
    _make_tree(plantdoc_root, PLANTDOC_CLASSES, seed=1)
    _make_tree(plantwild_root, PLANTWILD_CLASSES, seed=2)

    return Config({
        "data": {
            "root": str(pv_root),
            "plantdoc_root": str(plantdoc_root),
            "plantwild_root": str(plantwild_root),
            "image_size": 32,
            "num_nodes": 3,
            "seed": 0,
            "probe_set_fraction": 0.1,
            "probe_set_large_class_threshold": 1000,
            "probe_set_min_samples_small_class": 1,
            "probe_set_max_fraction_small_class": 0.5,
            "global_test_fraction": 0.1,
            "test_fraction": 0.2,
        },
        "training": {
            "batch_size": 4,
            "crop_class_balanced": True,
            "disease_class_balanced": True,
        },
    })


def test_returns_one_node_per_dataset(cfg):
    _, _, node_loaders, crop_classes, disease_classes, crop_w, disease_w = build_one_node_one_dataset_loaders(cfg)
    assert len(node_loaders) == len(NODE_ORDER) == 3
    assert len(crop_w) == len(disease_w) == 3
    assert set(crop_classes) >= {"Tomato", "Potato"}


def test_probe_and_global_test_span_all_three_datasets(cfg):
    probe_loader, global_test_loader, _, _, _, _, _ = build_one_node_one_dataset_loaders(cfg)
    for loader in (probe_loader, global_test_loader):
        parts = loader.dataset.datasets
        assert len(parts) == 3
        assert all(len(p) > 0 for p in parts)


def test_no_leakage_between_splits(cfg):
    probe_loader, global_test_loader, node_loaders, _, _, _, _ = build_one_node_one_dataset_loaders(cfg)
    for i, name in enumerate(NODE_ORDER):
        train_idx = set(node_loaders[i][0].dataset.indices)
        val_idx = set(node_loaders[i][1].dataset.indices)
        probe_idx = set(probe_loader.dataset.datasets[i].indices)
        gtest_idx = set(global_test_loader.dataset.datasets[i].indices)

        splits = {"train": train_idx, "val": val_idx, "probe": probe_idx, "global_test": gtest_idx}
        for a_name, a in splits.items():
            for b_name, b in splits.items():
                if a_name != b_name:
                    assert a.isdisjoint(b), f"{name}: {a_name} and {b_name} overlap"

        assert len(train_idx | val_idx | probe_idx | gtest_idx) == EXPECTED_LEN[name]


def test_train_split_is_augmented_validation_is_not(cfg):
    _, _, node_loaders, _, _, _, _ = build_one_node_one_dataset_loaders(cfg)
    for train_loader, val_loader in node_loaders:
        train_transform = train_loader.dataset.transform
        val_transform = val_loader.dataset.transform
        assert any(isinstance(t, transforms.RandomResizedCrop) for t in train_transform.transforms)
        assert not any(isinstance(t, transforms.RandomResizedCrop) for t in val_transform.transforms)


def test_num_nodes_must_match_dataset_count(cfg):
    cfg._data["data"]["num_nodes"] = 4
    with pytest.raises(ValueError, match="fixes node count"):
        build_one_node_one_dataset_loaders(cfg)
