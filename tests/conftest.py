"""Shared pytest fixtures and helpers for the mesh/scenario test suite."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data.plantvillage import PlantVillageDataset, make_subset, train_test_split_indices
from src.federated.node import Node
from src.models.factory import build_model

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


@pytest.fixture
def synthetic_dataset(tmp_path):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(8):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return PlantVillageDataset(root, image_size=32)


def build_nodes(dataset, shards, num_crop, num_disease, arch="mobilenet_v3_small"):
    nodes = []
    for i, shard in enumerate(shards):
        train_idx, test_idx = train_test_split_indices(dataset, shard, test_fraction=0.3, seed=1)
        train_loader = DataLoader(make_subset(dataset, train_idx), batch_size=4, shuffle=True, drop_last=True)
        test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=4, shuffle=False)
        model = build_model(arch, num_crop, num_disease, pretrained=False)
        nodes.append(Node(
            f"node_{i}", model, train_loader, test_loader, device="cpu",
            crop_classes=dataset.labels.crop_classes, disease_classes=dataset.labels.disease_classes,
        ))
    return nodes


@pytest.fixture
def energy_tracker(tmp_path):
    """CodeCarbon disabled, so the tracker uses its disclosed wall-clock ×
    fixed-power fallback — deterministic enough for tests and identical to
    the code path a machine without RAPL access actually takes.
    """
    from src.energy.tracker import ComputeEnergyTracker

    return ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)


@pytest.fixture
def wifi_comm_estimator():
    from src.energy.tracker import CommunicationCostEstimator

    return CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )


@pytest.fixture
def node1_scoped_config(tmp_path):
    """A tiny synthetic PlantVillage-shaped dataset (Tomato + Potato, 8
    images each) with a 2-node manual split, used across the
    src/validation/ test suite. "node_1" here means Potato, mirroring how
    config.yaml's real manual_node_crops assigns node_1 = Corn/Potato/
    Soybean/Strawberry/Squash — the mechanism under test doesn't care
    which literal crop names are used, only that partitioning +
    dedup-aware splitting work correctly.
    """
    from src.config import Config

    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    classes = [
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
    ]
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(8):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")

    cfg = Config(
        {
            "data": {
                "root": str(root),
                "image_size": 32,
                "num_nodes": 2,
                "non_iid_strategy": "manual",
                "manual_node_crops": {"node_0": ["Tomato"], "node_1": ["Potato"]},
                "test_fraction": 0.25,
                "seed": 42,
            }
        }
    )
    return cfg, root


@pytest.fixture
def corn_scoped_config(tmp_path):
    """A tiny synthetic Corn-shaped dataset (4 classes) plus one
    unrelated Tomato class (to prove Corn-filtering excludes other
    crops), used across the corn_mesh test suite.
    """
    from src.config import Config

    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    class_counts = {
        "Corn___healthy": 12,
        "Corn___Common_rust": 6,
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": 6,
        "Corn___Northern_Leaf_Blight": 6,
        "Tomato___healthy": 4,
    }
    for cls, count in class_counts.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(count):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")

    cfg = Config(
        {
            "data": {
                "root": str(root),
                "image_size": 32,
                "seed": 42,
                "test_fraction": 0.25,
                "probe_set_fraction": 0.1,
                "probe_set_large_class_threshold": 200,
                "probe_set_min_samples_small_class": 1,
                "probe_set_max_fraction_small_class": 0.5,
            },
            "corn_mesh": {
                "crop": "Corn",
                "node_diseases": {
                    "node_0": "Common_rust",
                    "node_1": "Cercospora_leaf_spot Gray_leaf_spot",
                    "node_2": "Northern_Leaf_Blight",
                },
                "healthy_dedup_threshold": 5,
                "rounds": 2,
            },
            "training": {
                "distill_epochs_per_round": 1,
                "distill_lr": 1e-3,
                "proto_weight": 0.5,
                "kd_weight": 0.5,
                "kd_temperature": 2.0,
            },
            "federated": {
                "aggregation": "trimmed_mean",
                "trim_fraction": 0.0,
                "krum_neighbors": 1,
            },
        }
    )
    return cfg, root


@pytest.fixture
def tomato_scoped_config(tmp_path):
    """A tiny synthetic 4-source Tomato dataset (PlantVillage + PlantDoc +
    PlantWild v1/v2) covering 3 canonical disease classes plus one
    off-crop PlantVillage control class, used across the tomato_mesh test
    suite.
    """
    from src.config import Config

    rng = np.random.RandomState(0)

    def _make_images(dir_path, count):
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")

    pv_root = tmp_path / "PlantVillage"
    for cls, count in {
        "Tomato___Bacterial_spot": 10,
        "Tomato___healthy": 10,
        "Tomato___Early_blight": 8,
        "Potato___healthy": 4,
    }.items():
        _make_images(pv_root / cls, count)

    plantdoc_root = tmp_path / "PlantDoc"
    _make_images(plantdoc_root / "train" / "Tomato leaf bacterial spot", 4)
    _make_images(plantdoc_root / "test" / "Tomato leaf bacterial spot", 2)
    _make_images(plantdoc_root / "train" / "Tomato leaf", 3)

    plantwild_v1_root = tmp_path / "PlantWild" / "plantwild" / "plantwild" / "images"
    _make_images(plantwild_v1_root / "tomato bacterial leaf spot", 4)
    _make_images(plantwild_v1_root / "tomato leaf", 3)

    plantwild_v2_root = tmp_path / "PlantWild" / "plantwild_v2" / "plantwild_v2"
    _make_images(plantwild_v2_root / "tomato bacterial leaf spot", 3)

    cfg = Config(
        {
            "data": {
                "root": str(pv_root),
                "image_size": 32,
                "seed": 42,
                "probe_set_fraction": 0.1,
                "probe_set_large_class_threshold": 200,
                "probe_set_min_samples_small_class": 1,
                "probe_set_max_fraction_small_class": 0.5,
            },
            "tomato_mesh": {
                "crop": "Tomato",
                "plantdoc_root": str(plantdoc_root),
                "plantwild_v1_root": str(plantwild_v1_root),
                "plantwild_v2_root": str(plantwild_v2_root),
                "num_nodes": 3,
                "dirichlet_alpha": 0.3,
                "test_fraction": 0.2,
                "dedup_threshold": 5,
                "dedup_max_group_size": 25,
                "rounds": 5,
            },
            "training": {
                "distill_epochs_per_round": 1,
                "distill_lr": 1e-3,
                "proto_weight": 0.5,
                "kd_weight": 0.5,
                "kd_temperature": 2.0,
            },
            "federated": {
                "aggregation": "trimmed_mean",
                "trim_fraction": 0.0,
                "krum_neighbors": 1,
            },
        }
    )
    return cfg
