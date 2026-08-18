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
