from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.split_node_data import split
from src.config import Config

SOURCE_CLASSES = {
    "Tomato___Bacterial_spot": 6,
    "Tomato___healthy": 6,
    "Potato___Early_blight": 6,
    "Potato___healthy": 6,
}


def _build_source_tree(root: Path) -> int:
    rng = np.random.RandomState(0)
    total = 0
    for cls, n in SOURCE_CLASSES.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n):
            arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
            total += 1
    return total


def test_split_partitions_every_source_file_exactly_once(tmp_path):
    source_root = tmp_path / "PlantVillage"
    total_files = _build_source_tree(source_root)

    cfg = Config(
        {
            "data": {
                "root": str(source_root),
                "image_size": 16,
                "num_nodes": 2,
                "probe_set_fraction": 0.1,
                "non_iid_strategy": "manual",
                "manual_node_crops": {"node_0": ["Tomato"], "node_1": ["Potato"]},
                "seed": 42,
            }
        }
    )
    output_root = tmp_path / "docker_mesh"
    split(cfg, output_root)

    assert (output_root / "classes.json").exists()

    seen_paths = set()
    for sub in ["probe", "node_0", "node_1"]:
        for f in (output_root / sub).rglob("*.jpg"):
            key = (sub, f.parent.name, f.name)
            assert key not in seen_paths, f"duplicate copy: {key}"
            seen_paths.add(key)
    assert len(seen_paths) == total_files

    # manual strategy: node_0's files are only ever Tomato classes, node_1's only Potato
    node_0_classes = {f.parent.name for f in (output_root / "node_0").rglob("*.jpg")}
    node_1_classes = {f.parent.name for f in (output_root / "node_1").rglob("*.jpg")}
    assert all(c.startswith("Tomato") for c in node_0_classes)
    assert all(c.startswith("Potato") for c in node_1_classes)


APPLE_TOMATO_SOURCE_CLASSES = {
    "Apple___Apple_scab": 6,
    "Apple___Black_rot": 6,
    "Apple___healthy": 6,
    "Tomato___Bacterial_spot": 6,
    "Tomato___healthy": 6,
    "Potato___healthy": 6,
}


def test_split_applies_docker_mesh_crop_and_disease_filters(tmp_path):
    source_root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls, n in APPLE_TOMATO_SOURCE_CLASSES.items():
        cls_dir = source_root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n):
            arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")

    cfg = Config(
        {
            "data": {
                "root": str(source_root),
                "image_size": 16,
                "num_nodes": 3,
                "probe_set_fraction": 0.1,
                "non_iid_strategy": "manual",  # deliberately left as "manual" -- docker_mesh below must override it
                "manual_node_crops": {
                    "node_0": ["Apple", "Cherry"],
                    "node_1": ["Potato"],
                    "node_2": ["Tomato"],
                },
                "seed": 42,
            },
            "docker_mesh": {
                "included_crops": ["Apple", "Tomato"],
                "excluded_diseases": {"Apple": ["Black_rot"]},
                "non_iid_strategy": "by_disease",
            },
        }
    )
    output_root = tmp_path / "docker_mesh"
    split(cfg, output_root)

    all_classes = set()
    for sub in ["probe", "node_0", "node_1", "node_2"]:
        all_classes |= {f.parent.name for f in (output_root / sub).rglob("*.jpg")}

    # Potato is excluded (not in included_crops) and Apple___Black_rot is
    # excluded by name -- neither should show up anywhere in the split.
    assert "Potato___healthy" not in all_classes
    assert "Apple___Black_rot" not in all_classes
    assert all_classes == {"Apple___Apple_scab", "Apple___healthy", "Tomato___Bacterial_spot", "Tomato___healthy"}

    # by_disease (not manual_node_crops' whole-crop assignment) means every
    # node should have received at least some data out of only 2 crops.
    for sub in ["node_0", "node_1", "node_2"]:
        assert any((output_root / sub).rglob("*.jpg")), f"{sub} got no data from the filtered split"
