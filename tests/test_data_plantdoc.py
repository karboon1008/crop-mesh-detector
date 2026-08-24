from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.plantdoc import PLANTDOC_CANONICAL_MAP, load_plantdoc_tomato_paths


def _make_images(dir_path: Path, count: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")


def test_canonical_map_has_no_target_spot_and_maps_leaf_to_healthy():
    assert "Tomato leaf" in PLANTDOC_CANONICAL_MAP
    assert PLANTDOC_CANONICAL_MAP["Tomato leaf"] == "healthy"
    assert "Target_Spot" not in PLANTDOC_CANONICAL_MAP.values()


def test_pools_train_and_test_subfolders(tmp_path):
    root = tmp_path / "PlantDoc"
    _make_images(root / "train" / "Tomato leaf bacterial spot", 3)
    _make_images(root / "test" / "Tomato leaf bacterial spot", 2)
    _make_images(root / "train" / "Apple leaf", 4)  # non-Tomato, must be ignored

    pairs = load_plantdoc_tomato_paths(root)
    assert len(pairs) == 5
    assert all(canonical == "Bacterial_spot" for _, canonical in pairs)


def test_missing_root_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PlantDoc"):
        load_plantdoc_tomato_paths(tmp_path / "does_not_exist")
