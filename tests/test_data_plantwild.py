from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.plantwild import (
    PLANTWILD_V1_CANONICAL_MAP,
    PLANTWILD_V2_CANONICAL_MAP,
    load_plantwild_v1_tomato_paths,
    load_plantwild_v2_tomato_paths,
)


def _make_images(dir_path: Path, count: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")


def test_v2_map_has_no_healthy_leaf_entry():
    assert "tomato leaf" in PLANTWILD_V1_CANONICAL_MAP
    assert "tomato leaf" not in PLANTWILD_V2_CANONICAL_MAP
    assert PLANTWILD_V1_CANONICAL_MAP["tomato leaf"] == "healthy"


def test_v1_loader_pools_known_classes(tmp_path):
    root = tmp_path / "plantwild_images"
    _make_images(root / "tomato bacterial leaf spot", 4)
    _make_images(root / "tomato leaf", 2)
    _make_images(root / "apple black rot", 3)  # non-Tomato, ignored

    pairs = load_plantwild_v1_tomato_paths(root)
    assert len(pairs) == 6
    canonical_names = {c for _, c in pairs}
    assert canonical_names == {"Bacterial_spot", "healthy"}


def test_v2_loader_ignores_healthy_folder_even_if_present(tmp_path):
    root = tmp_path / "plantwild_v2"
    _make_images(root / "tomato bacterial leaf spot", 3)
    _make_images(root / "tomato leaf", 2)  # not in v2's map, must be ignored

    pairs = load_plantwild_v2_tomato_paths(root)
    assert len(pairs) == 3
    assert all(c == "Bacterial_spot" for _, c in pairs)


def test_missing_roots_raise_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PlantWild"):
        load_plantwild_v1_tomato_paths(tmp_path / "missing_v1")
    with pytest.raises(FileNotFoundError, match="PlantWild"):
        load_plantwild_v2_tomato_paths(tmp_path / "missing_v2")
