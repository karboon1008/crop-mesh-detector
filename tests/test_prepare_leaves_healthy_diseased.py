"""scripts/prepare_leaves_healthy_diseased.py: Kaggle "Leaves: Healthy or Diseased" -> the
PlantVillage-style <Crop>___<Disease> folders the pipeline reads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.plantvillage import PlantVillageDataset

spec = importlib.util.spec_from_file_location(
    "prepare_leaves", Path(__file__).resolve().parent.parent / "scripts" / "prepare_leaves_healthy_diseased.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.mark.parametrize("path, expected", [
    ("Mango/healthy/a.jpg", ("Mango", "healthy")),
    ("Mango/diseased/a.jpg", ("Mango", "diseased")),
    ("train/Mango healthy (P0b)/a.jpg", ("Mango", "healthy")),
    ("Plants_2/test/Alstonia Scholaris diseased (P2a)/a.JPG", ("Alstonia_Scholaris", "diseased")),
    ("Guava_unhealthy/a.png", ("Guava", "diseased")),
    ("valid/healthy/a.jpg", ("Leaf", "healthy")),
    ("Leaves/Diseased/a.jpg", ("Leaf", "diseased")),
    ("lemon (P10)/Healthy/a.jpg", ("Lemon", "healthy")),
    ("Mango/a.jpg", None),
])
def test_classify_path(path, expected):
    assert prep.classify_path(Path(path)) == expected


def _img(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.RandomState(seed).randint(0, 255, size=(24, 24, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_convert_merges_splits_dedupes_and_loads_as_plantvillage(tmp_path):
    src = tmp_path / "kaggle"
    _img(src / "train" / "Mango healthy (P0b)" / "1.jpg", 1)
    _img(src / "train" / "Mango diseased (P0a)" / "2.jpg", 2)
    _img(src / "test" / "Mango diseased (P0a)" / "3.jpg", 3)
    _img(src / "train" / "Guava healthy (P3b)" / "4.jpg", 4)
    _img(src / "test" / "Guava healthy (P3b)" / "4_copy.jpg", 4)  # same photo in both splits
    (src / "README.txt").write_text("not an image")

    out = tmp_path / "LeavesHealthyDiseased"
    counts = prep.convert(src, out)
    assert counts == {"Mango___healthy": 1, "Mango___diseased": 2, "Guava___healthy": 1}

    ds = PlantVillageDataset(out, image_size=32)  # the pipeline's own loader
    assert len(ds) == 4
    assert sorted(ds.labels.crop_classes) == ["Guava", "Mango"]
    assert sorted(ds.labels.disease_classes) == ["diseased", "healthy"]

    with pytest.raises(SystemExit):
        prep.convert(src, out)  # refuses to overwrite without --overwrite
    prep.convert(src, out, overwrite=True)


def test_convert_stops_when_condition_unknown(tmp_path):
    _img(tmp_path / "src" / "Mango" / "1.jpg", 1)
    with pytest.raises(SystemExit, match="Couldn't tell healthy from diseased"):
        prep.convert(tmp_path / "src", tmp_path / "out")
