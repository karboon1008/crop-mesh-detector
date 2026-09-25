"""src/data/multi_source.py on a tiny synthetic PlantVillage + PlantDoc +
PlantWild tree: folder mapping, dropped folders, top-crop selection, and
the node assignment it writes into the config."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.config import Config
from src.data.multi_source import load_dataset


def _make(root, folders: dict[str, int]):
    rng = np.random.RandomState(0)
    for folder, n in folders.items():
        d = root / folder
        d.mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)).save(d / f"{i}.jpg")


@pytest.fixture
def sources(tmp_path):
    _make(tmp_path / "pv", {
        "Tomato___healthy": 4, "Tomato___Early_blight": 4,
        "Apple___healthy": 3, "Apple___Apple_scab": 3,
        "Potato___healthy": 2, "Potato___Early_blight": 2,
        "Orange___Haunglongbing_(Citrus_greening)": 9,  # biggest crop, but one class
    })
    _make(tmp_path / "pd", {"train/Tomato leaf": 2, "test/Apple_healthy": 1, "train/Potato leaf early blight": 1})
    _make(tmp_path / "pw", {
        "plantwild/images/Apple_Apple_scab": 4,      # PlantVillage-style name
        "plantwild/images/potato+early+blight": 3,   # free-text name
        "plantwild/images/banana leaf": 5,           # no PlantVillage crop -> dropped
        "__MACOSX/images/Apple_Apple_scab": 7,       # zip junk -> ignored
    })
    return tmp_path


def _cfg(root, **data):
    return Config({"data": {
        "root": str(root / "pv"), "image_size": 8,
        "extra_sources": {"plantdoc": [str(root / "pd")], "plantwild": [str(root / "pw")]},
        **data,
    }})


def _crop_counts(dataset):
    crops = dataset.labels.crop_classes
    return {
        crops[c]: sum(1 for t in dataset.targets if dataset.labels.class_to_crop_disease[t][0] == c)
        for c in range(len(crops))
    }


def test_merges_mapped_folders_and_drops_the_rest(sources):
    dataset = load_dataset(_cfg(sources))
    assert _crop_counts(dataset) == {"Apple": 3 + 3 + 1 + 4, "Orange": 9, "Potato": 4 + 1 + 3, "Tomato": 8 + 2}
    assert not any("banana" in p or "__MACOSX" in p for p, _ in dataset.base.samples)


def test_auto_picks_largest_multi_class_crops_and_assigns_node_groups(sources):
    cfg = _cfg(
        sources, included_crops="auto", auto_crop_count=2, auto_crop_min_diseases=2,
        auto_crop_node_groups=[["node_0", "node_1"], ["node_2", "node_3"]],
    )
    dataset = load_dataset(cfg)
    # Orange (9) is skipped for having a single class; Apple (11) > Tomato (10) > Potato (8)
    assert set(dataset.labels.crop_classes) == {"Apple", "Tomato"}
    assert cfg.get("data.manual_node_crops") == {
        "node_0": ["Apple"], "node_1": ["Apple"], "node_2": ["Tomato"], "node_3": ["Tomato"],
    }
