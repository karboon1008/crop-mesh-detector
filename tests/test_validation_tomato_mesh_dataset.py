from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.validation.tomato_mesh_dataset import (
    TOMATO_DISEASE_ORDER,
    TomatoRawItem,
    _items_from_paths,
    build_tomato_label_map,
    build_tomato_train_eval_datasets,
    load_plantvillage_tomato_items,
)


def test_tomato_disease_order_has_ten_canonical_classes():
    assert len(TOMATO_DISEASE_ORDER) == 10
    assert "healthy" in TOMATO_DISEASE_ORDER
    assert "Target_Spot" in TOMATO_DISEASE_ORDER


def test_build_tomato_label_map():
    label_map = build_tomato_label_map()
    assert label_map.crop_classes == ["Tomato"]
    assert label_map.disease_classes == TOMATO_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == TOMATO_DISEASE_ORDER.index("healthy")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_merged_dataset_getitem_and_duck_typed_surface(tmp_path):
    label_map = build_tomato_label_map()
    p0 = tmp_path / "a.jpg"
    p1 = tmp_path / "b.jpg"
    _write_image(p0)
    _write_image(p1)
    items = [
        TomatoRawItem(source="plantvillage", path=str(p0), canonical_disease_idx=0),
        TomatoRawItem(source="plantdoc", path=str(p1), canonical_disease_idx=2),
    ]
    train_ds, eval_ds = build_tomato_train_eval_datasets(items, label_map, image_size=16)

    assert len(train_ds) == 2
    image, crop_label, disease_label = eval_ds[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 16)
    assert crop_label == 0
    assert disease_label == 2

    # duck-typed surface required by evaluate_onnx.run_evaluation / export_onnx.export_checkpoint
    assert eval_ds.base.samples[1] == (str(p1), 2)
    assert eval_ds.base.targets == [0, 2]
    assert eval_ds.labels.class_to_crop_disease[2] == (0, 2)
    assert eval_ds.labels.crop_classes == ["Tomato"]
    assert eval_ds.labels.disease_classes == TOMATO_DISEASE_ORDER

    # train has augmentation, eval doesn't -- same style assertion as
    # tests/test_validation_node1_dataset.py's build_train_eval_datasets test
    train_types = [type(t) for t in train_ds.transform.transforms]
    eval_types = [type(t) for t in eval_ds.transform.transforms]
    assert train_types != eval_types
    assert len(eval_types) == 3  # Resize, ToTensor, Normalize


def _write_pv_class(tmp_path, crop_disease_folder, count):
    cls_dir = tmp_path / "PlantVillage" / crop_disease_folder
    cls_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


def test_load_plantvillage_tomato_items_filters_to_tomato_only(tmp_path):
    label_map = build_tomato_label_map()
    _write_pv_class(tmp_path, "Tomato___Bacterial_spot", 3)
    _write_pv_class(tmp_path, "Tomato___healthy", 2)
    _write_pv_class(tmp_path, "Potato___healthy", 4)  # must be excluded

    items = load_plantvillage_tomato_items(tmp_path / "PlantVillage", label_map)
    assert len(items) == 5
    assert all(item.source == "plantvillage" for item in items)
    canonical_names = {label_map.disease_classes[i.canonical_disease_idx] for i in items}
    assert canonical_names == {"Bacterial_spot", "healthy"}


def test_items_from_paths_maps_canonical_name_to_index():
    label_map = build_tomato_label_map()
    pairs = [("/a.jpg", "healthy"), ("/b.jpg", "Early_blight")]
    items = _items_from_paths(pairs, "plantdoc", label_map)
    assert items[0].canonical_disease_idx == label_map.name_to_disease_idx["healthy"]
    assert items[1].source == "plantdoc"
