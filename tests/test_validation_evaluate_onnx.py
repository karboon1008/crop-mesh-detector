from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, PlantVillageDataset
from src.models.factory import build_model
from src.validation.evaluate_onnx import build_report, predict_onnx, preprocess_image, run_evaluation
from src.validation.export_onnx import export_checkpoint


def test_preprocess_image_produces_expected_shape_and_dtype():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    result = preprocess_image(img, image_size=32, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    assert result.shape == (1, 3, 32, 32)
    assert result.dtype == np.float32


def test_predict_onnx_returns_valid_index_and_confidence(tmp_path):
    crop_classes = ["Corn", "Potato"]
    disease_classes = ["healthy", "rust"]
    image_size = 32

    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    onnx_path = export_checkpoint(checkpoint_path, crop_classes, disease_classes, image_size, tmp_path / "export")
    session = onnxruntime.InferenceSession(str(onnx_path))

    image_array = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    crop_idx, crop_conf, disease_idx, disease_conf = predict_onnx(session, image_array)

    assert crop_idx in (0, 1)
    assert disease_idx in (0, 1)
    assert 0.0 <= crop_conf <= 1.0
    assert 0.0 <= disease_conf <= 1.0


def test_build_report_computes_accuracy_per_class_and_top_confusions():
    records = [
        {
            "expected_crop": "Corn", "expected_disease": "healthy",
            "predicted_crop": "Corn", "predicted_disease": "healthy",
            "crop_correct": True, "disease_correct": True,
        },
        {
            "expected_crop": "Corn", "expected_disease": "rust",
            "predicted_crop": "Corn", "predicted_disease": "blight",
            "crop_correct": True, "disease_correct": False,
        },
        {
            "expected_crop": "Potato", "expected_disease": "rust",
            "predicted_crop": "Potato", "predicted_disease": "blight",
            "crop_correct": True, "disease_correct": False,
        },
        {
            "expected_crop": "Potato", "expected_disease": "healthy",
            "predicted_crop": "Corn", "predicted_disease": "healthy",
            "crop_correct": False, "disease_correct": True,
        },
    ]

    summary = build_report(records, model_name="mobilenet_v3_small", node="node_1")

    assert summary["num_test_images"] == 4
    assert summary["crop_accuracy"] == pytest.approx(0.75)
    assert summary["disease_accuracy"] == pytest.approx(0.5)
    assert summary["per_class_accuracy"]["disease"]["rust"] == pytest.approx(0.0)
    assert summary["top_confusions"][0] == {"expected": "rust", "predicted": "blight", "count": 2}


def test_run_evaluation_writes_report_matching_test_idx_length(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "PlantVillage"
    for cls in ["Corn___healthy", "Corn___rust"]:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    eval_ds = PlantVillageDataset(root, image_size=32)
    test_idx = list(range(len(eval_ds)))

    crop_classes = eval_ds.labels.crop_classes
    disease_classes = eval_ds.labels.disease_classes
    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    onnx_path = export_checkpoint(checkpoint_path, crop_classes, disease_classes, 32, tmp_path / "export")
    manifest = json.loads((tmp_path / "export" / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(onnx_path))

    report = run_evaluation(
        session, eval_ds, test_idx, manifest, "mobilenet_v3_small", "node_1", tmp_path / "report.json"
    )

    assert (tmp_path / "report.json").exists()
    assert report["summary"]["num_test_images"] == len(test_idx)
    assert len(report["results"]) == len(test_idx)
    for record in report["results"]:
        assert set(record.keys()) == {
            "filename", "expected_crop", "expected_disease", "predicted_crop", "predicted_disease",
            "crop_confidence", "disease_confidence", "crop_correct", "disease_correct",
        }
