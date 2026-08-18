from __future__ import annotations

import json
import sys
from pathlib import Path

import onnxruntime
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory import build_model
from src.validation.export_onnx import check_parity, export_checkpoint

CROP_CLASSES = ["Corn", "Potato"]
DISEASE_CLASSES = ["healthy", "rust"]
IMAGE_SIZE = 32


def _save_tiny_checkpoint(tmp_path) -> Path:
    model = build_model("mobilenet_v3_small", len(CROP_CLASSES), len(DISEASE_CLASSES), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path, model


def test_export_checkpoint_writes_onnx_and_manifest(tmp_path):
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path)
    output_dir = tmp_path / "export"

    onnx_path = export_checkpoint(checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir)

    assert onnx_path == output_dir / "model.onnx"
    assert onnx_path.exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["crop_classes"] == CROP_CLASSES
    assert manifest["disease_classes"] == DISEASE_CLASSES
    assert manifest["image_size"] == IMAGE_SIZE
    assert len(manifest["mean"]) == 3 and len(manifest["std"]) == 3

    session = onnxruntime.InferenceSession(str(onnx_path))
    outputs = session.get_outputs()
    assert outputs[0].shape[-1] == len(CROP_CLASSES)
    assert outputs[1].shape[-1] == len(DISEASE_CLASSES)


def test_check_parity_finds_zero_mismatches_for_the_exported_weights(tmp_path):
    checkpoint_path, model = _save_tiny_checkpoint(tmp_path)
    model.eval()
    output_dir = tmp_path / "export"

    onnx_path = export_checkpoint(checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir)
    session = onnxruntime.InferenceSession(str(onnx_path))

    samples = [(torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), 0, 0) for _ in range(4)]
    mismatches = check_parity(model, session, samples)

    assert mismatches == 0


def test_export_checkpoint_runs_parity_check_when_samples_given(tmp_path, capsys):
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path)
    output_dir = tmp_path / "export"
    parity_samples = [(torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), 0, 0) for _ in range(3)]

    onnx_path = export_checkpoint(
        checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir, parity_samples=parity_samples
    )

    assert onnx_path.exists()
    captured = capsys.readouterr()
    assert "Parity OK" in captured.out
