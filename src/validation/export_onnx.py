"""Exports a trained mobilenet_v3_small checkpoint from train_mobilenet.py
to ONNX + a manifest.json, and spot-checks PyTorch-vs-ONNX prediction
parity -- same shape/approach as scripts/export_for_pi.py, reimplemented
here as a self-contained validation-pipeline step (no quantization, since
this pass is about accuracy correctness, not on-device footprint).
"""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime
import torch

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD
from src.models.factory import build_model


def export_onnx(model: torch.nn.Module, image_size: int, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["crop_logits", "disease_logits"],
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


def check_parity(model: torch.nn.Module, session: onnxruntime.InferenceSession, samples) -> int:
    """Compares PyTorch vs. ONNX top-1 predictions on a few samples
    (image_tensor, crop_label, disease_label). Returns mismatch count.
    """
    input_name = session.get_inputs()[0].name
    mismatches = 0
    with torch.no_grad():
        for image, _, _ in samples:
            batch = image.unsqueeze(0)
            torch_crop, torch_disease = model(batch)
            onnx_crop, onnx_disease = session.run(None, {input_name: batch.numpy()})
            if torch_crop.argmax(1).item() != onnx_crop.argmax(1).item():
                mismatches += 1
            elif torch_disease.argmax(1).item() != onnx_disease.argmax(1).item():
                mismatches += 1
    return mismatches


def export_checkpoint(
    checkpoint_path: Path,
    crop_classes: list[str],
    disease_classes: list[str],
    image_size: int,
    output_dir: Path,
    parity_samples: list | None = None,
) -> Path:
    """Builds a fresh mobilenet_v3_small, loads `checkpoint_path`'s
    weights, exports to ONNX, writes manifest.json. Returns the ONNX path.

    If `parity_samples` (a list of (image_tensor, crop_label,
    disease_label) tuples) is given, also runs check_parity against the
    exported ONNX model and prints a warning on any mismatch — same idea
    as scripts/export_for_pi.py's check_parity, reimplemented here so an
    export bug is caught before evaluate_onnx.py blames the training
    itself.
    """
    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    export_onnx(model, image_size, onnx_path)

    manifest = {
        "crop_classes": crop_classes,
        "disease_classes": disease_classes,
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if parity_samples:
        session = onnxruntime.InferenceSession(str(onnx_path))
        mismatches = check_parity(model, session, parity_samples)
        if mismatches:
            print(
                f"WARNING: {mismatches}/{len(parity_samples)} samples disagree between the "
                f"PyTorch checkpoint and the exported ONNX model — inspect before trusting "
                f"evaluate_onnx.py's report.json."
            )
        else:
            print(f"Parity OK: all {len(parity_samples)} sampled predictions match.")

    return onnx_path
