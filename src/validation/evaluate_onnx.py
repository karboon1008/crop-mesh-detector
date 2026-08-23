"""Runs the exported ONNX model (see export_onnx.py) over a held-out
split, using the exact preprocessing already verified identical to
apps/crop_disease_detection/inference.py and pi/inference_service.py, and
writes a per-image + aggregate-summary report.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from src.data.plantvillage import PlantVillageDataset


def _win_long_path(path: str) -> str:
    """Applies the `\\\\?\\` extended-length-path prefix on Windows so
    Image.open() can read files whose absolute path exceeds MAX_PATH
    (260 chars) -- same fix as apple_mesh_dataset.py's helper of the same
    name, needed here too since run_evaluation() opens test-set images
    directly by path rather than through a dataset's __getitem__. No-op on
    non-Windows platforms and already-prefixed/UNC paths.
    """
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def preprocess_image(image: Image.Image, image_size: int, mean: list[float], std: list[float]) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_onnx(session: onnxruntime.InferenceSession, image_array: np.ndarray) -> tuple[int, float, int, float]:
    input_name = session.get_inputs()[0].name
    crop_logits, disease_logits = session.run(None, {input_name: image_array})
    crop_probs = _softmax(crop_logits)[0]
    disease_probs = _softmax(disease_logits)[0]
    crop_idx = int(crop_probs.argmax())
    disease_idx = int(disease_probs.argmax())
    return crop_idx, float(crop_probs[crop_idx]), disease_idx, float(disease_probs[disease_idx])


def build_report(
    records: list[dict], model_name: str, node: str, top_n_confusions: int = 10
) -> dict:
    """`records` already contain filename/expected_*/predicted_*/
    *_confidence/*_correct keys (see run_evaluation) -- this only computes
    the aggregate summary block.
    """
    num_images = len(records)
    crop_correct = sum(1 for r in records if r["crop_correct"])
    disease_correct = sum(1 for r in records if r["disease_correct"])

    per_class_crop: dict[str, list[int]] = {}
    per_class_disease: dict[str, list[int]] = {}
    confusion_counts: dict[tuple[str, str], int] = {}
    for r in records:
        crop_bucket = per_class_crop.setdefault(r["expected_crop"], [0, 0])
        crop_bucket[0] += int(r["crop_correct"])
        crop_bucket[1] += 1

        disease_bucket = per_class_disease.setdefault(r["expected_disease"], [0, 0])
        disease_bucket[0] += int(r["disease_correct"])
        disease_bucket[1] += 1

        if not r["disease_correct"]:
            key = (r["expected_disease"], r["predicted_disease"])
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

    top_confusions = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(
            confusion_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n_confusions]
    ]

    return {
        "model": model_name,
        "node": node,
        "num_test_images": num_images,
        "crop_accuracy": crop_correct / max(1, num_images),
        "disease_accuracy": disease_correct / max(1, num_images),
        "per_class_accuracy": {
            "crop": {name: correct / total for name, (correct, total) in per_class_crop.items()},
            "disease": {name: correct / total for name, (correct, total) in per_class_disease.items()},
        },
        "top_confusions": top_confusions,
    }


def run_evaluation(
    session: onnxruntime.InferenceSession,
    eval_ds: PlantVillageDataset,
    test_idx: list[int],
    manifest: dict,
    model_name: str,
    node: str,
    output_path: Path,
) -> dict:
    records = []
    for idx in test_idx:
        path, class_idx = eval_ds.base.samples[idx]
        crop_gt_idx, disease_gt_idx = eval_ds.labels.class_to_crop_disease[class_idx]
        expected_crop = eval_ds.labels.crop_classes[crop_gt_idx]
        expected_disease = eval_ds.labels.disease_classes[disease_gt_idx]

        with Image.open(_win_long_path(path)) as img:
            image_array = preprocess_image(img, manifest["image_size"], manifest["mean"], manifest["std"])
        crop_idx, crop_conf, disease_idx, disease_conf = predict_onnx(session, image_array)
        predicted_crop = manifest["crop_classes"][crop_idx]
        predicted_disease = manifest["disease_classes"][disease_idx]

        records.append(
            {
                "filename": str(path),
                "expected_crop": expected_crop,
                "expected_disease": expected_disease,
                "predicted_crop": predicted_crop,
                "predicted_disease": predicted_disease,
                "crop_confidence": crop_conf,
                "disease_confidence": disease_conf,
                "crop_correct": predicted_crop == expected_crop,
                "disease_correct": predicted_disease == expected_disease,
            }
        )

    summary = build_report(records, model_name, node)
    report = {"summary": summary, "results": records}
    output_path.write_text(json.dumps(report, indent=2))
    return report
