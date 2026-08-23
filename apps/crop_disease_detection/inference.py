"""Pure prediction logic for the Crop Disease Detection app: ONNX Runtime
session loading/validation, image preprocessing, and the crop/disease
softmax + confidence-tier classification. No Streamlit import here so this
module is unit-testable without a Streamlit runtime -- see app.py for the
Streamlit-specific caching wrapper around create_session().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.onnx"

IMAGE_SIZE = 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.60

# Fallback values used when no manifest.json sits next to model.onnx (the
# originally bundled 14-crop/21-disease model predates per-model manifests).
# create_session() overrides IMAGE_SIZE/MEAN/STD/CROP_CLASSES/DISEASE_CLASSES
# from that manifest when one is present, so the app adapts to whatever
# model is bundled (e.g. a 1-crop/10-disease Tomato-only model) instead of
# always expecting this exact shape.
# Order matches outputs/checkpoints/classes.json -- position is the model's
# class index, so these lists must stay in exactly this order.
CROP_CLASSES = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
    "Pepper,_bell", "Potato", "Raspberry", "Soybean", "Squash",
    "Strawberry", "Tomato",
]
DISEASE_CLASSES = [
    "Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy",
    "Powdery_mildew", "Cercospora_leaf_spot Gray_leaf_spot", "Common_rust",
    "Northern_Leaf_Blight", "Esca_(Black_Measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Haunglongbing_(Citrus_greening)", "Bacterial_spot", "Early_blight",
    "Late_blight", "Leaf_scorch", "Leaf_Mold", "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite", "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
]


def preprocess(image: Image.Image) -> np.ndarray:
    """RGB PIL image -> (1, 3, IMAGE_SIZE, IMAGE_SIZE) float32, normalized
    identically to training (resize -> [0,1] -> ImageNet mean/std -> CHW).
    """
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def classify_tier(crop_confidence: float, disease_confidence: float, disease_label: str) -> str:
    if crop_confidence < CONFIDENCE_THRESHOLD or disease_confidence < CONFIDENCE_THRESHOLD:
        return "uncertain"
    return "healthy" if disease_label == "healthy" else "diseased"


def _load_manifest(model_path: Path) -> dict | None:
    manifest_path = model_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def create_session(model_path: Path) -> ort.InferenceSession:
    """Loads and validates the ONNX model at model_path. Raises
    FileNotFoundError if it's missing, ValueError if its input/output shapes
    don't match the expected classes/image size.

    If a manifest.json (as written by src/validation/export_onnx.py's
    export_checkpoint()) sits next to model_path, its crop_classes/
    disease_classes/image_size/mean/std override this module's hardcoded
    defaults before validation runs -- this is what lets a differently
    shaped model (e.g. Tomato-only, 1 crop / 10 diseases) work here without
    a code change, instead of always expecting the original
    14-crop/21-disease shape.
    """
    global CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, MEAN, STD

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Place an ONNX model there with input "
            f"'image' shaped (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}) and outputs "
            f"'crop_logits' ({len(CROP_CLASSES)} classes) + 'disease_logits' "
            f"({len(DISEASE_CLASSES)} classes) -- or include a manifest.json "
            f"declaring a different shape (see src/validation/export_onnx.py)."
        )

    manifest = _load_manifest(model_path)
    if manifest is not None:
        CROP_CLASSES = manifest["crop_classes"]
        DISEASE_CLASSES = manifest["disease_classes"]
        IMAGE_SIZE = manifest["image_size"]
        MEAN = np.array(manifest["mean"], dtype=np.float32)
        STD = np.array(manifest["std"], dtype=np.float32)

    session = ort.InferenceSession(str(model_path))
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_input_shape = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    if len(inputs) != 1 or list(inputs[0].shape) != expected_input_shape:
        actual = inputs[0].shape if inputs else None
        raise ValueError(f"Model at {model_path} has input shape {actual}; expected {expected_input_shape}.")
    if (
        len(outputs) != 2
        or outputs[0].shape[-1] != len(CROP_CLASSES)
        or outputs[1].shape[-1] != len(DISEASE_CLASSES)
    ):
        raise ValueError(
            f"Model at {model_path} outputs don't match the expected "
            f"{len(CROP_CLASSES)}-crop / {len(DISEASE_CLASSES)}-disease classifier shape."
        )
    return session


def predict(session: ort.InferenceSession, image: Image.Image) -> dict:
    batch = preprocess(image)
    input_name = session.get_inputs()[0].name
    crop_logits, disease_logits = session.run(None, {input_name: batch})
    crop_probs = _softmax(crop_logits)
    disease_probs = _softmax(disease_logits)
    crop_idx = int(np.argmax(crop_probs, axis=1)[0])
    disease_idx = int(np.argmax(disease_probs, axis=1)[0])
    crop_confidence = float(crop_probs[0, crop_idx])
    disease_confidence = float(disease_probs[0, disease_idx])
    disease_label = DISEASE_CLASSES[disease_idx]
    return {
        "predicted_crop": CROP_CLASSES[crop_idx],
        "crop_confidence": crop_confidence,
        "predicted_disease": disease_label,
        "disease_confidence": disease_confidence,
        "tier": classify_tier(crop_confidence, disease_confidence, disease_label),
    }
