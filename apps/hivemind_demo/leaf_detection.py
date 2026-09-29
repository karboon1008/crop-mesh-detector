"""Stage 1 of the two-stage demo: find every leaf in a photo, so each one can
be classified on its own by the HiveMind farm model (stage 2).

The farm models only ever saw PlantVillage photos: one leaf filling the
frame on a plain background. A field or webcam photo has several leaves and
lots of background, which they misread. The leaf detector (SSDLite320-
MobileNetV3, COCO-initialised, fine-tuned with one "leaf" class; see
src/detection/) finds each leaf, and each padded square crop then looks like
a PlantVillage photo to the classifier.

Pure ONNX Runtime + Pillow, no torch, like inference.py. The detector is
optional: the app runs without it (centre-zoom only) until
models/detector/detector.onnx exists. Build it with:
    python -m src.detection.train_detector --train-dir <PlantDoc>/TRAIN --test-dir <PlantDoc>/TEST
    python scripts/export_detector_for_pi.py --output-dir apps/hivemind_demo/models/detector
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

DETECTOR_DIR = Path(__file__).resolve().parent / "models" / "detector"

TIER_COLORS = {"healthy": "#2e7d32", "diseased": "#c62828", "uncertain": "#ef6c00"}


@dataclass
class Leaf:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the photo's pixels
    score: float


@dataclass
class Detector:
    session: ort.InferenceSession
    manifest: dict

    @property
    def default_threshold(self) -> float:
        return float(self.manifest.get("score_threshold", 0.5))

    @property
    def map_50(self) -> float | None:
        metrics = self.manifest.get("metrics") or {}
        return metrics.get("map_50")


def load_detector(detector_dir: Path = DETECTOR_DIR) -> Detector | None:
    """The detector in `detector_dir`, or None if it hasn't been exported
    there (the app then runs single-leaf mode only).
    """
    onnx_path = detector_dir / "detector.onnx"
    manifest_path = detector_dir / "detector_manifest.json"
    if not onnx_path.exists() or not manifest_path.exists():
        return None
    return Detector(ort.InferenceSession(str(onnx_path)), json.loads(manifest_path.read_text()))


# Plant-colour check. The detector was trained only on photos that contain leaves, so it
# also boxes the main object of photos without any (a cup, a cat, sky, a wall). A box is kept
# only if at least MIN_PLANT_FRACTION of its pixels are vegetation-coloured: "excess green"
# 2g - r - b > 0.05 in normalised rgb (a standard crop-imaging index), on pixels with some
# colour (max - min channel >= 15, so grey walls and white paper don't count). Every sample
# leaf, diseased ones included, scores >= 0.13; cups, cats, bricks, coins and sky score <= 0.02.
# It does NOT reliably reject faces (a face with background can reach ~0.15): retraining with
# negative photos (src/detection/train_detector.py --negatives-dir) is the fix for people.
MIN_PLANT_FRACTION = 0.10


def plant_colour_fraction(image: Image.Image, box) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return 0.0
    crop = np.asarray(image.crop((x1, y1, x2, y2)).resize((64, 64), Image.BILINEAR), dtype=np.float32)
    total = crop.sum(axis=-1) + 1e-6
    r, g, b = crop[..., 0] / total, crop[..., 1] / total, crop[..., 2] / total
    colourful = crop.max(axis=-1) - crop.min(axis=-1) >= 15
    return float(((2 * g - r - b > 0.05) & colourful).mean())


def detect_leaves(
    detector: Detector, image: Image.Image, score_threshold: float, max_leaves: int = 10,
    min_box_fraction: float = 0.02, min_plant_fraction: float = MIN_PLANT_FRACTION,
) -> list[Leaf]:
    """Leaves in `image`, best first. The exported graph takes a 320x320
    RGB image in [0, 1] (it normalises internally, and NMS is already in
    it) and returns boxes in 320x320 pixels, rescaled here to the photo.
    Boxes whose short side is under `min_box_fraction` of the photo's are
    dropped: too small to classify once upscaled. Boxes that aren't
    plant-coloured (see MIN_PLANT_FRACTION) are dropped too; pass
    min_plant_fraction=0 to keep them.
    """
    image = image.convert("RGB")
    size = detector.manifest["image_size"]
    arr = (np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0).transpose(2, 0, 1)
    boxes, scores, labels = detector.session.run(None, {detector.session.get_inputs()[0].name: arr})

    width, height = image.size
    sx, sy = width / size, height / size
    min_side = min_box_fraction * min(width, height)
    leaves = []
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != detector.manifest["leaf_label"] or float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy
        if min(x2 - x1, y2 - y1) < min_side:
            continue
        if min_plant_fraction > 0 and plant_colour_fraction(image, (x1, y1, x2, y2)) < min_plant_fraction:
            continue
        leaves.append(Leaf((x1, y1, x2, y2), float(score)))
    leaves.sort(key=lambda leaf: leaf.score, reverse=True)
    return leaves[:max_leaves]


def square_crop_box(box, image_size, pad_fraction: float = 0.1) -> tuple[int, int, int, int]:
    """Same as src/detection/leaf_detector.py:square_crop_box: a padded
    square around the leaf, clamped to the photo. The classifier squashes
    its input to a square, and PlantVillage leaves sit in a square frame
    with a margin, so a square crop avoids stretching the leaf.
    """
    width, height = image_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * pad_fraction), width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def crop_leaf(image: Image.Image, leaf: Leaf) -> Image.Image:
    return image.convert("RGB").crop(square_crop_box(leaf.box, image.size))


DETECTOR_BOX_COLOR = "#ff3b1f"


def draw_leaves(
    image: Image.Image, leaves: list[Leaf], tiers: list[str] | None = None, style: str = "detector"
) -> Image.Image:
    """The photo with every leaf boxed and labelled on a filled tag.

    style="detector": red boxes labelled "leaf 0.86" (the detector's own
    view: what it found and how sure it is). style="diagnosis": boxes
    numbered 1, 2, ... to match the result cards and coloured by each
    leaf's result tier (needs `tiers`).

    Lower-scoring leaves are drawn first so the most confident leaves'
    labels end up on top where boxes overlap.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line = max(2, round(min(annotated.size) / 150))
    font_size = max(14, round(min(annotated.size) / 22))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        font = ImageFont.load_default()

    tiers = tiers or [None] * len(leaves)
    numbered = list(enumerate(zip(leaves, tiers), start=1))
    for number, (leaf, tier) in sorted(numbered, key=lambda item: item[1][0].score):
        if style == "diagnosis":
            color, label = TIER_COLORS.get(tier, "#1565c0"), str(number)
        else:
            color, label = DETECTOR_BOX_COLOR, f"leaf {leaf.score:.2f}"
        x1, y1, x2, y2 = leaf.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        pad = line
        tag_w, tag_h = (tx2 - tx1) + 2 * pad, (ty2 - ty1) + 2 * pad
        # Tag sits on the box's top edge; at the top of the photo it moves
        # inside the box, and it never runs off the right-hand side.
        tag_x = min(x1, annotated.size[0] - tag_w)
        tag_y = y1 - tag_h if y1 - tag_h >= 0 else y1
        draw.rectangle((tag_x, tag_y, tag_x + tag_w, tag_y + tag_h), fill=color)
        draw.text((tag_x + pad - tx1, tag_y + pad - ty1), label, fill="white", font=font)
    return annotated
