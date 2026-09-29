#!/usr/bin/env python3
"""How often does an exported leaf detector box something that isn't a leaf?

Runs a detector.onnx (the demo app's by default) over a folder of photos with NO leaves
(people, faces, rooms; see scripts/capture_negatives.py) and reports the share of photos
with at least one "leaf" box, at each of several score thresholds. Run it before and after
retraining with --negatives-dir to see the improvement. Needs only onnxruntime + Pillow.

    python scripts/check_false_alarms.py data/negatives
    python scripts/check_false_alarms.py data/negatives --detector-dir outputs/pi_export --save-boxes out/fa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "hivemind_demo"))

import leaf_detection  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Photos that contain no leaves")
    parser.add_argument("--detector-dir", default=str(leaf_detection.DETECTOR_DIR),
                        help="Folder with detector.onnx + detector_manifest.json")
    parser.add_argument("--save-boxes", default=None,
                        help="Also save each photo that got a box at 0.5, with the boxes drawn, here")
    args = parser.parse_args()

    detector = leaf_detection.load_detector(Path(args.detector_dir))
    if detector is None:
        raise SystemExit(f"No detector.onnx + detector_manifest.json in {args.detector_dir}")
    paths = [p for p in sorted(Path(args.folder).rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise SystemExit(f"No images in {args.folder}")

    hits = {t: 0 for t in THRESHOLDS}
    save_dir = Path(args.save_boxes) if args.save_boxes else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = Image.open(path).convert("RGB")
        leaves = leaf_detection.detect_leaves(detector, image, min(THRESHOLDS))
        for t in THRESHOLDS:
            hits[t] += any(leaf.score >= t for leaf in leaves)
        at_half = [leaf for leaf in leaves if leaf.score >= 0.5]
        if save_dir and at_half:
            leaf_detection.draw_leaves(image, at_half).save(save_dir / f"{path.stem}.jpg")

    print(f"{len(paths)} no-leaf photos, detector: {args.detector_dir}")
    for t in THRESHOLDS:
        print(f"  threshold {t:.1f}: {hits[t] / len(paths):6.1%} of photos get a false 'leaf' box")


if __name__ == "__main__":
    main()
