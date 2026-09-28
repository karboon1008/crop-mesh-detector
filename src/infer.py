"""Interactive inference using a mesh-trained checkpoint from a completed
`python -m src.train` run — for pointing your laptop's webcam at a leaf, or
classifying a handful of specific image files, and seeing the predicted
class and confidence immediately.

Unlike `src/predict.py` (batch folder + accuracy-vs-ground-truth report),
this is for ad-hoc single-shot use with no expectation of known labels.

Classify one or more image files:
    python -m src.infer --images leaf1.jpg leaf2.jpg

Classify a single snapshot from the laptop's webcam (opens a preview
window, press SPACE to capture and classify, 'q' to cancel):
    python -m src.infer --camera

Live webcam classification (continuously classifies every --interval
seconds, prediction overlaid on the preview window, 'q' to quit):
    python -m src.infer --camera --live

Two-stage mode for field photos with several leaves: add --detector with a
leaf-detector checkpoint (src/detection/train_detector.py). Each detected
leaf is cropped and classified on its own, giving one result row per leaf
(with its box); a photo with no detected leaf is classified whole:
    python -m src.infer --images field.jpg --detector outputs/detector/leaf_ssdlite.pt
    python -m src.infer --camera --live --detector outputs/detector/leaf_ssdlite.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.detection.leaf_detector import LeafDetector, crop_leaves
from src.predict import build_transform, load_model


def classify(model, transform, device, image: Image.Image) -> dict:
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        crop_logits, disease_logits = model(tensor)
    crop_probs = F.softmax(crop_logits, dim=1)[0]
    disease_probs = F.softmax(disease_logits, dim=1)[0]
    crop_conf, crop_idx = crop_probs.max(dim=0)
    disease_conf, disease_idx = disease_probs.max(dim=0)
    return {
        "crop_idx": crop_idx.item(),
        "crop_confidence": round(crop_conf.item(), 4),
        "disease_idx": disease_idx.item(),
        "disease_confidence": round(disease_conf.item(), 4),
    }


def format_result(crop_classes, disease_classes, raw: dict, source: str) -> dict:
    crop = crop_classes[raw["crop_idx"]]
    disease = disease_classes[raw["disease_idx"]]
    return {
        "file": source,
        "predicted_class": f"{crop}___{disease}",
        "predicted_crop": crop,
        "crop_confidence": raw["crop_confidence"],
        "predicted_disease": disease,
        "disease_confidence": raw["disease_confidence"],
    }


def print_result(result: dict) -> None:
    where = ""
    if "leaf_index" in result:
        where = f" [leaf {result['leaf_index']} box {result['box']}]" if result["leaf_index"] >= 0 else " [no leaf detected, whole frame]"
    print(
        f"{result['file']}{where} -> {result['predicted_class']}  "
        f"(crop: {result['predicted_crop']} {result['crop_confidence']:.2%}, "
        f"disease: {result['predicted_disease']} {result['disease_confidence']:.2%})"
    )


def analyse(
    image: Image.Image, model, transform, device, crop_classes, disease_classes, source: str,
    detector: LeafDetector | None = None,
) -> list[dict]:
    """Without a detector: one result for the whole image (the original
    behaviour). With one: one result per detected leaf crop, each with
    its box and detector score; the whole frame if no leaf is found.
    """
    if detector is None:
        raw = classify(model, transform, device, image)
        return [format_result(crop_classes, disease_classes, raw, source)]

    results = []
    for leaf_index, (leaf, detection) in enumerate(crop_leaves(image, detector.detect(image))):
        raw = classify(model, transform, device, leaf)
        result = format_result(crop_classes, disease_classes, raw, source)
        result["leaf_index"] = leaf_index if detection is not None else -1  # -1 = no leaf found, whole frame used
        result["box"] = "" if detection is None else " ".join(f"{v:.0f}" for v in detection.box)
        result["detector_score"] = "" if detection is None else round(detection.score, 4)
        results.append(result)
    return results


def classify_images(
    image_paths, model, transform, device, crop_classes, disease_classes, detector: LeafDetector | None = None
) -> list[dict]:
    results = []
    for path in image_paths:
        image = Image.open(path)
        for result in analyse(image, model, transform, device, crop_classes, disease_classes, str(path), detector):
            print_result(result)
            results.append(result)
    return results


def run_camera(
    model, transform, device, crop_classes, disease_classes, device_index: int, live: bool, interval: float,
    detector: LeafDetector | None = None,
) -> list[dict]:
    import cv2  # only imported when --camera is used, so --images-only use needs no camera deps

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device_index}")

    print("Live classification — press 'q' to quit." if live else "Press SPACE to capture and classify, 'q' to cancel.")
    results: list[dict] = []
    latest: list[dict] = []  # results for the most recent classified frame, drawn on the preview
    last_classified = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            display = frame.copy()
            for result in latest:
                if result.get("box"):
                    x1, y1, x2, y2 = (int(float(v)) for v in result["box"].split())
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display, result["predicted_class"], (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                else:
                    cv2.putText(
                        display, result["predicted_class"], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )
            cv2.imshow("crop-mesh-detector - laptop camera", display)

            key = cv2.waitKey(1 if live else 0) & 0xFF
            if key == ord("q"):
                break

            should_classify = (key == ord(" ")) if not live else (time.time() - last_classified >= interval)
            if should_classify:
                last_classified = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                latest = analyse(image, model, transform, device, crop_classes, disease_classes, "camera", detector)
                for result in latest:
                    print_result(result)
                results.extend(latest)
                if not live:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=None, help="One or more image files to classify")
    parser.add_argument("--camera", action="store_true", help="Classify snapshot(s) from the laptop's webcam instead of --images")
    parser.add_argument("--live", action="store_true", help="With --camera: continuously classify instead of a single snapshot")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between classifications in --live mode")
    parser.add_argument("--device-index", type=int, default=0, help="Webcam device index, only used with --camera")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--results", default="outputs/results_summary.json")
    parser.add_argument("--arch", default=None, help="Override: which architecture's checkpoint to use")
    parser.add_argument("--node", default=None, help="Override: which node's checkpoint to use, e.g. node_0")
    parser.add_argument("--output", default=None, help="Optional CSV path to save results")
    parser.add_argument("--detector", default=None,
                        help="Leaf-detector checkpoint: detect leaves first and classify each crop")
    parser.add_argument("--detector-threshold", type=float, default=0.5, help="Minimum leaf-detector score")
    parser.add_argument("--max-leaves", type=int, default=10, help="Most leaves classified per image")
    args = parser.parse_args()

    if not args.images and not args.camera:
        parser.error("Provide --images path [path ...] or --camera")
    if args.images and args.camera:
        parser.error("Use either --images or --camera, not both")

    model, crop_classes, disease_classes, image_size, device = load_model(
        Path(args.checkpoints_dir), Path(args.results), args.arch, args.node
    )
    transform = build_transform(image_size)
    detector = None
    if args.detector:
        detector = LeafDetector.from_checkpoint(
            Path(args.detector), device=device, score_threshold=args.detector_threshold, max_leaves=args.max_leaves
        )

    if args.camera:
        results = run_camera(
            model, transform, device, crop_classes, disease_classes, args.device_index, args.live, args.interval,
            detector,
        )
    else:
        results = classify_images(args.images, model, transform, device, crop_classes, disease_classes, detector)

    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {len(results)} result(s) to {output_path}")


if __name__ == "__main__":
    main()
