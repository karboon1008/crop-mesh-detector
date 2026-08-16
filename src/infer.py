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
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

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
    print(
        f"{result['file']} -> {result['predicted_class']}  "
        f"(crop: {result['predicted_crop']} {result['crop_confidence']:.2%}, "
        f"disease: {result['predicted_disease']} {result['disease_confidence']:.2%})"
    )


def classify_images(image_paths, model, transform, device, crop_classes, disease_classes) -> list[dict]:
    results = []
    for path in image_paths:
        image = Image.open(path)
        raw = classify(model, transform, device, image)
        result = format_result(crop_classes, disease_classes, raw, str(path))
        print_result(result)
        results.append(result)
    return results


def run_camera(
    model, transform, device, crop_classes, disease_classes, device_index: int, live: bool, interval: float
) -> list[dict]:
    import cv2  # only imported when --camera is used, so --images-only use needs no camera deps

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {device_index}")

    print("Live classification — press 'q' to quit." if live else "Press SPACE to capture and classify, 'q' to cancel.")
    results: list[dict] = []
    last_classified = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            display = frame.copy()
            if results:
                cv2.putText(
                    display, results[-1]["predicted_class"], (10, 30),
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
                raw = classify(model, transform, device, image)
                result = format_result(crop_classes, disease_classes, raw, "camera")
                print_result(result)
                results.append(result)
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
    args = parser.parse_args()

    if not args.images and not args.camera:
        parser.error("Provide --images path [path ...] or --camera")
    if args.images and args.camera:
        parser.error("Use either --images or --camera, not both")

    model, crop_classes, disease_classes, image_size, device = load_model(
        Path(args.checkpoints_dir), Path(args.results), args.arch, args.node
    )
    transform = build_transform(image_size)

    if args.camera:
        results = run_camera(
            model, transform, device, crop_classes, disease_classes, args.device_index, args.live, args.interval
        )
    else:
        results = classify_images(args.images, model, transform, device, crop_classes, disease_classes)

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
