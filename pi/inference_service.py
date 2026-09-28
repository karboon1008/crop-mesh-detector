"""Runs on the Raspberry Pi itself. Loads the ONNX bundle produced by
scripts/export_for_pi.py and classifies images from a camera (or a single
static file, for smoke-testing without a camera) on a fixed interval.

Deliberately has no torch/torchvision/timm dependency — only onnxruntime,
numpy, and Pillow (plus opencv-python-headless or picamera2 depending on
which camera you use).

Smoke test with no camera:
    python inference_service.py --model-dir pi_export --camera file --image test.jpg --once

Live camera (USB webcam):
    python inference_service.py --model-dir pi_export --camera opencv --interval 5

Live camera (official Pi Camera Module):
    python inference_service.py --model-dir pi_export --camera picamera2 --interval 5

Two-stage mode for field photos (several leaves per frame): add --detector,
pointing at detector.onnx from scripts/export_detector_for_pi.py. Each
detected leaf is cropped and classified, and logged as its own row with
its box; a frame with no detected leaf is classified whole (leaf_index -1):
    python inference_service.py --model-dir pi_export --detector pi_export/detector.onnx --camera opencv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image


class CameraSource:
    def read(self) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FileSource(CameraSource):
    """Not a camera at all — classifies one static image repeatedly.
    Used to smoke-test the pipeline before wiring up real hardware.
    """

    def __init__(self, path: str):
        self.image = Image.open(path).convert("RGB")

    def read(self) -> Image.Image:
        return self.image


class OpenCVSource(CameraSource):
    """USB webcam via OpenCV."""

    def __init__(self, device_index: int = 0):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device_index}")

    def read(self) -> Image.Image:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read a frame from the camera")
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def close(self) -> None:
        self.cap.release()


class Picamera2Source(CameraSource):
    """Official Raspberry Pi Camera Module via picamera2 (ships with Raspberry Pi OS)."""

    def __init__(self):
        from picamera2 import Picamera2

        self.picam2 = Picamera2()
        self.picam2.start()
        time.sleep(1)  # let auto-exposure settle

    def read(self) -> Image.Image:
        array = self.picam2.capture_array()
        return Image.fromarray(array).convert("RGB")

    def close(self) -> None:
        self.picam2.stop()


def preprocess(image: Image.Image, image_size: int, mean: list[float], std: list[float]) -> np.ndarray:
    # resize & normalized
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr[np.newaxis, ...].astype(np.float32)  # NCHW, batch size 1


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def detector_preprocess(image: Image.Image, image_size: int) -> np.ndarray:
    """RGB image -> (3, S, S) float32 in [0, 1]. No mean/std: the exported
    SSDLite graph normalises internally.
    """
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    return (np.asarray(resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def detect_leaves(
    session, image: Image.Image, manifest: dict, score_threshold: float, max_leaves: int,
    min_box_fraction: float = 0.02,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """(box in original-image pixels, score) per leaf, best first. Same
    filtering as src/detection/leaf_detector.py:LeafDetector.detect.
    """
    size = manifest["image_size"]
    boxes, scores, labels = session.run(None, {session.get_inputs()[0].name: detector_preprocess(image, size)})
    width, height = image.size
    sx, sy = width / size, height / size
    min_side = min_box_fraction * min(width, height)
    leaves = []
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != manifest["leaf_label"] or float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = float(box[0]) * sx, float(box[1]) * sy, float(box[2]) * sx, float(box[3]) * sy
        if min(x2 - x1, y2 - y1) < min_side:
            continue
        leaves.append(((x1, y1, x2, y2), float(score)))
    leaves.sort(key=lambda leaf: leaf[1], reverse=True)
    return leaves[:max_leaves]


def square_crop_box(box, image_size, pad_fraction: float = 0.1) -> tuple[int, int, int, int]:
    """Copy of src/detection/leaf_detector.py:square_crop_box (the Pi has
    no src/ package): padded square around the leaf, clamped to the image,
    so the classifier's square resize doesn't stretch it.
    """
    width, height = image_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(x2 - x1, y2 - y1) * (1 + 2 * pad_fraction), width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def classify(session, input_name: str, image: Image.Image, manifest: dict) -> tuple[int, float, int, float]:
    input_arr = preprocess(image, manifest["image_size"], manifest["mean"], manifest["std"])
    crop_logits, disease_logits = session.run(None, {input_name: input_arr})
    crop_probs = softmax(crop_logits[0])
    disease_probs = softmax(disease_logits[0])
    crop_idx = int(np.argmax(crop_probs))
    disease_idx = int(np.argmax(disease_probs))
    return crop_idx, float(crop_probs[crop_idx]), disease_idx, float(disease_probs[disease_idx])


LOG_HEADER = ["timestamp", "predicted_crop", "crop_confidence", "predicted_disease", "disease_confidence", "latency_ms"]
DETECTOR_LOG_HEADER = LOG_HEADER + ["leaf_index", "x1", "y1", "x2", "y2", "detector_score"]


def build_camera(args) -> CameraSource:
    if args.camera == "file":
        if not args.image:
            raise ValueError("--camera file requires --image path/to/test.jpg")
        return FileSource(args.image)
    if args.camera == "opencv":
        return OpenCVSource(args.device_index)
    if args.camera == "picamera2":
        return Picamera2Source()
    raise ValueError(f"Unknown --camera {args.camera}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="pi_export", help="Folder containing model.onnx + manifest.json")
    parser.add_argument("--camera", choices=["file", "opencv", "picamera2"], default="opencv")
    parser.add_argument("--image", default=None, help="Static image path, only used with --camera file")
    parser.add_argument("--device-index", type=int, default=0, help="USB camera index, only used with --camera opencv")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between classifications")
    parser.add_argument("--log", default="predictions_log.csv")
    parser.add_argument("--once", action="store_true", help="Classify a single frame and exit (smoke test)")
    parser.add_argument("--detector", default=None,
                        help="detector.onnx from scripts/export_detector_for_pi.py: detect leaves, classify each crop")
    parser.add_argument("--detector-threshold", type=float, default=None,
                        help="Minimum leaf score (default: detector_manifest.json's score_threshold)")
    parser.add_argument("--max-leaves", type=int, default=10)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(model_dir / "model.onnx"))
    input_name = session.get_inputs()[0].name
    print(f"Loaded {manifest['arch']}/{manifest['node_id']} — {len(manifest['crop_classes'])} crop classes, "
          f"{len(manifest['disease_classes'])} disease classes.")

    detector = detector_manifest = None
    if args.detector:
        detector_path = Path(args.detector)
        detector_manifest = json.loads((detector_path.parent / "detector_manifest.json").read_text())
        detector = onnxruntime.InferenceSession(str(detector_path))
        if args.detector_threshold is None:
            args.detector_threshold = detector_manifest["score_threshold"]
        print(f"Leaf detector loaded (score threshold {args.detector_threshold}).")

    header = DETECTOR_LOG_HEADER if detector is not None else LOG_HEADER
    log_path = Path(args.log)
    write_header = not log_path.exists()
    if not write_header:
        with log_path.open(newline="") as existing:
            existing_header = next(csv.reader(existing), None)
        if existing_header and existing_header != header:
            raise SystemExit(
                f"{log_path} was written {'without' if detector is not None else 'with'} --detector "
                f"(different columns); pass a different --log."
            )

    camera = build_camera(args)

    try:
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            while True:
                start = time.time()
                image = camera.read()
                if detector is None:
                    targets = [(image, None)]
                else:
                    leaves = detect_leaves(detector, image, detector_manifest, args.detector_threshold, args.max_leaves)
                    targets = [(image.crop(square_crop_box(box, image.size)), (i, box, score))
                               for i, (box, score) in enumerate(leaves)] or [(image, None)]

                predictions = [(classify(session, input_name, target, manifest), leaf) for target, leaf in targets]
                # Latency covers the whole frame: detection plus every leaf's classification.
                latency_ms = (time.time() - start) * 1000
                timestamp = datetime.now(timezone.utc).isoformat()

                for (crop_idx, crop_conf, disease_idx, disease_conf), leaf in predictions:
                    row = [
                        timestamp,
                        manifest["crop_classes"][crop_idx],
                        round(crop_conf, 4),
                        manifest["disease_classes"][disease_idx],
                        round(disease_conf, 4),
                        round(latency_ms, 1),
                    ]
                    if detector is not None:
                        if leaf is None:
                            row += [-1, "", "", "", "", ""]
                        else:
                            leaf_index, box, score = leaf
                            row += [leaf_index, *(round(v, 1) for v in box), round(score, 4)]
                    print(row)
                    writer.writerow(row)
                f.flush()

                if args.once:
                    break
                time.sleep(max(0.0, args.interval - (time.time() - start)))
    finally:
        camera.close()


if __name__ == "__main__":
    main()
