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
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(model_dir / "model.onnx"))
    input_name = session.get_inputs()[0].name
    print(f"Loaded {manifest['arch']}/{manifest['node_id']} — {len(manifest['crop_classes'])} crop classes, "
          f"{len(manifest['disease_classes'])} disease classes.")

    camera = build_camera(args)
    log_path = Path(args.log)
    write_header = not log_path.exists()

    try:
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["timestamp", "predicted_crop", "crop_confidence", "predicted_disease", "disease_confidence", "latency_ms"]
                )
            while True:
                start = time.time()
                image = camera.read()
                input_arr = preprocess(image, manifest["image_size"], manifest["mean"], manifest["std"])
                crop_logits, disease_logits = session.run(None, {input_name: input_arr})
                crop_probs = softmax(crop_logits[0])
                disease_probs = softmax(disease_logits[0])
                crop_idx = int(np.argmax(crop_probs))
                disease_idx = int(np.argmax(disease_probs))
                latency_ms = (time.time() - start) * 1000

                row = [
                    datetime.now(timezone.utc).isoformat(),
                    manifest["crop_classes"][crop_idx],
                    round(float(crop_probs[crop_idx]), 4),
                    manifest["disease_classes"][disease_idx],
                    round(float(disease_probs[disease_idx]), 4),
                    round(latency_ms, 1),
                ]
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
