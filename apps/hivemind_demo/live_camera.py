"""Live leaf camera: a Streamlit component (live_camera_web/index.html) that runs the leaf
detector in the browser on every camera frame and draws "leaf 0.86" boxes live. When the
user presses "Capture & diagnose" it returns the full-resolution frame, which app.py runs
through the normal two-stage pipeline (detect, then classify each leaf).
"""
from __future__ import annotations

import base64
import io
import mimetypes
import shutil
from pathlib import Path

import streamlit.components.v1 as components

import leaf_detection

WEB_DIR = Path(__file__).resolve().parent / "live_camera_web"

# The browser only runs the ONNX runtime's ES module and WebAssembly files when they are
# served with the right content types, and Python's mimetypes table doesn't know them on
# every platform (notably some Windows installs).
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

_component = components.declare_component("live_leaf_camera", path=str(WEB_DIR))


def _sync_detector() -> None:
    """The component can only serve files from its own folder, so it gets a copy of the
    detector from models/detector/ (refreshed whenever that one changes).
    """
    source = leaf_detection.DETECTOR_DIR / "detector.onnx"
    target = WEB_DIR / "detector.onnx"
    if not target.exists() or target.stat().st_size != source.stat().st_size \
            or target.stat().st_mtime < source.stat().st_mtime:
        shutil.copy2(source, target)


class CapturedPhoto(io.BytesIO):
    """A captured frame that looks like Streamlit's UploadedFile to app.py (name + file_id)."""

    def __init__(self, data: bytes, captured_at: int):
        super().__init__(data)
        self.name = f"camera-{captured_at}.jpg"
        self.file_id = f"camera-{captured_at}"


def live_camera(detector: leaf_detection.Detector, threshold: float, max_leaves: int = 10,
                min_plant_fraction: float = leaf_detection.MIN_PLANT_FRACTION,
                key: str = "live_camera") -> CapturedPhoto | None:
    """Shows the live camera; returns the last captured frame, or None before the first capture."""
    _sync_detector()
    value = _component(threshold=float(threshold), max_leaves=int(max_leaves),
                       leaf_label=int(detector.manifest["leaf_label"]), min_box_fraction=0.02,
                       min_plant_fraction=float(min_plant_fraction),
                       key=key, default=None)
    if not value or "image" not in value:
        return None
    header, _, encoded = value["image"].partition(",")
    if not header.startswith("data:image/"):
        return None
    return CapturedPhoto(base64.b64decode(encoded), int(value.get("captured_at", 0)))
