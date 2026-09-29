"""The HiveMind demo app's two-stage mode (apps/hivemind_demo/leaf_detection.py):
ONNX-only leaf detection, square crops and box drawing. The detector ONNX is
exported from a random-weight model, so only shapes and contracts are checked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "hivemind_demo"))

import leaf_detection  # noqa: E402


class _StubSession:
    """Stands in for onnxruntime.InferenceSession: fixed boxes in 320x320 space."""

    class _Input:
        name = "image"

    def __init__(self):
        self.last_input = None

    def get_inputs(self):
        return [self._Input()]

    def run(self, _, feeds):
        self.last_input = feeds["image"]
        boxes = np.array([[0, 0, 160, 160], [10, 10, 12, 12], [160, 80, 320, 240], [0, 0, 100, 100]], dtype=np.float32)
        return boxes, np.array([0.6, 0.95, 0.9, 0.2], dtype=np.float32), np.array([1, 1, 1, 1], dtype=np.int64)


def test_missing_detector_means_single_leaf_mode(tmp_path):
    assert leaf_detection.load_detector(tmp_path) is None


def test_detect_leaves_rescales_filters_and_sorts():
    session = _StubSession()
    detector = leaf_detection.Detector(session, {"image_size": 320, "leaf_label": 1, "score_threshold": 0.5})
    leaves = leaf_detection.detect_leaves(detector, Image.new("RGB", (640, 480)), score_threshold=0.5)

    assert session.last_input.shape == (3, 320, 320) and session.last_input.max() <= 1.0
    # 0.95 box is 4x3 px after rescale (< 2% of 480) -> dropped; 0.2 below threshold.
    assert [leaf.score for leaf in leaves] == pytest.approx([0.9, 0.6])
    assert leaves[0].box == pytest.approx((320, 120, 640, 360))  # x * 2, y * 1.5
    assert detector.default_threshold == 0.5 and detector.map_50 is None


def test_crop_is_square_and_inside_photo():
    image = Image.new("RGB", (640, 480))
    crop = leaf_detection.crop_leaf(image, leaf_detection.Leaf((600, 400, 640, 480), 0.9))
    assert crop.size[0] == crop.size[1]
    assert leaf_detection.square_crop_box((40, 40, 60, 60), (200, 100)) == (38, 38, 62, 62)


def test_draw_leaves_keeps_size_and_marks_boxes():
    image = Image.new("RGB", (300, 200), (255, 255, 255))
    leaves = [leaf_detection.Leaf((20, 40, 120, 160), 0.9), leaf_detection.Leaf((150, 30, 280, 180), 0.7)]
    annotated = leaf_detection.draw_leaves(image, leaves, ["healthy", "diseased"], style="diagnosis")
    assert annotated.size == image.size
    assert annotated.getpixel((20, 100)) == (0x2E, 0x7D, 0x32)  # healthy green edge
    assert annotated.getpixel((280, 100)) == (0xC6, 0x28, 0x28)  # diseased red edge
    assert image.getpixel((20, 100)) == (255, 255, 255)  # original untouched


def test_exported_detector_loads_and_runs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    import importlib.util

    from src.detection.leaf_detector import build_leaf_detector, save_detector

    torch.manual_seed(0)
    checkpoint = tmp_path / "det.pt"
    save_detector(build_leaf_detector(pretrained=False), checkpoint,
                  {"score_threshold": 0.4, "metrics": {"map_50": 0.61}})

    spec = importlib.util.spec_from_file_location("export_detector", REPO_ROOT / "scripts" / "export_detector_for_pi.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    out = tmp_path / "detector"
    sys.argv = ["export", "--checkpoint", str(checkpoint), "--output-dir", str(out)]
    exporter.main()

    detector = leaf_detection.load_detector(out)
    assert detector is not None
    assert detector.default_threshold == 0.4 and detector.map_50 == 0.61
    assert json.loads((out / "detector_manifest.json").read_text())["image_size"] == 320
    leaves = leaf_detection.detect_leaves(detector, Image.new("RGB", (500, 400), (40, 150, 40)), 0.0, max_leaves=3)
    assert len(leaves) <= 3
    for leaf in leaves:
        x1, y1, x2, y2 = leaf.box
        assert 0 <= x1 <= x2 <= 500 and 0 <= y1 <= y2 <= 400


def test_detector_style_labels_every_box_red():
    image = Image.new("RGB", (300, 200), (255, 255, 255))
    leaves = [leaf_detection.Leaf((20, 0, 120, 160), 0.86), leaf_detection.Leaf((150, 60, 299, 180), 0.21)]
    annotated = leaf_detection.draw_leaves(image, leaves)  # default style, no tiers needed
    red = (0xFF, 0x3B, 0x1F)
    assert annotated.getpixel((20, 100)) == red and annotated.getpixel((299, 120)) == red
    # The top-edge box's tag moves inside it; the other's sits above its top edge.
    def red_share(box):
        pixels = [annotated.getpixel((x, y)) for x in range(box[0], box[2]) for y in range(box[1], box[3])]
        return sum(px == red for px in pixels) / len(pixels)

    assert red_share((24, 4, 40, 10)) > 0.15  # inside the top box, under its top edge
    assert red_share((154, 50, 170, 56)) > 0.15  # above the second box
    assert red_share((154, 64, 170, 70)) == 0.0  # not inside the second box


def test_live_camera_decodes_capture_and_ships_detector(tmp_path, monkeypatch):
    import base64 as b64
    import io as _io

    import live_camera

    # The component serves only its own folder, so it gets a copy of the detector.
    src_dir, web_dir = tmp_path / "detector", tmp_path / "web"
    src_dir.mkdir(), web_dir.mkdir()
    (src_dir / "detector.onnx").write_bytes(b"onnx-bytes")
    monkeypatch.setattr(leaf_detection, "DETECTOR_DIR", src_dir)
    monkeypatch.setattr(live_camera, "WEB_DIR", web_dir)
    live_camera._sync_detector()
    assert (web_dir / "detector.onnx").read_bytes() == b"onnx-bytes"

    buf = _io.BytesIO()
    Image.new("RGB", (64, 48), (30, 140, 30)).save(buf, format="JPEG")
    frame = "data:image/jpeg;base64," + b64.b64encode(buf.getvalue()).decode()
    detector = leaf_detection.Detector(None, {"image_size": 320, "leaf_label": 1})
    monkeypatch.setattr(live_camera, "_component", lambda **kw: {"image": frame, "captured_at": 1234})
    photo = live_camera.live_camera(detector, 0.5)
    assert photo.file_id == "camera-1234" and Image.open(photo).size == (64, 48)

    monkeypatch.setattr(live_camera, "_component", lambda **kw: None)
    assert live_camera.live_camera(detector, 0.5) is None
