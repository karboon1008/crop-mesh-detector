"""Two-stage pipeline: leaf detector (src/detection/) + per-leaf classification
in src/infer.py and pi/inference_service.py. Synthetic images only, random
weights only (nothing downloaded).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.detection.dataset import LeafBoxDataset, load_voc_folder, parse_voc_xml
from src.detection.leaf_detector import (
    LEAF_LABEL, LeafDetection, LeafDetector, build_leaf_detector, crop_leaves, load_detector, square_crop_box,
)
from src.detection.metrics import average_precision, detection_report

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_voc(xml_path: Path, size: tuple[int, int], boxes: list[tuple[float, float, float, float]]) -> None:
    objects = "".join(
        f"<object><name>Tomato leaf</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for b in boxes
    )
    xml_path.write_text(
        f"<annotation><size><width>{size[0]}</width><height>{size[1]}</height><depth>3</depth></size>"
        f"{objects}</annotation>"
    )


@pytest.fixture
def voc_folder(tmp_path):
    folder = tmp_path / "voc"
    folder.mkdir()
    for i in range(4):
        img = Image.new("RGB", (96, 64), (120, 80, 40))
        img.paste((30, 160, 40), (10 + i, 10, 50 + i, 50))
        img.save(folder / f"img_{i}.jpg")
        _write_voc(folder / f"img_{i}.xml", (96, 64), [(10 + i, 10, 50 + i, 50)])
    Image.new("RGB", (96, 64)).save(folder / "unlabelled.jpg")  # no .xml -> skipped
    return folder


def test_parse_voc_rescales_mismatched_size_and_clamps(tmp_path):
    xml = tmp_path / "a.xml"
    # Annotated as 200x100, real image 100x50: boxes halve. Second box runs
    # off the image and is clamped; third is zero-area and dropped.
    _write_voc(xml, (200, 100), [(20, 10, 60, 50), (180, 0, 260, 90), (40, 40, 40, 80)])
    assert parse_voc_xml(xml, (100, 50)) == [(10.0, 5.0, 30.0, 25.0), (90.0, 0.0, 100.0, 45.0)]


def test_load_voc_folder_skips_unannotated(voc_folder):
    items = load_voc_folder(voc_folder)
    assert [item.image_path.name for item in items] == [f"img_{i}.jpg" for i in range(4)]


def test_dataset_flip_moves_boxes(voc_folder):
    items = load_voc_folder(voc_folder)
    image, target = LeafBoxDataset(items[:1], train=True, flip_prob=1.0)[0]
    assert image.shape == (3, 64, 96)
    assert target["boxes"].tolist() == [[46.0, 10.0, 86.0, 50.0]]  # 96 - 50, 96 - 10
    assert target["labels"].tolist() == [LEAF_LABEL]


def test_average_precision_known_cases():
    gt = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}]
    perfect = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9])}]
    assert average_precision(perfect, gt) == pytest.approx(1.0)

    # A higher-scoring false positive ahead of the true positive: precision
    # is 0.5 when recall reaches 1, so AP = 0.5.
    fp_first = [{"boxes": torch.tensor([[50.0, 50.0, 60.0, 60.0], [0.0, 0.0, 10.0, 10.0]]),
                 "scores": torch.tensor([0.9, 0.8])}]
    assert average_precision(fp_first, gt) == pytest.approx(0.5)

    # A duplicate of an already-matched box is a false positive, but it
    # ranks below the true positive, so AP stays 1.
    duplicate = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.0]]),
                  "scores": torch.tensor([0.9, 0.8])}]
    assert average_precision(duplicate, gt) == pytest.approx(1.0)

    empty = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0)}]
    assert average_precision(empty, gt) == 0.0

    report = detection_report(fp_first, gt, score_threshold=0.85)
    assert report["map_50"] == pytest.approx(0.5)
    assert report["precision_at_threshold"] == 0.0 and report["recall_at_threshold"] == 0.0


def test_square_crop_box_is_square_padded_and_clamped():
    assert square_crop_box((40, 40, 60, 60), (200, 100), pad_fraction=0.1) == (38, 38, 62, 62)
    # Near the corner: shifted back inside the image, still square.
    left, top, right, bottom = square_crop_box((0, 0, 30, 10), (200, 100), pad_fraction=0.5)
    assert (left, top) == (0, 0) and right - left == bottom - top == 60
    # Larger than the image's short side: capped to it.
    left, top, right, bottom = square_crop_box((0, 0, 200, 100), (200, 100))
    assert bottom - top == right - left == 100 and 0 <= left and right <= 200


def test_crop_leaves_falls_back_to_whole_frame():
    image = Image.new("RGB", (80, 60))
    [(whole, detection)] = crop_leaves(image, [])
    assert detection is None and whole.size == image.size
    crops = crop_leaves(image, [LeafDetection((10, 10, 30, 30), 0.9)])
    assert len(crops) == 1 and crops[0][0].size[0] == crops[0][0].size[1]


class _StubDetectionModel(torch.nn.Module):
    def forward(self, images):
        return [{
            "boxes": torch.tensor([[0.0, 0.0, 50.0, 50.0], [10.0, 10.0, 12.0, 12.0],
                                   [5.0, 5.0, 60.0, 40.0], [0.0, 0.0, 40.0, 40.0]]),
            "scores": torch.tensor([0.6, 0.95, 0.9, 0.3]),
            "labels": torch.tensor([LEAF_LABEL, LEAF_LABEL, LEAF_LABEL, LEAF_LABEL]),
        } for _ in images]


def test_leaf_detector_filters_score_size_and_sorts():
    detector = LeafDetector(_StubDetectionModel(), score_threshold=0.5, min_box_fraction=0.1)
    detections = detector.detect(Image.new("RGB", (100, 80)))
    # 0.95 box is 2px (< 10% of 80) -> dropped; 0.3 below threshold -> dropped.
    assert [d.score for d in detections] == pytest.approx([0.9, 0.6])
    detector.max_leaves = 1
    assert len(detector.detect(Image.new("RGB", (100, 80)))) == 1


def test_detector_trains_and_predicts_on_synthetic_boxes(voc_folder, tmp_path):
    from src.detection.train_detector import train_detector

    torch.manual_seed(0)
    checkpoint = tmp_path / "det" / "leaf.pt"
    report = train_detector(
        voc_folder, voc_folder, checkpoint, epochs=1, batch_size=2, pretrained=False,
        device="cpu", num_workers=0, energy_enabled=False,
    )
    assert checkpoint.exists()
    metrics = json.loads((checkpoint.parent / "detector_metrics.json").read_text())
    assert metrics["num_train_images"] == 4 and len(metrics["history"]) == 1
    assert 0.0 <= report["best"]["map_50"] <= 1.0

    model = load_detector(checkpoint)
    output = model([torch.rand(3, 64, 96)])[0]
    assert set(output) == {"boxes", "scores", "labels"}
    assert output["labels"].numel() == 0 or set(output["labels"].tolist()) == {LEAF_LABEL}


def test_infer_analyse_one_row_per_leaf():
    from src.infer import analyse
    from src.predict import build_transform

    class _Classifier(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[0.0, 5.0]]), torch.tensor([[3.0, 0.0, 0.0]])

    image = Image.new("RGB", (100, 80))
    args = (image, _Classifier(), build_transform(32), "cpu", ["Apple", "Tomato"], ["healthy", "a", "b"], "x.jpg")

    plain = analyse(*args)
    assert len(plain) == 1 and plain[0]["predicted_class"] == "Tomato___healthy" and "box" not in plain[0]

    rows = analyse(*args, detector=LeafDetector(_StubDetectionModel(), min_box_fraction=0.1))
    assert [r["leaf_index"] for r in rows] == [0, 1]
    assert rows[0]["box"] == "5 5 60 40" and rows[0]["detector_score"] == pytest.approx(0.9)

    class _NoLeaves(torch.nn.Module):
        def forward(self, images):
            return [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}]

    fallback = analyse(*args, detector=LeafDetector(_NoLeaves()))
    assert len(fallback) == 1 and fallback[0]["leaf_index"] == -1 and fallback[0]["box"] == ""


def test_onnx_export_runs_with_pi_service(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    import onnxruntime

    spec = importlib.util.spec_from_file_location("export_detector_for_pi", REPO_ROOT / "scripts" / "export_detector_for_pi.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    spec = importlib.util.spec_from_file_location("pi_inference_service", REPO_ROOT / "pi" / "inference_service.py")
    pi_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi_service)

    torch.manual_seed(0)
    model = build_leaf_detector(pretrained=False).eval()
    onnx_path = tmp_path / "detector.onnx"
    exporter.export_detector_onnx(model, onnx_path)
    assert exporter.check_parity(model, onnx_path, num_samples=1) < 1e-3

    session = onnxruntime.InferenceSession(str(onnx_path))
    manifest = {"image_size": 320, "leaf_label": LEAF_LABEL}
    image = Image.new("RGB", (640, 480), (40, 150, 40))
    leaves = pi_service.detect_leaves(session, image, manifest, score_threshold=0.0, max_leaves=5)
    assert len(leaves) <= 5
    for (x1, y1, x2, y2), score in leaves:
        assert 0 <= x1 <= x2 <= 640 and 0 <= y1 <= y2 <= 480 and 0.0 <= score <= 1.0
    assert pi_service.square_crop_box((40, 40, 60, 60), (200, 100)) == square_crop_box((40, 40, 60, 60), (200, 100))
