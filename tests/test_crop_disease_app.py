"""Tests for the pure logic behind the Crop Disease Detection Streamlit app
(apps/crop_disease_detection/) -- the Streamlit UI itself (app.py) isn't
unit-tested here; see docs/superpowers/plans/2026-08-18-crop-disease-streamlit-app.md
for the manual verification checklist that covers it.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from apps.crop_disease_detection import db, inference


# --- db.py ---

def test_save_and_get_recent_round_trip(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    db.save_detection(db_path, "2026-08-18T10:00:00", "Tomato", 0.91, "Early_blight", 0.83, "diseased")
    db.save_detection(db_path, "2026-08-18T10:01:00", "Apple", 0.95, "healthy", 0.88, "healthy")

    rows = db.get_recent(db_path, limit=20)

    assert len(rows) == 2
    # Most recently inserted row comes back first.
    assert rows[0]["predicted_crop"] == "Apple"
    assert rows[0]["predicted_disease"] == "healthy"
    assert rows[0]["tier"] == "healthy"
    assert rows[1]["predicted_crop"] == "Tomato"
    assert rows[1]["crop_confidence"] == pytest.approx(0.91)


def test_get_recent_respects_limit(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    for i in range(5):
        db.save_detection(db_path, f"2026-08-18T10:0{i}:00", "Tomato", 0.9, "healthy", 0.9, "healthy")

    rows = db.get_recent(db_path, limit=3)

    assert len(rows) == 3


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    db.save_detection(db_path, "2026-08-18T10:00:00", "Tomato", 0.9, "healthy", 0.9, "healthy")

    db.init_db(db_path)  # calling again must not wipe existing rows
    rows = db.get_recent(db_path)

    assert len(rows) == 1


# --- inference.py: preprocess ---

def test_preprocess_output_shape_and_dtype():
    image = Image.new("RGB", (320, 240), color=(100, 150, 200))

    result = inference.preprocess(image)

    assert result.shape == (1, 3, inference.IMAGE_SIZE, inference.IMAGE_SIZE)
    assert result.dtype == np.float32


def test_preprocess_normalizes_values():
    # A flat white image should map, per channel, to (1.0 - mean) / std.
    image = Image.new("RGB", (160, 160), color=(255, 255, 255))

    result = inference.preprocess(image)

    expected = (np.array([1.0, 1.0, 1.0], dtype=np.float32) - inference.MEAN) / inference.STD
    for c in range(3):
        assert result[0, c].mean() == pytest.approx(float(expected[c]), abs=1e-4)


# --- inference.py: classify_tier ---

@pytest.mark.parametrize(
    "crop_conf, disease_conf, disease_label, expected_tier",
    [
        (0.90, 0.90, "healthy", "healthy"),
        (0.90, 0.90, "Early_blight", "diseased"),
        (0.59, 0.90, "healthy", "uncertain"),
        (0.90, 0.59, "healthy", "uncertain"),
        (0.60, 0.60, "healthy", "healthy"),  # exactly at threshold counts as confident
    ],
)
def test_classify_tier(crop_conf, disease_conf, disease_label, expected_tier):
    assert inference.classify_tier(crop_conf, disease_conf, disease_label) == expected_tier


# --- inference.py: create_session ---

def test_create_session_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.create_session(tmp_path / "does_not_exist.onnx")


# --- inference.py: predict, end-to-end against the real bundled model ---

def test_predict_end_to_end_with_bundled_model():
    session = inference.create_session(inference.MODEL_PATH)
    image = Image.new("RGB", (200, 200), color=(80, 120, 60))

    result = inference.predict(session, image)

    assert result["predicted_crop"] in inference.CROP_CLASSES
    assert result["predicted_disease"] in inference.DISEASE_CLASSES
    assert 0.0 <= result["crop_confidence"] <= 1.0
    assert 0.0 <= result["disease_confidence"] <= 1.0
    assert result["tier"] in {"healthy", "diseased", "uncertain"}
