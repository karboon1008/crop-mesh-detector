"""Tests for the pure logic behind the Crop Disease Detection Streamlit app
(apps/crop_disease_detection/) -- the Streamlit UI itself (app.py) isn't
unit-tested here; see docs/superpowers/plans/2026-08-18-crop-disease-streamlit-app.md
for the manual verification checklist that covers it.
"""
from __future__ import annotations

import pytest

from apps.crop_disease_detection import db


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
