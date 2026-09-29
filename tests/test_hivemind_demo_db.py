"""Tests for the HiveMind demo's detection log (apps/hivemind_demo/db.py)."""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "hivemind_demo_db", Path(__file__).resolve().parents[1] / "apps" / "hivemind_demo" / "db.py"
)
db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(db)


def test_save_and_get_recent_records_the_model(tmp_path):
    path = tmp_path / "d.db"
    db.init_db(path)
    db.save_detection(path, "2026-01-01T00:00:00", "Apple", 0.9, "healthy", 0.8, "healthy", "EfficientNet-Lite0 · farm 0")
    db.save_detection(path, "2026-01-01T00:00:01", "Tomato", 0.7, "Leaf_Mold", 0.65, "diseased")
    rows = db.get_recent(path)
    assert [r["predicted_crop"] for r in rows] == ["Tomato", "Apple"]
    assert rows[0]["model"] == ""
    assert rows[1]["model"] == "EfficientNet-Lite0 · farm 0"


def test_init_db_adds_model_column_to_an_old_database(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE detections (id INTEGER PRIMARY KEY AUTOINCREMENT, captured_at TEXT NOT NULL, "
        "predicted_crop TEXT NOT NULL, crop_confidence REAL NOT NULL, predicted_disease TEXT NOT NULL, "
        "disease_confidence REAL NOT NULL, tier TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO detections VALUES (1, 't', 'Apple', 0.9, 'healthy', 0.9, 'healthy')")
    conn.commit()
    conn.close()

    db.init_db(path)
    db.init_db(path)  # idempotent
    assert db.get_recent(path)[0]["model"] == ""


def test_clear_detections(tmp_path):
    path = tmp_path / "d.db"
    db.init_db(path)
    db.save_detection(path, "t", "Apple", 0.9, "healthy", 0.9, "healthy", "m")
    db.clear_detections(path)
    assert db.get_recent(path) == []
