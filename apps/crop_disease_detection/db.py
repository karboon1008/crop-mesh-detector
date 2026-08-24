"""SQLite persistence for logged crop/disease detections. Pure stdlib —
no Streamlit or ML dependency here, so this is unit-testable in isolation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                predicted_crop TEXT NOT NULL,
                crop_confidence REAL NOT NULL,
                predicted_disease TEXT NOT NULL,
                disease_confidence REAL NOT NULL,
                tier TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_detection(
    db_path: Path,
    captured_at: str,
    predicted_crop: str,
    crop_confidence: float,
    predicted_disease: str,
    disease_confidence: float,
    tier: str,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO detections
                (captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent(db_path: Path, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier
            FROM detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
