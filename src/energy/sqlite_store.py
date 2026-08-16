"""SQLite persistence for per-round mesh metrics (energy, communication
bytes, accuracy). Written incrementally: a node's /round/start response
and its /round/gather response arrive at different times, so `upsert_row`
merges whichever columns are passed into one logical row rather than
requiring the whole row at once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS round_metrics (
    node_id TEXT NOT NULL,
    round_idx INTEGER NOT NULL,
    energy_kwh REAL,
    duration_s REAL,
    energy_method TEXT,
    knowledge_bytes_sent INTEGER,
    crop_accuracy REAL,
    disease_accuracy REAL,
    active INTEGER,
    recorded_at TEXT,
    PRIMARY KEY (node_id, round_idx)
);
"""


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def upsert_row(path: str | Path, node_id: str, round_idx: int, recorded_at: str, **fields) -> None:
    """Insert a row for (node_id, round_idx), or merge `fields` into an
    existing row. Columns not present in `fields` are left untouched on
    conflict, so a node's separate /round/start-sourced and
    /round/gather-sourced writes for the same (node_id, round_idx)
    accumulate into one row instead of clobbering each other.
    """
    init_db(path)
    columns = ["node_id", "round_idx", "recorded_at", *fields.keys()]
    placeholders = ", ".join("?" for _ in columns)
    values = [node_id, round_idx, recorded_at, *fields.values()]
    update_clause = "recorded_at=excluded.recorded_at" + (
        ", " + ", ".join(f"{col}=excluded.{col}" for col in fields) if fields else ""
    )
    sql = (
        f"INSERT INTO round_metrics ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(node_id, round_idx) DO UPDATE SET {update_clause}"
    )
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def read_all(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA_SQL)
        rows = conn.execute("SELECT * FROM round_metrics ORDER BY round_idx, node_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
