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
    baseline_crop_accuracy REAL,
    baseline_disease_accuracy REAL,
    baseline_energy_kwh REAL,
    baseline_duration_s REAL,
    recorded_at TEXT,
    PRIMARY KEY (node_id, round_idx)
);
"""

# One row per completed peer-to-peer knowledge pull (GET /knowledge/{round_idx}),
# recorded by the FETCHING node -- this is the actual transferred payload size,
# as opposed to round_metrics.knowledge_bytes_sent which is an estimate
# (one payload's size times the number of fetching peers).
TRANSFERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_transfers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_idx INTEGER NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(SCHEMA_SQL)
        conn.execute(TRANSFERS_SCHEMA_SQL)
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
    # Never runs CREATE TABLE here -- this is also called against
    # read-only-mounted files (the dashboard's view of each db), where a
    # table that doesn't exist yet must read back as "no rows", not attempt
    # a schema write that a read-only mount would reject.
    if not Path(path).exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM round_metrics ORDER BY round_idx, node_id").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def record_transfer(
    path: str | Path, round_idx: int, from_node: str, to_node: str, size_bytes: int, fetched_at: str
) -> None:
    """Records one completed peer-to-peer knowledge pull. Called by the
    fetching node right after a successful GET /knowledge/{round_idx} --
    outside any ComputeEnergyTracker scope, so this never affects a
    reported energy_kwh figure.
    """
    init_db(path)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO knowledge_transfers (round_idx, from_node, to_node, size_bytes, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (round_idx, from_node, to_node, size_bytes, fetched_at),
        )
        conn.commit()
    finally:
        conn.close()


def read_transfers(path: str | Path) -> list[dict]:
    # Same no-DDL-on-read rule as read_all: a knowledge_transfers table that
    # doesn't exist yet (nothing fetched this run, or an older db predating
    # this table) must read back empty, never trigger a schema write.
    if not Path(path).exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT round_idx, from_node, to_node, size_bytes, fetched_at "
            "FROM knowledge_transfers ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
