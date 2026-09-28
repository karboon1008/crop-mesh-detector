"""SQLite persistence for the farmer alerts service. Pure stdlib.

Tables:
  nodes      edge devices (Pi cameras): where they are, when last heard from
  alerts     one row per disease outbreak at one node; repeat detections of
             the same disease at the same node merge into it
  detections every individual report that fed an alert
  devices    iPhones registered for push notifications, per farm
  push_log   every notification attempted (also the whole output in dry-run mode)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

OPEN_STATUSES = ("new", "acknowledged")
ALL_STATUSES = ("new", "acknowledged", "treated", "false_alarm")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    farm_id TEXT NOT NULL,
    name TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    last_seen TEXT,
    battery_pct REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farm_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    crop TEXT NOT NULL,
    disease TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    max_confidence REAL NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    detection_count INTEGER NOT NULL DEFAULT 1,
    last_notified TEXT,
    image BLOB,
    note TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_farm ON alerts (farm_id, status, last_seen);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL REFERENCES alerts (id),
    captured_at TEXT NOT NULL,
    crop_confidence REAL NOT NULL,
    disease_confidence REAL NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS devices (
    token TEXT PRIMARY KEY,
    farm_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS push_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farm_id TEXT NOT NULL,
    token TEXT NOT NULL,
    alert_id INTEGER,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    sent_at TEXT NOT NULL
);
"""

ALERT_COLUMNS = (
    "id, farm_id, node_id, crop, disease, status, max_confidence, latitude, longitude, first_seen, "
    "last_seen, detection_count, last_notified, image IS NOT NULL AS has_image, note, updated_at"
)


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- nodes ---

    def upsert_node(self, node_id: str, farm_id: str, name: str, latitude: float | None, longitude: float | None) -> None:
        """Registers a node from config; keeps last_seen/battery, and keeps a
        location a heartbeat already reported if config gives none.
        """
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO nodes (node_id, farm_id, name, latitude, longitude) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (node_id) DO UPDATE SET farm_id = excluded.farm_id, name = excluded.name,
                       latitude = COALESCE(excluded.latitude, nodes.latitude),
                       longitude = COALESCE(excluded.longitude, nodes.longitude)""",
                (node_id, farm_id, name, latitude, longitude),
            )

    def heartbeat(self, node_id: str, seen_at: str, latitude=None, longitude=None, battery_pct=None) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE nodes SET last_seen = ?, latitude = COALESCE(?, latitude),
                       longitude = COALESCE(?, longitude), battery_pct = COALESCE(?, battery_pct)
                   WHERE node_id = ?""",
                (seen_at, latitude, longitude, battery_pct, node_id),
            )

    def get_node(self, node_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return dict(row) if row else None

    def list_nodes(self, farm_id: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if farm_id is None:
                rows = conn.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
            else:
                rows = conn.execute("SELECT * FROM nodes WHERE farm_id = ? ORDER BY node_id", (farm_id,)).fetchall()
        return [dict(r) for r in rows]

    # --- alerts ---

    def find_open_alert(self, farm_id: str, node_id: str, crop: str, disease: str, since: str) -> dict | None:
        """The open alert a new detection should merge into: same node, same
        crop + disease, still new/acknowledged, and seen since `since`.
        """
        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT {ALERT_COLUMNS} FROM alerts
                    WHERE farm_id = ? AND node_id = ? AND crop = ? AND disease = ?
                      AND status IN ('new', 'acknowledged') AND last_seen >= ?
                    ORDER BY last_seen DESC LIMIT 1""",
                (farm_id, node_id, crop, disease, since),
            ).fetchone()
        return dict(row) if row else None

    def create_alert(self, farm_id, node_id, crop, disease, confidence, latitude, longitude, seen_at, image) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO alerts (farm_id, node_id, crop, disease, max_confidence, latitude, longitude,
                                       first_seen, last_seen, image, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (farm_id, node_id, crop, disease, confidence, latitude, longitude, seen_at, seen_at, image, seen_at),
            )
            return cur.lastrowid

    def merge_into_alert(self, alert_id: int, confidence: float, latitude, longitude, seen_at: str, image) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE alerts SET detection_count = detection_count + 1,
                       max_confidence = MAX(max_confidence, ?), latitude = ?, longitude = ?,
                       last_seen = MAX(last_seen, ?), image = COALESCE(?, image), updated_at = ?
                   WHERE id = ?""",
                (confidence, latitude, longitude, seen_at, image, seen_at, alert_id),
            )

    def add_detection(self, alert_id, captured_at, crop_confidence, disease_confidence, latitude, longitude) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO detections (alert_id, captured_at, crop_confidence, disease_confidence, latitude, longitude)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (alert_id, captured_at, crop_confidence, disease_confidence, latitude, longitude),
            )

    def mark_notified(self, alert_id: int, at: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE alerts SET last_notified = ? WHERE id = ?", (at, alert_id))

    def get_alert(self, alert_id: int, farm_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {ALERT_COLUMNS} FROM alerts WHERE id = ? AND farm_id = ?", (alert_id, farm_id)
            ).fetchone()
        return dict(row) if row else None

    def get_alert_image(self, alert_id: int, farm_id: str) -> bytes | None:
        with self._conn() as conn:
            row = conn.execute("SELECT image FROM alerts WHERE id = ? AND farm_id = ?", (alert_id, farm_id)).fetchone()
        return row["image"] if row else None

    def list_alerts(self, farm_id: str, statuses: tuple[str, ...] = ALL_STATUSES, limit: int = 200) -> list[dict]:
        placeholders = ",".join("?" * len(statuses))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT {ALERT_COLUMNS} FROM alerts WHERE farm_id = ? AND status IN ({placeholders})
                    ORDER BY last_seen DESC LIMIT ?""",
                (farm_id, *statuses, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_open_alerts_excluding_farm(self, farm_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {ALERT_COLUMNS} FROM alerts WHERE farm_id != ? AND status IN ('new', 'acknowledged')",
                (farm_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_alert_status(self, alert_id: int, farm_id: str, status: str, note: str | None, at: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE alerts SET status = ?, note = COALESCE(?, note), updated_at = ? WHERE id = ? AND farm_id = ?",
                (status, note, at, alert_id, farm_id),
            )
            return cur.rowcount > 0

    def list_detections(self, alert_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT captured_at, crop_confidence, disease_confidence, latitude, longitude FROM detections "
                "WHERE alert_id = ? ORDER BY captured_at DESC LIMIT 50",
                (alert_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --- devices + push log ---

    def register_device(self, token: str, farm_id: str, platform: str, at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO devices (token, farm_id, platform, registered_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT (token) DO UPDATE SET farm_id = excluded.farm_id, registered_at = excluded.registered_at""",
                (token, farm_id, platform, at),
            )

    def remove_device(self, token: str, farm_id: str | None = None) -> None:
        with self._conn() as conn:
            if farm_id is None:
                conn.execute("DELETE FROM devices WHERE token = ?", (token,))
            else:
                conn.execute("DELETE FROM devices WHERE token = ? AND farm_id = ?", (token, farm_id))

    def device_tokens(self, farm_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT token FROM devices WHERE farm_id = ?", (farm_id,)).fetchall()
        return [r["token"] for r in rows]

    def log_push(self, farm_id, token, alert_id, kind, payload: dict, status, at) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO push_log (farm_id, token, alert_id, kind, payload, status, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (farm_id, token, alert_id, kind, json.dumps(payload), status, at),
            )

    def push_log(self, farm_id: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if farm_id is None:
                rows = conn.execute("SELECT * FROM push_log ORDER BY id").fetchall()
            else:
                rows = conn.execute("SELECT * FROM push_log WHERE farm_id = ? ORDER BY id", (farm_id,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
