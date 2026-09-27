"""The one shared knowledge database every node uploads to and retrieves
from in the continual mesh (src/federated/continual.py).

Each row is labelled with the node it came from. A node keeps exactly one
live entry — its latest prototypes and its latest probe logits, each
tagged with the batch it came from. A full upload (a teacher) replaces
both; a logits-only upload (every other node, every batch) replaces the
probe logits and class counts but keeps the node's previous prototypes. Only KnowledgePayloads (class prototypes and
probe-set logits) are ever stored: never images, labels, gradients, or
weights.

Every upload and every retrieval is also appended to a log table with its
byte size, so communication cost is read straight off the database.
"""

from __future__ import annotations

import io
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.federated.node import KnowledgePayload

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge (
    node_id TEXT PRIMARY KEY,
    batch_idx INTEGER NOT NULL,          -- batch the prototypes come from
    logits_batch_idx INTEGER NOT NULL,   -- batch the probe logits come from
    size_bytes INTEGER NOT NULL,
    payload BLOB NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    batch_idx INTEGER NOT NULL,
    kind TEXT NOT NULL,                  -- "full" | "logits"
    size_bytes INTEGER NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_idx INTEGER NOT NULL,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    from_batch_idx INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_payload(payload: KnowledgePayload) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "prototypes": payload.prototypes,
            "crop_logits": payload.crop_logits,
            "disease_logits": payload.disease_logits,
            "known_crop_classes": payload.known_crop_classes,
            "known_disease_classes": payload.known_disease_classes,
        },
        buffer,
    )
    return buffer.getvalue()


def decode_payload(data: bytes) -> KnowledgePayload:
    # weights_only=True: payloads arrive over the network on an edge
    # deployment (src/edge/), so only plain tensors, dicts, tuples and
    # numbers are accepted — never arbitrary pickled objects
    obj = torch.load(io.BytesIO(data), weights_only=True)
    return KnowledgePayload(
        prototypes=obj["prototypes"],
        crop_logits=obj["crop_logits"],
        disease_logits=obj["disease_logits"],
        known_crop_classes=obj["known_crop_classes"],
        known_disease_classes=obj["known_disease_classes"],
    )


def _migrate(conn: sqlite3.Connection) -> None:
    """Adds the columns introduced for logits-only uploads to a database
    created before them (a resumed run), treating old entries as full uploads.
    """
    knowledge_cols = {row[1] for row in conn.execute("PRAGMA table_info(knowledge)")}
    if "logits_batch_idx" not in knowledge_cols:
        conn.execute("ALTER TABLE knowledge ADD COLUMN logits_batch_idx INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE knowledge SET logits_batch_idx = batch_idx")
    upload_cols = {row[1] for row in conn.execute("PRAGMA table_info(uploads)")}
    if "kind" not in upload_cols:
        conn.execute("ALTER TABLE uploads ADD COLUMN kind TEXT NOT NULL DEFAULT 'full'")


class KnowledgeStore:
    """`size_bytes` is KnowledgePayload.size_bytes() — the float32 size of
    the prototypes + logits + class counts, the same figure the rest of the
    project uses for communication cost — not the serialized blob length
    (which adds torch.save container overhead a real radio link wouldn't).
    """

    def __init__(self, path: str | Path, reset: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.unlink(missing_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            _migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upload(self, node_id: str, batch_idx: int, payload: KnowledgePayload, logits_only: bool = False) -> int:
        """Full upload: replaces `node_id`'s prototypes and probe logits with
        `payload`'s. `logits_only`: replaces just the probe logits and class
        counts, keeping the prototypes (and their batch) already stored.
        Returns bytes uploaded (for a logits-only upload, just the logits and
        counts).
        """
        size = payload.size_bytes()
        now = _now()
        with self._connect() as conn:
            proto_batch = batch_idx
            if logits_only:
                row = conn.execute(
                    "SELECT batch_idx, payload FROM knowledge WHERE node_id = ?", (node_id,)
                ).fetchone()
                if row is not None:
                    proto_batch = row[0]
                    payload = replace(payload, prototypes=decode_payload(row[1]).prototypes)
            conn.execute(
                "INSERT INTO knowledge (node_id, batch_idx, logits_batch_idx, size_bytes, payload, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET batch_idx=excluded.batch_idx, "
                "logits_batch_idx=excluded.logits_batch_idx, size_bytes=excluded.size_bytes, "
                "payload=excluded.payload, uploaded_at=excluded.uploaded_at",
                (node_id, proto_batch, batch_idx, payload.size_bytes(), encode_payload(payload), now),
            )
            conn.execute(
                "INSERT INTO uploads (node_id, batch_idx, kind, size_bytes, uploaded_at) VALUES (?, ?, ?, ?, ?)",
                (node_id, batch_idx, "logits" if logits_only else "full", size, now),
            )
        return size

    def fetch_peers(
        self, node_id: str, batch_idx: int, logits_batch: int | None = None,
    ) -> dict[str, tuple[int, KnowledgePayload]]:
        """Every OTHER node's latest entry as {peer_id: (prototypes' batch, payload)}
        — the requesting node's own entry is never returned, so a node can't
        distil towards its own knowledge.

        `logits_batch`: probe logits only line up with the probe set of the
        batch they were computed on. When given, an entry whose logits come
        from any other batch comes back with its probe logits left out
        (empty tensors) — only its prototypes and class counts are
        retrieved, and only those bytes are logged.
        """
        now = _now()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, batch_idx, logits_batch_idx, payload FROM knowledge "
                "WHERE node_id != ? ORDER BY node_id",
                (node_id,),
            ).fetchall()
            peers = {}
            for peer, peer_batch, peer_logits_batch, blob in rows:
                payload = decode_payload(blob)
                if logits_batch is not None and peer_logits_batch != logits_batch:
                    payload = replace(
                        payload,
                        crop_logits=payload.crop_logits[:0],
                        disease_logits=payload.disease_logits[:0],
                    )
                peers[peer] = (peer_batch, payload)
            conn.executemany(
                "INSERT INTO retrievals (batch_idx, from_node, to_node, from_batch_idx, size_bytes, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(batch_idx, peer, node_id, peer_batch, payload.size_bytes(), now)
                 for peer, (peer_batch, payload) in peers.items()],
            )
        return peers

    def entries(self) -> list[dict]:
        """Current live entries (without the payload blobs)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, batch_idx, logits_batch_idx, size_bytes, uploaded_at FROM knowledge ORDER BY node_id"
            ).fetchall()
        return [
            {"node_id": n, "batch_idx": b, "logits_batch_idx": lb, "size_bytes": s, "uploaded_at": t}
            for n, b, lb, s, t in rows
        ]
