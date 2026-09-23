"""The one shared knowledge database every node uploads to and retrieves
from in the continual mesh (src/federated/continual.py).

Each row is labelled with the node it came from. A node keeps exactly one
live entry — its LATEST prototypes + probe logits — and a new upload
replaces the previous one. Only KnowledgePayloads (class prototypes and
probe-set logits) are ever stored: never images, labels, gradients, or
weights.

Every upload and every retrieval is also appended to a log table with its
byte size, so communication cost is read straight off the database.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.federated.node import KnowledgePayload

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge (
    node_id TEXT PRIMARY KEY,
    batch_idx INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    payload BLOB NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    batch_idx INTEGER NOT NULL,
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
    # weights_only=False: the tuple-keyed prototypes dict isn't loadable
    # otherwise. Only ever reads blobs this process wrote itself.
    obj = torch.load(io.BytesIO(data), weights_only=False)
    return KnowledgePayload(
        prototypes=obj["prototypes"],
        crop_logits=obj["crop_logits"],
        disease_logits=obj["disease_logits"],
        known_crop_classes=obj["known_crop_classes"],
        known_disease_classes=obj["known_disease_classes"],
    )


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

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def upload(self, node_id: str, batch_idx: int, payload: KnowledgePayload) -> int:
        """Replaces `node_id`'s previous entry with `payload`. Returns bytes uploaded."""
        size = payload.size_bytes()
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO knowledge (node_id, batch_idx, size_bytes, payload, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(node_id) DO UPDATE SET batch_idx=excluded.batch_idx, "
                "size_bytes=excluded.size_bytes, payload=excluded.payload, uploaded_at=excluded.uploaded_at",
                (node_id, batch_idx, size, encode_payload(payload), now),
            )
            conn.execute(
                "INSERT INTO uploads (node_id, batch_idx, size_bytes, uploaded_at) VALUES (?, ?, ?, ?)",
                (node_id, batch_idx, size, now),
            )
        return size

    def fetch_peers(self, node_id: str, batch_idx: int) -> dict[str, tuple[int, KnowledgePayload]]:
        """Every OTHER node's latest entry as {peer_id: (uploaded_in_batch, payload)}
        — the requesting node's own entry is never returned, so a node can't
        distil towards its own knowledge. Logs each retrieval's size.
        """
        now = _now()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, batch_idx, size_bytes, payload FROM knowledge WHERE node_id != ? ORDER BY node_id",
                (node_id,),
            ).fetchall()
            conn.executemany(
                "INSERT INTO retrievals (batch_idx, from_node, to_node, from_batch_idx, size_bytes, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(batch_idx, peer, node_id, peer_batch, size, now) for peer, peer_batch, size, _ in rows],
            )
        return {peer: (peer_batch, decode_payload(blob)) for peer, peer_batch, _, blob in rows}

    def entries(self) -> list[dict]:
        """Current live entries (without the payload blobs)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, batch_idx, size_bytes, uploaded_at FROM knowledge ORDER BY node_id"
            ).fetchall()
        return [
            {"node_id": n, "batch_idx": b, "size_bytes": s, "uploaded_at": t} for n, b, s, t in rows
        ]
