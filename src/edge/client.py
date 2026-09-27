"""HTTP client for the knowledge server. `KnowledgeClient` has the same
upload / fetch_peers / entries interface as the local KnowledgeStore, so the
round logic (src/federated/continual.py) runs unchanged on an edge device.
"""

from __future__ import annotations

import io
import time

import httpx
import torch

from src.federated.knowledge_store import decode_payload, encode_payload
from src.federated.node import KnowledgePayload


class KnowledgeClient:
    def __init__(self, server: str | httpx.Client, timeout: float = 60.0):
        """`server`: the server's base URL, or an httpx.Client already
        pointing at it (tests pass FastAPI's TestClient).
        """
        self.http = server if isinstance(server, httpx.Client) else httpx.Client(base_url=server, timeout=timeout)

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code >= 400:
            raise RuntimeError(f"knowledge server: {response.status_code} {response.text}")
        return response

    # ---- KnowledgeStore interface
    def upload(self, node_id: str, batch_idx: int, payload: KnowledgePayload, logits_only: bool = False) -> int:
        response = self._check(self.http.post(
            f"/knowledge/{node_id}", params={"batch_idx": batch_idx, "logits_only": logits_only},
            content=encode_payload(payload), headers={"content-type": "application/octet-stream"},
        ))
        return int(response.json()["size_bytes"])

    def fetch_peers(
        self, node_id: str, batch_idx: int, logits_batch: int | None = None,
    ) -> dict[str, tuple[int, KnowledgePayload]]:
        params = {"batch_idx": batch_idx}
        if logits_batch is not None:
            params["logits_batch"] = logits_batch
        response = self._check(self.http.get(f"/knowledge/{node_id}/peers", params=params))
        raw = torch.load(io.BytesIO(response.content), weights_only=True)
        return {peer: (int(entry["batch_idx"]), decode_payload(entry["payload"])) for peer, entry in raw.items()}

    def entries(self) -> list[dict]:
        return self._check(self.http.get("/entries")).json()["entries"]

    # ---- co-op hub
    def put_classes(self, classes: dict) -> None:
        self._check(self.http.put("/classes", json=classes))

    def get_classes(self) -> dict:
        return self._check(self.http.get("/classes")).json()

    def register(self, node_id: str) -> list[str]:
        return self._check(self.http.post(f"/nodes/{node_id}")).json()["nodes"]

    def announce_round(self, batch_idx: int, probe: list[str], nodes: list[str]) -> None:
        self._check(self.http.put(f"/rounds/{batch_idx}", json={"probe": probe, "nodes": nodes}))

    def rounds(self) -> list[int]:
        return self._check(self.http.get("/rounds")).json()["rounds"]

    def get_round(self, batch_idx: int) -> dict:
        return self._check(self.http.get(f"/rounds/{batch_idx}")).json()

    def logits_uploaded(self, batch_idx: int) -> list[str]:
        return self._check(self.http.get(f"/rounds/{batch_idx}/status")).json()["logits_uploaded"]

    def wait_for_uploads(
        self, batch_idx: int, expected: list[str], timeout_s: float, poll_s: float = 10.0,
    ) -> list[str]:
        """Blocks until every expected farm has refreshed its logits for this
        round, or `timeout_s` passes; returns the farms that did. A farm that
        misses the deadline simply isn't part of this round's logit consensus.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            done = self.logits_uploaded(batch_idx)
            if set(expected) <= set(done) or time.monotonic() >= deadline:
                return done
            time.sleep(poll_s)
