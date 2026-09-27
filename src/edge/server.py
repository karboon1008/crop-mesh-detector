"""The co-op's knowledge server: the one shared knowledge database, over HTTP.

It holds three things, and never any photo, label or model weight:

  - the label space (classes.json), so every farm's model heads line up
  - round announcements: for each round, which public probe images to use
    and which farms are expected to take part
  - the knowledge table (src/federated/knowledge_store.py): each farm's
    latest probe logits and prototypes, labelled with the farm

Run on the co-op server, a cloud VM, or one farm's gateway:

    python -m src.edge.server --db edge_run/knowledge.db --port 8000

Payloads travel as torch-serialized dicts of plain tensors and numbers and
are decoded with torch.load(weights_only=True), so a farm can't make the
server run arbitrary code.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Response

from src.federated.knowledge_store import KnowledgeStore, decode_payload, encode_payload

NODE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
OCTET = "application/octet-stream"


class _Hub:
    """Label space, registered farms and round announcements — small JSON
    state next to the knowledge database, written atomically.
    """

    def __init__(self, path: Path, reset: bool):
        self.path = path
        self.lock = threading.Lock()
        if reset:
            path.unlink(missing_ok=True)
        self.state = json.loads(path.read_text()) if path.exists() else {"classes": None, "nodes": [], "rounds": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)


def _check_node(node_id: str) -> None:
    if not NODE_ID.match(node_id):
        raise HTTPException(400, f"invalid node id {node_id!r}")


def create_app(db_path: str | Path, reset: bool = False) -> FastAPI:
    db_path = Path(db_path)
    store = KnowledgeStore(db_path, reset=reset)
    hub = _Hub(db_path.with_suffix(".hub.json"), reset=reset)
    app = FastAPI(title="Crop mesh knowledge server")

    @app.get("/health")
    def health():
        return {"ok": True}

    # ---- label space
    @app.put("/classes")
    async def put_classes(request: Request):
        data = await request.json()
        missing = {"crop_classes", "disease_classes", "name_to_crop_disease", "image_size"} - set(data)
        if missing:
            raise HTTPException(400, f"classes.json is missing {sorted(missing)}")
        with hub.lock:
            if hub.state["classes"] not in (None, data):
                raise HTTPException(409, "the co-op already has a different label space")
            hub.state["classes"] = data
            hub.save()
        return {"ok": True}

    @app.get("/classes")
    def get_classes():
        if hub.state["classes"] is None:
            raise HTTPException(404, "no label space yet")
        return hub.state["classes"]

    # ---- farms
    @app.post("/nodes/{node_id}")
    def register(node_id: str):
        _check_node(node_id)
        with hub.lock:
            if node_id not in hub.state["nodes"]:
                hub.state["nodes"].append(node_id)
                hub.save()
        return {"nodes": hub.state["nodes"]}

    @app.get("/nodes")
    def nodes():
        return {"nodes": hub.state["nodes"]}

    # ---- rounds
    @app.put("/rounds/{batch_idx}")
    async def announce_round(batch_idx: int, request: Request):
        data = await request.json()
        if not isinstance(data.get("probe"), list) or not isinstance(data.get("nodes"), list):
            raise HTTPException(400, "a round needs 'probe' (image paths) and 'nodes' (expected farms)")
        with hub.lock:
            hub.state["rounds"][str(batch_idx)] = {"probe": data["probe"], "nodes": data["nodes"]}
            hub.save()
        return {"ok": True}

    @app.get("/rounds")
    def rounds():
        return {"rounds": sorted(int(b) for b in hub.state["rounds"])}

    @app.get("/rounds/{batch_idx}")
    def get_round(batch_idx: int):
        entry = hub.state["rounds"].get(str(batch_idx))
        if entry is None:
            raise HTTPException(404, f"round {batch_idx} hasn't been announced")
        return {"batch_idx": batch_idx, **entry}

    @app.get("/rounds/{batch_idx}/status")
    def round_status(batch_idx: int):
        """Which farms have refreshed their probe logits for this round —
        what a learner waits on before it retrieves.
        """
        done = [e["node_id"] for e in store.entries() if e["logits_batch_idx"] == batch_idx]
        return {"batch_idx": batch_idx, "logits_uploaded": done}

    # ---- knowledge
    @app.post("/knowledge/{node_id}")
    async def upload(node_id: str, batch_idx: int, request: Request, logits_only: bool = False):
        _check_node(node_id)
        data = await request.body()
        if len(data) > MAX_PAYLOAD_BYTES:
            raise HTTPException(413, "payload too large")
        try:
            payload = decode_payload(data)
        except Exception as exc:  # malformed or unsafe blob
            raise HTTPException(400, f"could not decode payload: {exc}") from exc
        return {"size_bytes": store.upload(node_id, batch_idx, payload, logits_only=logits_only)}

    @app.get("/knowledge/{node_id}/peers")
    def peers(node_id: str, batch_idx: int, logits_batch: int | None = None):
        """Every OTHER farm's latest entry (the requester's own is never
        returned); logits from a different round than `logits_batch` are
        left out.
        """
        _check_node(node_id)
        fetched = store.fetch_peers(node_id, batch_idx, logits_batch=logits_batch)
        buffer = io.BytesIO()
        torch.save(
            {peer: {"batch_idx": peer_batch, "payload": encode_payload(p)} for peer, (peer_batch, p) in fetched.items()},
            buffer,
        )
        return Response(buffer.getvalue(), media_type=OCTET)

    @app.get("/entries")
    def entries():
        return {"entries": store.entries()}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="edge_run/knowledge.db", help="Knowledge database file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reset", action="store_true", help="Start from an empty database")
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(create_app(args.db, reset=args.reset), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
