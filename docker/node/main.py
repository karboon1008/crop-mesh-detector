# docker/node/main.py
"""FastAPI server for one mesh node. /round/start and /round/gather are
plain `def` routes (not `async def`) -- Node.local_train/distill are
blocking, synchronous PyTorch calls with no await points; FastAPI runs
`def` routes in a worker thread automatically, keeping /health responsive
while training runs. An `async def` route calling this code directly
would freeze the whole server's event loop for the duration of training.

Peer-knowledge fetches (needed inside /round/gather) use httpx.AsyncClient
+ asyncio.gather for concurrency, invoked via asyncio.run() from the
synchronous handler.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# In the container, /app has src/ copied alongside this file, so this
# insert is a harmless no-op there. Running locally (e.g. for the
# no-Docker verification workflow) this file's own directory does NOT
# contain src/ -- this makes `from src...` resolve in both cases.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
from fastapi import FastAPI, HTTPException, Response
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import (
    PlantVillageDataset,
    load_global_label_map,
    make_subset,
    train_test_split_indices,
)
from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import Node
from src.models.factory import build_model
from node_runner import NodeRunner


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def _fetch_one(client: httpx.AsyncClient, peer_id: str, base_url: str | None, round_idx: int):
    # base_url is None when peer_bases had no entry for this peer. Treat that
    # exactly like a failed fetch (return None) instead of raising: a KeyError
    # here would propagate out of the asyncio.gather() below and fail THIS
    # node's entire /round/gather, so one bad peer entry from a buggy caller
    # would cost the node its whole round rather than just that peer's
    # contribution.
    if not base_url:
        return peer_id, None
    try:
        resp = await client.get(f"{base_url}/knowledge/{round_idx}", timeout=30.0)
        if resp.status_code != 200:
            return peer_id, None
        # Recorded here (the fetching side), never inside a ComputeEnergyTracker
        # scope -- handle_round_gather's fetch/distill/evaluate work is already
        # outside the tracked block (see node_runner.py's ENERGY SCOPE CAVEAT),
        # so this write has no effect on any reported energy_kwh figure.
        sqlite_store.record_transfer(
            runner.db_path,
            round_idx,
            from_node=peer_id,
            to_node=runner.node_id,
            size_bytes=len(resp.content),
            fetched_at=_now_iso(),
        )
        return peer_id, resp.content
    except httpx.HTTPError:
        return peer_id, None


async def _fetch_all_knowledge_async(peer_ids: list, peer_bases: dict, round_idx: int) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_fetch_one(client, peer_id, peer_bases.get(peer_id), round_idx) for peer_id in peer_ids)
        )
    return dict(results)


def fetch_all_knowledge(peer_ids: list, peer_bases: dict, round_idx: int) -> dict:
    if not peer_ids:
        return {}
    return asyncio.run(_fetch_all_knowledge_async(peer_ids, peer_bases, round_idx))


def build_runner() -> NodeRunner:
    node_id = os.environ["NODE_ID"]
    data_root = os.environ["DATA_ROOT"]
    probe_root = os.environ["PROBE_ROOT"]
    classes_json = os.environ["CLASSES_JSON"]
    energy_db = os.environ["ENERGY_DB"]
    # Every container start is a fresh run, not a resume -- a stale db left
    # over from a previous run (same bind-mounted /energy dir) would show
    # old rounds/transfers under a "live" dashboard. Only this node ever
    # writes to its own db path, so deleting it here can't race another
    # container.
    log_file = os.environ.get("LOG_FILE")
    Path(energy_db).unlink(missing_ok=True)
    if log_file:
        Path(log_file).unlink(missing_ok=True)
    cfg = Config.load(os.environ.get("CONFIG_PATH"))

    global_map = load_global_label_map(classes_json)
    image_size = cfg.get("data.image_size", 160)
    local_dataset = PlantVillageDataset(data_root, image_size=image_size, global_label_map=global_map)
    probe_dataset = PlantVillageDataset(probe_root, image_size=image_size, global_label_map=global_map)

    train_idx, test_idx = train_test_split_indices(
        list(range(len(local_dataset))), cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
    )
    batch_size = cfg.get("training.batch_size", 32)
    train_loader = DataLoader(make_subset(local_dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(make_subset(local_dataset, test_idx), batch_size=batch_size, shuffle=False)
    # shuffle=False: every node must compute probe logits over the identical
    # image order for the positional peer-logit aggregation to be meaningful.
    probe_loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)

    arch = cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    model = build_model(
        arch,
        len(global_map.crop_classes),
        len(global_map.disease_classes),
        pretrained=cfg.get("models.pretrained", True),
    )
    node = Node(node_id, model, train_loader, test_loader, device="cpu")
    shadow_model = build_model(
        arch,
        len(global_map.crop_classes),
        len(global_map.disease_classes),
        pretrained=cfg.get("models.pretrained", True),
    )
    shadow_node = Node(node_id, shadow_model, train_loader, test_loader, device="cpu")
    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", False),
        output_dir="/tmp/codecarbon",
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )

    return NodeRunner(
        node_id=node_id,
        node=node,
        shadow_node=shadow_node,
        log_file=log_file,
        probe_loader=probe_loader,
        tracker=tracker,
        db_path=energy_db,
        fetch_all_knowledge=fetch_all_knowledge,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        local_epochs=cfg.get("training.local_epochs_per_round", 2),
        distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
        lr=cfg.get("training.lr", 0.001),
        distill_lr=cfg.get("training.distill_lr", 0.0005),
        proto_weight=cfg.get("training.proto_weight", 0.5),
        kd_weight=cfg.get("training.kd_weight", 0.5),
        temperature=cfg.get("training.kd_temperature", 2.0),
    )


runner = build_runner()
app = FastAPI()


@app.get("/health")
def health():
    return {"node_id": runner.node_id, "status": "online"}


@app.get("/log")
def get_log():
    return runner.activity_log[-200:]


def _require(body: dict, *keys: str) -> None:
    """Reject a malformed request body with a 400 naming the missing key,
    rather than letting a raw KeyError surface as an opaque 500.
    """
    for key in keys:
        if key not in body:
            raise HTTPException(status_code=400, detail=f"missing required field: {key!r}")


@app.post("/round/start")
def round_start(body: dict):
    _require(body, "round_idx")
    return runner.handle_round_start(body["round_idx"])


@app.get("/knowledge/{round_idx}")
def get_knowledge(round_idx: int):
    data = runner.get_knowledge_bytes(round_idx)
    if data is None:
        raise HTTPException(status_code=409, detail="knowledge not ready for this round")
    return Response(content=data, media_type="application/octet-stream")


@app.post("/round/gather")
def round_gather(body: dict):
    _require(body, "round_idx", "active_nodes", "peer_bases")
    return runner.handle_round_gather(body["round_idx"], body["active_nodes"], body["peer_bases"])


if __name__ == "__main__":
    import uvicorn

    # PORT defaults to 8000 -- exactly what docker-compose.yml and the
    # Dockerfile's EXPOSE assume -- but is overridable so several nodes can be
    # run side by side on one host (the no-Docker verification workflow) or a
    # non-default port can be used, without editing this file.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
