"""Edge deployment: the knowledge server over HTTP, its client, the
simulation feeder, and node agents running rounds against the server —
all in-process through FastAPI's TestClient."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.config import Config
from src.edge.client import KnowledgeClient
from src.edge.data import inbox_path
from src.edge.feeder import trigger
from src.edge.node_agent import NodeAgent
from src.edge.server import create_app
from src.federated.continual import LEARNER, TEACHER
from src.federated.knowledge_store import encode_payload
from src.federated.node import KnowledgePayload

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Apple___Black_rot",
    "Apple___healthy",
]


@pytest.fixture
def pv_root(tmp_path):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        (root / cls).mkdir(parents=True)
        for i in range(24):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / cls / f"img_{i}.jpg")
    return root


def _cfg(root, tmp_path, **continual) -> Config:
    return Config({
        "data": {
            "root": str(root), "image_size": 32, "num_nodes": 3, "seed": 0,
            "probe_set_fraction": 0.05, "test_fraction": 0.25,
            "non_iid_strategy": "dirichlet", "dirichlet_alpha": 1.0,
        },
        "continual": {
            "first_batch_size": 60, "next_batch_size": 24, "first_batch_local_epochs": 1,
            "labeled_fraction": 1.0, **continual,
        },
        "models": {"architectures": ["mobilenet_v3_small"], "pretrained": False},
        "training": {
            "local_epochs_per_round": 1, "distill_epochs_per_round": 1, "batch_size": 8,
            "lr": 1e-3, "distill_lr": 5e-4, "proto_weight": 0.1, "kd_weight": 0.5, "kd_temperature": 2.0,
        },
        "federated": {"aggregation": "trimmed_mean", "trim_fraction": 0.2, "krum_neighbors": 1},
        "energy": {"track_with_codecarbon": False},
        "edge": {"dir": str(tmp_path / "edge"), "round_timeout_s": 0, "poll_interval_s": 0, "export_onnx": False},
    })


@pytest.fixture
def client(tmp_path):
    return KnowledgeClient(TestClient(create_app(tmp_path / "server" / "knowledge.db")))


def _payload(value: float, num_probe: int = 4) -> KnowledgePayload:
    return KnowledgePayload(
        prototypes={("crop", 0): torch.full((8,), value), ("disease", 1): torch.full((8,), -value)},
        crop_logits=torch.full((num_probe, 3), value),
        disease_logits=torch.full((num_probe, 5), value),
        known_crop_classes={0: 3},
        known_disease_classes={1: 2},
    )


def test_server_round_trip_matches_the_local_store(client):
    client.upload("node_0", 0, _payload(1.0))
    client.upload("node_1", 0, _payload(2.0))
    peers = client.fetch_peers("node_0", 0, logits_batch=0)
    assert list(peers) == ["node_1"]  # never your own entry
    batch, payload = peers["node_1"]
    assert batch == 0
    assert torch.equal(payload.prototypes[("crop", 0)], torch.full((8,), 2.0))
    assert payload.known_disease_classes == {1: 2}

    # a logits-only upload keeps the old prototypes; stale logits are withheld
    client.upload("node_1", 1, _payload(5.0), logits_only=True)
    batch, payload = client.fetch_peers("node_0", 1, logits_batch=1)["node_1"]
    assert batch == 0 and torch.equal(payload.prototypes[("crop", 0)], torch.full((8,), 2.0))
    assert torch.equal(payload.crop_logits, torch.full((4, 3), 5.0))
    assert client.fetch_peers("node_1", 1, logits_batch=1)["node_0"][1].crop_logits.shape[0] == 0
    assert client.logits_uploaded(1) == ["node_1"]


def test_server_rejects_bad_ids_unsafe_payloads_and_a_second_label_space(client):
    with pytest.raises(RuntimeError, match="400"):
        client.upload("node 0;drop", 0, _payload(1.0))
    # a pickled arbitrary object is refused by weights_only decoding
    import pickle

    response = client.http.post(
        "/knowledge/node_0", params={"batch_idx": 0}, content=pickle.dumps(object()),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert client.entries() == []

    classes = {"crop_classes": ["a"], "disease_classes": ["b"], "name_to_crop_disease": {"a___b": [0, 0]}, "image_size": 32}
    client.put_classes(classes)
    client.put_classes(classes)  # idempotent
    with pytest.raises(RuntimeError, match="409"):
        client.put_classes({**classes, "image_size": 64})


def test_server_state_survives_a_restart(tmp_path):
    db = tmp_path / "knowledge.db"
    first = KnowledgeClient(TestClient(create_app(db)))
    first.register("node_0")
    first.announce_round(0, ["probe.jpg"], ["node_0"])
    first.http.post("/knowledge/node_0", params={"batch_idx": 0}, content=encode_payload(_payload(1.0)))
    second = KnowledgeClient(TestClient(create_app(db)))
    assert second.rounds() == [0]
    assert second.get_round(0)["probe"] == ["probe.jpg"]
    assert [e["node_id"] for e in second.entries()] == ["node_0"]
    fresh = KnowledgeClient(TestClient(create_app(db, reset=True)))
    assert fresh.rounds() == [] and fresh.entries() == []


def test_feeder_writes_inboxes_and_announces_the_round(pv_root, tmp_path, client):
    cfg = _cfg(pv_root, tmp_path)
    batch = trigger(cfg, client, cfg.get("edge.dir"), next_batch=False)
    announced = client.get_round(0)
    assert len(announced["probe"]) == len(batch["probe_idx"])
    assert set(announced["nodes"]) <= {"node_0", "node_1", "node_2"}
    for node_id, split in batch["nodes"].items():
        manifest = json.loads(inbox_path(cfg.get("edge.dir"), node_id, 0).read_text())
        assert len(manifest["train"]) == len(split["train_idx"])
        assert len(manifest["test"]) == len(split["test_idx"])
        assert all(e["class"] in CLASSES for e in manifest["train"] + manifest["test"])
    assert client.get_classes()["image_size"] == 32
    with pytest.raises(SystemExit):
        trigger(cfg, client, cfg.get("edge.dir"), next_batch=False)  # batch 0 exists already


def _run_round(agents: dict[str, NodeAgent], b: int) -> dict:
    """Every farm through one round with the server in between: all upload
    before any learner retrieves — what the learners' wait enforces on real
    devices."""
    contexts = {nid: agent.prepare(b) for nid, agent in agents.items()}
    for nid, ctx in contexts.items():
        if ctx is not None:
            agents[nid].train_and_upload(ctx)
    records = {}
    for nid, ctx in contexts.items():
        record = agents[nid].learn_and_finish(ctx) if ctx is not None else None
        agents[nid].save_round(b, record)
        records[nid] = record
    return records


def test_node_agents_run_the_continual_mesh_through_the_server(pv_root, tmp_path, client):
    cfg = _cfg(pv_root, tmp_path, labeled_fraction=0.3, next_batch_size=60)
    edge_dir = cfg.get("edge.dir")
    trigger(cfg, client, edge_dir, next_batch=False)
    agents = {f"node_{i}": NodeAgent(cfg, f"node_{i}", client, edge_dir, device="cpu") for i in range(3)}

    records = _run_round(agents, 0)
    present = [r for r in records.values() if r is not None]
    assert present and all(r.roles == [TEACHER, LEARNER] for r in present)
    assert all(r.uploaded and r.logits_uploaded for r in present)
    assert sorted(client.logits_uploaded(0)) == sorted(r.node_id for r in present)
    for r in present:
        if len(present) > 1:
            assert r.distilled and set(r.logit_peers) == {p.node_id for p in present} - {r.node_id}

    trigger(cfg, client, edge_dir, next_batch=True)
    records = _run_round(agents, 1)
    present = [r for r in records.values() if r is not None]
    for r in present:
        assert r.logits_uploaded  # every farm refreshes its logits every round
        assert r.uploaded == (TEACHER in r.roles)
    assert any(r.num_unlabeled > 0 for r in present)

    # each farm kept its own state and log, and a restarted agent picks up where it stopped
    for nid, agent in agents.items():
        state = json.loads((agent.dir / "agent.json").read_text())
        assert state["completed_rounds"] == [0, 1]
        assert len(json.loads((agent.dir / "rounds.json").read_text())) == 2
        restarted = NodeAgent(cfg, nid, client, edge_dir, device="cpu")
        assert restarted.pending_rounds() == []
        assert restarted.mesh.ema[nid] == state["ema"]
        if (agent.dir / "state.pt").exists():
            for a, b in zip(agent.node.model.state_dict().values(), restarted.node.model.state_dict().values()):
                assert torch.equal(a, b)


def test_a_lone_learner_times_out_and_distils_on_prototypes_only(pv_root, tmp_path, client):
    """One farm ahead of the others: no fresh peer logits arrive before the
    timeout, so it distils towards the peers' stored prototypes alone."""
    cfg = _cfg(pv_root, tmp_path)
    edge_dir = cfg.get("edge.dir")
    trigger(cfg, client, edge_dir, next_batch=False)
    agents = {f"node_{i}": NodeAgent(cfg, f"node_{i}", client, edge_dir, device="cpu") for i in range(3)}
    _run_round(agents, 0)

    trigger(cfg, client, edge_dir, next_batch=True)
    expected = client.get_round(1)["nodes"]
    if not expected:
        pytest.skip("no farm got enough photos in round 1 of this tiny synthetic stream")
    agent = agents[expected[0]]
    agent.mesh.ema[agent.node_id] = 0.0  # force the learner role
    record = agent.run_round(1)
    assert LEARNER in record.roles
    if record.distilled:
        assert record.logit_peers == [] and record.probe_images_used == 0


def test_agent_exports_the_updated_model_for_the_camera_app(pv_root, tmp_path, client):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    cfg = _cfg(pv_root, tmp_path)
    trigger(cfg, client, cfg.get("edge.dir"), next_batch=False)
    agent = NodeAgent(cfg, "node_0", client, cfg.get("edge.dir"), device="cpu")
    path = agent.export_onnx()
    session = ort.InferenceSession(str(path))
    crop, disease = session.run(None, {"image": np.zeros((1, 3, 32, 32), dtype=np.float32)})
    assert crop.shape == (1, len(agent.labels.crop_classes))
    assert disease.shape == (1, len(agent.labels.disease_classes))
