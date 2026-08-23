# tests/test_node_runner.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "node"))

import torch
from torch.utils.data import DataLoader

from knowledge_codec import encode_knowledge  # noqa: E402
from node_runner import NodeRunner  # noqa: E402
from src.data.plantvillage import make_subset, train_test_split_indices
from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import KnowledgePayload, Node
from src.models.factory import build_model


def _make_shadow_node(synthetic_dataset, train_loader, test_loader):
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    shadow_model = build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False)
    return Node("node_0", shadow_model, train_loader, test_loader, device="cpu")


def _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=None):
    train_idx, test_idx = train_test_split_indices(synthetic_dataset, list(range(len(synthetic_dataset))), 0.3, seed=1)
    train_loader = DataLoader(make_subset(synthetic_dataset, train_idx, train=True), batch_size=4, shuffle=True)
    test_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    probe_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    model = build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False)
    node = Node("node_0", model, train_loader, test_loader, device="cpu")
    shadow_node = _make_shadow_node(synthetic_dataset, train_loader, test_loader)
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    runner = NodeRunner(
        node_id="node_0",
        node=node,
        shadow_node=shadow_node,
        probe_loader=probe_loader,
        tracker=tracker,
        db_path=str(tmp_path / "node_0.db"),
        fetch_all_knowledge=fetch_all_knowledge or (lambda peer_ids, peer_bases, round_idx: {}),
    )
    return runner, probe_loader


def test_handle_round_start_returns_response_body_and_writes_db(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_start(0)

    assert response["round_idx"] == 0
    assert response["size_bytes"] > 0
    assert response["energy_kwh"] is not None

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert len(rows) == 1
    assert rows[0]["knowledge_bytes_sent"] == response["size_bytes"]


def test_handle_round_start_returns_baseline_fields_and_writes_them(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_start(0)

    assert "baseline_crop_accuracy" in response
    assert "baseline_disease_accuracy" in response
    assert response["baseline_energy_kwh"] is not None
    assert response["baseline_duration_s"] is not None

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["baseline_crop_accuracy"] == response["baseline_crop_accuracy"]


def test_shadow_model_never_receives_distilled_knowledge(tmp_path, synthetic_dataset):
    # The shadow model must never call .distill(...) -- handle_round_gather
    # must not touch it at all. Patch it to raise if ever called.
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.shadow_node.distill = lambda *a, **k: (_ for _ in ()).throw(AssertionError("shadow must not distill"))
    runner.handle_round_start(0)
    runner.handle_round_gather(0, ["node_0"], {})  # must not raise


def test_get_knowledge_bytes_returns_none_for_a_different_round(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_start(0)
    assert runner.get_knowledge_bytes(0) is not None
    assert runner.get_knowledge_bytes(1) is None


def test_handle_round_gather_ignores_peer_data_for_a_different_round(tmp_path, synthetic_dataset):
    n_probe_holder = {}

    def fake_fetch_all(peer_ids, peer_bases, round_idx):
        n_probe = n_probe_holder["n"]
        stale_payload = KnowledgePayload(
            prototypes={}, crop_logits=torch.zeros(n_probe, 2), disease_logits=torch.zeros(n_probe, 2),
            known_crop_classes={}, known_disease_classes={},
        )
        # encoded for round 99, but we are gathering round 0 -- must be dropped
        return {"node_1": encode_knowledge(99, stale_payload)}

    runner, probe_loader = _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=fake_fetch_all)
    n_probe_holder["n"] = len(probe_loader.dataset)

    response = runner.handle_round_gather(0, ["node_0", "node_1"], {"node_1": "http://node_1:8000"})
    assert "crop_accuracy" in response
    assert "disease_accuracy" in response

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["active"] == 1


def test_handle_round_gather_skips_a_peer_whose_fetch_failed(tmp_path, synthetic_dataset):
    def fake_fetch_all(peer_ids, peer_bases, round_idx):
        return {"node_1": None}  # peer timed out / errored

    runner, _ = _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=fake_fetch_all)
    response = runner.handle_round_gather(0, ["node_0", "node_1"], {"node_1": "http://node_1:8000"})
    assert "crop_accuracy" in response


def test_handle_round_gather_with_no_peers_still_evaluates(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_gather(0, ["node_0"], {})
    assert "crop_accuracy" in response


def test_handle_round_gather_updates_knowledge_bytes_sent_for_the_active_peer_count(
    tmp_path, synthetic_dataset
):
    # No central process re-derives this -- the node itself recomputes its
    # own knowledge_bytes_sent once it learns (via active_nodes, passed in
    # the gather request) how many peers were actually around to fetch its
    # payload this round. Same "broadcast to every other active peer"
    # multiplier as src/federated/mesh.py's RoundLog.total_bytes_exchanged.
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    start_response = runner.handle_round_start(0)
    runner.handle_round_gather(0, ["node_0", "node_1", "node_2"], {})

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["knowledge_bytes_sent"] == start_response["size_bytes"] * 2


def test_handle_round_gather_records_zero_knowledge_bytes_sent_when_alone(tmp_path, synthetic_dataset):
    # A lone active node has nobody to serve its payload to, so nothing is
    # actually transmitted -- mesh.py's max(0, active_n - 1) gives 0 here too.
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_start(0)
    runner.handle_round_gather(0, ["node_0"], {})

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["knowledge_bytes_sent"] == 0


def test_handle_round_start_records_activity_log_entries(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_start(0)

    assert len(runner.activity_log) >= 2
    assert all("ts" in e and "stage" in e and "message" in e for e in runner.activity_log)
    assert any("round 0" in e["message"] for e in runner.activity_log)


def test_handle_round_gather_records_activity_log_entries(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_gather(0, ["node_0"], {})

    stages = [e["stage"] for e in runner.activity_log]
    assert "round_gather" in stages


def test_activity_log_is_bounded(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    for i in range(350):
        runner._log("test", f"entry {i}")
    assert len(runner.activity_log) == 300
    assert runner.activity_log[-1]["message"] == "entry 349"


def test_log_file_receives_one_json_line_per_log_call(tmp_path, synthetic_dataset):
    log_path = tmp_path / "node_0.log"
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.log_file = str(log_path)
    runner._log("round_start", "hello")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "round_start"
    assert entry["message"] == "hello"
