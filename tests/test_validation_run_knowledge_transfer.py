from __future__ import annotations

import sys
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import make_subset
from src.federated.node import Node
from src.models.factory import build_model
from src.validation.corn_mesh_dataset import CornDiseaseView, prepare_corn_mesh_data
from src.validation.run_knowledge_transfer import run_kt_round


def _build_kt_nodes(data, batch_size=4):
    nodes = {}
    for node_id in ("node_0", "node_1", "node_2"):
        train_idx = data.per_node[node_id]["train_idx"]
        test_idx = data.per_node[node_id]["test_idx"]
        train_loader = DataLoader(
            CornDiseaseView(data.train_base, train_idx, data.label_map), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            CornDiseaseView(data.eval_base, test_idx, data.label_map), batch_size=batch_size, shuffle=False
        )
        model = build_model("mobilenet_v3_small", 1, 4, pretrained=False)
        nodes[node_id] = Node(node_id, model, train_loader, test_loader, device="cpu")
    return nodes


def test_run_kt_round_updates_all_nodes_and_reports_bytes(corn_scoped_config):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)

    before_control_params = {
        nid: [p.clone() for p in n.model.parameters()] for nid, n in control_nodes.items()
    }

    result = run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method="trimmed_mean",
        trim_fraction=0.0,
        krum_neighbors=1,
        distill_epochs=1,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )

    assert set(result["per_node_distill_loss"].keys()) == {"node_0", "node_1", "node_2"}
    for loss_dict in result["per_node_distill_loss"].values():
        assert set(loss_dict.keys()) == {"kd_loss", "sup_loss", "proto_loss", "total_loss"}
    assert result["total_bytes_exchanged"] > 0
    assert set(result["per_node_bytes_sent"].keys()) == {"node_0", "node_1", "node_2"}

    # control nodes' weights must have moved too (local_train ran)
    for nid, node in control_nodes.items():
        after = list(node.model.parameters())
        assert any(
            not before.equal(after_p) for before, after_p in zip(before_control_params[nid], after)
        )


def test_run_kt_round_tracks_energy_per_node_when_tracker_given(corn_scoped_config):
    from src.energy.tracker import ComputeEnergyTracker

    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)
    tracker = ComputeEnergyTracker(enabled=False, output_dir=root.parent / "energy_out", fallback_power_watts=15.0)

    run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method="trimmed_mean",
        trim_fraction=0.0,
        krum_neighbors=1,
        distill_epochs=1,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
        tracker=tracker,
    )

    assert tracker.summary()["num_tracked_blocks"] == 6  # 3 KT nodes + 3 control nodes
