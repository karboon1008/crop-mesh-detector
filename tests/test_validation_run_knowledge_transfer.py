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


def test_run_kt_round_single_node_has_no_peers_and_does_not_crash(corn_scoped_config):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    all_nodes = _build_kt_nodes(data)
    nodes = {"node_0": all_nodes["node_0"]}  # no peers to reconcile against
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)

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

    # node_0 has no peers this round, so it is skipped rather than crashing on
    # an empty peer_payloads list (mirrors mesh.py's MeshSimulator.run_round).
    assert "node_0" not in result["per_node_distill_loss"]


import json

from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.validation.run_knowledge_transfer import export_and_evaluate, run_round_with_io


def test_export_and_evaluate_writes_report_with_local_and_cross_node(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    node = nodes["node_0"]
    local_test_idx = data.per_node["node_0"]["test_idx"]
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    node_dir = tmp_path / "node_0"

    report = export_and_evaluate(
        node, data.label_map, data.image_size, data.eval_base, local_test_idx, cross_node_idx, node_dir, keep_onnx=True
    )

    assert set(report.keys()) == {"local", "cross_node"}
    assert (node_dir / "model.onnx").exists()
    assert (node_dir / "manifest.json").exists()
    on_disk = json.loads((node_dir / "report.json").read_text())
    assert on_disk == report


def test_export_and_evaluate_discards_onnx_for_control_arm(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    control_nodes = _build_kt_nodes(data)
    node = control_nodes["node_0"]
    local_test_idx = data.per_node["node_0"]["test_idx"]
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    node_dir = tmp_path / "node_0_control"

    export_and_evaluate(
        node, data.label_map, data.image_size, data.eval_base, local_test_idx, cross_node_idx, node_dir, keep_onnx=False
    )

    assert (node_dir / "checkpoint.pt").exists()
    assert (node_dir / "report.json").exists()
    assert not (node_dir / "model.onnx").exists()
    assert not (node_dir / "manifest.json").exists()


def test_run_round_with_io_writes_round_summary(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path / "energy", fallback_power_watts=15.0)
    comm_estimator = CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )
    round_dir = tmp_path / "round_1"

    result = run_round_with_io(
        1, nodes, control_nodes, probe_loader, data, cross_node_idx, cfg, tracker, comm_estimator, round_dir
    )

    summary = json.loads((round_dir / "round_summary.json").read_text())
    assert summary["round"] == 1
    assert set(summary["per_node_scores"].keys()) == {"node_0", "node_1", "node_2"}
    for node_id, scores in summary["per_node_scores"].items():
        assert set(scores.keys()) == {"collective", "local_only_control"}
        assert "local" in scores["collective"] and "cross_node" in scores["collective"]
    assert (round_dir / "node_0" / "model.onnx").exists()
    assert (round_dir / "node_0" / "local_only_control" / "checkpoint.pt").exists()
    assert not (round_dir / "node_0" / "local_only_control" / "model.onnx").exists()
    assert result == summary


from src.validation.run_corn_pipeline import (
    node_output_dir,
    run_evaluate_stage,
    run_export_stage,
    run_train_stage,
)
from src.validation.run_knowledge_transfer import (
    build_knowledge_transfer_summary,
    evaluate_round0_baseline,
    main,
)


def _train_stage1_checkpoints(data, base_dir):
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(base_dir, node_id)
        run_train_stage(data, node_id, node_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, node_dir)
        run_evaluate_stage(data, node_id, node_dir)


def test_evaluate_round0_baseline_uses_stage1_onnx_models(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    _train_stage1_checkpoints(data, stage1_dir)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]

    baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    assert set(baseline.keys()) == {"node_0", "node_1", "node_2"}
    for node_id, report in baseline.items():
        assert "disease" in report["summary"]["per_class_accuracy"]


def test_build_knowledge_transfer_summary_has_appendix_a1_fields(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    _train_stage1_checkpoints(data, stage1_dir)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    round0_baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    # build_knowledge_transfer_summary reads
    # per_node_scores[node_id]["collective"]["cross_node"]["summary"] (the
    # same nested shape run_round_with_io's real report produces via
    # export_and_evaluate) -- reuse round0_baseline[node_id] (a plain
    # run_evaluation report, {"summary":..., "results":...}) as the stand-in
    # value for both the "local" and "cross_node" sub-keys.
    fake_round_summary = {
        "round": 1,
        "per_node_scores": {
            node_id: {
                "collective": {"local": round0_baseline[node_id], "cross_node": round0_baseline[node_id]},
                "local_only_control": {
                    "local": round0_baseline[node_id],
                    "cross_node": round0_baseline[node_id],
                },
            }
            for node_id in ("node_0", "node_1", "node_2")
        },
        "total_bytes_exchanged": 100,
        "energy": {"total_compute_energy_kwh": 0.001},
    }

    summary = build_knowledge_transfer_summary(cfg, round0_baseline, [fake_round_summary], data)

    assert summary["node_count"] == 3
    for key in (
        "data_split",
        "local_only_budget",
        "collective_budget",
        "test_set_scope",
        "fairness_exception_reason",
        "delta_g_formula",
        "per_node_scores",
        "macro_avg_and_worst_node",
        "collaboration_gain_per_disease",
        "limitation_note",
    ):
        assert key in summary
    for node_id in ("node_0", "node_1", "node_2"):
        assert set(summary["collaboration_gain_per_disease"][node_id].keys()) == {
            "healthy",
            "Common_rust",
            "Cercospora_leaf_spot_Gray_leaf_spot",
            "Northern_Leaf_Blight",
        }


def test_main_rejects_rounds_outside_valid_range(tmp_path, corn_scoped_config, monkeypatch):
    cfg, root = corn_scoped_config
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    monkeypatch.setattr(
        "sys.argv",
        ["run_knowledge_transfer", "--config", str(config_path), "--rounds", "6"],
    )

    with pytest.raises(SystemExit):
        main()


def test_main_raises_clear_error_naming_missing_export_when_stage1_partial(
    tmp_path, corn_scoped_config, monkeypatch
):
    """Simulates someone having run only 'run_corn_pipeline --stage train'
    (checkpoint.pt exists) without 'export' (no model.onnx/manifest.json) --
    a realistic partial stage-1 state that evaluate_round0_baseline's ONNX
    load would otherwise crash on later, with no "run stage 1 first"
    guidance and no indication of which file is actually missing.
    """
    cfg, root = corn_scoped_config
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    stage1_dir = tmp_path / "outputs" / "validation" / "corn_mesh"
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "checkpoint.pt").write_text("stub")  # 'train' ran, 'export' did not

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_knowledge_transfer",
            "--config",
            str(config_path),
            "--output-dir",
            str(stage1_dir),
            "--rounds",
            "1",
        ],
    )

    with pytest.raises(FileNotFoundError, match="model.onnx"):
        main()


def test_main_runs_end_to_end(tmp_path, corn_scoped_config, monkeypatch):
    cfg, root = corn_scoped_config
    config_path = tmp_path / "config.yaml"
    import math

    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    stage1_dir = tmp_path / "outputs" / "validation" / "corn_mesh"
    data = prepare_corn_mesh_data(cfg)
    _train_stage1_checkpoints(data, stage1_dir)

    kt_dir = stage1_dir / "knowledge_transfer"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_knowledge_transfer",
            "--config",
            str(config_path),
            "--output-dir",
            str(stage1_dir),
            "--rounds",
            "2",
        ],
    )

    main()

    assert (kt_dir / "round_1" / "round_summary.json").exists()
    assert (kt_dir / "round_2" / "round_summary.json").exists()
    assert (kt_dir / "knowledge_transfer_summary.json").exists()

    round_2_summary = json.loads((kt_dir / "round_2" / "round_summary.json").read_text())
    kt_summary = json.loads((kt_dir / "knowledge_transfer_summary.json").read_text())

    assert kt_summary["rounds_run"] == 2
    assert kt_summary["cumulative_bytes_exchanged"] > 0
    assert math.isfinite(kt_summary["cumulative_energy_kwh"])
    # ComputeEnergyTracker.summary() is cumulative over its whole
    # append-only log (shared across all rounds), so round 2's own
    # energy figure already IS the 2-round total -- summing round 1's
    # and round 2's figures would double-count round 1. This assertion
    # is exactly what catches that regression.
    assert kt_summary["cumulative_energy_kwh"] == round_2_summary["energy"]["total_compute_energy_kwh"]

    gain_table = kt_summary["collaboration_gain_per_disease"]
    expected_diseases = {
        "healthy",
        "Common_rust",
        "Cercospora_leaf_spot_Gray_leaf_spot",
        "Northern_Leaf_Blight",
    }
    for node_id in ("node_0", "node_1", "node_2"):
        assert set(gain_table[node_id].keys()) == expected_diseases
        for disease_name in expected_diseases:
            entry = gain_table[node_id][disease_name]
            for metric_name in (
                "round_0_accuracy",
                "round_N_collective_accuracy",
                "round_N_local_only_control_accuracy",
                "gain_vs_round0",
                "gain_vs_local_only_control",
            ):
                assert math.isfinite(entry[metric_name])
