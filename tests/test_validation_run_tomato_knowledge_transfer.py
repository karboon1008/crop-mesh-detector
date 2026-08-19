# tests/test_validation_run_tomato_knowledge_transfer.py
from __future__ import annotations

import json

import pytest

from src.validation.run_tomato_knowledge_transfer import (
    build_tomato_knowledge_transfer_summary,
    evaluate_tomato_round0_baseline,
)
from src.validation.run_tomato_pipeline import run_evaluate_stage, run_export_stage, run_train_stage
from src.validation.tomato_mesh_dataset import prepare_tomato_mesh_data


def _run_stage1(tmp_path, cfg):
    data = prepare_tomato_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    (stage1_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "test_idx": data.test_idx,
            }
        )
    )
    node_ids = sorted(data.per_node.keys())
    for node_id in node_ids:
        node_dir = stage1_dir / f"{node_id}_mobilenet_v3_small"
        run_train_stage(data, node_id, node_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, node_dir)
        run_evaluate_stage(data, node_id, node_dir)
    return data, stage1_dir, node_ids


def test_round0_baseline_evaluates_every_node_against_global_test_set(tmp_path, tomato_scoped_config):
    data, stage1_dir, node_ids = _run_stage1(tmp_path, tomato_scoped_config)
    baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)
    assert set(baseline.keys()) == set(node_ids)
    for report in baseline.values():
        assert report["summary"]["num_test_images"] == len(data.test_idx)


def test_build_summary_has_appendix_a1_fields(tmp_path, tomato_scoped_config):
    data, stage1_dir, node_ids = _run_stage1(tmp_path, tomato_scoped_config)
    baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)

    # hand-build one fake round_summary matching the real shape, to test
    # the summary builder in isolation from the full round loop
    fake_scores = {
        node_id: {
            "collective": {"test": baseline[node_id]},
            "local_only_control": {"test": baseline[node_id]},
        }
        for node_id in node_ids
    }
    round_summaries = [
        {
            "round": 1,
            "total_bytes_exchanged": 100,
            "energy": {"total_compute_energy_kwh": 0.0},
            "per_node_scores": fake_scores,
        }
    ]
    summary = build_tomato_knowledge_transfer_summary(tomato_scoped_config, baseline, round_summaries, data)
    for key in (
        "node_count",
        "data_split",
        "local_only_budget",
        "collective_budget",
        "fairness_exception_reason",
        "test_set_scope",
        "rounds_run",
        "cumulative_bytes_exchanged",
        "cumulative_energy_kwh",
        "delta_g_formula",
        "per_node_scores",
        "macro_avg_and_worst_node",
        "collaboration_gain_per_disease",
        "limitation_note",
    ):
        assert key in summary
    assert summary["node_count"] == 3
    assert summary["data_split"]["dirichlet_alpha"] == 0.3
    assert "global" in summary["test_set_scope"]
