# tests/test_validation_run_tomato_knowledge_transfer.py
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.validation.run_tomato_knowledge_transfer import (
    _describe_data_split_strength,
    build_tomato_knowledge_transfer_summary,
    evaluate_tomato_round0_baseline,
    main,
)
from src.validation.run_tomato_pipeline import node_output_dir, run_evaluate_stage, run_export_stage, run_train_stage
from src.validation.tomato_mesh_dataset import build_tomato_label_map, prepare_tomato_mesh_data


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


def test_data_split_strength_reports_full_coverage_when_every_node_has_all_classes():
    label_map = build_tomato_label_map()
    num_classes = len(label_map.disease_classes)
    data = SimpleNamespace(
        label_map=label_map,
        partition_diagnostics={
            "node_0": {
                "num_classes_present": num_classes,
                "per_class_counts": {name: 1 for name in label_map.disease_classes},
            },
            "node_1": {
                "num_classes_present": num_classes,
                "per_class_counts": {name: 1 for name in label_map.disease_classes},
            },
        },
    )
    strength = _describe_data_split_strength(data)
    assert "every node has some training exposure to all" in strength
    assert "not exclusionary" in strength


def test_data_split_strength_names_missing_classes_when_a_node_lacks_full_coverage():
    """Regression for the real shipped-config finding: node_1 actually has
    only 8/10 classes present (Bacterial_spot and
    Tomato_Yellow_Leaf_Curl_Virus at zero samples). The disclosure string
    must name the affected node and classes rather than claim full
    coverage.
    """
    label_map = build_tomato_label_map()
    per_class_counts_full = {name: 5 for name in label_map.disease_classes}
    per_class_counts_missing = dict(per_class_counts_full)
    per_class_counts_missing["Bacterial_spot"] = 0
    per_class_counts_missing["Tomato_Yellow_Leaf_Curl_Virus"] = 0

    data = SimpleNamespace(
        label_map=label_map,
        partition_diagnostics={
            "node_0": {
                "num_classes_present": len(label_map.disease_classes),
                "per_class_counts": per_class_counts_full,
            },
            "node_1": {
                "num_classes_present": len(label_map.disease_classes) - 2,
                "per_class_counts": per_class_counts_missing,
            },
        },
    )
    strength = _describe_data_split_strength(data)
    assert "node_1" in strength
    assert "Bacterial_spot" in strength
    assert "Tomato_Yellow_Leaf_Curl_Virus" in strength
    assert "every node has some training exposure to all" not in strength


def test_main_raises_clear_error_when_stage1_split_does_not_match_stage2_recompute(
    tmp_path, tomato_scoped_config, monkeypatch
):
    """If stage 1's persisted classes.json (train_idx/test_idx) doesn't
    match what prepare_tomato_mesh_data(cfg) recomputes in stage 2 -- e.g.
    because data.seed/test_fraction/etc. or source data changed between
    the two runs -- main() must fail loudly rather than silently
    evaluating a node's stage-1 checkpoint on images it was trained on.
    Mirrors test_validation_run_knowledge_transfer.py's
    test_main_raises_clear_error_when_stage1_split_does_not_match_stage2_recompute
    for the corn pipeline.
    """
    cfg = tomato_scoped_config
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    _data, stage1_dir, node_ids = _run_stage1(tmp_path, cfg)
    assert "node_1" in node_ids

    # Tamper with node_1's persisted train_idx so it disagrees with what
    # prepare_tomato_mesh_data(cfg) will recompute inside main().
    node_1_dir = node_output_dir(stage1_dir, "node_1")
    classes_path = node_1_dir / "classes.json"
    classes = json.loads(classes_path.read_text())
    classes["train_idx"] = classes["train_idx"] + [999999]  # deliberately wrong
    classes_path.write_text(json.dumps(classes))

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_tomato_knowledge_transfer",
            "--config",
            str(config_path),
            "--output-dir",
            str(stage1_dir),
            "--rounds",
            "1",
        ],
    )

    with pytest.raises(ValueError, match="node_1"):
        main()
