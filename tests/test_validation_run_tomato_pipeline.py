# tests/test_validation_run_tomato_pipeline.py
from __future__ import annotations

import json

import pytest

from src.validation.run_tomato_pipeline import run_evaluate_stage, run_export_stage, run_train_stage
from src.validation.tomato_mesh_dataset import prepare_tomato_mesh_data


def test_run_export_stage_raises_clear_error_when_train_stage_not_run(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="train"):
        run_export_stage(data, "node_0", output_dir)


def test_run_evaluate_stage_raises_clear_error_when_export_stage_not_run(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="export"):
        run_evaluate_stage(data, "node_0", output_dir)


def test_full_pipeline_produces_a_well_formed_report_per_node(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)

    for node_id in sorted(data.per_node.keys()):
        output_dir = tmp_path / node_id
        run_train_stage(data, node_id, output_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, output_dir)
        run_evaluate_stage(data, node_id, output_dir)

        report = json.loads((output_dir / "report.json").read_text())
        assert "summary" in report and "results" in report
        assert report["summary"]["num_test_images"] == len(report["results"])
        assert report["summary"]["num_test_images"] == len(data.test_idx)

        classes = json.loads((output_dir / "classes.json").read_text())
        assert classes["crop_classes"] == ["Tomato"]
        assert len(classes["disease_classes"]) == 10
        assert classes["test_idx"] == data.test_idx
