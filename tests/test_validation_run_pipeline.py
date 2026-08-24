from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.run_pipeline import run_evaluate_stage, run_export_stage, run_train_stage


def test_run_export_stage_raises_clear_error_when_train_stage_not_run(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="train"):
        run_export_stage(cfg, output_dir)


def test_run_evaluate_stage_raises_clear_error_when_export_stage_not_run(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="export"):
        run_evaluate_stage(cfg, output_dir)


def test_full_pipeline_produces_a_well_formed_report(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"

    run_train_stage(cfg, output_dir, epochs=2, pretrained=False)
    run_export_stage(cfg, output_dir)
    run_evaluate_stage(cfg, output_dir)

    report = json.loads((output_dir / "report.json").read_text())
    assert "summary" in report and "results" in report
    assert report["summary"]["num_test_images"] == len(report["results"])
    assert report["summary"]["num_test_images"] > 0
