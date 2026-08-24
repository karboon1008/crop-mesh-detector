from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "dashboard"))

from data import (  # noqa: E402
    build_export_payload,
    build_fairness_disclosure,
    build_final_result_payload,
    build_log_lines,
    build_status_rows,
    is_run_complete,
    merge_transfer_rows,
    read_log_file,
    rows_for_node,
    scenario_paths,
    to_json_str,
)


def test_build_status_rows_sorts_by_node_id():
    rows = build_status_rows({"node_2": True, "node_0": False, "node_1": True})
    assert rows == [
        {"node_id": "node_0", "online": False},
        {"node_id": "node_1", "online": True},
        {"node_id": "node_2", "online": True},
    ]


def test_build_status_rows_handles_empty_input():
    assert build_status_rows({}) == []


def test_build_log_lines_keeps_chronological_order_and_replaces_ts_with_time_string():
    entries = [
        {"ts": 1786930091.0, "stage": "round_start", "message": "a"},
        {"ts": 1786930100.0, "stage": "round_gather", "message": "b"},
    ]
    rows = build_log_lines(entries)
    assert [r["message"] for r in rows] == ["a", "b"]  # oldest first, newest last
    assert "ts" not in rows[0]
    assert "time" in rows[0]


def test_build_log_lines_handles_empty_input():
    assert build_log_lines([]) == []


def test_rows_for_node_filters_by_node_id():
    rows = [
        {"node_id": "node_0", "round_idx": 0},
        {"node_id": "node_1", "round_idx": 0},
        {"node_id": "node_0", "round_idx": 1},
    ]
    assert rows_for_node(rows, "node_0") == [
        {"node_id": "node_0", "round_idx": 0},
        {"node_id": "node_0", "round_idx": 1},
    ]


def test_is_run_complete_true_only_when_flag_set():
    assert is_run_complete({"all_rounds_complete": True}) is True
    assert is_run_complete({"all_rounds_complete": False}) is False
    assert is_run_complete(None) is False
    assert is_run_complete({}) is False


def test_merge_transfer_rows_concatenates_and_sorts_newest_first():
    per_node = [
        [{"fetched_at": "2026-08-17T00:00:00Z", "from_node": "node_1"}],
        [{"fetched_at": "2026-08-17T00:00:05Z", "from_node": "node_0"}],
    ]
    merged = merge_transfer_rows(per_node)
    assert [r["from_node"] for r in merged] == ["node_0", "node_1"]


def test_build_final_result_payload_groups_rows_by_node():
    rows = [
        {"node_id": "node_0", "round_idx": 0, "crop_accuracy": 0.9},
        {"node_id": "node_1", "round_idx": 0, "crop_accuracy": 0.8},
    ]
    payload = build_final_result_payload(rows, transfers=[], status={"all_rounds_complete": True})
    assert set(payload["nodes"]) == {"node_0", "node_1"}
    assert payload["nodes"]["node_0"] == [rows[0]]
    assert payload["status"]["all_rounds_complete"] is True


def test_to_json_str_round_trips():
    obj = {"a": 1, "b": [1, 2, 3]}
    assert json.loads(to_json_str(obj)) == obj


def test_scenario_paths_builds_per_node_and_shared_paths():
    paths = scenario_paths("/energy", "disconnection", num_nodes=2)
    assert paths["merged_db"] == "/energy/disconnection/merged.db"
    assert paths["node_dbs"] == {
        "node_0": "/energy/disconnection/node_0.db",
        "node_1": "/energy/disconnection/node_1.db",
    }
    assert paths["status_path"] == "/energy/disconnection/status.json"
    assert paths["log_paths"]["node_0"] == "/energy/disconnection/node_0.log"
    assert paths["log_paths"]["coordinator"] == "/energy/disconnection/coordinator.log"


def test_read_log_file_returns_empty_list_when_file_missing(tmp_path):
    assert read_log_file(str(tmp_path / "missing.log")) == []


def test_read_log_file_parses_one_json_object_per_line(tmp_path):
    path = tmp_path / "node_0.log"
    path.write_text('{"ts": 1.0, "stage": "a", "message": "x"}\n{"ts": 2.0, "stage": "b", "message": "y"}\n')
    entries = read_log_file(str(path))
    assert [e["message"] for e in entries] == ["x", "y"]


def test_build_fairness_disclosure_reports_macro_avg_and_worst_node():
    rows = [
        {"node_id": "node_0", "round_idx": 0, "crop_accuracy": 0.80, "disease_accuracy": 0.70,
         "baseline_crop_accuracy": 0.60, "baseline_disease_accuracy": 0.50},
        {"node_id": "node_1", "round_idx": 0, "crop_accuracy": 0.55, "disease_accuracy": 0.55,
         "baseline_crop_accuracy": 0.50, "baseline_disease_accuracy": 0.50},
    ]
    cfg = {
        "training.local_epochs_per_round": 1, "training.distill_epochs_per_round": 1,
        "training.lr": 0.001, "training.distill_lr": 0.0005,
        "data.non_iid_strategy": "manual", "data.test_fraction": 0.15,
    }
    table = build_fairness_disclosure(rows, cfg)
    assert table["node_count"] == 2
    # node_0 delta: crop +0.20, disease +0.20 -> avg 0.20; node_1: crop +0.05, disease +0.05 -> avg 0.05
    assert table["macro_avg_and_worst_node"]["macro_avg"] == pytest.approx(0.125)
    assert table["macro_avg_and_worst_node"]["worst_node"] == pytest.approx(0.05)
    assert table["per_node_scores"]["node_0"]["mesh"]["crop_accuracy"] == 0.80
    assert table["per_node_scores"]["node_0"]["baseline"]["crop_accuracy"] == 0.60
    assert "delta_g_formula" in table and table["delta_g_formula"]


def test_build_export_payload_includes_fairness_disclosure_and_scenario_name():
    rows = [{"node_id": "node_0", "round_idx": 0, "crop_accuracy": 0.8, "disease_accuracy": 0.7,
             "baseline_crop_accuracy": 0.6, "baseline_disease_accuracy": 0.5}]
    cfg = {"training.local_epochs_per_round": 1, "training.distill_epochs_per_round": 1,
           "training.lr": 0.001, "training.distill_lr": 0.0005,
           "data.non_iid_strategy": "manual", "data.test_fraction": 0.15}
    payload = build_export_payload("disconnection", rows, transfers=[], status=None, cfg=cfg)
    assert payload["scenario"] == "disconnection"
    assert payload["complete"] is False
    assert "fairness_disclosure" in payload
    assert payload["nodes"]["node_0"] == rows
