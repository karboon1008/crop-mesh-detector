from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "dashboard"))

from data import (  # noqa: E402
    build_final_result_payload,
    build_log_lines,
    build_status_rows,
    is_run_complete,
    merge_transfer_rows,
    rows_for_node,
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
