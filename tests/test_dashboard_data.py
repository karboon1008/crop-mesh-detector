from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "dashboard"))

from data import build_status_rows  # noqa: E402


def test_build_status_rows_sorts_by_node_id():
    rows = build_status_rows({"node_2": True, "node_0": False, "node_1": True})
    assert rows == [
        {"node_id": "node_0", "online": False},
        {"node_id": "node_1", "online": True},
        {"node_id": "node_2", "online": True},
    ]


def test_build_status_rows_handles_empty_input():
    assert build_status_rows({}) == []
