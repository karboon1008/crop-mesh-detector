"""Pure helpers for the dashboard -- deliberately separated from Streamlit
rendering (docker/dashboard/app.py) so this logic is unit-testable without
a running Streamlit session or real HTTP calls.
"""

from __future__ import annotations

import json
import time


def build_status_rows(node_health: dict) -> list:
    """node_health: {node_id: bool} -> sorted list of {"node_id", "online"}
    rows, so table row order is a guarantee rather than an accident of
    dict iteration order.
    """
    return [{"node_id": node_id, "online": online} for node_id, online in sorted(node_health.items())]


def build_log_lines(entries: list) -> list:
    """Per-node/coordinator activity log entries ({"ts": float, ...}) ->
    oldest-first rows with ts replaced by an "HH:MM:SS" string. Oldest-first
    (not reversed) so the dashboard's scrolling console panel reads like a
    real log file/terminal, with the newest line at the bottom.
    """
    out = []
    for entry in entries:
        row = {k: v for k, v in entry.items() if k != "ts"}
        row["time"] = time.strftime("%H:%M:%S", time.localtime(entry["ts"]))
        out.append(row)
    return out


def rows_for_node(rows: list, node_id: str) -> list:
    """Filters merged round_metrics rows down to one node, preserving order."""
    return [r for r in rows if r.get("node_id") == node_id]


def is_run_complete(status: dict | None) -> bool:
    return bool(status) and bool(status.get("all_rounds_complete"))


def merge_transfer_rows(transfer_lists: list) -> list:
    """Concatenates each node's own knowledge_transfers rows (one list per
    node db) into a single feed, newest first.
    """
    merged = [row for rows in transfer_lists for row in rows]
    return sorted(merged, key=lambda r: r["fetched_at"], reverse=True)


def build_final_result_payload(rows: list, transfers: list, status: dict | None) -> dict:
    """Shape of the combined "final result" JSON export: per-node round
    history, the knowledge-transfer feed, and the completion marker.
    """
    node_ids = sorted({r["node_id"] for r in rows})
    return {
        "status": status,
        "nodes": {node_id: rows_for_node(rows, node_id) for node_id in node_ids},
        "knowledge_transfers": transfers,
    }


def to_json_str(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
