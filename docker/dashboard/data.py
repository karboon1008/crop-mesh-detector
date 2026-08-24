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


def merge_round_rows(round_metrics_lists: list) -> list:
    """Concatenates each node's own round_metrics rows (one list per node
    db) into a single combined view, sorted the same way
    sqlite_store.read_all sorts a single db (round_idx, then node_id).
    There is no coordinator-owned merged db to read instead -- every node's
    round history lives only in that node's own db, and the dashboard is
    the one place that ever combines them, purely for display.
    """
    merged = [row for rows in round_metrics_lists for row in rows]
    return sorted(merged, key=lambda r: (r["round_idx"], r["node_id"]))


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


def build_fairness_disclosure(rows: list, cfg: dict) -> dict:
    """Collaboration-Gain Fairness Disclosure table (Appendix A.1), read off
    each node's most recent round row -- the same macro_avg / worst_node
    framing the headline report uses in §7, but computed live from whatever
    rows the dashboard has polled so far.
    """
    latest_by_node: dict = {}
    for row in rows:
        latest_by_node[row["node_id"]] = row  # rows are in round order, so last write wins
    per_node_scores = {}
    per_node_avg_delta = {}
    for node_id, row in latest_by_node.items():
        mesh = {"crop_accuracy": row["crop_accuracy"], "disease_accuracy": row["disease_accuracy"]}
        baseline = {
            "crop_accuracy": row["baseline_crop_accuracy"],
            "disease_accuracy": row["baseline_disease_accuracy"],
        }
        per_node_scores[node_id] = {"mesh": mesh, "baseline": baseline}
        crop_delta = mesh["crop_accuracy"] - baseline["crop_accuracy"]
        disease_delta = mesh["disease_accuracy"] - baseline["disease_accuracy"]
        per_node_avg_delta[node_id] = (crop_delta + disease_delta) / 2
    deltas = list(per_node_avg_delta.values())
    return {
        "node_count": len(latest_by_node),
        "per_node_scores": per_node_scores,
        "macro_avg_and_worst_node": {
            "macro_avg": sum(deltas) / len(deltas) if deltas else 0.0,
            "worst_node": min(deltas) if deltas else 0.0,
        },
        "local_only_budget": {
            "epochs": cfg.get("training.local_epochs_per_round"),
            "lr": cfg.get("training.lr"),
        },
        "collective_budget": {
            "epochs": cfg.get("training.distill_epochs_per_round"),
            "lr": cfg.get("training.distill_lr"),
        },
        "non_iid_description": cfg.get("data.non_iid_strategy"),
        "test_set_scope": cfg.get("data.test_fraction"),
        "delta_g_formula": "mean(mesh.crop_accuracy - baseline.crop_accuracy, "
        "mesh.disease_accuracy - baseline.disease_accuracy), per node's latest round",
    }


def build_export_payload(scenario: str, rows: list, transfers: list, status: dict | None, cfg: dict) -> dict:
    """Superset of build_final_result_payload for the scenario dashboard's
    JSON export button: adds the scenario name and the fairness-disclosure
    table alongside the per-node round history.
    """
    payload = build_final_result_payload(rows, transfers, status)
    payload["scenario"] = scenario
    payload["complete"] = is_run_complete(status)
    payload["fairness_disclosure"] = build_fairness_disclosure(rows, cfg)
    return payload


def scenario_paths(energy_root: str, scenario: str, num_nodes: int) -> dict:
    """Per-scenario file layout under the shared energy volume: one sqlite
    db per node plus a merged one, a status marker, and one log file per
    node/coordinator -- so multiple scenarios can run without overwriting
    each other's evidence.
    """
    base = f"{energy_root}/{scenario}"
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    log_paths = {node_id: f"{base}/{node_id}.log" for node_id in node_ids}
    log_paths["coordinator"] = f"{base}/coordinator.log"
    return {
        "merged_db": f"{base}/merged.db",
        "node_dbs": {node_id: f"{base}/{node_id}.db" for node_id in node_ids},
        "status_path": f"{base}/status.json",
        "log_paths": log_paths,
    }


def read_log_file(path: str) -> list:
    """Reads a coordinator/node activity log (one JSON object per line, the
    same format sqlite_store's log writer produces) -- [] if the file
    doesn't exist yet, since a scenario's log appears only once it starts.
    """
    try:
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def to_json_str(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
