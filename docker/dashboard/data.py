"""Pure helpers for the dashboard -- deliberately separated from Streamlit
rendering (docker/dashboard/app.py) so this logic is unit-testable without
a running Streamlit session or real HTTP calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


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


SCENARIOS = ["full_run", "class_addition", "disconnection", "distribution_shift"]


def scenario_paths(energy_dir: str, scenario: str, num_nodes: int = 3) -> dict:
    """All namespaced paths for one scenario's data, rooted at energy_dir
    (e.g. "/energy"). Mirrors exactly what docker-compose.yml's
    ${SCENARIO}-templated ENERGY_DB/LOG_FILE env vars point node/coordinator
    containers at, so the dashboard reads from the same place they write to.
    """
    base = f"{energy_dir}/{scenario}"
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    return {
        "merged_db": f"{base}/merged.db",
        "node_dbs": {node_id: f"{base}/{node_id}.db" for node_id in node_ids},
        "status_path": f"{base}/status.json",
        "log_paths": {**{node_id: f"{base}/{node_id}.log" for node_id in node_ids}, "coordinator": f"{base}/coordinator.log"},
    }


def read_log_file(path: str) -> list:
    """Reads a JSONL activity log written by NodeRunner/CoordinatorRunner's
    _log(). Returns [] if the file doesn't exist yet (scenario never run) --
    same "no rows" convention as sqlite_store.read_all for a missing db.
    """
    if not Path(path).exists():
        return []
    entries = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A node/coordinator killed mid-write (e.g. docker compose
                # stop's timeout, hit on every scenario switch) can leave a
                # truncated final line. Skip it rather than taking down the
                # whole page render over one malformed line.
                continue
    return entries


def _cfg_get(cfg, key: str, default=None):
    """cfg may be a real src.config.Config or a plain dict keyed by dotted
    path (what the unit tests pass) -- both expose a compatible
    .get(key, default) method, so no branching is needed.
    """
    return cfg.get(key, default)


def build_fairness_disclosure(rows: list, cfg) -> dict:
    """Appendix A.1's Collaboration-Gain Fairness Disclosure Table, computed
    from the last round each node reported. ΔG per node/metric is
    mesh_accuracy - baseline_accuracy; macro_avg is the mean of every
    node's own (crop+disease)-averaged ΔG, worst_node is the minimum of
    those per-node averages -- reported separately per §4.3's "macro
    average and the worst-node score (not just the best or the overall
    average)".
    """
    node_ids = sorted({r["node_id"] for r in rows})
    per_node_scores = {}
    per_node_deltas = []
    rounds_used = {}
    for node_id in node_ids:
        node_rows = [r for r in rows if r["node_id"] == node_id]
        # "Last round" must mean the last COMPLETE round: baseline_* is
        # written at /round/start but crop_accuracy/disease_accuracy only
        # land at /round/gather, so if the last-recorded round is still in
        # progress (or that node's /round/gather failed/timed out), its row
        # has crop_accuracy/disease_accuracy still None. Filter to rows
        # where both are present before taking the max round_idx, so a
        # trailing in-progress/failed row never silently drops a node from
        # per_node_deltas without any trace in the payload.
        complete_rows = [r for r in node_rows if r.get("crop_accuracy") is not None and r.get("disease_accuracy") is not None]
        if not complete_rows:
            # No complete round recorded for this node yet (e.g. its most
            # recent /round/gather never landed) -- flag it via rounds_used
            # instead of silently synthesizing a score from an incomplete row.
            rounds_used[node_id] = None
            continue
        last = max(complete_rows, key=lambda r: r["round_idx"])
        rounds_used[node_id] = last["round_idx"]
        mesh = {"crop_accuracy": last.get("crop_accuracy"), "disease_accuracy": last.get("disease_accuracy")}
        baseline = {
            "crop_accuracy": last.get("baseline_crop_accuracy"),
            "disease_accuracy": last.get("baseline_disease_accuracy"),
        }
        per_node_scores[node_id] = {"mesh": mesh, "baseline": baseline}
        deltas = [mesh[m] - baseline[m] for m in ("crop_accuracy", "disease_accuracy")
                  if mesh[m] is not None and baseline[m] is not None]
        if deltas:
            per_node_deltas.append(sum(deltas) / len(deltas))

    macro_avg = sum(per_node_deltas) / len(per_node_deltas) if per_node_deltas else None
    worst_node = min(per_node_deltas) if per_node_deltas else None

    local_epochs = _cfg_get(cfg, "training.local_epochs_per_round")
    lr = _cfg_get(cfg, "training.lr")
    distill_epochs = _cfg_get(cfg, "training.distill_epochs_per_round")
    distill_lr = _cfg_get(cfg, "training.distill_lr")

    return {
        "task_metric": "crop_accuracy, disease_accuracy",
        "node_count": len(node_ids),
        "data_split": f"per-node held-out test split, test_fraction={_cfg_get(cfg, 'data.test_fraction')}",
        "non_iid_description": f"non_iid_strategy={_cfg_get(cfg, 'data.non_iid_strategy')}",
        "test_set_scope": "local test (per-node held-out split; test samples never enter any training set)",
        "local_only_budget": {"local_epochs_per_round": local_epochs, "lr": lr},
        "collective_budget": {
            "local_epochs_per_round": local_epochs, "lr": lr,
            "distill_epochs_per_round": distill_epochs, "distill_lr": distill_lr,
        },
        "per_node_scores": per_node_scores,
        "macro_avg_and_worst_node": {"macro_avg": macro_avg, "worst_node": worst_node},
        "rounds_used": rounds_used,
        "delta_g_formula": (
            "per-node delta = mean(mesh_accuracy - baseline_accuracy) over {crop_accuracy, disease_accuracy}, "
            "using each node's own last completed round; macro_avg is the mean across nodes, "
            "worst_node is the minimum across nodes"
        ),
    }


def build_export_payload(scenario: str, rows: list, transfers: list, status: dict | None, cfg) -> dict:
    """The single export schema shared by every tab (spec §6) -- extends
    the existing final_result.json shape (status/nodes/knowledge_transfers)
    with the fairness disclosure table. `complete` mirrors is_run_complete
    so a partial/crashed run's export is never mistaken for a finished one.
    """
    base = build_final_result_payload(rows, transfers, status)
    return {
        "scenario": scenario,
        "target_node": None,
        "perturbation_applied": False,
        "complete": is_run_complete(status),
        **base,
        "fairness_disclosure": build_fairness_disclosure(rows, cfg),
    }
