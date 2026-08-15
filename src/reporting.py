"""Round-over-round reporting: flattens the nested per-round eval/loss
dicts (RoundLog, ScenarioRoundRecord) into one row per (round, node), so a
weight-tuning question like "is kd_weight=0.5 helping" is a glance at a
CSV/plot instead of a diff across N nested JSON blobs.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.evaluate import scalar_metrics


def flatten_eval(eval_: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in scalar_metrics(eval_).items()}


def build_round_log_rows(round_logs: list[dict]) -> list[dict]:
    """One row per (round, node) from outputs/round_logs_{arch}.json's
    "rounds" list (the RoundLog shape: per_node_train_loss,
    pre_distill_eval, per_node_distill_loss, per_node_eval).
    """
    rows = []
    for rl in round_logs:
        round_idx = rl["round_idx"]
        node_ids = sorted(set(rl.get("per_node_train_loss", {})) | set(rl.get("per_node_eval", {})))
        for node_id in node_ids:
            row = {"round_idx": round_idx, "node_id": node_id}
            train_loss = rl.get("per_node_train_loss", {}).get(node_id)
            if train_loss is not None:
                row["train_loss"] = train_loss
            row.update(rl.get("per_node_distill_loss", {}).get(node_id, {}))
            pre_eval = rl.get("pre_distill_eval", {}).get(node_id)
            if pre_eval is not None:
                row.update(flatten_eval(pre_eval, "pre_"))
            post_eval = rl.get("per_node_eval", {}).get(node_id)
            if post_eval is not None:
                row.update(flatten_eval(post_eval, "post_"))
            rows.append(row)
    return rows


def build_scenario_rows(records: list[dict]) -> list[dict]:
    """One row per (round, node) from a scenario report's "rounds" list
    (the ScenarioRoundRecord shape: baseline_eval, mesh_eval, plus the
    same loss fields as RoundLog for the mesh side).
    """
    rows = []
    for r in records:
        round_idx = r["round_idx"]
        node_ids = sorted(set(r.get("mesh_eval", {})) | set(r.get("baseline_eval", {})))
        for node_id in node_ids:
            row = {"round_idx": round_idx, "node_id": node_id}
            train_loss = r.get("per_node_train_loss", {}).get(node_id)
            if train_loss is not None:
                row["mesh_train_loss"] = train_loss
            row.update({f"mesh_{k}": v for k, v in r.get("per_node_distill_loss", {}).get(node_id, {}).items()})
            mesh_eval = r.get("mesh_eval", {}).get(node_id)
            if mesh_eval is not None:
                row.update(flatten_eval(mesh_eval, "mesh_"))
            baseline_eval = r.get("baseline_eval", {}).get(node_id)
            if baseline_eval is not None:
                row.update(flatten_eval(baseline_eval, "baseline_"))
            rows.append(row)
    return rows


def build_per_class_rows(rounds: list[dict], eval_specs: list[tuple[str, str]]) -> list[dict]:
    """One row per (round, node, phase, head, class), pulling per-class
    accuracy/precision/recall/f1/support out of detail.{crop,disease}.per_class
    -- the confusion matrix itself still isn't representable as flat rows,
    but the derived per-class metrics it's built from are.

    eval_specs is a list of (phase_label, dict_key) pairs identifying which
    eval dict(s) in each round dict to pull from, e.g.
    [("pre", "pre_distill_eval"), ("post", "per_node_eval")] for round logs,
    or [("mesh", "mesh_eval"), ("baseline", "baseline_eval")] for scenarios.
    """
    rows = []
    for r in rounds:
        round_idx = r["round_idx"]
        for phase, key in eval_specs:
            for node_id, ev in r.get(key, {}).items():
                for head, head_detail in ev.get("detail", {}).items():
                    for class_name, stats in head_detail["per_class"].items():
                        rows.append({
                            "round_idx": round_idx,
                            "node_id": node_id,
                            "phase": phase,
                            "head": head,
                            "class_name": class_name,
                            "accuracy": stats["accuracy"],
                            "precision": stats["precision"],
                            "recall": stats["recall"],
                            "f1": stats["f1"],
                            "support": stats["support"],
                        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict], output_path: Path, title: str) -> None:
    """One line per node per metric, over rounds: top panel accuracy-type
    metrics, bottom panel loss-type metrics. Reads whichever "*_accuracy"/
    "*_loss" columns are present, so the same function serves both the
    main sweep's rows (pre_/post_ prefixes) and a scenario's rows
    (mesh_/baseline_ prefixes) without change.
    """
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_ids = sorted({r["node_id"] for r in rows})
    accuracy_keys = [k for k in rows[0] if k.endswith("_accuracy")]
    loss_keys = [k for k in rows[0] if k.endswith("_loss")]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for node_id in node_ids:
        node_rows = sorted((r for r in rows if r["node_id"] == node_id), key=lambda r: r["round_idx"])
        x = [r["round_idx"] for r in node_rows]
        for key in accuracy_keys:
            y = [r.get(key) for r in node_rows]
            if any(v is not None for v in y):
                axes[0].plot(x, y, marker="o", label=f"{node_id}: {key}")
        for key in loss_keys:
            y = [r.get(key) for r in node_rows]
            if any(v is not None for v in y):
                axes[1].plot(x, y, marker="o", label=f"{node_id}: {key}")

    axes[0].set_ylabel("accuracy")
    axes[0].set_title(title)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("loss")
    axes[1].set_xlabel("round")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
