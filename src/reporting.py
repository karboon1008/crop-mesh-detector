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
            kd_weight = rl.get("per_node_kd_weight", {}).get(node_id)
            if kd_weight is not None:
                row["kd_weight"] = kd_weight
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
    precision/recall/f1/support out of detail.{crop,disease}.per_class --
    the confusion matrix itself still isn't representable as flat rows,
    but the derived per-class metrics it's built from are. No "accuracy"
    column: for one class in a multi-class confusion matrix, accuracy and
    recall are the same number.

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
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_ids = sorted({r["node_id"] for r in rows})
    accuracy_keys = [k for k in rows[0] if k.endswith("_accuracy")]
    loss_keys = [k for k in rows[0] if k.endswith("_loss")]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    accuracy_colors = {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(accuracy_keys)}
    loss_colors = {k: color_cycle[i % len(color_cycle)] for i, k in enumerate(loss_keys)}

    fig, axes = plt.subplots(
        len(node_ids), 2, figsize=(11, 3 * len(node_ids)), sharex=True, squeeze=False,
    )
    for row_idx, node_id in enumerate(node_ids):
        node_rows = sorted((r for r in rows if r["node_id"] == node_id), key=lambda r: r["round_idx"])
        x = [r["round_idx"] for r in node_rows]

        ax_acc, ax_loss = axes[row_idx]
        for key in accuracy_keys:
            y = [r.get(key) for r in node_rows]
            if any(v is not None for v in y):
                ax_acc.plot(x, y, marker="o", color=accuracy_colors[key], label=key)
        for key in loss_keys:
            y = [r.get(key) for r in node_rows]
            if any(v is not None for v in y):
                ax_loss.plot(x, y, marker="o", color=loss_colors[key], label=key)

        ax_acc.set_ylabel(f"{node_id}\naccuracy")
        ax_loss.set_ylabel("loss")
        ax_acc.legend(fontsize=6, loc="best")
        ax_loss.legend(fontsize=6, loc="best")
        if row_idx == 0:
            ax_acc.set_title("accuracy")
            ax_loss.set_title("loss")

    axes[-1][0].set_xlabel("round")
    axes[-1][1].set_xlabel("round")
    fig.suptitle(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
