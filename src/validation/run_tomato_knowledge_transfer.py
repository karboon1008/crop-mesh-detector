"""Stage 2 for the Tomato Dirichlet-mesh pipeline: knowledge-transfer
rounds on top of the 3 per-node checkpoints stage 1
(run_tomato_pipeline.py) produces, evaluated against the single GLOBAL
held-out test set (not a per-node/cross-node union), plus the
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.
Aggregation mechanics (run_kt_round) are reused unchanged from
run_knowledge_transfer.py -- this pipeline changes the data split, not
the aggregation algorithm.

    python -m src.validation.run_tomato_knowledge_transfer --rounds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import make_subset
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.node import Node
from src.models.factory import build_model
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.run_knowledge_transfer import _build_per_node_energy_breakdown, run_kt_round
from src.validation.run_tomato_pipeline import node_output_dir
from src.validation.tomato_mesh_dataset import TomatoLabelMap, TomatoMeshData, prepare_tomato_mesh_data

MODEL_NAME = "mobilenet_v3_small"


def export_and_evaluate_tomato(
    node: Node,
    label_map: TomatoLabelMap,
    image_size: int,
    eval_base,
    test_idx: list[int],
    node_dir: Path,
    keep_onnx: bool,
) -> dict:
    """Same shape as run_knowledge_transfer.export_and_evaluate, but
    against the ONE shared global test set: report.json = {"test": ...}
    rather than {"local": ..., "cross_node": ...}, since there is no
    per-node test split to distinguish from a cross-node union here.
    """
    node_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = node_dir / "checkpoint.pt"
    torch.save(node.model.state_dict(), checkpoint_path)

    onnx_path = export_checkpoint(
        checkpoint_path, label_map.crop_classes, label_map.disease_classes, image_size, node_dir
    )
    manifest = json.loads((node_dir / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(onnx_path))

    scratch = node_dir / "_scratch_report.json"
    test_report = run_evaluation(session, eval_base, test_idx, manifest, MODEL_NAME, node.node_id, scratch)
    scratch.unlink(missing_ok=True)

    combined = {"test": test_report}
    (node_dir / "report.json").write_text(json.dumps(combined, indent=2))

    if not keep_onnx:
        onnx_path.unlink(missing_ok=True)
        (node_dir / "manifest.json").unlink(missing_ok=True)

    return combined


def _load_tomato_node_from_checkpoint(
    checkpoint_path: Path, data: TomatoMeshData, node_id: str, batch_size: int
) -> Node:
    model = build_model(
        MODEL_NAME, len(data.label_map.crop_classes), len(data.label_map.disease_classes), pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    train_idx = data.per_node[node_id]["train_idx"]
    train_loader = DataLoader(
        make_subset(data.train_base, train_idx), batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(make_subset(data.eval_base, data.test_idx), batch_size=batch_size, shuffle=False)
    return Node(node_id, model, train_loader, test_loader, device="cpu")


def evaluate_tomato_round0_baseline(data: TomatoMeshData, stage1_dir: Path, node_ids: list[str]) -> dict[str, dict]:
    """Evaluates each node's ALREADY-EXPORTED stage-1 model.onnx (no
    re-export) against the shared global test set -- this is round 0's
    baseline for the per-class collaboration-gain table.
    """
    baseline: dict[str, dict] = {}
    for node_id in node_ids:
        node_dir = node_output_dir(stage1_dir, node_id)
        manifest = json.loads((node_dir / "manifest.json").read_text())
        session = onnxruntime.InferenceSession(str(node_dir / "model.onnx"))
        scratch = node_dir / "_round0_scratch.json"
        report = run_evaluation(session, data.eval_base, data.test_idx, manifest, MODEL_NAME, node_id, scratch)
        scratch.unlink(missing_ok=True)
        baseline[node_id] = report
    return baseline


def _per_class_gain_table_tomato(
    round0_baseline: dict[str, dict],
    final_round_scores: dict[str, dict],
    disease_classes: list[str],
    partition_diagnostics: dict[str, dict],
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for node_id, baseline_report in round0_baseline.items():
        table[node_id] = {}
        collective_summary = final_round_scores[node_id]["collective"]["test"]["summary"]
        control_summary = final_round_scores[node_id]["local_only_control"]["test"]["summary"]
        low_rep = set(partition_diagnostics[node_id]["low_representation_classes"])
        for disease_name in disease_classes:
            round0_acc = baseline_report["summary"]["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            collective_acc = collective_summary["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            control_acc = control_summary["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            table[node_id][disease_name] = {
                "round_0_accuracy": round0_acc,
                "round_N_collective_accuracy": collective_acc,
                "round_N_local_only_control_accuracy": control_acc,
                "gain_vs_round0": collective_acc - round0_acc,
                "gain_vs_local_only_control": collective_acc - control_acc,
                "low_representation_for_this_node": disease_name in low_rep,
            }
    return table


def run_tomato_round_with_io(
    round_idx: int,
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    data: TomatoMeshData,
    cfg: Config,
    tracker: ComputeEnergyTracker,
    comm_estimator,
    round_dir: Path,
) -> dict:
    round_dir.mkdir(parents=True, exist_ok=True)

    kt_result = run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
        distill_lr=cfg.get("training.distill_lr", 0.0005),
        proto_weight=cfg.get("training.proto_weight", 0.5),
        kd_weight=cfg.get("training.kd_weight", 0.5),
        temperature=cfg.get("training.kd_temperature", 2.0),
        crop_kd_weight=cfg.get("training.crop_kd_weight", None),
        tracker=tracker,
        round_idx=round_idx,
    )

    per_node_scores: dict[str, dict] = {}
    for node_id in nodes:
        collective_report = export_and_evaluate_tomato(
            nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            data.test_idx,
            round_dir / node_id,
            keep_onnx=True,
        )
        control_report = export_and_evaluate_tomato(
            control_nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            data.test_idx,
            round_dir / node_id / "local_only_control",
            keep_onnx=False,
        )
        per_node_scores[node_id] = {"collective": collective_report, "local_only_control": control_report}

    comm_estimate = comm_estimator.estimate_all_radios(kt_result["total_bytes_exchanged"])
    energy_summary = tracker.summary()
    per_node_energy, per_node_local_only_control_energy = _build_per_node_energy_breakdown(
        tracker.log, nodes.keys(), round_idx
    )
    energy_summary = {
        **energy_summary,
        "per_node": per_node_energy,
        "per_node_local_only_control": per_node_local_only_control_energy,
    }

    summary = {
        "round": round_idx,
        "per_node_distill_loss": kt_result["per_node_distill_loss"],
        "per_node_bytes_sent": kt_result["per_node_bytes_sent"],
        "total_bytes_exchanged": kt_result["total_bytes_exchanged"],
        "energy": energy_summary,
        "communication_estimate": comm_estimate,
        "per_node_scores": per_node_scores,
    }
    (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _describe_data_split_strength(data: TomatoMeshData) -> str:
    """Derives the data_split.strength disclosure string from the ACTUAL
    partition_diagnostics rather than asserting full class coverage as a
    constant -- this feeds knowledge_transfer_summary.json, an Appendix
    A.1 research-disclosure artifact, so it must never overstate what the
    real partition achieved. Against real config/data, some nodes can
    have zero training samples for some canonical classes even though the
    partition method (Dirichlet) is proportional rather than deliberately
    exclusionary.
    """
    disease_classes = data.label_map.disease_classes
    num_classes = len(disease_classes)
    min_classes_present = min(
        diagnostics["num_classes_present"] for diagnostics in data.partition_diagnostics.values()
    )
    if min_classes_present >= num_classes:
        return (
            "every node has some training exposure to all "
            f"{num_classes} canonical disease classes; skew is proportional (Dirichlet-drawn), "
            "not exclusionary"
        )

    missing_by_node = {
        node_id: [
            name for name, count in diagnostics["per_class_counts"].items() if count == 0
        ]
        for node_id, diagnostics in data.partition_diagnostics.items()
    }
    missing_by_node = {node_id: names for node_id, names in missing_by_node.items() if names}
    return (
        "skew is proportional (Dirichlet-drawn), but NOT every node has training exposure to all "
        f"{num_classes} canonical disease classes: at least one node has zero training samples for "
        f"some classes -- missing classes by node: {missing_by_node}"
    )


def build_tomato_knowledge_transfer_summary(
    cfg: Config, round0_baseline: dict[str, dict], round_summaries: list[dict], data: TomatoMeshData
) -> dict:
    final_round = round_summaries[-1]
    final_scores = final_round["per_node_scores"]

    collective_evals = {nid: s["collective"]["test"]["summary"] for nid, s in final_scores.items()}
    control_evals = {nid: s["local_only_control"]["test"]["summary"] for nid, s in final_scores.items()}
    gain = compute_collaboration_gain(
        {
            nid: {"crop_accuracy": s["crop_accuracy"], "disease_accuracy": s["disease_accuracy"]}
            for nid, s in collective_evals.items()
        },
        {
            nid: {"crop_accuracy": s["crop_accuracy"], "disease_accuracy": s["disease_accuracy"]}
            for nid, s in control_evals.items()
        },
    )

    dirichlet_alpha = cfg.get("tomato_mesh.dirichlet_alpha", 0.3)
    cumulative_bytes = sum(r["total_bytes_exchanged"] for r in round_summaries)
    cumulative_energy = round_summaries[-1]["energy"]["total_compute_energy_kwh"]

    return {
        "node_count": len(data.per_node),
        "data_split": {
            "strategy": (
                f"Dirichlet label-skew partition (alpha={dirichlet_alpha}) over a merged, "
                "deduplicated multi-source Tomato pool (PlantVillage, PlantDoc, PlantWild v1+v2), "
                "with a global test split carved out before partitioning"
            ),
            "strength": _describe_data_split_strength(data),
            "sources": ["PlantVillage", "PlantDoc", "PlantWild_v1", "PlantWild_v2"],
            "dirichlet_alpha": dirichlet_alpha,
            "partition_diagnostics": data.partition_diagnostics,
        },
        "local_only_budget": {
            "epochs_per_round": cfg.get("training.distill_epochs_per_round", 1),
            "lr": cfg.get("training.distill_lr", 0.0005),
            "rounds": len(round_summaries),
        },
        "collective_budget": {
            "distill_epochs_per_round": cfg.get("training.distill_epochs_per_round", 1),
            "lr": cfg.get("training.distill_lr", 0.0005),
            "kd_weight": cfg.get("training.kd_weight", 0.5),
            "proto_weight": cfg.get("training.proto_weight", 0.5),
            "rounds": len(round_summaries),
        },
        "fairness_exception_reason": (
            "Local-supervised training budgets are aligned round-for-round between the collective "
            "and local-only-control arms. The one intentional, disclosed asymmetry is the "
            "collective arm's extra KD phase over the shared probe set — that is the mechanism "
            "under test, not an unaligned budget."
        ),
        "test_set_scope": (
            "single global held-out test set, shared across all nodes, carved out before "
            "Dirichlet partitioning; test samples never enter any training set"
        ),
        "rounds_run": len(round_summaries),
        "cumulative_bytes_exchanged": cumulative_bytes,
        "cumulative_energy_kwh": cumulative_energy,
        "delta_g_formula": "Score(collective, round_N) - Score(local_only_control, round_N), per metric per node",
        "per_node_scores": {
            nid: {"collective": collective_evals[nid], "local_only_control": control_evals[nid]}
            for nid in collective_evals
        },
        "macro_avg_and_worst_node": {
            "macro_gain": gain["macro_gain"],
            "worst_node_gain": gain["worst_node_gain"],
        },
        "collaboration_gain_per_disease": _per_class_gain_table_tomato(
            round0_baseline, final_scores, data.label_map.disease_classes, data.partition_diagnostics
        ),
        "limitation_note": (
            "Cross-class-representation gain above comes from probe-set logit distillation only, "
            "not prototype alignment — prototype exchange only reinforces classes a node already "
            "has local samples for, since a node's proto_loss only runs over its own local batches."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/tomato_mesh")
    parser.add_argument("--rounds", type=int, default=None, help="Number of knowledge-transfer rounds (1-20)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    rounds = args.rounds if args.rounds is not None else cfg.get("tomato_mesh.rounds", 5)
    if not (1 <= rounds <= 20):
        parser.error(f"--rounds must be between 1 and 20, got {rounds}")

    stage1_dir = Path(args.output_dir)
    data = prepare_tomato_mesh_data(cfg)
    node_ids = sorted(data.per_node.keys())

    for node_id in node_ids:
        node_dir = node_output_dir(stage1_dir, node_id)
        for required_name in ("checkpoint.pt", "model.onnx", "manifest.json", "classes.json"):
            required_path = node_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(
                    f"{required_path} not found — run 'python -m src.validation.run_tomato_pipeline' first."
                )

    top_level_classes_path = stage1_dir / "classes.json"
    if not top_level_classes_path.exists():
        raise FileNotFoundError(f"{top_level_classes_path} not found — run run_tomato_pipeline.py first.")
    top_level_classes = json.loads(top_level_classes_path.read_text())
    if top_level_classes.get("test_idx") != data.test_idx:
        raise ValueError(
            f"{top_level_classes_path}: the global test split persisted by stage 1 does not match "
            f"the split prepare_tomato_mesh_data(cfg) produces now. Stage 1 and stage 2 must be run "
            f"against the same config and the same source data — otherwise stage 2 would silently "
            f"evaluate checkpoints on images they were trained on."
        )
    for node_id in node_ids:
        classes_path = node_output_dir(stage1_dir, node_id) / "classes.json"
        stage1_classes = json.loads(classes_path.read_text())
        if stage1_classes.get("train_idx") != data.per_node[node_id]["train_idx"]:
            raise ValueError(
                f"{node_id}: the train split persisted in {classes_path} by stage 1 "
                f"(run_tomato_pipeline.py) does not match the split prepare_tomato_mesh_data(cfg) "
                f"produces now. Stage 1 and stage 2 must be run against the same config and the "
                f"same source data — otherwise stage 2 would silently evaluate this node's "
                f"checkpoint on images it was trained on."
            )

    print("=== round 0 baseline (stage 1 checkpoints, global test set) ===")
    round0_baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)

    kt_dir = stage1_dir / "knowledge_transfer"
    kt_dir.mkdir(parents=True, exist_ok=True)
    (kt_dir / "round_0_baseline.json").write_text(json.dumps(round0_baseline, indent=2))

    nodes = {
        node_id: _load_tomato_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in node_ids
    }
    control_nodes = {
        node_id: _load_tomato_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in node_ids
    }
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=args.batch_size, shuffle=False)

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", False),
        output_dir=kt_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}), cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125)
    )

    round_summaries = []
    for round_idx in range(1, rounds + 1):
        print(f"=== knowledge-transfer round {round_idx}/{rounds} ===")
        round_dir = kt_dir / f"round_{round_idx}"
        summary = run_tomato_round_with_io(
            round_idx, nodes, control_nodes, probe_loader, data, cfg, tracker, comm_estimator, round_dir
        )
        round_summaries.append(summary)

    kt_summary = build_tomato_knowledge_transfer_summary(cfg, round0_baseline, round_summaries, data)
    (kt_dir / "knowledge_transfer_summary.json").write_text(json.dumps(kt_summary, indent=2))
    print(f"Done. Knowledge-transfer outputs in {kt_dir}/")


if __name__ == "__main__":
    main()
