"""Stage 2: 1-5 rounds of knowledge-transfer distillation on top of the
3 per-node checkpoints stage 1 (run_corn_pipeline.py) produces, plus a
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_knowledge_transfer --rounds 2
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import onnxruntime
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import Node
from src.validation.corn_mesh_dataset import CornLabelMap, CornMeshData
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint

MODEL_NAME = "mobilenet_v3_small"


def run_kt_round(
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    aggregation_method: str,
    trim_fraction: float,
    krum_neighbors: int,
    distill_epochs: int,
    distill_lr: float,
    proto_weight: float,
    kd_weight: float,
    temperature: float,
    tracker: ComputeEnergyTracker | None = None,
) -> dict:
    """One knowledge-transfer round: `nodes` distill toward their peers'
    consensus (no separate local_train call); `control_nodes` run the
    same-budget local_train with no exchange at all, for the Appendix
    A.1 fairness-aligned comparison. Both dicts are mutated in place
    (each Node's .model is updated); nothing is reloaded from disk here.
    """
    payloads = {node_id: node.compute_knowledge(probe_loader) for node_id, node in nodes.items()}
    per_node_bytes_sent = {node_id: payload.size_bytes() for node_id, payload in payloads.items()}
    total_bytes_exchanged = sum(per_node_bytes_sent.values()) * max(0, len(nodes) - 1)

    per_node_distill_loss: dict[str, dict[str, float]] = {}
    for node_id, node in nodes.items():
        peer_payloads = [p for nid, p in payloads.items() if nid != node_id]
        if not peer_payloads:
            continue  # single-node mesh: nothing to reconcile (mirrors mesh.py's run_round)
        consensus_prototypes = aggregate_prototypes(
            [p.prototypes for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_crop_logits = aggregate_logits(
            [p.crop_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_disease_logits = aggregate_logits(
            [p.disease_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        ctx = tracker.track(f"{node_id}_kt_distill") if tracker is not None else nullcontext()
        with ctx:
            per_node_distill_loss[node_id] = node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                probe_loader,
                epochs=distill_epochs,
                lr=distill_lr,
                proto_weight=proto_weight,
                kd_weight=kd_weight,
                temperature=temperature,
            )

    for node_id, node in control_nodes.items():
        ctx = tracker.track(f"{node_id}_local_only_control") if tracker is not None else nullcontext()
        with ctx:
            node.local_train(epochs=distill_epochs, lr=distill_lr)

    return {
        "per_node_distill_loss": per_node_distill_loss,
        "per_node_bytes_sent": per_node_bytes_sent,
        "total_bytes_exchanged": total_bytes_exchanged,
    }


def export_and_evaluate(
    node: Node,
    label_map: CornLabelMap,
    image_size: int,
    eval_base,
    local_test_idx: list[int],
    cross_node_idx: list[int],
    node_dir: Path,
    keep_onnx: bool,
) -> dict:
    """Saves node.model's current weights, exports to ONNX, evaluates
    against both the local held-out test set and the cross-node union
    set, and writes report.json = {"local": ..., "cross_node": ...}.
    When keep_onnx is False (the local-only control arm — not a
    deployment artifact), model.onnx/manifest.json are deleted again
    right after the transient evaluation session is built.
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
    local_report = run_evaluation(session, eval_base, local_test_idx, manifest, MODEL_NAME, node.node_id, scratch)
    cross_node_report = run_evaluation(
        session, eval_base, cross_node_idx, manifest, MODEL_NAME, node.node_id, scratch
    )
    scratch.unlink(missing_ok=True)

    combined = {"local": local_report, "cross_node": cross_node_report}
    (node_dir / "report.json").write_text(json.dumps(combined, indent=2))

    if not keep_onnx:
        onnx_path.unlink(missing_ok=True)
        (node_dir / "manifest.json").unlink(missing_ok=True)

    return combined


def run_round_with_io(
    round_idx: int,
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    data: CornMeshData,
    cross_node_idx: list[int],
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
        tracker=tracker,
    )

    per_node_scores: dict[str, dict] = {}
    for node_id in nodes:
        local_test_idx = data.per_node[node_id]["test_idx"]
        collective_report = export_and_evaluate(
            nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            local_test_idx,
            cross_node_idx,
            round_dir / node_id,
            keep_onnx=True,
        )
        control_report = export_and_evaluate(
            control_nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            local_test_idx,
            cross_node_idx,
            round_dir / node_id / "local_only_control",
            keep_onnx=False,
        )
        per_node_scores[node_id] = {"collective": collective_report, "local_only_control": control_report}

    comm_estimate = comm_estimator.estimate_all_radios(kt_result["total_bytes_exchanged"])
    energy_summary = tracker.summary()

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
