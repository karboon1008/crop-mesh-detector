"""Stage 2: 1-5 rounds of knowledge-transfer distillation on top of the
3 per-node checkpoints stage 1 (run_corn_pipeline.py) produces, plus a
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_knowledge_transfer --rounds 2
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import onnxruntime
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import make_subset
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import Node
from src.models.factory import build_model
from src.validation.corn_mesh_dataset import CornDiseaseView, CornLabelMap, CornMeshData, prepare_corn_mesh_data
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.run_corn_pipeline import node_output_dir

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
    round_idx: int | str = "na",
) -> dict:
    """One knowledge-transfer round: `nodes` distill toward their peers'
    consensus (no separate local_train call); `control_nodes` run the
    same-budget local_train with no exchange at all, for the Appendix
    A.1 fairness-aligned comparison. Both dicts are mutated in place
    (each Node's .model is updated); nothing is reloaded from disk here.

    `round_idx` is folded into each tracked block's label (e.g.
    f"{node_id}_kt_compute_knowledge_round_{round_idx}") purely so
    run_round_with_io can later filter tracker.log down to just this
    round's blocks when building the per-node energy breakdown --
    tracker.log is an append-only history shared across every round.

    compute_knowledge is still called for ALL nodes upfront (required
    for the peer-consensus math -- each node's distill step needs its
    peers' already-computed payloads), but each node's own call is
    wrapped in its own tracked block, separate from that node's
    `distill` block, so the two can be summed together into one
    per-node energy figure without misattributing one node's compute to
    another.
    """
    payloads = {}
    for node_id, node in nodes.items():
        ctx = (
            tracker.track(f"{node_id}_kt_compute_knowledge_round_{round_idx}")
            if tracker is not None
            else nullcontext()
        )
        with ctx:
            payloads[node_id] = node.compute_knowledge(probe_loader)
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
        ctx = (
            tracker.track(f"{node_id}_kt_distill_round_{round_idx}") if tracker is not None else nullcontext()
        )
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
        ctx = (
            tracker.track(f"{node_id}_local_only_control_round_{round_idx}")
            if tracker is not None
            else nullcontext()
        )
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


def _sum_tracked_blocks(records: list[dict]) -> dict:
    """Collapses a list of ComputeEnergyTracker.log records (each has
    duration_s/energy_kwh/method) into one summed {duration_s,
    energy_kwh, method} dict, matching the per-node shape the spec's
    round_summary.json calls for.
    """
    if not records:
        return {"duration_s": 0.0, "energy_kwh": 0.0, "method": None}
    methods = {r.get("method") for r in records}
    return {
        "duration_s": sum(r.get("duration_s", 0.0) for r in records),
        "energy_kwh": sum(r.get("energy_kwh", 0.0) for r in records),
        "method": next(iter(methods)) if len(methods) == 1 else "mixed",
    }


def _build_per_node_energy_breakdown(
    tracker_log: list[dict], node_ids, round_idx: int | str
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Filters tracker.log (an append-only history across every round
    run so far) down to just this round's blocks, using the
    f"..._round_{round_idx}" label suffix run_kt_round tags each block
    with -- otherwise a later round's summary would double-count
    earlier rounds' energy under "per_node".
    """
    per_node: dict[str, dict] = {}
    per_node_local_only_control: dict[str, dict] = {}
    for node_id in node_ids:
        kt_labels = {
            f"{node_id}_kt_compute_knowledge_round_{round_idx}",
            f"{node_id}_kt_distill_round_{round_idx}",
        }
        per_node[node_id] = _sum_tracked_blocks([r for r in tracker_log if r.get("label") in kt_labels])

        control_label = f"{node_id}_local_only_control_round_{round_idx}"
        per_node_local_only_control[node_id] = _sum_tracked_blocks(
            [r for r in tracker_log if r.get("label") == control_label]
        )
    return per_node, per_node_local_only_control


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
        round_idx=round_idx,
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


def evaluate_round0_baseline(data: CornMeshData, stage1_dir: Path, cross_node_idx: list[int]) -> dict[str, dict]:
    """Evaluates each node's ALREADY-EXPORTED stage-1 model.onnx (no
    re-export) against the cross-node union set -- this is round 0's
    baseline for the per-disease collaboration-gain table, since stage
    1's own report.json only covers each node's local test set.
    """
    baseline: dict[str, dict] = {}
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        manifest = json.loads((node_dir / "manifest.json").read_text())
        session = onnxruntime.InferenceSession(str(node_dir / "model.onnx"))
        scratch = node_dir / "_round0_scratch.json"
        report = run_evaluation(session, data.eval_base, cross_node_idx, manifest, MODEL_NAME, node_id, scratch)
        scratch.unlink(missing_ok=True)
        baseline[node_id] = report
    return baseline


def _load_node_from_checkpoint(checkpoint_path: Path, data: CornMeshData, node_id: str, batch_size: int) -> Node:
    model = build_model(
        MODEL_NAME, len(data.label_map.crop_classes), len(data.label_map.disease_classes), pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.per_node[node_id]["test_idx"]
    train_loader = DataLoader(
        CornDiseaseView(data.train_base, train_idx, data.label_map),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        CornDiseaseView(data.eval_base, test_idx, data.label_map), batch_size=batch_size, shuffle=False
    )
    return Node(node_id, model, train_loader, test_loader, device="cpu")


def _per_disease_gain_table(
    round0_baseline: dict[str, dict], final_round_scores: dict[str, dict], disease_classes: list[str]
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for node_id, baseline_report in round0_baseline.items():
        table[node_id] = {}
        collective_cross = final_round_scores[node_id]["collective"]["cross_node"]["summary"]
        control_cross = final_round_scores[node_id]["local_only_control"]["cross_node"]["summary"]
        for disease_name in disease_classes:
            round0_acc = baseline_report["summary"]["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            collective_acc = collective_cross["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            control_acc = control_cross["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            table[node_id][disease_name] = {
                "round_0_accuracy": round0_acc,
                "round_N_collective_accuracy": collective_acc,
                "round_N_local_only_control_accuracy": control_acc,
                "gain_vs_round0": collective_acc - round0_acc,
                "gain_vs_local_only_control": collective_acc - control_acc,
            }
    return table


NODE_DISEASE_DEFAULT_FALLBACK = {
    "node_0": "Common_rust",
    "node_1": "Cercospora_leaf_spot Gray_leaf_spot",
    "node_2": "Northern_Leaf_Blight",
}


def _flat_accuracy_metrics(summary: dict) -> dict[str, float]:
    """compute_collaboration_gain's macro_average/worst_node helpers sum
    every value in the per-node dict across all nodes, so they need a
    dict of plain numeric metrics -- not evaluate_onnx's full "summary"
    block, which also carries "model"/"node" (str) and
    "per_class_accuracy"/"top_confusions" (nested dict/list) fields that
    would make that summation raise a TypeError.
    """
    return {"crop_accuracy": summary["crop_accuracy"], "disease_accuracy": summary["disease_accuracy"]}


def build_knowledge_transfer_summary(
    cfg: Config, round0_baseline: dict[str, dict], round_summaries: list[dict], data: CornMeshData
) -> dict:
    final_round = round_summaries[-1]
    final_scores = final_round["per_node_scores"]

    collective_evals = {nid: s["collective"]["cross_node"]["summary"] for nid, s in final_scores.items()}
    control_evals = {nid: s["local_only_control"]["cross_node"]["summary"] for nid, s in final_scores.items()}
    gain = compute_collaboration_gain(
        {nid: _flat_accuracy_metrics(s) for nid, s in collective_evals.items()},
        {nid: _flat_accuracy_metrics(s) for nid, s in control_evals.items()},
    )

    node_diseases = cfg.get("corn_mesh.node_diseases", NODE_DISEASE_DEFAULT_FALLBACK)
    # total_bytes_exchanged is reported per-round (not cumulative), so
    # summing across rounds is correct. total_compute_energy_kwh, in
    # contrast, comes from ComputeEnergyTracker.summary() -- and the same
    # tracker instance is shared across all rounds in main(), so each
    # round's figure already IS the running total over its append-only
    # log. Summing round_summaries' energy figures would double-count
    # every round but the last (e.g. 2 rounds -> E1 + (E1+E2) instead of
    # E1+E2), so the final round's own figure is the correct cumulative
    # total.
    cumulative_bytes = sum(r["total_bytes_exchanged"] for r in round_summaries)
    cumulative_energy = round_summaries[-1]["energy"]["total_compute_energy_kwh"]

    return {
        "node_count": 3,
        "data_split": {
            "strategy": "disjoint disease-label skew within one crop (Corn)",
            "strength": (
                "complete disjoint — each node has exactly one assigned disease class, with zero "
                "overlap with its peers'; only the shared healthy class is split (dedup-aware, "
                "~1/3 each, no image duplicated across nodes)"
            ),
            "node_diseases": node_diseases,
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
            "both — local (own node's held-out test set) and global held-out (union of all 3 "
            "nodes' held-out test sets); test samples never enter any training set"
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
        "collaboration_gain_per_disease": _per_disease_gain_table(
            round0_baseline, final_scores, data.label_map.disease_classes
        ),
        "limitation_note": (
            "Cross-node disease-recognition gain above comes from probe-set logit distillation "
            "only, not prototype alignment — prototype exchange only reinforces the shared "
            "healthy class, since a node's proto_loss only runs over its own local batches."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/corn_mesh")
    parser.add_argument("--rounds", type=int, default=None, help="Number of knowledge-transfer rounds (1-5)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    rounds = args.rounds if args.rounds is not None else cfg.get("corn_mesh.rounds", 2)
    if not (1 <= rounds <= 5):
        parser.error(f"--rounds must be between 1 and 5, got {rounds}")

    stage1_dir = Path(args.output_dir)
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        # checkpoint.pt is read by _load_node_from_checkpoint;
        # model.onnx/manifest.json are read by evaluate_round0_baseline
        # (no re-export in stage 2); classes.json is read below to verify
        # stage 1's persisted split against this run's split -- all four
        # must exist, i.e. stage 1's train AND export stages both ran for
        # this node.
        for required_name in ("checkpoint.pt", "model.onnx", "manifest.json", "classes.json"):
            required_path = node_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(
                    f"{required_path} not found — run "
                    f"'python -m src.validation.run_corn_pipeline' first."
                )

    data = prepare_corn_mesh_data(cfg)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]

    # Stage 1 (run_corn_pipeline.py's run_train_stage) persists each node's
    # EXACT train_idx/test_idx into that node's classes.json. Stage 2 here
    # calls prepare_corn_mesh_data(cfg) again and just trusts it reproduces
    # the identical split deterministically -- but if data.seed,
    # data.test_fraction, data.probe_set_*, corn_mesh.healthy_dedup_threshold,
    # or the contents of data/PlantVillage differ between the two runs,
    # stage 2 would otherwise silently evaluate stage-1 checkpoints on
    # images they were trained on (a train/test leak) with no error and no
    # visible symptom besides implausibly high accuracy. Fail loudly instead.
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        classes_path = node_dir / "classes.json"
        stage1_classes = json.loads(classes_path.read_text())
        if stage1_classes.get("train_idx") != data.per_node[node_id]["train_idx"] or stage1_classes.get(
            "test_idx"
        ) != data.per_node[node_id]["test_idx"]:
            raise ValueError(
                f"{node_id}: the train/test split persisted in {classes_path} by stage 1 "
                f"(run_corn_pipeline.py) does not match the split prepare_corn_mesh_data(cfg) "
                f"produces now. Stage 1 and stage 2 must be run against the same config and the "
                f"same data/PlantVillage contents -- otherwise stage 2 would silently evaluate "
                f"this node's checkpoint on images it was trained on."
            )

    print("=== round 0 baseline (stage 1 checkpoints, cross-node eval) ===")
    round0_baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    kt_dir = stage1_dir / "knowledge_transfer"
    kt_dir.mkdir(parents=True, exist_ok=True)
    (kt_dir / "round_0_baseline.json").write_text(json.dumps(round0_baseline, indent=2))

    nodes = {
        node_id: _load_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in ("node_0", "node_1", "node_2")
    }
    control_nodes = {
        node_id: _load_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in ("node_0", "node_1", "node_2")
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
        summary = run_round_with_io(
            round_idx, nodes, control_nodes, probe_loader, data, cross_node_idx, cfg, tracker, comm_estimator, round_dir
        )
        round_summaries.append(summary)

    kt_summary = build_knowledge_transfer_summary(cfg, round0_baseline, round_summaries, data)
    (kt_dir / "knowledge_transfer_summary.json").write_text(json.dumps(kt_summary, indent=2))
    print(f"Done. Knowledge-transfer outputs in {kt_dir}/")


if __name__ == "__main__":
    main()
