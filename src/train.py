"""Entry point: one end-to-end continual-mesh run on PlantVillage — no
separate local-only stage, no warm-start from a previous run.

    python -m src.train --config config.yaml
    python -m src.train --config config.yaml --arch mobilenet_v3_small

For every configured architecture:

  1. PlantVillage -> 5% global probe set + 95% private pool -> 6 non-IID
     node shards -> each shard cut into continual.num_batches batches, each
     with its own private train/test split (see src/data/splits.py).
  2. Batch by batch, every node trains locally, evaluates, and — depending
     on its EMA-based role — uploads its knowledge to the shared database
     and/or retrieves its peers' knowledge and distils towards it (see
     src/federated/continual.py). Models carry over between batches.
  3. Per node and batch it records pre- vs. post-distill accuracy, bytes
     uploaded/downloaded, and compute energy per phase.

Outputs (under output.dir, default outputs/):
  continual/<arch>/knowledge.db         the shared knowledge database
  continual/<arch>/batch_logs.json      every NodeBatchRecord, in full
  continual/<arch>/batch_summary.csv    one row per (batch, node)
  results_<arch>.json, results_summary.json
  checkpoints/<arch>/<node>.pt, checkpoints/classes.json
  sustainability_report.json / .md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from pathlib import Path

import torch

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.data.splits import build_batch_loaders, build_continual_splits, build_probe_loader
from src.energy.tracker import (
    CommunicationCostEstimator,
    ComputeEnergyTracker,
    sweep_totals_from_emissions_csv,
    write_sustainability_report,
)
from src.evaluate import scalar_metrics
from src.federated.continual import BatchLog, ContinualMesh
from src.federated.knowledge_store import KnowledgeStore
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb
from src.reporting import write_csv


def scale_epochs_by_node_size(base_epochs: int, train_sizes: list[int], max_epochs: int) -> list[int]:
    """Per-node epoch count so a node with a much smaller train batch still
    gets roughly as many gradient steps as a median-sized node. Scales UP
    only (typical/large nodes keep `base_epochs`), capped at `max_epochs`.
    """
    reference = statistics.median(train_sizes)
    return [min(max_epochs, round(base_epochs * max(1.0, reference / max(1, size)))) for size in train_sizes]


def build_nodes(cfg, arch: str, dataset, num_nodes: int, device: str) -> list[Node]:
    crop_classes, disease_classes = dataset.labels.crop_classes, dataset.labels.disease_classes
    nodes = []
    for i in range(num_nodes):
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True),
            freeze_low_layers_=cfg.get("training.freeze_low_layers_in_mesh", False),
        )
        # loaders and class weights are swapped in per batch by ContinualMesh.run_batch
        nodes.append(Node(
            f"node_{i}", model, train_loader=None, test_loader=None, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            loss_type=cfg.get("training.loss_type", "cross_entropy"),
            focal_gamma=cfg.get("training.focal_gamma", 2.0),
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
        ))
    return nodes


def batch_summary_rows(batch_logs: list[BatchLog], metric: str, comm_estimator: CommunicationCostEstimator) -> list[dict]:
    rows = []
    for log in batch_logs:
        for record in log.per_node.values():
            pre, post = record.pre_distill_eval, record.post_distill_eval
            wifi = comm_estimator.estimate(record.total_bytes, "wifi") if "wifi" in comm_estimator.radio_energy_j_per_byte else {}
            rows.append({
                "batch": record.batch_idx,
                "node": record.node_id,
                "num_train": record.num_train,
                "num_test": record.num_test,
                "roles": "+".join(record.roles) or "idle",
                "ema": round(record.ema, 4),
                "prev_ema": None if record.prev_ema is None else round(record.prev_ema, 4),
                "uploaded": record.uploaded,
                "distilled": record.distilled,
                "peers_used": len(record.peers_used),
                f"pre_{metric}": round(pre[metric], 4),
                f"post_{metric}": round(post[metric], 4),
                f"gain_{metric}": round(post[metric] - pre[metric], 4),
                "pre_crop_accuracy": round(pre["crop_accuracy"], 4),
                "post_crop_accuracy": round(post["crop_accuracy"], 4),
                "pre_disease_accuracy": round(pre["disease_accuracy"], 4),
                "post_disease_accuracy": round(post["disease_accuracy"], 4),
                "improved": record.improved,
                "bytes_uploaded": record.bytes_uploaded,
                "bytes_downloaded": record.bytes_downloaded,
                "compute_energy_kwh": record.total_energy_kwh,
                "wifi_comm_energy_kwh": wifi.get("energy_kwh", 0.0),
                "duration_s": round(sum(record.duration_s.values()), 2),
            })
    return rows


def summarise_run(batch_logs: list[BatchLog], metric: str) -> dict:
    """Headline numbers for one architecture: how often distillation
    helped, by how much, and what it cost.
    """
    distilled = [r for log in batch_logs for r in log.per_node.values() if r.distilled]
    per_batch = []
    for log in batch_logs:
        records = list(log.per_node.values())
        batch_distilled = [r for r in records if r.distilled]
        per_batch.append({
            "batch": log.batch_idx,
            "teachers": [r.node_id for r in records if "teacher" in r.roles],
            "learners": [r.node_id for r in records if "learner" in r.roles],
            f"mean_pre_{metric}": statistics.mean(r.pre_distill_eval[metric] for r in records),
            f"mean_post_{metric}": statistics.mean(r.post_distill_eval[metric] for r in records),
            "num_distilled": len(batch_distilled),
            "num_improved": sum(r.improved for r in batch_distilled),
            "bytes_uploaded": sum(r.bytes_uploaded for r in records),
            "bytes_downloaded": sum(r.bytes_downloaded for r in records),
            "compute_energy_kwh": log.total_energy_kwh,
        })
    return {
        "metric": metric,
        "num_batches": len(batch_logs),
        "num_distillations": len(distilled),
        "num_improved": sum(r.improved for r in distilled),
        "mean_distill_gain": (
            statistics.mean(r.distill_gain[metric] for r in distilled) if distilled else 0.0
        ),
        "total_bytes_uploaded": sum(b["bytes_uploaded"] for b in per_batch),
        "total_bytes_downloaded": sum(b["bytes_downloaded"] for b in per_batch),
        "total_bytes_exchanged": sum(log.total_bytes for log in batch_logs),
        "total_compute_energy_kwh": sum(log.total_energy_kwh for log in batch_logs),
        "per_batch": per_batch,
    }


def run_continual(cfg, arch, dataset, splits, probe_loader, tracker, comm_estimator, device, output_dir: Path) -> dict:
    run_dir = output_dir / "continual" / arch
    run_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = len(splits.node_batches)
    num_batches = len(splits.node_batches[0])
    metric = cfg.get("continual.ema_metric", "pair_accuracy")

    nodes = build_nodes(cfg, arch, dataset, num_nodes, device)
    mesh = ContinualMesh(
        nodes, probe_loader, KnowledgeStore(run_dir / "knowledge.db"), tracker,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        ema_alpha=cfg.get("continual.ema_alpha", 0.5),
        ema_threshold=cfg.get("continual.ema_threshold", 0.8),
        ema_metric=metric,
        label_prefix=f"{arch}_",
    )

    base_local_epochs = cfg.get(
        f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
    )
    max_epochs = cfg.get("training.max_epochs_per_node", 40)
    batch_logs: list[BatchLog] = []
    for b in range(num_batches):
        batch_loaders = {
            node.node_id: build_batch_loaders(cfg, dataset, splits.node_batches[i][b])
            for i, node in enumerate(nodes)
        }
        base = cfg.get("continual.first_batch_local_epochs", base_local_epochs) if b == 0 else base_local_epochs
        epochs = scale_epochs_by_node_size(
            base, [len(batch_loaders[n.node_id][0].dataset) for n in nodes], max_epochs
        )
        log = mesh.run_batch(
            b, batch_loaders,
            local_epochs={node.node_id: e for node, e in zip(nodes, epochs)},
            lr=cfg.get("training.lr", 0.001),
            distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
            distill_lr=cfg.get("training.distill_lr", 0.0005),
            proto_weight=cfg.get("training.proto_weight", 0.5),
            kd_weight=cfg.get("training.kd_weight", 0.5),
            crop_kd_weight=cfg.get("training.crop_kd_weight", None),
            temperature=cfg.get("training.kd_temperature", 2.0),
        )
        batch_logs.append(log)
        print(f"  batch {b}: {log.total_bytes} bytes exchanged, {log.total_energy_kwh:.6f} kWh compute")
        for record in log.per_node.values():
            pre, post = record.pre_distill_eval[metric], record.post_distill_eval[metric]
            print(
                f"    {record.node_id}: roles={'+'.join(record.roles) or 'idle':<15} "
                f"EMA={record.ema:.3f} {metric} pre={pre:.3f} post={post:.3f} "
                f"({'improved' if record.improved else 'distilled, no gain' if record.distilled else 'no distill'})"
            )

    def _jsonable(record):
        d = dataclasses.asdict(record)
        d["pre_distill_eval"] = scalar_metrics(record.pre_distill_eval)
        d["post_distill_eval"] = scalar_metrics(record.post_distill_eval)
        return d

    (run_dir / "batch_logs.json").write_text(json.dumps(
        [{"batch_idx": log.batch_idx, "per_node": {nid: _jsonable(r) for nid, r in log.per_node.items()}}
         for log in batch_logs],
        indent=2,
    ))
    write_csv(batch_summary_rows(batch_logs, metric, comm_estimator), run_dir / "batch_summary.csv")

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    summary = summarise_run(batch_logs, metric)
    last = batch_logs[-1].per_node
    return {
        "architecture": arch,
        "params": count_parameters(nodes[0].model),
        "model_size_mb": model_size_mb(nodes[0].model),
        # each node's final post-distill eval on its last batch's test set —
        # the key src/model_selection.py scores when picking a checkpoint
        "mesh_eval": {nid: scalar_metrics(r.post_distill_eval) for nid, r in last.items()},
        "continual": summary,
        "knowledge_store": mesh.store.entries(),
        "total_bytes_exchanged": summary["total_bytes_exchanged"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch", default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # CodeCarbon appends to emissions.csv; start clean so the report's
    # emissions.csv total covers exactly this run
    (output_dir / "emissions.csv").unlink(missing_ok=True)

    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 224))
    print(
        f"Loaded {len(dataset)} PlantVillage images, {len(dataset.labels.crop_classes)} crop classes, "
        f"{len(dataset.labels.disease_classes)} disease classes."
    )
    splits = build_continual_splits(cfg, dataset)
    probe_loader = build_probe_loader(cfg, dataset, splits.probe_idx)
    print(f"Global probe set: {len(splits.probe_idx)} images")
    for i, batches in enumerate(splits.node_batches):
        sizes = ", ".join(f"{len(s.train_idx)}/{len(s.test_idx)}" for s in batches)
        print(f"  node_{i} batches (train/test): {sizes}")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "classes.json").write_text(json.dumps(
        {
            "crop_classes": dataset.labels.crop_classes,
            "disease_classes": dataset.labels.disease_classes,
            "image_size": cfg.get("data.image_size", 224),
        },
        indent=2,
    ))

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=output_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    all_results = {}
    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} ===")
        result = run_continual(cfg, arch, dataset, splits, probe_loader, tracker, comm_estimator, device, output_dir)
        all_results[arch] = result
        (output_dir / f"results_{arch}.json").write_text(json.dumps(result, indent=2))
        s = result["continual"]
        print(
            f"distillation improved {s['num_improved']}/{s['num_distillations']} node-batches, "
            f"mean gain {s['mean_distill_gain']:+.4f} {s['metric']}, "
            f"{s['total_bytes_exchanged']} bytes, {s['total_compute_energy_kwh']:.6f} kWh"
        )

    (output_dir / "results_summary.json").write_text(json.dumps(all_results, indent=2))

    total_bytes = sum(r["total_bytes_exchanged"] for r in all_results.values())
    write_sustainability_report(
        output_dir / "sustainability_report",
        tracker.summary(),
        comm_estimator.estimate_all_radios(total_bytes),
        {
            arch: {k: r["continual"][k] for k in ("metric", "num_distillations", "num_improved", "mean_distill_gain")}
            for arch, r in all_results.items()
        },
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(output_dir),
    )
    print(f"\nDone. Results and sustainability report written to {output_dir}/")


if __name__ == "__main__":
    main()
