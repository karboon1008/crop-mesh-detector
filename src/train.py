"""Entry point for the continual mesh on PlantVillage. Each invocation is
one trigger, and data arrives batch by batch, like in the field:

    python -m src.train                 # first run: set up the stream, run batch 0
                                        # (continual.first_batch_size images, default 20,000)
    python -m src.train --next-batch    # a new batch arrives (continual.next_batch_size,
                                        # default 3,000) and is run on top of the saved models
    python -m src.train --next-batch    # ...and again, until PlantVillage runs out
    python -m src.train --reset         # throw everything away and start again from batch 0

Add --arch <name> to any of these to run only one architecture.

A trigger:
  1. draws the new batch — a stratified sample of the images no earlier
     batch used — carves a stratified 5% of it into this batch's public
     probe set, hands every other image to the node that owns it, and lets
     each node split what it got into private train/test 85/15
     (src/data/stream.py)
  2. for every architecture, reloads each node's model/optimizer, EMA, and
     the shared knowledge database, and runs the batch: local training,
     pre-distill eval, EMA roles, upload/retrieve + distil, post-distill
     eval (src/federated/continual.py)
  3. saves everything again and rewrites the reports.

An architecture that missed earlier batches (e.g. a crashed or --arch-only
trigger) is brought up to date before the new batch, so every
architecture always ends up having seen the same stream.

Outputs (under output.dir, default outputs/):
  continual/stream.json                 which images arrived in which batch: probe slice + per-node train/test
  continual/<arch>/knowledge.db         the shared knowledge database
  continual/<arch>/nodes/<node>.pt      each node's model + optimizer (for the next trigger)
  continual/<arch>/state.json           batches completed + each node's EMA
  continual/<arch>/batch_logs.json      every node's record for every batch
  continual/<arch>/batch_summary.csv    one row per (batch, node)
  results_<arch>.json, results_summary.json
  checkpoints/<arch>/<node>.pt, checkpoints/classes.json
  sustainability_report.json / .md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import statistics
from pathlib import Path

import torch

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.data.splits import build_batch_loaders, build_probe_loader
from src.data.stream import DataStream
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


def batch_log_to_json(log: BatchLog) -> dict:
    def record_json(record):
        d = dataclasses.asdict(record)
        d["pre_distill_eval"] = scalar_metrics(record.pre_distill_eval)
        d["post_distill_eval"] = scalar_metrics(record.post_distill_eval)
        d["total_bytes"] = record.total_bytes
        d["total_energy_kwh"] = record.total_energy_kwh
        return d

    return {
        "batch_idx": log.batch_idx,
        "absent": log.absent,
        "per_node": {nid: record_json(r) for nid, r in log.per_node.items()},
    }


def batch_summary_rows(batch_logs: list[dict], metric: str, comm_estimator: CommunicationCostEstimator) -> list[dict]:
    has_wifi = "wifi" in comm_estimator.radio_energy_j_per_byte
    rows = []
    for log in batch_logs:
        for r in log["per_node"].values():
            pre, post = r["pre_distill_eval"], r["post_distill_eval"]
            rows.append({
                "batch": r["batch_idx"],
                "node": r["node_id"],
                "num_train": r["num_train"],
                "num_test": r["num_test"],
                "roles": "+".join(r["roles"]) or "idle",
                "ema": round(r["ema"], 4),
                "prev_ema": None if r["prev_ema"] is None else round(r["prev_ema"], 4),
                "uploaded": r["uploaded"],
                "distilled": r["distilled"],
                "peers_used": len(r["peers_used"]),
                "logit_peers": len(r["logit_peers"]),
                "probe_images_used": r["probe_images_used"],
                f"pre_{metric}": round(pre[metric], 4),
                f"post_{metric}": round(post[metric], 4),
                f"gain_{metric}": round(post[metric] - pre[metric], 4),
                "pre_crop_accuracy": round(pre["crop_accuracy"], 4),
                "post_crop_accuracy": round(post["crop_accuracy"], 4),
                "pre_disease_accuracy": round(pre["disease_accuracy"], 4),
                "post_disease_accuracy": round(post["disease_accuracy"], 4),
                "improved": r["improved"],
                "bytes_uploaded": r["bytes_uploaded"],
                "bytes_downloaded": r["bytes_downloaded"],
                "compute_energy_kwh": r["total_energy_kwh"],
                "wifi_comm_energy_kwh": (
                    comm_estimator.estimate(r["total_bytes"], "wifi")["energy_kwh"] if has_wifi else 0.0
                ),
                "duration_s": round(sum(r["duration_s"].values()), 2),
            })
    return rows


def summarise_run(batch_logs: list[dict], metric: str) -> dict:
    """Headline numbers for one architecture across every batch so far: how
    often distillation helped, by how much, and what it cost.
    """
    distilled = [r for log in batch_logs for r in log["per_node"].values() if r["distilled"]]
    per_batch = []
    for log in batch_logs:
        records = list(log["per_node"].values())
        per_batch.append({
            "batch": log["batch_idx"],
            "nodes_present": [r["node_id"] for r in records],
            "nodes_absent": log["absent"],
            "teachers": [r["node_id"] for r in records if "teacher" in r["roles"]],
            "learners": [r["node_id"] for r in records if "learner" in r["roles"]],
            f"mean_pre_{metric}": statistics.mean(r["pre_distill_eval"][metric] for r in records) if records else None,
            f"mean_post_{metric}": statistics.mean(r["post_distill_eval"][metric] for r in records) if records else None,
            "num_distilled": sum(r["distilled"] for r in records),
            "num_improved": sum(r["improved"] for r in records),
            "bytes_uploaded": sum(r["bytes_uploaded"] for r in records),
            "bytes_downloaded": sum(r["bytes_downloaded"] for r in records),
            "compute_energy_kwh": sum(r["total_energy_kwh"] for r in records),
        })
    return {
        "metric": metric,
        "num_batches": len(batch_logs),
        "num_distillations": len(distilled),
        "num_improved": sum(r["improved"] for r in distilled),
        "mean_distill_gain": statistics.mean(r["distill_gain"][metric] for r in distilled) if distilled else 0.0,
        "total_bytes_uploaded": sum(b["bytes_uploaded"] for b in per_batch),
        "total_bytes_downloaded": sum(b["bytes_downloaded"] for b in per_batch),
        "total_bytes_exchanged": sum(b["bytes_uploaded"] + b["bytes_downloaded"] for b in per_batch),
        "total_compute_energy_kwh": sum(b["compute_energy_kwh"] for b in per_batch),
        "per_batch": per_batch,
    }


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def run_architecture(cfg, arch, dataset, stream, tracker, comm_estimator, device, output_dir: Path) -> dict:
    """Runs every stream batch this architecture hasn't run yet, saving its
    state after each one, then rewrites its reports.
    """
    run_dir = output_dir / "continual" / arch
    node_dir = run_dir / "nodes"
    node_dir.mkdir(parents=True, exist_ok=True)
    state_path, logs_path = run_dir / "state.json", run_dir / "batch_logs.json"
    state = _read_json(state_path, {"completed_batches": 0, "ema": {}})
    batch_logs = _read_json(logs_path, [])
    metric = cfg.get("continual.ema_metric", "pair_accuracy")
    lr = cfg.get("training.lr", 0.001)

    nodes = build_nodes(cfg, arch, dataset, stream.num_nodes, device)
    if state["completed_batches"] > 0:
        for node in nodes:
            node.load_state(torch.load(node_dir / f"{node.node_id}.pt", map_location=device), lr)
    mesh = ContinualMesh(
        nodes,
        KnowledgeStore(run_dir / "knowledge.db", reset=state["completed_batches"] == 0), tracker,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        ema_alpha=cfg.get("continual.ema_alpha", 0.5),
        ema_threshold=cfg.get("continual.ema_threshold", 0.8),
        ema_metric=metric,
        label_prefix=f"{arch}_",
    )
    mesh.ema.update(state["ema"])

    base_local_epochs = cfg.get(
        f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
    )
    for b in range(state["completed_batches"], stream.num_batches):
        # a node needs >= 2 train images (BatchNorm) and >= 1 test image to take part
        batch_loaders = {}
        for node in nodes:
            split = stream.node_split(b, node.node_id)
            if len(split.train_idx) >= 2 and split.test_idx:
                batch_loaders[node.node_id] = build_batch_loaders(cfg, dataset, split)
        present = [n for n in nodes if n.node_id in batch_loaders]
        if b == 0:
            # the big first batch: scale epochs up for small nodes so every
            # node gets a comparable number of gradient steps from ImageNet init
            epochs = scale_epochs_by_node_size(
                cfg.get("continual.first_batch_local_epochs", base_local_epochs),
                [len(batch_loaders[n.node_id][0].dataset) for n in present],
                cfg.get("training.max_epochs_per_node", 40),
            ) if present else []
        else:
            # small follow-up batches: a flat epoch count — scaling a node with a
            # handful of new images up to dozens of epochs would just overfit them
            epochs = [base_local_epochs] * len(present)

        probe_idx = stream.batches[b]["probe_idx"]
        print(f"  batch {b}: {stream.batches[b]['size']} images, probe set {len(probe_idx)} images")
        log = mesh.run_batch(
            b, batch_loaders,
            probe_loader=build_probe_loader(cfg, dataset, probe_idx),
            local_epochs={node.node_id: e for node, e in zip(present, epochs)},
            lr=lr,
            distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
            distill_lr=cfg.get("training.distill_lr", 0.0005),
            proto_weight=cfg.get("training.proto_weight", 0.5),
            kd_weight=cfg.get("training.kd_weight", 0.5),
            crop_kd_weight=cfg.get("training.crop_kd_weight", None),
            temperature=cfg.get("training.kd_temperature", 2.0),
        )
        for record in log.per_node.values():
            pre, post = record.pre_distill_eval[metric], record.post_distill_eval[metric]
            print(
                f"    {record.node_id}: {record.num_train:>5} train / {record.num_test:>4} test  "
                f"roles={'+'.join(record.roles) or 'idle':<15} EMA={record.ema:.3f} "
                f"{metric} pre={pre:.3f} post={post:.3f} "
                f"({'improved' if record.improved else 'distilled, no gain' if record.distilled else 'no distill'})"
            )
        for node_id in log.absent:
            print(f"    {node_id}: too few images this batch — sat it out")
        print(f"    {log.total_bytes} bytes exchanged, {log.total_energy_kwh:.6f} kWh compute")

        # save after every batch so a crash never loses a finished one
        for node in nodes:
            torch.save(node.state(), node_dir / f"{node.node_id}.pt")
        batch_logs.append(batch_log_to_json(log))
        logs_path.write_text(json.dumps(batch_logs, indent=2))
        state = {"completed_batches": b + 1, "ema": mesh.ema}
        state_path.write_text(json.dumps(state, indent=2))

    write_csv(batch_summary_rows(batch_logs, metric, comm_estimator), run_dir / "batch_summary.csv")
    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    # each node's post-distill eval from the latest batch it took part in —
    # the key src/model_selection.py scores when picking a checkpoint
    latest_eval = {}
    for log in batch_logs:
        for node_id, r in log["per_node"].items():
            latest_eval[node_id] = r["post_distill_eval"]
    summary = summarise_run(batch_logs, metric)
    return {
        "architecture": arch,
        "params": count_parameters(nodes[0].model),
        "model_size_mb": model_size_mb(nodes[0].model),
        "mesh_eval": latest_eval,
        "continual": summary,
        "knowledge_store": mesh.store.entries(),
        "total_bytes_exchanged": summary["total_bytes_exchanged"],
    }


def reset_outputs(output_dir: Path) -> None:
    shutil.rmtree(output_dir / "continual", ignore_errors=True)
    shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
    for name in ("emissions.csv", "run_state.json", "results_summary.json"):
        (output_dir / name).unlink(missing_ok=True)
    for f in output_dir.glob("results_*.json"):
        f.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch", default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    trigger = parser.add_mutually_exclusive_group()
    trigger.add_argument(
        "--next-batch", action="store_true",
        help="A new batch of continual.next_batch_size images arrives and is run on top of the saved models",
    )
    trigger.add_argument(
        "--reset", action="store_true",
        help="Delete the saved stream, models, and knowledge database, and start again from batch 0",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / "continual" / "stream.json"
    seed = cfg.get("data.seed", 42)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    test_fraction = cfg.get("data.test_fraction", 0.15)

    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 224))
    print(
        f"Loaded {len(dataset)} PlantVillage images, {len(dataset.labels.crop_classes)} crop classes, "
        f"{len(dataset.labels.disease_classes)} disease classes."
    )

    if args.reset:
        reset_outputs(output_dir)
    if not stream_path.exists():
        if args.next_batch:
            raise SystemExit("No stream yet — run `python -m src.train` first to start it with batch 0.")
        stream = DataStream.create(cfg, dataset, stream_path)
        batch = stream.next_batch(dataset, cfg.get("continual.first_batch_size", 20000), probe_fraction, test_fraction, seed)
    else:
        stream = DataStream.load(cfg, dataset, stream_path)
        if args.next_batch:
            batch = stream.next_batch(dataset, cfg.get("continual.next_batch_size", 3000), probe_fraction, test_fraction, seed)
        else:
            batch = None
            print(
                f"Stream already has {stream.num_batches} batch(es); no new batch was triggered "
                f"(pass --next-batch for one). Bringing any architecture that is behind up to date."
            )

    if batch is not None:
        print(f"New batch {batch['batch_idx']}: {batch['size']} images (stratified), "
              f"{len(stream.remaining())} images left for future batches")
        print(f"  probe: {len(batch['probe_idx'])} images carved from this batch")
        for node_id, split in batch["nodes"].items():
            print(f"  {node_id}: {len(split['train_idx'])} train / {len(split['test_idx'])} test")

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

    results_path = output_dir / "results_summary.json"
    all_results = _read_json(results_path, {})
    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} ===")
        result = run_architecture(cfg, arch, dataset, stream, tracker, comm_estimator, device, output_dir)
        all_results[arch] = result
        (output_dir / f"results_{arch}.json").write_text(json.dumps(result, indent=2))
        s = result["continual"]
        print(
            f"so far ({s['num_batches']} batches): distillation improved {s['num_improved']}/"
            f"{s['num_distillations']} node-batches, mean gain {s['mean_distill_gain']:+.4f} {s['metric']}, "
            f"{s['total_bytes_exchanged']} bytes, {s['total_compute_energy_kwh']:.6f} kWh"
        )
    results_path.write_text(json.dumps(all_results, indent=2))

    # compute energy accumulates across triggers (each is its own process)
    run_state_path = output_dir / "run_state.json"
    run_state = _read_json(run_state_path, {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0})
    this_run = tracker.summary()
    for key in ("total_compute_energy_kwh", "total_duration_s", "num_tracked_blocks"):
        run_state[key] += this_run[key]
    run_state["all_blocks_cpu_measured"] = this_run["all_blocks_cpu_measured"] and run_state.get("all_blocks_cpu_measured", True)
    run_state_path.write_text(json.dumps(run_state, indent=2))

    total_bytes = sum(r["total_bytes_exchanged"] for r in all_results.values())
    write_sustainability_report(
        output_dir / "sustainability_report",
        run_state,
        comm_estimator.estimate_all_radios(total_bytes),
        {
            arch: {k: r["continual"][k] for k in ("metric", "num_batches", "num_distillations", "num_improved", "mean_distill_gain")}
            for arch, r in all_results.items()
        },
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(output_dir),
    )
    print(
        f"\nDone. {len(stream.remaining())} images left — run `python -m src.train --next-batch` "
        f"for the next batch of {cfg.get('continual.next_batch_size', 3000)}."
    )


if __name__ == "__main__":
    main()
