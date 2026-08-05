"""Entry point: for every configured architecture, trains an
aligned-budget local-only baseline and the decentralised mesh (prototype
+ probe-logit exchange only — no raw images, no weights) on PlantVillage,
then reports the collaboration gain and a compute + communication
sustainability accounting.

Run:
    python -m src.train --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import (
    carve_public_probe_set,
    load_full_dataset,
    make_subset,
    partition_nodes,
    train_test_split_indices,
)
from src.energy.tracker import (
    CommunicationCostEstimator,
    ComputeEnergyTracker,
    write_sustainability_report,
)
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb


def build_dataloaders(cfg: Config, dataset):
    probe_idx, remaining_idx = carve_public_probe_set(
        dataset, cfg.get("data.probe_set_fraction", 0.05), cfg.get("data.seed", 42)
    )
    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "by_crop"),
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
    )
    batch_size = cfg.get("training.batch_size", 32)
    probe_loader = DataLoader(make_subset(dataset, probe_idx), batch_size=batch_size, shuffle=False)

    node_loaders = []
    for shard in shards:
        train_idx, test_idx = train_test_split_indices(
            shard, cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
        )
        train_loader = DataLoader(make_subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
        node_loaders.append((train_loader, test_loader))
    return probe_loader, node_loaders


def run_baseline(cfg, arch, node_loaders, num_crop, num_disease, tracker, device):
    """Local-only training, no exchange at all — the comparison point
    the collaboration gain is measured against.
    """
    evals = {}
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        node_id = f"node_{i}"
        model = build_model(arch, num_crop, num_disease, pretrained=cfg.get("models.pretrained", True))
        node = Node(node_id, model, train_loader, test_loader, device=device)
        with tracker.track(f"{arch}_baseline_{node_id}"):
            node.local_train(cfg.get("training.baseline_epochs", 10), cfg.get("training.lr", 0.001))
        evals[node_id] = node.evaluate()
    return evals


def run_mesh(cfg, arch, node_loaders, probe_loader, num_crop, num_disease, tracker, device, output_dir):
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        model = build_model(arch, num_crop, num_disease, pretrained=cfg.get("models.pretrained", True))
        nodes.append(Node(f"node_{i}", model, train_loader, test_loader, device=device))

    mesh = MeshSimulator(
        nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    total_bytes = 0
    for r in range(cfg.get("training.rounds", 5)):
        with tracker.track(f"{arch}_mesh_round_{r}"):
            round_log = mesh.run_round(
                r,
                local_epochs=cfg.get("training.local_epochs_per_round", 2),
                distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
                lr=cfg.get("training.lr", 0.001),
                distill_lr=cfg.get("training.distill_lr", 0.0005),
                proto_weight=cfg.get("training.proto_weight", 0.5),
                kd_weight=cfg.get("training.kd_weight", 0.5),
                temperature=cfg.get("training.kd_temperature", 2.0),
            )
        total_bytes += round_log.total_bytes_exchanged
        print(f"  round {r}: {round_log.total_bytes_exchanged} bytes exchanged")

    final_evals = {node.node_id: node.evaluate() for node in nodes}

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    return final_evals, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    print(f"Loaded {len(dataset)} images, {num_crop} crop classes, {num_disease} disease classes.")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": dataset.labels.crop_classes,
                "disease_classes": dataset.labels.disease_classes,
                "image_size": cfg.get("data.image_size", 160),
            },
            indent=2,
        )
    )

    probe_loader, node_loaders = build_dataloaders(cfg, dataset)

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=output_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    all_results = {}
    grand_total_bytes = 0

    for arch in cfg.get("models.architectures", []):
        print(f"\n=== Architecture: {arch} ===")
        print("-- baseline (local-only) --")
        baseline_evals = run_baseline(cfg, arch, node_loaders, num_crop, num_disease, tracker, device)
        print("-- mesh (prototype + logit exchange) --")
        mesh_evals, total_bytes = run_mesh(
            cfg, arch, node_loaders, probe_loader, num_crop, num_disease, tracker, device, output_dir
        )
        grand_total_bytes += total_bytes
        gain = compute_collaboration_gain(mesh_evals, baseline_evals)

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "baseline_eval": baseline_evals,
            "mesh_eval": mesh_evals,
            "collaboration_gain": gain,
            "total_bytes_exchanged": total_bytes,
        }
        all_results[arch] = arch_result
        (output_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        print(f"macro_gain: {gain['macro_gain']}")
        print(f"worst_node_gain: {gain['worst_node_gain']}")

    (output_dir / "results_summary.json").write_text(json.dumps(all_results, indent=2))

    compute_summary = tracker.summary()
    comm_estimate = comm_estimator.estimate_all_radios(grand_total_bytes)
    overall_gain = {arch: res["collaboration_gain"]["macro_gain"] for arch, res in all_results.items()}
    write_sustainability_report(
        output_dir / "sustainability_report",
        compute_summary,
        comm_estimate,
        overall_gain,
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )
    print(f"\nDone. Results and sustainability report written to {output_dir}/")


if __name__ == "__main__":
    main()
