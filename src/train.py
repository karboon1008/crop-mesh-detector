"""Entry point: for every configured architecture, trains an
aligned-budget local-only baseline and the decentralised mesh (prototype
+ probe-logit exchange only — no raw images, no weights) on PlantVillage,
then reports the collaboration gain and a compute + communication
sustainability accounting.

Run all architectures in config.yaml's models.architectures list in one go:
    python -m src.train --config config.yaml

Or restrict a single run to one architecture (e.g. to split a full sweep
across several shorter Colab sessions to stay under free-tier GPU usage
limits) — results merge into any existing outputs/results_summary.json
rather than overwriting it, so running each architecture separately still
produces one combined comparison at the end:
    python -m src.train --config config.yaml --arch mobilenet_v3_small
    python -m src.train --config config.yaml --arch efficientnet_lite0
    python -m src.train --config config.yaml --arch mobilevit_xxs

Pass --fresh to start a new sweep instead of merging into a previous one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import (
    carve_global_test_set,
    carve_public_probe_set,
    compute_disease_class_weights,
    load_full_dataset,
    make_subset,
    partition_nodes,
    train_test_split_indices,
)
from src.energy.tracker import (
    CommunicationCostEstimator,
    ComputeEnergyTracker,
    sweep_totals_from_emissions_csv,
    write_sustainability_report,
)
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb
from src.reporting import build_per_class_rows, build_round_log_rows, plot_training_curves, write_csv


def build_dataloaders(cfg: Config, dataset):
    probe_idx, remaining_idx = carve_public_probe_set(
        dataset,
        cfg.get("data.probe_set_fraction", 0.05),
        cfg.get("data.seed", 42),
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    # measure of whether a node's model actually generalizes.
    global_test_idx, remaining_idx = carve_global_test_set(
        dataset,
        remaining_idx,
        cfg.get("data.global_test_fraction", 0.05),
        cfg.get("data.seed", 42),
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "by_crop"),
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    batch_size = cfg.get("training.batch_size", 32)
    probe_loader = DataLoader(make_subset(dataset, probe_idx), batch_size=batch_size, shuffle=False)
    global_test_loader = DataLoader(make_subset(dataset, global_test_idx), batch_size=batch_size, shuffle=False)

    node_loaders = []
    disease_class_weights = []
    for shard in shards:
        train_idx, test_idx = train_test_split_indices(
            shard, cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
        )
        train_loader = DataLoader(
            make_subset(dataset, train_idx, train=True), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
        node_loaders.append((train_loader, test_loader))
        disease_class_weights.append(
            compute_disease_class_weights(dataset, train_idx)
            if cfg.get("training.disease_class_balanced", True)
            else None
        )
    return probe_loader, global_test_loader, node_loaders, disease_class_weights


def run_baseline(
    cfg, arch, node_loaders, global_test_loader, crop_classes, disease_classes, tracker, device,
    disease_class_weights=None,
):
    """Local-only training, no exchange at all — the comparison point
    the collaboration gain is measured against.
    """
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    evals = {}
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        node_id = f"node_{i}"
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        node = Node(
            node_id, model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            disease_class_weights=disease_class_weights[i],
        )
        with tracker.track(f"{arch}_baseline_{node_id}"):
            node.local_train(cfg.get("training.baseline_epochs", 10), cfg.get("training.lr", 0.001))
        # top-level metrics: this node's own (skewed) local test split.
        # "global": the untouched global_test_loader — the only number
        # that reflects whether this node can classify the full catalogue.
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        evals[node_id] = node_eval
    return evals


def run_mesh(
    cfg, arch, node_loaders, probe_loader, global_test_loader, crop_classes, disease_classes, tracker, device,
    output_dir, disease_class_weights=None,
):
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        nodes.append(Node(
            f"node_{i}", model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            disease_class_weights=disease_class_weights[i],
        ))

    mesh = MeshSimulator(
        nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    total_bytes = 0
    round_logs = []
    for r in range(cfg.get("training.rounds", 5)):
        with tracker.track(f"{arch}_mesh_round_{r}"):
            local_epochs = cfg.get(
                f"training.local_epochs_per_round_overrides.{arch}",
                cfg.get("training.local_epochs_per_round", 2),
            )
            round_log = mesh.run_round(
                r,
                local_epochs=local_epochs,
                distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
                lr=cfg.get("training.lr", 0.001),
                distill_lr=cfg.get("training.distill_lr", 0.0005),
                proto_weight=cfg.get("training.proto_weight", 0.5),
                kd_weight=cfg.get("training.kd_weight", 0.5),
                temperature=cfg.get("training.kd_temperature", 2.0),
            )
        total_bytes += round_log.total_bytes_exchanged
        round_logs.append(round_log)
        print(f"  round {r}: {round_log.total_bytes_exchanged} bytes exchanged")

    round_log_dicts = [dataclasses.asdict(rl) for rl in round_logs]
    trend_rows = build_round_log_rows(round_log_dicts)
    per_class_rows = build_per_class_rows(
        round_log_dicts, [("pre", "pre_distill_eval"), ("post", "per_node_eval")]
    )
    (output_dir / f"round_logs_{arch}.json").write_text(
        json.dumps({"rounds": round_log_dicts, "trend": trend_rows, "per_class_trend": per_class_rows}, indent=2)
    )
    if cfg.get("output.save_plots", True):
        write_csv(trend_rows, output_dir / f"trend_{arch}.csv")
        write_csv(per_class_rows, output_dir / f"trend_{arch}_per_class.csv")
        plot_training_curves(
            trend_rows, output_dir / "plots" / f"{arch}_mesh_training_curves.png",
            f"{arch}: mesh training curves",
        )

    final_evals = {}
    for node in nodes:
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        final_evals[node.node_id] = node_eval

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    return final_evals, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch",
        default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing results_summary.json / run_state.json instead of merging into them",
    )
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

    probe_loader, global_test_loader, node_loaders, disease_class_weights = build_dataloaders(cfg, dataset)

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

    results_summary_path = output_dir / "results_summary.json"
    run_state_path = output_dir / "run_state.json"
    if args.fresh:
        results_summary_path.unlink(missing_ok=True)
        run_state_path.unlink(missing_ok=True)

    all_results = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    run_state = (
        json.loads(run_state_path.read_text())
        if run_state_path.exists()
        else {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0, "total_bytes_exchanged": 0}
    )
    grand_total_bytes = 0

    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} ===")
        print("-- baseline (local-only) --")
        baseline_evals = run_baseline(
            cfg, arch, node_loaders, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device,
            disease_class_weights=disease_class_weights,
        )
        print("-- mesh (prototype + logit exchange) --")
        mesh_evals, total_bytes = run_mesh(
            cfg, arch, node_loaders, probe_loader, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device, output_dir,
            disease_class_weights=disease_class_weights,
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
        if "global_macro_gain" in gain:
            print(f"global_macro_gain: {gain['global_macro_gain']}")
            print(f"global_worst_node_gain: {gain['global_worst_node_gain']}")

    results_summary_path.write_text(json.dumps(all_results, indent=2))

    # Accumulate this run's compute energy/bytes on top of any prior separate
    # run(s) (e.g. one Colab session per architecture), so the sustainability
    # report reflects the whole sweep rather than just the architecture(s)
    # trained in this particular invocation.
    this_run_compute = tracker.summary()
    run_state["total_compute_energy_kwh"] += this_run_compute["total_compute_energy_kwh"]
    run_state["total_duration_s"] += this_run_compute["total_duration_s"]
    run_state["num_tracked_blocks"] += this_run_compute["num_tracked_blocks"]
    run_state["total_bytes_exchanged"] += grand_total_bytes
    run_state_path.write_text(json.dumps(run_state, indent=2))

    comm_estimate = comm_estimator.estimate_all_radios(run_state["total_bytes_exchanged"])
    overall_gain = {arch: res["collaboration_gain"]["macro_gain"] for arch, res in all_results.items()}
    write_sustainability_report(
        output_dir / "sustainability_report",
        {k: v for k, v in run_state.items() if k != "total_bytes_exchanged"},
        comm_estimate,
        overall_gain,
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(output_dir),
    )
    print(f"\nDone. Results and sustainability report written to {output_dir}/ "
          f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})")


if __name__ == "__main__":
    main()
