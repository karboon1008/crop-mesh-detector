"""Stage 2 alternative: classic FedAvg (client-server weight averaging)
continuing FROM stage 1's saved local models (src/train_local.py), instead
of the mesh's prototype/logit exchange (src/train_mesh.py). Same warm
start, same round/epoch budget as the mesh arm for the same architecture
-- the only thing that differs between this script and train_mesh.py is
the collaboration mechanism itself, so the two are a fair "which knowledge
exchange scheme" comparison, not a different experiment.

Meant to run after stage 1, same as train_mesh.py:
    python -m src.train_local --config config.yaml     # stage 1
    python -m src.train_fedavg --config config.yaml     # this script

Or restrict to one architecture (results merge into any existing
outputs/stage2_fedavg/results_summary.json rather than overwriting it):
    python -m src.train_fedavg --config config.yaml --arch mobilenet_v3_small

Pass --fresh to start a new run instead of merging into a previous one.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch

from src.config import Config
from src.data.merged import load_merged_dataset
from src.energy.tracker import (
    CommunicationCostEstimator,
    ComputeEnergyTracker,
    sweep_totals_from_emissions_csv,
    write_sustainability_report,
)
from src.evaluate import compute_collaboration_gain
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb
from src.train import build_dataloaders, check_manifest_match, dataset_manifest, scale_epochs_by_node_size


def federated_average(state_dicts: list[dict], weights: list[int]) -> dict:
    total = sum(weights)
    new_state = copy.deepcopy(state_dicts[0])
    for key in new_state:
        stacked = torch.stack(
            [sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights)], dim=0
        )
        summed = stacked.sum(dim=0)
        new_state[key] = (
            summed.to(state_dicts[0][key].dtype)
            if state_dicts[0][key].dtype.is_floating_point
            else summed.round().to(state_dicts[0][key].dtype)
        )
    return new_state


def run_fedavg(
    cfg, arch, node_loaders, global_test_loader, crop_classes, disease_classes, tracker, device, output_dir,
    crop_class_weights=None, disease_class_weights=None, local_epochs_per_node=None,
    warm_start_checkpoint_dir=None, pair_class_names=None, class_to_crop_disease=None,
):
    """Mirrors run_mesh's signature/warm-start/return shape exactly so the
    two are a controlled comparison. Every round: each node trains locally
    for its (size-scaled) epoch count starting from the current global
    weights, then all node weights are averaged (sample-count-weighted)
    into the new global model, broadcast back to every node.
    """
    crop_class_weights = crop_class_weights or [None] * len(node_loaders)
    disease_class_weights = disease_class_weights or [None] * len(node_loaders)
    default_local_epochs = cfg.get(
        f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
    )
    local_epochs_per_node = local_epochs_per_node or [default_local_epochs] * len(node_loaders)
    warm_start_checkpoint_dir = Path(warm_start_checkpoint_dir) if warm_start_checkpoint_dir else None

    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        node_id = f"node_{i}"
        model = build_model(
            arch, len(crop_classes), len(disease_classes), pretrained=cfg.get("models.pretrained", True)
        )
        if warm_start_checkpoint_dir is not None:
            ckpt_path = warm_start_checkpoint_dir / f"{node_id}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Missing stage-1 checkpoint for {node_id} at {ckpt_path} — run "
                    f"'python -m src.train_local --config ...' for this architecture first."
                )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        nodes.append(Node(
            node_id, model, train_loader, test_loader, device=device,
            crop_classes=crop_classes, disease_classes=disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            crop_class_weights=crop_class_weights[i], disease_class_weights=disease_class_weights[i],
            loss_type=cfg.get("training.loss_type", "cross_entropy"), focal_gamma=cfg.get("training.focal_gamma", 2.0),
            pair_class_names=pair_class_names, class_to_crop_disease=class_to_crop_disease,
        ))

    pre_round_evals = {}
    for node in nodes:
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        pre_round_evals[node.node_id] = node_eval

    node_weights = [len(train_loader.dataset) for train_loader, _ in node_loaders]
    global_state = copy.deepcopy(nodes[0].model.state_dict())
    for node in nodes:
        node.model.load_state_dict(global_state)
    model_bytes = count_parameters(nodes[0].model) * 4  # FP32 state_dict

    total_bytes = 0
    for r in range(cfg.get("training.rounds", 5)):
        with tracker.track(f"{arch}_fedavg_round_{r}"):
            for node, epochs in zip(nodes, local_epochs_per_node):
                node.model.load_state_dict(global_state)
                node.local_train(epochs, cfg.get("training.lr", 0.001))
            client_states = [node.model.state_dict() for node in nodes]
            global_state = federated_average(client_states, node_weights)
            for node in nodes:
                node.model.load_state_dict(global_state)

        # Client-server bytes: each node uploads its full model once and
        # receives the aggregated global model back once per round --
        # unlike the mesh's peer-to-peer gossip broadcast, FedAvg goes
        # through one central aggregator, so cost is linear in node count.
        round_bytes = 2 * len(nodes) * model_bytes
        total_bytes += round_bytes
        print(f"  round {r}: {round_bytes} bytes exchanged (client-server FedAvg)")

    final_evals = {}
    for node in nodes:
        node_eval = node.evaluate()
        node_eval["global"] = node.evaluate(global_test_loader)
        final_evals[node.node_id] = node_eval

    checkpoint_dir = output_dir / "checkpoints" / arch
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        torch.save(node.model.state_dict(), checkpoint_dir / f"{node.node_id}.pt")

    return final_evals, total_bytes, pre_round_evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch", default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Discard any existing stage2_fedavg results/run_state instead of merging into them",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    stage1_dir = output_dir / "stage1_local"
    stage2_dir = output_dir / "stage2_fedavg"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    stage1_manifest_path = stage1_dir / "manifest.json"
    if not stage1_manifest_path.exists():
        raise FileNotFoundError(
            f"No stage-1 manifest at {stage1_manifest_path} — run 'python -m src.train_local "
            f"--config {args.config or 'config.yaml'}' first."
        )

    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 224), cfg.get("data.seed", 42)
    )
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    print(f"Loaded {len(dataset)} images, {num_crop} crop classes, {num_disease} disease classes.")

    probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights = build_dataloaders(
        cfg, dataset
    )

    check_manifest_match(
        json.loads(stage1_manifest_path.read_text()), dataset_manifest(cfg, dataset, node_loaders), stage1_manifest_path
    )
    print(f"Stage 1 manifest matches — safe to warm-start from {stage1_dir}/checkpoints/.")

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True), output_dir=stage2_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}), cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    results_summary_path = stage2_dir / "results_summary.json"
    run_state_path = stage2_dir / "run_state.json"
    if args.fresh:
        results_summary_path.unlink(missing_ok=True)
        run_state_path.unlink(missing_ok=True)

    all_results = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    run_state = (
        json.loads(run_state_path.read_text()) if run_state_path.exists()
        else {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0, "total_bytes_exchanged": 0}
    )
    grand_total_bytes = 0

    max_epochs_per_node = cfg.get("training.max_epochs_per_node", 40)
    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} (stage 2: FedAvg, warm-started from stage 1) ===")
        stage1_checkpoint_dir = stage1_dir / "checkpoints" / arch
        if not stage1_checkpoint_dir.exists():
            raise FileNotFoundError(
                f"No stage-1 checkpoints for {arch!r} at {stage1_checkpoint_dir} — run "
                f"'python -m src.train_local --config {args.config or 'config.yaml'} --arch {arch}' first."
            )
        base_local_epochs = cfg.get(
            f"training.local_epochs_per_round_overrides.{arch}", cfg.get("training.local_epochs_per_round", 2)
        )
        local_epochs_per_node = scale_epochs_by_node_size(base_local_epochs, node_loaders, max_epochs_per_node)
        print(f"Per-node FedAvg local_epochs (size-scaled): {local_epochs_per_node}")

        fedavg_evals, total_bytes, pre_round_evals = run_fedavg(
            cfg, arch, node_loaders, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device, stage2_dir,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            local_epochs_per_node=local_epochs_per_node, warm_start_checkpoint_dir=stage1_checkpoint_dir,
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
        )
        grand_total_bytes += total_bytes
        gain = compute_collaboration_gain(fedavg_evals, pre_round_evals)

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "pre_round_eval": pre_round_evals,
            "fedavg_eval": fedavg_evals,
            "step1_to_step2_gain": gain,
            "total_bytes_exchanged": total_bytes,
        }
        all_results[arch] = arch_result
        (stage2_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        print(f"macro_gain (post-FedAvg vs stage-1 checkpoint): {gain['macro_gain']}")
        print(f"worst_node_gain: {gain['worst_node_gain']}")
        if "global_macro_gain" in gain:
            print(f"global_macro_gain: {gain['global_macro_gain']}")
            print(f"global_worst_node_gain: {gain['global_worst_node_gain']}")

    results_summary_path.write_text(json.dumps(all_results, indent=2))

    this_run_compute = tracker.summary()
    run_state["total_compute_energy_kwh"] += this_run_compute["total_compute_energy_kwh"]
    run_state["total_duration_s"] += this_run_compute["total_duration_s"]
    run_state["num_tracked_blocks"] += this_run_compute["num_tracked_blocks"]
    run_state["total_bytes_exchanged"] += grand_total_bytes
    run_state_path.write_text(json.dumps(run_state, indent=2))
    if cfg.path is not None:
        shutil.copy2(cfg.path, stage2_dir / "config.yaml")

    comm_estimate = comm_estimator.estimate_all_radios(run_state["total_bytes_exchanged"])
    overall_gain = {arch: res["step1_to_step2_gain"]["macro_gain"] for arch, res in all_results.items()}
    write_sustainability_report(
        stage2_dir / "sustainability_report",
        {k: v for k, v in run_state.items() if k != "total_bytes_exchanged"},
        comm_estimate, overall_gain, cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(stage2_dir),
    )
    print(
        f"\nStage 2 (FedAvg) done. Results and sustainability report written to {stage2_dir}/ "
        f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})"
    )


if __name__ == "__main__":
    main()
