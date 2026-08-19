"""Stage 2 of the two-stage training pipeline: knowledge transfer (mesh
prototype + probe-logit exchange) continuing FROM stage 1's saved,
already-verified local models (src/train_local.py) — never a fresh random
init. Rebuilds the exact same dataset split as stage 1 (same config,
same seed) and checks it against stage 1's manifest before trusting its
checkpoints, so a config drifted between the two runs fails loudly
instead of silently pairing a checkpoint with the wrong node's data.

Meant to run as its own process, after stage 1 has been inspected and
judged good enough to continue:
    python -m src.train_local --config config.yaml     # stage 1, run + inspect first
    python -m src.train_mesh --config config.yaml       # stage 2, only once stage 1 looks good

Or restrict to one architecture (results merge into any existing
outputs/stage2_mesh/results_summary.json rather than overwriting it):
    python -m src.train_mesh --config config.yaml --arch mobilenet_v3_small

Pass --fresh to start a new stage-2 run instead of merging into a previous one.
"""

from __future__ import annotations

import argparse
import json
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
from src.models.factory import build_model, count_parameters, model_size_mb
from src.train import build_dataloaders, check_manifest_match, dataset_manifest, run_mesh, scale_epochs_by_node_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch", default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Discard any existing stage2_mesh results/run_state instead of merging into them",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    stage1_dir = output_dir / "stage1_local"
    stage2_dir = output_dir / "stage2_mesh"
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
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=stage2_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )

    results_summary_path = stage2_dir / "results_summary.json"
    run_state_path = stage2_dir / "run_state.json"
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

    max_epochs_per_node = cfg.get("training.max_epochs_per_node", 40)
    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} (stage 2: mesh, warm-started from stage 1) ===")
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
        print(f"Per-node mesh local_epochs (size-scaled): {local_epochs_per_node}")

        mesh_evals, total_bytes, pre_round_evals = run_mesh(
            cfg, arch, node_loaders, probe_loader, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device, stage2_dir,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            local_epochs_per_node=local_epochs_per_node, warm_start_checkpoint_dir=stage1_checkpoint_dir,
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
        )
        grand_total_bytes += total_bytes

        # NOTE on interpreting this gain: pre_round_evals is stage 1's model
        # evaluated fresh right after loading, before round 0 does anything;
        # mesh_evals is after all mesh rounds. Each mesh round does its own
        # local_train on top of the warm start BEFORE distilling, so this
        # gain reflects "continuing training via the mesh procedure" as a
        # whole (more local epochs + distillation together) — it does NOT
        # isolate distillation's effect alone from just training longer
        # locally. For that, you'd need a control arm that continues
        # LOCAL-ONLY training (no exchange) from the same stage-1
        # checkpoint for the same additional epoch budget.
        gain = compute_collaboration_gain(mesh_evals, pre_round_evals)

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "pre_round_eval": pre_round_evals,
            "mesh_eval": mesh_evals,
            "step1_to_step2_gain": gain,
            "total_bytes_exchanged": total_bytes,
        }
        all_results[arch] = arch_result
        (stage2_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        print(f"macro_gain (post-mesh vs stage-1 checkpoint): {gain['macro_gain']}")
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

    comm_estimate = comm_estimator.estimate_all_radios(run_state["total_bytes_exchanged"])
    overall_gain = {arch: res["step1_to_step2_gain"]["macro_gain"] for arch, res in all_results.items()}
    write_sustainability_report(
        stage2_dir / "sustainability_report",
        {k: v for k, v in run_state.items() if k != "total_bytes_exchanged"},
        comm_estimate,
        overall_gain,
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
        emissions_csv_totals=sweep_totals_from_emissions_csv(stage2_dir),
    )
    print(
        f"\nStage 2 done. Results and sustainability report written to {stage2_dir}/ "
        f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})"
    )


if __name__ == "__main__":
    main()
