"""Stage 1 of the two-stage training pipeline: local-only supervised
training per node, NO knowledge exchange at all. Each node's trained
model and metrics are persisted to disk so the run can be inspected and
judged "good to go" before spending any more compute — only then does
stage 2 (src/train_mesh.py) run, warm-starting every node from exactly
these checkpoints and continuing into mesh rounds (prototype + probe-
logit exchange) rather than starting from a fresh random init.

Meant to run as its own process, independently of stage 2 — e.g. submit
this as an Isambard job, wait for it to finish, inspect
outputs/stage1_local/results_summary.json, and only then submit stage 2.

Run all architectures in config.yaml's models.architectures list:
    python -m src.train_local --config config.yaml

Or restrict to one architecture (results merge into any existing
outputs/stage1_local/results_summary.json rather than overwriting it):
    python -m src.train_local --config config.yaml --arch mobilenet_v3_small

Pass --fresh to start a new stage-1 run instead of merging into a previous one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import Config
from src.data.merged import load_merged_dataset
from src.energy.tracker import ComputeEnergyTracker
from src.models.factory import build_model, count_parameters, model_size_mb
from src.train import build_dataloaders, dataset_manifest, run_baseline, scale_epochs_by_node_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument(
        "--arch", default=None,
        help="Restrict this run to a single architecture, overriding config.yaml's models.architectures list",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Discard any existing stage1_local results/manifest instead of merging into them",
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage_dir = Path(cfg.get("output.dir", "outputs")) / "stage1_local"
    stage_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 160), cfg.get("data.seed", 42)
    )
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    print(f"Loaded {len(dataset)} images, {num_crop} crop classes, {num_disease} disease classes.")

    (stage_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": dataset.labels.crop_classes,
                "disease_classes": dataset.labels.disease_classes,
                "image_size": cfg.get("data.image_size", 160),
            },
            indent=2,
        )
    )

    probe_loader, global_test_loader, node_loaders, crop_class_weights, disease_class_weights = build_dataloaders(
        cfg, dataset
    )

    manifest_path = stage_dir / "manifest.json"
    results_summary_path = stage_dir / "results_summary.json"
    run_state_path = stage_dir / "run_state.json"
    if args.fresh:
        manifest_path.unlink(missing_ok=True)
        results_summary_path.unlink(missing_ok=True)
        run_state_path.unlink(missing_ok=True)

    # Written once (not per-architecture) — this is what src/train_mesh.py
    # checks against before it trusts these checkpoints as its starting point.
    manifest_path.write_text(json.dumps(dataset_manifest(cfg, dataset, node_loaders), indent=2))

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=stage_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )

    all_results = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    run_state = (
        json.loads(run_state_path.read_text())
        if run_state_path.exists()
        else {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0}
    )

    max_epochs_per_node = cfg.get("training.max_epochs_per_node", 40)
    baseline_epochs_per_node = scale_epochs_by_node_size(
        cfg.get("training.baseline_epochs", 10), node_loaders, max_epochs_per_node
    )
    print(f"Per-node local-training epochs (size-scaled): {baseline_epochs_per_node}")

    min_pair_accuracy = cfg.get("training.min_local_pair_accuracy", 0.3)
    flagged = []  # (arch, node_id, pair_accuracy) below the sanity floor — printed as a banner at the end

    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} (stage 1: local-only) ===")
        checkpoint_dir = stage_dir / "checkpoints" / arch
        evals = run_baseline(
            cfg, arch, node_loaders, global_test_loader,
            dataset.labels.crop_classes, dataset.labels.disease_classes, tracker, device,
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            epochs_per_node=baseline_epochs_per_node, checkpoint_dir=checkpoint_dir,
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
            resume=not args.fresh,
        )

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "local_eval": evals,
        }
        all_results[arch] = arch_result
        (stage_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        for node_id, node_eval in evals.items():
            pair_accuracy = node_eval["pair_accuracy"]
            flag = " ⚠ BELOW FLOOR" if pair_accuracy < min_pair_accuracy else ""
            print(
                f"  {node_id}: local pair_accuracy={pair_accuracy:.3f} disease_accuracy={node_eval['disease_accuracy']:.3f} "
                f"global disease_accuracy={node_eval['global']['disease_accuracy']:.3f}{flag}"
            )
            if pair_accuracy < min_pair_accuracy:
                flagged.append((arch, node_id, pair_accuracy))

    results_summary_path.write_text(json.dumps(all_results, indent=2))

    this_run_compute = tracker.summary()
    run_state["total_compute_energy_kwh"] += this_run_compute["total_compute_energy_kwh"]
    run_state["total_duration_s"] += this_run_compute["total_duration_s"]
    run_state["num_tracked_blocks"] += this_run_compute["num_tracked_blocks"]
    run_state_path.write_text(json.dumps(run_state, indent=2))

    print(
        f"\nStage 1 done. Results, manifest, and checkpoints written to {stage_dir}/ "
        f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})."
    )
    if flagged:
        print(
            f"\n⚠ {len(flagged)} node(s) below the local pair_accuracy floor "
            f"({min_pair_accuracy}) — review before proceeding to stage 2:"
        )
        for arch, node_id, pair_accuracy in flagged:
            print(f"    {arch} / {node_id}: pair_accuracy={pair_accuracy:.3f}")
    else:
        print(f"\nAll nodes cleared the local pair_accuracy floor ({min_pair_accuracy}) — safe to proceed:")
    print(
        f"    python -m src.train_mesh --config {args.config or 'config.yaml'}"
    )


if __name__ == "__main__":
    main()
