"""Positive control / Official Problem Statement's Baseline A: pools ALL
nodes' train data into one continuous dataset and trains ONE model per
architecture from scratch -- the "upload everything to a central cloud"
ceiling that src/train_mesh.py (MESH) and src/train_fedavg.py (FedAvg) are
compared against. No node isolation, no warm start, no per-round exchange
-- an independent script, not part of the two-stage pipeline, matching
§4.1 Baseline A: "a counterfactual baseline used only for comparison ...
must not serve as the initialisation weights of the submitted system or
as any part of the collaboration workflow."

Run:
    python -m src.train_centralized --config config.yaml
    python -m src.train_centralized --config config.yaml --arch efficientnet_lite0

Pass --fresh to start a new run instead of merging into a previous one.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.merged import load_merged_dataset
from src.data.plantvillage import compute_crop_class_weights, compute_disease_class_weights, make_subset
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import Node
from src.models.factory import build_model, count_parameters, model_size_mb
from src.train import build_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--arch", default=None, help="Restrict this run to a single architecture")
    parser.add_argument("--fresh", action="store_true", help="Discard any existing results instead of merging")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    stage_dir = output_dir / "centralized_pooled"
    stage_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_merged_dataset(
        cfg.get("data.root"), cfg.get("data.plantdoc_root"), cfg.get("data.image_size", 224), cfg.get("data.seed", 42)
    )
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    print(f"Loaded {len(dataset)} images, {num_crop} crop classes, {num_disease} disease classes.")

    _probe_loader, global_test_loader, node_loaders, _cw, _dw = build_dataloaders(cfg, dataset)

    # Pool every node's own train indices into one set -- this is exactly
    # the "raw data uploaded to a central cloud" counterfactual: no node
    # boundary, one model sees everything. Recompute class weights over the
    # POOLED distribution, not any single node's skewed one.
    pooled_train_idx = [i for train_loader, _ in node_loaders for i in train_loader.dataset.indices]
    print(f"Pooled train set: {len(pooled_train_idx)} images across all {len(node_loaders)} nodes (Baseline A: centralised cloud)")

    batch_size = cfg.get("training.batch_size", 32)
    pooled_train_loader = DataLoader(
        make_subset(dataset, pooled_train_idx, train=True), batch_size=batch_size, shuffle=True
    )
    crop_class_weights = (
        compute_crop_class_weights(dataset, pooled_train_idx) if cfg.get("training.crop_class_balanced", False) else None
    )
    disease_class_weights = (
        compute_disease_class_weights(dataset, pooled_train_idx) if cfg.get("training.disease_class_balanced", True) else None
    )

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True), output_dir=stage_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
        fallback_power_watts=cfg.get("energy.fallback_power_watts", 15.0),
    )

    results_summary_path = stage_dir / "results_summary.json"
    run_state_path = stage_dir / "run_state.json"
    if args.fresh:
        results_summary_path.unlink(missing_ok=True)
        run_state_path.unlink(missing_ok=True)
    all_results = json.loads(results_summary_path.read_text()) if results_summary_path.exists() else {}
    run_state = (
        json.loads(run_state_path.read_text()) if run_state_path.exists()
        else {"total_compute_energy_kwh": 0.0, "total_duration_s": 0.0, "num_tracked_blocks": 0}
    )

    architectures = [args.arch] if args.arch else cfg.get("models.architectures", [])
    for arch in architectures:
        print(f"\n=== Architecture: {arch} (centralized, pooled-data ceiling) ===")
        model = build_model(arch, num_crop, num_disease, pretrained=cfg.get("models.pretrained", True))
        # global_test_loader doubles as this arm's only "test_loader" and
        # its early-stopping validation set -- there's no single node's own
        # split for a model that was never trained per-node in the first
        # place; global_test_loader is the closest untouched, unbiased set.
        pooled_node = Node(
            "centralized", model, pooled_train_loader, global_test_loader, device=device,
            crop_classes=dataset.labels.crop_classes, disease_classes=dataset.labels.disease_classes,
            crop_loss_weight=cfg.get("training.crop_loss_weight", 1.0),
            disease_loss_weight=cfg.get("training.disease_loss_weight", 1.0),
            crop_class_weights=crop_class_weights, disease_class_weights=disease_class_weights,
            loss_type=cfg.get("training.loss_type", "cross_entropy"), focal_gamma=cfg.get("training.focal_gamma", 2.0),
            pair_class_names=dataset.base.classes, class_to_crop_disease=dataset.labels.class_to_crop_disease,
        )
        with tracker.track(f"{arch}_centralized"):
            pooled_node.local_train(
                cfg.get("training.baseline_epochs", 10), cfg.get("training.lr", 0.001),
                val_loader=global_test_loader, patience=cfg.get("training.early_stopping_patience", 5),
                weight_decay=cfg.get("training.weight_decay", 0.0),
            )

        checkpoint_dir = stage_dir / "checkpoints" / arch
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pooled_node.model.state_dict(), checkpoint_dir / "centralized.pt")

        # Evaluate this ONE pooled model against EACH node's own test split
        # (comparable, per-node, to the local-only/mesh/FedAvg arms) plus
        # the untouched global_test_loader.
        per_node_evals = {}
        for i, (_, test_loader) in enumerate(node_loaders):
            node_eval = pooled_node.evaluate(test_loader)
            node_eval["global"] = pooled_node.evaluate(global_test_loader)
            per_node_evals[f"node_{i}"] = node_eval

        sample_model = build_model(arch, num_crop, num_disease, pretrained=False)
        arch_result = {
            "architecture": arch,
            "params": count_parameters(sample_model),
            "model_size_mb": model_size_mb(sample_model),
            "pooled_train_size": len(pooled_train_idx),
            "centralized_eval": per_node_evals,
        }
        all_results[arch] = arch_result
        (stage_dir / f"results_{arch}.json").write_text(json.dumps(arch_result, indent=2))
        for node_id, node_eval in per_node_evals.items():
            print(
                f"  {node_id}: pair_accuracy={node_eval['pair_accuracy']:.3f} "
                f"global disease_accuracy={node_eval['global']['disease_accuracy']:.3f}"
            )

    results_summary_path.write_text(json.dumps(all_results, indent=2))
    this_run_compute = tracker.summary()
    run_state["total_compute_energy_kwh"] += this_run_compute["total_compute_energy_kwh"]
    run_state["total_duration_s"] += this_run_compute["total_duration_s"]
    run_state["num_tracked_blocks"] += this_run_compute["num_tracked_blocks"]
    run_state_path.write_text(json.dumps(run_state, indent=2))
    if cfg.path is not None:
        shutil.copy2(cfg.path, stage_dir / "config.yaml")

    print(f"\nCentralized (pooled) run done. Written to {stage_dir}/ "
          f"(now covering {len(all_results)} architecture(s): {list(all_results.keys())})")


if __name__ == "__main__":
    main()
