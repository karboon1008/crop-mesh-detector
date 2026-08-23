"""Preview the Tomato+Apple combined Dirichlet-mesh data split (per-node,
per-crop, per-class counts) without running any training -- calls the
exact same prepare_tomato_apple_mesh_data(cfg) that
run_tomato_apple_pipeline.py / run_tomato_apple_knowledge_transfer.py use,
so the printed split is guaranteed to match what a real run would train
against for the current config.yaml (tomato_apple_mesh.dirichlet_alpha,
tomato_apple_mesh.exclude_diseases, etc.). Direct mirror of
scripts/preview_apple_dirichlet_split.py, extended with a per-crop
breakdown since this pool spans two crops.

    python -m scripts.preview_tomato_apple_dirichlet_split
    python -m scripts.preview_tomato_apple_dirichlet_split --config config.yaml
"""

from __future__ import annotations

import argparse

from src.config import Config
from src.validation.tomato_apple_mesh_dataset import prepare_tomato_apple_mesh_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    alpha = cfg.get("tomato_apple_mesh.dirichlet_alpha", 0.3)
    excluded = cfg.get("tomato_apple_mesh.exclude_diseases", []) or []

    data = prepare_tomato_apple_mesh_data(cfg)
    disease_classes = data.label_map.disease_classes
    crop_classes = data.label_map.crop_classes

    print(f"alpha={alpha}  excluded_diseases={excluded}")
    print(f"crop_classes={crop_classes}")
    print(f"disease_classes ({len(disease_classes)}): {disease_classes}")
    print(f"global test set: {len(data.test_idx)} images | probe set: {len(data.probe_idx)} images")
    print()

    node_ids = sorted(data.per_node.keys())

    print("Per-node crop mix:")
    crop_header = f"{'Node':<8}{'Samples':>10}" + "".join(f"{c:>14}" for c in crop_classes) + f"{'DominantCrop':>16}"
    print(crop_header)
    print("-" * len(crop_header))
    for node_id in node_ids:
        d = data.partition_diagnostics[node_id]
        crop_cells = "".join(f"{d['per_crop_counts'][c]:>14}" for c in crop_classes)
        dominant = f"{d['dominant_crop']} ({d['dominant_crop_fraction']*100:.0f}%)"
        print(f"{node_id:<8}{d['num_samples']:>10}{crop_cells}{dominant:>16}")

    print()
    print("Per-node disease-class summary:")
    header = f"{'Node':<8}{'Samples':>10}{'Dominant':>20}{'Share':>8}{'JSdiv':>8}{'Classes':>10}"
    print(header)
    print("-" * len(header))
    for node_id in node_ids:
        d = data.partition_diagnostics[node_id]
        classes_present = f"{d['num_classes_present']}/{len(disease_classes)}"
        print(
            f"{node_id:<8}{d['num_samples']:>10}{d['dominant_class']:>20}"
            f"{d['dominant_class_fraction']*100:>7.1f}%{d['js_divergence_from_uniform']:>8.4f}{classes_present:>10}"
        )

    print()
    print("Full per-class counts (aggregated across both crops; 'healthy' spans Apple+Tomato):")
    col_header = f"{'Class':<40}" + "".join(f"{nid:>12}" for nid in node_ids) + f"{'pooled':>12}"
    print(col_header)
    print("-" * len(col_header))
    for cls in disease_classes:
        counts = [data.partition_diagnostics[nid]["per_class_counts"][cls] for nid in node_ids]
        pooled = sum(counts)
        row = f"{cls:<40}" + "".join(f"{c:>12}" for c in counts) + f"{pooled:>12}"
        print(row)

    print()
    for crop in crop_classes:
        print(f"Full per-class counts within {crop} only:")
        col_header = f"{'Class':<40}" + "".join(f"{nid:>12}" for nid in node_ids) + f"{'pooled':>12}"
        print(col_header)
        print("-" * len(col_header))
        for cls in disease_classes:
            counts = [data.partition_diagnostics[nid]["per_crop_disease_counts"][crop][cls] for nid in node_ids]
            pooled = sum(counts)
            if pooled == 0:
                continue
            row = f"{cls:<40}" + "".join(f"{c:>12}" for c in counts) + f"{pooled:>12}"
            print(row)
        print()

    zero_sample = {
        nid: [cls for cls in disease_classes if data.partition_diagnostics[nid]["per_class_counts"][cls] == 0]
        for nid in node_ids
    }
    zero_sample = {nid: names for nid, names in zero_sample.items() if names}
    if zero_sample:
        print(f"WARNING -- zero-sample classes by node: {zero_sample}")
    else:
        print("Every node has >=1 training sample of every remaining disease class.")

    zero_crop = {
        nid: [c for c in crop_classes if data.partition_diagnostics[nid]["per_crop_counts"][c] == 0]
        for nid in node_ids
    }
    zero_crop = {nid: names for nid, names in zero_crop.items() if names}
    if zero_crop:
        print(f"WARNING -- zero-sample CROPS by node: {zero_crop}")
    else:
        print("Every node has >=1 training sample of every crop.")


if __name__ == "__main__":
    main()
