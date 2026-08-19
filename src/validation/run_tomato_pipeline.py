"""CLI entrypoint chaining the Tomato Dirichlet-mesh validation pipeline
across 3 nodes: train -> export -> evaluate. Mirrors
run_corn_pipeline.py's structure, adapted for a Dirichlet label-skew
partition over a merged multi-source Tomato pool and a single GLOBAL
held-out test set (not per-node) -- see
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.

    python -m src.validation.run_tomato_pipeline
    python -m src.validation.run_tomato_pipeline --stage train
    python -m src.validation.run_tomato_pipeline --stage export
    python -m src.validation.run_tomato_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.tomato_mesh_dataset import TomatoMeshData, prepare_tomato_mesh_data
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"


def node_output_dir(base_output_dir: Path, node_id: str) -> Path:
    return base_output_dir / f"{node_id}_{MODEL_NAME}"


def run_train_stage(
    data: TomatoMeshData, node_id: str, output_dir: Path, epochs: int = 2, pretrained: bool = True
) -> None:
    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.test_idx  # shared global test set, not per-node

    run_training(
        data.train_base,
        train_idx,
        data.eval_base,
        test_idx,
        len(data.label_map.crop_classes),
        len(data.label_map.disease_classes),
        output_dir,
        epochs=epochs,
        pretrained=pretrained,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "train_idx": train_idx,
                "test_idx": test_idx,
            },
            indent=2,
        )
    )


def run_export_stage(data: TomatoMeshData, node_id: str, output_dir: Path) -> None:
    """Reads crop_classes/disease_classes/image_size/test_idx from the
    classes.json persisted by run_train_stage rather than from `data`
    directly. `data` is freshly recomputed via prepare_tomato_mesh_data(cfg)
    on every separate CLI invocation (this module supports
    `--stage train`, then a later `--stage export`, etc. as documented in
    its module docstring) -- if config or source data drifts between
    invocations, using `data`'s freshly-recomputed scalars/indices here
    could silently disagree with the split `train` actually trained
    against. `data.eval_base` (the underlying image dataset) is still used
    directly since it's a deterministic rebuild from the same source
    directories and seed -- the risk is specifically in the persisted
    scalar/index values, not in eval_base's image content.
    """
    checkpoint_path = output_dir / "checkpoint.pt"
    classes_path = output_dir / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} and/or {classes_path} not found — run the 'train' stage first."
        )
    classes = json.loads(classes_path.read_text())
    test_idx = classes["test_idx"]

    parity_samples = [data.eval_base[i] for i in test_idx[:5]]
    export_checkpoint(
        checkpoint_path,
        classes["crop_classes"],
        classes["disease_classes"],
        classes["image_size"],
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(data: TomatoMeshData, node_id: str, output_dir: Path) -> None:
    """Same rationale as run_export_stage above: test_idx comes from the
    classes.json persisted by run_train_stage, not from the freshly
    recomputed `data.test_idx`.
    """
    manifest_path = output_dir / "manifest.json"
    classes_path = output_dir / "classes.json"
    onnx_path = output_dir / "model.onnx"
    if not manifest_path.exists() or not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} and/or {manifest_path} not found — run the 'export' stage first."
        )
    manifest = json.loads(manifest_path.read_text())
    classes = json.loads(classes_path.read_text())  # raises FileNotFoundError if missing
    test_idx = classes["test_idx"]

    session = onnxruntime.InferenceSession(str(onnx_path))
    run_evaluation(session, data.eval_base, test_idx, manifest, MODEL_NAME, node_id, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/tomato_mesh")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)
    data = prepare_tomato_mesh_data(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "test_idx": data.test_idx,
                "partition_diagnostics": data.partition_diagnostics,
            },
            indent=2,
        )
    )

    for node_id in sorted(data.per_node.keys()):
        node_dir = node_output_dir(output_dir, node_id)
        print(f"=== {node_id} ===")
        for stage in ([args.stage] if args.stage else list(STAGES)):
            print(f"  -- stage: {stage} --")
            if stage == "train":
                run_train_stage(data, node_id, node_dir)
            elif stage == "export":
                run_export_stage(data, node_id, node_dir)
            elif stage == "evaluate":
                run_evaluate_stage(data, node_id, node_dir)

    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
