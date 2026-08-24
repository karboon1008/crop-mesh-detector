"""CLI entrypoint chaining the Tomato+Apple combined Dirichlet-mesh
validation pipeline across N nodes: train -> export -> evaluate. Direct
mirror of run_apple_pipeline.py / run_tomato_pipeline.py, adapted for the
combined multi-crop pool (tomato_apple_mesh_dataset.py) and a single
GLOBAL held-out test set (not per-node).

    python -m src.validation.run_tomato_apple_pipeline
    python -m src.validation.run_tomato_apple_pipeline --stage train
    python -m src.validation.run_tomato_apple_pipeline --stage export
    python -m src.validation.run_tomato_apple_pipeline --stage evaluate
    python -m src.validation.run_tomato_apple_pipeline --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.tomato_apple_mesh_dataset import TomatoAppleMeshData, prepare_tomato_apple_mesh_data
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"


def node_output_dir(base_output_dir: Path, node_id: str) -> Path:
    return base_output_dir / f"{node_id}_{MODEL_NAME}"


def run_train_stage(
    data: TomatoAppleMeshData, node_id: str, output_dir: Path, epochs: int = 2, pretrained: bool = True
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


def run_export_stage(data: TomatoAppleMeshData, node_id: str, output_dir: Path) -> None:
    """Reads crop_classes/disease_classes/image_size/test_idx from the
    classes.json persisted by run_train_stage rather than from `data`
    directly -- same rationale as run_apple_pipeline.py's
    run_export_stage.
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


def run_evaluate_stage(data: TomatoAppleMeshData, node_id: str, output_dir: Path) -> None:
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
    parser.add_argument("--output-dir", default="outputs/validation/tomato_apple_mesh")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per node (default: 2)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)
    epochs = args.epochs if args.epochs is not None else 2
    data = prepare_tomato_apple_mesh_data(cfg)

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
                run_train_stage(data, node_id, node_dir, epochs=epochs)
            elif stage == "export":
                run_export_stage(data, node_id, node_dir)
            elif stage == "evaluate":
                run_evaluate_stage(data, node_id, node_dir)

    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
