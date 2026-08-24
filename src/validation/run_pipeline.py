"""CLI entrypoint chaining the node_1 / mobilenet_v3_small validation
pipeline: train -> export -> evaluate. Each stage reads/writes files under
one output directory so stages can be re-run independently.

    python -m src.validation.run_pipeline
    python -m src.validation.run_pipeline --stage train
    python -m src.validation.run_pipeline --stage export
    python -m src.validation.run_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.data.plantvillage import PlantVillageDataset
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.node1_dataset import prepare_node1_data
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"
NODE_NAME = "node_1"


def run_train_stage(cfg: Config, output_dir: Path, epochs: int = 15, pretrained: bool = True) -> None:
    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)
    num_crop = len(eval_ds.labels.crop_classes)
    num_disease = len(eval_ds.labels.disease_classes)

    run_training(
        train_ds, train_idx, eval_ds, test_idx, num_crop, num_disease, output_dir,
        epochs=epochs, pretrained=pretrained,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": eval_ds.labels.crop_classes,
                "disease_classes": eval_ds.labels.disease_classes,
                "image_size": cfg.get("data.image_size", 160),
                "test_idx": test_idx,
            },
            indent=2,
        )
    )


def run_export_stage(cfg: Config, output_dir: Path, num_parity_samples: int = 8) -> None:
    checkpoint_path = output_dir / "checkpoint.pt"
    classes_path = output_dir / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} and/or {classes_path} not found — run the 'train' stage first."
        )
    classes = json.loads(classes_path.read_text())
    image_size = classes["image_size"]
    test_idx = classes["test_idx"]

    # Parity samples come from the same held-out split the train stage
    # already resolved (classes.json's "test_idx"), through the plain
    # unmodified PlantVillageDataset (its __getitem__ applies the clean
    # eval transform) -- no need to redo prepare_node1_data's dedup split.
    eval_ds = PlantVillageDataset(cfg.get("data.root", "data/PlantVillage"), image_size=image_size)
    sample_idx = test_idx[: min(num_parity_samples, len(test_idx))]
    parity_samples = [eval_ds[idx] for idx in sample_idx]

    export_checkpoint(
        checkpoint_path,
        classes["crop_classes"],
        classes["disease_classes"],
        image_size,
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(cfg: Config, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    classes_path = output_dir / "classes.json"
    onnx_path = output_dir / "model.onnx"
    if not manifest_path.exists() or not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} and/or {manifest_path} not found — run the 'export' stage first."
        )
    manifest = json.loads(manifest_path.read_text())
    classes = json.loads(classes_path.read_text())
    test_idx = classes["test_idx"]

    # Reads the stored split rather than recomputing prepare_node1_data's
    # dedup-aware split (which is O(n^2) in node_1's shard size) -- the
    # train stage already resolved train_idx/test_idx once.
    eval_ds = PlantVillageDataset(cfg.get("data.root", "data/PlantVillage"), image_size=classes["image_size"])
    session = onnxruntime.InferenceSession(str(onnx_path))
    run_evaluation(session, eval_ds, test_idx, manifest, MODEL_NAME, NODE_NAME, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/node_1_mobilenet_v3_small")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)

    for stage in ([args.stage] if args.stage else list(STAGES)):
        print(f"=== stage: {stage} ===")
        if stage == "train":
            run_train_stage(cfg, output_dir)
        elif stage == "export":
            run_export_stage(cfg, output_dir)
        elif stage == "evaluate":
            run_evaluate_stage(cfg, output_dir)

    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
