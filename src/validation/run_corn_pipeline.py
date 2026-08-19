"""CLI entrypoint chaining the Corn disease-split validation pipeline
across all 3 nodes: train -> export -> evaluate. Generalizes
run_pipeline.py's single-node/single-model chaining to node_0/1/2, each
holding one disjoint Corn disease class plus a disjoint share of
healthy -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_corn_pipeline
    python -m src.validation.run_corn_pipeline --stage train
    python -m src.validation.run_corn_pipeline --stage export
    python -m src.validation.run_corn_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import onnxruntime
from torch.utils.data import Dataset

from src.config import Config
from src.validation.corn_mesh_dataset import CornDiseaseView, CornMeshData, prepare_corn_mesh_data
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"
NODE_LABELS = {"node_0": "common_rust", "node_1": "cercospora", "node_2": "northern_leaf_blight"}


def node_output_dir(base_output_dir: Path, node_id: str) -> Path:
    return base_output_dir / f"{node_id}_{NODE_LABELS[node_id]}_{MODEL_NAME}"


class _TrainingLabelView(Dataset):
    """Wraps a CornDiseaseView with the `.labels.class_to_crop_disease` /
    `.base.targets` attribute surface that train_mobilenet.run_training's
    internal disease_labels_for_indices helper expects from its `train_ds`
    argument (a PlantVillageDataset elsewhere). run_training reuses the
    same (dataset, indices) pair both for that class-weight lookup and for
    the Subset/DataLoader draws that actually train the model, so the
    lookup must be keyed by the SAME per-position indices (0..N-1) as the
    DataLoader uses -- CornDiseaseView itself is a compact Corn-only label
    remapper, not a PlantVillageDataset, so it doesn't expose this surface
    and calling run_training with a bare CornDiseaseView raises
    AttributeError before training even starts.
    """

    def __init__(self, view: CornDiseaseView):
        self._view = view
        compact_labels = [self._compact_disease_label(view, pos) for pos in range(len(view))]
        self.base = SimpleNamespace(targets=list(range(len(view))))
        self.labels = SimpleNamespace(
            class_to_crop_disease={pos: (0, label) for pos, label in enumerate(compact_labels)}
        )

    @staticmethod
    def _compact_disease_label(view: CornDiseaseView, pos: int) -> int:
        raw_idx = view.indices[pos]
        raw_class_idx = view.base.targets[raw_idx]
        _, disease_idx = view.base.labels.class_to_crop_disease[raw_class_idx]
        disease_name = view.base.labels.disease_classes[disease_idx]
        return view.label_map.name_to_disease_idx[disease_name]

    def __len__(self) -> int:
        return len(self._view)

    def __getitem__(self, pos: int):
        return self._view[pos]


def run_train_stage(
    data: CornMeshData, node_id: str, output_dir: Path, epochs: int = 15, pretrained: bool = True
) -> None:
    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.per_node[node_id]["test_idx"]
    train_view = CornDiseaseView(data.train_base, train_idx, data.label_map)
    eval_view = CornDiseaseView(data.eval_base, test_idx, data.label_map)

    run_training(
        _TrainingLabelView(train_view),
        list(range(len(train_idx))),
        eval_view,
        list(range(len(test_idx))),
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


def run_export_stage(data: CornMeshData, node_id: str, output_dir: Path, num_parity_samples: int = 8) -> None:
    checkpoint_path = output_dir / "checkpoint.pt"
    classes_path = output_dir / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} and/or {classes_path} not found — run the 'train' stage first."
        )
    classes = json.loads(classes_path.read_text())
    test_idx = classes["test_idx"]

    eval_view = CornDiseaseView(data.eval_base, test_idx, data.label_map)
    sample_idx = list(range(min(num_parity_samples, len(eval_view))))
    parity_samples = [eval_view[i] for i in sample_idx]

    export_checkpoint(
        checkpoint_path,
        classes["crop_classes"],
        classes["disease_classes"],
        classes["image_size"],
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(data: CornMeshData, node_id: str, output_dir: Path) -> None:
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

    session = onnxruntime.InferenceSession(str(onnx_path))
    run_evaluation(session, data.eval_base, test_idx, manifest, MODEL_NAME, node_id, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/corn_mesh")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)
    data = prepare_corn_mesh_data(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
            },
            indent=2,
        )
    )

    for node_id in ("node_0", "node_1", "node_2"):
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
