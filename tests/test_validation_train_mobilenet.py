from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.node1_dataset import get_node1_indices
from src.validation.train_mobilenet import compute_class_weights, disease_labels_for_indices


def test_compute_class_weights_upweights_rare_classes_and_zeros_absent_ones():
    weights = compute_class_weights(labels=[0, 0, 0, 1], num_classes=3)

    assert weights[2] == pytest.approx(0.0)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(1.5)
    assert weights[1] > weights[0] > 0


def test_disease_labels_for_indices_matches_label_maps(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    node1_indices = get_node1_indices(dataset, cfg)

    disease_labels = disease_labels_for_indices(dataset, node1_indices)

    assert len(disease_labels) == len(node1_indices)
    for idx, disease_label in zip(node1_indices, disease_labels):
        class_idx = dataset.base.targets[idx]
        _, expected_disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        assert disease_label == expected_disease_idx


import json

from src.validation.node1_dataset import prepare_node1_data
from src.validation.train_mobilenet import run_training


def test_run_training_writes_log_and_best_checkpoint(node1_scoped_config, tmp_path):
    cfg, root = node1_scoped_config
    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)
    num_crop = len(eval_ds.labels.crop_classes)
    num_disease = len(eval_ds.labels.disease_classes)
    output_dir = tmp_path / "run"

    result = run_training(
        train_ds,
        train_idx,
        eval_ds,
        test_idx,
        num_crop,
        num_disease,
        output_dir,
        epochs=2,
        batch_size=4,
        pretrained=False,
        device="cpu",
    )

    assert (output_dir / "checkpoint.pt").exists()
    log = json.loads((output_dir / "training_log.json").read_text())
    assert len(log) == 2
    assert set(log[0].keys()) == {"epoch", "train_loss", "test_crop_accuracy", "test_disease_accuracy"}
    assert result["best_disease_accuracy"] == max(e["test_disease_accuracy"] for e in log)
