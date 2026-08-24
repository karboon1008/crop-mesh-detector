# tests/test_node_progress_cb.py
"""Covers the optional progress_cb hook added to Node.local_train/distill
for the Docker dashboard's per-node activity log -- it must fire with
per-batch progress info when given, and every existing no-callback caller
must keep working unchanged.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.data.plantvillage import make_subset, train_test_split_indices
from src.federated.node import Node
from src.models.factory import build_model


def _make_node(dataset):
    train_idx, test_idx = train_test_split_indices(dataset, list(range(len(dataset))), 0.3, seed=1)
    train_loader = DataLoader(make_subset(dataset, train_idx, train=True), batch_size=4, shuffle=True)
    # batch_size=3 divides the 9-sample test split evenly (3,3,3) -- batch_size=4
    # would leave a trailing batch of size 1, which BatchNorm rejects in
    # model.train() mode (a pre-existing fixture-sizing trap, not something
    # introduced by the progress_cb feature these tests exercise).
    test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=3, shuffle=False)
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    model = build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False)
    return Node("node_0", model, train_loader, test_loader, device="cpu"), train_loader, test_loader


def test_local_train_calls_progress_cb_once_per_batch(synthetic_dataset):
    node, train_loader, _ = _make_node(synthetic_dataset)
    calls = []
    node.local_train(epochs=2, lr=1e-3, progress_cb=calls.append)

    num_batches = len(train_loader)
    assert len(calls) == 2 * num_batches
    assert calls[0]["phase"] == "local_train"
    assert calls[0]["epoch"] == 1
    assert calls[0]["epochs"] == 2
    assert calls[0]["batch"] == 1
    assert calls[0]["num_batches"] == num_batches
    assert isinstance(calls[0]["loss"], float)
    assert calls[-1]["epoch"] == 2
    assert calls[-1]["batch"] == num_batches


def test_local_train_without_progress_cb_still_works(synthetic_dataset):
    node, _, _ = _make_node(synthetic_dataset)
    loss = node.local_train(epochs=1, lr=1e-3)
    assert isinstance(loss, float)


def test_distill_calls_progress_cb_for_both_kd_and_sup_phases(synthetic_dataset):
    node, _, test_loader = _make_node(synthetic_dataset)
    num_probe = len(test_loader.dataset)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    calls = []

    node.distill(
        consensus_prototypes={},
        consensus_crop_logits=torch.zeros(num_probe, num_crop),
        crop_known_mask=torch.ones(num_crop, dtype=torch.bool),
        consensus_disease_logits=torch.zeros(num_probe, num_disease),
        disease_known_mask=torch.ones(num_disease, dtype=torch.bool),
        probe_loader=test_loader,
        epochs=1,
        lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        crop_kd_weight=0.5,
        temperature=2.0,
        progress_cb=calls.append,
    )

    phases = {c["phase"] for c in calls}
    assert phases == {"distill_kd", "distill_sup"}


def test_distill_without_progress_cb_still_works(synthetic_dataset):
    node, _, test_loader = _make_node(synthetic_dataset)
    num_probe = len(test_loader.dataset)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    result = node.distill(
        consensus_prototypes={},
        consensus_crop_logits=torch.zeros(num_probe, num_crop),
        crop_known_mask=torch.ones(num_crop, dtype=torch.bool),
        consensus_disease_logits=torch.zeros(num_probe, num_disease),
        disease_known_mask=torch.ones(num_disease, dtype=torch.bool),
        probe_loader=test_loader,
        epochs=1,
        lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        crop_kd_weight=0.5,
        temperature=2.0,
    )
    assert "kd_loss" in result
