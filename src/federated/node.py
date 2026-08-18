"""A single simulated farm node: trains only on its own private shard,
and exposes knowledge (prototypes + probe-set logits) without ever
exposing images, labels-in-bulk, or model weights to anyone else.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.metrics import head_metrics

Prototypes = dict[tuple[str, int], torch.Tensor]

@dataclass
class KnowledgePayload:
    """Everything a node sends to its peers in one round. Deliberately
    small and non-invertible: mean embeddings and soft probabilities,
    never a raw image, gradient, or weight tensor.
    """

    prototypes: Prototypes  # mean future embedding per class (crop type and disease type)
    crop_logits: torch.Tensor  # (num_probe, num_crop_classes)
    disease_logits: torch.Tensor  # (num_probe, num_disease_classes)

    # estimates communication cost for measuring bandwidth efficiency
    def size_bytes(self) -> int:
        proto_bytes = sum(v.numel() * 4 for v in self.prototypes.values())
        logit_bytes = (self.crop_logits.numel() + self.disease_logits.numel()) * 4
        return proto_bytes + logit_bytes


class Node:
    def __init__(
        self,
        node_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        active: bool = True,
        crop_classes: list[str] | None = None,
        disease_classes: list[str] | None = None,
        crop_loss_weight: float = 1.0,
        disease_loss_weight: float = 1.0,
        crop_class_weights: torch.Tensor | None = None,
        disease_class_weights: torch.Tensor | None = None,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        pair_class_names: list[str] | None = None,
        class_to_crop_disease: dict[int, tuple[int, int]] | None = None,
    ):
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.active = active
        # class names for readable per-class metrics/confusion matrices in
        # evaluate() — falls back to positional labels if not provided.
        self.crop_classes = crop_classes or [f"crop_{i}" for i in range(model.crop_head.out_features)]
        self.disease_classes = disease_classes or [f"disease_{i}" for i in range(model.disease_head.out_features)]
        self.crop_loss_weight = crop_loss_weight
        self.disease_loss_weight = disease_loss_weight
        self.crop_class_weights = (
            crop_class_weights.to(device) if crop_class_weights is not None else None
        )
        self.disease_class_weights = (
            disease_class_weights.to(device) if disease_class_weights is not None else None
        )
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        # crop_accuracy and disease_accuracy are scored independently, so a
        # model can get disease_accuracy right by pattern-matching lesion
        # texture while guessing the wrong crop, and that never shows up.
        # pair_class_names/class_to_crop_disease (dataset.base.classes /
        # dataset.labels.class_to_crop_disease) let evaluate() also score
        # the JOINT (crop, disease) pair per sample — the real test of
        # whether mesh knowledge transfer generalizes a node to another
        # node's crop, not just its disease vocabulary. Optional: without
        # them, evaluate() only reports the two independent accuracies.
        self.pair_class_names = pair_class_names
        self.pair_to_class_idx = (
            {pair: idx for idx, pair in class_to_crop_disease.items()} if class_to_crop_disease else None
        )

    # local supervised training (data never leaves this method)
    def local_train(
        self, epochs: int, lr: float, val_loader: DataLoader | None = None, patience: int | None = None,
    ) -> float:
        """If `val_loader` and `patience` are both given, evaluates this
        node's pair_accuracy (crop AND disease both correct — see
        evaluate()) on `val_loader` after every epoch, keeps the
        best-scoring epoch's weights, and stops early once `patience`
        epochs pass with no improvement — so a node whose epoch count was
        scaled up for having a small local shard (see
        src.train.scale_epochs_by_node_size) doesn't just overfit through
        all of them. The mesh's own per-round local_train calls (1-3
        epochs) don't pass these, so this is opt-in and only used by
        src.train.run_baseline's stage-1 local-only training.
        """
        early_stopping = val_loader is not None and patience is not None
        if early_stopping and self.pair_to_class_idx is None:
            raise ValueError(
                "local_train early stopping needs pair_class_names/class_to_crop_disease set on this Node"
            )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(self.train_loader), epochs=epochs
        )
        best_score, best_state, epochs_without_improvement = -1.0, None, 0
        total_loss, total_batches = 0.0, 0
        for _epoch in range(epochs):
            self.model.train()
            for images, crop_labels, disease_labels in self.train_loader:
                images = images.to(self.device)
                crop_labels = crop_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)

                optimizer.zero_grad()
                crop_logits, disease_logits = self.model(images)
                loss = self.crop_loss_weight * _classification_loss(
                    crop_logits, crop_labels, self.loss_type, self.crop_class_weights, self.focal_gamma
                ) + self.disease_loss_weight * _classification_loss(
                    disease_logits, disease_labels, self.loss_type, self.disease_class_weights, self.focal_gamma
                )
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_batches += 1

            if early_stopping:
                val_score = self.evaluate(val_loader)["pair_accuracy"]
                if val_score > best_score:
                    best_score = val_score
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

        if early_stopping and best_state is not None:
            self.model.load_state_dict(best_state)
        return total_loss / max(1, total_batches)

    # knowledge extraction: prototypes + public-probe logits
    # no_grad - no training session here
    @torch.no_grad()
    def compute_prototypes(self) -> Prototypes:
        self.model.eval()
        crop_sums: dict[int, torch.Tensor] = {}
        crop_counts: dict[int, int] = {}
        disease_sums: dict[int, torch.Tensor] = {}
        disease_counts: dict[int, int] = {}

        for images, crop_labels, disease_labels in self.train_loader:
            images = images.to(self.device)
            # extract features
            _, _, feats = self.model(images, return_features=True)
            feats = feats.cpu()
            for f, c, d in zip(feats, crop_labels, disease_labels):
                c, d = int(c), int(d)
                # sum the crop/disease feat and labels
                crop_sums[c] = crop_sums.get(c, torch.zeros_like(f)) + f
                crop_counts[c] = crop_counts.get(c, 0) + 1
                disease_sums[d] = disease_sums.get(d, torch.zeros_like(f)) + f
                disease_counts[d] = disease_counts.get(d, 0) + 1

        prototypes: Prototypes = {}
        # compute average
        for c, s in crop_sums.items():
            prototypes[("crop", c)] = s / crop_counts[c]
        for d, s in disease_sums.items():
            prototypes[("disease", d)] = s / disease_counts[d]
        return prototypes

    @torch.no_grad()
    def compute_probe_logits(self, probe_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        crop_logits_all, disease_logits_all = [], []
        ## only load images
        for images, _, _ in probe_loader:
            images = images.to(self.device)
            # run inference
            crop_logits, disease_logits = self.model(images)
            crop_logits_all.append(crop_logits.cpu())
            disease_logits_all.append(disease_logits.cpu())
        return torch.cat(crop_logits_all, dim=0), torch.cat(disease_logits_all, dim=0)

    # pack knowledge
    def compute_knowledge(self, probe_loader: DataLoader) -> KnowledgePayload:
        prototypes = self.compute_prototypes()
        crop_logits, disease_logits = self.compute_probe_logits(probe_loader)
        return KnowledgePayload(prototypes, crop_logits, disease_logits)

    # knowledge distillation towards a peer consensus (no peer data ever seen) 
    def distill(
        self,
        consensus_prototypes: Prototypes,
        consensus_crop_logits: torch.Tensor,
        consensus_disease_logits: torch.Tensor,
        probe_loader: DataLoader, # shared public probe dataset
        epochs: int,
        lr: float, # learning rate
        proto_weight: float,
        kd_weight: float,
        temperature: float,
    ) -> dict[str, float]:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        total_steps = epochs * (len(probe_loader) + len(self.train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
        consensus_crop_logits = consensus_crop_logits.to(self.device)
        consensus_disease_logits = consensus_disease_logits.to(self.device)

        kd_loss_sum, kd_batches = 0.0, 0
        sup_loss_sum, proto_loss_sum, sup_batches = 0.0, 0.0, 0
        for _ in range(epochs):
            # (a) knowledge-distillation using the shared public probe dataset
            for batch_idx, (images, _, _) in enumerate(probe_loader):
                images = images.to(self.device)
                start = batch_idx * probe_loader.batch_size
                end = start + images.shape[0]

                # generate student logits
                crop_logits, disease_logits = self.model(images)
                kd_loss = _soft_kd_loss(
                    crop_logits, consensus_crop_logits[start:end], temperature
                ) + _soft_kd_loss(disease_logits, consensus_disease_logits[start:end], temperature)

                optimizer.zero_grad()
                (kd_weight * kd_loss).backward()
                optimizer.step()
                scheduler.step()

                kd_loss_sum += kd_loss.item()
                kd_batches += 1

            # (b) supervised learning + prototype alignment on local labeled data only
            for images, crop_labels, disease_labels in self.train_loader:
                images = images.to(self.device)
                crop_labels = crop_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)

                crop_logits, disease_logits, feats = self.model(images, return_features=True)
                sup_loss = self.crop_loss_weight * _classification_loss(
                    crop_logits, crop_labels, self.loss_type, self.crop_class_weights, self.focal_gamma
                ) + self.disease_loss_weight * _classification_loss(
                    disease_logits, disease_labels, self.loss_type, self.disease_class_weights, self.focal_gamma
                )
                proto_loss = _prototype_alignment_loss(
                    feats, crop_labels, disease_labels, consensus_prototypes, self.device
                )

                loss = sup_loss + proto_weight * proto_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                sup_loss_sum += sup_loss.item()
                proto_loss_sum += proto_loss.item()
                sup_batches += 1

        return {
            "kd_loss": kd_loss_sum / max(1, kd_batches),
            "sup_loss": sup_loss_sum / max(1, sup_batches),
            "proto_loss": proto_loss_sum / max(1, sup_batches),
        }

    # evaluation
    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict:
        """To measure whether this node actually gained knowledge of
        classes outside its own shard via the mesh
        """
        self.model.eval()
        loader = loader if loader is not None else self.test_loader
        crop_true, crop_pred, disease_true, disease_pred = [], [], [], []
        for images, crop_labels, disease_labels in loader:
            images = images.to(self.device)
            crop_logits, disease_logits = self.model(images)
            crop_true.extend(crop_labels.tolist())
            crop_pred.extend(crop_logits.argmax(dim=1).cpu().tolist())
            disease_true.extend(disease_labels.tolist())
            disease_pred.extend(disease_logits.argmax(dim=1).cpu().tolist())

        total = max(1, len(crop_true))
        correct_crop = sum(t == p for t, p in zip(crop_true, crop_pred))
        correct_disease = sum(t == p for t, p in zip(disease_true, disease_pred))
        # Both heads right on the SAME sample — the two accuracies above
        # can each look fine while the model rarely gets the full class
        # right (e.g. correct disease, wrong crop), which is exactly the
        # failure mode that matters for cross-node generalization.
        correct_pair = sum(
            cp == ct and dp == dt for cp, ct, dp, dt in zip(crop_pred, crop_true, disease_pred, disease_true)
        )

        crop_metrics = head_metrics(crop_true, crop_pred, self.crop_classes)
        disease_metrics = head_metrics(disease_true, disease_pred, self.disease_classes)

        result = {
            "crop_accuracy": correct_crop / total,
            "disease_accuracy": correct_disease / total,
            "pair_accuracy": correct_pair / total,
            "crop_macro_precision": crop_metrics["macro_precision"],
            "crop_macro_recall": crop_metrics["macro_recall"],
            "crop_macro_f1": crop_metrics["macro_f1"],
            "disease_macro_precision": disease_metrics["macro_precision"],
            "disease_macro_recall": disease_metrics["macro_recall"],
            "disease_macro_f1": disease_metrics["macro_f1"],
            "detail": {
                "crop": {
                    "per_class": crop_metrics["per_class"],
                    "confusion_matrix": crop_metrics["confusion_matrix"],
                },
                "disease": {
                    "per_class": disease_metrics["per_class"],
                    "confusion_matrix": disease_metrics["confusion_matrix"],
                },
            },
        }

        if self.pair_to_class_idx is not None and self.pair_class_names is not None:
            # The two heads predict independently, so a predicted
            # (crop, disease) combo can be one that never occurs in the
            # real 27-class label space (e.g. Peach + Tomato_mosaic_virus)
            # — bucket those as "invalid_combo" rather than crashing or
            # silently dropping them, since predicting a nonsense combo is
            # itself a real failure worth counting.
            invalid_idx = len(self.pair_class_names)
            pair_class_names = self.pair_class_names + ["invalid_combo"]
            pair_true = [self.pair_to_class_idx[(c, d)] for c, d in zip(crop_true, disease_true)]
            pair_pred = [self.pair_to_class_idx.get((c, d), invalid_idx) for c, d in zip(crop_pred, disease_pred)]
            pair_metrics = head_metrics(pair_true, pair_pred, pair_class_names)
            result["pair_macro_precision"] = pair_metrics["macro_precision"]
            result["pair_macro_recall"] = pair_metrics["macro_recall"]
            result["pair_macro_f1"] = pair_metrics["macro_f1"]
            result["detail"]["pair"] = {
                "per_class": pair_metrics["per_class"],
                "confusion_matrix": pair_metrics["confusion_matrix"],
            }

        return result

def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    weight: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    if loss_type == "focal":
        ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** gamma * ce).mean()
    return F.cross_entropy(logits, labels, weight=weight)


# compute kl divergence loss
def _soft_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1) #(−∞,0)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1) #(0, 1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

# compute prototype alignment loss
def _prototype_alignment_loss(
    feats: torch.Tensor,
    crop_labels: torch.Tensor,
    disease_labels: torch.Tensor,
    consensus_prototypes: Prototypes,
    device: str,
) -> torch.Tensor:
    losses = []
    for f, c, d in zip(feats, crop_labels, disease_labels):
        c, d = int(c), int(d)
        if ("crop", c) in consensus_prototypes:
            target = consensus_prototypes[("crop", c)].to(device)
            losses.append(F.mse_loss(f, target))
        if ("disease", d) in consensus_prototypes:
            target = consensus_prototypes[("disease", d)].to(device)
            losses.append(F.mse_loss(f, target))
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
