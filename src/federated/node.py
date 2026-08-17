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
        disease_class_weights: torch.Tensor | None = None,
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
        self.disease_class_weights = (
            disease_class_weights.to(device) if disease_class_weights is not None else None
        )

    # local supervised training (data never leaves this method)
    def local_train(self, epochs: int, lr: float) -> float:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        total_loss, total_batches = 0.0, 0
        for _ in range(epochs):
            for images, crop_labels, disease_labels in self.train_loader:
                images = images.to(self.device)
                crop_labels = crop_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)

                optimizer.zero_grad()
                crop_logits, disease_logits = self.model(images)
                loss = self.crop_loss_weight * F.cross_entropy(
                    crop_logits, crop_labels
                ) + self.disease_loss_weight * F.cross_entropy(
                    disease_logits, disease_labels, weight=self.disease_class_weights
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
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

                kd_loss_sum += kd_loss.item()
                kd_batches += 1

            # (b) supervised learning + prototype alignment on local labeled data only
            for images, crop_labels, disease_labels in self.train_loader:
                images = images.to(self.device)
                crop_labels = crop_labels.to(self.device)
                disease_labels = disease_labels.to(self.device)

                crop_logits, disease_logits, feats = self.model(images, return_features=True)
                sup_loss = self.crop_loss_weight * F.cross_entropy(
                    crop_logits, crop_labels
                ) + self.disease_loss_weight * F.cross_entropy(
                    disease_logits, disease_labels, weight=self.disease_class_weights
                )
                proto_loss = _prototype_alignment_loss(
                    feats, crop_labels, disease_labels, consensus_prototypes, self.device
                )

                loss = sup_loss + proto_weight * proto_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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

        crop_metrics = head_metrics(crop_true, crop_pred, self.crop_classes)
        disease_metrics = head_metrics(disease_true, disease_pred, self.disease_classes)

        return {
            "crop_accuracy": correct_crop / total,
            "disease_accuracy": correct_disease / total,
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
