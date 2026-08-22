"""Model factory: three lightweight backbones (mobilenet_v3_small,
efficientnet_lite0, mobilevit_xxs) sharing one architecture pattern —
a single feature extractor with two linear heads, one for crop type and
one for disease. A shared backbone means one detector per node instead
of two, halving the on-device footprint for the same accuracy target.

All backbones come from `timm` so parameter counts, pretrained weights,
and the pooled-feature interface (`num_classes=0`) are consistent across
architectures, which keeps the energy/accuracy comparison in
src/train.py apples-to-apples.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# config.yaml key -> timm model name
ARCH_TO_TIMM: dict[str, str] = {
    "mobilenet_v3_small": "mobilenetv3_small_100",
    "efficientnet_lite0": "tf_efficientnet_lite0",
    "mobilevit_xxs": "mobilevit_xxs",
}

SUPPORTED_ARCHITECTURES = tuple(ARCH_TO_TIMM.keys())

# Backbone module-name prefixes to freeze, for architectures where
# freezing is enabled. Both lightweight backbones are back-loaded (most
# params sit in their last two stages + head), so freezing only the stem
# + first two feature stages would lock under 1% of parameters — too
# small to plausibly affect anything. These cutoffs instead go deep
# enough to lock a real share of capacity:
#   mobilenet_v3_small: through blocks.3 (~51% of backbone params) —
#     blocks.4/5 + conv_head, left trainable, hold the other ~49%.
#   mobilevit_xxs: through stages.2 (~18% of backbone params) — going
#     further would freeze stages.3/4, the transformer blocks that hold
#     96% of this backbone's capacity, which is too much to lock.
# Past the first stage or so these are no longer purely generic
# ImageNet features — this trades some task-adaptation capacity for
# round-to-round stability, not a free stabilizer. mobilenet_v3_small
# and mobilevit_xxs show large round-to-round accuracy swings during the
# mesh phase's short (2-3 epoch) local fine-tunes; efficientnet_lite0
# doesn't (see training.local_epochs_per_round_overrides comment in
# config.yaml) and is deliberately left out here — freezing it would
# only remove capacity from a backbone with no instability to fix.
FREEZE_LOW_LAYER_PREFIXES: dict[str, tuple[str, ...]] = {
    "mobilenet_v3_small": ("conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2", "blocks.3"),
    "mobilevit_xxs": ("stem", "stages.0", "stages.1", "stages.2"),
}


def freeze_low_layers(backbone: nn.Module, arch_key: str) -> int:
    """Sets requires_grad=False on the FREEZE_LOW_LAYER_PREFIXES[arch_key]
    modules of `backbone`, in place. No-op for any arch_key not in
    FREEZE_LOW_LAYER_PREFIXES. Returns the number of parameters frozen.
    """
    prefixes = FREEZE_LOW_LAYER_PREFIXES.get(arch_key, ())
    frozen = 0
    for name, param in backbone.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


class MultiTaskNet(nn.Module):
    """Shared backbone + two heads: crop type and disease status."""

    def __init__(self, backbone: nn.Module, embed_dim: int, num_crop_classes: int, num_disease_classes: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.crop_head = nn.Linear(embed_dim, num_crop_classes)
        self.disease_head = nn.Linear(embed_dim, num_disease_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)  # (B, embed_dim), already globally pooled
        crop_logits = self.crop_head(feats)
        disease_logits = self.disease_head(feats)
        if return_features:
            return crop_logits, disease_logits, feats
        return crop_logits, disease_logits


def build_model(
    arch_key: str,
    num_crop_classes: int,
    num_disease_classes: int,
    pretrained: bool = True,
    freeze_low_layers_: bool = False,
) -> MultiTaskNet:
    if arch_key not in ARCH_TO_TIMM:
        raise ValueError(
            f"Unknown architecture '{arch_key}'. Supported: {SUPPORTED_ARCHITECTURES}"
        )
    timm_name = ARCH_TO_TIMM[arch_key]
    backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    embed_dim = _probe_embedding_dim(backbone)
    if freeze_low_layers_:
        freeze_low_layers(backbone, arch_key)
    return MultiTaskNet(backbone, embed_dim, num_crop_classes, num_disease_classes)


def _probe_embedding_dim(backbone: nn.Module) -> int:
    """`backbone.num_features` is unreliable across timm architectures —
    some (e.g. mobilenetv3_small_100) report the pre-head-conv channel
    count rather than the actual pooled output width. A dummy forward
    pass is the only architecture-agnostic way to get the real dimension.
    """
    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        dim = backbone(dummy).shape[1]
    backbone.train(was_training)
    return dim


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate on-disk footprint assuming FP32 weights (4 bytes/param)."""
    return count_parameters(model) * 4 / (1024 ** 2)


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of the linear heads — a cheap, CPU-only
    demonstration of the compute/memory savings the report's INT8 export
    path (van Baalen et al.) is aiming for. Full static quantization of
    the convolutional backbone needs calibration data and is out of scope
    for this simulation-only project.
    """
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
