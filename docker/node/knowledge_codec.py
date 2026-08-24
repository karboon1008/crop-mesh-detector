"""Serializes a KnowledgePayload (prototypes + probe-set logits — never
raw images, gradients, or weights) to bytes for the GET /knowledge/{round_idx}
HTTP response, tagged with the round it was computed in.
"""

from __future__ import annotations

import io

import torch

from src.federated.node import KnowledgePayload


def encode_knowledge(round_idx: int, payload: KnowledgePayload) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "round_idx": round_idx,
            "prototypes": payload.prototypes,
            "crop_logits": payload.crop_logits,
            "disease_logits": payload.disease_logits,
            "known_crop_classes": payload.known_crop_classes,
            "known_disease_classes": payload.known_disease_classes,
        },
        buffer,
    )
    return buffer.getvalue()


def decode_knowledge(data: bytes) -> tuple[int, KnowledgePayload]:
    # weights_only=False: this deserializes our own KnowledgePayload dict
    # (tuple-keyed prototypes dict + tensors), fetched only from our own
    # node containers over the internal Docker network -- not untrusted input.
    obj = torch.load(io.BytesIO(data), weights_only=False)
    payload = KnowledgePayload(
        prototypes=obj["prototypes"],
        crop_logits=obj["crop_logits"],
        disease_logits=obj["disease_logits"],
        known_crop_classes=obj["known_crop_classes"],
        known_disease_classes=obj["known_disease_classes"],
    )
    return obj["round_idx"], payload
