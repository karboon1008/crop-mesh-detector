from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "node"))

import torch

from knowledge_codec import decode_knowledge, encode_knowledge  # noqa: E402
from src.federated.node import KnowledgePayload  # noqa: E402


def test_encode_decode_roundtrip_preserves_tensors_and_round_idx():
    payload = KnowledgePayload(
        prototypes={("crop", 0): torch.randn(8), ("disease", 1): torch.randn(8)},
        crop_logits=torch.randn(5, 3),
        disease_logits=torch.randn(5, 2),
        known_crop_classes={0: 4},
        known_disease_classes={1: 4},
    )
    data = encode_knowledge(7, payload)
    assert isinstance(data, bytes)

    round_idx, decoded = decode_knowledge(data)
    assert round_idx == 7
    assert torch.equal(decoded.crop_logits, payload.crop_logits)
    assert torch.equal(decoded.disease_logits, payload.disease_logits)
    for key in payload.prototypes:
        assert torch.equal(decoded.prototypes[key], payload.prototypes[key])
    assert decoded.known_crop_classes == payload.known_crop_classes
    assert decoded.known_disease_classes == payload.known_disease_classes


def test_encoded_payload_is_nonempty_bytes():
    payload = KnowledgePayload(
        prototypes={}, crop_logits=torch.zeros(2, 2), disease_logits=torch.zeros(2, 2),
        known_crop_classes={}, known_disease_classes={},
    )
    data = encode_knowledge(0, payload)
    assert len(data) > 0
