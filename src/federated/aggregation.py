"""Byzantine-robust aggregation over peer-shared knowledge artefacts
(class prototypes and public-probe-set logits) — never over weights or
gradients. Every node runs this locally on what its peers sent it; there
is no central aggregator (that is what makes this a mesh, not a
federated server).
"""

from __future__ import annotations

import torch


def _trimmed_mean(stacked: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    """stacked: (N, D). Trims `trim_fraction` of peers from each end,
    per-dimension, then averages what remains. Robust to a minority of
    stale or adversarial peers without needing to identify them.
    """
    n = stacked.shape[0]
    k = int(n * trim_fraction)
    if n - 2 * k < 1:
        k = max(0, (n - 1) // 2)
    sorted_vals, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[k : n - k] if k > 0 else sorted_vals
    return trimmed.mean(dim=0)


def _krum(stacked: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    """stacked: (N, D). Scores each peer by the sum of squared distances
    to its `num_neighbors` closest peers, then returns the peer with the
    lowest score — the single most "agreed-with" contribution, immune to
    outliers pulling a mean off course (Blanchard et al.-style Krum).
    """
    n = stacked.shape[0]
    if n == 1:
        return stacked[0]
    num_neighbors = max(1, min(num_neighbors, n - 1))
    dists = torch.cdist(stacked, stacked, p=2) ** 2  # (N, N)
    scores = torch.zeros(n)
    for i in range(n):
        row = dists[i].clone()
        row[i] = float("inf")
        closest, _ = torch.topk(row, num_neighbors, largest=False)
        scores[i] = closest.sum()
    best = torch.argmin(scores).item()
    return stacked[best]


def aggregate_vectors(
    vectors: list[torch.Tensor],
    method: str = "trimmed_mean",
    trim_fraction: float = 0.2,
    krum_neighbors: int = 2,
) -> torch.Tensor:
    """Aggregate a list of same-shape 1-D (or flattenable) tensors from peers."""
    shapes = {v.shape for v in vectors}
    if len(shapes) != 1:
        raise ValueError(f"All peer vectors must share a shape, got {shapes}")
    orig_shape = vectors[0].shape
    stacked = torch.stack([v.reshape(-1).float() for v in vectors], dim=0)

    if method == "trimmed_mean":
        result = _trimmed_mean(stacked, trim_fraction)
    elif method == "krum":
        result = _krum(stacked, krum_neighbors)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return result.reshape(orig_shape)


def aggregate_prototypes(
    peer_prototypes: list[dict[tuple[str, int], torch.Tensor]],
    method: str = "trimmed_mean",
    trim_fraction: float = 0.2,
    krum_neighbors: int = 2,
) -> dict[tuple[str, int], torch.Tensor]:
    """peer_prototypes: one dict per peer, keyed by (task, class_id) -> mean
    embedding for that class. Missing keys (a peer never saw that class)
    are simply skipped for that key — the mesh only reconciles knowledge
    peers actually have.
    """
    all_keys: set[tuple[str, int]] = set()
    for proto in peer_prototypes:
        all_keys.update(proto.keys())

    consensus: dict[tuple[str, int], torch.Tensor] = {}
    for key in all_keys:
        contributions = [p[key] for p in peer_prototypes if key in p]
        if not contributions:
            continue
        consensus[key] = aggregate_vectors(
            contributions, method=method, trim_fraction=trim_fraction, krum_neighbors=krum_neighbors
        )
    return consensus


def aggregate_logits(
    peer_logits: list[torch.Tensor],
    method: str = "trimmed_mean",
    trim_fraction: float = 0.2,
    krum_neighbors: int = 2,
) -> torch.Tensor:
    """peer_logits: one (num_probe, num_classes) tensor per peer, computed
    on the identical shared public probe set. Returns the consensus
    logits of the same shape.
    """
    return aggregate_vectors(
        peer_logits, method=method, trim_fraction=trim_fraction, krum_neighbors=krum_neighbors
    )
