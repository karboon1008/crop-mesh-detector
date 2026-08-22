"""Byzantine-robust aggregation over peer-shared knowledge artefacts
(class prototypes and public-probe-set logits) — never over weights or
gradients. Every node runs this locally on what its peers sent it; there
is no central aggregator (that is what makes this a mesh, not a
federated server).
"""

from __future__ import annotations

import torch


def _trimmed_mean(stacked: torch.Tensor, trim_fraction: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    """stacked: (N, D). Trims `trim_fraction` of peers from each end,
    per-dimension, then averages what remains. Robust to a minority of
    stale or adversarial peers without needing to identify them.

    `weights` (shape (N,)), when given, replaces the plain mean of the
    surviving peers with a weighted average — e.g. each peer's local
    sample count for the class being aggregated, so a peer with 5
    examples doesn't count as much as one with 200. Trimming itself still
    goes by value, not weight, so a confidently-wrong high-volume peer is
    still trimmed like anyone else; weighting only affects the average
    among peers that already survived that cut.
    """
    n = stacked.shape[0]
    k = int(n * trim_fraction)
    if n - 2 * k < 1:
        k = max(0, (n - 1) // 2)
    sorted_vals, sort_idx = torch.sort(stacked, dim=0)
    if k > 0:
        trimmed, trimmed_idx = sorted_vals[k : n - k], sort_idx[k : n - k]
    else:
        trimmed, trimmed_idx = sorted_vals, sort_idx
    if weights is None:
        return trimmed.mean(dim=0)
    # weights is per-peer, but sort_idx (and so trimmed_idx) tracks which
    # peer landed at each sorted position independently per column, so the
    # weight has to be gathered the same way rather than just sliced.
    weights_per_cell = weights.reshape(-1, *([1] * (stacked.dim() - 1))).expand_as(stacked)
    trimmed_weights = torch.gather(weights_per_cell, 0, trimmed_idx)
    return (trimmed * trimmed_weights).sum(dim=0) / trimmed_weights.sum(dim=0).clamp_min(1e-8)


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


def aggregate_masked_logits(
    peer_logits: list[torch.Tensor],
    peer_class_counts: list[dict[int, int]],
    method: str = "trimmed_mean",
    trim_fraction: float = 0.2,
    krum_neighbors: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """peer_logits: one (num_probe, num_classes) tensor per peer (crop or
    disease head), computed on the identical shared public probe set.
    peer_class_counts: the matching per-peer {class_id: local_sample_count}
    map for classes that peer actually has local training examples for
    (see Node.compute_prototypes) — membership (`c in counts`) marks a peer
    as informed the same way a plain set would; the count value additionally
    weights that peer's say in the trimmed-mean average (see _trimmed_mean).

    Unlike a plain per-cell trimmed mean/Krum over all peers, this
    aggregates each class column only from the peers who actually have
    that class — many classes belong to only a subset of nodes (a single
    node under non_iid_strategy="manual", a Dirichlet-skewed subset under
    "dirichlet"), so for those columns an uninformed peer would otherwise
    be confidently voting on a class it has never seen, and trimmed-mean/
    Krum can't tell that apart from a genuinely informed peer. Weighting by
    count on top of that means a peer with only a handful of examples for
    a class doesn't get the same say as one with hundreds. Krum picks a
    single peer's contribution rather than averaging, so counts don't
    apply there — every informed peer still competes on agreement alone.

    Returns (consensus_logits, known_mask): known_mask (bool, shape
    (num_classes,)) marks which columns had at least one informed peer —
    columns with none are left at 0 in consensus_logits and False in
    known_mask, and callers should exclude them from any loss that reads
    consensus_logits (see Node._soft_kd_loss's class_mask).
    """
    num_probe, num_classes = peer_logits[0].shape
    consensus = torch.zeros(num_probe, num_classes)
    known_mask = torch.zeros(num_classes, dtype=torch.bool)

    for c in range(num_classes):
        informed = [
            (logits[:, c], counts[c])
            for logits, counts in zip(peer_logits, peer_class_counts)
            if c in counts
        ]
        if not informed:
            continue
        known_mask[c] = True
        stacked = torch.stack([logits for logits, _ in informed], dim=0)  # (n_informed, num_probe)
        if method == "trimmed_mean":
            weights = torch.tensor([float(count) for _, count in informed])
            consensus[:, c] = _trimmed_mean(stacked, trim_fraction, weights=weights)
        elif method == "krum":
            consensus[:, c] = _krum(stacked, krum_neighbors)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return consensus, known_mask
