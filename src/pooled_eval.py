"""Pooled test evaluation for the continual stream.

Every node is also scored on a pooled test set: a stratified sample (capped at
`max_images`) of the private test images of ALL nodes over batches 0..b. No image in it
was ever trained on by any node, and it contains classes a node may never have seen, so
it measures what a node knows beyond its own shard. It is scored at the same two points
as the node's own test set: right after local training (pre) and right after the
exchange (post).
"""

from __future__ import annotations

from contextlib import contextmanager

from src.data.splits import build_probe_loader, stratified_sample
from src.evaluate import scalar_metrics


def build_pooled_test_loader(cfg, dataset, stream, up_to_batch: int, max_images: int, seed: int):
    pool = sorted({
        idx
        for b in range(up_to_batch + 1)
        for i in range(stream.num_nodes)
        for idx in stream.node_split(b, f"node_{i}").test_idx
    })
    if len(pool) > max_images:
        pool = sorted(stratified_sample(dataset, pool, max_images, seed))
    return build_probe_loader(cfg, dataset, pool)  # unshuffled, evaluation transforms


@contextmanager
def pooled_eval_taps(nodes, loader):
    """While active, every no-argument node.evaluate() (the pre- and post-exchange own-test
    evaluations, the only ones a batch makes) is followed by an evaluation on `loader`.
    Yields node_id -> list of scalar-metric dicts in call order: [pre] or [pre, post].
    """
    logs = {node.node_id: [] for node in nodes}
    for node in nodes:
        original = node.evaluate

        def tapped(eval_loader=None, _original=original, _log=logs[node.node_id]):
            result = _original(eval_loader)
            if eval_loader is None:
                _log.append(scalar_metrics(_original(loader)))
            return result

        node.evaluate = tapped
    try:
        yield logs
    finally:
        for node in nodes:
            del node.evaluate  # drop the instance attribute; the class method is back
