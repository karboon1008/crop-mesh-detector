# tests/test_node_main.py
"""Covers the thin HTTP-glue layer in docker/node/main.py that NodeRunner's
own tests (tests/test_node_runner.py) deliberately don't reach: request-body
validation and the concurrent peer-fetch helper's handling of a peer_bases
entry that simply isn't there.

main.py builds a real NodeRunner at import time, so the fixture below stands
up a minimal on-disk node/probe split plus a classes.json and config first,
then loads the module by explicit file path (not `import main`) so it can't
collide in sys.modules with docker/coordinator/main.py.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

NODE_DIR = Path(__file__).resolve().parent.parent / "docker" / "node"
sys.path.insert(0, str(NODE_DIR))

from src.data.plantvillage import (  # noqa: E402
    PlantVillageDataset,
    build_global_label_map,
    save_global_label_map,
)

CLASSES = ["Tomato___Bacterial_spot", "Tomato___healthy", "Potato___healthy"]

MINIMAL_CONFIG = """
data:
  image_size: 32
  test_fraction: 0.3
  seed: 42
models:
  architectures: ["mobilenet_v3_small"]
  pretrained: false
training:
  batch_size: 4
  local_epochs_per_round: 1
  distill_epochs_per_round: 1
energy:
  track_with_codecarbon: false
"""


def _write_images(root: Path, n_per_class: int = 4) -> None:
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


@pytest.fixture(scope="module")
def node_main(tmp_path_factory):
    root = tmp_path_factory.mktemp("node_main")
    _write_images(root / "node_0")
    _write_images(root / "probe", n_per_class=2)
    save_global_label_map(
        build_global_label_map(PlantVillageDataset(root / "node_0", image_size=32)),
        root / "classes.json",
    )
    config_path = root / "config.yaml"
    config_path.write_text(MINIMAL_CONFIG)

    env = {
        "NODE_ID": "node_0",
        "DATA_ROOT": str(root / "node_0"),
        "PROBE_ROOT": str(root / "probe"),
        "CLASSES_JSON": str(root / "classes.json"),
        "ENERGY_DB": str(root / "node_0.db"),
        "CONFIG_PATH": str(config_path),
    }
    with pytest.MonkeyPatch.context() as mp:
        for key, value in env.items():
            mp.setenv(key, value)
        spec = importlib.util.spec_from_file_location("docker_node_main", NODE_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_fetch_skips_a_peer_with_no_peer_bases_entry_instead_of_raising(node_main):
    # peer_bases is missing node_9 entirely. Indexing it would raise a KeyError
    # out of asyncio.gather and fail this node's whole /round/gather; the peer
    # must instead be reported as a plain fetch failure (None).
    result = asyncio.run(node_main._fetch_all_knowledge_async(["node_9"], {}, 0))
    assert result == {"node_9": None}


def test_fetch_skips_only_the_missing_peer_not_the_whole_gather(node_main):
    # node_1 has no base URL; node_2's is unreachable. Both come back None,
    # and crucially the call returns at all rather than blowing up.
    async def scenario():
        return await node_main._fetch_all_knowledge_async(
            ["node_1", "node_2"], {"node_2": "http://127.0.0.1:9"}, 0
        )

    result = asyncio.run(scenario())
    assert result == {"node_1": None, "node_2": None}


def test_fetch_all_knowledge_returns_empty_for_no_peers(node_main):
    assert node_main.fetch_all_knowledge([], {"node_1": "http://node_1:8000"}, 0) == {}


def test_round_start_rejects_a_body_missing_round_idx_with_400(node_main):
    with pytest.raises(HTTPException) as exc_info:
        node_main.round_start({})
    assert exc_info.value.status_code == 400
    assert "round_idx" in exc_info.value.detail


@pytest.mark.parametrize("missing", ["round_idx", "active_nodes", "peer_bases"])
def test_round_gather_rejects_a_body_missing_any_required_key_with_400(node_main, missing):
    body = {"round_idx": 0, "active_nodes": ["node_0"], "peer_bases": {}}
    del body[missing]
    with pytest.raises(HTTPException) as exc_info:
        node_main.round_gather(body)
    assert exc_info.value.status_code == 400
    assert missing in exc_info.value.detail
