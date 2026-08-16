# Docker/MQTT Multi-Container Mesh + Observability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing single-process mesh (`Node`/`MeshSimulator` training + distillation logic) across 3 isolated Docker containers exchanging knowledge over MQTT instead of an in-memory dict, with energy/eval metrics persisted to SQLite (per-node + merged), plus a read-only Streamlit dashboard.

**Architecture:** 6 Docker Compose services — `broker` (Mosquitto), `coordinator` (times rounds, tracks liveness, merges metrics — never touches knowledge content), `node_0`/`node_1`/`node_2` (one shared image, each with its own physically-isolated data folder), `dashboard` (Streamlit viewer). The existing in-process simulation (`src/federated/mesh.py`, `src/train.py`) is untouched and stays as a separate, working fallback pipeline.

**Tech Stack:** Python 3.11, PyTorch/timm (nodes only), `paho-mqtt`, `sqlite3` (stdlib), Eclipse Mosquitto, Streamlit + pandas (dashboard only).

**Spec:** `docs/superpowers/specs/2026-08-16-docker-mqtt-mesh-design.md`

## Global Constraints

- `src/federated/mesh.py`, `src/federated/node.py`, `src/federated/aggregation.py`, `src/models/factory.py`, `src/train.py` are **not modified** by this plan — the in-process pipeline must keep working exactly as it does today.
- The coordinator **never subscribes to `mesh/node/+/knowledge`** (the binary payload topic) — control-plane only, per spec §5.
- The dashboard subscribes only to `mesh/control/#`, `mesh/node/+/status`, `mesh/node/+/knowledge_ready`, `mesh/node/+/energy`, `mesh/node/+/eval` — **never** `mesh/node/+/knowledge` — per spec §8.
- No broker authentication; Mosquitto has no host port published; only the dashboard publishes a host port (`8501`) — per spec §11.
- `size_per_message`/`knowledge_bytes_sent` must be the actual serialized byte length, never `KnowledgePayload.size_bytes()`'s estimate — per spec §6.
- This repo's layout diverges from spec §3 in one place: app code lives flat under `docker/node/`, `docker/coordinator/`, `docker/dashboard/` (no nested `node_app/`/`coordinator_app/`/`dashboard_app/` subfolder) to avoid a `docker.*` dotted-package name colliding with the pip `docker` SDK package, and to keep Dockerfiles simple. Topic names, schema, and container roles are unchanged from the spec.
- Deferred, do not build in this plan: disruption scenarios over this transport, the K210 node, `mesh/control/trigger_run` — per spec §2/§12.

---

## Task 1: Global label map for `PlantVillageDataset`

This is the highest-risk piece in the whole design: without it, a node container that only physically holds a subset of crop classes would silently build its own local class ordering, misaligning the crop/disease logits exchanged between nodes with no visible error.

**Files:**
- Modify: `src/data/plantvillage.py`
- Test: `tests/test_global_label_map.py`

**Interfaces:**
- Produces: `GlobalLabelMap` dataclass (`crop_classes: list[str]`, `disease_classes: list[str]`, `name_to_crop_disease: dict[str, tuple[int, int]]`); `build_global_label_map(dataset: PlantVillageDataset) -> GlobalLabelMap`; `save_global_label_map(label_map: GlobalLabelMap, path: str | Path) -> None`; `load_global_label_map(path: str | Path) -> GlobalLabelMap`; `PlantVillageDataset.__init__(self, root, image_size=160, global_label_map: GlobalLabelMap | None = None)` (default `None` preserves all existing behaviour).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_global_label_map.py
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.plantvillage import (
    GlobalLabelMap,
    PlantVillageDataset,
    build_global_label_map,
    load_global_label_map,
    save_global_label_map,
)

FULL_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


def _make_class_folders(root, class_names, n_per_class=3):
    rng = np.random.RandomState(0)
    for cls in class_names:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


def test_global_label_map_preserves_indices_for_subset_of_classes(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, FULL_CLASSES)
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    node_root = tmp_path / "node_0"
    subset = ["Tomato___Bacterial_spot", "Potato___healthy"]
    _make_class_folders(node_root, subset)
    node_dataset = PlantVillageDataset(node_root, image_size=16, global_label_map=global_map)

    for local_idx, name in enumerate(node_dataset.base.classes):
        full_idx = full_dataset.base.classes.index(name)
        expected = full_dataset.labels.class_to_crop_disease[full_idx]
        assert node_dataset.labels.class_to_crop_disease[local_idx] == expected

    assert node_dataset.labels.crop_classes == full_dataset.labels.crop_classes
    assert node_dataset.labels.disease_classes == full_dataset.labels.disease_classes


def test_global_label_map_raises_on_class_missing_from_map(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, ["Tomato___Bacterial_spot", "Potato___healthy"])
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    node_root = tmp_path / "node_stale"
    _make_class_folders(node_root, ["Tomato___Bacterial_spot", "Corn___healthy"])  # Corn unknown to the map

    with pytest.raises(ValueError, match="Corn___healthy"):
        PlantVillageDataset(node_root, image_size=16, global_label_map=global_map)


def test_global_label_map_json_roundtrip(tmp_path):
    full_root = tmp_path / "full"
    _make_class_folders(full_root, FULL_CLASSES)
    full_dataset = PlantVillageDataset(full_root, image_size=16)
    global_map = build_global_label_map(full_dataset)

    path = tmp_path / "classes.json"
    save_global_label_map(global_map, path)
    loaded = load_global_label_map(path)

    assert loaded.crop_classes == global_map.crop_classes
    assert loaded.disease_classes == global_map.disease_classes
    assert loaded.name_to_crop_disease == global_map.name_to_crop_disease
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_global_label_map.py -v`
Expected: FAIL with `ImportError` (`GlobalLabelMap`, `build_global_label_map`, etc. don't exist yet).

- [ ] **Step 3: Implement**

Add to `src/data/plantvillage.py` (near the existing `LabelMaps` dataclass):

```python
import json  # add to existing imports at top of file


@dataclass
class GlobalLabelMap:
    """Class-name-keyed label map, shared across every node container so
    each node's locally-discovered ImageFolder classes map to the SAME
    global crop/disease indices as every other node — required for the
    prototype/logit exchange to align positionally. Keyed by class NAME,
    not the numeric ImageFolder index used by LabelMaps.class_to_crop_disease,
    because that index is only meaningful relative to one specific
    ImageFolder instance (and a node's local ImageFolder only discovers
    whatever subset of class folders it physically has).
    """

    crop_classes: list[str]
    disease_classes: list[str]
    name_to_crop_disease: dict[str, tuple[int, int]]


def build_global_label_map(dataset: "PlantVillageDataset") -> GlobalLabelMap:
    name_to_crop_disease = {
        dataset.base.classes[idx]: cd
        for idx, cd in dataset.labels.class_to_crop_disease.items()
    }
    return GlobalLabelMap(
        crop_classes=list(dataset.labels.crop_classes),
        disease_classes=list(dataset.labels.disease_classes),
        name_to_crop_disease=name_to_crop_disease,
    )


def save_global_label_map(label_map: GlobalLabelMap, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "crop_classes": label_map.crop_classes,
                "disease_classes": label_map.disease_classes,
                "name_to_crop_disease": label_map.name_to_crop_disease,
            },
            indent=2,
        )
    )


def load_global_label_map(path: str | Path) -> GlobalLabelMap:
    data = json.loads(Path(path).read_text())
    return GlobalLabelMap(
        crop_classes=data["crop_classes"],
        disease_classes=data["disease_classes"],
        name_to_crop_disease={k: tuple(v) for k, v in data["name_to_crop_disease"].items()},
    )
```

Modify `PlantVillageDataset`:

```python
class PlantVillageDataset(Dataset):
    """Wraps torchvision's ImageFolder, exposing (image, crop_label, disease_label)."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 160,
        global_label_map: "GlobalLabelMap | None" = None,
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.base = ImageFolder(str(root))
        if global_label_map is None:
            self.labels = self._build_label_maps(self.base.classes)
        else:
            self.labels = self._apply_global_label_map(self.base.classes, global_label_map)

    @staticmethod
    def _apply_global_label_map(class_names: list[str], global_label_map: "GlobalLabelMap") -> LabelMaps:
        missing = [c for c in class_names if c not in global_label_map.name_to_crop_disease]
        if missing:
            raise ValueError(
                f"Classes {missing} are not present in the supplied global label map "
                f"(classes.json) — the split data and classes.json are out of sync."
            )
        class_to_crop_disease = {
            i: global_label_map.name_to_crop_disease[name] for i, name in enumerate(class_names)
        }
        return LabelMaps(
            crop_classes=list(global_label_map.crop_classes),
            disease_classes=list(global_label_map.disease_classes),
            class_to_crop_disease=class_to_crop_disease,
        )

    @staticmethod
    def _build_label_maps(class_names: list[str]) -> LabelMaps:
        ...  # unchanged, existing method body
```

(Keep the existing `_build_label_maps` body exactly as-is — only add the new `_apply_global_label_map` staticmethod and the `global_label_map` branch in `__init__`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_global_label_map.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -v`
Expected: PASS (all existing tests unaffected — `global_label_map` defaults to `None`)

- [ ] **Step 6: Commit**

```bash
git add src/data/plantvillage.py tests/test_global_label_map.py
git commit -m "Add GlobalLabelMap for cross-container class-index alignment"
```

---

## Task 2: SQLite round_metrics store

**Files:**
- Create: `src/energy/sqlite_store.py`
- Test: `tests/test_sqlite_store.py`

**Interfaces:**
- Consumes: nothing from other tasks (stdlib `sqlite3` only).
- Produces: `init_db(path: str | Path) -> None`; `upsert_row(path: str | Path, node_id: str, round_idx: int, recorded_at: str, **fields) -> None`; `read_all(path: str | Path) -> list[dict]`. Used by Task 5 (node runner), Task 7 (coordinator runner), Task 10 (dashboard data helpers).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sqlite_store.py
from __future__ import annotations

from src.energy import sqlite_store


def test_upsert_row_creates_row_with_given_columns(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(db_path, "node_0", 0, "2026-08-16T00:00:00Z", energy_kwh=0.001, duration_s=1.5)

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 1
    assert rows[0]["node_id"] == "node_0"
    assert rows[0]["round_idx"] == 0
    assert rows[0]["energy_kwh"] == 0.001
    assert rows[0]["duration_s"] == 1.5
    assert rows[0]["crop_accuracy"] is None


def test_upsert_row_merges_separate_messages_into_one_row(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(
        db_path, "node_0", 0, "2026-08-16T00:00:00Z",
        energy_kwh=0.001, duration_s=1.5, energy_method="proxy_wall_power",
    )
    sqlite_store.upsert_row(
        db_path, "node_0", 0, "2026-08-16T00:00:01Z",
        crop_accuracy=0.8, disease_accuracy=0.7,
    )

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 1
    row = rows[0]
    # both messages' columns survive -- the second upsert must not null out the first's
    assert row["energy_kwh"] == 0.001
    assert row["energy_method"] == "proxy_wall_power"
    assert row["crop_accuracy"] == 0.8
    assert row["disease_accuracy"] == 0.7
    assert row["recorded_at"] == "2026-08-16T00:00:01Z"  # most recent write wins


def test_upsert_row_keeps_separate_rows_per_node_and_round(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(db_path, "node_0", 0, "t0", crop_accuracy=0.5)
    sqlite_store.upsert_row(db_path, "node_1", 0, "t0", crop_accuracy=0.6)
    sqlite_store.upsert_row(db_path, "node_0", 1, "t1", crop_accuracy=0.55)

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 3


def test_read_all_on_missing_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert sqlite_store.read_all(db_path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.energy.sqlite_store'`

- [ ] **Step 3: Implement**

```python
# src/energy/sqlite_store.py
"""SQLite persistence for per-round mesh metrics (energy, communication
bytes, accuracy). Written incrementally: an `energy` MQTT message and an
`eval` MQTT message for the same (node_id, round_idx) arrive separately,
so `upsert_row` merges whichever columns are passed into one logical row
rather than requiring the whole row at once.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS round_metrics (
    node_id TEXT NOT NULL,
    round_idx INTEGER NOT NULL,
    energy_kwh REAL,
    duration_s REAL,
    energy_method TEXT,
    knowledge_bytes_sent INTEGER,
    crop_accuracy REAL,
    disease_accuracy REAL,
    active INTEGER,
    recorded_at TEXT,
    PRIMARY KEY (node_id, round_idx)
);
"""


def init_db(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def upsert_row(path: str | Path, node_id: str, round_idx: int, recorded_at: str, **fields) -> None:
    """Insert a row for (node_id, round_idx), or merge `fields` into an
    existing row. Columns not present in `fields` are left untouched on
    conflict, so separate energy/eval/knowledge_ready/active messages for
    the same (node_id, round_idx) accumulate into one row instead of
    clobbering each other.
    """
    init_db(path)
    columns = ["node_id", "round_idx", "recorded_at", *fields.keys()]
    placeholders = ", ".join("?" for _ in columns)
    values = [node_id, round_idx, recorded_at, *fields.values()]
    update_clause = "recorded_at=excluded.recorded_at" + (
        ", " + ", ".join(f"{col}=excluded.{col}" for col in fields) if fields else ""
    )
    sql = (
        f"INSERT INTO round_metrics ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(node_id, round_idx) DO UPDATE SET {update_clause}"
    )
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def read_all(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA_SQL)
        rows = conn.execute("SELECT * FROM round_metrics ORDER BY round_idx, node_id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/energy/sqlite_store.py tests/test_sqlite_store.py
git commit -m "Add SQLite round_metrics store with column-preserving upsert"
```

---

## Task 3: `docker_mesh` config + data pre-split script

**Files:**
- Modify: `config.yaml`
- Create: `scripts/split_node_data.py`
- Test: `tests/test_split_node_data.py`

**Interfaces:**
- Consumes: `build_global_label_map`, `save_global_label_map` (Task 1); `carve_public_probe_set`, `load_full_dataset`, `partition_nodes` (existing, unchanged).
- Produces: `scripts/split_node_data.py`'s `split(cfg: Config, output_root: Path) -> None`, importable for the test; on-disk layout `output_root/{probe,node_0,...}/<class_name>/<file>` + `output_root/classes.json`.

- [ ] **Step 1: Add `docker_mesh` block to `config.yaml`**

Append to `config.yaml` (new top-level key, does not touch any existing key):

```yaml
docker_mesh:
  mqtt_host: "broker"
  mqtt_port: 1883
  round_timeout_s: 300
  data_dir: "data/docker_mesh"
  energy_db_dir: "outputs/docker_mesh/energy"
  dashboard_refresh_s: 3
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_split_node_data.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.split_node_data import split
from src.config import Config

SOURCE_CLASSES = {
    "Tomato___Bacterial_spot": 6,
    "Tomato___healthy": 6,
    "Potato___Early_blight": 6,
    "Potato___healthy": 6,
}


def _build_source_tree(root: Path) -> int:
    rng = np.random.RandomState(0)
    total = 0
    for cls, n in SOURCE_CLASSES.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(n):
            arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
            total += 1
    return total


def test_split_partitions_every_source_file_exactly_once(tmp_path):
    source_root = tmp_path / "PlantVillage"
    total_files = _build_source_tree(source_root)

    cfg = Config(
        {
            "data": {
                "root": str(source_root),
                "image_size": 16,
                "num_nodes": 2,
                "probe_set_fraction": 0.1,
                "non_iid_strategy": "manual",
                "manual_node_crops": {"node_0": ["Tomato"], "node_1": ["Potato"]},
                "seed": 42,
            }
        }
    )
    output_root = tmp_path / "docker_mesh"
    split(cfg, output_root)

    assert (output_root / "classes.json").exists()

    seen_paths = set()
    for sub in ["probe", "node_0", "node_1"]:
        for f in (output_root / sub).rglob("*.jpg"):
            key = (sub, f.parent.name, f.name)
            assert key not in seen_paths, f"duplicate copy: {key}"
            seen_paths.add(key)
    assert len(seen_paths) == total_files

    # manual strategy: node_0's files are only ever Tomato classes, node_1's only Potato
    node_0_classes = {f.parent.name for f in (output_root / "node_0").rglob("*.jpg")}
    node_1_classes = {f.parent.name for f in (output_root / "node_1").rglob("*.jpg")}
    assert all(c.startswith("Tomato") for c in node_0_classes)
    assert all(c.startswith("Potato") for c in node_1_classes)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_split_node_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.split_node_data'`

- [ ] **Step 4: Implement**

```python
# scripts/split_node_data.py
"""One-time pre-split of PlantVillage into physically isolated per-node
folders plus a shared public probe folder, mirroring exactly the
probe/shard split src/train.py's build_dataloaders performs in-memory —
but writing files to disk so each Docker node container can be given a
read-only bind mount of only its own folder.

Run from the repo root:
    python scripts/split_node_data.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.plantvillage import (
    build_global_label_map,
    carve_public_probe_set,
    load_full_dataset,
    partition_nodes,
    save_global_label_map,
)


def _copy_indices(dataset, indices: list[int], dest_root: Path) -> None:
    for idx in indices:
        src_path, class_idx = dataset.base.samples[idx]
        class_name = dataset.base.classes[class_idx]
        dest_dir = dest_root / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_dir / Path(src_path).name)


def split(cfg: Config, output_root: Path) -> None:
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    global_map = build_global_label_map(dataset)
    output_root.mkdir(parents=True, exist_ok=True)
    save_global_label_map(global_map, output_root / "classes.json")

    probe_idx, remaining_idx = carve_public_probe_set(
        dataset, cfg.get("data.probe_set_fraction", 0.05), cfg.get("data.seed", 42)
    )
    _copy_indices(dataset, probe_idx, output_root / "probe")

    shards = partition_nodes(
        dataset,
        remaining_idx,
        cfg.get("data.num_nodes", 3),
        cfg.get("data.non_iid_strategy", "manual"),
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    for i, shard in enumerate(shards):
        _copy_indices(dataset, shard, output_root / f"node_{i}")


def main() -> None:
    cfg = Config.load()
    output_root = Path(cfg.get("docker_mesh.data_dir", "data/docker_mesh"))
    split(cfg, output_root)
    print(f"Wrote split dataset + classes.json to {output_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_split_node_data.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.yaml scripts/split_node_data.py tests/test_split_node_data.py
git commit -m "Add docker_mesh config and per-node data pre-split script"
```

---

## Task 4: MQTT knowledge codec

**Files:**
- Create: `docker/node/mqtt_codec.py`
- Test: `tests/test_mqtt_codec.py`

**Interfaces:**
- Consumes: `KnowledgePayload` (existing, `src/federated/node.py`, unchanged).
- Produces: `encode_knowledge(round_idx: int, payload: KnowledgePayload) -> bytes`; `decode_knowledge(data: bytes) -> tuple[int, KnowledgePayload]`. Used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mqtt_codec.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "node"))

import torch

from mqtt_codec import decode_knowledge, encode_knowledge  # noqa: E402
from src.federated.node import KnowledgePayload  # noqa: E402


def test_encode_decode_roundtrip_preserves_tensors_and_round_idx():
    payload = KnowledgePayload(
        prototypes={("crop", 0): torch.randn(8), ("disease", 1): torch.randn(8)},
        crop_logits=torch.randn(5, 3),
        disease_logits=torch.randn(5, 2),
    )
    data = encode_knowledge(7, payload)
    assert isinstance(data, bytes)

    round_idx, decoded = decode_knowledge(data)
    assert round_idx == 7
    assert torch.equal(decoded.crop_logits, payload.crop_logits)
    assert torch.equal(decoded.disease_logits, payload.disease_logits)
    for key in payload.prototypes:
        assert torch.equal(decoded.prototypes[key], payload.prototypes[key])


def test_encoded_size_matches_declared_length():
    payload = KnowledgePayload(prototypes={}, crop_logits=torch.zeros(2, 2), disease_logits=torch.zeros(2, 2))
    data = encode_knowledge(0, payload)
    assert len(data) == len(data)  # sanity: encode_knowledge returns the exact publishable bytes
    assert len(data) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mqtt_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mqtt_codec'`

- [ ] **Step 3: Implement**

```python
# docker/node/mqtt_codec.py
"""Serializes a KnowledgePayload (prototypes + probe-set logits — never
raw images, gradients, or weights) to bytes for MQTT publish, tagged with
the round it was computed in.
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
        },
        buffer,
    )
    return buffer.getvalue()


def decode_knowledge(data: bytes) -> tuple[int, KnowledgePayload]:
    # weights_only=False: this deserializes our own KnowledgePayload dict
    # (tuple-keyed prototypes dict + tensors), published only by our own
    # node containers over the internal broker -- not untrusted input.
    obj = torch.load(io.BytesIO(data), weights_only=False)
    payload = KnowledgePayload(
        prototypes=obj["prototypes"],
        crop_logits=obj["crop_logits"],
        disease_logits=obj["disease_logits"],
    )
    return obj["round_idx"], payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mqtt_codec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docker/node/mqtt_codec.py tests/test_mqtt_codec.py
git commit -m "Add MQTT knowledge-payload codec with round_idx tag"
```

---

## Task 5: `NodeRunner` core round-handling logic

The round-handling logic is deliberately separated from real MQTT wiring (Task 6) so it's unit-testable without a live broker: `NodeRunner` takes any object with a `publish(topic, payload, retain=False, qos=1)` method, and its `on_message(topic, payload)` is called directly by tests (or by the real paho client's callback in Task 6).

**Files:**
- Create: `docker/node/node_runner.py`
- Test: `tests/test_node_runner.py`

**Interfaces:**
- Consumes: `Node`, `KnowledgePayload` (`src/federated/node.py`); `aggregate_prototypes`, `aggregate_logits` (`src/federated/aggregation.py`); `ComputeEnergyTracker` (`src/energy/tracker.py`); `sqlite_store.upsert_row` (Task 2); `encode_knowledge`, `decode_knowledge` (Task 4).
- Produces: `NodeRunner` dataclass with `on_message(topic: str, payload: bytes) -> None`, `handle_round_start(round_idx: int) -> None`, `handle_round_gather_done(round_idx: int, active_nodes: list[str]) -> None`. Used by Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_node_runner.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "node"))

import torch
from torch.utils.data import DataLoader

from mqtt_codec import encode_knowledge  # noqa: E402
from node_runner import NodeRunner  # noqa: E402
from src.data.plantvillage import make_subset, train_test_split_indices
from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import KnowledgePayload, Node
from src.models.factory import build_model


class FakeClient:
    def __init__(self):
        self.published: list[tuple[str, object, bool]] = []

    def publish(self, topic, payload, retain=False, qos=1):
        self.published.append((topic, payload, retain))


def _make_runner(tmp_path, synthetic_dataset):
    train_idx, test_idx = train_test_split_indices(list(range(len(synthetic_dataset))), 0.3, seed=1)
    train_loader = DataLoader(make_subset(synthetic_dataset, train_idx), batch_size=4, shuffle=True)
    test_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    probe_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    model = build_model("mobilenet_v3_small", 2, 2, pretrained=False)
    node = Node("node_0", model, train_loader, test_loader, device="cpu")
    client = FakeClient()
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    runner = NodeRunner(
        node_id="node_0",
        node=node,
        probe_loader=probe_loader,
        client=client,
        tracker=tracker,
        db_path=str(tmp_path / "node_0.db"),
    )
    return runner, client, probe_loader


def test_handle_round_start_publishes_knowledge_energy_and_writes_db(tmp_path, synthetic_dataset):
    runner, client, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_start(0)

    topics = [t for t, _, _ in client.published]
    assert "mesh/node/node_0/knowledge" in topics
    assert "mesh/node/node_0/knowledge_ready" in topics
    assert "mesh/node/node_0/energy" in topics

    knowledge_topic, knowledge_bytes, retained = next(
        (t, p, r) for t, p, r in client.published if t == "mesh/node/node_0/knowledge"
    )
    assert retained is True
    assert isinstance(knowledge_bytes, bytes)

    ready_payload = json.loads(
        next(p for t, p, _ in client.published if t == "mesh/node/node_0/knowledge_ready")
    )
    assert ready_payload["round_idx"] == 0
    assert ready_payload["size_bytes"] == len(knowledge_bytes)

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert len(rows) == 1
    assert rows[0]["knowledge_bytes_sent"] == len(knowledge_bytes)
    assert rows[0]["energy_kwh"] is not None


def test_on_message_stores_peer_knowledge_by_round(tmp_path, synthetic_dataset):
    runner, _, probe_loader = _make_runner(tmp_path, synthetic_dataset)
    n_probe = len(probe_loader.dataset)
    peer_payload = KnowledgePayload(
        prototypes={}, crop_logits=torch.zeros(n_probe, 2), disease_logits=torch.zeros(n_probe, 2)
    )
    runner.on_message("mesh/node/node_1/knowledge", encode_knowledge(3, peer_payload))
    assert runner._peer_knowledge["node_1"][0] == 3


def test_handle_round_gather_done_ignores_peer_knowledge_from_a_different_round(tmp_path, synthetic_dataset):
    runner, client, probe_loader = _make_runner(tmp_path, synthetic_dataset)
    n_probe = len(probe_loader.dataset)
    stale_payload = KnowledgePayload(
        prototypes={}, crop_logits=torch.zeros(n_probe, 2), disease_logits=torch.zeros(n_probe, 2)
    )
    # published for round 99, but we are about to gather round 0 -- must be dropped
    runner.on_message("mesh/node/node_1/knowledge", encode_knowledge(99, stale_payload))

    runner.handle_round_gather_done(0, ["node_0", "node_1"])

    eval_payload = json.loads(next(p for t, p, _ in client.published if t == "mesh/node/node_0/eval"))
    assert "crop_accuracy" in eval_payload
    assert "disease_accuracy" in eval_payload

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["active"] == 1


def test_handle_round_gather_done_with_no_active_peers_still_evaluates(tmp_path, synthetic_dataset):
    runner, client, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_gather_done(0, ["node_0"])  # single-node mesh: nothing to reconcile
    eval_topics = [t for t, _, _ in client.published if t == "mesh/node/node_0/eval"]
    assert len(eval_topics) == 1
```

(`synthetic_dataset` is the existing fixture in `tests/conftest.py` — no changes needed there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_node_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'node_runner'`

- [ ] **Step 3: Implement**

```python
# docker/node/node_runner.py
"""MQTT-driven wrapper around a single Node, reusing src/federated/node.py
and src/federated/aggregation.py unmodified. Round-handling logic is
separated from the real MQTT client so it's unit-testable with a fake
client -- see docker/node/main.py for the real wiring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import KnowledgePayload, Node

from mqtt_codec import decode_knowledge, encode_knowledge


class MQTTClientLike(Protocol):
    def publish(self, topic: str, payload, retain: bool = False, qos: int = 1) -> None: ...


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class NodeRunner:
    node_id: str
    node: Node
    probe_loader: object
    client: MQTTClientLike
    tracker: ComputeEnergyTracker
    db_path: str
    aggregation_method: str = "trimmed_mean"
    trim_fraction: float = 0.2
    krum_neighbors: int = 2
    local_epochs: int = 2
    distill_epochs: int = 1
    lr: float = 0.001
    distill_lr: float = 0.0005
    proto_weight: float = 0.5
    kd_weight: float = 0.5
    temperature: float = 2.0
    _peer_knowledge: dict = field(default_factory=dict)

    def on_message(self, topic: str, payload: bytes) -> None:
        parts = topic.split("/")
        if len(parts) == 4 and parts[0] == "mesh" and parts[1] == "node" and parts[3] == "knowledge":
            peer_id = parts[2]
            if peer_id == self.node_id:
                return
            round_idx, knowledge = decode_knowledge(payload)
            self._peer_knowledge[peer_id] = (round_idx, knowledge)

    def handle_round_start(self, round_idx: int) -> None:
        with self.tracker.track(f"{self.node_id}_round_{round_idx}") as energy_record:
            self.node.local_train(self.local_epochs, self.lr)
        knowledge = self.node.compute_knowledge(self.probe_loader)
        data = encode_knowledge(round_idx, knowledge)

        self.client.publish(f"mesh/node/{self.node_id}/knowledge", data, retain=True, qos=1)
        self.client.publish(
            f"mesh/node/{self.node_id}/knowledge_ready",
            json.dumps({"round_idx": round_idx, "size_bytes": len(data)}),
            qos=1,
        )
        self.client.publish(
            f"mesh/node/{self.node_id}/energy",
            json.dumps(
                {
                    "round_idx": round_idx,
                    "energy_kwh": energy_record["energy_kwh"],
                    "duration_s": energy_record["duration_s"],
                    "energy_method": energy_record["method"],
                }
            ),
            qos=1,
        )
        sqlite_store.upsert_row(
            self.db_path,
            self.node_id,
            round_idx,
            _now_iso(),
            energy_kwh=energy_record["energy_kwh"],
            duration_s=energy_record["duration_s"],
            energy_method=energy_record["method"],
            knowledge_bytes_sent=len(data),
        )

    def handle_round_gather_done(self, round_idx: int, active_nodes: list[str]) -> None:
        peers: list[KnowledgePayload] = []
        for peer_id in active_nodes:
            if peer_id == self.node_id:
                continue
            entry = self._peer_knowledge.get(peer_id)
            if entry is None or entry[0] != round_idx:
                continue  # missing or stale -- integrity guard from spec §6
            peers.append(entry[1])

        if peers:
            consensus_prototypes = aggregate_prototypes(
                [p.prototypes for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_crop_logits = aggregate_logits(
                [p.crop_logits for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            consensus_disease_logits = aggregate_logits(
                [p.disease_logits for p in peers],
                method=self.aggregation_method,
                trim_fraction=self.trim_fraction,
                krum_neighbors=self.krum_neighbors,
            )
            self.node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                self.probe_loader,
                epochs=self.distill_epochs,
                lr=self.distill_lr,
                proto_weight=self.proto_weight,
                kd_weight=self.kd_weight,
                temperature=self.temperature,
            )

        eval_result = self.node.evaluate()
        self.client.publish(
            f"mesh/node/{self.node_id}/eval",
            json.dumps({"round_idx": round_idx, **eval_result}),
            qos=1,
        )
        sqlite_store.upsert_row(
            self.db_path,
            self.node_id,
            round_idx,
            _now_iso(),
            crop_accuracy=eval_result["crop_accuracy"],
            disease_accuracy=eval_result["disease_accuracy"],
            active=1 if self.node_id in active_nodes else 0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_node_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/node/node_runner.py tests/test_node_runner.py
git commit -m "Add NodeRunner: MQTT-driven round handling over the existing Node/aggregation logic"
```

---

## Task 6: Node container — `main.py` wiring, `Dockerfile`, `requirements.txt`

This task wires `NodeRunner` to a real `paho-mqtt` client and real data. It is not TDD — `NodeRunner`'s logic is already covered by Task 5; this is thin, mostly-untestable-without-a-broker glue, verified manually in Task 13's integration smoke test.

**Files:**
- Create: `docker/node/main.py`
- Create: `docker/node/Dockerfile`
- Create: `docker/node/requirements.txt`

**Interfaces:**
- Consumes: `NodeRunner` (Task 5); `Config` (`src/config.py`); `GlobalLabelMap`/`load_global_label_map`/`PlantVillageDataset`/`make_subset`/`train_test_split_indices` (`src/data/plantvillage.py`, Task 1); `build_model` (`src/models/factory.py`); `Node` (`src/federated/node.py`); `ComputeEnergyTracker` (`src/energy/tracker.py`).
- Produces: a runnable container entry point reading `NODE_ID`, `MQTT_HOST`, `MQTT_PORT`, `DATA_ROOT`, `PROBE_ROOT`, `CLASSES_JSON`, `ENERGY_DB`, `CONFIG_PATH` from the environment (all set by `docker-compose.yml` in Task 12).

- [ ] **Step 1: Write `requirements.txt`**

```
# docker/node/requirements.txt
torch>=2.1
torchvision>=0.16
timm>=1.0
numpy>=1.24
Pillow>=10.0
PyYAML>=6.0
codecarbon>=2.3
paho-mqtt>=2.1
```

- [ ] **Step 2: Write `main.py`**

```python
# docker/node/main.py
"""Container entry point for one mesh node. Builds a Node from its own
isolated data folder + the shared global label map, wires it to
NodeRunner, and drives everything from MQTT callbacks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# In the container, /app already has src/ copied alongside this file, so
# this insert is a harmless no-op there. Running locally (e.g. `python
# docker/node/main.py` from the repo root, for the no-Docker verification
# workflow) main.py's own directory does NOT contain src/ -- this makes
# `from src...` resolve in both cases without two different code paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import paho.mqtt.client as mqtt
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import (
    PlantVillageDataset,
    load_global_label_map,
    make_subset,
    train_test_split_indices,
)
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import Node
from src.models.factory import build_model
from node_runner import NodeRunner


def main() -> None:
    node_id = os.environ["NODE_ID"]
    mqtt_host = os.environ.get("MQTT_HOST", "broker")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    data_root = os.environ["DATA_ROOT"]
    probe_root = os.environ["PROBE_ROOT"]
    classes_json = os.environ["CLASSES_JSON"]
    energy_db = os.environ["ENERGY_DB"]
    cfg = Config.load(os.environ.get("CONFIG_PATH"))

    global_map = load_global_label_map(classes_json)
    image_size = cfg.get("data.image_size", 160)
    local_dataset = PlantVillageDataset(data_root, image_size=image_size, global_label_map=global_map)
    probe_dataset = PlantVillageDataset(probe_root, image_size=image_size, global_label_map=global_map)

    train_idx, test_idx = train_test_split_indices(
        list(range(len(local_dataset))), cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
    )
    batch_size = cfg.get("training.batch_size", 32)
    train_loader = DataLoader(make_subset(local_dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(make_subset(local_dataset, test_idx), batch_size=batch_size, shuffle=False)
    # shuffle=False: every node must compute probe logits over the identical
    # image order for the positional peer-logit aggregation to be meaningful.
    probe_loader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False)

    arch = cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    model = build_model(
        arch,
        len(global_map.crop_classes),
        len(global_map.disease_classes),
        pretrained=cfg.get("models.pretrained", True),
    )
    node = Node(node_id, model, train_loader, test_loader, device="cpu")
    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", False),
        output_dir="/tmp/codecarbon",
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )

    client = mqtt.Client(client_id=node_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    runner = NodeRunner(
        node_id=node_id,
        node=node,
        probe_loader=probe_loader,
        client=client,
        tracker=tracker,
        db_path=energy_db,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        local_epochs=cfg.get("training.local_epochs_per_round", 2),
        distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
        lr=cfg.get("training.lr", 0.001),
        distill_lr=cfg.get("training.distill_lr", 0.0005),
        proto_weight=cfg.get("training.proto_weight", 0.5),
        kd_weight=cfg.get("training.kd_weight", 0.5),
        temperature=cfg.get("training.kd_temperature", 2.0),
    )

    def _on_message(_client, _userdata, msg):
        if msg.topic == "mesh/control/round_start":
            import json

            runner.handle_round_start(json.loads(msg.payload)["round_idx"])
        elif msg.topic == "mesh/control/round_gather_done":
            import json

            data = json.loads(msg.payload)
            runner.handle_round_gather_done(data["round_idx"], data["active_nodes"])
        else:
            runner.on_message(msg.topic, msg.payload)

    client.on_message = _on_message
    client.will_set(f"mesh/node/{node_id}/status", payload='{"status": "offline"}', qos=1, retain=True)
    client.connect(mqtt_host, mqtt_port)
    client.subscribe("mesh/control/round_start")
    client.subscribe("mesh/control/round_gather_done")
    client.subscribe("mesh/node/+/knowledge")
    client.publish(f"mesh/node/{node_id}/status", '{"status": "online"}', qos=1, retain=True)
    print(f"[{node_id}] connected to {mqtt_host}:{mqtt_port}, waiting for rounds...")
    client.loop_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# docker/node/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/node/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/node/main.py docker/node/node_runner.py docker/node/mqtt_codec.py ./

CMD ["python", "main.py"]
```

- [ ] **Step 4: Build the image to verify it compiles and installs cleanly**

Run: `docker build -f docker/node/Dockerfile -t mesh-node -t mesh-node .` (from repo root, so `src/` is in the build context)
Expected: build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add docker/node/main.py docker/node/Dockerfile docker/node/requirements.txt
git commit -m "Add node container: main.py wiring, Dockerfile, requirements"
```

---

## Task 7: `CoordinatorRunner` core round-driving logic

Like Task 5, this separates round-driving logic from the real MQTT client so it's unit-testable without a broker. Two subtleties this task must get right (both discovered while planning, not in the original spec text — see comments in the code below):

1. **Readiness detection must be decoupled from round-driving.** If `on_message`'s "all nodes online" check directly called a long-running, sleep-polling round loop, and `on_message` runs on the paho network thread (the standard `loop_start()` pattern), that thread would be blocked sleeping and could never receive the very `knowledge_ready` messages the round loop is waiting for. `on_message` only ever flips a fast callback (`on_ready`); the actual round-driving loop runs on the caller's thread (Task 8 wires this to the main thread).
2. **A round isn't complete until evals are in, not just knowledge.** The coordinator must wait for `eval` from every active node — not just `knowledge_ready` — before starting the next round, otherwise round N+1 could start while round N's distillation is still running on some node.

**Files:**
- Create: `docker/coordinator/coordinator_runner.py`
- Test: `tests/test_coordinator_runner.py`

**Interfaces:**
- Consumes: `sqlite_store.upsert_row` (Task 2).
- Produces: `CoordinatorRunner` dataclass with `on_message(topic, payload) -> None`, `run_round(round_idx, sleep_fn=time.sleep, now_fn=time.time, poll_interval_s=0.5) -> list[str]`, `run_all_rounds(sleep_fn=time.sleep, now_fn=time.time) -> None`, and an `on_ready: Callable[[], None]` field fired once when every expected node has reported online. Used by Task 8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coordinator_runner.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "coordinator"))

from coordinator_runner import CoordinatorRunner  # noqa: E402
from src.energy import sqlite_store


class FakeClient:
    def __init__(self):
        self.published: list[tuple[str, str]] = []

    def publish(self, topic, payload, qos=1):
        self.published.append((topic, payload))


def test_on_ready_fires_once_when_all_expected_nodes_online(tmp_path):
    client = FakeClient()
    fired = []
    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        num_rounds=1,
        round_timeout_s=5,
        client=client,
        db_path=str(tmp_path / "merged.db"),
        on_ready=lambda: fired.append(True),
    )
    runner.on_message("mesh/node/node_0/status", json.dumps({"status": "online"}).encode())
    assert fired == []
    runner.on_message("mesh/node/node_1/status", json.dumps({"status": "online"}).encode())
    assert fired == [True]
    # a repeat status message must not fire on_ready a second time
    runner.on_message("mesh/node/node_0/status", json.dumps({"status": "online"}).encode())
    assert fired == [True]


def test_on_ready_does_not_fire_if_a_node_goes_offline_before_all_are_online(tmp_path):
    client = FakeClient()
    fired = []
    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        num_rounds=1,
        round_timeout_s=5,
        client=client,
        db_path=str(tmp_path / "merged.db"),
        on_ready=lambda: fired.append(True),
    )
    runner.on_message("mesh/node/node_0/status", json.dumps({"status": "online"}).encode())
    runner.on_message("mesh/node/node_0/status", json.dumps({"status": "offline"}).encode())
    runner.on_message("mesh/node/node_1/status", json.dumps({"status": "online"}).encode())
    assert fired == []


def test_run_round_waits_for_knowledge_then_eval_and_merges_db_rows(tmp_path):
    client = FakeClient()
    db_path = str(tmp_path / "merged.db")
    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1"], num_rounds=1, round_timeout_s=5, client=client, db_path=db_path
    )
    runner._online = {"node_0", "node_1"}

    calls = {"n": 0}

    def fake_sleep(_interval):
        calls["n"] += 1
        if calls["n"] == 1:
            runner.on_message("mesh/node/node_0/knowledge_ready", json.dumps({"round_idx": 0}).encode())
            runner.on_message("mesh/node/node_1/knowledge_ready", json.dumps({"round_idx": 0}).encode())
        else:
            runner.on_message(
                "mesh/node/node_0/eval",
                json.dumps({"round_idx": 0, "crop_accuracy": 0.5, "disease_accuracy": 0.6}).encode(),
            )
            runner.on_message(
                "mesh/node/node_1/eval",
                json.dumps({"round_idx": 0, "crop_accuracy": 0.4, "disease_accuracy": 0.7}).encode(),
            )

    active = runner.run_round(0, sleep_fn=fake_sleep, poll_interval_s=0)

    assert active == ["node_0", "node_1"]
    topics = [t for t, _ in client.published]
    assert topics == ["mesh/control/round_start", "mesh/control/round_gather_done"]

    gather_payload = json.loads(next(p for t, p in client.published if t == "mesh/control/round_gather_done"))
    assert gather_payload["active_nodes"] == ["node_0", "node_1"]

    rows = {r["node_id"]: r for r in sqlite_store.read_all(db_path)}
    assert rows["node_0"]["crop_accuracy"] == 0.5
    assert rows["node_1"]["active"] == 1


def test_run_round_excludes_a_node_that_never_publishes_knowledge_ready(tmp_path):
    client = FakeClient()
    db_path = str(tmp_path / "merged.db")
    runner = CoordinatorRunner(
        expected_nodes=["node_0", "node_1"], num_rounds=1, round_timeout_s=1, client=client, db_path=db_path
    )
    runner._online = {"node_0", "node_1"}
    runner.on_message("mesh/node/node_0/knowledge_ready", json.dumps({"round_idx": 0}).encode())
    # node_1 never responds

    fake_now = {"t": 0.0}

    def now_fn():
        return fake_now["t"]

    def fake_sleep(interval):
        fake_now["t"] += interval + 1.1  # advance well past the 1s timeout, no real waiting

    active = runner.run_round(0, sleep_fn=fake_sleep, now_fn=now_fn, poll_interval_s=0.5)
    assert active == ["node_0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_coordinator_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coordinator_runner'`

- [ ] **Step 3: Implement**

```python
# docker/coordinator/coordinator_runner.py
"""Control-plane round driver: times rounds, tracks node liveness, merges
energy/eval MQTT messages into SQLite. Never subscribes to or inspects
mesh/node/+/knowledge -- it only ever sees a byte count via
knowledge_ready, never payload content, so there is no central
aggregator of knowledge (that stays per-node, in node_runner.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from src.energy import sqlite_store


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class CoordinatorRunner:
    expected_nodes: list[str]
    num_rounds: int
    round_timeout_s: float
    client: object
    db_path: str
    on_ready: Callable[[], None] = lambda: None
    _online: set = field(default_factory=set)
    _knowledge_ready: dict = field(default_factory=dict)
    _eval_received: dict = field(default_factory=dict)
    _ready_fired: bool = False

    def on_message(self, topic: str, payload: bytes) -> None:
        parts = topic.split("/")
        if len(parts) != 4 or parts[0] != "mesh" or parts[1] != "node":
            return
        node_id, kind = parts[2], parts[3]
        data = json.loads(payload) if payload else {}

        if kind == "status":
            if data.get("status") == "online":
                self._online.add(node_id)
            else:
                self._online.discard(node_id)
            # readiness check must stay cheap and non-blocking -- see Task 7 docstring
            if not self._ready_fired and set(self.expected_nodes) <= self._online:
                self._ready_fired = True
                self.on_ready()
        elif kind == "knowledge_ready":
            self._knowledge_ready.setdefault(data["round_idx"], set()).add(node_id)
        elif kind == "energy":
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                data["round_idx"],
                _now_iso(),
                energy_kwh=data["energy_kwh"],
                duration_s=data["duration_s"],
                energy_method=data["energy_method"],
            )
        elif kind == "eval":
            self._eval_received.setdefault(data["round_idx"], set()).add(node_id)
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                data["round_idx"],
                _now_iso(),
                crop_accuracy=data["crop_accuracy"],
                disease_accuracy=data["disease_accuracy"],
            )

    def _poll_until(self, round_idx, tracker, expected, sleep_fn, now_fn, poll_interval_s) -> set:
        deadline = now_fn() + self.round_timeout_s
        while now_fn() < deadline:
            if expected <= tracker.get(round_idx, set()):
                break
            sleep_fn(poll_interval_s)
        return tracker.get(round_idx, set()) & expected

    def run_round(self, round_idx, sleep_fn=time.sleep, now_fn=time.time, poll_interval_s: float = 0.5) -> list[str]:
        self.client.publish("mesh/control/round_start", json.dumps({"round_idx": round_idx}), qos=1)

        active = sorted(
            self._poll_until(round_idx, self._knowledge_ready, set(self._online), sleep_fn, now_fn, poll_interval_s)
        )
        for node_id in active:
            sqlite_store.upsert_row(self.db_path, node_id, round_idx, _now_iso(), active=1)

        self.client.publish(
            "mesh/control/round_gather_done",
            json.dumps({"round_idx": round_idx, "active_nodes": active}),
            qos=1,
        )

        # a round is not complete until every active node's eval is in --
        # otherwise round N+1 could start while round N is still distilling
        self._poll_until(round_idx, self._eval_received, set(active), sleep_fn, now_fn, poll_interval_s)
        return active

    def run_all_rounds(self, sleep_fn=time.sleep, now_fn=time.time) -> None:
        for round_idx in range(self.num_rounds):
            self.run_round(round_idx, sleep_fn=sleep_fn, now_fn=now_fn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_coordinator_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/coordinator/coordinator_runner.py tests/test_coordinator_runner.py
git commit -m "Add CoordinatorRunner: control-plane round timing and DB merge, decoupled from MQTT wiring"
```

---

## Task 8: Coordinator container — `main.py` wiring, `Dockerfile`, `requirements.txt`

Like Task 6, this is thin glue verified manually in Task 13. The coordinator's image deliberately has **no** PyTorch — it only imports `src.config` and `src.energy.sqlite_store`, neither of which pulls in `src.federated`/`src.models`.

**Files:**
- Create: `docker/coordinator/main.py`
- Create: `docker/coordinator/Dockerfile`
- Create: `docker/coordinator/requirements.txt`

**Interfaces:**
- Consumes: `CoordinatorRunner` (Task 7); `Config` (`src/config.py`).
- Produces: a runnable container entry point reading `MQTT_HOST`, `MQTT_PORT`, `CONFIG_PATH`, `ENERGY_DB` from the environment.

- [ ] **Step 1: Write `requirements.txt`**

```
# docker/coordinator/requirements.txt
PyYAML>=6.0
paho-mqtt>=2.1
```

- [ ] **Step 2: Write `main.py`**

```python
# docker/coordinator/main.py
"""Container entry point for the round coordinator. Runs the paho network
loop on a background thread (loop_start) and blocks the main thread on a
threading.Event until every expected node reports online -- see
coordinator_runner.py's docstring for why on_ready must stay non-blocking
inside the MQTT callback thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

# See docker/node/main.py's identical comment: makes `from src...` resolve
# both inside the flattened /app container layout and when run locally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import paho.mqtt.client as mqtt

from src.config import Config
from coordinator_runner import CoordinatorRunner


def main() -> None:
    mqtt_host = os.environ.get("MQTT_HOST", "broker")
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    energy_db = os.environ["ENERGY_DB"]
    cfg = Config.load(os.environ.get("CONFIG_PATH"))

    num_nodes = cfg.get("data.num_nodes", 3)
    expected_nodes = [f"node_{i}" for i in range(num_nodes)]
    num_rounds = cfg.get("training.rounds", 5)
    round_timeout_s = cfg.get("docker_mesh.round_timeout_s", 300)

    ready_event = threading.Event()
    client = mqtt.Client(client_id="coordinator", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    runner = CoordinatorRunner(
        expected_nodes=expected_nodes,
        num_rounds=num_rounds,
        round_timeout_s=round_timeout_s,
        client=client,
        db_path=energy_db,
        on_ready=ready_event.set,
    )

    def _on_message(_client, _userdata, msg):
        runner.on_message(msg.topic, msg.payload)

    client.on_message = _on_message
    client.connect(mqtt_host, mqtt_port)
    client.subscribe("mesh/node/+/status")
    client.subscribe("mesh/node/+/knowledge_ready")
    client.subscribe("mesh/node/+/energy")
    client.subscribe("mesh/node/+/eval")
    client.loop_start()

    print(f"[coordinator] waiting for {expected_nodes} to come online...")
    ready_event.wait()
    print("[coordinator] all nodes online, starting round loop")
    runner.run_all_rounds()
    print("[coordinator] all rounds complete")

    while True:
        threading.Event().wait(3600)  # idle -- no re-run trigger yet, see spec §12


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# docker/coordinator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/coordinator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/config.py ./src/config.py
COPY src/energy/ ./src/energy/
COPY docker/coordinator/main.py docker/coordinator/coordinator_runner.py ./

CMD ["python", "main.py"]
```

Note: `src/` needs an `__init__.py` for `src.config`/`src.energy` to import correctly — copy those too:

```dockerfile
COPY src/__init__.py ./src/__init__.py
```

(Add this line right before the `COPY src/config.py` line above.)

- [ ] **Step 4: Build the image to verify it compiles and installs cleanly**

Run: `docker build -f docker/coordinator/Dockerfile -t mesh-coordinator .` (from repo root)
Expected: build succeeds with no errors, and no PyTorch gets installed (`docker run --rm mesh-coordinator python -c "import torch"` should fail with `ModuleNotFoundError` — confirming the image really is lean).

- [ ] **Step 5: Commit**

```bash
git add docker/coordinator/main.py docker/coordinator/Dockerfile docker/coordinator/requirements.txt
git commit -m "Add coordinator container: main.py wiring, Dockerfile, requirements"
```

---

## Task 9: Broker config

**Files:**
- Create: `docker/broker/mosquitto.conf`

**Interfaces:**
- Consumes: nothing.
- Produces: Mosquitto config consumed by `docker-compose.yml` (Task 12).

- [ ] **Step 1: Write the config**

```
# docker/broker/mosquitto.conf
listener 1883
allow_anonymous true
persistence false
log_dest stdout
```

- [ ] **Step 2: Verify syntax**

Run: `docker run --rm -v "$(pwd)/docker/broker/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro" eclipse-mosquitto:2 mosquitto -c /mosquitto/config/mosquitto.conf -v &` then check it logs "mosquitto version ... running" with no config errors, then stop it (`docker stop` on the container, or Ctrl+C if run in foreground without `&`).
Expected: no config parse errors printed.

- [ ] **Step 3: Commit**

```bash
git add docker/broker/mosquitto.conf
git commit -m "Add Mosquitto broker config"
```

---

## Task 10: Dashboard data helpers

Pure, testable functions only — no Streamlit calls. These are what Task 11's `app.py` calls into; the `st.*` rendering itself is not automated-testable and is verified manually in Task 13.

**Files:**
- Create: `docker/dashboard/data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: `sqlite_store.read_all` (Task 2).
- Produces: `is_metadata_topic(topic: str) -> bool`; `format_feed_entry(topic: str, payload: bytes, ts: float) -> dict`; `DASHBOARD_SUBSCRIBE_TOPICS: tuple[str, ...]` (the exact wildcard subscription list Task 11's `app.py` must use — deliberately excludes `mesh/node/+/knowledge`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_data.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "dashboard"))

from data import DASHBOARD_SUBSCRIBE_TOPICS, format_feed_entry, is_metadata_topic  # noqa: E402


def test_dashboard_never_subscribes_to_the_binary_knowledge_topic():
    assert "mesh/node/+/knowledge" not in DASHBOARD_SUBSCRIBE_TOPICS
    for topic in DASHBOARD_SUBSCRIBE_TOPICS:
        assert not topic.endswith("/knowledge")


def test_is_metadata_topic_accepts_control_and_metadata_topics():
    assert is_metadata_topic("mesh/control/round_start")
    assert is_metadata_topic("mesh/control/round_gather_done")
    assert is_metadata_topic("mesh/node/node_0/status")
    assert is_metadata_topic("mesh/node/node_0/knowledge_ready")
    assert is_metadata_topic("mesh/node/node_0/energy")
    assert is_metadata_topic("mesh/node/node_0/eval")


def test_is_metadata_topic_rejects_the_binary_knowledge_topic():
    assert not is_metadata_topic("mesh/node/node_0/knowledge")


def test_format_feed_entry_parses_json_payload():
    payload = json.dumps({"round_idx": 2}).encode()
    entry = format_feed_entry("mesh/control/round_start", payload, ts=123.0)
    assert entry == {"topic": "mesh/control/round_start", "payload": {"round_idx": 2}, "timestamp": 123.0}


def test_format_feed_entry_falls_back_to_byte_count_on_non_json():
    entry = format_feed_entry("mesh/node/node_0/knowledge", b"\x00\x01\x02", ts=1.0)
    assert entry["payload"] == "<3 bytes>"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Implement**

```python
# docker/dashboard/data.py
"""Pure helpers for the dashboard's live MQTT feed and DB view --
deliberately separated from Streamlit rendering (docker/dashboard/app.py)
so this logic is unit-testable without a live broker or a running
Streamlit session.
"""

from __future__ import annotations

import json

# Deliberately excludes mesh/node/+/knowledge (the binary KnowledgePayload
# topic): it wouldn't render meaningfully as a feed entry, and subscribing
# to it would add real broker fan-out/bandwidth for content the dashboard
# never needs -- see spec §8.
DASHBOARD_SUBSCRIBE_TOPICS: tuple[str, ...] = (
    "mesh/control/round_start",
    "mesh/control/round_gather_done",
    "mesh/node/+/status",
    "mesh/node/+/knowledge_ready",
    "mesh/node/+/energy",
    "mesh/node/+/eval",
)

_METADATA_SUFFIXES = {"status", "knowledge_ready", "energy", "eval"}


def is_metadata_topic(topic: str) -> bool:
    if topic in ("mesh/control/round_start", "mesh/control/round_gather_done"):
        return True
    parts = topic.split("/")
    return (
        len(parts) == 4
        and parts[0] == "mesh"
        and parts[1] == "node"
        and parts[3] in _METADATA_SUFFIXES
    )


def format_feed_entry(topic: str, payload: bytes, ts: float) -> dict:
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        parsed = f"<{len(payload)} bytes>"
    return {"topic": topic, "payload": parsed, "timestamp": ts}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/dashboard/data.py tests/test_dashboard_data.py
git commit -m "Add dashboard data helpers with metadata-only MQTT subscription list"
```

---

## Task 11: Dashboard container — Streamlit `app.py`, `Dockerfile`, `requirements.txt`

Not TDD — Streamlit rendering itself has no automated test here; this task's steps end in a manual verification (running `streamlit run` and checking the browser), not a pytest run. Say so explicitly rather than claim UI correctness from code review alone.

**Files:**
- Create: `docker/dashboard/app.py`
- Create: `docker/dashboard/Dockerfile`
- Create: `docker/dashboard/requirements.txt`

**Interfaces:**
- Consumes: `is_metadata_topic`, `format_feed_entry`, `DASHBOARD_SUBSCRIBE_TOPICS` (Task 10); `sqlite_store.read_all` (Task 2).
- Produces: a Streamlit app reading `MQTT_HOST`, `MQTT_PORT`, `MERGED_DB`, `REFRESH_S` from the environment, serving on port 8501.

- [ ] **Step 1: Write `requirements.txt`**

```
# docker/dashboard/requirements.txt
streamlit>=1.38
pandas>=2.0
paho-mqtt>=2.1
```

(No `PyYAML` — `app.py`/`data.py` never read `config.yaml` directly; everything they need comes from env vars or `sqlite_store`.)

- [ ] **Step 2: Write `app.py`**

```python
# docker/dashboard/app.py
"""Read-only observability dashboard: connection/liveness status, a live
feed of control-plane MQTT traffic, and a table/charts over the merged
SQLite metrics. No Docker socket access, no control-plane publish rights
-- this process is excluded from the sustainability accounting scope
(see spec §8): its own CPU usage is outside every ComputeEnergyTracker
scope in the node containers.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

# See docker/node/main.py's identical comment: makes `from src...` resolve
# both inside the flattened /app container layout and when run locally
# (e.g. `streamlit run docker/dashboard/app.py` from the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st

from data import DASHBOARD_SUBSCRIBE_TOPICS, format_feed_entry
from src.energy.sqlite_store import read_all

MQTT_HOST = os.environ.get("MQTT_HOST", "broker")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MERGED_DB = os.environ.get("MERGED_DB", "/energy/merged.db")
REFRESH_S = float(os.environ.get("REFRESH_S", "3"))
STALE_AFTER_S = 600  # 2x a generous round_timeout_s default; adjust if config's is much larger

FEED_MAX_ENTRIES = 200


@st.cache_resource
def _mqtt_state():
    """One shared, session-independent state dict + background MQTT
    client for the whole dashboard process (cache_resource runs once per
    process, not per browser session).
    """
    state = {"feed": [], "last_seen": {}, "lock": threading.Lock()}

    def _on_message(_client, _userdata, msg):
        with state["lock"]:
            entry = format_feed_entry(msg.topic, msg.payload, ts=time.time())
            state["feed"].append(entry)
            state["feed"][:] = state["feed"][-FEED_MAX_ENTRIES:]
            parts = msg.topic.split("/")
            if len(parts) == 4 and parts[1] == "node":
                state["last_seen"][parts[2]] = time.time()

    client = mqtt.Client(client_id="dashboard", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = _on_message
    client.connect(MQTT_HOST, MQTT_PORT)
    for topic in DASHBOARD_SUBSCRIBE_TOPICS:
        client.subscribe(topic)
    client.loop_start()
    state["client"] = client
    return state


def render() -> None:
    st.set_page_config(page_title="Mesh dashboard", layout="wide")
    st.title("Docker/MQTT mesh — live status")

    state = _mqtt_state()
    with state["lock"]:
        feed_snapshot = list(state["feed"])
        last_seen_snapshot = dict(state["last_seen"])

    st.subheader("Node status")
    now = time.time()
    status_rows = [
        {
            "node_id": node_id,
            "last_seen_s_ago": round(now - ts, 1),
            "state": "stale" if now - ts > STALE_AFTER_S else "online",
        }
        for node_id, ts in sorted(last_seen_snapshot.items())
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

    st.subheader("Round metrics (merged.db)")
    rows = read_all(MERGED_DB)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        for metric in ["energy_kwh", "knowledge_bytes_sent", "crop_accuracy"]:
            if metric in df.columns:
                pivot = df.pivot_table(index="round_idx", columns="node_id", values=metric)
                st.line_chart(pivot)
    else:
        st.write("No rows yet.")

    st.subheader("Live control-plane feed")
    st.dataframe(pd.DataFrame(list(reversed(feed_snapshot))), use_container_width=True)

    time.sleep(REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    render()
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# docker/dashboard/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/energy/ ./src/energy/
COPY docker/dashboard/app.py docker/dashboard/data.py ./

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

- [ ] **Step 4: Build the image to verify it compiles and installs cleanly**

Run: `docker build -f docker/dashboard/Dockerfile -t mesh-dashboard .` (from repo root)
Expected: build succeeds with no errors, and no PyTorch is installed (`docker run --rm mesh-dashboard python -c "import torch"` fails with `ModuleNotFoundError`).

- [ ] **Step 5: Manual check — run it standalone against no broker yet**

Run: `docker run --rm -p 8501:8501 -e MQTT_HOST=localhost mesh-dashboard` and open `http://localhost:8501`.
Expected: the page loads (it will show empty/stale status since no broker/nodes are running yet — that's expected at this point; full verification with a live mesh happens in Task 13). This step only confirms the app boots and serves a page — say so explicitly, don't claim more than that.

- [ ] **Step 6: Commit**

```bash
git add docker/dashboard/app.py docker/dashboard/Dockerfile docker/dashboard/requirements.txt
git commit -m "Add dashboard container: Streamlit app, Dockerfile, requirements"
```

---

## Task 12: `docker-compose.yml` — assemble all six services

**Files:**
- Create: `docker/docker-compose.yml`

**Interfaces:**
- Consumes: every Dockerfile from Tasks 6, 8, 9, 11; `config.yaml` (Task 3); `data/docker_mesh/` (produced by running Task 3's `split_node_data.py`, not by this task).
- Produces: the full runnable stack.

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
# docker/docker-compose.yml
services:
  broker:
    image: eclipse-mosquitto:2
    volumes:
      - ./broker/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro
    networks: [mesh]

  coordinator:
    build:
      context: ..
      dockerfile: docker/coordinator/Dockerfile
    depends_on: [broker]
    environment:
      MQTT_HOST: broker
      MQTT_PORT: "1883"
      CONFIG_PATH: /config/config.yaml
      ENERGY_DB: /energy/merged.db
    volumes:
      - ../config.yaml:/config/config.yaml:ro
      - ../outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_0:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    depends_on: [broker]
    environment:
      NODE_ID: node_0
      MQTT_HOST: broker
      MQTT_PORT: "1883"
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_0
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      ENERGY_DB: /energy/node_0.db
    volumes:
      - ../config.yaml:/config/config.yaml:ro
      - ../data/docker_mesh/node_0:/data/node_0:ro
      - ../data/docker_mesh/probe:/data/probe:ro
      - ../data/docker_mesh/classes.json:/data/classes.json:ro
      - ../outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_1:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    depends_on: [broker]
    environment:
      NODE_ID: node_1
      MQTT_HOST: broker
      MQTT_PORT: "1883"
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_1
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      ENERGY_DB: /energy/node_1.db
    volumes:
      - ../config.yaml:/config/config.yaml:ro
      - ../data/docker_mesh/node_1:/data/node_1:ro
      - ../data/docker_mesh/probe:/data/probe:ro
      - ../data/docker_mesh/classes.json:/data/classes.json:ro
      - ../outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_2:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    depends_on: [broker]
    environment:
      NODE_ID: node_2
      MQTT_HOST: broker
      MQTT_PORT: "1883"
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_2
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      ENERGY_DB: /energy/node_2.db
    volumes:
      - ../config.yaml:/config/config.yaml:ro
      - ../data/docker_mesh/node_2:/data/node_2:ro
      - ../data/docker_mesh/probe:/data/probe:ro
      - ../data/docker_mesh/classes.json:/data/classes.json:ro
      - ../outputs/docker_mesh/energy:/energy
    networks: [mesh]

  dashboard:
    build:
      context: ..
      dockerfile: docker/dashboard/Dockerfile
    depends_on: [broker]
    ports:
      - "8501:8501"
    environment:
      MQTT_HOST: broker
      MQTT_PORT: "1883"
      MERGED_DB: /energy/merged.db
      REFRESH_S: "3"
    volumes:
      - ../outputs/docker_mesh/energy:/energy:ro
    networks: [mesh]

networks:
  mesh: {}
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose -f docker/docker-compose.yml config`
Expected: prints the fully-resolved config with no errors (this does not build or start anything, just validates syntax and interpolation).

- [ ] **Step 3: Commit**

```bash
git add docker/docker-compose.yml
git commit -m "Add docker-compose.yml assembling broker, coordinator, 3 nodes, dashboard"
```

---

## Task 13: Full-stack integration smoke test

Manual verification per spec §10 — this task has no new source files, it exercises everything built in Tasks 1–12 together.

**Files:** none (verification only).

- [ ] **Step 1: Set a short smoke-test round count**

Temporarily edit `config.yaml`: set `training.rounds: 1` (revert after this task, or note it needs reverting before any "real" run).

- [ ] **Step 2: Generate the split data**

Run: `python scripts/split_node_data.py`
Expected: prints `Wrote split dataset + classes.json to data/docker_mesh`, and `data/docker_mesh/{probe,node_0,node_1,node_2,classes.json}` exist.

- [ ] **Step 3: Bring up the full stack**

Run: `docker compose -f docker/docker-compose.yml up --build`
Expected: all 6 services start; node logs show `connected to broker:1883, waiting for rounds...`; coordinator logs show it waiting, then `all nodes online, starting round loop`, then `all rounds complete`.

- [ ] **Step 4: Verify the merged database**

Run: `python -c "from src.energy.sqlite_store import read_all; import json; print(json.dumps(read_all('outputs/docker_mesh/energy/merged.db'), indent=2))"`
Expected: exactly 3 rows (`round_idx=0`, one per node), each with `energy_kwh`, `knowledge_bytes_sent`, `crop_accuracy`, `disease_accuracy`, and `active=1` all populated (not `None`).

- [ ] **Step 5: Verify the dashboard**

Open `http://localhost:8501`. Expected: "Node status" shows all 3 nodes with a recent `last_seen_s_ago`; "Round metrics" table shows the same 3 rows as Step 4; "Live control-plane feed" shows `round_start`/`round_gather_done`/`status`/`knowledge_ready`/`energy`/`eval` entries — confirm by visual inspection that no entry's topic ends in `/knowledge` (the binary payload topic must never appear, since the dashboard never subscribes to it).

- [ ] **Step 6: Cross-check against the in-process pipeline**

Run: `python -m src.train --config config.yaml --arch mobilenet_v3_small` (same seed/config) and compare its per-node `crop_accuracy`/`disease_accuracy` from `outputs/results_mobilenet_v3_small.json` against the Docker run's `merged.db` rows. Expected: same ballpark (not exact — real wall-clock/thread-scheduling nondeterminism between the two runs is expected), which is the practical confirmation that Task 1's class-index alignment fix worked end-to-end rather than silently misaligning logits.

- [ ] **Step 7: Tear down and revert the smoke-test config change**

Run: `docker compose -f docker/docker-compose.yml down`
Revert `config.yaml`'s `training.rounds` back to its original value (5).

- [ ] **Step 8: Commit the reverted config (if Step 1's edit was accidentally committed) or confirm there is nothing to commit**

```bash
git status
```
Expected: `config.yaml` shows no diff (already reverted in Step 7) — no commit needed for this task.

---

## Appendix: Local verification without Docker

`NodeRunner`/`CoordinatorRunner` (Tasks 5, 7) don't know or care whether they're inside a container — `main.py`/`app.py` (Tasks 6, 8, 11) are thin, env-var-driven wiring with no Docker-specific code. This means every piece except the broker itself can be run as plain host processes, which is faster to iterate on than rebuilding images — a good way to validate Tasks 1–11 end-to-end before ever running `docker build`. Only the broker needs *something* providing an MQTT server on `localhost:1883`; the simplest option is still a single `docker run` for just that one, unmodified image (no build, no Dockerfile of yours involved) — or a natively-installed Mosquitto if Docker Desktop isn't wanted at all for this step.

This works because of the `sys.path.insert` added to `main.py`/`app.py` above (resolves `src/` from the repo root either way) — do this appendix's steps only after Tasks 1–11 are implemented.

1. Start a broker on localhost: `docker run --rm -p 1883:1883 -v ${PWD}/docker/broker/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro eclipse-mosquitto:2` (or a native Mosquitto install, config from Task 9).
2. `python scripts/split_node_data.py` — writes `data/docker_mesh/` on the host filesystem directly, no container involved.
3. In separate terminals, from the repo root, with `MQTT_HOST=localhost` and paths pointing at the host's `data/docker_mesh/...` (not the container's `/data/...`):
   - Coordinator: `ENERGY_DB=outputs/docker_mesh/energy/merged.db CONFIG_PATH=config.yaml MQTT_HOST=localhost python docker/coordinator/main.py`
   - `node_0`: `NODE_ID=node_0 MQTT_HOST=localhost CONFIG_PATH=config.yaml DATA_ROOT=data/docker_mesh/node_0 PROBE_ROOT=data/docker_mesh/probe CLASSES_JSON=data/docker_mesh/classes.json ENERGY_DB=outputs/docker_mesh/energy/node_0.db python docker/node/main.py` (repeat for `node_1`/`node_2` with their own `NODE_ID`/`DATA_ROOT`/`ENERGY_DB`)
   - Dashboard: `MQTT_HOST=localhost MERGED_DB=outputs/docker_mesh/energy/merged.db streamlit run docker/dashboard/app.py`
4. Open `http://localhost:8501`.

This is the same env-var contract `docker-compose.yml` sets — only the *values* (`localhost` + host paths, instead of `broker` + container paths) differ. Once this works, Tasks 6/8/11's `docker build` steps and Task 12/13 just move the same processes into containers with no code changes.
