# Docker/HTTP Multi-Container Mesh + Observability Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing single-process mesh (`Node`/`MeshSimulator` training + distillation logic) across 3 isolated Docker containers exchanging knowledge over direct HTTP request/response instead of an in-memory dict, with energy/eval metrics persisted to SQLite (per-node + merged), plus a read-only Streamlit dashboard.

**Architecture:** 5 Docker Compose services — `coordinator` (times rounds via concurrent HTTP fan-out, tracks liveness, merges metrics — never touches knowledge content), `node_0`/`node_1`/`node_2` (one shared image, each with its own physically-isolated data folder, running a small FastAPI server), `dashboard` (Streamlit viewer, polls nodes' `/health` + coordinator's `/events`). No broker. The existing in-process simulation (`src/federated/mesh.py`, `src/train.py`) is untouched and stays as a separate, working fallback pipeline.

**Tech Stack:** Python 3.11, PyTorch/timm (nodes only), FastAPI + uvicorn (nodes + coordinator's `/events` server), `httpx` (async HTTP client, used for the coordinator's fan-out and each node's peer-knowledge fetch), `sqlite3` (stdlib), Streamlit + pandas (dashboard only).

**Spec:** `docs/superpowers/specs/2026-08-16-docker-mqtt-mesh-design.md` (filename predates the MQTT→HTTP revision; content is current)

## Global Constraints

- `src/federated/mesh.py`, `src/federated/node.py`, `src/federated/aggregation.py`, `src/models/factory.py`, `src/train.py` are **not modified** by this plan — the in-process pipeline must keep working exactly as it does today.
- The coordinator **never calls a node's `/knowledge/{round_idx}` endpoint** — control-plane only (`/health`, `/round/start`, `/round/gather`), per spec §5. Peer knowledge is fetched node-to-node, never through the coordinator.
- The dashboard only ever issues `GET` requests (`/health` on nodes, `/events` on the coordinator) — it never calls `/round/start` or `/round/gather`, so it cannot trigger a run, per spec §8.
- Node containers publish no host port; only the dashboard (`8501`) and the coordinator's `/events` server (`9000`) do — per spec §11.
- `size_bytes`/`knowledge_bytes_sent` must be the actual serialized byte length, never `KnowledgePayload.size_bytes()`'s estimate — per spec §6.
- **Concurrency is load-bearing, not incidental**: the coordinator's fan-out to nodes, and each node's fetch of its peers, must be concurrent (`asyncio.gather` over `httpx.AsyncClient`). Sequential calls would silently turn parallel training into serial training — a functional regression with no error message, only much slower rounds.
- Node FastAPI routes that call into `NodeRunner` (`/round/start`, `/round/gather`) must be defined as plain `def`, never `async def` — `Node.local_train`/`distill` are blocking synchronous PyTorch calls; FastAPI runs `def` routes in a worker thread automatically, keeping `/health` responsive during training. An `async def` route calling this code directly would freeze the whole server for the duration of training.
- This repo's layout diverges from spec §3 in one place: app code lives flat under `docker/node/`, `docker/coordinator/`, `docker/dashboard/` (no nested subfolder) to keep Dockerfiles simple and avoid any `docker.*` dotted-package naming question.
- Deferred, do not build in this plan: disruption scenarios over this transport, the K210 node, a dashboard "trigger new run" control — per spec §2/§12.

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
- Produces: `init_db(path: str | Path) -> None`; `upsert_row(path: str | Path, node_id: str, round_idx: int, recorded_at: str, **fields) -> None`; `read_all(path: str | Path) -> list[dict]`. Used by Task 5 (node runner), Task 7 (coordinator runner), Task 9 (dashboard).

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
    # both writes' columns survive -- the second upsert must not null out the first's
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
bytes, accuracy). Written incrementally: a node's /round/start response
and its /round/gather response arrive at different times, so `upsert_row`
merges whichever columns are passed into one logical row rather than
requiring the whole row at once.
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
    conflict, so a node's separate /round/start-sourced and
    /round/gather-sourced writes for the same (node_id, round_idx)
    accumulate into one row instead of clobbering each other.
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

## Task 4: Knowledge codec

**Files:**
- Create: `docker/node/knowledge_codec.py`
- Test: `tests/test_knowledge_codec.py`

**Interfaces:**
- Consumes: `KnowledgePayload` (existing, `src/federated/node.py`, unchanged).
- Produces: `encode_knowledge(round_idx: int, payload: KnowledgePayload) -> bytes`; `decode_knowledge(data: bytes) -> tuple[int, KnowledgePayload]`. `encode_knowledge`'s return value is the exact HTTP response body for `GET /knowledge/{round_idx}` (Task 6). Used by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knowledge_codec.py
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
    )
    data = encode_knowledge(7, payload)
    assert isinstance(data, bytes)

    round_idx, decoded = decode_knowledge(data)
    assert round_idx == 7
    assert torch.equal(decoded.crop_logits, payload.crop_logits)
    assert torch.equal(decoded.disease_logits, payload.disease_logits)
    for key in payload.prototypes:
        assert torch.equal(decoded.prototypes[key], payload.prototypes[key])


def test_encoded_payload_is_nonempty_bytes():
    payload = KnowledgePayload(prototypes={}, crop_logits=torch.zeros(2, 2), disease_logits=torch.zeros(2, 2))
    data = encode_knowledge(0, payload)
    assert len(data) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge_codec'`

- [ ] **Step 3: Implement**

```python
# docker/node/knowledge_codec.py
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
    )
    return obj["round_idx"], payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_codec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docker/node/knowledge_codec.py tests/test_knowledge_codec.py
git commit -m "Add knowledge-payload codec with round_idx tag"
```

---

## Task 5: `NodeRunner` core round-handling logic

Round-handling logic is separated from the real HTTP server and client so it's unit-testable without a running FastAPI app: `NodeRunner.fetch_all_knowledge` is an injected callable (`(peer_ids, peer_bases, round_idx) -> {peer_id: bytes | None}`), fully fake-able in tests. The real implementation (Task 6) uses `asyncio.gather` + `httpx.AsyncClient`, but `NodeRunner` itself never knows or cares.

**Files:**
- Create: `docker/node/node_runner.py`
- Test: `tests/test_node_runner.py`

**Interfaces:**
- Consumes: `Node`, `KnowledgePayload` (`src/federated/node.py`); `aggregate_prototypes`, `aggregate_logits` (`src/federated/aggregation.py`); `ComputeEnergyTracker` (`src/energy/tracker.py`); `sqlite_store.upsert_row` (Task 2); `encode_knowledge`, `decode_knowledge` (Task 4).
- Produces: `NodeRunner` dataclass with `handle_round_start(round_idx: int) -> dict`, `get_knowledge_bytes(round_idx: int) -> bytes | None`, `handle_round_gather(round_idx: int, active_nodes: list[str], peer_bases: dict[str, str]) -> dict`. Used by Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_node_runner.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "node"))

import torch
from torch.utils.data import DataLoader

from knowledge_codec import encode_knowledge  # noqa: E402
from node_runner import NodeRunner  # noqa: E402
from src.data.plantvillage import make_subset, train_test_split_indices
from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.node import KnowledgePayload, Node
from src.models.factory import build_model


def _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=None):
    train_idx, test_idx = train_test_split_indices(list(range(len(synthetic_dataset))), 0.3, seed=1)
    train_loader = DataLoader(make_subset(synthetic_dataset, train_idx), batch_size=4, shuffle=True)
    test_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    probe_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    model = build_model("mobilenet_v3_small", 2, 2, pretrained=False)
    node = Node("node_0", model, train_loader, test_loader, device="cpu")
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    runner = NodeRunner(
        node_id="node_0",
        node=node,
        probe_loader=probe_loader,
        tracker=tracker,
        db_path=str(tmp_path / "node_0.db"),
        fetch_all_knowledge=fetch_all_knowledge or (lambda peer_ids, peer_bases, round_idx: {}),
    )
    return runner, probe_loader


def test_handle_round_start_returns_response_body_and_writes_db(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_start(0)

    assert response["round_idx"] == 0
    assert response["size_bytes"] > 0
    assert response["energy_kwh"] is not None

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert len(rows) == 1
    assert rows[0]["knowledge_bytes_sent"] == response["size_bytes"]


def test_get_knowledge_bytes_returns_none_for_a_different_round(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.handle_round_start(0)
    assert runner.get_knowledge_bytes(0) is not None
    assert runner.get_knowledge_bytes(1) is None


def test_handle_round_gather_ignores_peer_data_for_a_different_round(tmp_path, synthetic_dataset):
    n_probe_holder = {}

    def fake_fetch_all(peer_ids, peer_bases, round_idx):
        n_probe = n_probe_holder["n"]
        stale_payload = KnowledgePayload(
            prototypes={}, crop_logits=torch.zeros(n_probe, 2), disease_logits=torch.zeros(n_probe, 2)
        )
        # encoded for round 99, but we are gathering round 0 -- must be dropped
        return {"node_1": encode_knowledge(99, stale_payload)}

    runner, probe_loader = _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=fake_fetch_all)
    n_probe_holder["n"] = len(probe_loader.dataset)

    response = runner.handle_round_gather(0, ["node_0", "node_1"], {"node_1": "http://node_1:8000"})
    assert "crop_accuracy" in response
    assert "disease_accuracy" in response

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["active"] == 1


def test_handle_round_gather_skips_a_peer_whose_fetch_failed(tmp_path, synthetic_dataset):
    def fake_fetch_all(peer_ids, peer_bases, round_idx):
        return {"node_1": None}  # peer timed out / errored

    runner, _ = _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=fake_fetch_all)
    response = runner.handle_round_gather(0, ["node_0", "node_1"], {"node_1": "http://node_1:8000"})
    assert "crop_accuracy" in response


def test_handle_round_gather_with_no_peers_still_evaluates(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_gather(0, ["node_0"], {})
    assert "crop_accuracy" in response
```

(`synthetic_dataset` is the existing fixture in `tests/conftest.py` — no changes needed there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_node_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'node_runner'`

- [ ] **Step 3: Implement**

```python
# docker/node/node_runner.py
"""HTTP-driven wrapper around a single Node, reusing src/federated/node.py
and src/federated/aggregation.py unmodified. Round-handling logic takes its
peer-fetch mechanism as an injected callable so it's unit-testable with a
fake fetcher -- no real HTTP server, no real concurrency, needed in tests.
See docker/node/main.py for the real FastAPI + httpx wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from src.energy import sqlite_store
from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import KnowledgePayload, Node

from knowledge_codec import decode_knowledge, encode_knowledge


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# fetch_all_knowledge(peer_ids, peer_bases, round_idx) -> {peer_id: bytes | None}
# None means that peer's fetch failed, timed out, or (checked by the caller)
# returned a mismatched round.
FetchAllKnowledge = Callable[[list, dict, int], dict]


@dataclass
class NodeRunner:
    node_id: str
    node: Node
    probe_loader: object
    tracker: ComputeEnergyTracker
    db_path: str
    fetch_all_knowledge: FetchAllKnowledge
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
    _last_round_idx: int | None = field(default=None, init=False)
    _last_knowledge_bytes: bytes | None = field(default=None, init=False)

    def handle_round_start(self, round_idx: int) -> dict:
        with self.tracker.track(f"{self.node_id}_round_{round_idx}") as energy_record:
            self.node.local_train(self.local_epochs, self.lr)
        knowledge = self.node.compute_knowledge(self.probe_loader)
        data = encode_knowledge(round_idx, knowledge)
        self._last_round_idx = round_idx
        self._last_knowledge_bytes = data

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
        return {
            "round_idx": round_idx,
            "size_bytes": len(data),
            "energy_kwh": energy_record["energy_kwh"],
            "duration_s": energy_record["duration_s"],
            "energy_method": energy_record["method"],
        }

    def get_knowledge_bytes(self, round_idx: int) -> bytes | None:
        if self._last_round_idx != round_idx:
            return None
        return self._last_knowledge_bytes

    def handle_round_gather(self, round_idx: int, active_nodes: list, peer_bases: dict) -> dict:
        peer_ids = [n for n in active_nodes if n != self.node_id]
        fetched = self.fetch_all_knowledge(peer_ids, peer_bases, round_idx)

        peers: list[KnowledgePayload] = []
        for peer_id in peer_ids:
            data = fetched.get(peer_id)
            if data is None:
                continue
            peer_round_idx, knowledge = decode_knowledge(data)
            if peer_round_idx != round_idx:
                continue  # defensive: peer returned data for a different round
            peers.append(knowledge)

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
        sqlite_store.upsert_row(
            self.db_path,
            self.node_id,
            round_idx,
            _now_iso(),
            crop_accuracy=eval_result["crop_accuracy"],
            disease_accuracy=eval_result["disease_accuracy"],
            active=1,
        )
        return {"round_idx": round_idx, **eval_result}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_node_runner.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/node/node_runner.py tests/test_node_runner.py
git commit -m "Add NodeRunner: HTTP-driven round handling over the existing Node/aggregation logic"
```

---

## Task 6: Node container — FastAPI `main.py`, `Dockerfile`, `requirements.txt`

This task wires `NodeRunner` to a real FastAPI server and a real `httpx.AsyncClient` for peer fetches. It is not TDD — `NodeRunner`'s logic is already covered by Task 5; this is thin, mostly-untestable-without-real-network glue, verified manually in Task 12's integration smoke test.

**Files:**
- Create: `docker/node/main.py`
- Create: `docker/node/Dockerfile`
- Create: `docker/node/requirements.txt`

**Interfaces:**
- Consumes: `NodeRunner` (Task 5); `Config` (`src/config.py`); `GlobalLabelMap`/`load_global_label_map`/`PlantVillageDataset`/`make_subset`/`train_test_split_indices` (`src/data/plantvillage.py`, Task 1); `build_model` (`src/models/factory.py`); `Node` (`src/federated/node.py`); `ComputeEnergyTracker` (`src/energy/tracker.py`).
- Produces: a FastAPI app on port `8000` with `GET /health`, `POST /round/start`, `GET /knowledge/{round_idx}`, `POST /round/gather`, reading `NODE_ID`, `DATA_ROOT`, `PROBE_ROOT`, `CLASSES_JSON`, `ENERGY_DB`, `CONFIG_PATH` from the environment (all set by `docker-compose.yml` in Task 11).

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
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
```

- [ ] **Step 2: Write `main.py`**

```python
# docker/node/main.py
"""FastAPI server for one mesh node. /round/start and /round/gather are
plain `def` routes (not `async def`) -- Node.local_train/distill are
blocking, synchronous PyTorch calls with no await points; FastAPI runs
`def` routes in a worker thread automatically, keeping /health responsive
while training runs. An `async def` route calling this code directly
would freeze the whole server's event loop for the duration of training.

Peer-knowledge fetches (needed inside /round/gather) use httpx.AsyncClient
+ asyncio.gather for concurrency, invoked via asyncio.run() from the
synchronous handler.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# In the container, /app has src/ copied alongside this file, so this
# insert is a harmless no-op there. Running locally (e.g. for the
# no-Docker verification workflow) this file's own directory does NOT
# contain src/ -- this makes `from src...` resolve in both cases.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
from fastapi import FastAPI, HTTPException, Response
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


async def _fetch_one(client: httpx.AsyncClient, peer_id: str, base_url: str, round_idx: int):
    try:
        resp = await client.get(f"{base_url}/knowledge/{round_idx}", timeout=30.0)
        if resp.status_code != 200:
            return peer_id, None
        return peer_id, resp.content
    except httpx.HTTPError:
        return peer_id, None


async def _fetch_all_knowledge_async(peer_ids: list, peer_bases: dict, round_idx: int) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_fetch_one(client, peer_id, peer_bases[peer_id], round_idx) for peer_id in peer_ids)
        )
    return dict(results)


def fetch_all_knowledge(peer_ids: list, peer_bases: dict, round_idx: int) -> dict:
    if not peer_ids:
        return {}
    return asyncio.run(_fetch_all_knowledge_async(peer_ids, peer_bases, round_idx))


def build_runner() -> NodeRunner:
    node_id = os.environ["NODE_ID"]
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

    return NodeRunner(
        node_id=node_id,
        node=node,
        probe_loader=probe_loader,
        tracker=tracker,
        db_path=energy_db,
        fetch_all_knowledge=fetch_all_knowledge,
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


runner = build_runner()
app = FastAPI()


@app.get("/health")
def health():
    return {"node_id": runner.node_id, "status": "online"}


@app.post("/round/start")
def round_start(body: dict):
    return runner.handle_round_start(body["round_idx"])


@app.get("/knowledge/{round_idx}")
def get_knowledge(round_idx: int):
    data = runner.get_knowledge_bytes(round_idx)
    if data is None:
        raise HTTPException(status_code=409, detail="knowledge not ready for this round")
    return Response(content=data, media_type="application/octet-stream")


@app.post("/round/gather")
def round_gather(body: dict):
    return runner.handle_round_gather(body["round_idx"], body["active_nodes"], body["peer_bases"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: Write `Dockerfile`**

```dockerfile
# docker/node/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY docker/node/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/node/main.py docker/node/node_runner.py docker/node/knowledge_codec.py ./

EXPOSE 8000
CMD ["python", "main.py"]
```

- [ ] **Step 4: Build the image to verify it compiles and installs cleanly**

Run: `docker build -f docker/node/Dockerfile -t mesh-node .` (from repo root, so `src/` is in the build context)
Expected: build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add docker/node/main.py docker/node/Dockerfile docker/node/requirements.txt
git commit -m "Add node container: FastAPI main.py wiring, Dockerfile, requirements"
```

---

## Task 7: `CoordinatorRunner` core round-driving logic

Like Task 5, this separates round-driving logic from the real HTTP client so it's unit-testable without a network. `post_all`/`health_check` are injected callables — the real implementation (Task 8) uses `asyncio.gather` + `httpx`, but `CoordinatorRunner` never knows or cares. This is meaningfully simpler than the equivalent MQTT design would have been: there's no separate poll-with-timeout loop to test, since a synchronous `post_all` call either returns a node's result or `None` (timed out/errored) by the time it returns — the timeout itself is the real client's concern (Task 8), not this class's.

**Files:**
- Create: `docker/coordinator/coordinator_runner.py`
- Test: `tests/test_coordinator_runner.py`

**Interfaces:**
- Consumes: `sqlite_store.upsert_row` (Task 2).
- Produces: `CoordinatorRunner` dataclass with `wait_until_all_online(sleep_fn=time.sleep, poll_interval_s=1.0) -> None`, `run_round(round_idx: int) -> list[str]`, `run_all_rounds() -> None`. Used by Task 8.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coordinator_runner.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "coordinator"))

from coordinator_runner import CoordinatorRunner  # noqa: E402
from src.energy import sqlite_store


def _make_runner(tmp_path, post_all, health_check=lambda n: True, num_rounds=1):
    return CoordinatorRunner(
        expected_nodes=["node_0", "node_1"],
        node_base_urls={"node_0": "http://node_0:8000", "node_1": "http://node_1:8000"},
        num_rounds=num_rounds,
        round_timeout_s=30,
        db_path=str(tmp_path / "merged.db"),
        post_all=post_all,
        health_check=health_check,
    )


def test_wait_until_all_online_blocks_until_health_check_passes_for_every_node():
    calls = {"n": 0}

    def health_check(_node_id):
        calls["n"] += 1
        return calls["n"] > 2  # first couple of checks report unhealthy

    runner = CoordinatorRunner(
        expected_nodes=["node_0"],
        node_base_urls={"node_0": "http://node_0:8000"},
        num_rounds=1,
        round_timeout_s=30,
        db_path="unused.db",
        post_all=lambda *a: {},
        health_check=health_check,
    )
    slept = []
    runner.wait_until_all_online(sleep_fn=slept.append, poll_interval_s=0.1)
    assert len(slept) >= 1


def test_run_round_writes_energy_and_eval_rows_from_response_bodies(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                n: {"energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "proxy_wall_power", "size_bytes": 100}
                for n in node_ids
            }
        assert path == "/round/gather"
        assert body["active_nodes"] == ["node_0", "node_1"]
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.6} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    active = runner.run_round(0)

    assert active == ["node_0", "node_1"]
    rows = {r["node_id"]: r for r in sqlite_store.read_all(str(tmp_path / "merged.db"))}
    assert rows["node_0"]["crop_accuracy"] == 0.5
    assert rows["node_1"]["knowledge_bytes_sent"] == 100


def test_run_round_excludes_a_node_that_failed_round_start(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                "node_0": {"energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "x", "size_bytes": 10},
                "node_1": None,  # timed out / errored
            }
        assert set(body["active_nodes"]) == {"node_0"}
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.5} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    active = runner.run_round(0)
    assert active == ["node_0"]


def test_run_all_rounds_calls_run_round_for_every_configured_round(tmp_path):
    seen_rounds = []

    def post_all(node_ids, path, body):
        if path == "/round/start":
            seen_rounds.append(body["round_idx"])
            return {
                n: {"energy_kwh": 0.0, "duration_s": 0.0, "energy_method": "x", "size_bytes": 1} for n in node_ids
            }
        return {n: {"crop_accuracy": 0.0, "disease_accuracy": 0.0} for n in node_ids}

    runner = _make_runner(tmp_path, post_all, num_rounds=3)
    runner.run_all_rounds()
    assert seen_rounds == [0, 1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_coordinator_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coordinator_runner'`

- [ ] **Step 3: Implement**

```python
# docker/coordinator/coordinator_runner.py
"""Control-plane round driver: times rounds, tracks node liveness, merges
per-round HTTP response bodies into SQLite. Never calls a node's
/knowledge endpoint itself -- nodes fetch peer knowledge directly from
each other, so there is no central aggregator of knowledge (that logic
stays entirely in node_runner.py). Transport (concurrent HTTP fan-out,
health polling) is injected as plain callables so this class is
unit-testable with fakes -- see docker/coordinator/main.py for the real
asyncio/httpx wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from src.energy import sqlite_store


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# post_all(node_ids, path, body) -> {node_id: response_json | None}
# None means that node's request failed, timed out, or returned non-200.
PostAll = Callable[[list, str, dict], dict]
HealthCheck = Callable[[str], bool]


@dataclass
class CoordinatorRunner:
    expected_nodes: list
    node_base_urls: dict
    num_rounds: int
    round_timeout_s: float
    db_path: str
    post_all: PostAll
    health_check: HealthCheck

    def wait_until_all_online(self, sleep_fn=time.sleep, poll_interval_s: float = 1.0) -> None:
        while not all(self.health_check(n) for n in self.expected_nodes):
            sleep_fn(poll_interval_s)

    def run_round(self, round_idx: int) -> list:
        start_results = self.post_all(self.expected_nodes, "/round/start", {"round_idx": round_idx})
        active = sorted(n for n, r in start_results.items() if r is not None)
        for node_id in active:
            r = start_results[node_id]
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                round_idx,
                _now_iso(),
                energy_kwh=r["energy_kwh"],
                duration_s=r["duration_s"],
                energy_method=r["energy_method"],
                knowledge_bytes_sent=r["size_bytes"],
                active=1,
            )

        gather_body = {
            "round_idx": round_idx,
            "active_nodes": active,
            "peer_bases": {n: self.node_base_urls[n] for n in active},
        }
        gather_results = self.post_all(active, "/round/gather", gather_body)
        for node_id in active:
            r = gather_results.get(node_id)
            if r is None:
                continue
            sqlite_store.upsert_row(
                self.db_path,
                node_id,
                round_idx,
                _now_iso(),
                crop_accuracy=r["crop_accuracy"],
                disease_accuracy=r["disease_accuracy"],
            )
        return active

    def run_all_rounds(self) -> None:
        for round_idx in range(self.num_rounds):
            self.run_round(round_idx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_coordinator_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/coordinator/coordinator_runner.py tests/test_coordinator_runner.py
git commit -m "Add CoordinatorRunner: HTTP round timing and DB merge, decoupled from transport"
```

---

## Task 8: Coordinator container — `main.py`, `Dockerfile`, `requirements.txt`

Like Task 6, this is thin glue verified manually in Task 12. The coordinator's image deliberately has **no** PyTorch — it only imports `src.config` and `src.energy.sqlite_store`, neither of which pulls in `src.federated`/`src.models`.

**Files:**
- Create: `docker/coordinator/main.py`
- Create: `docker/coordinator/Dockerfile`
- Create: `docker/coordinator/requirements.txt`

**Interfaces:**
- Consumes: `CoordinatorRunner` (Task 7); `Config` (`src/config.py`).
- Produces: a runnable container entry point reading `CONFIG_PATH`, `ENERGY_DB` from the environment; also serves `GET /events` on port `9000` for the dashboard.

- [ ] **Step 1: Write `requirements.txt`**

```
# docker/coordinator/requirements.txt
PyYAML>=6.0
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
```

- [ ] **Step 2: Write `main.py`**

```python
# docker/coordinator/main.py
"""Entry point for the round coordinator. Fan-out to nodes uses
asyncio.gather over httpx.AsyncClient for concurrency -- sequential calls
would force nodes to train one at a time instead of in parallel. Exposes
a tiny /events endpoint purely for the dashboard to poll -- it records
only which path was called and what status came back, never touching
knowledge content.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import uvicorn
from fastapi import FastAPI

from src.config import Config
from coordinator_runner import CoordinatorRunner

events: list = []
MAX_EVENTS = 200


async def _post_one(client: httpx.AsyncClient, node_id: str, url: str, body: dict, timeout: float):
    try:
        resp = await client.post(url, json=body, timeout=timeout)
        events.append({"ts": time.time(), "node_id": node_id, "path": url, "status": resp.status_code})
        events[:] = events[-MAX_EVENTS:]
        if resp.status_code != 200:
            return node_id, None
        return node_id, resp.json()
    except httpx.HTTPError:
        events.append({"ts": time.time(), "node_id": node_id, "path": url, "status": "error"})
        events[:] = events[-MAX_EVENTS:]
        return node_id, None


async def _post_all_async(node_base_urls: dict, node_ids: list, path: str, body: dict, timeout: float) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_post_one(client, n, f"{node_base_urls[n]}{path}", body, timeout) for n in node_ids)
        )
    return dict(results)


def make_post_all(node_base_urls: dict, timeout: float):
    def post_all(node_ids: list, path: str, body: dict) -> dict:
        if not node_ids:
            return {}
        return asyncio.run(_post_all_async(node_base_urls, node_ids, path, body, timeout))

    return post_all


def make_health_check(node_base_urls: dict):
    def health_check(node_id: str) -> bool:
        try:
            resp = httpx.get(f"{node_base_urls[node_id]}/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    return health_check


def main() -> None:
    cfg = Config.load(os.environ.get("CONFIG_PATH"))
    num_nodes = cfg.get("data.num_nodes", 3)
    expected_nodes = [f"node_{i}" for i in range(num_nodes)]
    node_base_urls = {n: f"http://{n}:8000" for n in expected_nodes}
    round_timeout_s = cfg.get("docker_mesh.round_timeout_s", 300)
    energy_db = os.environ["ENERGY_DB"]

    runner = CoordinatorRunner(
        expected_nodes=expected_nodes,
        node_base_urls=node_base_urls,
        num_rounds=cfg.get("training.rounds", 5),
        round_timeout_s=round_timeout_s,
        db_path=energy_db,
        post_all=make_post_all(node_base_urls, round_timeout_s),
        health_check=make_health_check(node_base_urls),
    )

    events_app = FastAPI()

    @events_app.get("/events")
    def get_events():
        return events[-MAX_EVENTS:]

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(events_app, host="0.0.0.0", port=9000), daemon=True
    )
    server_thread.start()

    print(f"[coordinator] waiting for {expected_nodes} to come online...")
    runner.wait_until_all_online()
    print("[coordinator] all nodes online, starting round loop")
    runner.run_all_rounds()
    print("[coordinator] all rounds complete")

    threading.Event().wait()  # idle -- no re-run trigger yet, see spec §12


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

COPY src/__init__.py ./src/__init__.py
COPY src/config.py ./src/config.py
COPY src/energy/ ./src/energy/
COPY docker/coordinator/main.py docker/coordinator/coordinator_runner.py ./

EXPOSE 9000
CMD ["python", "main.py"]
```

- [ ] **Step 4: Build the image to verify it compiles, installs cleanly, and stays lean**

Run: `docker build -f docker/coordinator/Dockerfile -t mesh-coordinator .` (from repo root)
Expected: build succeeds; `docker run --rm mesh-coordinator python -c "import torch"` fails with `ModuleNotFoundError` (confirms the image really has no PyTorch).

- [ ] **Step 5: Commit**

```bash
git add docker/coordinator/main.py docker/coordinator/Dockerfile docker/coordinator/requirements.txt
git commit -m "Add coordinator container: asyncio/httpx fan-out, /events endpoint, Dockerfile"
```

---

## Task 9: Dashboard data helpers

**Files:**
- Create: `docker/dashboard/data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: nothing (pure function, no I/O).
- Produces: `build_status_rows(node_health: dict[str, bool]) -> list[dict]`. Used by Task 10.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_data.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "dashboard"))

from data import build_status_rows  # noqa: E402


def test_build_status_rows_sorts_by_node_id():
    rows = build_status_rows({"node_2": True, "node_0": False, "node_1": True})
    assert rows == [
        {"node_id": "node_0", "online": False},
        {"node_id": "node_1", "online": True},
        {"node_id": "node_2", "online": True},
    ]


def test_build_status_rows_handles_empty_input():
    assert build_status_rows({}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Implement**

```python
# docker/dashboard/data.py
"""Pure helpers for the dashboard -- deliberately separated from Streamlit
rendering (docker/dashboard/app.py) so this logic is unit-testable without
a running Streamlit session or real HTTP calls.
"""

from __future__ import annotations


def build_status_rows(node_health: dict) -> list:
    """node_health: {node_id: bool} -> sorted list of {"node_id", "online"}
    rows, so table row order is a guarantee rather than an accident of
    dict iteration order.
    """
    return [{"node_id": node_id, "online": online} for node_id, online in sorted(node_health.items())]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/dashboard/data.py tests/test_dashboard_data.py
git commit -m "Add dashboard status-table helper"
```

---

## Task 10: Dashboard container — Streamlit `app.py`, `Dockerfile`, `requirements.txt`

Not TDD — Streamlit rendering itself has no automated test here; this task's steps end in a manual verification (running `streamlit run` and checking the browser), not a pytest run. Say so explicitly rather than claim UI correctness from code review alone.

**Files:**
- Create: `docker/dashboard/app.py`
- Create: `docker/dashboard/Dockerfile`
- Create: `docker/dashboard/requirements.txt`

**Interfaces:**
- Consumes: `build_status_rows` (Task 9); `sqlite_store.read_all` (Task 2).
- Produces: a Streamlit app reading `NUM_NODES`, `COORDINATOR_EVENTS_URL`, `MERGED_DB`, `REFRESH_S` from the environment, serving on port 8501.

- [ ] **Step 1: Write `requirements.txt`**

```
# docker/dashboard/requirements.txt
streamlit>=1.38
pandas>=2.0
httpx>=0.27
```

(No `PyYAML` — `app.py`/`data.py` never read `config.yaml` directly; everything they need comes from env vars or `sqlite_store`.)

- [ ] **Step 2: Write `app.py`**

```python
# docker/dashboard/app.py
"""Read-only observability dashboard: polls each node's /health directly,
polls the coordinator's /events for a live request log, and reads the
merged SQLite metrics. No Docker socket access, and it only ever issues
GET requests -- it cannot trigger a run. Excluded from the sustainability
accounting scope (see spec §8): its own CPU usage is outside every
ComputeEnergyTracker scope in the node containers.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import pandas as pd
import streamlit as st

from data import build_status_rows
from src.energy.sqlite_store import read_all

NUM_NODES = int(os.environ.get("NUM_NODES", "3"))
NODE_BASE_URLS = {f"node_{i}": f"http://node_{i}:8000" for i in range(NUM_NODES)}
COORDINATOR_EVENTS_URL = os.environ.get("COORDINATOR_EVENTS_URL", "http://coordinator:9000/events")
MERGED_DB = os.environ.get("MERGED_DB", "/energy/merged.db")
REFRESH_S = float(os.environ.get("REFRESH_S", "3"))


def _poll_health() -> dict:
    health = {}
    for node_id, base_url in NODE_BASE_URLS.items():
        try:
            resp = httpx.get(f"{base_url}/health", timeout=3.0)
            health[node_id] = resp.status_code == 200
        except httpx.HTTPError:
            health[node_id] = False
    return health


def _poll_events() -> list:
    try:
        resp = httpx.get(COORDINATOR_EVENTS_URL, timeout=3.0)
        return resp.json() if resp.status_code == 200 else []
    except httpx.HTTPError:
        return []


def render() -> None:
    st.set_page_config(page_title="Mesh dashboard", layout="wide")
    st.title("Docker/HTTP mesh — live status")

    st.subheader("Node status")
    st.dataframe(pd.DataFrame(build_status_rows(_poll_health())), use_container_width=True)

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

    st.subheader("Coordinator event log")
    st.dataframe(pd.DataFrame(list(reversed(_poll_events()))), use_container_width=True)

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
Expected: build succeeds; `docker run --rm mesh-dashboard python -c "import torch"` fails with `ModuleNotFoundError`.

- [ ] **Step 5: Manual check — run it standalone before any nodes/coordinator exist**

Run: `docker run --rm -p 8501:8501 mesh-dashboard` and open `http://localhost:8501`.
Expected: the page loads showing all nodes unreachable and no rows yet — that's expected at this point; full verification with a live mesh happens in Task 12. This step only confirms the app boots and serves a page — say so explicitly, don't claim more than that.

- [ ] **Step 6: Commit**

```bash
git add docker/dashboard/app.py docker/dashboard/Dockerfile docker/dashboard/requirements.txt
git commit -m "Add dashboard container: Streamlit app polling nodes and coordinator over HTTP"
```

---

## Task 11: `docker-compose.yml` — assemble all five services

**Files:**
- Create: `docker/docker-compose.yml`

**Interfaces:**
- Consumes: every Dockerfile from Tasks 6, 8, 10; `config.yaml` (Task 3); `data/docker_mesh/` (produced by running Task 3's `split_node_data.py`, not by this task).
- Produces: the full runnable stack.

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
# docker/docker-compose.yml
services:
  coordinator:
    build:
      context: ..
      dockerfile: docker/coordinator/Dockerfile
    depends_on: [node_0, node_1, node_2]
    ports:
      - "9000:9000"
    environment:
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
    environment:
      NODE_ID: node_0
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
    environment:
      NODE_ID: node_1
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
    environment:
      NODE_ID: node_2
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
    depends_on: [coordinator, node_0, node_1, node_2]
    ports:
      - "8501:8501"
    environment:
      NUM_NODES: "3"
      COORDINATOR_EVENTS_URL: http://coordinator:9000/events
      MERGED_DB: /energy/merged.db
      REFRESH_S: "3"
    volumes:
      - ../outputs/docker_mesh/energy:/energy:ro
    networks: [mesh]

networks:
  mesh: {}
```

Node services don't strictly need to wait for anything (they serve `/health` immediately on startup), and the coordinator's own `wait_until_all_online` already tolerates nodes not being up yet — `depends_on` here is just for a tidier startup order in `docker compose up` logs, not a correctness requirement.

- [ ] **Step 2: Validate the compose file**

Run: `docker compose -f docker/docker-compose.yml config`
Expected: prints the fully-resolved config with no errors (this does not build or start anything, just validates syntax and interpolation).

- [ ] **Step 3: Commit**

```bash
git add docker/docker-compose.yml
git commit -m "Add docker-compose.yml assembling coordinator, 3 nodes, dashboard (no broker)"
```

---

## Task 12: Full-stack integration smoke test

Manual verification per spec §10 — this task has no new source files, it exercises everything built in Tasks 1–11 together.

**Files:** none (verification only).

- [ ] **Step 1: Set a short smoke-test round count**

Temporarily edit `config.yaml`: set `training.rounds: 1` (revert after this task, or note it needs reverting before any "real" run).

- [ ] **Step 2: Generate the split data**

Run: `python scripts/split_node_data.py`
Expected: prints `Wrote split dataset + classes.json to data/docker_mesh`, and `data/docker_mesh/{probe,node_0,node_1,node_2,classes.json}` exist.

- [ ] **Step 3: Bring up the full stack**

Run: `docker compose -f docker/docker-compose.yml up --build`
Expected: all 5 services start; coordinator logs show it waiting, then `all nodes online, starting round loop`, then `all rounds complete`.

- [ ] **Step 4: Verify the merged database**

Run: `python -c "from src.energy.sqlite_store import read_all; import json; print(json.dumps(read_all('outputs/docker_mesh/energy/merged.db'), indent=2))"`
Expected: exactly 3 rows (`round_idx=0`, one per node), each with `energy_kwh`, `knowledge_bytes_sent`, `crop_accuracy`, `disease_accuracy`, and `active=1` all populated (not `None`).

- [ ] **Step 5: Verify the coordinator's event log directly**

Run: `curl http://localhost:9000/events` (or open it in a browser).
Expected: a JSON list of `{ts, node_id, path, status}` entries for every `/round/start`/`/round/gather` call — confirm none of them include payload content, only path/status.

- [ ] **Step 6: Verify the dashboard**

Open `http://localhost:8501`. Expected: "Node status" shows all 3 nodes online; "Round metrics" table shows the same 3 rows as Step 4; "Coordinator event log" shows the same entries as Step 5.

- [ ] **Step 7: Cross-check against the in-process pipeline**

Run: `python -m src.train --config config.yaml --arch mobilenet_v3_small` (same seed/config) and compare its per-node `crop_accuracy`/`disease_accuracy` from `outputs/results_mobilenet_v3_small.json` against the Docker run's `merged.db` rows. Expected: same ballpark (not exact — real wall-clock/thread-scheduling nondeterminism between the two runs is expected), which is the practical confirmation that Task 1's class-index alignment fix worked end-to-end rather than silently misaligning logits.

- [ ] **Step 8: Tear down and revert the smoke-test config change**

Run: `docker compose -f docker/docker-compose.yml down`
Revert `config.yaml`'s `training.rounds` back to its original value (5).

- [ ] **Step 9: Commit the reverted config (if Step 1's edit was accidentally committed) or confirm there is nothing to commit**

```bash
git status
```
Expected: `config.yaml` shows no diff (already reverted in Step 8) — no commit needed for this task.

---

## Appendix: Local verification without Docker

`NodeRunner`/`CoordinatorRunner` (Tasks 5, 7) don't know or care whether they're inside a container — `main.py`/`app.py` (Tasks 6, 8, 10) are thin, env-var-driven wiring with no Docker-specific code. Every piece can run as plain host processes — no broker, no Docker at all needed for this workflow (the HTTP redesign removed even the one piece — Mosquitto — that would have needed *something* running).

This works because of the `sys.path.insert` in `main.py`/`app.py` above (resolves `src/` from the repo root either way) — do this appendix's steps only after Tasks 1–10 are implemented.

1. `python scripts/split_node_data.py` — writes `data/docker_mesh/` on the host filesystem directly.
2. In separate terminals, from the repo root, with paths pointing at the host's `data/docker_mesh/...` (not the container's `/data/...`):
   - `node_0`: `NODE_ID=node_0 CONFIG_PATH=config.yaml DATA_ROOT=data/docker_mesh/node_0 PROBE_ROOT=data/docker_mesh/probe CLASSES_JSON=data/docker_mesh/classes.json ENERGY_DB=outputs/docker_mesh/energy/node_0.db python docker/node/main.py` (this starts a `uvicorn` server on port 8000 — repeat for `node_1`/`node_2` with their own `NODE_ID`/`DATA_ROOT`/`ENERGY_DB`, but each needs its own port: pass `--port 8001`/`--port 8002` by editing the `uvicorn.run(...)` call's port for the second/third instance, or run each node from a separate working directory with a small per-node override — simplest is temporarily editing the port per terminal for this manual check.)
   - Coordinator: since `docker/coordinator/main.py` derives node base URLs as `http://node_i:8000` (a Docker Compose DNS assumption), running it locally against nodes on `localhost:8000/8001/8002` needs those URLs overridden — the cleanest way to do this locally is to temporarily hardcode `node_base_urls` in `main()` to `{"node_0": "http://localhost:8000", "node_1": "http://localhost:8001", "node_2": "http://localhost:8002"}` for this manual check only (don't commit that change): `ENERGY_DB=outputs/docker_mesh/energy/merged.db CONFIG_PATH=config.yaml python docker/coordinator/main.py`.
   - Dashboard: same idea — temporarily override `NODE_BASE_URLS` to point at `localhost:8000/8001/8002`, then `COORDINATOR_EVENTS_URL=http://localhost:9000/events MERGED_DB=outputs/docker_mesh/energy/merged.db streamlit run docker/dashboard/app.py`.
3. Open `http://localhost:8501`.

Once this works, moving to real Docker images (Tasks 6/8/10's `docker build` steps, then Task 11's compose file) is just deleting the temporary localhost-port overrides — Compose's per-service DNS (`http://node_0:8000`) is what the code already assumes by default.
