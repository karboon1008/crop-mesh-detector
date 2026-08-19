# Corn Disease-Split Knowledge-Transfer Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-node, single-crop (Corn), disease-disjoint local training pipeline plus a knowledge-transfer stage that reuses this repo's existing federated exchange mechanics unchanged, producing a per-round ONNX model and a fairness-aligned JSON report (energy, bytes, accuracy) per node.

**Architecture:** Stage 1 (`src/validation/run_corn_pipeline.py`) trains 3 independent per-node baselines using the existing node1-validation recipe, scoped via a new `src/validation/corn_mesh_dataset.py` data layer. Stage 2 (`src/validation/run_knowledge_transfer.py`) loads those 3 checkpoints and runs 1-5 rounds of `distill()`-only knowledge exchange (`src/federated/node.py`/`mesh.py`, unmodified) alongside a budget-aligned local-only control arm, exporting ONNX + evaluating (local + cross-node) + logging energy/bytes each round.

**Tech Stack:** PyTorch, timm (via existing `src/models/factory.py`), onnxruntime, pytest — all already in this repo's `requirements.txt`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md`

## Global Constraints

- Do not modify `src/federated/node.py`, `src/federated/mesh.py`, `src/federated/aggregation.py`, `src/energy/tracker.py`, `src/data/plantvillage.py`, `src/models/factory.py`, `src/validation/node1_dataset.py`, `src/validation/train_mobilenet.py`, `src/validation/export_onnx.py`, or `src/validation/evaluate_onnx.py` — import and reuse their existing public functions/classes as-is.
- Dataset build stays in-memory — filter `data/PlantVillage` sample indices at load time, no physical per-node copy.
- Crop: Corn only, 4 classes: `healthy` (1,162), `Common_rust` (1,192), `Cercospora_leaf_spot_Gray_leaf_spot` (513), `Northern_Leaf_Blight` (985).
- Node/disease assignment (fixed): `node_0`=`Common_rust`, `node_1`=`Cercospora_leaf_spot_Gray_leaf_spot`, `node_2`=`Northern_Leaf_Blight`, each plus a disjoint ~1/3 share of `healthy`.
- Model: `mobilenet_v3_small` only. Compact label space: `crop_classes=["Corn"]`, `disease_classes=["healthy","Common_rust","Cercospora_leaf_spot_Gray_leaf_spot","Northern_Leaf_Blight"]`.
- Stage 2 rounds never call `node.local_train()` on the knowledge-transfer nodes — `node.distill()` alone is the per-round update, continuing from the previous round's (or stage 1's) in-memory model. Nothing resets to a fresh/random model between rounds.
- The local-only control arm's `node.local_train(epochs=training.distill_epochs_per_round, lr=training.distill_lr)` budget must match the knowledge-transfer arm's local-supervised phase budget round-for-round (Appendix A.1 fairness requirement).
- All new config keys live under a new, additive `corn_mesh:` section in `config.yaml`; everything else (`data.*`, `training.*`, `federated.*`, `energy.*`) is read from existing sections unchanged.

---

## File Structure

```
config.yaml                                  # + corn_mesh: section (Task 1)
src/validation/
  corn_mesh_dataset.py                       # NEW — CornLabelMap, CornDiseaseView, filtering, splits (Tasks 1-3)
  run_corn_pipeline.py                        # NEW — stage 1 CLI (Task 4)
  run_knowledge_transfer.py                   # NEW — stage 2 CLI (Tasks 5-7)
tests/
  test_validation_corn_mesh_dataset.py        # NEW (Tasks 1-3)
  test_validation_run_corn_pipeline.py        # NEW (Task 4)
  test_validation_run_knowledge_transfer.py   # NEW (Tasks 5-7)
tests/conftest.py                             # + corn_scoped_config fixture (Task 1)
```

---

### Task 1: Config section + CornLabelMap + Corn/disease index filtering

**Files:**
- Modify: `config.yaml` (add `corn_mesh:` section at end of file)
- Modify: `tests/conftest.py` (add `corn_scoped_config` fixture)
- Create: `src/validation/corn_mesh_dataset.py`
- Test: `tests/test_validation_corn_mesh_dataset.py`

**Interfaces:**
- Produces: `CornLabelMap` (dataclass: `crop_classes: list[str]`, `disease_classes: list[str]`, `name_to_disease_idx: dict[str, int]`), `CORN_DISEASE_ORDER: list[str]`, `NODE_DISEASE_DEFAULT: dict[str, str]`, `build_corn_label_map() -> CornLabelMap`, `get_corn_indices(dataset: PlantVillageDataset) -> list[int]`, `get_corn_disease_indices(dataset: PlantVillageDataset, indices: list[int], disease_name: str) -> list[int]`.

- [ ] **Step 1: Add the `corn_mesh` config section**

Append to `config.yaml`:

```yaml

corn_mesh:
  crop: "Corn"
  node_diseases:
    node_0: "Common_rust"
    node_1: "Cercospora_leaf_spot_Gray_leaf_spot"
    node_2: "Northern_Leaf_Blight"
  healthy_dedup_threshold: 5
  rounds: 2
  output_dir: "outputs/validation/corn_mesh"
```

- [ ] **Step 2: Add the `corn_scoped_config` fixture to `tests/conftest.py`**

Add this fixture (place it after `node1_scoped_config`):

```python
@pytest.fixture
def corn_scoped_config(tmp_path):
    """A tiny synthetic Corn-shaped dataset (4 classes) plus one
    unrelated Tomato class (to prove Corn-filtering excludes other
    crops), used across the corn_mesh test suite.
    """
    from src.config import Config

    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    class_counts = {
        "Corn___healthy": 12,
        "Corn___Common_rust": 6,
        "Corn___Cercospora_leaf_spot_Gray_leaf_spot": 6,
        "Corn___Northern_Leaf_Blight": 6,
        "Tomato___healthy": 4,
    }
    for cls, count in class_counts.items():
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(count):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")

    cfg = Config(
        {
            "data": {
                "root": str(root),
                "image_size": 32,
                "seed": 42,
                "test_fraction": 0.25,
                "probe_set_fraction": 0.1,
                "probe_set_large_class_threshold": 200,
                "probe_set_min_samples_small_class": 1,
                "probe_set_max_fraction_small_class": 0.5,
            },
            "corn_mesh": {
                "crop": "Corn",
                "node_diseases": {
                    "node_0": "Common_rust",
                    "node_1": "Cercospora_leaf_spot_Gray_leaf_spot",
                    "node_2": "Northern_Leaf_Blight",
                },
                "healthy_dedup_threshold": 5,
                "rounds": 2,
            },
            "training": {
                "distill_epochs_per_round": 1,
                "distill_lr": 1e-3,
                "proto_weight": 0.5,
                "kd_weight": 0.5,
                "kd_temperature": 2.0,
            },
            "federated": {
                "aggregation": "trimmed_mean",
                "trim_fraction": 0.0,
                "krum_neighbors": 1,
            },
        }
    )
    return cfg, root
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_validation_corn_mesh_dataset.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.corn_mesh_dataset import (
    CORN_DISEASE_ORDER,
    build_corn_label_map,
    get_corn_disease_indices,
    get_corn_indices,
)


def test_build_corn_label_map_has_one_crop_and_four_diseases():
    label_map = build_corn_label_map()

    assert label_map.crop_classes == ["Corn"]
    assert label_map.disease_classes == CORN_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == 0
    assert label_map.name_to_disease_idx["Common_rust"] == 1


def test_get_corn_indices_excludes_other_crops(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    corn_indices = get_corn_indices(dataset)

    assert len(corn_indices) == 12 + 6 + 6 + 6  # healthy + 3 diseases, no Tomato
    for idx in corn_indices:
        crop_idx, _ = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        assert dataset.labels.crop_classes[crop_idx] == "Corn"


def test_get_corn_disease_indices_filters_to_named_disease(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)

    rust_indices = get_corn_disease_indices(dataset, corn_indices, "Common_rust")
    healthy_indices = get_corn_disease_indices(dataset, corn_indices, "healthy")

    assert len(rust_indices) == 6
    assert len(healthy_indices) == 12
    assert set(rust_indices).isdisjoint(set(healthy_indices))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.corn_mesh_dataset'`

- [ ] **Step 5: Implement `src/validation/corn_mesh_dataset.py`**

```python
"""Scopes the full PlantVillage dataset down to Corn only, and remaps
Corn's 4 classes (healthy + 3 diseases) into a compact, Corn-only label
space -- see docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md
for why the head is scoped this tightly instead of reusing the full
14-crop/~22-disease global label space.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.plantvillage import PlantVillageDataset

CORN_DISEASE_ORDER = [
    "healthy",
    "Common_rust",
    "Cercospora_leaf_spot_Gray_leaf_spot",
    "Northern_Leaf_Blight",
]

NODE_DISEASE_DEFAULT = {
    "node_0": "Common_rust",
    "node_1": "Cercospora_leaf_spot_Gray_leaf_spot",
    "node_2": "Northern_Leaf_Blight",
}


@dataclass
class CornLabelMap:
    crop_classes: list[str]
    disease_classes: list[str]
    name_to_disease_idx: dict[str, int]


def build_corn_label_map(disease_names: list[str] = CORN_DISEASE_ORDER) -> CornLabelMap:
    return CornLabelMap(
        crop_classes=["Corn"],
        disease_classes=list(disease_names),
        name_to_disease_idx={name: i for i, name in enumerate(disease_names)},
    )


def get_corn_indices(dataset: PlantVillageDataset) -> list[int]:
    corn_crop_idx = dataset.labels.crop_classes.index("Corn")
    indices = []
    for idx in range(len(dataset)):
        crop_idx, _ = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        if crop_idx == corn_crop_idx:
            indices.append(idx)
    return indices


def get_corn_disease_indices(
    dataset: PlantVillageDataset, indices: list[int], disease_name: str
) -> list[int]:
    result = []
    for idx in indices:
        _, disease_idx = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
        if dataset.labels.disease_classes[disease_idx] == disease_name:
            result.append(idx)
    return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add config.yaml tests/conftest.py src/validation/corn_mesh_dataset.py tests/test_validation_corn_mesh_dataset.py
git commit -m "feat: add corn_mesh config + Corn label map and index filtering"
```

---

### Task 2: Healthy 3-way dedup-aware split

**Files:**
- Modify: `src/validation/corn_mesh_dataset.py`
- Test: `tests/test_validation_corn_mesh_dataset.py`

**Interfaces:**
- Consumes: `src.validation.node1_dataset.compute_image_hashes(dataset, indices) -> dict[int, int]`, `src.validation.node1_dataset.group_duplicates(indices, hashes, threshold) -> dict[int, int]` (both already dataset-agnostic, reused unmodified).
- Produces: `split_healthy_3way(dataset: PlantVillageDataset, healthy_indices: list[int], num_nodes: int, seed: int, threshold: int = 5) -> list[list[int]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation_corn_mesh_dataset.py`:

```python
from src.validation.corn_mesh_dataset import split_healthy_3way


def test_split_healthy_3way_produces_disjoint_complete_shares(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)
    healthy_indices = get_corn_disease_indices(dataset, corn_indices, "healthy")

    shares = split_healthy_3way(dataset, healthy_indices, num_nodes=3, seed=0, threshold=5)

    assert len(shares) == 3
    all_assigned = [idx for share in shares for idx in share]
    assert sorted(all_assigned) == sorted(healthy_indices)
    for i in range(3):
        for j in range(i + 1, 3):
            assert set(shares[i]).isdisjoint(set(shares[j]))
    # 12 healthy images, real (non-duplicate) random noise -> roughly balanced
    assert max(len(s) for s in shares) - min(len(s) for s in shares) <= 2


def test_split_healthy_3way_keeps_duplicate_groups_together():
    from src.data.plantvillage import PlantVillageDataset as PVD

    class _FakeLabels:
        pass

    class _FakeDataset:
        """Minimal stand-in exposing exactly what compute_image_hashes and
        group_duplicates need (dataset.base.samples), so this test doesn't
        need to write real duplicate-hash image files to disk.
        """

    # compute_image_hashes opens real image files, so this test drives
    # split_healthy_3way's grouping logic directly via group_duplicates
    # instead of through compute_image_hashes -- verifies groups never
    # split across shares regardless of where the hashes came from.
    from src.validation.node1_dataset import group_duplicates

    indices = [0, 1, 2, 3, 4, 5]
    hashes = {0: 0, 1: 1, 2: 0b1111_0000, 3: 0b1111_0001, 4: 0b0101_0101, 5: 0b1010_1010}
    groups = group_duplicates(indices, hashes, threshold=1)
    group_members: dict[int, list[int]] = {}
    for idx, gid in groups.items():
        group_members.setdefault(gid, []).append(idx)

    # Reimplement just the greedy-assignment half (the part under test)
    # against these hand-built groups, mirroring what split_healthy_3way
    # does internally after grouping.
    from src.validation.corn_mesh_dataset import _assign_groups_to_shares

    shares = _assign_groups_to_shares(list(group_members.values()), num_nodes=3, seed=0)
    idx_to_share = {idx: i for i, share in enumerate(shares) for idx in share}
    assert idx_to_share[0] == idx_to_share[1]
    assert idx_to_share[2] == idx_to_share[3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v -k split_healthy`
Expected: FAIL with `ImportError: cannot import name 'split_healthy_3way'`

- [ ] **Step 3: Implement `split_healthy_3way` and `_assign_groups_to_shares`**

Add to `src/validation/corn_mesh_dataset.py` (add `import random` to the top imports, and the `node1_dataset` import):

```python
import random

from src.validation.node1_dataset import compute_image_hashes, group_duplicates
```

```python
def _assign_groups_to_shares(
    group_list: list[list[int]], num_nodes: int, seed: int
) -> list[list[int]]:
    """Greedy load-balancing: largest groups first, each assigned to
    whichever share currently has the fewest images -- keeps every
    duplicate-group intact on one share while balancing share sizes.
    """
    rng = random.Random(seed)
    shuffled = list(group_list)
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    shares: list[list[int]] = [[] for _ in range(num_nodes)]
    for members in shuffled:
        target = min(range(num_nodes), key=lambda i: len(shares[i]))
        shares[target].extend(members)
    return shares


def split_healthy_3way(
    dataset: PlantVillageDataset,
    healthy_indices: list[int],
    num_nodes: int,
    seed: int,
    threshold: int = 5,
) -> list[list[int]]:
    hashes = compute_image_hashes(dataset, healthy_indices)
    groups = group_duplicates(healthy_indices, hashes, threshold)
    group_members: dict[int, list[int]] = {}
    for idx, group_id in groups.items():
        group_members.setdefault(group_id, []).append(idx)
    return _assign_groups_to_shares(list(group_members.values()), num_nodes, seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v -k split_healthy`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/corn_mesh_dataset.py tests/test_validation_corn_mesh_dataset.py
git commit -m "feat: add dedup-aware 3-way healthy split with no cross-node duplicates"
```

---

### Task 3: `prepare_corn_mesh_data` orchestration + `CornDiseaseView`

**Files:**
- Modify: `src/validation/corn_mesh_dataset.py`
- Test: `tests/test_validation_corn_mesh_dataset.py`

**Interfaces:**
- Consumes: `src.data.plantvillage.load_full_dataset`, `carve_public_probe_set` (unmodified); `src.validation.node1_dataset.build_train_eval_datasets`, `compute_image_hashes`, `dedup_aware_split` (unmodified); Task 1/2's `get_corn_indices`, `get_corn_disease_indices`, `split_healthy_3way`, `build_corn_label_map`.
- Produces: `CornMeshData` (dataclass: `train_base: PlantVillageDataset`, `eval_base: PlantVillageDataset`, `label_map: CornLabelMap`, `image_size: int`, `probe_idx: list[int]`, `per_node: dict[str, dict[str, list[int]]]` — each node's dict has keys `"train_idx"`/`"test_idx"`), `prepare_corn_mesh_data(cfg: Config) -> CornMeshData`, `CornDiseaseView(Dataset)` with `__init__(self, base: PlantVillageDataset, indices: list[int], label_map: CornLabelMap)`, `__len__`, `__getitem__(pos) -> (image, 0, compact_disease_idx)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation_corn_mesh_dataset.py`:

```python
from src.validation.corn_mesh_dataset import CornDiseaseView, prepare_corn_mesh_data


def test_corn_disease_view_remaps_labels_to_compact_space(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    corn_indices = get_corn_indices(dataset)
    rust_indices = get_corn_disease_indices(dataset, corn_indices, "Common_rust")
    label_map = build_corn_label_map()

    view = CornDiseaseView(dataset, rust_indices, label_map)

    assert len(view) == len(rust_indices)
    for pos in range(len(view)):
        image, crop_label, disease_label = view[pos]
        assert crop_label == 0
        assert disease_label == label_map.name_to_disease_idx["Common_rust"]


def test_prepare_corn_mesh_data_produces_disjoint_per_node_splits_and_probe_set(corn_scoped_config):
    cfg, root = corn_scoped_config

    data = prepare_corn_mesh_data(cfg)

    assert set(data.per_node.keys()) == {"node_0", "node_1", "node_2"}
    all_train_test = []
    for node_id, splits in data.per_node.items():
        train_idx, test_idx = splits["train_idx"], splits["test_idx"]
        assert set(train_idx).isdisjoint(set(test_idx))
        all_train_test.extend(train_idx + test_idx)

    # no index appears in more than one node's local data (disjoint diseases
    # + disjoint healthy shares), and none overlap the shared probe set
    assert len(all_train_test) == len(set(all_train_test))
    assert set(all_train_test).isdisjoint(set(data.probe_idx))
    assert len(data.probe_idx) > 0
    assert data.label_map.crop_classes == ["Corn"]


def test_prepare_corn_mesh_data_each_node_only_has_its_own_disease_plus_healthy(corn_scoped_config):
    cfg, root = corn_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    data = prepare_corn_mesh_data(cfg)

    node_diseases = cfg.get("corn_mesh.node_diseases")
    for node_id, own_disease in node_diseases.items():
        combined = data.per_node[node_id]["train_idx"] + data.per_node[node_id]["test_idx"]
        disease_names = set()
        for idx in combined:
            _, disease_idx = dataset.labels.class_to_crop_disease[dataset.base.targets[idx]]
            disease_names.add(dataset.labels.disease_classes[disease_idx])
        assert disease_names <= {own_disease, "healthy"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v -k "corn_disease_view or prepare_corn_mesh"`
Expected: FAIL with `ImportError: cannot import name 'CornDiseaseView'`

- [ ] **Step 3: Implement `CornDiseaseView` and `prepare_corn_mesh_data`**

Add to `src/validation/corn_mesh_dataset.py` (add these imports at the top):

```python
from dataclasses import dataclass, field

from torch.utils.data import Dataset

from src.config import Config
from src.data.plantvillage import carve_public_probe_set, load_full_dataset
from src.validation.node1_dataset import build_train_eval_datasets, dedup_aware_split
```

```python
class CornDiseaseView(Dataset):
    """Wraps a PlantVillageDataset + a fixed list of its raw sample
    indices, remapping each sample's disease label into the compact
    Corn-only label space (see build_corn_label_map). Crop label is
    always 0 -- there is only one crop in this experiment.
    """

    def __init__(self, base: PlantVillageDataset, indices: list[int], label_map: CornLabelMap):
        self.base = base
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        image, _, disease_idx = self.base[self.indices[pos]]
        disease_name = self.base.labels.disease_classes[disease_idx]
        compact_idx = self.label_map.name_to_disease_idx[disease_name]
        return image, 0, compact_idx


@dataclass
class CornMeshData:
    train_base: PlantVillageDataset
    eval_base: PlantVillageDataset
    label_map: CornLabelMap
    image_size: int
    probe_idx: list[int]
    per_node: dict[str, dict[str, list[int]]] = field(default_factory=dict)


def prepare_corn_mesh_data(cfg: Config) -> CornMeshData:
    image_size = cfg.get("data.image_size", 160)
    root = cfg.get("data.root", "data/PlantVillage")
    seed = cfg.get("data.seed", 42)
    test_fraction = cfg.get("data.test_fraction", 0.15)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)
    healthy_dedup_threshold = cfg.get("corn_mesh.healthy_dedup_threshold", 5)
    node_diseases = cfg.get("corn_mesh.node_diseases", NODE_DISEASE_DEFAULT)

    dataset = load_full_dataset(root, image_size)
    label_map = build_corn_label_map()

    corn_indices = get_corn_indices(dataset)
    corn_set = set(corn_indices)

    probe_idx_full, remaining_idx_full = carve_public_probe_set(
        dataset,
        probe_fraction,
        seed,
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )
    probe_idx = [i for i in probe_idx_full if i in corn_set]
    remaining_corn_idx = [i for i in remaining_idx_full if i in corn_set]

    healthy_remaining = get_corn_disease_indices(dataset, remaining_corn_idx, "healthy")
    healthy_shares = split_healthy_3way(
        dataset, healthy_remaining, num_nodes=3, seed=seed, threshold=healthy_dedup_threshold
    )

    per_node: dict[str, dict[str, list[int]]] = {}
    for node_id, disease_name in node_diseases.items():
        node_idx = int(str(node_id).rsplit("_", 1)[-1])
        disease_idx_list = get_corn_disease_indices(dataset, remaining_corn_idx, disease_name)
        combined = disease_idx_list + healthy_shares[node_idx]
        hashes = compute_image_hashes(dataset, combined)
        train_idx, test_idx = dedup_aware_split(
            combined, hashes, test_fraction, seed, threshold=healthy_dedup_threshold
        )
        per_node[node_id] = {"train_idx": train_idx, "test_idx": test_idx}

    train_base, eval_base = build_train_eval_datasets(root, image_size)
    return CornMeshData(
        train_base=train_base,
        eval_base=eval_base,
        label_map=label_map,
        image_size=image_size,
        probe_idx=probe_idx,
        per_node=per_node,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_corn_mesh_dataset.py -v`
Expected: PASS (all tests in this file, ~9 total)

- [ ] **Step 5: Commit**

```bash
git add src/validation/corn_mesh_dataset.py tests/test_validation_corn_mesh_dataset.py
git commit -m "feat: add prepare_corn_mesh_data orchestration and CornDiseaseView"
```

---

### Task 4: Stage 1 — `run_corn_pipeline.py`

**Files:**
- Create: `src/validation/run_corn_pipeline.py`
- Test: `tests/test_validation_run_corn_pipeline.py`

**Interfaces:**
- Consumes: Task 3's `CornMeshData`, `CornDiseaseView`, `prepare_corn_mesh_data`; `src.validation.train_mobilenet.run_training` (unmodified, signature: `run_training(train_ds, train_idx, eval_ds, test_idx, num_crop_classes, num_disease_classes, output_dir, epochs=15, batch_size=32, lr=0.001, weight_decay=1e-4, pretrained=True, device="cpu") -> dict`); `src.validation.export_onnx.export_checkpoint` (unmodified); `src.validation.evaluate_onnx.run_evaluation` (unmodified).
- Produces: `NODE_LABELS: dict[str, str]`, `node_output_dir(base_output_dir: Path, node_id: str) -> Path`, `run_train_stage(data: CornMeshData, node_id: str, output_dir: Path, epochs=15, pretrained=True) -> None`, `run_export_stage(data, node_id, output_dir, num_parity_samples=8) -> None`, `run_evaluate_stage(data, node_id, output_dir) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_run_corn_pipeline.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.corn_mesh_dataset import prepare_corn_mesh_data
from src.validation.run_corn_pipeline import (
    run_evaluate_stage,
    run_export_stage,
    run_train_stage,
)


def test_run_export_stage_raises_clear_error_when_train_stage_not_run(tmp_path, corn_scoped_config):
    cfg, _ = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="train"):
        run_export_stage(data, "node_0", output_dir)


def test_run_evaluate_stage_raises_clear_error_when_export_stage_not_run(tmp_path, corn_scoped_config):
    cfg, _ = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="export"):
        run_evaluate_stage(data, "node_0", output_dir)


def test_full_pipeline_produces_a_well_formed_report_per_node(tmp_path, corn_scoped_config):
    cfg, _ = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)

    for node_id in ("node_0", "node_1", "node_2"):
        output_dir = tmp_path / node_id
        run_train_stage(data, node_id, output_dir, epochs=2, pretrained=False)
        run_export_stage(data, node_id, output_dir)
        run_evaluate_stage(data, node_id, output_dir)

        report = json.loads((output_dir / "report.json").read_text())
        assert "summary" in report and "results" in report
        assert report["summary"]["num_test_images"] == len(report["results"])
        assert report["summary"]["num_test_images"] > 0
        classes = json.loads((output_dir / "classes.json").read_text())
        assert classes["crop_classes"] == ["Corn"]
        assert len(classes["disease_classes"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_run_corn_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.run_corn_pipeline'`

- [ ] **Step 3: Implement `src/validation/run_corn_pipeline.py`**

```python
"""CLI entrypoint chaining the Corn disease-split validation pipeline
across all 3 nodes: train -> export -> evaluate. Generalizes
run_pipeline.py's single-node/single-model chaining to node_0/1/2, each
holding one disjoint Corn disease class plus a disjoint share of
healthy -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_corn_pipeline
    python -m src.validation.run_corn_pipeline --stage train
    python -m src.validation.run_corn_pipeline --stage export
    python -m src.validation.run_corn_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.validation.corn_mesh_dataset import CornDiseaseView, CornMeshData, prepare_corn_mesh_data
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"
NODE_LABELS = {"node_0": "common_rust", "node_1": "cercospora", "node_2": "northern_leaf_blight"}


def node_output_dir(base_output_dir: Path, node_id: str) -> Path:
    return base_output_dir / f"{node_id}_{NODE_LABELS[node_id]}_{MODEL_NAME}"


def run_train_stage(
    data: CornMeshData, node_id: str, output_dir: Path, epochs: int = 15, pretrained: bool = True
) -> None:
    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.per_node[node_id]["test_idx"]
    train_view = CornDiseaseView(data.train_base, train_idx, data.label_map)
    eval_view = CornDiseaseView(data.eval_base, test_idx, data.label_map)

    run_training(
        train_view,
        list(range(len(train_idx))),
        eval_view,
        list(range(len(test_idx))),
        len(data.label_map.crop_classes),
        len(data.label_map.disease_classes),
        output_dir,
        epochs=epochs,
        pretrained=pretrained,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "train_idx": train_idx,
                "test_idx": test_idx,
            },
            indent=2,
        )
    )


def run_export_stage(data: CornMeshData, node_id: str, output_dir: Path, num_parity_samples: int = 8) -> None:
    checkpoint_path = output_dir / "checkpoint.pt"
    classes_path = output_dir / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} and/or {classes_path} not found — run the 'train' stage first."
        )
    classes = json.loads(classes_path.read_text())
    test_idx = classes["test_idx"]

    eval_view = CornDiseaseView(data.eval_base, test_idx, data.label_map)
    sample_idx = list(range(min(num_parity_samples, len(eval_view))))
    parity_samples = [eval_view[i] for i in sample_idx]

    export_checkpoint(
        checkpoint_path,
        classes["crop_classes"],
        classes["disease_classes"],
        classes["image_size"],
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(data: CornMeshData, node_id: str, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    classes_path = output_dir / "classes.json"
    onnx_path = output_dir / "model.onnx"
    if not manifest_path.exists() or not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} and/or {manifest_path} not found — run the 'export' stage first."
        )
    manifest = json.loads(manifest_path.read_text())
    classes = json.loads(classes_path.read_text())
    test_idx = classes["test_idx"]

    session = onnxruntime.InferenceSession(str(onnx_path))
    run_evaluation(session, data.eval_base, test_idx, manifest, MODEL_NAME, node_id, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/corn_mesh")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)
    data = prepare_corn_mesh_data(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
            },
            indent=2,
        )
    )

    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(output_dir, node_id)
        print(f"=== {node_id} ===")
        for stage in ([args.stage] if args.stage else list(STAGES)):
            print(f"  -- stage: {stage} --")
            if stage == "train":
                run_train_stage(data, node_id, node_dir)
            elif stage == "export":
                run_export_stage(data, node_id, node_dir)
            elif stage == "evaluate":
                run_evaluate_stage(data, node_id, node_dir)

    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_run_corn_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/run_corn_pipeline.py tests/test_validation_run_corn_pipeline.py
git commit -m "feat: add stage 1 per-node Corn validation pipeline CLI"
```

---

### Task 5: Stage 2 core mechanics — `run_kt_round`

**Files:**
- Create: `src/validation/run_knowledge_transfer.py`
- Test: `tests/test_validation_run_knowledge_transfer.py`

**Interfaces:**
- Consumes: `src.federated.node.Node`, `KnowledgePayload` (unmodified — `Node.compute_knowledge(probe_loader) -> KnowledgePayload`, `Node.distill(consensus_prototypes, consensus_crop_logits, consensus_disease_logits, probe_loader, epochs, lr, proto_weight, kd_weight, temperature) -> dict[str, float]`, `Node.local_train(epochs, lr, progress_cb=None) -> float`); `src.federated.aggregation.aggregate_prototypes`, `aggregate_logits` (unmodified); `src.energy.tracker.ComputeEnergyTracker` (unmodified, `.track(label)` context manager).
- Produces: `run_kt_round(nodes: dict[str, Node], control_nodes: dict[str, Node], probe_loader: DataLoader, aggregation_method: str, trim_fraction: float, krum_neighbors: int, distill_epochs: int, distill_lr: float, proto_weight: float, kd_weight: float, temperature: float, tracker: ComputeEnergyTracker | None = None) -> dict` — returns `{"per_node_distill_loss": {...}, "per_node_bytes_sent": {...}, "total_bytes_exchanged": int}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation_run_knowledge_transfer.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import make_subset
from src.federated.node import Node
from src.models.factory import build_model
from src.validation.corn_mesh_dataset import CornDiseaseView, prepare_corn_mesh_data
from src.validation.run_knowledge_transfer import run_kt_round


def _build_kt_nodes(data, batch_size=4):
    nodes = {}
    for node_id in ("node_0", "node_1", "node_2"):
        train_idx = data.per_node[node_id]["train_idx"]
        test_idx = data.per_node[node_id]["test_idx"]
        train_loader = DataLoader(
            CornDiseaseView(data.train_base, train_idx, data.label_map), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            CornDiseaseView(data.eval_base, test_idx, data.label_map), batch_size=batch_size, shuffle=False
        )
        model = build_model("mobilenet_v3_small", 1, 4, pretrained=False)
        nodes[node_id] = Node(node_id, model, train_loader, test_loader, device="cpu")
    return nodes


def test_run_kt_round_updates_all_nodes_and_reports_bytes(corn_scoped_config):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)

    before_control_params = {
        nid: [p.clone() for p in n.model.parameters()] for nid, n in control_nodes.items()
    }

    result = run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method="trimmed_mean",
        trim_fraction=0.0,
        krum_neighbors=1,
        distill_epochs=1,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )

    assert set(result["per_node_distill_loss"].keys()) == {"node_0", "node_1", "node_2"}
    for loss_dict in result["per_node_distill_loss"].values():
        assert set(loss_dict.keys()) == {"kd_loss", "sup_loss", "proto_loss", "total_loss"}
    assert result["total_bytes_exchanged"] > 0
    assert set(result["per_node_bytes_sent"].keys()) == {"node_0", "node_1", "node_2"}

    # control nodes' weights must have moved too (local_train ran)
    for nid, node in control_nodes.items():
        after = list(node.model.parameters())
        assert any(
            not before.equal(after_p) for before, after_p in zip(before_control_params[nid], after)
        )


def test_run_kt_round_tracks_energy_per_node_when_tracker_given(corn_scoped_config):
    from src.energy.tracker import ComputeEnergyTracker

    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)
    tracker = ComputeEnergyTracker(enabled=False, output_dir=root.parent / "energy_out", fallback_power_watts=15.0)

    run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method="trimmed_mean",
        trim_fraction=0.0,
        krum_neighbors=1,
        distill_epochs=1,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
        tracker=tracker,
    )

    assert tracker.summary()["num_tracked_blocks"] == 6  # 3 KT nodes + 3 control nodes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.run_knowledge_transfer'`

- [ ] **Step 3: Implement the core of `src/validation/run_knowledge_transfer.py`**

```python
"""Stage 2: 1-5 rounds of knowledge-transfer distillation on top of the
3 per-node checkpoints stage 1 (run_corn_pipeline.py) produces, plus a
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md.

    python -m src.validation.run_knowledge_transfer --rounds 2
"""

from __future__ import annotations

from contextlib import nullcontext

from torch.utils.data import DataLoader

from src.energy.tracker import ComputeEnergyTracker
from src.federated.aggregation import aggregate_logits, aggregate_prototypes
from src.federated.node import Node

MODEL_NAME = "mobilenet_v3_small"


def run_kt_round(
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    aggregation_method: str,
    trim_fraction: float,
    krum_neighbors: int,
    distill_epochs: int,
    distill_lr: float,
    proto_weight: float,
    kd_weight: float,
    temperature: float,
    tracker: ComputeEnergyTracker | None = None,
) -> dict:
    """One knowledge-transfer round: `nodes` distill toward their peers'
    consensus (no separate local_train call); `control_nodes` run the
    same-budget local_train with no exchange at all, for the Appendix
    A.1 fairness-aligned comparison. Both dicts are mutated in place
    (each Node's .model is updated); nothing is reloaded from disk here.
    """
    payloads = {node_id: node.compute_knowledge(probe_loader) for node_id, node in nodes.items()}
    per_node_bytes_sent = {node_id: payload.size_bytes() for node_id, payload in payloads.items()}
    total_bytes_exchanged = sum(per_node_bytes_sent.values()) * max(0, len(nodes) - 1)

    per_node_distill_loss: dict[str, dict[str, float]] = {}
    for node_id, node in nodes.items():
        peer_payloads = [p for nid, p in payloads.items() if nid != node_id]
        consensus_prototypes = aggregate_prototypes(
            [p.prototypes for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_crop_logits = aggregate_logits(
            [p.crop_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        consensus_disease_logits = aggregate_logits(
            [p.disease_logits for p in peer_payloads],
            method=aggregation_method,
            trim_fraction=trim_fraction,
            krum_neighbors=krum_neighbors,
        )
        ctx = tracker.track(f"{node_id}_kt_distill") if tracker is not None else nullcontext()
        with ctx:
            per_node_distill_loss[node_id] = node.distill(
                consensus_prototypes,
                consensus_crop_logits,
                consensus_disease_logits,
                probe_loader,
                epochs=distill_epochs,
                lr=distill_lr,
                proto_weight=proto_weight,
                kd_weight=kd_weight,
                temperature=temperature,
            )

    for node_id, node in control_nodes.items():
        ctx = tracker.track(f"{node_id}_local_only_control") if tracker is not None else nullcontext()
        with ctx:
            node.local_train(epochs=distill_epochs, lr=distill_lr)

    return {
        "per_node_distill_loss": per_node_distill_loss,
        "per_node_bytes_sent": per_node_bytes_sent,
        "total_bytes_exchanged": total_bytes_exchanged,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/run_knowledge_transfer.py tests/test_validation_run_knowledge_transfer.py
git commit -m "feat: add knowledge-transfer round mechanics with fairness control arm"
```

---

### Task 6: Per-round export + dual evaluation + `round_summary.json`

**Files:**
- Modify: `src/validation/run_knowledge_transfer.py`
- Test: `tests/test_validation_run_knowledge_transfer.py`

**Interfaces:**
- Consumes: Task 5's `run_kt_round`; `src.validation.export_onnx.export_checkpoint` (unmodified); `src.validation.evaluate_onnx.run_evaluation` (unmodified); `src.energy.tracker.CommunicationCostEstimator` (unmodified).
- Produces: `export_and_evaluate(node: Node, label_map, image_size, eval_base, local_test_idx, cross_node_idx, node_dir: Path, keep_onnx: bool) -> dict`, `run_round_with_io(round_idx: int, nodes, control_nodes, probe_loader, data, cross_node_idx, cfg, tracker, comm_estimator, round_dir: Path) -> dict` (writes `round_dir/round_summary.json`, returns its contents plus `per_node_scores`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validation_run_knowledge_transfer.py`:

```python
import json

from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.validation.run_knowledge_transfer import export_and_evaluate, run_round_with_io


def test_export_and_evaluate_writes_report_with_local_and_cross_node(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    node = nodes["node_0"]
    local_test_idx = data.per_node["node_0"]["test_idx"]
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    node_dir = tmp_path / "node_0"

    report = export_and_evaluate(
        node, data.label_map, data.image_size, data.eval_base, local_test_idx, cross_node_idx, node_dir, keep_onnx=True
    )

    assert set(report.keys()) == {"local", "cross_node"}
    assert (node_dir / "model.onnx").exists()
    assert (node_dir / "manifest.json").exists()
    on_disk = json.loads((node_dir / "report.json").read_text())
    assert on_disk == report


def test_export_and_evaluate_discards_onnx_for_control_arm(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    control_nodes = _build_kt_nodes(data)
    node = control_nodes["node_0"]
    local_test_idx = data.per_node["node_0"]["test_idx"]
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    node_dir = tmp_path / "node_0_control"

    export_and_evaluate(
        node, data.label_map, data.image_size, data.eval_base, local_test_idx, cross_node_idx, node_dir, keep_onnx=False
    )

    assert (node_dir / "checkpoint.pt").exists()
    assert (node_dir / "report.json").exists()
    assert not (node_dir / "model.onnx").exists()
    assert not (node_dir / "manifest.json").exists()


def test_run_round_with_io_writes_round_summary(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    nodes = _build_kt_nodes(data)
    control_nodes = _build_kt_nodes(data)
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=4, shuffle=False)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path / "energy", fallback_power_watts=15.0)
    comm_estimator = CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )
    round_dir = tmp_path / "round_1"

    result = run_round_with_io(
        1, nodes, control_nodes, probe_loader, data, cross_node_idx, cfg, tracker, comm_estimator, round_dir
    )

    summary = json.loads((round_dir / "round_summary.json").read_text())
    assert summary["round"] == 1
    assert set(summary["per_node_scores"].keys()) == {"node_0", "node_1", "node_2"}
    for node_id, scores in summary["per_node_scores"].items():
        assert set(scores.keys()) == {"collective", "local_only_control"}
        assert "local" in scores["collective"] and "cross_node" in scores["collective"]
    assert (round_dir / "node_0" / "model.onnx").exists()
    assert (round_dir / "node_0" / "local_only_control" / "checkpoint.pt").exists()
    assert not (round_dir / "node_0" / "local_only_control" / "model.onnx").exists()
    assert result == summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v -k "export_and_evaluate or run_round_with_io"`
Expected: FAIL with `ImportError: cannot import name 'export_and_evaluate'`

- [ ] **Step 3: Implement `export_and_evaluate` and `run_round_with_io`**

Add to `src/validation/run_knowledge_transfer.py` (add these imports):

```python
import json
from pathlib import Path

import onnxruntime
import torch

from src.config import Config
from src.validation.corn_mesh_dataset import CornLabelMap, CornMeshData
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
```

```python
def export_and_evaluate(
    node: Node,
    label_map: CornLabelMap,
    image_size: int,
    eval_base,
    local_test_idx: list[int],
    cross_node_idx: list[int],
    node_dir: Path,
    keep_onnx: bool,
) -> dict:
    """Saves node.model's current weights, exports to ONNX, evaluates
    against both the local held-out test set and the cross-node union
    set, and writes report.json = {"local": ..., "cross_node": ...}.
    When keep_onnx is False (the local-only control arm — not a
    deployment artifact), model.onnx/manifest.json are deleted again
    right after the transient evaluation session is built.
    """
    node_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = node_dir / "checkpoint.pt"
    torch.save(node.model.state_dict(), checkpoint_path)

    onnx_path = export_checkpoint(
        checkpoint_path, label_map.crop_classes, label_map.disease_classes, image_size, node_dir
    )
    manifest = json.loads((node_dir / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(onnx_path))

    scratch = node_dir / "_scratch_report.json"
    local_report = run_evaluation(session, eval_base, local_test_idx, manifest, MODEL_NAME, node.node_id, scratch)
    cross_node_report = run_evaluation(
        session, eval_base, cross_node_idx, manifest, MODEL_NAME, node.node_id, scratch
    )
    scratch.unlink(missing_ok=True)

    combined = {"local": local_report, "cross_node": cross_node_report}
    (node_dir / "report.json").write_text(json.dumps(combined, indent=2))

    if not keep_onnx:
        onnx_path.unlink(missing_ok=True)
        (node_dir / "manifest.json").unlink(missing_ok=True)

    return combined


def run_round_with_io(
    round_idx: int,
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    data: CornMeshData,
    cross_node_idx: list[int],
    cfg: Config,
    tracker: ComputeEnergyTracker,
    comm_estimator,
    round_dir: Path,
) -> dict:
    round_dir.mkdir(parents=True, exist_ok=True)

    kt_result = run_kt_round(
        nodes,
        control_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
        distill_epochs=cfg.get("training.distill_epochs_per_round", 1),
        distill_lr=cfg.get("training.distill_lr", 0.0005),
        proto_weight=cfg.get("training.proto_weight", 0.5),
        kd_weight=cfg.get("training.kd_weight", 0.5),
        temperature=cfg.get("training.kd_temperature", 2.0),
        tracker=tracker,
    )

    per_node_scores: dict[str, dict] = {}
    for node_id in nodes:
        local_test_idx = data.per_node[node_id]["test_idx"]
        collective_report = export_and_evaluate(
            nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            local_test_idx,
            cross_node_idx,
            round_dir / node_id,
            keep_onnx=True,
        )
        control_report = export_and_evaluate(
            control_nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            local_test_idx,
            cross_node_idx,
            round_dir / node_id / "local_only_control",
            keep_onnx=False,
        )
        per_node_scores[node_id] = {"collective": collective_report, "local_only_control": control_report}

    comm_estimate = comm_estimator.estimate_all_radios(kt_result["total_bytes_exchanged"])
    energy_summary = tracker.summary()

    summary = {
        "round": round_idx,
        "per_node_distill_loss": kt_result["per_node_distill_loss"],
        "per_node_bytes_sent": kt_result["per_node_bytes_sent"],
        "total_bytes_exchanged": kt_result["total_bytes_exchanged"],
        "energy": energy_summary,
        "communication_estimate": comm_estimate,
        "per_node_scores": per_node_scores,
    }
    (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/run_knowledge_transfer.py tests/test_validation_run_knowledge_transfer.py
git commit -m "feat: add per-round ONNX export, dual evaluation, and round_summary.json"
```

---

### Task 7: Cross-round summary, CLI entrypoint, end-to-end test

**Files:**
- Modify: `src/validation/run_knowledge_transfer.py`
- Test: `tests/test_validation_run_knowledge_transfer.py`

**Interfaces:**
- Consumes: Task 4's `node_output_dir`, `NODE_LABELS` (from `run_corn_pipeline.py`); Task 6's `run_round_with_io`.
- Produces: `evaluate_round0_baseline(...) -> dict[str, dict]`, `build_knowledge_transfer_summary(cfg, round0_baseline, round_summaries: list[dict], data) -> dict`, `main()` CLI with `--rounds` (validated to `[1, 5]`), `--config`, `--output-dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validation_run_knowledge_transfer.py`:

```python
from src.validation.run_corn_pipeline import (
    node_output_dir,
    run_evaluate_stage,
    run_export_stage,
    run_train_stage,
)
from src.validation.run_knowledge_transfer import (
    build_knowledge_transfer_summary,
    evaluate_round0_baseline,
    main,
)


def _train_stage1_checkpoints(data, base_dir):
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(base_dir, node_id)
        run_train_stage(data, node_id, node_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, node_dir)
        run_evaluate_stage(data, node_id, node_dir)


def test_evaluate_round0_baseline_uses_stage1_onnx_models(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    _train_stage1_checkpoints(data, stage1_dir)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]

    baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    assert set(baseline.keys()) == {"node_0", "node_1", "node_2"}
    for node_id, report in baseline.items():
        assert "disease" in report["summary"]["per_class_accuracy"]


def test_build_knowledge_transfer_summary_has_appendix_a1_fields(corn_scoped_config, tmp_path):
    cfg, root = corn_scoped_config
    data = prepare_corn_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    _train_stage1_checkpoints(data, stage1_dir)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]
    round0_baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    fake_round_summary = {
        "round": 1,
        "per_node_scores": {
            node_id: {
                "collective": round0_baseline[node_id],
                "local_only_control": round0_baseline[node_id],
            }
            for node_id in ("node_0", "node_1", "node_2")
        },
        "total_bytes_exchanged": 100,
        "energy": {"total_compute_energy_kwh": 0.001},
    }

    summary = build_knowledge_transfer_summary(cfg, round0_baseline, [fake_round_summary], data)

    assert summary["node_count"] == 3
    for key in (
        "data_split",
        "local_only_budget",
        "collective_budget",
        "test_set_scope",
        "fairness_exception_reason",
        "delta_g_formula",
        "per_node_scores",
        "macro_avg_and_worst_node",
        "collaboration_gain_per_disease",
        "limitation_note",
    ):
        assert key in summary
    for node_id in ("node_0", "node_1", "node_2"):
        assert set(summary["collaboration_gain_per_disease"][node_id].keys()) == {
            "healthy",
            "Common_rust",
            "Cercospora_leaf_spot_Gray_leaf_spot",
            "Northern_Leaf_Blight",
        }


def test_main_rejects_rounds_outside_valid_range(tmp_path, corn_scoped_config, monkeypatch):
    cfg, root = corn_scoped_config
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    monkeypatch.setattr(
        "sys.argv",
        ["run_knowledge_transfer", "--config", str(config_path), "--rounds", "6"],
    )

    with pytest.raises(SystemExit):
        main()


def test_main_runs_end_to_end(tmp_path, corn_scoped_config, monkeypatch):
    cfg, root = corn_scoped_config
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.as_dict()))
    stage1_dir = tmp_path / "outputs" / "validation" / "corn_mesh"
    data = prepare_corn_mesh_data(cfg)
    _train_stage1_checkpoints(data, stage1_dir)

    kt_dir = stage1_dir / "knowledge_transfer"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_knowledge_transfer",
            "--config",
            str(config_path),
            "--output-dir",
            str(stage1_dir),
            "--rounds",
            "1",
        ],
    )

    main()

    assert (kt_dir / "round_1" / "round_summary.json").exists()
    assert (kt_dir / "knowledge_transfer_summary.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v -k "round0_baseline or knowledge_transfer_summary or main_"`
Expected: FAIL with `ImportError: cannot import name 'evaluate_round0_baseline'`

- [ ] **Step 3: Implement the rest of `src/validation/run_knowledge_transfer.py`**

Add to the imports at the top of `src/validation/run_knowledge_transfer.py`:

```python
import argparse

from src.data.plantvillage import make_subset
from src.energy.tracker import CommunicationCostEstimator
from src.evaluate import compute_collaboration_gain
from src.models.factory import build_model
from src.validation.corn_mesh_dataset import CornDiseaseView, prepare_corn_mesh_data
from src.validation.run_corn_pipeline import node_output_dir
```

```python
def evaluate_round0_baseline(data: CornMeshData, stage1_dir: Path, cross_node_idx: list[int]) -> dict[str, dict]:
    """Evaluates each node's ALREADY-EXPORTED stage-1 model.onnx (no
    re-export) against the cross-node union set -- this is round 0's
    baseline for the per-disease collaboration-gain table, since stage
    1's own report.json only covers each node's local test set.
    """
    baseline: dict[str, dict] = {}
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        manifest = json.loads((node_dir / "manifest.json").read_text())
        session = onnxruntime.InferenceSession(str(node_dir / "model.onnx"))
        scratch = node_dir / "_round0_scratch.json"
        report = run_evaluation(session, data.eval_base, cross_node_idx, manifest, MODEL_NAME, node_id, scratch)
        scratch.unlink(missing_ok=True)
        baseline[node_id] = report
    return baseline


def _load_node_from_checkpoint(checkpoint_path: Path, data: CornMeshData, node_id: str, batch_size: int) -> Node:
    model = build_model(
        MODEL_NAME, len(data.label_map.crop_classes), len(data.label_map.disease_classes), pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.per_node[node_id]["test_idx"]
    train_loader = DataLoader(
        CornDiseaseView(data.train_base, train_idx, data.label_map), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        CornDiseaseView(data.eval_base, test_idx, data.label_map), batch_size=batch_size, shuffle=False
    )
    return Node(node_id, model, train_loader, test_loader, device="cpu")


def _per_disease_gain_table(
    round0_baseline: dict[str, dict], final_round_scores: dict[str, dict], disease_classes: list[str]
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for node_id, baseline_report in round0_baseline.items():
        table[node_id] = {}
        collective_cross = final_round_scores[node_id]["collective"]["cross_node"]["summary"]
        control_cross = final_round_scores[node_id]["local_only_control"]["cross_node"]["summary"]
        for disease_name in disease_classes:
            round0_acc = baseline_report["summary"]["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            collective_acc = collective_cross["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            control_acc = control_cross["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            table[node_id][disease_name] = {
                "round_0_accuracy": round0_acc,
                "round_N_collective_accuracy": collective_acc,
                "round_N_local_only_control_accuracy": control_acc,
                "gain_vs_round0": collective_acc - round0_acc,
                "gain_vs_local_only_control": collective_acc - control_acc,
            }
    return table


def build_knowledge_transfer_summary(
    cfg: Config, round0_baseline: dict[str, dict], round_summaries: list[dict], data: CornMeshData
) -> dict:
    final_round = round_summaries[-1]
    final_scores = final_round["per_node_scores"]

    collective_evals = {nid: s["collective"]["cross_node"]["summary"] for nid, s in final_scores.items()}
    control_evals = {nid: s["local_only_control"]["cross_node"]["summary"] for nid, s in final_scores.items()}
    gain = compute_collaboration_gain(collective_evals, control_evals)

    node_diseases = cfg.get("corn_mesh.node_diseases", NODE_DISEASE_DEFAULT_FALLBACK)
    cumulative_bytes = sum(r["total_bytes_exchanged"] for r in round_summaries)
    cumulative_energy = sum(r["energy"]["total_compute_energy_kwh"] for r in round_summaries)

    return {
        "node_count": 3,
        "data_split": {
            "strategy": "disjoint disease-label skew within one crop (Corn)",
            "strength": (
                "complete disjoint — each node's 3 disease classes have zero overlap with its "
                "peers'; only the shared healthy class is split (dedup-aware, ~1/3 each, no image "
                "duplicated across nodes)"
            ),
            "node_diseases": node_diseases,
        },
        "local_only_budget": {
            "epochs_per_round": cfg.get("training.distill_epochs_per_round", 1),
            "lr": cfg.get("training.distill_lr", 0.0005),
            "rounds": len(round_summaries),
        },
        "collective_budget": {
            "distill_epochs_per_round": cfg.get("training.distill_epochs_per_round", 1),
            "lr": cfg.get("training.distill_lr", 0.0005),
            "kd_weight": cfg.get("training.kd_weight", 0.5),
            "proto_weight": cfg.get("training.proto_weight", 0.5),
            "rounds": len(round_summaries),
        },
        "fairness_exception_reason": (
            "Local-supervised training budgets are aligned round-for-round between the collective "
            "and local-only-control arms. The one intentional, disclosed asymmetry is the "
            "collective arm's extra KD phase over the shared probe set — that is the mechanism "
            "under test, not an unaligned budget."
        ),
        "test_set_scope": (
            "both — local (own node's held-out test set) and global held-out (union of all 3 "
            "nodes' held-out test sets); test samples never enter any training set"
        ),
        "rounds_run": len(round_summaries),
        "cumulative_bytes_exchanged": cumulative_bytes,
        "cumulative_energy_kwh": cumulative_energy,
        "delta_g_formula": "Score(collective, round_N) - Score(local_only_control, round_N), per metric per node",
        "per_node_scores": {
            nid: {"collective": collective_evals[nid], "local_only_control": control_evals[nid]}
            for nid in collective_evals
        },
        "macro_avg_and_worst_node": {
            "macro_gain": gain["macro_gain"],
            "worst_node_gain": gain["worst_node_gain"],
        },
        "collaboration_gain_per_disease": _per_disease_gain_table(
            round0_baseline, final_scores, data.label_map.disease_classes
        ),
        "limitation_note": (
            "Cross-node disease-recognition gain above comes from probe-set logit distillation "
            "only, not prototype alignment — prototype exchange only reinforces the shared "
            "healthy class, since a node's proto_loss only runs over its own local batches."
        ),
    }


NODE_DISEASE_DEFAULT_FALLBACK = {
    "node_0": "Common_rust",
    "node_1": "Cercospora_leaf_spot_Gray_leaf_spot",
    "node_2": "Northern_Leaf_Blight",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/corn_mesh")
    parser.add_argument("--rounds", type=int, default=None, help="Number of knowledge-transfer rounds (1-5)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    rounds = args.rounds if args.rounds is not None else cfg.get("corn_mesh.rounds", 2)
    if not (1 <= rounds <= 5):
        parser.error(f"--rounds must be between 1 and 5, got {rounds}")

    stage1_dir = Path(args.output_dir)
    for node_id in ("node_0", "node_1", "node_2"):
        node_dir = node_output_dir(stage1_dir, node_id)
        if not (node_dir / "checkpoint.pt").exists() or not (stage1_dir / "classes.json").exists():
            raise FileNotFoundError(
                f"{node_dir / 'checkpoint.pt'} not found — run "
                f"'python -m src.validation.run_corn_pipeline' first."
            )

    data = prepare_corn_mesh_data(cfg)
    cross_node_idx = [i for n in data.per_node.values() for i in n["test_idx"]]

    print("=== round 0 baseline (stage 1 checkpoints, cross-node eval) ===")
    round0_baseline = evaluate_round0_baseline(data, stage1_dir, cross_node_idx)

    kt_dir = stage1_dir / "knowledge_transfer"
    kt_dir.mkdir(parents=True, exist_ok=True)
    (kt_dir / "round_0_baseline.json").write_text(json.dumps(round0_baseline, indent=2))

    nodes = {
        node_id: _load_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in ("node_0", "node_1", "node_2")
    }
    control_nodes = {
        node_id: _load_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in ("node_0", "node_1", "node_2")
    }
    probe_loader = DataLoader(make_subset(data.eval_base, data.probe_idx), batch_size=args.batch_size, shuffle=False)

    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", False),
        output_dir=kt_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}), cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125)
    )

    round_summaries = []
    for round_idx in range(1, rounds + 1):
        print(f"=== knowledge-transfer round {round_idx}/{rounds} ===")
        round_dir = kt_dir / f"round_{round_idx}"
        summary = run_round_with_io(
            round_idx, nodes, control_nodes, probe_loader, data, cross_node_idx, cfg, tracker, comm_estimator, round_dir
        )
        round_summaries.append(summary)

    kt_summary = build_knowledge_transfer_summary(cfg, round0_baseline, round_summaries, data)
    (kt_dir / "knowledge_transfer_summary.json").write_text(json.dumps(kt_summary, indent=2))
    print(f"Done. Knowledge-transfer outputs in {kt_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validation_run_knowledge_transfer.py -v`
Expected: PASS (9 tests total in this file)

- [ ] **Step 5: Run the full new test suite together**

Run: `pytest tests/test_validation_corn_mesh_dataset.py tests/test_validation_run_corn_pipeline.py tests/test_validation_run_knowledge_transfer.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the full existing test suite to confirm no regressions**

Run: `pytest tests/ -v`
Expected: PASS (all pre-existing tests plus the new ones)

- [ ] **Step 7: Commit**

```bash
git add src/validation/run_knowledge_transfer.py tests/test_validation_run_knowledge_transfer.py
git commit -m "feat: add round-0 baseline, Appendix A.1 summary, and CLI for knowledge transfer"
```

---

## Self-Review Notes

- **Spec coverage:** Crop/disease assignment (Task 1/3), no-duplicate healthy split (Task 2, tested directly), compact label space (Task 1/3), stage 1 recipe reuse (Task 4), `distill()`-only rounds with no `local_train()` on the KT arm (Task 5), fairness-aligned local-only control (Task 5/6/7), per-round ONNX + dual eval (Task 6), energy/bytes tracking (Task 5/6), round-0 cross-node baseline (Task 7), Appendix A.1 field names + per-disease gain table + limitation note (Task 7), `--rounds` validation (Task 7). The one spec item with no dedicated task is the `--rounds` outside `[1,5]` failing at "argument parsing" specifically — implemented via `parser.error()` (which calls `sys.exit(2)`), covered by Task 7's `test_main_rejects_rounds_outside_valid_range`.
- **Type consistency check:** `CornMeshData.per_node[node_id]` is `{"train_idx": list[int], "test_idx": list[int]}` everywhere it's used (Tasks 3, 4, 5, 6, 7). `export_and_evaluate`'s return shape (`{"local":..., "cross_node":...}`) matches what `_per_disease_gain_table` and `build_knowledge_transfer_summary` read (`s["collective"]["cross_node"]["summary"]`). `run_kt_round`'s return keys (`per_node_distill_loss`, `per_node_bytes_sent`, `total_bytes_exchanged`) match exactly what `run_round_with_io` reads off `kt_result`.
- **Fixed during review:** Task 7's `_load_node_from_checkpoint` initially referenced `CornDiseaseView` via an inline `__import__` instead of a top-level import — corrected in place: Task 7's import block now includes `CornDiseaseView` directly (`from src.validation.corn_mesh_dataset import CornDiseaseView, prepare_corn_mesh_data`), and `_load_node_from_checkpoint` calls it as a plain name. `CornMeshData` itself is only used as a type hint in Task 6/7 function signatures that already import it where needed (Task 6's block imports `CornMeshData` alongside `CornLabelMap`; Task 7 doesn't re-annotate it, so no separate import is required there).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-19-corn-disease-knowledge-transfer.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
