# Node-1 Validation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `src/validation/` pipeline that trains `mobilenet_v3_small` on node_1's PlantVillage crops (Corn, Potato, Soybean, Strawberry, Squash) with augmentation and a dedup-aware train/test split, exports it to ONNX, and evaluates the exported model against the held-out split into a `report.json` with per-image and aggregate accuracy — so we can tell whether these fixes close the gap between reported 80-99% accuracy and bad real-world predictions, before rolling out to the full sweep.

**Architecture:** Four pure-function-first modules (`node1_dataset.py`, `train_mobilenet.py`, `export_onnx.py`, `evaluate_onnx.py`) that only *reuse* the existing `src/data/plantvillage.py`, `src/models/factory.py`, and `src/config.py` (imported, never edited), plus a thin CLI (`run_pipeline.py`) that chains them. Each module's core logic is a small set of testable pure/near-pure functions; only the outermost orchestration functions touch disk/network/model weights.

**Tech Stack:** PyTorch + torchvision (training, reused `PlantVillageDataset`/`build_model`), `timm` (backbone, via `build_model`), ONNX + `onnxruntime` (export/inference), Pillow/numpy (hashing, preprocessing), `pytest` (tests) — all already dependencies of this repo, no new packages needed.

**Spec:** `docs/superpowers/specs/2026-08-18-node1-validation-pipeline-design.md`

## Global Constraints

- Do NOT modify `src/data/plantvillage.py`, `src/train.py`, `src/federated/node.py`, or `src/models/factory.py` — only import and call their existing public functions/classes.
- Scope for this pass: node_1 only (`Corn, Potato, Soybean, Strawberry, Squash`, from `config.yaml`'s `manual_node_crops`), `mobilenet_v3_small` only.
- Dedup-aware split threshold: 64-bit average-hash, Hamming distance <= 5 groups images as near-duplicates.
- Training recipe: 15 epochs, Adam with `weight_decay=1e-4`, `CosineAnnealingLR` from `lr=0.001`, disease-head cross-entropy weighted by inverse class frequency (crop head unweighted).
- `train_ds`/`eval_ds` are two separate `PlantVillageDataset` instances against the same root; only `train_ds.transform` is overwritten (post-construction, plain attribute assignment) with an augmented pipeline. `eval_ds.transform` stays the original clean `Resize -> ToTensor -> Normalize`.
- ONNX export: opset 17, static `(1, 3, image_size, image_size)` NCHW input, matching `scripts/export_for_pi.py`'s approach (no quantization needed for this validation pass).
- All new code lives under `src/validation/`; all new tests under `tests/test_validation_*.py` plus one new shared fixture in `tests/conftest.py`.

---

### Task 1: Package scaffold + perceptual hashing utilities

**Files:**
- Create: `src/validation/__init__.py`
- Create: `src/validation/hashing.py`
- Test: `tests/test_validation_hashing.py`

**Interfaces:**
- Produces: `average_hash(image: PIL.Image.Image, hash_size: int = 8) -> int`, `hamming_distance(a: int, b: int) -> int` — used by Task 2's dedup grouping.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_hashing.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.hashing import average_hash, hamming_distance


def _image_from_gray_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_identical_images_have_zero_hamming_distance():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img1 = _image_from_gray_array(arr)
    img2 = _image_from_gray_array(arr.copy())

    assert hamming_distance(average_hash(img1), average_hash(img2)) == 0


def test_slightly_brightened_image_is_a_near_duplicate():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    brightened = np.clip(arr.astype(int) + 10, 0, 255).astype(np.uint8)
    img1 = _image_from_gray_array(arr)
    img2 = _image_from_gray_array(brightened)

    assert hamming_distance(average_hash(img1), average_hash(img2)) <= 5


def test_unrelated_images_have_large_hamming_distance():
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    arr1 = rng1.randint(0, 255, size=(64, 64), dtype=np.uint8)
    arr2 = rng2.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img1 = _image_from_gray_array(arr1)
    img2 = _image_from_gray_array(arr2)

    assert hamming_distance(average_hash(img1), average_hash(img2)) > 5


def test_average_hash_is_64_bits_for_default_hash_size():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64), dtype=np.uint8)
    img = _image_from_gray_array(arr)

    assert 0 <= average_hash(img) < (1 << 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation'`

- [ ] **Step 3: Implement**

Create `src/validation/__init__.py` (empty file).

Create `src/validation/hashing.py`:

```python
"""64-bit average-hash perceptual hashing, used to group near-duplicate
PlantVillage images before splitting train/test (see node1_dataset.py) —
PlantVillage is known to contain near-identical shots of the same physical
leaf, which a plain random split would happily place on both sides.
"""

from __future__ import annotations

from PIL import Image


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for i, pixel in enumerate(pixels):
        if pixel > mean:
            bits |= 1 << i
    return bits


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_hashing.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/__init__.py src/validation/hashing.py tests/test_validation_hashing.py
git commit -m "feat: add perceptual-hash utilities for validation pipeline"
```

---

### Task 2: Shared test fixture + node_1 scoping and dedup-aware split

**Files:**
- Create: `src/validation/node1_dataset.py` (partial — scoping + split only)
- Modify: `tests/conftest.py` (add one new fixture)
- Test: `tests/test_validation_node1_dataset.py` (partial)

**Interfaces:**
- Consumes: `average_hash`, `hamming_distance` from Task 1 (`src/validation/hashing.py`); `load_full_dataset`, `partition_nodes`, `PlantVillageDataset` from `src/data/plantvillage.py` (reused, unmodified); `Config` from `src/config.py`.
- Produces: `get_node1_indices(dataset, cfg) -> list[int]`, `compute_image_hashes(dataset, indices) -> dict[int, int]`, `group_duplicates(indices, hashes, threshold=5) -> dict[int, int]`, `dedup_aware_split(indices, hashes, test_fraction, seed, threshold=5) -> tuple[list[int], list[int]]` — Task 3 builds on all four.

- [ ] **Step 1: Add the shared fixture**

Add to `tests/conftest.py` (append at the end of the file):

```python
@pytest.fixture
def node1_scoped_config(tmp_path):
    """A tiny synthetic PlantVillage-shaped dataset (Tomato + Potato, 8
    images each) with a 2-node manual split, used across the
    src/validation/ test suite. "node_1" here means Potato, mirroring how
    config.yaml's real manual_node_crops assigns node_1 = Corn/Potato/
    Soybean/Strawberry/Squash — the mechanism under test doesn't care
    which literal crop names are used, only that partitioning +
    dedup-aware splitting work correctly.
    """
    from src.config import Config

    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    classes = [
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
        "Potato___Early_blight",
        "Potato___healthy",
    ]
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(8):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")

    cfg = Config(
        {
            "data": {
                "root": str(root),
                "image_size": 32,
                "num_nodes": 2,
                "non_iid_strategy": "manual",
                "manual_node_crops": {"node_0": ["Tomato"], "node_1": ["Potato"]},
                "test_fraction": 0.25,
                "seed": 42,
            }
        }
    )
    return cfg, root
```

(`numpy as np`, `PIL.Image`, and `pytest` are already imported at the top of `tests/conftest.py` — no new imports needed there.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_validation_node1_dataset.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.node1_dataset import (
    compute_image_hashes,
    dedup_aware_split,
    get_node1_indices,
    group_duplicates,
)


def test_get_node1_indices_returns_only_the_configured_node1_crop(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)

    node1_indices = get_node1_indices(dataset, cfg)

    assert len(node1_indices) == 16  # 8 Potato___Early_blight + 8 Potato___healthy
    for idx in node1_indices:
        _, class_idx = dataset.base.samples[idx]
        crop_idx, _ = dataset.labels.class_to_crop_disease[class_idx]
        assert dataset.labels.crop_classes[crop_idx] == "Potato"


def test_compute_image_hashes_returns_one_hash_per_index(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    node1_indices = get_node1_indices(dataset, cfg)

    hashes = compute_image_hashes(dataset, node1_indices)

    assert set(hashes.keys()) == set(node1_indices)
    assert all(isinstance(h, int) for h in hashes.values())


def test_group_duplicates_merges_near_identical_hashes_and_separates_distinct_ones():
    hashes = {0: 0b0000_0000, 1: 0b0000_0001, 2: 0b1111_1111, 3: 0b1111_1110}

    groups = group_duplicates([0, 1, 2, 3], hashes, threshold=1)

    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


def test_dedup_aware_split_keeps_duplicate_groups_on_one_side():
    indices = [0, 1, 2, 3, 4, 5]
    # (0, 1) near-duplicates; (2, 3) near-duplicates; 4 and 5 are singletons far from everything.
    hashes = {0: 0, 1: 1, 2: 0b1111_0000, 3: 0b1111_0001, 4: 0b0101_0101, 5: 0b1010_1010}

    train_idx, test_idx = dedup_aware_split(indices, hashes, test_fraction=0.34, seed=0, threshold=1)

    train_set, test_set = set(train_idx), set(test_idx)
    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(indices)
    assert (0 in train_set) == (1 in train_set)
    assert (2 in train_set) == (3 in train_set)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_node1_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (no `node1_dataset.py` yet)

- [ ] **Step 4: Implement**

Create `src/validation/node1_dataset.py`:

```python
"""Scopes the full PlantVillage dataset down to node_1's crops (per
config.yaml's manual_node_crops) and splits it into a train/eval pair using
a dedup-aware split, so near-duplicate PlantVillage shots (the same
physical leaf photographed more than once) can't leak across the split.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from src.config import Config
from src.data.plantvillage import PlantVillageDataset, partition_nodes
from src.validation.hashing import average_hash, hamming_distance

# Position of "node_1" in partition_nodes()'s output list — node keys in
# config.yaml's manual_node_crops are literally "node_0", "node_1", "node_2",
# and _manual_partition() (src/data/plantvillage.py) maps them to this same
# integer position regardless of dict iteration order.
NODE1_INDEX = 1


def get_node1_indices(dataset: PlantVillageDataset, cfg: Config) -> list[int]:
    """Node_1's raw sample indices, via the same partition_nodes() the real
    mesh pipeline uses — no probe-set carve-out here (this is a local-only
    training run, not mesh/distillation), so every sample index is passed
    in as "remaining".
    """
    all_indices = list(range(len(dataset)))
    shards = partition_nodes(
        dataset,
        all_indices,
        cfg.get("data.num_nodes", 3),
        "manual",
        cfg.get("data.dirichlet_alpha", 0.3),
        cfg.get("data.seed", 42),
        manual_node_crops=cfg.get("data.manual_node_crops", None),
    )
    return shards[NODE1_INDEX]


def compute_image_hashes(dataset: PlantVillageDataset, indices: list[int]) -> dict[int, int]:
    """Perceptual hash per sample index (index is the same one used by
    dataset.base.samples / dataset.base.targets everywhere else).
    """
    hashes: dict[int, int] = {}
    for idx in indices:
        path, _ = dataset.base.samples[idx]
        with Image.open(path) as img:
            hashes[idx] = average_hash(img.convert("RGB"))
    return hashes


def group_duplicates(indices: list[int], hashes: dict[int, int], threshold: int = 5) -> dict[int, int]:
    """Union-find over `indices`: any two whose hashes are within Hamming
    distance <= threshold land in the same group. Returns index -> group_id
    (group_id is one representative index per group).

    O(n^2) pairwise comparisons — fine for node_1's few-thousand-image
    scope, not intended for the full dataset.
    """
    parent = {idx: idx for idx in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(indices):
        for b in indices[i + 1 :]:
            if hamming_distance(hashes[a], hashes[b]) <= threshold:
                union(a, b)

    return {idx: find(idx) for idx in indices}


def dedup_aware_split(
    indices: list[int],
    hashes: dict[int, int],
    test_fraction: float,
    seed: int,
    threshold: int = 5,
) -> tuple[list[int], list[int]]:
    """Like the existing train_test_split_indices, but splits at the
    duplicate-group level (see group_duplicates) instead of the raw index
    level.
    """
    groups = group_duplicates(indices, hashes, threshold)
    group_members: dict[int, list[int]] = {}
    for idx, group_id in groups.items():
        group_members.setdefault(group_id, []).append(idx)

    group_ids = list(group_members.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_test_target = max(1, int(len(indices) * test_fraction)) if len(indices) > 1 else 0
    train_idx: list[int] = []
    test_idx: list[int] = []
    for group_id in group_ids:
        members = group_members[group_id]
        if len(test_idx) < n_test_target:
            test_idx.extend(members)
        else:
            train_idx.extend(members)
    return train_idx, test_idx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_node1_dataset.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/validation/node1_dataset.py tests/test_validation_node1_dataset.py tests/conftest.py
git commit -m "feat: add node_1 scoping and dedup-aware train/test split"
```

---

### Task 3: Augmented train transform + end-to-end node_1 data prep

**Files:**
- Modify: `src/validation/node1_dataset.py` (append)
- Test: `tests/test_validation_node1_dataset.py` (append)

**Interfaces:**
- Consumes: everything from Task 2, plus `IMAGENET_MEAN`/`IMAGENET_STD` from `src/data/plantvillage.py` (reused, unmodified).
- Produces: `build_train_eval_datasets(root, image_size) -> tuple[PlantVillageDataset, PlantVillageDataset]`, `prepare_node1_data(cfg) -> tuple[PlantVillageDataset, list[int], PlantVillageDataset, list[int]]` (train_ds, train_idx, eval_ds, test_idx) — Tasks 5 and 8 consume this tuple shape directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validation_node1_dataset.py`:

```python
from torchvision import transforms

from src.validation.node1_dataset import build_train_eval_datasets, prepare_node1_data


def test_build_train_eval_datasets_augments_train_only(node1_scoped_config):
    cfg, root = node1_scoped_config

    train_ds, eval_ds = build_train_eval_datasets(root, image_size=32)

    train_types = [type(t) for t in train_ds.transform.transforms]
    eval_types = [type(t) for t in eval_ds.transform.transforms]

    assert transforms.RandomHorizontalFlip in train_types
    assert transforms.ColorJitter in train_types
    assert eval_types == [transforms.Resize, transforms.ToTensor, transforms.Normalize]


def test_prepare_node1_data_returns_disjoint_indices_and_full_label_space(node1_scoped_config):
    cfg, root = node1_scoped_config

    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(train_idx) + len(test_idx) == 16
    # both crops are in the label space even though only Potato samples were
    # selected for node_1 -- the model must predict among the full global
    # label set, matching how the real pipeline sizes its heads.
    assert set(eval_ds.labels.crop_classes) == {"Tomato", "Potato"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_node1_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_train_eval_datasets'`

- [ ] **Step 3: Implement**

Append to `src/validation/node1_dataset.py`:

```python
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, load_full_dataset


def build_train_eval_datasets(
    root: str | Path, image_size: int
) -> tuple[PlantVillageDataset, PlantVillageDataset]:
    """Two PlantVillageDataset instances against the same root, so they
    share identical ImageFolder ordering/labels. `train_ds.transform` is
    overwritten (plain attribute assignment, no class edit) with an
    augmented pipeline; `eval_ds` keeps the original clean transform.
    """
    train_ds = PlantVillageDataset(root, image_size=image_size)
    eval_ds = PlantVillageDataset(root, image_size=image_size)
    train_ds.transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_ds, eval_ds


def prepare_node1_data(
    cfg: Config,
) -> tuple[PlantVillageDataset, list[int], PlantVillageDataset, list[int]]:
    """End-to-end: load the full dataset, scope to node_1, dedup-aware
    split. Returns (train_ds, train_idx, eval_ds, test_idx).
    """
    image_size = cfg.get("data.image_size", 160)
    root = cfg.get("data.root", "data/PlantVillage")
    dataset = load_full_dataset(root, image_size)
    node1_indices = get_node1_indices(dataset, cfg)
    hashes = compute_image_hashes(dataset, node1_indices)
    train_idx, test_idx = dedup_aware_split(
        node1_indices, hashes, cfg.get("data.test_fraction", 0.15), cfg.get("data.seed", 42)
    )
    train_ds, eval_ds = build_train_eval_datasets(root, image_size)
    return train_ds, train_idx, eval_ds, test_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_node1_dataset.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/node1_dataset.py tests/test_validation_node1_dataset.py
git commit -m "feat: add augmented train transform and end-to-end node_1 data prep"
```

---

### Task 4: Disease class-weighting helpers

**Files:**
- Create: `src/validation/train_mobilenet.py` (partial — weighting helpers only)
- Test: `tests/test_validation_train_mobilenet.py` (partial)

**Interfaces:**
- Consumes: `PlantVillageDataset` (for typing only).
- Produces: `disease_labels_for_indices(dataset, indices) -> list[int]`, `compute_class_weights(labels, num_classes) -> torch.Tensor` — Task 5's `run_training` uses both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_train_mobilenet.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import PlantVillageDataset
from src.validation.node1_dataset import get_node1_indices
from src.validation.train_mobilenet import compute_class_weights, disease_labels_for_indices


def test_compute_class_weights_upweights_rare_classes_and_zeros_absent_ones():
    weights = compute_class_weights(labels=[0, 0, 0, 1], num_classes=3)

    assert weights[2] == pytest.approx(0.0)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(1.5)
    assert weights[1] > weights[0] > 0


def test_disease_labels_for_indices_matches_label_maps(node1_scoped_config):
    cfg, root = node1_scoped_config
    dataset = PlantVillageDataset(root, image_size=32)
    node1_indices = get_node1_indices(dataset, cfg)

    disease_labels = disease_labels_for_indices(dataset, node1_indices)

    assert len(disease_labels) == len(node1_indices)
    for idx, disease_label in zip(node1_indices, disease_labels):
        class_idx = dataset.base.targets[idx]
        _, expected_disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        assert disease_label == expected_disease_idx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_train_mobilenet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.train_mobilenet'`

- [ ] **Step 3: Implement**

Create `src/validation/train_mobilenet.py`:

```python
"""Trains mobilenet_v3_small on node_1's dedup-aware split (see
node1_dataset.py) with an improved recipe over config.yaml's current
defaults: more epochs, weight decay, a cosine LR schedule, and disease-head
class weighting -- see docs/superpowers/specs/2026-08-18-
node1-validation-pipeline-design.md for why each of these was added.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.plantvillage import PlantVillageDataset
from src.models.factory import build_model


def disease_labels_for_indices(dataset: PlantVillageDataset, indices: list[int]) -> list[int]:
    return [dataset.labels.class_to_crop_disease[dataset.base.targets[idx]][1] for idx in indices]


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, rescaled so the mean weight across
    *present* classes is 1.0 (keeps the loss magnitude comparable to
    unweighted cross-entropy). Classes absent from `labels` get weight 0 --
    node_1 only ever has samples for the diseases of its own crops, but the
    disease head still spans the full global disease label space.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if present.any():
        weights[present] = weights[present] * (present.sum() / weights[present].sum())
    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_train_mobilenet.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/train_mobilenet.py tests/test_validation_train_mobilenet.py
git commit -m "feat: add disease class-weighting helpers for validation training"
```

---

### Task 5: Training loop with checkpoint selection

**Files:**
- Modify: `src/validation/train_mobilenet.py` (append)
- Test: `tests/test_validation_train_mobilenet.py` (append)

**Interfaces:**
- Consumes: `build_model` from `src/models/factory.py` (reused, unmodified); `disease_labels_for_indices`, `compute_class_weights` from Task 4; the `(train_ds, train_idx, eval_ds, test_idx)` tuple shape from Task 3.
- Produces: `train_one_epoch(model, loader, optimizer, class_weights, device) -> float`, `evaluate(model, loader, device) -> dict[str, float]`, `run_training(train_ds, train_idx, eval_ds, test_idx, num_crop_classes, num_disease_classes, output_dir, epochs=15, batch_size=32, lr=0.001, weight_decay=1e-4, pretrained=True, device="cpu") -> dict` — Task 8's `run_train_stage` calls `run_training` directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_validation_train_mobilenet.py`:

```python
import json

from src.validation.node1_dataset import prepare_node1_data
from src.validation.train_mobilenet import run_training


def test_run_training_writes_log_and_best_checkpoint(node1_scoped_config, tmp_path):
    cfg, root = node1_scoped_config
    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)
    num_crop = len(eval_ds.labels.crop_classes)
    num_disease = len(eval_ds.labels.disease_classes)
    output_dir = tmp_path / "run"

    result = run_training(
        train_ds,
        train_idx,
        eval_ds,
        test_idx,
        num_crop,
        num_disease,
        output_dir,
        epochs=2,
        batch_size=4,
        pretrained=False,
        device="cpu",
    )

    assert (output_dir / "checkpoint.pt").exists()
    log = json.loads((output_dir / "training_log.json").read_text())
    assert len(log) == 2
    assert set(log[0].keys()) == {"epoch", "train_loss", "test_crop_accuracy", "test_disease_accuracy"}
    assert result["best_disease_accuracy"] == max(e["test_disease_accuracy"] for e in log)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_validation_train_mobilenet.py::test_run_training_writes_log_and_best_checkpoint -v`
Expected: FAIL with `ImportError: cannot import name 'run_training'`

- [ ] **Step 3: Implement**

Append to `src/validation/train_mobilenet.py`:

```python
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    disease_class_weights: torch.Tensor,
    device: str,
) -> float:
    model.train()
    total_loss, total_batches = 0.0, 0
    weights = disease_class_weights.to(device)
    for images, crop_labels, disease_labels in loader:
        images = images.to(device)
        crop_labels = crop_labels.to(device)
        disease_labels = disease_labels.to(device)

        optimizer.zero_grad()
        crop_logits, disease_logits = model(images)
        loss = F.cross_entropy(crop_logits, crop_labels) + F.cross_entropy(
            disease_logits, disease_labels, weight=weights
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct_crop, correct_disease, total = 0, 0, 0
    for images, crop_labels, disease_labels in loader:
        images = images.to(device)
        crop_labels = crop_labels.to(device)
        disease_labels = disease_labels.to(device)
        crop_logits, disease_logits = model(images)
        correct_crop += (crop_logits.argmax(dim=1) == crop_labels).sum().item()
        correct_disease += (disease_logits.argmax(dim=1) == disease_labels).sum().item()
        total += images.shape[0]
    total = max(1, total)
    return {"crop_accuracy": correct_crop / total, "disease_accuracy": correct_disease / total}


def run_training(
    train_ds: PlantVillageDataset,
    train_idx: list[int],
    eval_ds: PlantVillageDataset,
    test_idx: list[int],
    num_crop_classes: int,
    num_disease_classes: int,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    pretrained: bool = True,
    device: str = "cpu",
) -> dict:
    disease_labels = disease_labels_for_indices(train_ds, train_idx)
    class_weights = compute_class_weights(disease_labels, num_disease_classes)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(eval_ds, test_idx), batch_size=batch_size, shuffle=False)

    model = build_model("mobilenet_v3_small", num_crop_classes, num_disease_classes, pretrained=pretrained).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_disease_accuracy = -1.0
    log_entries = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, class_weights, device)
        scheduler.step()
        eval_metrics = evaluate(model, eval_loader, device)
        log_entries.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_crop_accuracy": eval_metrics["crop_accuracy"],
                "test_disease_accuracy": eval_metrics["disease_accuracy"],
            }
        )
        if eval_metrics["disease_accuracy"] > best_disease_accuracy:
            best_disease_accuracy = eval_metrics["disease_accuracy"]
            torch.save(model.state_dict(), output_dir / "checkpoint.pt")

    (output_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2))
    return {"log_entries": log_entries, "best_disease_accuracy": best_disease_accuracy}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_validation_train_mobilenet.py -v`
Expected: PASS (3 tests). This test actually trains 2 tiny epochs on 16 32x32 images with `pretrained=False`, so it should finish in a few seconds with no network access needed.

- [ ] **Step 5: Commit**

```bash
git add src/validation/train_mobilenet.py tests/test_validation_train_mobilenet.py
git commit -m "feat: add mobilenet_v3_small training loop with checkpoint selection"
```

---

### Task 6: ONNX export + PyTorch/ONNX parity check

**Files:**
- Create: `src/validation/export_onnx.py`
- Test: `tests/test_validation_export_onnx.py`

**Interfaces:**
- Consumes: `build_model` from `src/models/factory.py` (reused, unmodified); `IMAGENET_MEAN`/`IMAGENET_STD` from `src/data/plantvillage.py` (reused, unmodified).
- Produces: `export_checkpoint(checkpoint_path, crop_classes, disease_classes, image_size, output_dir, parity_samples=None) -> Path` (writes `model.onnx` + `manifest.json`, runs `check_parity` and prints a warning if `parity_samples` is given, returns the ONNX path), `check_parity(model, session, samples) -> int` — Task 8's `run_export_stage` calls `export_checkpoint` with `parity_samples` built from the held-out split.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_export_onnx.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import onnxruntime
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.factory import build_model
from src.validation.export_onnx import check_parity, export_checkpoint

CROP_CLASSES = ["Corn", "Potato"]
DISEASE_CLASSES = ["healthy", "rust"]
IMAGE_SIZE = 32


def _save_tiny_checkpoint(tmp_path) -> Path:
    model = build_model("mobilenet_v3_small", len(CROP_CLASSES), len(DISEASE_CLASSES), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path, model


def test_export_checkpoint_writes_onnx_and_manifest(tmp_path):
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path)
    output_dir = tmp_path / "export"

    onnx_path = export_checkpoint(checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir)

    assert onnx_path == output_dir / "model.onnx"
    assert onnx_path.exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["crop_classes"] == CROP_CLASSES
    assert manifest["disease_classes"] == DISEASE_CLASSES
    assert manifest["image_size"] == IMAGE_SIZE
    assert len(manifest["mean"]) == 3 and len(manifest["std"]) == 3

    session = onnxruntime.InferenceSession(str(onnx_path))
    outputs = session.get_outputs()
    assert outputs[0].shape[-1] == len(CROP_CLASSES)
    assert outputs[1].shape[-1] == len(DISEASE_CLASSES)


def test_check_parity_finds_zero_mismatches_for_the_exported_weights(tmp_path):
    checkpoint_path, model = _save_tiny_checkpoint(tmp_path)
    model.eval()
    output_dir = tmp_path / "export"

    onnx_path = export_checkpoint(checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir)
    session = onnxruntime.InferenceSession(str(onnx_path))

    samples = [(torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), 0, 0) for _ in range(4)]
    mismatches = check_parity(model, session, samples)

    assert mismatches == 0


def test_export_checkpoint_runs_parity_check_when_samples_given(tmp_path, capsys):
    checkpoint_path, _ = _save_tiny_checkpoint(tmp_path)
    output_dir = tmp_path / "export"
    parity_samples = [(torch.rand(3, IMAGE_SIZE, IMAGE_SIZE), 0, 0) for _ in range(3)]

    onnx_path = export_checkpoint(
        checkpoint_path, CROP_CLASSES, DISEASE_CLASSES, IMAGE_SIZE, output_dir, parity_samples=parity_samples
    )

    assert onnx_path.exists()
    captured = capsys.readouterr()
    assert "Parity OK" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_export_onnx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.export_onnx'`

- [ ] **Step 3: Implement**

Create `src/validation/export_onnx.py`:

```python
"""Exports a trained mobilenet_v3_small checkpoint from train_mobilenet.py
to ONNX + a manifest.json, and spot-checks PyTorch-vs-ONNX prediction
parity -- same shape/approach as scripts/export_for_pi.py, reimplemented
here as a self-contained validation-pipeline step (no quantization, since
this pass is about accuracy correctness, not on-device footprint).
"""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime
import torch

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD
from src.models.factory import build_model


def export_onnx(model: torch.nn.Module, image_size: int, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["crop_logits", "disease_logits"],
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


def check_parity(model: torch.nn.Module, session: onnxruntime.InferenceSession, samples) -> int:
    """Compares PyTorch vs. ONNX top-1 predictions on a few samples
    (image_tensor, crop_label, disease_label). Returns mismatch count.
    """
    input_name = session.get_inputs()[0].name
    mismatches = 0
    with torch.no_grad():
        for image, _, _ in samples:
            batch = image.unsqueeze(0)
            torch_crop, torch_disease = model(batch)
            onnx_crop, onnx_disease = session.run(None, {input_name: batch.numpy()})
            if torch_crop.argmax(1).item() != onnx_crop.argmax(1).item():
                mismatches += 1
            elif torch_disease.argmax(1).item() != onnx_disease.argmax(1).item():
                mismatches += 1
    return mismatches


def export_checkpoint(
    checkpoint_path: Path,
    crop_classes: list[str],
    disease_classes: list[str],
    image_size: int,
    output_dir: Path,
    parity_samples: list | None = None,
) -> Path:
    """Builds a fresh mobilenet_v3_small, loads `checkpoint_path`'s
    weights, exports to ONNX, writes manifest.json. Returns the ONNX path.

    If `parity_samples` (a list of (image_tensor, crop_label,
    disease_label) tuples) is given, also runs check_parity against the
    exported ONNX model and prints a warning on any mismatch — same idea
    as scripts/export_for_pi.py's check_parity, reimplemented here so an
    export bug is caught before evaluate_onnx.py blames the training
    itself.
    """
    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    export_onnx(model, image_size, onnx_path)

    manifest = {
        "crop_classes": crop_classes,
        "disease_classes": disease_classes,
        "image_size": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if parity_samples:
        session = onnxruntime.InferenceSession(str(onnx_path))
        mismatches = check_parity(model, session, parity_samples)
        if mismatches:
            print(
                f"WARNING: {mismatches}/{len(parity_samples)} samples disagree between the "
                f"PyTorch checkpoint and the exported ONNX model — inspect before trusting "
                f"evaluate_onnx.py's report.json."
            )
        else:
            print(f"Parity OK: all {len(parity_samples)} sampled predictions match.")

    return onnx_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_export_onnx.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/export_onnx.py tests/test_validation_export_onnx.py
git commit -m "feat: add ONNX export and PyTorch/ONNX parity check"
```

---

### Task 7: ONNX evaluation + report builder

**Files:**
- Create: `src/validation/evaluate_onnx.py`
- Test: `tests/test_validation_evaluate_onnx.py`

**Interfaces:**
- Consumes: `export_checkpoint` from Task 6 (test-only, to get a real ONNX session to evaluate against); `PlantVillageDataset` from `src/data/plantvillage.py` (reused, unmodified) for image/label lookups.
- Produces: `preprocess_image(image, image_size, mean, std) -> np.ndarray`, `predict_onnx(session, image_array) -> tuple[int, float, int, float]` (crop_idx, crop_confidence, disease_idx, disease_confidence), `build_report(records, model_name, node, top_n_confusions=10) -> dict`, `run_evaluation(session, eval_ds, test_idx, manifest, model_name, node, output_path) -> dict` — Task 8's `run_evaluate_stage` calls `run_evaluation` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_evaluate_onnx.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD, PlantVillageDataset
from src.models.factory import build_model
from src.validation.evaluate_onnx import build_report, predict_onnx, preprocess_image, run_evaluation
from src.validation.export_onnx import export_checkpoint


def test_preprocess_image_produces_expected_shape_and_dtype():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    result = preprocess_image(img, image_size=32, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    assert result.shape == (1, 3, 32, 32)
    assert result.dtype == np.float32


def test_predict_onnx_returns_valid_index_and_confidence(tmp_path):
    crop_classes = ["Corn", "Potato"]
    disease_classes = ["healthy", "rust"]
    image_size = 32

    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    onnx_path = export_checkpoint(checkpoint_path, crop_classes, disease_classes, image_size, tmp_path / "export")
    session = onnxruntime.InferenceSession(str(onnx_path))

    image_array = np.random.rand(1, 3, image_size, image_size).astype(np.float32)
    crop_idx, crop_conf, disease_idx, disease_conf = predict_onnx(session, image_array)

    assert crop_idx in (0, 1)
    assert disease_idx in (0, 1)
    assert 0.0 <= crop_conf <= 1.0
    assert 0.0 <= disease_conf <= 1.0


def test_build_report_computes_accuracy_per_class_and_top_confusions():
    records = [
        {
            "expected_crop": "Corn", "expected_disease": "healthy",
            "predicted_crop": "Corn", "predicted_disease": "healthy",
            "crop_correct": True, "disease_correct": True,
        },
        {
            "expected_crop": "Corn", "expected_disease": "rust",
            "predicted_crop": "Corn", "predicted_disease": "blight",
            "crop_correct": True, "disease_correct": False,
        },
        {
            "expected_crop": "Potato", "expected_disease": "rust",
            "predicted_crop": "Potato", "predicted_disease": "blight",
            "crop_correct": True, "disease_correct": False,
        },
        {
            "expected_crop": "Potato", "expected_disease": "healthy",
            "predicted_crop": "Corn", "predicted_disease": "healthy",
            "crop_correct": False, "disease_correct": True,
        },
    ]

    summary = build_report(records, model_name="mobilenet_v3_small", node="node_1")

    assert summary["num_test_images"] == 4
    assert summary["crop_accuracy"] == pytest.approx(0.75)
    assert summary["disease_accuracy"] == pytest.approx(0.5)
    assert summary["per_class_accuracy"]["disease"]["rust"] == pytest.approx(0.0)
    assert summary["top_confusions"][0] == {"expected": "rust", "predicted": "blight", "count": 2}


def test_run_evaluation_writes_report_matching_test_idx_length(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "PlantVillage"
    for cls in ["Corn___healthy", "Corn___rust"]:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    eval_ds = PlantVillageDataset(root, image_size=32)
    test_idx = list(range(len(eval_ds)))

    crop_classes = eval_ds.labels.crop_classes
    disease_classes = eval_ds.labels.disease_classes
    model = build_model("mobilenet_v3_small", len(crop_classes), len(disease_classes), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    onnx_path = export_checkpoint(checkpoint_path, crop_classes, disease_classes, 32, tmp_path / "export")
    manifest = json.loads((tmp_path / "export" / "manifest.json").read_text())
    session = onnxruntime.InferenceSession(str(onnx_path))

    report = run_evaluation(
        session, eval_ds, test_idx, manifest, "mobilenet_v3_small", "node_1", tmp_path / "report.json"
    )

    assert (tmp_path / "report.json").exists()
    assert report["summary"]["num_test_images"] == len(test_idx)
    assert len(report["results"]) == len(test_idx)
    for record in report["results"]:
        assert set(record.keys()) == {
            "filename", "expected_crop", "expected_disease", "predicted_crop", "predicted_disease",
            "crop_confidence", "disease_confidence", "crop_correct", "disease_correct",
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_evaluate_onnx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.evaluate_onnx'`

- [ ] **Step 3: Implement**

Create `src/validation/evaluate_onnx.py`:

```python
"""Runs the exported ONNX model (see export_onnx.py) over a held-out
split, using the exact preprocessing already verified identical to
apps/crop_disease_detection/inference.py and pi/inference_service.py, and
writes a per-image + aggregate-summary report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from src.data.plantvillage import PlantVillageDataset


def preprocess_image(image: Image.Image, image_size: int, mean: list[float], std: list[float]) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_onnx(session: onnxruntime.InferenceSession, image_array: np.ndarray) -> tuple[int, float, int, float]:
    input_name = session.get_inputs()[0].name
    crop_logits, disease_logits = session.run(None, {input_name: image_array})
    crop_probs = _softmax(crop_logits)[0]
    disease_probs = _softmax(disease_logits)[0]
    crop_idx = int(crop_probs.argmax())
    disease_idx = int(disease_probs.argmax())
    return crop_idx, float(crop_probs[crop_idx]), disease_idx, float(disease_probs[disease_idx])


def build_report(
    records: list[dict], model_name: str, node: str, top_n_confusions: int = 10
) -> dict:
    """`records` already contain filename/expected_*/predicted_*/
    *_confidence/*_correct keys (see run_evaluation) -- this only computes
    the aggregate summary block.
    """
    num_images = len(records)
    crop_correct = sum(1 for r in records if r["crop_correct"])
    disease_correct = sum(1 for r in records if r["disease_correct"])

    per_class_crop: dict[str, list[int]] = {}
    per_class_disease: dict[str, list[int]] = {}
    confusion_counts: dict[tuple[str, str], int] = {}
    for r in records:
        crop_bucket = per_class_crop.setdefault(r["expected_crop"], [0, 0])
        crop_bucket[0] += int(r["crop_correct"])
        crop_bucket[1] += 1

        disease_bucket = per_class_disease.setdefault(r["expected_disease"], [0, 0])
        disease_bucket[0] += int(r["disease_correct"])
        disease_bucket[1] += 1

        if not r["disease_correct"]:
            key = (r["expected_disease"], r["predicted_disease"])
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

    top_confusions = [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in sorted(
            confusion_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:top_n_confusions]
    ]

    return {
        "model": model_name,
        "node": node,
        "num_test_images": num_images,
        "crop_accuracy": crop_correct / max(1, num_images),
        "disease_accuracy": disease_correct / max(1, num_images),
        "per_class_accuracy": {
            "crop": {name: correct / total for name, (correct, total) in per_class_crop.items()},
            "disease": {name: correct / total for name, (correct, total) in per_class_disease.items()},
        },
        "top_confusions": top_confusions,
    }


def run_evaluation(
    session: onnxruntime.InferenceSession,
    eval_ds: PlantVillageDataset,
    test_idx: list[int],
    manifest: dict,
    model_name: str,
    node: str,
    output_path: Path,
) -> dict:
    records = []
    for idx in test_idx:
        path, class_idx = eval_ds.base.samples[idx]
        crop_gt_idx, disease_gt_idx = eval_ds.labels.class_to_crop_disease[class_idx]
        expected_crop = eval_ds.labels.crop_classes[crop_gt_idx]
        expected_disease = eval_ds.labels.disease_classes[disease_gt_idx]

        with Image.open(path) as img:
            image_array = preprocess_image(img, manifest["image_size"], manifest["mean"], manifest["std"])
        crop_idx, crop_conf, disease_idx, disease_conf = predict_onnx(session, image_array)
        predicted_crop = manifest["crop_classes"][crop_idx]
        predicted_disease = manifest["disease_classes"][disease_idx]

        records.append(
            {
                "filename": str(path),
                "expected_crop": expected_crop,
                "expected_disease": expected_disease,
                "predicted_crop": predicted_crop,
                "predicted_disease": predicted_disease,
                "crop_confidence": crop_conf,
                "disease_confidence": disease_conf,
                "crop_correct": predicted_crop == expected_crop,
                "disease_correct": predicted_disease == expected_disease,
            }
        )

    summary = build_report(records, model_name, node)
    report = {"summary": summary, "results": records}
    output_path.write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_evaluate_onnx.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/evaluate_onnx.py tests/test_validation_evaluate_onnx.py
git commit -m "feat: add ONNX evaluation and JSON report builder"
```

---

### Task 8: CLI pipeline entrypoint chaining all stages

**Files:**
- Create: `src/validation/run_pipeline.py`
- Test: `tests/test_validation_run_pipeline.py`

**Interfaces:**
- Consumes: `prepare_node1_data` (Task 3), `run_training` (Task 5), `export_checkpoint` (Task 6), `run_evaluation` (Task 7); `PlantVillageDataset` from `src/data/plantvillage.py` (reused, unmodified); `Config` from `src/config.py` (reused, unmodified).
- Produces: `run_train_stage(cfg, output_dir, epochs=15, pretrained=True) -> None`, `run_export_stage(cfg, output_dir) -> None`, `run_evaluate_stage(cfg, output_dir) -> None`, and a `main()` CLI entrypoint. No later task consumes these — this is the outermost layer.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation_run_pipeline.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.run_pipeline import run_evaluate_stage, run_export_stage, run_train_stage


def test_run_export_stage_raises_clear_error_when_train_stage_not_run(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="train"):
        run_export_stage(cfg, output_dir)


def test_run_evaluate_stage_raises_clear_error_when_export_stage_not_run(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="export"):
        run_evaluate_stage(cfg, output_dir)


def test_full_pipeline_produces_a_well_formed_report(tmp_path, node1_scoped_config):
    cfg, _ = node1_scoped_config
    output_dir = tmp_path / "run"

    run_train_stage(cfg, output_dir, epochs=2, pretrained=False)
    run_export_stage(cfg, output_dir)
    run_evaluate_stage(cfg, output_dir)

    report = json.loads((output_dir / "report.json").read_text())
    assert "summary" in report and "results" in report
    assert report["summary"]["num_test_images"] == len(report["results"])
    assert report["summary"]["num_test_images"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validation_run_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.run_pipeline'`

- [ ] **Step 3: Implement**

Create `src/validation/run_pipeline.py`:

```python
"""CLI entrypoint chaining the node_1 / mobilenet_v3_small validation
pipeline: train -> export -> evaluate. Each stage reads/writes files under
one output directory so stages can be re-run independently.

    python -m src.validation.run_pipeline
    python -m src.validation.run_pipeline --stage train
    python -m src.validation.run_pipeline --stage export
    python -m src.validation.run_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.data.plantvillage import PlantVillageDataset
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.node1_dataset import prepare_node1_data
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"
NODE_NAME = "node_1"


def run_train_stage(cfg: Config, output_dir: Path, epochs: int = 15, pretrained: bool = True) -> None:
    train_ds, train_idx, eval_ds, test_idx = prepare_node1_data(cfg)
    num_crop = len(eval_ds.labels.crop_classes)
    num_disease = len(eval_ds.labels.disease_classes)

    run_training(
        train_ds, train_idx, eval_ds, test_idx, num_crop, num_disease, output_dir,
        epochs=epochs, pretrained=pretrained,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": eval_ds.labels.crop_classes,
                "disease_classes": eval_ds.labels.disease_classes,
                "image_size": cfg.get("data.image_size", 160),
                "test_idx": test_idx,
            },
            indent=2,
        )
    )


def run_export_stage(cfg: Config, output_dir: Path, num_parity_samples: int = 8) -> None:
    checkpoint_path = output_dir / "checkpoint.pt"
    classes_path = output_dir / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} and/or {classes_path} not found — run the 'train' stage first."
        )
    classes = json.loads(classes_path.read_text())
    image_size = classes["image_size"]
    test_idx = classes["test_idx"]

    # Parity samples come from the same held-out split the train stage
    # already resolved (classes.json's "test_idx"), through the plain
    # unmodified PlantVillageDataset (its __getitem__ applies the clean
    # eval transform) -- no need to redo prepare_node1_data's dedup split.
    eval_ds = PlantVillageDataset(cfg.get("data.root", "data/PlantVillage"), image_size=image_size)
    sample_idx = test_idx[: min(num_parity_samples, len(test_idx))]
    parity_samples = [eval_ds[idx] for idx in sample_idx]

    export_checkpoint(
        checkpoint_path,
        classes["crop_classes"],
        classes["disease_classes"],
        image_size,
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(cfg: Config, output_dir: Path) -> None:
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

    # Reads the stored split rather than recomputing prepare_node1_data's
    # dedup-aware split (which is O(n^2) in node_1's shard size) -- the
    # train stage already resolved train_idx/test_idx once.
    eval_ds = PlantVillageDataset(cfg.get("data.root", "data/PlantVillage"), image_size=classes["image_size"])
    session = onnxruntime.InferenceSession(str(onnx_path))
    run_evaluation(session, eval_ds, test_idx, manifest, MODEL_NAME, NODE_NAME, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/node_1_mobilenet_v3_small")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)

    for stage in ([args.stage] if args.stage else list(STAGES)):
        print(f"=== stage: {stage} ===")
        if stage == "train":
            run_train_stage(cfg, output_dir)
        elif stage == "export":
            run_export_stage(cfg, output_dir)
        elif stage == "evaluate":
            run_evaluate_stage(cfg, output_dir)

    print(f"Done. Outputs in {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validation_run_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full validation test suite together**

Run: `python -m pytest tests/test_validation_hashing.py tests/test_validation_node1_dataset.py tests/test_validation_train_mobilenet.py tests/test_validation_export_onnx.py tests/test_validation_evaluate_onnx.py tests/test_validation_run_pipeline.py -v`
Expected: PASS (all tests across every module in this plan)

- [ ] **Step 6: Commit**

```bash
git add src/validation/run_pipeline.py tests/test_validation_run_pipeline.py
git commit -m "feat: add CLI pipeline chaining train, export, and evaluate stages"
```

---

### Task 9: Manual end-to-end run against the real dataset (not automated)

This task has no pytest coverage — it needs the real `data/PlantVillage` dataset (via `python scripts/download_plantvillage.py` if not already present) and takes real training time (15 epochs on node_1's actual image count), so it's a manual verification step, matching the spec's own "Testing/verification" section.

**Files:** none (verification only).

- [ ] **Step 1: Run the full pipeline against the real dataset**

Run: `python -m src.validation.run_pipeline`

Expected: no errors; `outputs/validation/node_1_mobilenet_v3_small/` contains `checkpoint.pt`, `classes.json`, `training_log.json`, `model.onnx`, `manifest.json`, `report.json`.

- [ ] **Step 2: Inspect training_log.json for a sensible accuracy trend**

Open `outputs/validation/node_1_mobilenet_v3_small/training_log.json` and confirm `test_disease_accuracy` is logged per-epoch (not just one final number like today's `outputs/results_summary.json`), and note whether it's still climbing at epoch 15 (if so, more epochs may help further) or has plateaued.

- [ ] **Step 3: Compare against today's baseline number**

Open `outputs/results_summary.json`, find `mobilenet_v3_small.baseline_eval.node_1.disease_accuracy` (today's 1-epoch, non-deduped, non-augmented number), and compare it to `report.json`'s `summary.disease_accuracy`. Note the direction and magnitude of the change — a lower-but-more-honest number would confirm the leakage/optimism hypothesis from the design spec; a similar or higher number would suggest the domain-gap/augmentation fix mattered more than the evaluation methodology fix.

- [ ] **Step 4: Skim report.json's per-image results for remaining failure patterns**

Look at `report.json`'s `results` entries where `disease_correct` is `false` but `crop_correct` is `true` (the "genuine confusion within the correct crop" pattern already observed on real photos) — note whether this pattern is reduced, and check `summary.top_confusions` for which specific disease pairs are still most confused, to guide any follow-up (e.g. targeted real-photo fine-tuning for those specific diseases).

No commit for this task — it's a diagnostic run, not a code change. If the results are promising, the next conversation can decide whether to roll the same recipe out to node_0/node_2 and the other two architectures, or to prioritize real-photo fine-tuning instead.
