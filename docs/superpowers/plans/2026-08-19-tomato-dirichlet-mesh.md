# Tomato Multi-Source Dirichlet Knowledge-Transfer Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Tomato-only, 3-node federated knowledge-transfer pipeline that pools PlantVillage + PlantDoc + PlantWild (v1+v2) into one canonical label space, partitions it with a Dirichlet label-skew split (not the corn pipeline's complete-disjoint split), and evaluates against one global held-out test set — retesting whether the corn pipeline's null knowledge-transfer result was caused by its zero-cross-node-class-overlap split design.

**Architecture:** A new `src/validation/tomato_mesh_dataset.py` module merges four image sources into one flat list of `(path, canonical_disease_idx)` items, backed by a duck-typed `TomatoMergedDataset` that satisfies the same attribute surface (`.base.samples`, `.base.targets`, `.labels.class_to_crop_disease`, `.labels.crop_classes`, `.labels.disease_classes`) the existing reused training/export/evaluation code (`train_mobilenet.run_training`, `export_onnx.export_checkpoint`, `evaluate_onnx.run_evaluation`) already expects from a `PlantVillageDataset` — no corn-style `CornDiseaseView`/label-remap wrapper is needed here because the merged pool is canonical from construction, not a global-to-compact remap. A global dedup-aware test split is carved out first; the remaining pool is Dirichlet-partitioned (α=0.3) across 3 nodes reusing `plantvillage._dirichlet_partition`'s exact algorithm. Stage 1 (`run_tomato_pipeline.py`) and Stage 2 (`run_tomato_knowledge_transfer.py`) mirror `run_corn_pipeline.py`/`run_knowledge_transfer.py`'s structure, reusing `run_kt_round`/`_sum_tracked_blocks`/`_build_per_node_energy_breakdown` unchanged via import (they're already crop-agnostic), and only reimplementing the corn-specific parts (evaluation report shape, per-node checkpoint loading, the summary builder).

**Tech Stack:** Python, PyTorch, torchvision, NumPy, PIL, ONNX Runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md`

## Global Constraints

- Crop scope: Tomato only, pooled from `data/PlantVillage`, `data/PlantDoc`, `data/PlantWild/plantwild/plantwild/images` (v1), `data/PlantWild/plantwild_v2/plantwild_v2` (v2).
- Federated topology: 3 nodes, Dirichlet concentration α = 0.3.
- Model: `mobilenet_v3_small` only.
- Knowledge-transfer rounds: default 5, CLI-configurable 1-5 (validated range, `parser.error` outside it).
- Stage 1 per-node training: 2 epochs (reduced from the corn/node_1 recipe's original 15 — same `train_mobilenet.run_training` call, just a smaller `epochs` argument).
- Test split is **global** (one shared held-out set, carved out before any per-node partitioning), fraction controlled by a new `tomato_mesh.test_fraction` key (default 0.20) — deliberately **not** the shared `data.test_fraction` key (0.15), which stays scoped to the per-node splits the general mesh and corn pipelines already use.
- Perceptual-hash dedup threshold: Hamming distance ≤ 5 (`tomato_mesh.dedup_threshold`). A max-group-size cap (`tomato_mesh.dedup_max_group_size`, default 25) additionally caps any single duplicate-group at `min(25, 5% of that canonical class's pooled count)` — a starting heuristic that must be checked against the real merged pool's group-size histogram before the first real run is trusted.
- Aggregation mechanism is **unchanged**: `trimmed_mean`/`krum` + KD/prototype distillation from `src/federated/`. No AdaClass, no change to `src/federated/aggregation.py`, `src/federated/node.py`, or `src/federated/mesh.py`. This plan changes the *data split*, not the *aggregation weighting*.
- Do not modify: `src/data/plantvillage.py`, `src/validation/node1_dataset.py`, `src/validation/corn_mesh_dataset.py`, `src/validation/run_corn_pipeline.py`, `src/validation/run_knowledge_transfer.py`, anything under `src/federated/`. All are reused read-only (including one deliberate cross-module import of `plantvillage._dirichlet_partition`, a pure, already-generic function).
- The `healthy` label mapping for PlantDoc's `"Tomato leaf"` and PlantWild v1's `"tomato leaf"` folders is asserted by this plan but has **not** been visually verified against real images — flag this in the PR/results write-up, don't silently treat it as confirmed.

---

### Task 1: PlantDoc Tomato loader

**Files:**
- Create: `src/data/plantdoc.py`
- Test: `tests/test_data_plantdoc.py`

**Interfaces:**
- Produces: `PLANTDOC_CANONICAL_MAP: dict[str, str]` (PlantDoc folder name → canonical disease name, no `Tomato___` prefix, matching PlantVillage's own disease-name strings). `load_plantdoc_tomato_paths(root: str | Path) -> list[tuple[str, str]]` returning `(absolute_or_given_path: str, canonical_disease_name: str)` pairs, pooling both `train/` and `test/` subfolders. Raises `FileNotFoundError` if `root` itself doesn't exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_plantdoc.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.plantdoc import PLANTDOC_CANONICAL_MAP, load_plantdoc_tomato_paths


def _make_images(dir_path: Path, count: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")


def test_canonical_map_has_no_target_spot_and_maps_leaf_to_healthy():
    assert "Tomato leaf" in PLANTDOC_CANONICAL_MAP
    assert PLANTDOC_CANONICAL_MAP["Tomato leaf"] == "healthy"
    assert "Target_Spot" not in PLANTDOC_CANONICAL_MAP.values()


def test_pools_train_and_test_subfolders(tmp_path):
    root = tmp_path / "PlantDoc"
    _make_images(root / "train" / "Tomato leaf bacterial spot", 3)
    _make_images(root / "test" / "Tomato leaf bacterial spot", 2)
    _make_images(root / "train" / "Apple leaf", 4)  # non-Tomato, must be ignored

    pairs = load_plantdoc_tomato_paths(root)
    assert len(pairs) == 5
    assert all(canonical == "Bacterial_spot" for _, canonical in pairs)


def test_missing_root_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PlantDoc"):
        load_plantdoc_tomato_paths(tmp_path / "does_not_exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_plantdoc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.plantdoc'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/plantdoc.py
"""Loader for PlantDoc's Tomato-labeled images, folded into the Tomato
Dirichlet mesh pipeline's canonical label space (see
src/validation/tomato_mesh_dataset.py). PlantDoc ships its own train/test
split; both subfolders are pooled here since the mesh pipeline defines its
own global split downstream.
"""

from __future__ import annotations

from pathlib import Path

PLANTDOC_CANONICAL_MAP: dict[str, str] = {
    "Tomato leaf": "healthy",
    "Tomato Early blight leaf": "Early_blight",
    "Tomato leaf bacterial spot": "Bacterial_spot",
    "Tomato leaf late blight": "Late_blight",
    "Tomato leaf mosaic virus": "Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Leaf_Mold",
    "Tomato Septoria leaf spot": "Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": "Spider_mites Two-spotted_spider_mite",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def load_plantdoc_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    """Returns (path, canonical_disease_name) pairs for every image under
    root/train and root/test whose immediate folder name is a known
    PlantDoc Tomato class (see PLANTDOC_CANONICAL_MAP). Folders not in the
    map (other crops, or unrecognized names) are ignored.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PlantDoc root not found: {root}")

    results: list[tuple[str, str]] = []
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for folder_name, canonical_name in PLANTDOC_CANONICAL_MAP.items():
            class_dir = split_dir / folder_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix in IMAGE_EXTENSIONS:
                    results.append((str(path), canonical_name))
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_plantdoc.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/plantdoc.py tests/test_data_plantdoc.py
git commit -m "feat: add PlantDoc Tomato loader with canonical label mapping"
```

---

### Task 2: PlantWild (v1 + v2) Tomato loader

**Files:**
- Create: `src/data/plantwild.py`
- Test: `tests/test_data_plantwild.py`

**Interfaces:**
- Produces: `PLANTWILD_V1_CANONICAL_MAP: dict[str, str]`, `PLANTWILD_V2_CANONICAL_MAP: dict[str, str]` (v2 has no `"tomato leaf"`/healthy entry). `load_plantwild_v1_tomato_paths(root: str | Path) -> list[tuple[str, str]]`, `load_plantwild_v2_tomato_paths(root: str | Path) -> list[tuple[str, str]]` — same `(path, canonical_disease_name)` shape as Task 1. Both raise `FileNotFoundError` if their root doesn't exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_plantwild.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.plantwild import (
    PLANTWILD_V1_CANONICAL_MAP,
    PLANTWILD_V2_CANONICAL_MAP,
    load_plantwild_v1_tomato_paths,
    load_plantwild_v2_tomato_paths,
)


def _make_images(dir_path: Path, count: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")


def test_v2_map_has_no_healthy_leaf_entry():
    assert "tomato leaf" in PLANTWILD_V1_CANONICAL_MAP
    assert "tomato leaf" not in PLANTWILD_V2_CANONICAL_MAP
    assert PLANTWILD_V1_CANONICAL_MAP["tomato leaf"] == "healthy"


def test_v1_loader_pools_known_classes(tmp_path):
    root = tmp_path / "plantwild_images"
    _make_images(root / "tomato bacterial leaf spot", 4)
    _make_images(root / "tomato leaf", 2)
    _make_images(root / "apple black rot", 3)  # non-Tomato, ignored

    pairs = load_plantwild_v1_tomato_paths(root)
    assert len(pairs) == 6
    canonical_names = {c for _, c in pairs}
    assert canonical_names == {"Bacterial_spot", "healthy"}


def test_v2_loader_ignores_healthy_folder_even_if_present(tmp_path):
    root = tmp_path / "plantwild_v2"
    _make_images(root / "tomato bacterial leaf spot", 3)
    _make_images(root / "tomato leaf", 2)  # not in v2's map, must be ignored

    pairs = load_plantwild_v2_tomato_paths(root)
    assert len(pairs) == 3
    assert all(c == "Bacterial_spot" for _, c in pairs)


def test_missing_roots_raise_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="PlantWild"):
        load_plantwild_v1_tomato_paths(tmp_path / "missing_v1")
    with pytest.raises(FileNotFoundError, match="PlantWild"):
        load_plantwild_v2_tomato_paths(tmp_path / "missing_v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_plantwild.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.plantwild'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/plantwild.py
"""Loader for PlantWild's Tomato-labeled images (v1 and v2 builds), folded
into the Tomato Dirichlet mesh pipeline's canonical label space (see
src/validation/tomato_mesh_dataset.py). Both versions are pooled by the
caller; global perceptual-hash dedup (tomato_mesh_dataset.py) is
responsible for catching any near-duplicate images between the two.
"""

from __future__ import annotations

from pathlib import Path

PLANTWILD_V1_CANONICAL_MAP: dict[str, str] = {
    "tomato leaf": "healthy",
    "tomato bacterial leaf spot": "Bacterial_spot",
    "tomato early blight": "Early_blight",
    "tomato late blight": "Late_blight",
    "tomato leaf mold": "Leaf_Mold",
    "tomato mosaic virus": "Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato_Yellow_Leaf_Curl_Virus",
}

# v2 has no healthy/leaf equivalent for Tomato -- confirmed against the
# real download, not assumed.
PLANTWILD_V2_CANONICAL_MAP: dict[str, str] = {
    k: v for k, v in PLANTWILD_V1_CANONICAL_MAP.items() if k != "tomato leaf"
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _load_from_root(root: str | Path, canonical_map: dict[str, str]) -> list[tuple[str, str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PlantWild root not found: {root}")

    results: list[tuple[str, str]] = []
    for folder_name, canonical_name in canonical_map.items():
        class_dir = root / folder_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix in IMAGE_EXTENSIONS:
                results.append((str(path), canonical_name))
    return results


def load_plantwild_v1_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    return _load_from_root(root, PLANTWILD_V1_CANONICAL_MAP)


def load_plantwild_v2_tomato_paths(root: str | Path) -> list[tuple[str, str]]:
    return _load_from_root(root, PLANTWILD_V2_CANONICAL_MAP)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_plantwild.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/plantwild.py tests/test_data_plantwild.py
git commit -m "feat: add PlantWild v1/v2 Tomato loader with canonical label mapping"
```

---

### Task 3: Tomato label map and merged dataset core

**Files:**
- Create: `src/validation/tomato_mesh_dataset.py`
- Test: `tests/test_validation_tomato_mesh_dataset.py`

**Interfaces:**
- Consumes: nothing from earlier tasks yet (this task is the module's foundation).
- Produces: `TOMATO_DISEASE_ORDER: list[str]` (10 canonical names, PlantVillage-style, no `Tomato___` prefix). `TomatoLabelMap` (`@dataclass`: `crop_classes: list[str]`, `disease_classes: list[str]`, `name_to_disease_idx: dict[str, int]`). `build_tomato_label_map(disease_names: list[str] = TOMATO_DISEASE_ORDER) -> TomatoLabelMap`. `TomatoRawItem` (`@dataclass`: `source: str`, `path: str`, `canonical_disease_idx: int`). `TomatoMergedDataset(Dataset)` — `__init__(self, items: list[TomatoRawItem], label_map: TomatoLabelMap, transform)`, `__len__`, `__getitem__(self, pos: int) -> tuple[Tensor, int, int]` returning `(image, 0, canonical_disease_idx)`, plus `.base` (`SimpleNamespace(samples=[(path, canonical_disease_idx), ...], targets=[canonical_disease_idx, ...])`) and `.labels` (`SimpleNamespace(crop_classes=[...], disease_classes=[...], class_to_crop_disease={i: (0, i) for i in range(10)})`) attributes matching `PlantVillageDataset`'s duck-typed surface. `build_tomato_train_eval_datasets(items: list[TomatoRawItem], label_map: TomatoLabelMap, image_size: int) -> tuple[TomatoMergedDataset, TomatoMergedDataset]` (train has augmentation transform, eval doesn't, both share the same `items` ordering).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_tomato_mesh_dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.validation.tomato_mesh_dataset import (
    TOMATO_DISEASE_ORDER,
    TomatoRawItem,
    build_tomato_label_map,
    build_tomato_train_eval_datasets,
)


def test_tomato_disease_order_has_ten_canonical_classes():
    assert len(TOMATO_DISEASE_ORDER) == 10
    assert "healthy" in TOMATO_DISEASE_ORDER
    assert "Target_Spot" in TOMATO_DISEASE_ORDER


def test_build_tomato_label_map():
    label_map = build_tomato_label_map()
    assert label_map.crop_classes == ["Tomato"]
    assert label_map.disease_classes == TOMATO_DISEASE_ORDER
    assert label_map.name_to_disease_idx["healthy"] == TOMATO_DISEASE_ORDER.index("healthy")


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_merged_dataset_getitem_and_duck_typed_surface(tmp_path):
    label_map = build_tomato_label_map()
    p0 = tmp_path / "a.jpg"
    p1 = tmp_path / "b.jpg"
    _write_image(p0)
    _write_image(p1)
    items = [
        TomatoRawItem(source="plantvillage", path=str(p0), canonical_disease_idx=0),
        TomatoRawItem(source="plantdoc", path=str(p1), canonical_disease_idx=2),
    ]
    train_ds, eval_ds = build_tomato_train_eval_datasets(items, label_map, image_size=16)

    assert len(train_ds) == 2
    image, crop_label, disease_label = eval_ds[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 16, 16)
    assert crop_label == 0
    assert disease_label == 2

    # duck-typed surface required by evaluate_onnx.run_evaluation / export_onnx.export_checkpoint
    assert eval_ds.base.samples[1] == (str(p1), 2)
    assert eval_ds.base.targets == [0, 2]
    assert eval_ds.labels.class_to_crop_disease[2] == (0, 2)
    assert eval_ds.labels.crop_classes == ["Tomato"]
    assert eval_ds.labels.disease_classes == TOMATO_DISEASE_ORDER

    # train has augmentation, eval doesn't -- same style assertion as
    # tests/test_validation_node1_dataset.py's build_train_eval_datasets test
    train_types = [type(t) for t in train_ds.transform.transforms]
    eval_types = [type(t) for t in eval_ds.transform.transforms]
    assert train_types != eval_types
    assert len(eval_types) == 3  # Resize, ToTensor, Normalize
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.tomato_mesh_dataset'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/validation/tomato_mesh_dataset.py
"""Tomato multi-source Dirichlet mesh dataset preparation -- pools
PlantVillage + PlantDoc + PlantWild (v1+v2) into one canonical label
space, carves a GLOBAL dedup-aware test split, then Dirichlet-partitions
the remaining training pool across 3 nodes. See
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.

Unlike corn_mesh_dataset.py, no CornDiseaseView-style label remap is
needed here: the merged item list is built directly in the canonical
compact label space (crop always index 0, disease already 0-9), so
TomatoMergedDataset's `.labels.class_to_crop_disease` is simply
{i: (0, i) for i in range(len(disease_classes))} -- there is no
global-vs-compact index mismatch to bridge, because there is no single
underlying ImageFolder spanning other crops the way PlantVillage's full
dataset does for corn_mesh_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.plantvillage import IMAGENET_MEAN, IMAGENET_STD

TOMATO_DISEASE_ORDER = [
    "Bacterial_spot",
    "Early_blight",
    "healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus",
]


@dataclass
class TomatoLabelMap:
    crop_classes: list[str]
    disease_classes: list[str]
    name_to_disease_idx: dict[str, int]


def build_tomato_label_map(disease_names: list[str] = TOMATO_DISEASE_ORDER) -> TomatoLabelMap:
    return TomatoLabelMap(
        crop_classes=["Tomato"],
        disease_classes=list(disease_names),
        name_to_disease_idx={name: i for i, name in enumerate(disease_names)},
    )


@dataclass
class TomatoRawItem:
    source: str
    path: str
    canonical_disease_idx: int


class TomatoMergedDataset(Dataset):
    """A flat, canonical-label-space dataset over pooled multi-source
    Tomato items. Exposes the same `.base.samples`/`.base.targets`/
    `.labels.class_to_crop_disease`/`.labels.crop_classes`/
    `.labels.disease_classes` surface as PlantVillageDataset so
    train_mobilenet.run_training, export_onnx.export_checkpoint, and
    evaluate_onnx.run_evaluation all work against it unmodified.
    """

    def __init__(self, items: list[TomatoRawItem], label_map: TomatoLabelMap, transform: transforms.Compose):
        self.items = items
        self.transform = transform
        self.labels = SimpleNamespace(
            crop_classes=list(label_map.crop_classes),
            disease_classes=list(label_map.disease_classes),
            class_to_crop_disease={i: (0, i) for i in range(len(label_map.disease_classes))},
        )
        self.base = SimpleNamespace(
            samples=[(item.path, item.canonical_disease_idx) for item in items],
            targets=[item.canonical_disease_idx for item in items],
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, pos: int):
        item = self.items[pos]
        with Image.open(item.path) as img:
            image = self.transform(img.convert("RGB"))
        return image, 0, item.canonical_disease_idx


def build_tomato_train_eval_datasets(
    items: list[TomatoRawItem], label_map: TomatoLabelMap, image_size: int
) -> tuple[TomatoMergedDataset, TomatoMergedDataset]:
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(
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
    train_ds = TomatoMergedDataset(items, label_map, train_transform)
    eval_ds = TomatoMergedDataset(items, label_map, eval_transform)
    return train_ds, eval_ds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/tomato_mesh_dataset.py tests/test_validation_tomato_mesh_dataset.py
git commit -m "feat: add TomatoLabelMap and merged multi-source dataset core"
```

---

### Task 4: Load and merge all four sources into canonical items

**Files:**
- Modify: `src/validation/tomato_mesh_dataset.py`
- Test: `tests/test_validation_tomato_mesh_dataset.py` (extend)

**Interfaces:**
- Consumes: `load_plantdoc_tomato_paths` (Task 1), `load_plantwild_v1_tomato_paths`/`load_plantwild_v2_tomato_paths` (Task 2), `TomatoRawItem`/`TomatoLabelMap` (Task 3).
- Produces: `load_plantvillage_tomato_items(root: str | Path, label_map: TomatoLabelMap) -> list[TomatoRawItem]`. `_items_from_paths(pairs: list[tuple[str, str]], source: str, label_map: TomatoLabelMap) -> list[TomatoRawItem]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_validation_tomato_mesh_dataset.py
from src.validation.tomato_mesh_dataset import _items_from_paths, load_plantvillage_tomato_items


def _write_pv_class(tmp_path, crop_disease_folder, count):
    cls_dir = tmp_path / "PlantVillage" / crop_disease_folder
    cls_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(count):
        arr = rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")


def test_load_plantvillage_tomato_items_filters_to_tomato_only(tmp_path):
    label_map = build_tomato_label_map()
    _write_pv_class(tmp_path, "Tomato___Bacterial_spot", 3)
    _write_pv_class(tmp_path, "Tomato___healthy", 2)
    _write_pv_class(tmp_path, "Potato___healthy", 4)  # must be excluded

    items = load_plantvillage_tomato_items(tmp_path / "PlantVillage", label_map)
    assert len(items) == 5
    assert all(item.source == "plantvillage" for item in items)
    canonical_names = {label_map.disease_classes[i.canonical_disease_idx] for i in items}
    assert canonical_names == {"Bacterial_spot", "healthy"}


def test_items_from_paths_maps_canonical_name_to_index():
    label_map = build_tomato_label_map()
    pairs = [("/a.jpg", "healthy"), ("/b.jpg", "Early_blight")]
    items = _items_from_paths(pairs, "plantdoc", label_map)
    assert items[0].canonical_disease_idx == label_map.name_to_disease_idx["healthy"]
    assert items[1].source == "plantdoc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v -k "plantvillage_tomato_items or items_from_paths"`
Expected: FAIL with `ImportError: cannot import name 'load_plantvillage_tomato_items'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/validation/tomato_mesh_dataset.py` (after the imports, add `from src.data.plantvillage import load_full_dataset` alongside the existing `IMAGENET_MEAN, IMAGENET_STD` import; place the two functions after `build_tomato_train_eval_datasets`):

```python
def load_plantvillage_tomato_items(root, label_map: TomatoLabelMap) -> list[TomatoRawItem]:
    dataset = load_full_dataset(root, image_size=32)  # image_size unused beyond this scan
    tomato_crop_idx = dataset.labels.crop_classes.index("Tomato")
    items: list[TomatoRawItem] = []
    for idx in range(len(dataset)):
        class_idx = dataset.base.targets[idx]
        crop_idx, disease_idx = dataset.labels.class_to_crop_disease[class_idx]
        if crop_idx != tomato_crop_idx:
            continue
        disease_name = dataset.labels.disease_classes[disease_idx]
        if disease_name not in label_map.name_to_disease_idx:
            continue
        path, _ = dataset.base.samples[idx]
        items.append(
            TomatoRawItem(
                source="plantvillage",
                path=path,
                canonical_disease_idx=label_map.name_to_disease_idx[disease_name],
            )
        )
    return items


def _items_from_paths(
    pairs: list[tuple[str, str]], source: str, label_map: TomatoLabelMap
) -> list[TomatoRawItem]:
    return [
        TomatoRawItem(source=source, path=path, canonical_disease_idx=label_map.name_to_disease_idx[canonical_name])
        for path, canonical_name in pairs
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/validation/tomato_mesh_dataset.py tests/test_validation_tomato_mesh_dataset.py
git commit -m "feat: load and remap PlantVillage's Tomato subset into canonical items"
```

---

### Task 5: Merged-pool hashing and capped dedup split

**Files:**
- Modify: `src/validation/tomato_mesh_dataset.py`
- Test: `tests/test_validation_tomato_mesh_dataset.py` (extend)

**Interfaces:**
- Consumes: `TomatoRawItem` (Task 3), `average_hash` (`src/validation/hashing.py`, existing), `group_duplicates` (`src/validation/node1_dataset.py`, existing).
- Produces: `compute_merged_image_hashes(items: list[TomatoRawItem], indices: list[int]) -> dict[int, int]`. `enforce_max_group_size(indices: list[int], groups: dict[int, int], items: list[TomatoRawItem], max_group_size: int, max_group_fraction_of_class: float = 0.05) -> dict[int, int]`. `capped_dedup_split(indices: list[int], hashes: dict[int, int], items: list[TomatoRawItem], test_fraction: float, seed: int, threshold: int, max_group_size: int) -> tuple[list[int], list[int]]` returning `(train_idx, test_idx)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_validation_tomato_mesh_dataset.py
from src.validation.tomato_mesh_dataset import (
    capped_dedup_split,
    compute_merged_image_hashes,
    enforce_max_group_size,
)


def test_compute_merged_image_hashes_reads_each_item_path(tmp_path):
    p0, p1 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _write_image(p0)
    _write_image(p1)
    items = [
        TomatoRawItem(source="plantvillage", path=str(p0), canonical_disease_idx=0),
        TomatoRawItem(source="plantdoc", path=str(p1), canonical_disease_idx=0),
    ]
    hashes = compute_merged_image_hashes(items, [0, 1])
    assert set(hashes.keys()) == {0, 1}
    assert all(isinstance(h, int) for h in hashes.values())


def test_enforce_max_group_size_breaks_up_oversized_group():
    # 5 items, all in canonical class 0, all hashed into ONE group by the
    # caller (hand-built, mirrors test_validation_node1_dataset.py's style).
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(5)]
    indices = [0, 1, 2, 3, 4]
    groups = {i: 0 for i in indices}  # one big group, id=0

    fixed = enforce_max_group_size(indices, groups, items, max_group_size=2, max_group_fraction_of_class=1.0)
    # group of 5 > cap of 2 -> broken into singletons
    assert len({fixed[i] for i in indices}) == 5
    for i in indices:
        assert fixed[i] == i


def test_enforce_max_group_size_leaves_small_groups_alone():
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(3)]
    indices = [0, 1, 2]
    groups = {0: 0, 1: 0, 2: 2}  # group {0,1} size 2, group {2} size 1

    fixed = enforce_max_group_size(indices, groups, items, max_group_size=10, max_group_fraction_of_class=1.0)
    assert fixed == groups


def test_capped_dedup_split_partitions_all_indices():
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(6)]
    indices = list(range(6))
    hashes = {0: 0b0000, 1: 0b0000, 2: 0b1111, 3: 0b1111, 4: 0b0101, 5: 0b0110}

    train_idx, test_idx = capped_dedup_split(
        indices, hashes, items, test_fraction=0.5, seed=1, threshold=1, max_group_size=10
    )
    assert sorted(train_idx + test_idx) == indices
    assert set(train_idx).isdisjoint(test_idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v -k "hashes or enforce_max or capped_dedup"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

Add `import random` and `from src.validation.hashing import average_hash` and `from src.validation.node1_dataset import group_duplicates` to `tomato_mesh_dataset.py`'s imports, then append:

```python
def compute_merged_image_hashes(items: list[TomatoRawItem], indices: list[int]) -> dict[int, int]:
    hashes: dict[int, int] = {}
    for idx in indices:
        with Image.open(items[idx].path) as img:
            hashes[idx] = average_hash(img.convert("RGB"))
    return hashes


def enforce_max_group_size(
    indices: list[int],
    groups: dict[int, int],
    items: list[TomatoRawItem],
    max_group_size: int,
    max_group_fraction_of_class: float = 0.05,
) -> dict[int, int]:
    """Any duplicate-group larger than min(max_group_size, fraction * that
    group's dominant class's pooled count) is broken into singleton
    groups -- this is the fix for the exact failure mode that let 97% of
    node_1's Soybean class collapse into one "duplicate" group
    (docs/node1_validation_tuning_and_results.md).
    """
    class_totals: dict[int, int] = {}
    for idx in indices:
        cls = items[idx].canonical_disease_idx
        class_totals[cls] = class_totals.get(cls, 0) + 1

    members_by_group: dict[int, list[int]] = {}
    for idx in indices:
        members_by_group.setdefault(groups[idx], []).append(idx)

    fixed = dict(groups)
    for members in members_by_group.values():
        dominant_cls = items[members[0]].canonical_disease_idx
        cap = min(max_group_size, max(1, int(class_totals[dominant_cls] * max_group_fraction_of_class)))
        if len(members) > cap:
            for member in members:
                fixed[member] = member
    return fixed


def capped_dedup_split(
    indices: list[int],
    hashes: dict[int, int],
    items: list[TomatoRawItem],
    test_fraction: float,
    seed: int,
    threshold: int,
    max_group_size: int,
) -> tuple[list[int], list[int]]:
    """Same greedy-fill algorithm as node1_dataset.dedup_aware_split, plus
    the max-group-size cap above. Reimplemented (not calling
    dedup_aware_split directly) because that function recomputes grouping
    internally and has no cap parameter to inject.
    """
    groups = group_duplicates(indices, hashes, threshold)
    groups = enforce_max_group_size(indices, groups, items, max_group_size)

    group_members: dict[int, list[int]] = {}
    for idx in indices:
        group_members.setdefault(groups[idx], []).append(idx)

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/validation/tomato_mesh_dataset.py tests/test_validation_tomato_mesh_dataset.py
git commit -m "feat: add merged-pool hashing and max-group-size-capped dedup split"
```

---

### Task 6: Probe carve, Dirichlet partition, diagnostics, and orchestration

**Files:**
- Modify: `src/validation/tomato_mesh_dataset.py`
- Modify: `tests/conftest.py` (add `tomato_scoped_config` fixture)
- Test: `tests/test_validation_tomato_mesh_dataset.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1-5, plus `plantvillage._dirichlet_partition` (existing, imported directly — a pure, generic function; see Global Constraints).
- Produces: `carve_merged_probe_set(items, indices, probe_fraction, seed, large_class_threshold=200, min_samples_small_class=8, max_fraction_small_class=0.2) -> tuple[list[int], list[int]]` (returns `(probe_idx, remaining_idx)`). `_jensen_shannon_divergence(p, q) -> float`. `_summarize_partition(items, shard, label_map) -> dict` (keys: `num_samples`, `per_class_counts`, `dominant_class`, `dominant_class_fraction`, `js_divergence_from_uniform`, `num_classes_present`, `low_representation_classes`). `TomatoMeshData` (`@dataclass`: `train_base`, `eval_base`, `label_map`, `image_size`, `probe_idx: list[int]`, `test_idx: list[int]`, `per_node: dict[str, dict[str, list[int]]]`, `partition_diagnostics: dict[str, dict]`). `prepare_tomato_mesh_data(cfg: Config) -> TomatoMeshData`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_validation_tomato_mesh_dataset.py
import numpy as np

from src.validation.tomato_mesh_dataset import (
    TomatoMeshData,
    _jensen_shannon_divergence,
    _summarize_partition,
    carve_merged_probe_set,
    prepare_tomato_mesh_data,
)


def test_jensen_shannon_divergence_zero_for_identical_distributions():
    p = np.array([0.5, 0.5])
    assert _jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_divergence_one_for_disjoint_supports():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert _jensen_shannon_divergence(p, q) == pytest.approx(1.0, abs=1e-9)


def test_carve_merged_probe_set_disjoint_and_covers_all_indices():
    label_map = build_tomato_label_map()
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=i % 3) for i in range(30)]
    indices = list(range(30))
    probe_idx, remaining_idx = carve_merged_probe_set(
        items, indices, probe_fraction=0.2, seed=1, min_samples_small_class=1, max_fraction_small_class=0.5
    )
    assert set(probe_idx).isdisjoint(remaining_idx)
    assert sorted(probe_idx + remaining_idx) == indices
    assert len(probe_idx) > 0


def test_summarize_partition_flags_low_representation_classes():
    label_map = build_tomato_label_map()
    items = [TomatoRawItem(source="x", path=f"/{i}.jpg", canonical_disease_idx=0) for i in range(20)]
    items += [TomatoRawItem(source="x", path=f"/h{i}.jpg", canonical_disease_idx=2) for i in range(1)]
    shard = list(range(21))
    summary = _summarize_partition(items, shard, label_map)
    assert summary["num_samples"] == 21
    assert summary["dominant_class"] == label_map.disease_classes[0]
    assert label_map.disease_classes[2] in summary["low_representation_classes"]


def test_prepare_tomato_mesh_data_end_to_end(tomato_scoped_config):
    cfg = tomato_scoped_config
    data = prepare_tomato_mesh_data(cfg)
    assert isinstance(data, TomatoMeshData)
    assert len(data.per_node) == 3

    all_train = [i for shard in data.per_node.values() for i in shard["train_idx"]]
    assert set(all_train).isdisjoint(data.test_idx)
    assert set(all_train).isdisjoint(data.probe_idx)
    # every node's shard is disjoint from every other node's
    node_sets = [set(v["train_idx"]) for v in data.per_node.values()]
    for i in range(len(node_sets)):
        for j in range(i + 1, len(node_sets)):
            assert node_sets[i].isdisjoint(node_sets[j])

    for node_id, diagnostics in data.partition_diagnostics.items():
        assert diagnostics["num_samples"] == len(data.per_node[node_id]["train_idx"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v -k "jensen or carve_merged or summarize_partition or prepare_tomato_mesh_data"`
Expected: FAIL — `ImportError` for the new names, and `fixture 'tomato_scoped_config' not found` for the last test.

- [ ] **Step 3: Add the `tomato_scoped_config` fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def tomato_scoped_config(tmp_path):
    """A tiny synthetic 4-source Tomato dataset (PlantVillage + PlantDoc +
    PlantWild v1/v2) covering 3 canonical disease classes plus one
    off-crop PlantVillage control class, used across the tomato_mesh test
    suite.
    """
    from src.config import Config

    rng = np.random.RandomState(0)

    def _make_images(dir_path, count):
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(dir_path / f"img_{i}.jpg")

    pv_root = tmp_path / "PlantVillage"
    for cls, count in {
        "Tomato___Bacterial_spot": 10,
        "Tomato___healthy": 10,
        "Tomato___Early_blight": 8,
        "Potato___healthy": 4,
    }.items():
        _make_images(pv_root / cls, count)

    plantdoc_root = tmp_path / "PlantDoc"
    _make_images(plantdoc_root / "train" / "Tomato leaf bacterial spot", 4)
    _make_images(plantdoc_root / "test" / "Tomato leaf bacterial spot", 2)
    _make_images(plantdoc_root / "train" / "Tomato leaf", 3)

    plantwild_v1_root = tmp_path / "PlantWild" / "plantwild" / "plantwild" / "images"
    _make_images(plantwild_v1_root / "tomato bacterial leaf spot", 4)
    _make_images(plantwild_v1_root / "tomato leaf", 3)

    plantwild_v2_root = tmp_path / "PlantWild" / "plantwild_v2" / "plantwild_v2"
    _make_images(plantwild_v2_root / "tomato bacterial leaf spot", 3)

    cfg = Config(
        {
            "data": {
                "root": str(pv_root),
                "image_size": 32,
                "seed": 42,
                "probe_set_fraction": 0.1,
                "probe_set_large_class_threshold": 200,
                "probe_set_min_samples_small_class": 1,
                "probe_set_max_fraction_small_class": 0.5,
            },
            "tomato_mesh": {
                "crop": "Tomato",
                "plantdoc_root": str(plantdoc_root),
                "plantwild_v1_root": str(plantwild_v1_root),
                "plantwild_v2_root": str(plantwild_v2_root),
                "num_nodes": 3,
                "dirichlet_alpha": 0.3,
                "test_fraction": 0.2,
                "dedup_threshold": 5,
                "dedup_max_group_size": 25,
                "rounds": 5,
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
    return cfg
```

- [ ] **Step 4: Write minimal implementation**

Add `import numpy as np` and `from src.data.plantvillage import _dirichlet_partition` (alongside the existing `load_full_dataset`, `IMAGENET_MEAN`, `IMAGENET_STD` import) to `tomato_mesh_dataset.py`, then append:

```python
def carve_merged_probe_set(
    items: list[TomatoRawItem],
    indices: list[int],
    probe_fraction: float,
    seed: int,
    large_class_threshold: int = 200,
    min_samples_small_class: int = 8,
    max_fraction_small_class: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Same stratified-by-class algorithm as plantvillage.carve_public_probe_set,
    adapted to operate on a subset of positions (indices) into the merged
    items list rather than an entire PlantVillageDataset -- needed because
    the probe set here is carved from just the post-test-split training
    pool, not the whole dataset.
    """
    targets = np.array([items[i].canonical_disease_idx for i in indices])
    rng = random.Random(seed)
    probe_idx: list[int] = []
    remaining_idx: list[int] = []
    for cls in sorted(set(targets.tolist())):
        cls_indices = [indices[i] for i in np.flatnonzero(targets == cls).tolist()]
        rng.shuffle(cls_indices)
        count = len(cls_indices)
        if count >= large_class_threshold:
            n_probe_cls = max(1, round(count * probe_fraction))
        else:
            target_n = max(min_samples_small_class, round(count * probe_fraction))
            n_probe_cls = min(target_n, int(count * max_fraction_small_class), count)
        probe_idx.extend(cls_indices[:n_probe_cls])
        remaining_idx.extend(cls_indices[n_probe_cls:])
    rng.shuffle(probe_idx)
    rng.shuffle(remaining_idx)
    return probe_idx, remaining_idx


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _summarize_partition(items: list[TomatoRawItem], shard: list[int], label_map: TomatoLabelMap) -> dict:
    num_classes = len(label_map.disease_classes)
    counts = np.zeros(num_classes, dtype=int)
    for idx in shard:
        counts[items[idx].canonical_disease_idx] += 1
    total = int(counts.sum())
    proportions = counts / total if total > 0 else counts.astype(float)
    uniform = np.full(num_classes, 1.0 / num_classes)
    dominant_idx = int(np.argmax(counts))
    nonzero_counts = counts[counts > 0]
    low_rep_threshold = np.percentile(nonzero_counts, 25) if len(nonzero_counts) > 0 else 0
    return {
        "num_samples": total,
        "per_class_counts": {name: int(c) for name, c in zip(label_map.disease_classes, counts)},
        "dominant_class": label_map.disease_classes[dominant_idx],
        "dominant_class_fraction": float(proportions[dominant_idx]) if total > 0 else 0.0,
        "js_divergence_from_uniform": _jensen_shannon_divergence(proportions, uniform),
        "num_classes_present": int(np.count_nonzero(counts)),
        "low_representation_classes": [
            label_map.disease_classes[i]
            for i in range(num_classes)
            if 0 < counts[i] <= low_rep_threshold
        ],
    }


@dataclass
class TomatoMeshData:
    train_base: TomatoMergedDataset
    eval_base: TomatoMergedDataset
    label_map: TomatoLabelMap
    image_size: int
    probe_idx: list[int]
    test_idx: list[int]
    per_node: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    partition_diagnostics: dict[str, dict] = field(default_factory=dict)


def prepare_tomato_mesh_data(cfg) -> TomatoMeshData:
    from src.data.plantdoc import load_plantdoc_tomato_paths
    from src.data.plantwild import load_plantwild_v1_tomato_paths, load_plantwild_v2_tomato_paths

    image_size = cfg.get("data.image_size", 160)
    pv_root = cfg.get("data.root", "data/PlantVillage")
    plantdoc_root = cfg.get("tomato_mesh.plantdoc_root", "data/PlantDoc")
    plantwild_v1_root = cfg.get("tomato_mesh.plantwild_v1_root", "data/PlantWild/plantwild/plantwild/images")
    plantwild_v2_root = cfg.get("tomato_mesh.plantwild_v2_root", "data/PlantWild/plantwild_v2/plantwild_v2")
    seed = cfg.get("data.seed", 42)
    num_nodes = cfg.get("tomato_mesh.num_nodes", 3)
    dirichlet_alpha = cfg.get("tomato_mesh.dirichlet_alpha", 0.3)
    test_fraction = cfg.get("tomato_mesh.test_fraction", 0.20)
    dedup_threshold = cfg.get("tomato_mesh.dedup_threshold", 5)
    dedup_max_group_size = cfg.get("tomato_mesh.dedup_max_group_size", 25)
    probe_fraction = cfg.get("data.probe_set_fraction", 0.05)

    label_map = build_tomato_label_map()

    pv_items = load_plantvillage_tomato_items(pv_root, label_map)
    pd_items = _items_from_paths(load_plantdoc_tomato_paths(plantdoc_root), "plantdoc", label_map)
    pw1_items = _items_from_paths(
        load_plantwild_v1_tomato_paths(plantwild_v1_root), "plantwild_v1", label_map
    )
    pw2_items = _items_from_paths(
        load_plantwild_v2_tomato_paths(plantwild_v2_root), "plantwild_v2", label_map
    )
    items = pv_items + pd_items + pw1_items + pw2_items
    if not items:
        raise FileNotFoundError(
            "No Tomato images found across PlantVillage/PlantDoc/PlantWild -- check "
            "data.root/tomato_mesh.plantdoc_root/tomato_mesh.plantwild_v1_root/"
            "tomato_mesh.plantwild_v2_root."
        )

    indices = list(range(len(items)))
    hashes = compute_merged_image_hashes(items, indices)
    train_idx_pool, test_idx = capped_dedup_split(
        indices, hashes, items, test_fraction, seed, threshold=dedup_threshold, max_group_size=dedup_max_group_size
    )

    probe_idx, remaining_idx = carve_merged_probe_set(
        items,
        train_idx_pool,
        probe_fraction,
        seed,
        large_class_threshold=cfg.get("data.probe_set_large_class_threshold", 200),
        min_samples_small_class=cfg.get("data.probe_set_min_samples_small_class", 8),
        max_fraction_small_class=cfg.get("data.probe_set_max_fraction_small_class", 0.2),
    )

    targets = np.array([items[i].canonical_disease_idx for i in remaining_idx])
    rng = np.random.RandomState(seed)
    shards = _dirichlet_partition(remaining_idx, targets, num_nodes, dirichlet_alpha, rng)

    per_node: dict[str, dict[str, list[int]]] = {}
    partition_diagnostics: dict[str, dict] = {}
    for node_idx, shard in enumerate(shards):
        node_id = f"node_{node_idx}"
        per_node[node_id] = {"train_idx": shard}
        partition_diagnostics[node_id] = _summarize_partition(items, shard, label_map)

    train_base, eval_base = build_tomato_train_eval_datasets(items, label_map, image_size)
    return TomatoMeshData(
        train_base=train_base,
        eval_base=eval_base,
        label_map=label_map,
        image_size=image_size,
        probe_idx=probe_idx,
        test_idx=test_idx,
        per_node=per_node,
        partition_diagnostics=partition_diagnostics,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_validation_tomato_mesh_dataset.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 6: Commit**

```bash
git add src/validation/tomato_mesh_dataset.py tests/test_validation_tomato_mesh_dataset.py tests/conftest.py
git commit -m "feat: add probe carve, Dirichlet partition, and prepare_tomato_mesh_data orchestration"
```

---

### Task 7: Add `tomato_mesh` config block

**Files:**
- Modify: `config.yaml`
- Test: `tests/test_config_tomato_mesh.py`

**Interfaces:**
- Consumes: `Config.load`/`Config.get` (existing, unchanged).
- Produces: no new code — just new keys under a `tomato_mesh:` top-level block in the real `config.yaml`, readable via `Config.get("tomato_mesh.<key>", default)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_tomato_mesh.py
from __future__ import annotations

from src.config import Config


def test_config_yaml_has_tomato_mesh_block():
    cfg = Config.load()  # repo-root config.yaml
    assert cfg.get("tomato_mesh.crop") == "Tomato"
    assert cfg.get("tomato_mesh.num_nodes") == 3
    assert cfg.get("tomato_mesh.dirichlet_alpha") == 0.3
    assert cfg.get("tomato_mesh.test_fraction") == 0.20
    assert cfg.get("tomato_mesh.dedup_threshold") == 5
    assert cfg.get("tomato_mesh.dedup_max_group_size") == 25
    assert cfg.get("tomato_mesh.rounds") == 5
    assert cfg.get("tomato_mesh.plantdoc_root") == "data/PlantDoc"
    assert cfg.get("tomato_mesh.plantwild_v1_root") == "data/PlantWild/plantwild/plantwild/images"
    assert cfg.get("tomato_mesh.plantwild_v2_root") == "data/PlantWild/plantwild_v2/plantwild_v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_tomato_mesh.py -v`
Expected: FAIL — every assertion against `tomato_mesh.*` returns `None`

- [ ] **Step 3: Add the config block**

Append to `config.yaml` (after the existing `corn_mesh:` block):

```yaml
tomato_mesh:
  crop: "Tomato"
  plantdoc_root: "data/PlantDoc"
  plantwild_v1_root: "data/PlantWild/plantwild/plantwild/images"
  plantwild_v2_root: "data/PlantWild/plantwild_v2/plantwild_v2"
  num_nodes: 3
  dirichlet_alpha: 0.3
  test_fraction: 0.20              # global split fraction, deliberately separate from data.test_fraction (per-node, 0.15)
  dedup_threshold: 5               # perceptual-hash Hamming distance (same default as node_1/corn)
  dedup_max_group_size: 25         # starting heuristic (also capped at 5% of a class's pooled count) -- re-tune against real data
  rounds: 5
  output_dir: "outputs/validation/tomato_mesh"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_tomato_mesh.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml tests/test_config_tomato_mesh.py
git commit -m "feat: add tomato_mesh config block"
```

---

### Task 8: Stage 1 — per-node training pipeline

**Files:**
- Create: `src/validation/run_tomato_pipeline.py`
- Test: `tests/test_validation_run_tomato_pipeline.py`

**Interfaces:**
- Consumes: `TomatoMeshData`/`prepare_tomato_mesh_data` (Task 6), `run_training` (`src/validation/train_mobilenet.py`, existing, unmodified), `export_checkpoint` (`src/validation/export_onnx.py`, existing), `run_evaluation` (`src/validation/evaluate_onnx.py`, existing).
- Produces: `node_output_dir(base_output_dir: Path, node_id: str) -> Path`. `run_train_stage(data, node_id, output_dir, epochs=2, pretrained=True) -> None`. `run_export_stage(data, node_id, output_dir) -> None`. `run_evaluate_stage(data, node_id, output_dir) -> None`. `main()` CLI (`--config`, `--output-dir`, `--stage`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_run_tomato_pipeline.py
from __future__ import annotations

import json

import pytest

from src.validation.run_tomato_pipeline import run_evaluate_stage, run_export_stage, run_train_stage
from src.validation.tomato_mesh_dataset import prepare_tomato_mesh_data


def test_run_export_stage_raises_clear_error_when_train_stage_not_run(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="train"):
        run_export_stage(data, "node_0", output_dir)


def test_run_evaluate_stage_raises_clear_error_when_export_stage_not_run(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="export"):
        run_evaluate_stage(data, "node_0", output_dir)


def test_full_pipeline_produces_a_well_formed_report_per_node(tmp_path, tomato_scoped_config):
    data = prepare_tomato_mesh_data(tomato_scoped_config)

    for node_id in sorted(data.per_node.keys()):
        output_dir = tmp_path / node_id
        run_train_stage(data, node_id, output_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, output_dir)
        run_evaluate_stage(data, node_id, output_dir)

        report = json.loads((output_dir / "report.json").read_text())
        assert "summary" in report and "results" in report
        assert report["summary"]["num_test_images"] == len(report["results"])
        assert report["summary"]["num_test_images"] == len(data.test_idx)

        classes = json.loads((output_dir / "classes.json").read_text())
        assert classes["crop_classes"] == ["Tomato"]
        assert len(classes["disease_classes"]) == 10
        assert classes["test_idx"] == data.test_idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_run_tomato_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.run_tomato_pipeline'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/validation/run_tomato_pipeline.py
"""CLI entrypoint chaining the Tomato Dirichlet-mesh validation pipeline
across 3 nodes: train -> export -> evaluate. Mirrors
run_corn_pipeline.py's structure, adapted for a Dirichlet label-skew
partition over a merged multi-source Tomato pool and a single GLOBAL
held-out test set (not per-node) -- see
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.

    python -m src.validation.run_tomato_pipeline
    python -m src.validation.run_tomato_pipeline --stage train
    python -m src.validation.run_tomato_pipeline --stage export
    python -m src.validation.run_tomato_pipeline --stage evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime

from src.config import Config
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.tomato_mesh_dataset import TomatoMeshData, prepare_tomato_mesh_data
from src.validation.train_mobilenet import run_training

STAGES = ("train", "export", "evaluate")
MODEL_NAME = "mobilenet_v3_small"


def node_output_dir(base_output_dir: Path, node_id: str) -> Path:
    return base_output_dir / f"{node_id}_{MODEL_NAME}"


def run_train_stage(
    data: TomatoMeshData, node_id: str, output_dir: Path, epochs: int = 2, pretrained: bool = True
) -> None:
    train_idx = data.per_node[node_id]["train_idx"]
    test_idx = data.test_idx  # shared global test set, not per-node

    run_training(
        data.train_base,
        train_idx,
        data.eval_base,
        test_idx,
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


def run_export_stage(data: TomatoMeshData, node_id: str, output_dir: Path) -> None:
    checkpoint_path = output_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run the 'train' stage first.")
    parity_samples = [data.eval_base[i] for i in data.test_idx[:5]]
    export_checkpoint(
        checkpoint_path,
        data.label_map.crop_classes,
        data.label_map.disease_classes,
        data.image_size,
        output_dir,
        parity_samples=parity_samples,
    )


def run_evaluate_stage(data: TomatoMeshData, node_id: str, output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found — run the 'export' stage first.")
    manifest = json.loads(manifest_path.read_text())
    session = onnxruntime.InferenceSession(str(output_dir / "model.onnx"))
    run_evaluation(session, data.eval_base, data.test_idx, manifest, MODEL_NAME, node_id, output_dir / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/tomato_mesh")
    parser.add_argument("--stage", choices=STAGES, default=None, help="Run only this stage; default runs all")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = Path(args.output_dir)
    data = prepare_tomato_mesh_data(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "test_idx": data.test_idx,
                "partition_diagnostics": data.partition_diagnostics,
            },
            indent=2,
        )
    )

    for node_id in sorted(data.per_node.keys()):
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_run_tomato_pipeline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/validation/run_tomato_pipeline.py tests/test_validation_run_tomato_pipeline.py
git commit -m "feat: add Stage 1 Tomato per-node training pipeline"
```

---

### Task 9: Stage 2 — knowledge-transfer rounds

**Files:**
- Create: `src/validation/run_tomato_knowledge_transfer.py`
- Test: `tests/test_validation_run_tomato_knowledge_transfer.py`

**Interfaces:**
- Consumes: `TomatoMeshData`/`TomatoLabelMap`/`prepare_tomato_mesh_data` (Task 6), `node_output_dir` (Task 8), `run_kt_round`/`_build_per_node_energy_breakdown` (`src/validation/run_knowledge_transfer.py`, existing, reused unchanged via import — `_build_per_node_energy_breakdown` internally calls `_sum_tracked_blocks` itself, so this file does not need to import that helper directly), `Node` (`src/federated/node.py`, existing), `compute_collaboration_gain` (`src/evaluate.py`, existing).
- Produces: `export_and_evaluate_tomato(node, label_map, image_size, eval_base, test_idx, node_dir, keep_onnx) -> dict` (returns `{"test": report}`). `_load_tomato_node_from_checkpoint(checkpoint_path, data, node_id, batch_size) -> Node`. `evaluate_tomato_round0_baseline(data, stage1_dir, node_ids) -> dict[str, dict]`. `_per_class_gain_table_tomato(round0_baseline, final_round_scores, disease_classes, partition_diagnostics) -> dict`. `run_tomato_round_with_io(round_idx, nodes, control_nodes, probe_loader, data, cfg, tracker, comm_estimator, round_dir) -> dict`. `build_tomato_knowledge_transfer_summary(cfg, round0_baseline, round_summaries, data) -> dict`. `main()` CLI (`--config`, `--output-dir`, `--rounds`, `--batch-size`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_run_tomato_knowledge_transfer.py
from __future__ import annotations

import json

import pytest

from src.validation.run_tomato_knowledge_transfer import (
    build_tomato_knowledge_transfer_summary,
    evaluate_tomato_round0_baseline,
)
from src.validation.run_tomato_pipeline import run_evaluate_stage, run_export_stage, run_train_stage
from src.validation.tomato_mesh_dataset import prepare_tomato_mesh_data


def _run_stage1(tmp_path, cfg):
    data = prepare_tomato_mesh_data(cfg)
    stage1_dir = tmp_path / "stage1"
    stage1_dir.mkdir()
    (stage1_dir / "classes.json").write_text(
        json.dumps(
            {
                "crop_classes": data.label_map.crop_classes,
                "disease_classes": data.label_map.disease_classes,
                "image_size": data.image_size,
                "test_idx": data.test_idx,
            }
        )
    )
    node_ids = sorted(data.per_node.keys())
    for node_id in node_ids:
        node_dir = stage1_dir / f"{node_id}_mobilenet_v3_small"
        run_train_stage(data, node_id, node_dir, epochs=1, pretrained=False)
        run_export_stage(data, node_id, node_dir)
        run_evaluate_stage(data, node_id, node_dir)
    return data, stage1_dir, node_ids


def test_round0_baseline_evaluates_every_node_against_global_test_set(tmp_path, tomato_scoped_config):
    data, stage1_dir, node_ids = _run_stage1(tmp_path, tomato_scoped_config)
    baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)
    assert set(baseline.keys()) == set(node_ids)
    for report in baseline.values():
        assert report["summary"]["num_test_images"] == len(data.test_idx)


def test_build_summary_has_appendix_a1_fields(tmp_path, tomato_scoped_config):
    data, stage1_dir, node_ids = _run_stage1(tmp_path, tomato_scoped_config)
    baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)

    # hand-build one fake round_summary matching the real shape, to test
    # the summary builder in isolation from the full round loop
    fake_scores = {
        node_id: {
            "collective": {"test": baseline[node_id]},
            "local_only_control": {"test": baseline[node_id]},
        }
        for node_id in node_ids
    }
    round_summaries = [
        {
            "round": 1,
            "total_bytes_exchanged": 100,
            "energy": {"total_compute_energy_kwh": 0.0},
            "per_node_scores": fake_scores,
        }
    ]
    summary = build_tomato_knowledge_transfer_summary(tomato_scoped_config, baseline, round_summaries, data)
    for key in (
        "node_count",
        "data_split",
        "local_only_budget",
        "collective_budget",
        "fairness_exception_reason",
        "test_set_scope",
        "rounds_run",
        "cumulative_bytes_exchanged",
        "cumulative_energy_kwh",
        "delta_g_formula",
        "per_node_scores",
        "macro_avg_and_worst_node",
        "collaboration_gain_per_disease",
        "limitation_note",
    ):
        assert key in summary
    assert summary["node_count"] == 3
    assert summary["data_split"]["dirichlet_alpha"] == 0.3
    assert "global" in summary["test_set_scope"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validation_run_tomato_knowledge_transfer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.validation.run_tomato_knowledge_transfer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/validation/run_tomato_knowledge_transfer.py
"""Stage 2 for the Tomato Dirichlet-mesh pipeline: knowledge-transfer
rounds on top of the 3 per-node checkpoints stage 1
(run_tomato_pipeline.py) produces, evaluated against the single GLOBAL
held-out test set (not a per-node/cross-node union), plus the
budget-aligned local-only control arm -- see
docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md.
Aggregation mechanics (run_kt_round) are reused unchanged from
run_knowledge_transfer.py -- this pipeline changes the data split, not
the aggregation algorithm.

    python -m src.validation.run_tomato_knowledge_transfer --rounds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.plantvillage import make_subset
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.node import Node
from src.models.factory import build_model
from src.validation.evaluate_onnx import run_evaluation
from src.validation.export_onnx import export_checkpoint
from src.validation.run_knowledge_transfer import _build_per_node_energy_breakdown, run_kt_round
from src.validation.run_tomato_pipeline import node_output_dir
from src.validation.tomato_mesh_dataset import TomatoLabelMap, TomatoMeshData, prepare_tomato_mesh_data

MODEL_NAME = "mobilenet_v3_small"


def export_and_evaluate_tomato(
    node: Node,
    label_map: TomatoLabelMap,
    image_size: int,
    eval_base,
    test_idx: list[int],
    node_dir: Path,
    keep_onnx: bool,
) -> dict:
    """Same shape as run_knowledge_transfer.export_and_evaluate, but
    against the ONE shared global test set: report.json = {"test": ...}
    rather than {"local": ..., "cross_node": ...}, since there is no
    per-node test split to distinguish from a cross-node union here.
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
    test_report = run_evaluation(session, eval_base, test_idx, manifest, MODEL_NAME, node.node_id, scratch)
    scratch.unlink(missing_ok=True)

    combined = {"test": test_report}
    (node_dir / "report.json").write_text(json.dumps(combined, indent=2))

    if not keep_onnx:
        onnx_path.unlink(missing_ok=True)
        (node_dir / "manifest.json").unlink(missing_ok=True)

    return combined


def _load_tomato_node_from_checkpoint(
    checkpoint_path: Path, data: TomatoMeshData, node_id: str, batch_size: int
) -> Node:
    model = build_model(
        MODEL_NAME, len(data.label_map.crop_classes), len(data.label_map.disease_classes), pretrained=False
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    train_idx = data.per_node[node_id]["train_idx"]
    train_loader = DataLoader(
        make_subset(data.train_base, train_idx), batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(make_subset(data.eval_base, data.test_idx), batch_size=batch_size, shuffle=False)
    return Node(node_id, model, train_loader, test_loader, device="cpu")


def evaluate_tomato_round0_baseline(data: TomatoMeshData, stage1_dir: Path, node_ids: list[str]) -> dict[str, dict]:
    """Evaluates each node's ALREADY-EXPORTED stage-1 model.onnx (no
    re-export) against the shared global test set -- this is round 0's
    baseline for the per-class collaboration-gain table.
    """
    baseline: dict[str, dict] = {}
    for node_id in node_ids:
        node_dir = node_output_dir(stage1_dir, node_id)
        manifest = json.loads((node_dir / "manifest.json").read_text())
        session = onnxruntime.InferenceSession(str(node_dir / "model.onnx"))
        scratch = node_dir / "_round0_scratch.json"
        report = run_evaluation(session, data.eval_base, data.test_idx, manifest, MODEL_NAME, node_id, scratch)
        scratch.unlink(missing_ok=True)
        baseline[node_id] = report
    return baseline


def _per_class_gain_table_tomato(
    round0_baseline: dict[str, dict],
    final_round_scores: dict[str, dict],
    disease_classes: list[str],
    partition_diagnostics: dict[str, dict],
) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for node_id, baseline_report in round0_baseline.items():
        table[node_id] = {}
        collective_summary = final_round_scores[node_id]["collective"]["test"]["summary"]
        control_summary = final_round_scores[node_id]["local_only_control"]["test"]["summary"]
        low_rep = set(partition_diagnostics[node_id]["low_representation_classes"])
        for disease_name in disease_classes:
            round0_acc = baseline_report["summary"]["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            collective_acc = collective_summary["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            control_acc = control_summary["per_class_accuracy"]["disease"].get(disease_name, 0.0)
            table[node_id][disease_name] = {
                "round_0_accuracy": round0_acc,
                "round_N_collective_accuracy": collective_acc,
                "round_N_local_only_control_accuracy": control_acc,
                "gain_vs_round0": collective_acc - round0_acc,
                "gain_vs_local_only_control": collective_acc - control_acc,
                "low_representation_for_this_node": disease_name in low_rep,
            }
    return table


def run_tomato_round_with_io(
    round_idx: int,
    nodes: dict[str, Node],
    control_nodes: dict[str, Node],
    probe_loader: DataLoader,
    data: TomatoMeshData,
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
        round_idx=round_idx,
    )

    per_node_scores: dict[str, dict] = {}
    for node_id in nodes:
        collective_report = export_and_evaluate_tomato(
            nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            data.test_idx,
            round_dir / node_id,
            keep_onnx=True,
        )
        control_report = export_and_evaluate_tomato(
            control_nodes[node_id],
            data.label_map,
            data.image_size,
            data.eval_base,
            data.test_idx,
            round_dir / node_id / "local_only_control",
            keep_onnx=False,
        )
        per_node_scores[node_id] = {"collective": collective_report, "local_only_control": control_report}

    comm_estimate = comm_estimator.estimate_all_radios(kt_result["total_bytes_exchanged"])
    energy_summary = tracker.summary()
    per_node_energy, per_node_local_only_control_energy = _build_per_node_energy_breakdown(
        tracker.log, nodes.keys(), round_idx
    )
    energy_summary = {
        **energy_summary,
        "per_node": per_node_energy,
        "per_node_local_only_control": per_node_local_only_control_energy,
    }

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


def build_tomato_knowledge_transfer_summary(
    cfg: Config, round0_baseline: dict[str, dict], round_summaries: list[dict], data: TomatoMeshData
) -> dict:
    final_round = round_summaries[-1]
    final_scores = final_round["per_node_scores"]

    collective_evals = {nid: s["collective"]["test"]["summary"] for nid, s in final_scores.items()}
    control_evals = {nid: s["local_only_control"]["test"]["summary"] for nid, s in final_scores.items()}
    gain = compute_collaboration_gain(
        {
            nid: {"crop_accuracy": s["crop_accuracy"], "disease_accuracy": s["disease_accuracy"]}
            for nid, s in collective_evals.items()
        },
        {
            nid: {"crop_accuracy": s["crop_accuracy"], "disease_accuracy": s["disease_accuracy"]}
            for nid, s in control_evals.items()
        },
    )

    dirichlet_alpha = cfg.get("tomato_mesh.dirichlet_alpha", 0.3)
    cumulative_bytes = sum(r["total_bytes_exchanged"] for r in round_summaries)
    cumulative_energy = round_summaries[-1]["energy"]["total_compute_energy_kwh"]

    return {
        "node_count": len(data.per_node),
        "data_split": {
            "strategy": (
                f"Dirichlet label-skew partition (alpha={dirichlet_alpha}) over a merged, "
                "deduplicated multi-source Tomato pool (PlantVillage, PlantDoc, PlantWild v1+v2), "
                "with a global test split carved out before partitioning"
            ),
            "strength": (
                "every node has some training exposure to all "
                f"{len(data.label_map.disease_classes)} canonical disease classes; skew is "
                "proportional (Dirichlet-drawn), not exclusionary"
            ),
            "sources": ["PlantVillage", "PlantDoc", "PlantWild_v1", "PlantWild_v2"],
            "dirichlet_alpha": dirichlet_alpha,
            "partition_diagnostics": data.partition_diagnostics,
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
            "single global held-out test set, shared across all nodes, carved out before "
            "Dirichlet partitioning; test samples never enter any training set"
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
        "collaboration_gain_per_disease": _per_class_gain_table_tomato(
            round0_baseline, final_scores, data.label_map.disease_classes, data.partition_diagnostics
        ),
        "limitation_note": (
            "Cross-class-representation gain above comes from probe-set logit distillation only, "
            "not prototype alignment — prototype exchange only reinforces classes a node already "
            "has local samples for, since a node's proto_loss only runs over its own local batches."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo root)")
    parser.add_argument("--output-dir", default="outputs/validation/tomato_mesh")
    parser.add_argument("--rounds", type=int, default=None, help="Number of knowledge-transfer rounds (1-5)")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    rounds = args.rounds if args.rounds is not None else cfg.get("tomato_mesh.rounds", 5)
    if not (1 <= rounds <= 5):
        parser.error(f"--rounds must be between 1 and 5, got {rounds}")

    stage1_dir = Path(args.output_dir)
    data = prepare_tomato_mesh_data(cfg)
    node_ids = sorted(data.per_node.keys())

    for node_id in node_ids:
        node_dir = node_output_dir(stage1_dir, node_id)
        for required_name in ("checkpoint.pt", "model.onnx", "manifest.json", "classes.json"):
            required_path = node_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(
                    f"{required_path} not found — run 'python -m src.validation.run_tomato_pipeline' first."
                )

    top_level_classes_path = stage1_dir / "classes.json"
    if not top_level_classes_path.exists():
        raise FileNotFoundError(f"{top_level_classes_path} not found — run run_tomato_pipeline.py first.")
    top_level_classes = json.loads(top_level_classes_path.read_text())
    if top_level_classes.get("test_idx") != data.test_idx:
        raise ValueError(
            f"{top_level_classes_path}: the global test split persisted by stage 1 does not match "
            f"the split prepare_tomato_mesh_data(cfg) produces now. Stage 1 and stage 2 must be run "
            f"against the same config and the same source data — otherwise stage 2 would silently "
            f"evaluate checkpoints on images they were trained on."
        )
    for node_id in node_ids:
        classes_path = node_output_dir(stage1_dir, node_id) / "classes.json"
        stage1_classes = json.loads(classes_path.read_text())
        if stage1_classes.get("train_idx") != data.per_node[node_id]["train_idx"]:
            raise ValueError(
                f"{node_id}: the train split persisted in {classes_path} by stage 1 "
                f"(run_tomato_pipeline.py) does not match the split prepare_tomato_mesh_data(cfg) "
                f"produces now. Stage 1 and stage 2 must be run against the same config and the "
                f"same source data — otherwise stage 2 would silently evaluate this node's "
                f"checkpoint on images it was trained on."
            )

    print("=== round 0 baseline (stage 1 checkpoints, global test set) ===")
    round0_baseline = evaluate_tomato_round0_baseline(data, stage1_dir, node_ids)

    kt_dir = stage1_dir / "knowledge_transfer"
    kt_dir.mkdir(parents=True, exist_ok=True)
    (kt_dir / "round_0_baseline.json").write_text(json.dumps(round0_baseline, indent=2))

    nodes = {
        node_id: _load_tomato_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in node_ids
    }
    control_nodes = {
        node_id: _load_tomato_node_from_checkpoint(
            node_output_dir(stage1_dir, node_id) / "checkpoint.pt", data, node_id, args.batch_size
        )
        for node_id in node_ids
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
        summary = run_tomato_round_with_io(
            round_idx, nodes, control_nodes, probe_loader, data, cfg, tracker, comm_estimator, round_dir
        )
        round_summaries.append(summary)

    kt_summary = build_tomato_knowledge_transfer_summary(cfg, round0_baseline, round_summaries, data)
    (kt_dir / "knowledge_transfer_summary.json").write_text(json.dumps(kt_summary, indent=2))
    print(f"Done. Knowledge-transfer outputs in {kt_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validation_run_tomato_knowledge_transfer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full new test suite together**

Run: `pytest tests/test_data_plantdoc.py tests/test_data_plantwild.py tests/test_validation_tomato_mesh_dataset.py tests/test_config_tomato_mesh.py tests/test_validation_run_tomato_pipeline.py tests/test_validation_run_tomato_knowledge_transfer.py -v`
Expected: PASS (every test across all 6 new/modified test files)

- [ ] **Step 6: Commit**

```bash
git add src/validation/run_tomato_knowledge_transfer.py tests/test_validation_run_tomato_knowledge_transfer.py
git commit -m "feat: add Stage 2 Tomato knowledge-transfer rounds against global test set"
```

---

## Not covered by this plan (see spec's "Out of scope")

- Empirically re-tuning `dedup_max_group_size`/`dedup_threshold` against the real merged pool's actual hash-distance histogram (needs a real run against `data/PlantVillage` + `data/PlantDoc` + `data/PlantWild` — the synthetic test fixtures use random-noise images that won't exercise realistic duplicate clustering).
- Visually confirming PlantDoc's `"Tomato leaf"` and PlantWild v1's `"tomato leaf"` folders are genuinely healthy leaves.
- Running the real `python -m src.validation.run_tomato_pipeline` / `run_tomato_knowledge_transfer` against the actual downloaded datasets and interpreting the results.
- AdaClass or any other aggregation-weighting change, `efficientnet_lite0`/`mobilevit_xxs`, or applying this pattern to crops other than Tomato.
