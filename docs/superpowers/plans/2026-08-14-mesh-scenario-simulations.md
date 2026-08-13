# Mesh Scenario Simulations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three runnable scenario simulations (node disconnection/reconnection, runtime class addition, distribution shift) that demonstrate the decentralised mesh (`src/federated/`) continuing to function — and recovering faster than a no-exchange baseline — under each disruption, with results written to JSON.

**Architecture:** A minimal, backward-compatible extension to `Node`/`MeshSimulator` adds an `active` flag; a new `src/scenarios/` package provides a shared round-driver (`harness.py`) plus one independently-runnable script per scenario, each building a no-exchange baseline node set and a mesh node set from the same starting data shards (reusing `src.train.build_dataloaders`) and applying the identical perturbation to both.

**Tech Stack:** Python, PyTorch, torchvision, pytest — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-14-mesh-scenario-simulations-design.md`

## Global Constraints

- No real network/transport layer — "disconnection" is simulated by excluding a node from the broadcast/aggregation step of the existing synchronous, fully-connected `MeshSimulator.run_round`, not by modelling an actual network partition.
- `Node.active` must default to `True` and `RoundLog.active_nodes` must be additive — every existing test in `tests/test_pipeline.py` must keep passing unmodified in behavior (only its fixture *location* changes, in Task 1).
- Output is JSON only, under `outputs/scenarios/{name}.json` — no `.log` file.
- No new config-loading code — new config lives under a `scenarios:` top-level key in `config.yaml`, read via the existing `Config.get("scenarios.xxx.yyy", default)` dotted-key accessor.
- All new tests must run against the `synthetic_dataset` pytest fixture (tiny in-memory images), never the real PlantVillage dataset — this keeps `pytest` runnable immediately after `pip install -r requirements.txt`, matching the existing `tests/test_pipeline.py` convention.
- Reuse existing code rather than duplicating it: `src.train.build_dataloaders` for partitioning/loading, `src.evaluate.compute_collaboration_gain` for the mesh-vs-baseline metric, `src.data.plantvillage.train_test_split_indices`/`make_subset` for index-level dataset operations.
- Avoid the Python `X | Y` runtime union syntax (needs 3.10+) in non-annotation (eagerly-evaluated) code — use `typing.Optional` instead, matching the safer pattern.

---

## Task 1: Move shared test fixtures into `tests/conftest.py`

Pure refactor with no behavior change — needed so the new `tests/test_scenarios.py` (Tasks 2-6) can reuse the same synthetic dataset and node-building helper that `tests/test_pipeline.py` already defines, without duplicating them.

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_pipeline.py` (remove the fixture/helper it currently defines, import them instead)

**Interfaces:**
- Produces: `synthetic_dataset` pytest fixture (auto-discovered by pytest for every test module under `tests/`, no import needed) — yields a `PlantVillageDataset` over a tiny synthetic image tree (4 classes, 8 images each, 32x32).
- Produces: `build_nodes(dataset, shards, num_crop, num_disease, arch="mobilenet_v3_small") -> list[Node]` — a plain function (not a fixture), imported explicitly via `from tests.conftest import build_nodes`.

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures and helpers for the mesh/scenario test suite."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data.plantvillage import PlantVillageDataset, make_subset, train_test_split_indices
from src.federated.node import Node
from src.models.factory import build_model

CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


@pytest.fixture
def synthetic_dataset(tmp_path):
    root = tmp_path / "PlantVillage"
    rng = np.random.RandomState(0)
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(8):
            arr = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i}.jpg")
    return PlantVillageDataset(root, image_size=32)


def build_nodes(dataset, shards, num_crop, num_disease, arch="mobilenet_v3_small"):
    nodes = []
    for i, shard in enumerate(shards):
        train_idx, test_idx = train_test_split_indices(shard, test_fraction=0.3, seed=1)
        train_loader = DataLoader(make_subset(dataset, train_idx), batch_size=4, shuffle=True)
        test_loader = DataLoader(make_subset(dataset, test_idx), batch_size=4, shuffle=False)
        model = build_model(arch, num_crop, num_disease, pretrained=False)
        nodes.append(Node(f"node_{i}", model, train_loader, test_loader, device="cpu"))
    return nodes
```

- [ ] **Step 2: Rewrite `tests/test_pipeline.py` to use the shared fixture/helper**

Replace the entire file with:

```python
"""End-to-end smoke test on synthetic images — exercises the full
pipeline (label parsing, non-IID partition, model build, one mesh round,
energy/communication accounting) without needing the real PlantVillage
dataset, so `pytest` works right after `pip install -r requirements.txt`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.plantvillage import carve_public_probe_set, make_subset, partition_nodes
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.models.factory import build_model, count_parameters, model_size_mb

from tests.conftest import build_nodes


def test_label_parsing(synthetic_dataset):
    assert set(synthetic_dataset.labels.crop_classes) == {"Tomato", "Potato"}
    assert "healthy" in synthetic_dataset.labels.disease_classes
    assert "Bacterial_spot" in synthetic_dataset.labels.disease_classes


def test_partition_by_crop_is_disjoint_and_complete(synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=0
    )
    assert set(probe_idx).isdisjoint(set(remaining_idx))
    all_shard_idx = [i for shard in shards for i in shard]
    assert sorted(all_shard_idx) == sorted(remaining_idx)


def test_partition_manual_assigns_named_crops_to_named_nodes(synthetic_dataset):
    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset,
        remaining_idx,
        num_nodes=2,
        strategy="manual",
        dirichlet_alpha=0.3,
        seed=0,
        manual_node_crops={"node_0": ["Tomato"], "node_1": ["Potato"]},
    )
    class_to_crop = {c: cd[0] for c, cd in synthetic_dataset.labels.class_to_crop_disease.items()}
    crops_per_shard = [
        {synthetic_dataset.labels.crop_classes[class_to_crop[synthetic_dataset.targets[i]]] for i in shard}
        for shard in shards
    ]
    assert crops_per_shard[0] == {"Tomato"}
    assert crops_per_shard[1] == {"Potato"}


def test_partition_manual_rejects_incomplete_crop_assignment(synthetic_dataset):
    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    with pytest.raises(ValueError, match="missing an assignment"):
        partition_nodes(
            synthetic_dataset,
            remaining_idx,
            num_nodes=2,
            strategy="manual",
            dirichlet_alpha=0.3,
            seed=0,
            manual_node_crops={"node_0": ["Tomato"]},
        )


@pytest.mark.parametrize("arch", ["mobilenet_v3_small", "efficientnet_lite0", "mobilevit_xxs"])
def test_model_forward_shapes(synthetic_dataset, arch):
    model = build_model(arch, num_crop_classes=2, num_disease_classes=3, pretrained=False)
    images = torch.stack([synthetic_dataset[i][0] for i in range(4)])
    crop_logits, disease_logits = model(images)
    assert crop_logits.shape == (4, 2)
    assert disease_logits.shape == (4, 3)
    assert count_parameters(model) > 0
    assert model_size_mb(model) > 0


def test_end_to_end_mesh_round_beats_no_exchange_smoke(synthetic_dataset):
    """Not a statistical claim (too little synthetic data for that) —
    just proves the mesh round runs, exchanges only prototypes/logits,
    and produces a well-formed collaboration-gain report.
    """
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=1)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=1
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)

    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    baseline_evals = {}
    for node in baseline_nodes:
        node.local_train(epochs=1, lr=1e-3)
        baseline_evals[node.node_id] = node.evaluate()

    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )
    round_log = mesh.run_round(
        0,
        local_epochs=1,
        distill_epochs=1,
        lr=1e-3,
        distill_lr=1e-3,
        proto_weight=0.5,
        kd_weight=0.5,
        temperature=2.0,
    )

    assert round_log.total_bytes_exchanged > 0
    mesh_evals = {node.node_id: node.evaluate() for node in mesh_nodes}

    gain = compute_collaboration_gain(mesh_evals, baseline_evals)
    assert "macro_gain" in gain
    assert "worst_node_gain" in gain
    assert set(gain["per_node"].keys()) == {"node_0", "node_1"}


def test_energy_and_communication_tracking(tmp_path):
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    with tracker.track("dummy_block") as record:
        _ = sum(range(1000))
    assert record["method"] == "proxy_wall_power"
    assert record["energy_kwh"] >= 0
    assert tracker.summary()["num_tracked_blocks"] == 1

    estimator = CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )
    result = estimator.estimate(total_bytes=10_000, radio="wifi")
    assert result["energy_kwh"] > 0
    assert result["co2_kg"] > 0
```

- [ ] **Step 3: Run the existing suite to confirm nothing broke**

Run: `pytest tests/test_pipeline.py -v`
Expected: all 7 tests PASS, identical to before the refactor.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_pipeline.py
git commit -m "Move shared test fixtures into tests/conftest.py for reuse by scenario tests"
```

---

## Task 2: Extend `Node` and `MeshSimulator` with an `active` flag

**Files:**
- Modify: `src/federated/node.py:34-46` (`Node.__init__`)
- Modify: `src/federated/mesh.py:16-23` (`RoundLog`) and `src/federated/mesh.py:41-111` (`MeshSimulator.run_round`)
- Create: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `build_nodes(dataset, shards, num_crop, num_disease, arch=...)` and `synthetic_dataset` fixture from Task 1.
- Produces: `Node(..., active: bool = True)` — new keyword param, `node.active` attribute. `RoundLog.active_nodes: list[str]` — new field, default `[]`.

- [ ] **Step 1: Write the failing tests in `tests/test_scenarios.py`**

```python
"""Tests for the disconnection/class-addition/distribution-shift scenario
simulations, and the core mesh 'active' flag they rely on.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.plantvillage import (
    carve_public_probe_set,
    make_subset,
    partition_nodes,
    train_test_split_indices,
)
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model
from src.scenarios.harness import run_scenario, write_scenario_report

from tests.conftest import build_nodes


def test_node_active_defaults_to_true(synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=0)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=1, strategy="by_crop", dirichlet_alpha=0.3, seed=0
    )
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    node = build_nodes(synthetic_dataset, shards, num_crop, num_disease)[0]
    assert node.active is True


def test_inactive_node_excluded_from_broadcast_and_distill(synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=1)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=3, strategy="by_crop", dirichlet_alpha=0.3, seed=1
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    nodes[0].active = False

    mesh = MeshSimulator(
        nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )
    round_log = mesh.run_round(
        0, local_epochs=1, distill_epochs=1, lr=1e-3, distill_lr=1e-3,
        proto_weight=0.5, kd_weight=0.5, temperature=2.0,
    )

    assert round_log.active_nodes == ["node_1", "node_2"]
    # the disconnected node still trained locally and was evaluated...
    assert "node_0" in round_log.per_node_train_loss
    assert "node_0" in round_log.per_node_eval
    # ...but never distilled towards a peer consensus this round.
    assert "node_0" not in round_log.per_node_distill_loss
    # active peers still exchanged and distilled as normal.
    assert "node_1" in round_log.per_node_distill_loss
    assert "node_2" in round_log.per_node_distill_loss
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scenarios.py -v`
Expected: FAIL — `TypeError: Node.__init__() got an unexpected keyword argument` is not raised (since `active` is set as a plain attribute, not via constructor, in this test — instead expect `AttributeError` or the assertion `node.active is True` failing with `AttributeError: 'Node' object has no attribute 'active'`), and a collection error on `from src.scenarios.harness import ...` since that module doesn't exist yet (expected — Task 3 creates it). For now, comment out the `from src.scenarios.harness import ...` line so this task's two tests can run in isolation; Task 3 will add it back.

- [ ] **Step 3: Implement `Node.active` in `src/federated/node.py`**

Change:
```python
    def __init__(
        self,
        node_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
    ):
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
```
to:
```python
    def __init__(
        self,
        node_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        active: bool = True,
    ):
        self.node_id = node_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.active = active
```

- [ ] **Step 4: Implement `RoundLog.active_nodes` and the filtering logic in `src/federated/mesh.py`**

Change the `RoundLog` dataclass from:
```python
@dataclass
class RoundLog:
    round_idx: int
    per_node_train_loss: dict[str, float] = field(default_factory=dict)
    # node_id -> {"kd_loss", "sup_loss", "proto_loss", "total_loss"}
    per_node_distill_loss: dict[str, dict[str, float]] = field(default_factory=dict)
    per_node_eval: dict[str, dict[str, float]] = field(default_factory=dict)
    total_bytes_exchanged: int = 0
```
to:
```python
@dataclass
class RoundLog:
    round_idx: int
    per_node_train_loss: dict[str, float] = field(default_factory=dict)
    # node_id -> {"kd_loss", "sup_loss", "proto_loss", "total_loss"}
    per_node_distill_loss: dict[str, dict[str, float]] = field(default_factory=dict)
    per_node_eval: dict[str, dict[str, float]] = field(default_factory=dict)
    total_bytes_exchanged: int = 0
    active_nodes: list[str] = field(default_factory=list)
```

Change `MeshSimulator.run_round` from:
```python
        log = RoundLog(round_idx=round_idx)

        # 1) local supervised training, private data never leaves this loop
        for node in self.nodes:
            log.per_node_train_loss[node.node_id] = node.local_train(local_epochs, lr)

        # 2) each node computes its small, non-invertible knowledge payload
        payloads: dict[str, KnowledgePayload] = {
            node.node_id: node.compute_knowledge(self.probe_loader) for node in self.nodes
        }

        # simulate a fully-connected broadcast: every payload is sent to
        # every OTHER peer once (an upper bound — a real gossip relay with
        # partial connectivity would use less bandwidth than this).
        n = len(self.nodes)
        log.total_bytes_exchanged = sum(p.size_bytes() for p in payloads.values()) * (n - 1)

        # 3) each node aggregates what it received from PEERS (excluding
        #    its own payload) with a robust rule, then distils towards it
        for node in self.nodes:
            peer_payloads = [p for nid, p in payloads.items() if nid != node.node_id]
            if not peer_payloads:
                continue  # single-node mesh: nothing to reconcile
```
to:
```python
        log = RoundLog(round_idx=round_idx)
        log.active_nodes = [node.node_id for node in self.nodes if node.active]

        # 1) local supervised training, private data never leaves this loop.
        # Disconnected nodes keep training locally — they drift, but are
        # not frozen.
        for node in self.nodes:
            log.per_node_train_loss[node.node_id] = node.local_train(local_epochs, lr)

        # 2) each ACTIVE node computes its small, non-invertible knowledge
        # payload. A disconnected node's payload never enters the pool.
        payloads: dict[str, KnowledgePayload] = {
            node.node_id: node.compute_knowledge(self.probe_loader)
            for node in self.nodes
            if node.active
        }

        # simulate a fully-connected broadcast among the currently-connected
        # nodes only: every payload is sent to every other ACTIVE peer once
        # (an upper bound — a real gossip relay with partial connectivity
        # would use less bandwidth than this).
        active_n = len(payloads)
        log.total_bytes_exchanged = sum(p.size_bytes() for p in payloads.values()) * max(0, active_n - 1)

        # 3) each ACTIVE node aggregates what it received from PEERS
        # (excluding its own payload) with a robust rule, then distils
        # towards it. A disconnected node receives nothing and is skipped.
        for node in self.nodes:
            if not node.active:
                continue
            peer_payloads = [p for nid, p in payloads.items() if nid != node.node_id]
            if not peer_payloads:
                continue  # single-node mesh: nothing to reconcile
```

The rest of the method (aggregation, `distill` call, and the final evaluation loop) is unchanged — the evaluation loop already iterates `for node in self.nodes` unconditionally, which is what we want (a disconnected node is still evaluated every round).

- [ ] **Step 5: Re-enable the harness import in `tests/test_scenarios.py`**

Uncomment (or leave in place) the `from src.scenarios.harness import run_scenario, write_scenario_report` line — it will start resolving once Task 3 creates that module. For now, since Task 3 hasn't run yet, temporarily comment it out again if it blocks collection, and run only the two tests added in this task:

Run: `pytest tests/test_scenarios.py -v -k "active"`
Expected: both tests PASS.

- [ ] **Step 6: Run the full existing suite to confirm no regression**

Run: `pytest tests/test_pipeline.py -v`
Expected: all 7 tests still PASS (default `active=True` means the existing mesh smoke test is unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/federated/node.py src/federated/mesh.py tests/test_scenarios.py
git commit -m "Add Node.active flag and MeshSimulator round-filtering for disconnected nodes"
```

---

## Task 3: Shared scenario harness (`src/scenarios/harness.py`)

**Files:**
- Create: `src/scenarios/__init__.py` (empty, matches the empty `__init__.py` convention used by every other `src/` subpackage)
- Create: `src/scenarios/harness.py`
- Modify: `tests/test_scenarios.py` (uncomment the harness import from Task 2, add a new test)

**Interfaces:**
- Consumes: `Node`, `MeshSimulator` (from `src.federated`), `compute_collaboration_gain` (from `src.evaluate`), `build_model` (from `src.models.factory`).
- Produces (used by Tasks 4-6):
  - `ScenarioEvent(round_idx: int, event_type: str, node_id: str, details: dict)` — dataclass.
  - `ScenarioRoundRecord(round_idx, baseline_eval, mesh_eval, collaboration_gain, events)` — dataclass.
  - `node_ids_for(node_loaders) -> list[str]`
  - `require_target_node(target_node: str, node_loaders) -> None` (raises `ValueError`)
  - `build_node_set(cfg, arch: str, node_loaders, num_crop: int, num_disease: int, device: str) -> list[Node]`
  - `run_scenario(baseline_nodes, mesh, num_rounds, perturbation_hook, round_kwargs) -> list[ScenarioRoundRecord]`
  - `write_scenario_report(output_dir, scenario_name, target_node_id, disruption_start_round, disruption_end_round, config_snapshot, records) -> Path`

- [ ] **Step 1: Create `src/scenarios/__init__.py`**

Empty file (0 bytes) — matches `src/federated/__init__.py`, `src/models/__init__.py`, etc.

- [ ] **Step 2: Create `src/scenarios/harness.py`**

```python
"""Shared round-driver for the mesh scenario simulations (node
disconnection, runtime class addition, distribution shift): runs a
no-exchange baseline and the mesh side by side under the same
perturbation schedule, and writes the comparison to JSON.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Optional

from src.evaluate import compute_collaboration_gain
from src.federated.mesh import MeshSimulator
from src.federated.node import Node
from src.models.factory import build_model

RECOVERY_TOLERANCE = 0.05  # accuracy points a node must be within to count as "recovered"


@dataclasses.dataclass
class ScenarioEvent:
    round_idx: int
    event_type: str  # "disconnect" | "reconnect" | "class_added" | "shift_applied"
    node_id: str
    details: dict


@dataclasses.dataclass
class ScenarioRoundRecord:
    round_idx: int
    baseline_eval: dict[str, dict[str, float]]
    mesh_eval: dict[str, dict[str, float]]
    collaboration_gain: dict
    events: list[ScenarioEvent] = dataclasses.field(default_factory=list)


PerturbationHook = Callable[[int, "list[Node]", Optional[MeshSimulator]], "list[ScenarioEvent]"]


def node_ids_for(node_loaders) -> list[str]:
    return [f"node_{i}" for i in range(len(node_loaders))]


def require_target_node(target_node: str, node_loaders) -> None:
    ids = node_ids_for(node_loaders)
    if target_node not in ids:
        raise ValueError(f"target_node '{target_node}' is not one of {ids}")


def build_node_set(cfg, arch: str, node_loaders, num_crop: int, num_disease: int, device: str) -> list[Node]:
    """Builds one fresh, independent Node per shard in `node_loaders` — used
    to construct both the no-exchange baseline set and the mesh set from
    the same starting shards.
    """
    nodes = []
    for i, (train_loader, test_loader) in enumerate(node_loaders):
        model = build_model(arch, num_crop, num_disease, pretrained=cfg.get("models.pretrained", True))
        nodes.append(Node(f"node_{i}", model, train_loader, test_loader, device=device))
    return nodes


def run_scenario(
    baseline_nodes: list[Node],
    mesh: MeshSimulator,
    num_rounds: int,
    perturbation_hook: PerturbationHook,
    round_kwargs: dict,
) -> list[ScenarioRoundRecord]:
    """Drives `num_rounds` rounds of a baseline (no-exchange) node set and a
    mesh node set through the same perturbation schedule.

    Convention: `perturbation_hook` is called once per round for the
    baseline set (third argument None) and once for the mesh set (third
    argument the MeshSimulator) — it should mutate whichever node set it's
    given, but only return a non-empty list of ScenarioEvents on the mesh
    call, so each real-world event is recorded once even though it is
    applied identically to both parallel simulations.
    """
    records: list[ScenarioRoundRecord] = []
    for round_idx in range(num_rounds):
        events = perturbation_hook(round_idx, baseline_nodes, None)
        events = events + perturbation_hook(round_idx, mesh.nodes, mesh)

        baseline_eval = {}
        for node in baseline_nodes:
            node.local_train(round_kwargs["local_epochs"], round_kwargs["lr"])
            baseline_eval[node.node_id] = node.evaluate()

        mesh.run_round(
            round_idx,
            local_epochs=round_kwargs["local_epochs"],
            distill_epochs=round_kwargs["distill_epochs"],
            lr=round_kwargs["lr"],
            distill_lr=round_kwargs["distill_lr"],
            proto_weight=round_kwargs["proto_weight"],
            kd_weight=round_kwargs["kd_weight"],
            temperature=round_kwargs["temperature"],
        )
        mesh_eval = {node.node_id: node.evaluate() for node in mesh.nodes}

        gain = compute_collaboration_gain(mesh_eval, baseline_eval)
        records.append(ScenarioRoundRecord(round_idx, baseline_eval, mesh_eval, gain, events))
        print(f"  round {round_idx}: {len(events)} event(s), macro_gain={gain['macro_gain']}")
    return records


def _recovery_round(
    records: list[ScenarioRoundRecord],
    node_id: str,
    disruption_start_round: int,
    disruption_end_round: int,
    eval_key: str,
) -> Optional[int]:
    """First round index >= disruption_end_round where `node_id`'s eval
    (from `eval_key`, "mesh_eval" or "baseline_eval") is back within
    RECOVERY_TOLERANCE of its value from the round right before the
    disruption started, or None if it never recovers within the run.
    """
    pre_round = max(0, disruption_start_round - 1)
    pre_eval = getattr(records[pre_round], eval_key).get(node_id)
    if pre_eval is None:
        return None
    for record in records:
        if record.round_idx < disruption_end_round:
            continue
        current = getattr(record, eval_key).get(node_id)
        if current is None:
            continue
        if all(abs(current[m] - pre_eval[m]) <= RECOVERY_TOLERANCE for m in pre_eval):
            return record.round_idx
    return None


def write_scenario_report(
    output_dir: Path,
    scenario_name: str,
    target_node_id: str,
    disruption_start_round: int,
    disruption_end_round: int,
    config_snapshot: dict,
    records: list[ScenarioRoundRecord],
) -> Path:
    """Writes outputs/scenarios/{scenario_name}.json and returns its path."""
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "scenario": scenario_name,
        "target_node": target_node_id,
        "config": config_snapshot,
        "rounds": [dataclasses.asdict(r) for r in records],
        "summary": {
            "recovery_round_mesh": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "mesh_eval"
            ),
            "recovery_round_baseline": _recovery_round(
                records, target_node_id, disruption_start_round, disruption_end_round, "baseline_eval"
            ),
        },
    }
    path = scenarios_dir / f"{scenario_name}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
```

- [ ] **Step 3: Uncomment the harness import in `tests/test_scenarios.py` and add a harness-level test**

Ensure the top of `tests/test_scenarios.py` has (uncommented):
```python
from src.scenarios.harness import run_scenario, write_scenario_report
```

Append this test to `tests/test_scenarios.py`:

```python
def test_run_scenario_and_write_report(tmp_path, synthetic_dataset):
    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=2)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=2
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    def no_op_hook(round_idx, nodes, mesh_or_none):
        return []

    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=no_op_hook, round_kwargs=round_kwargs)
    assert len(records) == 2
    assert records[0].round_idx == 0 and records[1].round_idx == 1

    report_path = write_scenario_report(
        tmp_path, "unit_test_scenario", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"note": "test"}, records=records,
    )
    assert report_path == tmp_path / "scenarios" / "unit_test_scenario.json"
    written = json.loads(report_path.read_text())
    assert written["scenario"] == "unit_test_scenario"
    assert written["target_node"] == "node_0"
    assert len(written["rounds"]) == 2
    assert "recovery_round_mesh" in written["summary"]
    assert "recovery_round_baseline" in written["summary"]
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scenarios.py -v`
Expected: all 3 tests PASS (the two from Task 2 plus this new one).

- [ ] **Step 5: Commit**

```bash
git add src/scenarios/__init__.py src/scenarios/harness.py tests/test_scenarios.py
git commit -m "Add shared scenario harness: run_scenario, write_scenario_report, build_node_set"
```

---

## Task 4: Disconnection scenario (`src/scenarios/disconnection.py`)

**Files:**
- Create: `src/scenarios/disconnection.py`
- Modify: `config.yaml` (add `scenarios:` section)
- Modify: `tests/test_scenarios.py` (add tests)

**Interfaces:**
- Consumes: everything from Task 3's `harness.py`, plus `Config` (`src.config`), `load_full_dataset` (`src.data.plantvillage`), `build_dataloaders` (`src.train`).
- Produces: `make_disconnect_hook(target_node: str, disconnect_round: int, reconnect_round: int) -> PerturbationHook`, and a `main()` CLI entry point runnable as `python -m src.scenarios.disconnection`.

- [ ] **Step 1: Write the failing smoke test in `tests/test_scenarios.py`**

Append:

```python
def test_disconnection_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.disconnection import make_disconnect_hook

    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=3)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=3, strategy="by_crop", dirichlet_alpha=0.3, seed=3
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_disconnect_hook("node_1", disconnect_round=1, reconnect_round=2)
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=3, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "disconnection", "node_1",
        disruption_start_round=1, disruption_end_round=2,
        config_snapshot={"target_node": "node_1", "disconnect_round": 1, "reconnect_round": 2},
        records=records,
    )
    report = json.loads(report_path.read_text())

    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["disconnect", "reconnect"]
    # node_1 is evaluated every round, connected or not
    assert all("node_1" in r["mesh_eval"] for r in report["rounds"])
    assert "recovery_round_mesh" in report["summary"]
    assert "recovery_round_baseline" in report["summary"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scenarios.py -v -k disconnection`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scenarios.disconnection'`.

- [ ] **Step 3: Create `src/scenarios/disconnection.py`**

```python
"""Scenario: a node disconnects mid-run and later reconnects. Demonstrates
that the mesh tolerates a node dropping out (it keeps training locally,
just isolated) and that reconnecting lets it catch back up towards peer
consensus faster than relying on local training alone (the baseline).

Run: python -m src.scenarios.disconnection [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders


def make_disconnect_hook(target_node: str, disconnect_round: int, reconnect_round: int):
    def hook(round_idx, nodes, mesh):
        target = next(n for n in nodes if n.node_id == target_node)
        events = []
        if round_idx == disconnect_round:
            target.active = False
            if mesh is not None:
                events.append(ScenarioEvent(round_idx, "disconnect", target_node, {}))
        elif round_idx == reconnect_round:
            target.active = True
            if mesh is not None:
                events.append(ScenarioEvent(round_idx, "reconnect", target_node, {}))
        return events

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    scfg = cfg.get("scenarios.disconnection")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.disconnection section")
    target_node = scfg["target_node"]
    disconnect_round = scfg["disconnect_round"]
    reconnect_round = scfg["reconnect_round"]
    num_rounds = cfg.get("scenarios.rounds", 6)

    if not (0 <= disconnect_round < reconnect_round < num_rounds):
        raise ValueError(
            f"scenarios.disconnection needs 0 <= disconnect_round ({disconnect_round}) "
            f"< reconnect_round ({reconnect_round}) < scenarios.rounds ({num_rounds})"
        )

    device = "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    probe_loader, node_loaders = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]

    require_target_node(target_node, node_loaders)

    baseline_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh = MeshSimulator(
        mesh_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(
        baseline_nodes, mesh, num_rounds,
        make_disconnect_hook(target_node, disconnect_round, reconnect_round),
        round_kwargs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "disconnection", target_node,
        disruption_start_round=disconnect_round, disruption_end_round=reconnect_round,
        config_snapshot=scfg, records=records,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the `scenarios:` section to `config.yaml`**

Append this to the end of `config.yaml` (after the existing `output:` section — do not touch any existing keys):

```yaml

scenarios:
  rounds: 6
  disconnection:
    target_node: "node_1"
    disconnect_round: 2
    reconnect_round: 4
  class_addition:
    target_node: "node_0"
    source_crop: "Tomato"       # must belong to a different node in manual_node_crops (currently node_2)
    inject_round: 2
    reserve_fraction: 0.3
  distribution_shift:
    target_node: "node_2"
    shift_round: 2
    corruption: "brightness_blur_noise"
    severity: 0.5
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_scenarios.py -v`
Expected: all tests PASS, including `test_disconnection_scenario_end_to_end_smoke`.

- [ ] **Step 6: Commit**

```bash
git add src/scenarios/disconnection.py config.yaml tests/test_scenarios.py
git commit -m "Add disconnection scenario: node drops out and reconnects mid-run"
```

---

## Task 5: Class-addition scenario (`src/scenarios/class_addition.py`)

**Files:**
- Create: `src/scenarios/class_addition.py`
- Modify: `tests/test_scenarios.py` (add tests)

**Interfaces:**
- Consumes: everything from Task 3's `harness.py`; `train_test_split_indices`, `make_subset`, `load_full_dataset` from `src.data.plantvillage`; `build_dataloaders` from `src.train`.
- Produces: `carve_reserve_pool(dataset, source_train_indices, source_crop, reserve_fraction, seed, test_fraction) -> tuple[list[int], list[int], list[int]]`, `find_source_node(manual_node_crops, source_crop) -> str`, `make_class_addition_hook(target_node, source_crop, inject_round, reserve_train_idx, reserve_test_idx, dataset, batch_size) -> PerturbationHook`, and `main()`.

- [ ] **Step 1: Write the failing tests in `tests/test_scenarios.py`**

Append:

```python
def test_find_source_node_returns_owning_node_and_raises_for_unknown_crop():
    from src.scenarios.class_addition import find_source_node

    manual_node_crops = {"node_0": ["Potato"], "node_1": ["Tomato"]}
    assert find_source_node(manual_node_crops, "Tomato") == "node_1"

    with pytest.raises(ValueError, match="not assigned to any node"):
        find_source_node(manual_node_crops, "Corn")


def test_carve_reserve_pool_is_disjoint_from_remaining_source(synthetic_dataset):
    from src.scenarios.class_addition import carve_reserve_pool

    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=4)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="manual", dirichlet_alpha=0.3, seed=4,
        manual_node_crops={"node_0": ["Tomato"], "node_1": ["Potato"]},
    )
    source_train_idx, _ = train_test_split_indices(shards[0], test_fraction=0.3, seed=4)  # node_0 grows Tomato

    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
        synthetic_dataset, source_train_idx, source_crop="Tomato", reserve_fraction=0.5, seed=4, test_fraction=0.3,
    )

    reserve_all = set(reserve_train_idx) | set(reserve_test_idx)
    assert reserve_all.isdisjoint(remaining_source)
    assert set(remaining_source) | reserve_all == set(source_train_idx)
    assert len(reserve_all) > 0


def test_class_addition_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.class_addition import carve_reserve_pool, make_class_addition_hook

    _, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.1, seed=5)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="manual", dirichlet_alpha=0.3, seed=5,
        manual_node_crops={"node_0": ["Potato"], "node_1": ["Tomato"]},
    )
    probe_idx, _ = carve_public_probe_set(synthetic_dataset, 0.1, seed=5)
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    source_train_idx, source_test_idx = train_test_split_indices(shards[1], test_fraction=0.3, seed=5)  # node_1: Tomato
    target_train_idx, target_test_idx = train_test_split_indices(shards[0], test_fraction=0.3, seed=5)  # node_0: Potato

    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
        synthetic_dataset, source_train_idx, source_crop="Tomato", reserve_fraction=0.5, seed=5, test_fraction=0.3,
    )

    def make_loaders(train_idx, test_idx):
        return (
            DataLoader(make_subset(synthetic_dataset, train_idx), batch_size=4, shuffle=True),
            DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False),
        )

    node_loaders = [
        make_loaders(target_train_idx, target_test_idx),
        make_loaders(remaining_source, source_test_idx),
    ]

    baseline_nodes = [
        Node(f"node_{i}", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), tl, sl, device="cpu")
        for i, (tl, sl) in enumerate(node_loaders)
    ]
    mesh_nodes = [
        Node(f"node_{i}", build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False), tl, sl, device="cpu")
        for i, (tl, sl) in enumerate(node_loaders)
    ]
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_class_addition_hook(
        "node_0", "Tomato", inject_round=1, reserve_train_idx=reserve_train_idx,
        reserve_test_idx=reserve_test_idx, dataset=synthetic_dataset, batch_size=4,
    )
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "class_addition", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"target_node": "node_0", "source_crop": "Tomato", "inject_round": 1},
        records=records,
    )
    report = json.loads(report_path.read_text())
    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["class_added"]
    assert report["rounds"][1]["events"][0]["details"]["crop"] == "Tomato"
```

Add these two imports to the top of `tests/test_scenarios.py` (alongside the existing ones):
```python
from src.federated.node import Node
from src.models.factory import build_model
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scenarios.py -v -k class_addition`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scenarios.class_addition'`.

- [ ] **Step 3: Create `src/scenarios/class_addition.py`**

```python
"""Scenario: a crop that's currently only grown elsewhere in the mesh
starts appearing at one node partway through the run (e.g. crop
rotation). Demonstrates that the mesh helps that node learn the new class
faster than training on it alone (the baseline), because peers who
already know the crop contribute it to the shared knowledge consensus.

Run: python -m src.scenarios.class_addition [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config
from src.data.plantvillage import load_full_dataset, make_subset, train_test_split_indices
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders


def carve_reserve_pool(dataset, source_train_indices, source_crop, reserve_fraction, seed, test_fraction):
    """Removes a `reserve_fraction` slice of `source_crop` samples from
    `source_train_indices`, returning (remaining_source_indices,
    reserve_train_indices, reserve_test_indices). The reserve is a subset
    of what the source node already owned — nothing is duplicated across
    nodes.
    """
    crop_idx = dataset.labels.crop_classes.index(source_crop)
    matching = [
        idx for idx in source_train_indices
        if dataset.labels.class_to_crop_disease[dataset.targets[idx]][0] == crop_idx
    ]
    rng = random.Random(seed)
    shuffled = matching.copy()
    rng.shuffle(shuffled)
    n_reserve = max(1, int(len(shuffled) * reserve_fraction))
    reserve = set(shuffled[:n_reserve])
    remaining_source = [idx for idx in source_train_indices if idx not in reserve]
    reserve_train_idx, reserve_test_idx = train_test_split_indices(list(reserve), test_fraction, seed)
    return remaining_source, reserve_train_idx, reserve_test_idx


def find_source_node(manual_node_crops: dict, source_crop: str) -> str:
    for node_key, crops in manual_node_crops.items():
        if source_crop in crops:
            node_idx = int(str(node_key).rsplit("_", 1)[-1])
            return f"node_{node_idx}"
    raise ValueError(f"source_crop '{source_crop}' is not assigned to any node in data.manual_node_crops")


def make_class_addition_hook(
    target_node: str, source_crop: str, inject_round: int, reserve_train_idx, reserve_test_idx, dataset, batch_size: int
):
    def hook(round_idx, nodes, mesh):
        if round_idx != inject_round:
            return []
        target = next(n for n in nodes if n.node_id == target_node)
        target.train_loader = DataLoader(
            ConcatDataset([target.train_loader.dataset, make_subset(dataset, reserve_train_idx)]),
            batch_size=batch_size, shuffle=True,
        )
        target.test_loader = DataLoader(
            ConcatDataset([target.test_loader.dataset, make_subset(dataset, reserve_test_idx)]),
            batch_size=batch_size, shuffle=False,
        )
        if mesh is not None:
            return [ScenarioEvent(
                round_idx, "class_added", target_node,
                {"crop": source_crop, "n_injected": len(reserve_train_idx) + len(reserve_test_idx)},
            )]
        return []

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    if cfg.get("data.non_iid_strategy") != "manual":
        raise ValueError(
            "scenarios.class_addition requires data.non_iid_strategy == 'manual' "
            "(it looks up crop ownership via data.manual_node_crops)"
        )

    scfg = cfg.get("scenarios.class_addition")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.class_addition section")
    target_node = scfg["target_node"]
    source_crop = scfg["source_crop"]
    inject_round = scfg["inject_round"]
    reserve_fraction = scfg["reserve_fraction"]
    num_rounds = cfg.get("scenarios.rounds", 6)
    seed = cfg.get("data.seed", 42)

    if not (0 <= inject_round < num_rounds):
        raise ValueError(f"scenarios.class_addition.inject_round ({inject_round}) must be in [0, {num_rounds})")

    manual_node_crops = cfg.get("data.manual_node_crops", {})
    source_node = find_source_node(manual_node_crops, source_crop)
    if source_node == target_node:
        raise ValueError(
            f"source_crop '{source_crop}' is already assigned to target_node '{target_node}' — "
            "pick a crop owned by a DIFFERENT node so the mesh has something to teach it"
        )

    device = "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    if source_crop not in dataset.labels.crop_classes:
        raise ValueError(f"source_crop '{source_crop}' is not a known crop: {dataset.labels.crop_classes}")

    probe_loader, node_loaders = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    batch_size = cfg.get("training.batch_size", 32)

    require_target_node(target_node, node_loaders)
    require_target_node(source_node, node_loaders)

    source_idx = int(source_node.rsplit("_", 1)[-1])
    source_train_loader, source_test_loader = node_loaders[source_idx]
    remaining_source, reserve_train_idx, reserve_test_idx = carve_reserve_pool(
        dataset, source_train_loader.dataset.indices, source_crop, reserve_fraction, seed,
        cfg.get("data.test_fraction", 0.15),
    )
    node_loaders[source_idx] = (
        DataLoader(make_subset(dataset, remaining_source), batch_size=batch_size, shuffle=True),
        source_test_loader,
    )

    baseline_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh = MeshSimulator(
        mesh_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    hook = make_class_addition_hook(
        target_node, source_crop, inject_round, reserve_train_idx, reserve_test_idx, dataset, batch_size
    )
    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds, hook, round_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "class_addition", target_node,
        disruption_start_round=inject_round, disruption_end_round=inject_round,
        config_snapshot=scfg, records=records,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scenarios.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scenarios/class_addition.py tests/test_scenarios.py
git commit -m "Add class-addition scenario: a node gains a crop it never grew before, mid-run"
```

---

## Task 6: Distribution-shift scenario (`src/scenarios/distribution_shift.py`)

**Files:**
- Create: `src/scenarios/distribution_shift.py`
- Modify: `tests/test_scenarios.py` (add tests)

**Interfaces:**
- Consumes: everything from Task 3's `harness.py`; `load_full_dataset` from `src.data.plantvillage`; `build_dataloaders` from `src.train`.
- Produces: `CorruptedDataset(base: Dataset, severity: float)` (a `torch.utils.data.Dataset`), `make_shift_hook(target_node, shift_round, corruption, severity, batch_size) -> PerturbationHook`, and `main()`.

- [ ] **Step 1: Write the failing tests in `tests/test_scenarios.py`**

Append:

```python
def test_corrupted_dataset_is_noop_at_zero_severity_and_differs_otherwise(synthetic_dataset):
    from src.scenarios.distribution_shift import CorruptedDataset

    clean = CorruptedDataset(synthetic_dataset, severity=0.0)
    image_a, crop_a, disease_a = clean[0]
    image_b, crop_b, disease_b = synthetic_dataset[0]
    assert torch.equal(image_a, image_b)
    assert crop_a == crop_b and disease_a == disease_b

    corrupted = CorruptedDataset(synthetic_dataset, severity=0.5)
    image_c, _, _ = corrupted[0]
    assert not torch.equal(image_c, image_b)


def test_distribution_shift_scenario_end_to_end_smoke(tmp_path, synthetic_dataset):
    from src.scenarios.distribution_shift import make_shift_hook

    probe_idx, remaining_idx = carve_public_probe_set(synthetic_dataset, 0.2, seed=6)
    shards = partition_nodes(
        synthetic_dataset, remaining_idx, num_nodes=2, strategy="by_crop", dirichlet_alpha=0.3, seed=6
    )
    probe_loader = DataLoader(make_subset(synthetic_dataset, probe_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)

    baseline_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh_nodes = build_nodes(synthetic_dataset, shards, num_crop, num_disease)
    mesh = MeshSimulator(
        mesh_nodes, probe_loader, aggregation_method="trimmed_mean", trim_fraction=0.0, krum_neighbors=1
    )

    hook = make_shift_hook("node_0", shift_round=1, corruption="brightness_blur_noise", severity=0.5, batch_size=4)
    round_kwargs = {
        "local_epochs": 1, "distill_epochs": 1, "lr": 1e-3, "distill_lr": 1e-3,
        "proto_weight": 0.5, "kd_weight": 0.5, "temperature": 2.0,
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs)

    report_path = write_scenario_report(
        tmp_path, "distribution_shift", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={"target_node": "node_0", "shift_round": 1, "severity": 0.5},
        records=records,
    )
    report = json.loads(report_path.read_text())
    event_types = [e["event_type"] for r in report["rounds"] for e in r["events"]]
    assert event_types == ["shift_applied"]
    assert report["rounds"][1]["events"][0]["details"]["severity"] == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scenarios.py -v -k distribution_shift`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.scenarios.distribution_shift'`.

- [ ] **Step 3: Create `src/scenarios/distribution_shift.py`**

```python
"""Scenario: one node's local camera/lighting degrades partway through the
run (brightness shift + blur + noise applied to its images from that round
onward). Demonstrates that the mesh dampens the resulting accuracy drop
compared to a node coping with the shift on local training alone (the
baseline).

Run: python -m src.scenarios.distribution_shift [--config path] [--arch name]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.config import Config
from src.data.plantvillage import load_full_dataset
from src.federated.mesh import MeshSimulator
from src.scenarios.harness import (
    ScenarioEvent,
    build_node_set,
    require_target_node,
    run_scenario,
    write_scenario_report,
)
from src.train import build_dataloaders


class CorruptedDataset(Dataset):
    """Wraps a dataset, applying a severity-scaled brightness shift +
    Gaussian blur + additive noise to the already-normalized image tensor —
    simulating a degraded camera/lighting condition. severity=0 is a no-op.
    """

    def __init__(self, base: Dataset, severity: float):
        self.base = base
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, crop, disease = self.base[idx]
        if self.severity > 0:
            image = image * (1.0 + self.severity)
            kernel_size = max(3, int(self.severity * 6) | 1)
            image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])
            image = image + torch.randn_like(image) * self.severity
        return image, crop, disease


def make_shift_hook(target_node: str, shift_round: int, corruption: str, severity: float, batch_size: int):
    def hook(round_idx, nodes, mesh):
        if round_idx != shift_round:
            return []
        target = next(n for n in nodes if n.node_id == target_node)
        target.train_loader = DataLoader(
            CorruptedDataset(target.train_loader.dataset, severity), batch_size=batch_size, shuffle=True
        )
        target.test_loader = DataLoader(
            CorruptedDataset(target.test_loader.dataset, severity), batch_size=batch_size, shuffle=False
        )
        if mesh is not None:
            return [ScenarioEvent(round_idx, "shift_applied", target_node,
                                   {"corruption": corruption, "severity": severity})]
        return []

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    cfg = Config.load(args.config)

    scfg = cfg.get("scenarios.distribution_shift")
    if scfg is None:
        raise ValueError("config.yaml is missing a scenarios.distribution_shift section")
    target_node = scfg["target_node"]
    shift_round = scfg["shift_round"]
    corruption = scfg["corruption"]
    severity = scfg["severity"]
    num_rounds = cfg.get("scenarios.rounds", 6)

    if not (0 <= shift_round < num_rounds):
        raise ValueError(f"scenarios.distribution_shift.shift_round ({shift_round}) must be in [0, {num_rounds})")

    device = "cpu"
    output_dir = Path(cfg.get("output.dir", "outputs"))
    dataset = load_full_dataset(cfg.get("data.root"), cfg.get("data.image_size", 160))
    num_crop = len(dataset.labels.crop_classes)
    num_disease = len(dataset.labels.disease_classes)
    probe_loader, node_loaders = build_dataloaders(cfg, dataset)
    arch = args.arch or cfg.get("models.architectures", ["mobilenet_v3_small"])[0]
    batch_size = cfg.get("training.batch_size", 32)

    require_target_node(target_node, node_loaders)

    baseline_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh_nodes = build_node_set(cfg, arch, node_loaders, num_crop, num_disease, device)
    mesh = MeshSimulator(
        mesh_nodes,
        probe_loader,
        aggregation_method=cfg.get("federated.aggregation", "trimmed_mean"),
        trim_fraction=cfg.get("federated.trim_fraction", 0.2),
        krum_neighbors=cfg.get("federated.krum_neighbors", 2),
    )

    hook = make_shift_hook(target_node, shift_round, corruption, severity, batch_size)
    round_kwargs = {
        "local_epochs": cfg.get("training.local_epochs_per_round", 2),
        "distill_epochs": cfg.get("training.distill_epochs_per_round", 1),
        "lr": cfg.get("training.lr", 0.001),
        "distill_lr": cfg.get("training.distill_lr", 0.0005),
        "proto_weight": cfg.get("training.proto_weight", 0.5),
        "kd_weight": cfg.get("training.kd_weight", 0.5),
        "temperature": cfg.get("training.kd_temperature", 2.0),
    }
    records = run_scenario(baseline_nodes, mesh, num_rounds, hook, round_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_scenario_report(
        output_dir, "distribution_shift", target_node,
        disruption_start_round=shift_round, disruption_end_round=shift_round,
        config_snapshot=scfg, records=records,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test suite to verify everything passes**

Run: `pytest tests/ -v`
Expected: every test in `tests/test_pipeline.py` and `tests/test_scenarios.py` PASSes (this is the full regression check across all 6 tasks).

- [ ] **Step 5: Commit**

```bash
git add src/scenarios/distribution_shift.py tests/test_scenarios.py
git commit -m "Add distribution-shift scenario: a node's images degrade mid-run"
```

---

## After implementation: manual verification on the real dataset

Not a task with automated steps (it needs the real, already-downloaded PlantVillage data and takes real training time, so it isn't suited to a fast subagent test cycle) — once all 6 tasks are done and committed, run each scenario against the real config to see the actual demonstration output:

```bash
python -m src.scenarios.disconnection
python -m src.scenarios.class_addition
python -m src.scenarios.distribution_shift
```

Check `outputs/scenarios/disconnection.json`, `outputs/scenarios/class_addition.json`, `outputs/scenarios/distribution_shift.json` — each should show the target node's accuracy dipping around the disruption round and recovering (or not) afterward, with `collaboration_gain` per round showing whether the mesh outperformed the baseline under the same disruption.
