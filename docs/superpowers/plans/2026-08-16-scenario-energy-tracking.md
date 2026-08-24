# Scenario Energy Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `ComputeEnergyTracker` / `CommunicationCostEstimator` (already used by `src/train.py`) into the three mesh-disruption scenario scripts, so every scenario's JSON report and the published per-round markdown tables carry real Joule/kWh figures and a `gain_per_joule` metric — closing the one gap identified in `docs/sustainability_energy_plan.md` §3.

**Architecture:** `src/scenarios/harness.py`'s `run_scenario()` gains two new required parameters (a `ComputeEnergyTracker`, a `CommunicationCostEstimator`) and wraps the baseline nodes' `local_train()` calls and the mesh's `run_round()` call in `tracker.track(...)`, exactly the call pattern `src/train.py` already uses. `write_scenario_report()` gains a `sustainability` summary block, including a `gain_per_joule` figure. The three scenario entry points (`disconnection.py`, `class_addition.py`, `distribution_shift.py`) each instantiate the tracker/estimator from `config.yaml`'s existing `energy:` section and pass them through. No new energy math is introduced — this plan only wires up classes that already exist and are already tested.

**Tech Stack:** Python, pytest, PyTorch (existing stack — no new dependencies).

## Global Constraints

- Baseline for all sustainability comparisons is **Baseline B (local-only)** — this is what the scenario harness's baseline nodeset already is; do not introduce a second baseline.
- Default radio for the communication-energy conversion is `"wifi"` (`config.yaml` → `energy.radio_energy_j_per_byte.wifi = 0.00003` J/byte) — matches the figure already used throughout `docs/sustainability_energy_plan.md`.
- Do not modify `ComputeEnergyTracker` or `CommunicationCostEstimator` (`src/energy/tracker.py`) — both are already implemented, already unit-tested (`tests/test_pipeline.py::test_energy_and_communication_tracking`), and already used by `src/train.py`; this plan only calls them from new call sites.
- Do not change `config.yaml`'s `energy.track_with_codecarbon` default (currently `false`) — whichever energy-measurement method is active (CodeCarbon or the disclosed wall-power fallback) is a user/environment decision outside this plan's scope.
- Submission deadline for the Cambridge Edge AI Challenge is 2026-08-24 — Task 4 (real-dataset re-run) is the only wall-clock-heavy step; everything else is fast unit-test-driven code.

---

### Task 1: Sustainability fields on `ScenarioRoundRecord` + energy wiring in `run_scenario()`

**Files:**
- Modify: `src/scenarios/harness.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `ComputeEnergyTracker` (`src/energy/tracker.py`) — `tracker.track(label: str)` context manager yielding a dict with key `"energy_kwh"` (float) once the `with` block exits. `CommunicationCostEstimator` (`src/energy/tracker.py`) — `estimator.estimate(total_bytes: int, radio: str) -> dict` returning a dict with key `"energy_kwh"` (float).
- Produces: `ScenarioRoundRecord` gains three new fields — `baseline_compute_energy_kwh: float`, `mesh_compute_energy_kwh: float`, `communication_energy_j: float` (all default `0.0`). `run_scenario(baseline_nodes, mesh, num_rounds, perturbation_hook, round_kwargs, tracker, comm_estimator, radio="wifi")` — two new required keyword params, `tracker` and `comm_estimator`, and one optional `radio` (default `"wifi"`). Later tasks (Task 2, Task 3) rely on these exact names.

- [ ] **Step 1: Add shared energy fixtures to `tests/conftest.py`**

Add to the end of `tests/conftest.py`:

```python
@pytest.fixture
def energy_tracker(tmp_path):
    from src.energy.tracker import ComputeEnergyTracker
    return ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)


@pytest.fixture
def wifi_comm_estimator():
    from src.energy.tracker import CommunicationCostEstimator
    return CommunicationCostEstimator(
        radio_energy_j_per_byte={"wifi": 0.00003}, grid_carbon_intensity_gco2_per_kwh=125
    )
```

- [ ] **Step 2: Write the failing test**

Add to `tests/test_scenarios.py` (near `test_run_scenario_and_write_report`):

```python
def test_run_scenario_tracks_compute_and_communication_energy(
    tmp_path, synthetic_dataset, energy_tracker, wifi_comm_estimator
):
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
    records = run_scenario(
        baseline_nodes, mesh, num_rounds=2, perturbation_hook=no_op_hook, round_kwargs=round_kwargs,
        tracker=energy_tracker, comm_estimator=wifi_comm_estimator, radio="wifi",
    )

    assert len(records) == 2
    for record in records:
        assert record.baseline_compute_energy_kwh >= 0
        assert record.mesh_compute_energy_kwh >= 0
        # both nodes are active every round in this test, so bytes (and
        # therefore communication energy) must be strictly positive.
        assert record.communication_energy_j > 0

    expected_j = records[0].total_bytes_exchanged * 0.00003
    assert records[0].communication_energy_j == pytest.approx(expected_j)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_scenarios.py::test_run_scenario_tracks_compute_and_communication_energy -v`
Expected: FAIL with `TypeError: run_scenario() missing 2 required keyword-only arguments: 'tracker' and 'comm_estimator'` (or similar — `run_scenario` does not yet accept these kwargs).

- [ ] **Step 4: Implement the fields and the wiring in `src/scenarios/harness.py`**

Add the three new fields to `ScenarioRoundRecord` (after `total_bytes_exchanged`):

```python
@dataclasses.dataclass
class ScenarioRoundRecord:
    round_idx: int
    baseline_eval: dict[str, dict[str, float]]
    mesh_eval: dict[str, dict[str, float]]
    collaboration_gain: dict
    events: list[ScenarioEvent] = dataclasses.field(default_factory=list)
    active_nodes: list[str] = dataclasses.field(default_factory=list)
    total_bytes_exchanged: int = 0
    baseline_compute_energy_kwh: float = 0.0
    mesh_compute_energy_kwh: float = 0.0
    communication_energy_j: float = 0.0
```

Add the import near the top of the file, alongside the other `src.*` imports:

```python
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
```

Replace `run_scenario`'s signature and body:

```python
def run_scenario(
    baseline_nodes: list[Node],
    mesh: MeshSimulator,
    num_rounds: int,
    perturbation_hook: PerturbationHook,
    round_kwargs: dict,
    tracker: ComputeEnergyTracker,
    comm_estimator: CommunicationCostEstimator,
    radio: str = "wifi",
) -> list[ScenarioRoundRecord]:
    """Drives `num_rounds` rounds of a baseline (no-exchange) node set and a
    mesh node set through the same perturbation schedule, tracking compute
    energy (via `tracker`, the same ComputeEnergyTracker src/train.py uses)
    and communication energy (via `comm_estimator`, fed by the mesh round's
    already-measured `total_bytes_exchanged`) for every round of both.

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
        baseline_compute_energy_kwh = 0.0
        for node in baseline_nodes:
            with tracker.track(f"baseline_{node.node_id}_round_{round_idx}") as energy_record:
                node.local_train(round_kwargs["local_epochs"], round_kwargs["lr"])
            baseline_compute_energy_kwh += energy_record["energy_kwh"]
            baseline_eval[node.node_id] = node.evaluate()

        with tracker.track(f"mesh_round_{round_idx}") as mesh_energy_record:
            round_log = mesh.run_round(
                round_idx,
                local_epochs=round_kwargs["local_epochs"],
                distill_epochs=round_kwargs["distill_epochs"],
                lr=round_kwargs["lr"],
                distill_lr=round_kwargs["distill_lr"],
                proto_weight=round_kwargs["proto_weight"],
                kd_weight=round_kwargs["kd_weight"],
                temperature=round_kwargs["temperature"],
            )
        mesh_compute_energy_kwh = mesh_energy_record["energy_kwh"]
        mesh_eval = {node.node_id: node.evaluate() for node in mesh.nodes}

        comm_result = comm_estimator.estimate(round_log.total_bytes_exchanged, radio)
        communication_energy_j = comm_result["energy_kwh"] * 3_600_000

        gain = compute_collaboration_gain(mesh_eval, baseline_eval)
        records.append(ScenarioRoundRecord(
            round_idx, baseline_eval, mesh_eval, gain, events,
            active_nodes=round_log.active_nodes,
            total_bytes_exchanged=round_log.total_bytes_exchanged,
            baseline_compute_energy_kwh=baseline_compute_energy_kwh,
            mesh_compute_energy_kwh=mesh_compute_energy_kwh,
            communication_energy_j=communication_energy_j,
        ))
        print(f"  round {round_idx}: {len(events)} event(s), macro_gain={gain['macro_gain']}")
    return records
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_scenarios.py::test_run_scenario_tracks_compute_and_communication_energy -v`
Expected: PASS

- [ ] **Step 6: Fix the other existing `run_scenario(...)` call sites in `tests/test_scenarios.py`**

The signature change breaks four existing tests. Add `energy_tracker, wifi_comm_estimator` to each test function's parameters and pass them through:

In `test_run_scenario_and_write_report`, change the function signature to
`def test_run_scenario_and_write_report(tmp_path, synthetic_dataset, energy_tracker, wifi_comm_estimator):`
and the call to:
```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds=2, perturbation_hook=no_op_hook, round_kwargs=round_kwargs,
        tracker=energy_tracker, comm_estimator=wifi_comm_estimator,
    )
```

In `test_disconnection_scenario_end_to_end_smoke`, change the signature to
`def test_disconnection_scenario_end_to_end_smoke(tmp_path, synthetic_dataset, energy_tracker, wifi_comm_estimator):`
and the call to:
```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds=3, perturbation_hook=hook, round_kwargs=round_kwargs,
        tracker=energy_tracker, comm_estimator=wifi_comm_estimator,
    )
```

In `test_class_addition_scenario_end_to_end_smoke`, change the signature to
`def test_class_addition_scenario_end_to_end_smoke(tmp_path, synthetic_dataset, energy_tracker, wifi_comm_estimator):`
and the call to:
```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs,
        tracker=energy_tracker, comm_estimator=wifi_comm_estimator,
    )
```

In `test_distribution_shift_scenario_end_to_end_smoke`, change the signature to
`def test_distribution_shift_scenario_end_to_end_smoke(tmp_path, synthetic_dataset, energy_tracker, wifi_comm_estimator):`
and the call to:
```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds=2, perturbation_hook=hook, round_kwargs=round_kwargs,
        tracker=energy_tracker, comm_estimator=wifi_comm_estimator,
    )
```

- [ ] **Step 7: Run the full scenario test file to verify nothing is broken**

Run: `pytest tests/test_scenarios.py -v`
Expected: All tests PASS (previously-passing tests still pass; the new energy test passes).

- [ ] **Step 8: Commit**

```bash
git add src/scenarios/harness.py tests/conftest.py tests/test_scenarios.py
git commit -m "feat: track compute and communication energy per scenario round"
```

---

### Task 2: `gain_per_joule` and a `sustainability` summary block in `write_scenario_report()`

**Files:**
- Modify: `src/scenarios/harness.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: `ScenarioRoundRecord.baseline_compute_energy_kwh`, `.mesh_compute_energy_kwh`, `.communication_energy_j`, `.collaboration_gain` (all from Task 1).
- Produces: a module-level `_gain_per_joule(records: list[ScenarioRoundRecord]) -> Optional[float]` helper (importable directly, mirroring the existing `_recovery_round` pattern), and a `report["summary"]["sustainability"]` dict with keys `total_baseline_compute_energy_kwh`, `total_mesh_compute_energy_kwh`, `total_communication_energy_j`, `gain_per_joule` — this exact shape is what Task 4's doc-writing step will read from the regenerated JSON files.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_scenarios.py`:

```python
def test_write_scenario_report_includes_sustainability_summary(tmp_path):
    records = [
        ScenarioRoundRecord(
            0, {"node_0": {"crop_accuracy": 0.5}}, {"node_0": {"crop_accuracy": 0.6}}, {"macro_gain": 0.1},
            baseline_compute_energy_kwh=0.0001, mesh_compute_energy_kwh=0.0002, communication_energy_j=50.0,
        ),
        ScenarioRoundRecord(
            1, {"node_0": {"crop_accuracy": 0.55}}, {"node_0": {"crop_accuracy": 0.7}}, {"macro_gain": 0.15},
            baseline_compute_energy_kwh=0.0001, mesh_compute_energy_kwh=0.0002, communication_energy_j=50.0,
        ),
    ]
    report_path = write_scenario_report(
        tmp_path, "unit_test_energy", "node_0",
        disruption_start_round=1, disruption_end_round=1,
        config_snapshot={}, records=records,
    )
    report = json.loads(report_path.read_text())
    sustainability = report["summary"]["sustainability"]
    assert sustainability["total_baseline_compute_energy_kwh"] == pytest.approx(0.0002)
    assert sustainability["total_mesh_compute_energy_kwh"] == pytest.approx(0.0004)
    assert sustainability["total_communication_energy_j"] == pytest.approx(100.0)
    expected_total_mesh_j = 0.0004 * 3_600_000 + 100.0
    assert sustainability["gain_per_joule"] == pytest.approx(0.15 / expected_total_mesh_j)


def test_gain_per_joule_is_none_when_no_energy_recorded():
    from src.scenarios.harness import _gain_per_joule

    records = [
        ScenarioRoundRecord(0, {"node_0": {"crop_accuracy": 0.5}}, {"node_0": {"crop_accuracy": 0.6}}, {"macro_gain": 0.1}),
    ]
    assert _gain_per_joule(records) is None
    assert _gain_per_joule([]) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scenarios.py::test_write_scenario_report_includes_sustainability_summary tests/test_scenarios.py::test_gain_per_joule_is_none_when_no_energy_recorded -v`
Expected: FAIL — `test_write_scenario_report_includes_sustainability_summary` fails with `KeyError: 'sustainability'`; `test_gain_per_joule_is_none_when_no_energy_recorded` fails with `ImportError: cannot import name '_gain_per_joule'`.

- [ ] **Step 3: Implement `_gain_per_joule` and the summary block in `src/scenarios/harness.py`**

Add this helper directly above `write_scenario_report`:

```python
def _gain_per_joule(records: list[ScenarioRoundRecord]) -> Optional[float]:
    """The final round's macro collaboration gain divided by the total
    energy (compute + communication) the mesh spent across the whole run —
    Appendix A's gain_per_joule metric. Returns None when no energy was
    recorded at all, to avoid a divide-by-zero silently reporting 0.0 as if
    it were a measured (rather than absent) figure.
    """
    if not records:
        return None
    total_mesh_energy_j = (
        sum(r.mesh_compute_energy_kwh for r in records) * 3_600_000
        + sum(r.communication_energy_j for r in records)
    )
    if total_mesh_energy_j <= 0:
        return None
    return records[-1].collaboration_gain.get("macro_gain", 0.0) / total_mesh_energy_j
```

Change `write_scenario_report`'s `report` dict to add the `sustainability` key inside `summary`:

```python
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
            "sustainability": {
                "total_baseline_compute_energy_kwh": sum(r.baseline_compute_energy_kwh for r in records),
                "total_mesh_compute_energy_kwh": sum(r.mesh_compute_energy_kwh for r in records),
                "total_communication_energy_j": sum(r.communication_energy_j for r in records),
                "gain_per_joule": _gain_per_joule(records),
            },
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scenarios.py::test_write_scenario_report_includes_sustainability_summary tests/test_scenarios.py::test_gain_per_joule_is_none_when_no_energy_recorded -v`
Expected: PASS

- [ ] **Step 5: Run the full scenario test file to verify nothing is broken**

Run: `pytest tests/test_scenarios.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scenarios/harness.py tests/test_scenarios.py
git commit -m "feat: add gain_per_joule sustainability summary to scenario reports"
```

---

### Task 3: Wire the tracker/estimator into the three scenario entry points

**Files:**
- Modify: `src/scenarios/disconnection.py`
- Modify: `src/scenarios/class_addition.py`
- Modify: `src/scenarios/distribution_shift.py`

**Interfaces:**
- Consumes: `ComputeEnergyTracker(enabled, output_dir, country_iso_code)`, `CommunicationCostEstimator(radio_energy_j_per_byte, grid_carbon_intensity_gco2_per_kwh)` (both from `src/energy/tracker.py`); `run_scenario(..., tracker=..., comm_estimator=...)` (from Task 1).
- Produces: nothing new consumed by later tasks — this task's deliverable is that running any of the three scripts on the real dataset (Task 4) now produces energy-populated JSON reports.

This is the same three-line change, applied identically in each of the three files: import the two classes, instantiate them from `cfg` exactly as `src/train.py` does, and pass them into `run_scenario`.

- [ ] **Step 1: Wire `src/scenarios/disconnection.py`**

Add to the imports (alongside the existing `from src.train import build_dataloaders`):

```python
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
```

Insert immediately after the `mesh = MeshSimulator(...)` block (i.e. right before the `round_kwargs = {` line):

```python
    tracker = ComputeEnergyTracker(
        enabled=cfg.get("energy.track_with_codecarbon", True),
        output_dir=output_dir,
        country_iso_code=cfg.get("energy.country_iso_code", "GBR"),
    )
    comm_estimator = CommunicationCostEstimator(
        cfg.get("energy.radio_energy_j_per_byte", {}),
        cfg.get("energy.grid_carbon_intensity_gco2_per_kwh", 125),
    )
```

Change the `run_scenario(...)` call from:

```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds,
        make_disconnect_hook(target_node, disconnect_round, reconnect_round),
        round_kwargs,
    )
```

to:

```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds,
        make_disconnect_hook(target_node, disconnect_round, reconnect_round),
        round_kwargs,
        tracker=tracker, comm_estimator=comm_estimator,
    )
```

- [ ] **Step 2: Wire `src/scenarios/class_addition.py`**

Add the same import line alongside its existing `from src.train import build_dataloaders`:

```python
from src.energy.tracker import CommunicationCostEstimator, ComputeEnergyTracker
```

Insert the same `tracker`/`comm_estimator` block immediately after its `mesh = MeshSimulator(...)` block (before the `hook = make_class_addition_hook(...)` line).

Change its `run_scenario(...)` call from:

```python
    records = run_scenario(baseline_nodes, mesh, num_rounds, hook, round_kwargs)
```

to:

```python
    records = run_scenario(
        baseline_nodes, mesh, num_rounds, hook, round_kwargs,
        tracker=tracker, comm_estimator=comm_estimator,
    )
```

- [ ] **Step 3: Wire `src/scenarios/distribution_shift.py`**

Same pattern: add the import line, insert the `tracker`/`comm_estimator` block right after its `mesh = MeshSimulator(...)` block (before the `hook = make_shift_hook(...)` line), and change its `run_scenario(...)` call the same way as Step 2's.

- [ ] **Step 4: Verify all three modules still import and parse correctly**

Run: `python -c "import src.scenarios.disconnection, src.scenarios.class_addition, src.scenarios.distribution_shift"`
Expected: no output, exit code 0 (a syntax or import error would print a traceback and exit non-zero).

- [ ] **Step 5: Run the full scenario test file to verify nothing is broken**

Run: `pytest tests/test_scenarios.py -v`
Expected: All tests PASS (these tests exercise `run_scenario`/`write_scenario_report` and each module's hook-builder functions directly, not `main()`, so this step is a regression check on the surrounding code, not a direct test of the new wiring — Task 4's real run is what actually exercises `main()`).

- [ ] **Step 6: Commit**

```bash
git add src/scenarios/disconnection.py src/scenarios/class_addition.py src/scenarios/distribution_shift.py
git commit -m "feat: wire energy tracker/estimator into scenario entry points"
```

---

### Task 4: Re-run the three scenarios on the real PlantVillage dataset

**Files:**
- None (produces `outputs/scenarios/disconnection.json`, `outputs/scenarios/class_addition.json`, `outputs/scenarios/distribution_shift.json` — all git-ignored, matching the rest of `outputs/`)

**Interfaces:**
- Consumes: the wired-up scripts from Task 3.
- Produces: three regenerated JSON reports, each now containing non-zero `baseline_compute_energy_kwh`, `mesh_compute_energy_kwh`, and `communication_energy_j` per round, and a populated `summary.sustainability` block — this is what Task 5 reads from directly.

This task has no automated test — it is the real-dataset run that Task 3's `main()` changes can't be exercised by otherwise. Requires `data/PlantVillage` already downloaded (per `docs/execution_guide.md`) and `config.yaml` as currently checked in.

- [ ] **Step 1: Run the disconnection scenario**

Run: `python -m src.scenarios.disconnection`
Expected: prints per-round progress ending in `Wrote outputs\scenarios\disconnection.json` (or `outputs/scenarios/disconnection.json` on non-Windows).

- [ ] **Step 2: Run the class-addition scenario**

Run: `python -m src.scenarios.class_addition`
Expected: prints per-round progress ending in `Wrote outputs\scenarios\class_addition.json`.

- [ ] **Step 3: Run the distribution-shift scenario**

Run: `python -m src.scenarios.distribution_shift`
Expected: prints per-round progress ending in `Wrote outputs\scenarios\distribution_shift.json`.

- [ ] **Step 4: Verify the regenerated reports actually contain energy data**

Run (PowerShell):
```powershell
python -c "import json; d = json.load(open('outputs/scenarios/disconnection.json')); print(d['rounds'][0]['communication_energy_j']); print(d['summary']['sustainability'])"
```
Expected: prints a positive float (the round-0 communication energy in Joules) followed by a dict with keys `total_baseline_compute_energy_kwh`, `total_mesh_compute_energy_kwh`, `total_communication_energy_j`, `gain_per_joule`. Repeat for `class_addition.json` and `distribution_shift.json`.

No commit for this task — the `outputs/` directory is git-ignored (matches the existing convention documented in `mesh_disruption_scenarios.md`'s "Output files" section).

---

### Task 5: Update `docs/mesh_disruption_scenarios.md` with the real energy figures

**Files:**
- Modify: `docs/mesh_disruption_scenarios.md`

**Interfaces:**
- Consumes: `outputs/scenarios/{disconnection,class_addition,distribution_shift}.json` from Task 4 — specifically each round's `communication_energy_j` field and each report's `summary.sustainability` block.
- Produces: nothing consumed by later tasks — this is the final documentation deliverable.

- [ ] **Step 1: Add a "Comm. energy (J)" column to each of the three per-round tables**

For each of the three per-round markdown tables (Disconnection §1, Class Addition §2, Distribution Shift §3), add a `Comm. energy (J)` column, populated from the matching round's `communication_energy_j` value in that scenario's regenerated JSON (Task 4). Keep every existing column as-is — this is an addition, not a rewrite of the existing accuracy/byte-count evidence.

- [ ] **Step 2: Add one sustainability sentence per scenario section**

Immediately after each scenario's existing "**Explanation:**" paragraph, add one sentence reporting that scenario's `summary.sustainability.gain_per_joule` figure (or, for scenarios where it's `null`, state that plainly — e.g. distribution shift, where the near-ceiling accuracy plateau may leave no measurable gain to divide by energy spent, exactly as `docs/sustainability_energy_plan.md` §4 anticipated).

- [ ] **Step 3: Add a `gain_per_joule` column to the "Overall summary" table**

Extend the existing summary table (currently: Scenario / Target node / Disruption / Recovery — mesh / Recovery — baseline / Mesh advantage) with one more column holding each scenario's `gain_per_joule` value.

- [ ] **Step 4: Add a closing cross-reference**

At the end of the document, add one sentence pointing to `docs/sustainability_energy_plan.md` for the full Axis A / Appendix E rubric justification, so a judge following either document can find the other.

- [ ] **Step 5: Proofread against the regenerated JSON**

Re-open each of the three `outputs/scenarios/*.json` files from Task 4 and confirm every number just written into the markdown matches exactly (no transcription errors) — this is the task's verification step in place of an automated test.

- [ ] **Step 6: Commit**

```bash
git add docs/mesh_disruption_scenarios.md
git commit -m "docs: add real energy/gain-per-joule figures to scenario tables"
```
