# Dashboard Scenario Tabs + Start/Stop Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Streamlit dashboard 4 tabs (Full Run, Class Addition, Disconnection, Distribution Shift), each independently start/stoppable through a new `controller` service, with mutual exclusion across tabs, per-scenario data namespacing, a `local_only` shadow-model arm on every tab, and a unified export/fairness-disclosure schema.

**Architecture:** A new `controller` container (Docker-socket + host-project bind mount) drives `docker compose stop`/`up -d --force-recreate` on `coordinator`/`node_0/1/2` via a testable `ControllerRunner` state machine. Those 4 services read a `SCENARIO` env var to namespace their SQLite dbs and activity logs under `/energy/{scenario}/`. Each node additionally trains a `local_only` shadow model every round (sequentially, before its mesh step) so every tab's own data satisfies the Challenge's §6.1.3 baseline-comparison requirement. The dashboard renders 4 `st.tabs`, each reading its own namespaced data and calling the controller's HTTP API.

**Tech Stack:** Python 3.11, FastAPI/Uvicorn, Streamlit, SQLite (`src/energy/sqlite_store.py`), Docker Compose v2, `docker:27-cli` base image for the controller.

**Spec:** [docs/superpowers/specs/2026-08-17-dashboard-scenario-control-plane-design.md](../specs/2026-08-17-dashboard-scenario-control-plane-design.md)

## Global Constraints

- Nodes are hardcoded to `device="cpu"` (`docker/node/main.py`) — never introduce GPU code paths.
- Only 3 node containers + 1 coordinator ever run at once, for any scenario — no design may spin up additional containers per scenario.
- The shadow (`local_only`) step and the mesh step run **sequentially** within a node's process, shadow first — never concurrently (wall-clock energy measurement validity, per spec §4).
- Every container start is a fresh run, not a resume: existing code already wipes its own db/log/status files on startup — new code must preserve this, keyed by the active `SCENARIO`, not the whole `/energy` tree.
- `docker compose` project name is pinned via `name: crop-mesh` in `docker/docker-compose.yml` (top-level key) — this must stay in sync between the host-native compose file and the controller's nested invocation of the *same* file, so both resolve to the same containers.
- All new Docker volume paths use `${HOST_PROJECT_ROOT}` (an explicit, user-supplied absolute host path), never relative `..`, on any service the `controller` might need to recreate — relative paths silently break when `docker compose` is invoked from inside a container talking to the host daemon over the socket.

---

## File structure

```
docker/
  docker-compose.yml            # MODIFY: SCENARIO/LOG_FILE env vars, ${HOST_PROJECT_ROOT}, controller service, name: crop-mesh
  controller/                   # NEW
    Dockerfile
    requirements.txt
    controller_runner.py        # pure state machine, DI'd start/stop callables
    main.py                     # FastAPI wiring + real subprocess-based docker compose calls
  node/
    node_runner.py               # MODIFY: shadow model, per-scenario log file
    main.py                      # MODIFY: build shadow Node, LOG_FILE wiring
  coordinator/
    coordinator_runner.py        # MODIFY: persist baseline_* fields, per-scenario log file
    main.py                      # MODIFY: LOG_FILE wiring
  dashboard/
    data.py                      # MODIFY: scenario path helpers, fairness disclosure, export schema
    app.py                       # MODIFY: 4 tabs, Start/Stop, collapsed panels, log-file reads
src/energy/
  sqlite_store.py                # MODIFY: 4 new baseline_* columns
tests/
  test_controller_runner.py      # NEW
  test_sqlite_store.py           # MODIFY
  test_node_runner.py            # MODIFY
  test_coordinator_runner.py     # MODIFY
  test_dashboard_data.py         # MODIFY
docs/
  docker_mesh_execution_guide.md # MODIFY: HOST_PROJECT_ROOT setup, controller usage
```

---

### Task 1: Extend `round_metrics` schema with baseline columns

**Files:**
- Modify: `src/energy/sqlite_store.py:13-27`
- Test: `tests/test_sqlite_store.py`

**Interfaces:**
- Produces: `upsert_row(..., baseline_crop_accuracy=..., baseline_disease_accuracy=..., baseline_energy_kwh=..., baseline_duration_s=...)` — accepted like any other `**fields` kwarg, no signature change.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sqlite_store.py`:
```python
def test_upsert_row_accepts_baseline_columns(tmp_path):
    path = tmp_path / "node_0.db"
    sqlite_store.upsert_row(
        path, "node_0", 0, "2026-08-17T00:00:00Z",
        baseline_crop_accuracy=0.7, baseline_disease_accuracy=0.6,
        baseline_energy_kwh=0.001, baseline_duration_s=12.0,
    )
    rows = sqlite_store.read_all(path)
    assert rows[0]["baseline_crop_accuracy"] == 0.7
    assert rows[0]["baseline_disease_accuracy"] == 0.6
    assert rows[0]["baseline_energy_kwh"] == 0.001
    assert rows[0]["baseline_duration_s"] == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sqlite_store.py::test_upsert_row_accepts_baseline_columns -v`
Expected: FAIL with `sqlite3.OperationalError: table round_metrics has no column named baseline_crop_accuracy`

- [ ] **Step 3: Add the columns**

In `src/energy/sqlite_store.py`, change `SCHEMA_SQL` (lines 13-27) to:
```python
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
    baseline_crop_accuracy REAL,
    baseline_disease_accuracy REAL,
    baseline_energy_kwh REAL,
    baseline_duration_s REAL,
    recorded_at TEXT,
    PRIMARY KEY (node_id, round_idx)
);
"""
```
No other change is needed — `upsert_row` already builds its column list from whatever `**fields` it's called with, and `init_db` is idempotent (`CREATE TABLE IF NOT EXISTS`), so this is additive for any fresh db. Existing dbs from before this change are always wiped on container startup (per Global Constraints), so no migration path is needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sqlite_store.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/energy/sqlite_store.py tests/test_sqlite_store.py
git commit -m "feat: add baseline columns to round_metrics for the local_only shadow arm"
```

---

### Task 2: Shadow (local_only) model in NodeRunner

**Files:**
- Modify: `docker/node/node_runner.py`
- Modify: `docker/node/main.py:94-154` (`build_runner`)
- Test: `tests/test_node_runner.py`

**Interfaces:**
- Consumes: `sqlite_store.upsert_row` (Task 1's new columns), `Node.local_train`/`Node.evaluate` (`src/federated/node.py:58,241`, unchanged signatures).
- Produces: `NodeRunner` gains a required `shadow_node: Node` field. `handle_round_start` return dict gains `baseline_crop_accuracy`, `baseline_disease_accuracy`, `baseline_energy_kwh`, `baseline_duration_s` (all floats).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_node_runner.py`, extending `_make_runner` to build a second model/Node sharing the same loaders, and asserting the new fields:
```python
def _make_shadow_node(synthetic_dataset, train_loader, test_loader):
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    shadow_model = build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False)
    return Node("node_0", shadow_model, train_loader, test_loader, device="cpu")


def test_handle_round_start_returns_baseline_fields_and_writes_them(tmp_path, synthetic_dataset):
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    response = runner.handle_round_start(0)

    assert "baseline_crop_accuracy" in response
    assert "baseline_disease_accuracy" in response
    assert response["baseline_energy_kwh"] is not None
    assert response["baseline_duration_s"] is not None

    rows = sqlite_store.read_all(tmp_path / "node_0.db")
    assert rows[0]["baseline_crop_accuracy"] == response["baseline_crop_accuracy"]


def test_shadow_model_never_receives_distilled_knowledge(tmp_path, synthetic_dataset):
    # The shadow model must never call .distill(...) -- handle_round_gather
    # must not touch it at all. Patch it to raise if ever called.
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.shadow_node.distill = lambda *a, **k: (_ for _ in ()).throw(AssertionError("shadow must not distill"))
    runner.handle_round_start(0)
    runner.handle_round_gather(0, ["node_0"], {})  # must not raise
```

Update `_make_runner` in the same file to also build and pass `shadow_node`:
```python
def _make_runner(tmp_path, synthetic_dataset, fetch_all_knowledge=None):
    train_idx, test_idx = train_test_split_indices(list(range(len(synthetic_dataset))), 0.3, seed=1)
    train_loader = DataLoader(make_subset(synthetic_dataset, train_idx), batch_size=4, shuffle=True)
    test_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    probe_loader = DataLoader(make_subset(synthetic_dataset, test_idx), batch_size=4, shuffle=False)
    num_crop = len(synthetic_dataset.labels.crop_classes)
    num_disease = len(synthetic_dataset.labels.disease_classes)
    model = build_model("mobilenet_v3_small", num_crop, num_disease, pretrained=False)
    node = Node("node_0", model, train_loader, test_loader, device="cpu")
    shadow_node = _make_shadow_node(synthetic_dataset, train_loader, test_loader)
    tracker = ComputeEnergyTracker(enabled=False, output_dir=tmp_path, fallback_power_watts=15.0)
    runner = NodeRunner(
        node_id="node_0",
        node=node,
        shadow_node=shadow_node,
        probe_loader=probe_loader,
        tracker=tracker,
        db_path=str(tmp_path / "node_0.db"),
        fetch_all_knowledge=fetch_all_knowledge or (lambda peer_ids, peer_bases, round_idx: {}),
    )
    return runner, probe_loader
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_runner.py -v`
Expected: FAIL — `TypeError: NodeRunner.__init__() got an unexpected keyword argument 'shadow_node'`

- [ ] **Step 3: Implement the shadow step**

In `docker/node/node_runner.py`, add the field to the dataclass (after `node: Node` on line 43):
```python
    node: Node
    shadow_node: Node
```

Replace `handle_round_start` (lines 91-129) with:
```python
    def handle_round_start(self, round_idx: int) -> dict:
        self._log("round_start", f"round {round_idx}: received, starting local_train")
        # Shadow (local_only) arm runs fully BEFORE the mesh arm, sequentially
        # on the same CPU -- never concurrently. Running them concurrently
        # would have both contend for the same cores, inflating both
        # ComputeEnergyTracker.track() durations unpredictably and corrupting
        # the energy figures this run's Axis A evidence depends on. This
        # mirrors src/scenarios/harness.py's run_scenario() ordering exactly
        # (baseline arm fully computed, then the mesh arm), so the two are
        # honestly comparable under the same round_kwargs.
        with self.tracker.track(f"{self.node_id}_baseline_round_{round_idx}") as baseline_energy_record:
            self.shadow_node.local_train(self.local_epochs, self.lr)
            baseline_eval = self.shadow_node.evaluate()
        self._log(
            "round_start",
            f"round {round_idx}: local_only baseline done, "
            f"crop_acc={baseline_eval['crop_accuracy']:.4f} disease_acc={baseline_eval['disease_accuracy']:.4f}",
        )

        with self.tracker.track(f"{self.node_id}_round_{round_idx}") as energy_record:
            self.node.local_train(
                self.local_epochs, self.lr, progress_cb=self._make_progress_cb(f"round_{round_idx}_local_train")
            )
            self._log("round_start", f"round {round_idx}: local_train done, computing knowledge")
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
            baseline_crop_accuracy=baseline_eval["crop_accuracy"],
            baseline_disease_accuracy=baseline_eval["disease_accuracy"],
            baseline_energy_kwh=baseline_energy_record["energy_kwh"],
            baseline_duration_s=baseline_energy_record["duration_s"],
        )
        self._log(
            "round_start",
            f"round {round_idx}: knowledge ready ({len(data)} bytes), responding to coordinator",
        )
        return {
            "round_idx": round_idx,
            "size_bytes": len(data),
            "energy_kwh": energy_record["energy_kwh"],
            "duration_s": energy_record["duration_s"],
            "energy_method": energy_record["method"],
            "baseline_crop_accuracy": baseline_eval["crop_accuracy"],
            "baseline_disease_accuracy": baseline_eval["disease_accuracy"],
            "baseline_energy_kwh": baseline_energy_record["energy_kwh"],
            "baseline_duration_s": baseline_energy_record["duration_s"],
        }
```

In `docker/node/main.py`, `build_runner()` (lines 94-154): after the existing `model`/`node` construction (lines 124-130), build a second, independently-initialized model for the shadow arm, sharing the same loaders (matching `src/scenarios/harness.py::build_node_set`'s convention that both arms start from the same shards but independent weights):
```python
    node = Node(node_id, model, train_loader, test_loader, device="cpu")
    shadow_model = build_model(
        arch,
        len(global_map.crop_classes),
        len(global_map.disease_classes),
        pretrained=cfg.get("models.pretrained", True),
    )
    shadow_node = Node(node_id, shadow_model, train_loader, test_loader, device="cpu")
```
Then pass it through in the `NodeRunner(...)` construction (line 137): add `shadow_node=shadow_node,` right after `node=node,`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_runner.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add docker/node/node_runner.py docker/node/main.py tests/test_node_runner.py
git commit -m "feat: train a local_only shadow model on every node, every round"
```

---

### Task 3: Persist baseline fields in CoordinatorRunner

**Files:**
- Modify: `docker/coordinator/coordinator_runner.py:62-90`
- Test: `tests/test_coordinator_runner.py`

**Interfaces:**
- Consumes: `/round/start` response dict now containing `baseline_crop_accuracy`/`baseline_disease_accuracy`/`baseline_energy_kwh`/`baseline_duration_s` (Task 2).
- Produces: `merged.db`'s `round_metrics` rows carry the same 4 columns, sourced entirely from the `/round/start` response (no `/round/gather` contract change needed — the shadow arm has no gather-phase dependency).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_coordinator_runner.py`:
```python
def test_run_round_persists_baseline_fields_from_round_start_response(tmp_path):
    def post_all(node_ids, path, body):
        if path == "/round/start":
            return {
                n: {
                    "energy_kwh": 0.01, "duration_s": 1.0, "energy_method": "proxy_wall_power",
                    "size_bytes": 100,
                    "baseline_crop_accuracy": 0.55, "baseline_disease_accuracy": 0.45,
                    "baseline_energy_kwh": 0.009, "baseline_duration_s": 0.9,
                }
                for n in node_ids
            }
        return {n: {"crop_accuracy": 0.5, "disease_accuracy": 0.6} for n in node_ids}

    runner = _make_runner(tmp_path, post_all)
    runner.run_round(0)

    rows = {r["node_id"]: r for r in sqlite_store.read_all(str(tmp_path / "merged.db"))}
    assert rows["node_0"]["baseline_crop_accuracy"] == 0.55
    assert rows["node_0"]["baseline_disease_accuracy"] == 0.45
    assert rows["node_0"]["baseline_energy_kwh"] == 0.009
    assert rows["node_0"]["baseline_duration_s"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coordinator_runner.py::test_run_round_persists_baseline_fields_from_round_start_response -v`
Expected: FAIL — `KeyError: 'baseline_crop_accuracy'` (column exists from Task 1, but nothing writes it yet)

- [ ] **Step 3: Persist the fields**

In `docker/coordinator/coordinator_runner.py`, in `run_round` (lines 67-89), extend the `upsert_row` call inside the `for node_id in active:` loop that follows `/round/start`:
```python
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
                knowledge_bytes_sent=r["size_bytes"] * max(0, len(active) - 1),
                active=1,
                baseline_crop_accuracy=r["baseline_crop_accuracy"],
                baseline_disease_accuracy=r["baseline_disease_accuracy"],
                baseline_energy_kwh=r["baseline_energy_kwh"],
                baseline_duration_s=r["baseline_duration_s"],
            )
```
(Only the added `baseline_*` kwargs are new; the rest of the call is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_coordinator_runner.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add docker/coordinator/coordinator_runner.py tests/test_coordinator_runner.py
git commit -m "feat: persist local_only baseline fields into merged.db"
```

---

### Task 4: Per-scenario log files for NodeRunner and CoordinatorRunner

**Files:**
- Modify: `docker/node/node_runner.py` (`_log`, dataclass fields)
- Modify: `docker/coordinator/coordinator_runner.py` (`_log`, dataclass fields)
- Test: `tests/test_node_runner.py`, `tests/test_coordinator_runner.py`

**Interfaces:**
- Produces: `NodeRunner(..., log_file: str | None = None)` and `CoordinatorRunner(..., log_file: str | None = None)`. When set, every `_log(...)` call appends one JSON line (`{"ts": float, "stage": str, "message": str}` for nodes; `{"ts": float, "message": str}` for the coordinator) to that path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_node_runner.py`:
```python
def test_log_file_receives_one_json_line_per_log_call(tmp_path, synthetic_dataset):
    log_path = tmp_path / "node_0.log"
    runner, _ = _make_runner(tmp_path, synthetic_dataset)
    runner.log_file = str(log_path)
    runner._log("round_start", "hello")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "round_start"
    assert entry["message"] == "hello"
```
(Add `import json` to the top of `tests/test_node_runner.py` if not already present.)

Add to `tests/test_coordinator_runner.py`:
```python
def test_log_file_receives_one_json_line_per_log_call(tmp_path):
    log_path = tmp_path / "coordinator.log"
    runner = _make_runner(tmp_path, lambda *a: {})
    runner.log_file = str(log_path)
    runner._log("hello")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["message"] == "hello"
```
(`json` is already imported at the top of `tests/test_coordinator_runner.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_node_runner.py::test_log_file_receives_one_json_line_per_log_call tests/test_coordinator_runner.py::test_log_file_receives_one_json_line_per_log_call -v`
Expected: FAIL — `AttributeError`/`TypeError` (`log_file` field does not exist yet)

- [ ] **Step 3: Implement**

In `docker/node/node_runner.py`, add `import json` to the imports (near `import time`). Add the new field **after** `temperature: float = 2.0` (the last existing defaulted field, right before the `init=False` bookkeeping fields) — it must come after every field without a default (`db_path`, `fetch_all_knowledge`, etc.), or the dataclass definition raises `TypeError` at import time:
```python
    temperature: float = 2.0
    log_file: str | None = None
    _last_round_idx: int | None = field(default=None, init=False)
```
Replace `_log` (lines 63-68) with:
```python
    def _log(self, stage: str, message: str) -> None:
        entry = {"ts": time.time(), "stage": stage, "message": message}
        self.activity_log.append(entry)
        del self.activity_log[:-MAX_ACTIVITY_LOG]
        if self.log_file:
            if not self._log_dir_ready:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_dir_ready = True
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
```
Add the supporting import (`from pathlib import Path`) and dataclass field:
```python
    _log_dir_ready: bool = field(default=False, init=False)
```

In `docker/coordinator/coordinator_runner.py`, `Path` is already imported at line 16. Add the new field **after** `status_path: str | None = None` (the last existing defaulted field, right before `activity_log`, which is `init=False`):
```python
    status_path: str | None = None
    log_file: str | None = None
    activity_log: list = field(default_factory=list, init=False)
    _log_dir_ready: bool = field(default=False, init=False)
```
Then replace `_log` (lines 49-54) with:
```python
    def _log(self, message: str) -> None:
        entry = {"ts": time.time(), "message": message}
        self.activity_log.append(entry)
        del self.activity_log[:-MAX_ACTIVITY_LOG]
        if self.log_file:
            if not self._log_dir_ready:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                self._log_dir_ready = True
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_node_runner.py tests/test_coordinator_runner.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add docker/node/node_runner.py docker/coordinator/coordinator_runner.py tests/test_node_runner.py tests/test_coordinator_runner.py
git commit -m "feat: write activity log lines to a file, not just in-memory"
```

---

### Task 5: SCENARIO-namespaced paths, wired through main.py and docker-compose.yml

**Files:**
- Modify: `docker/node/main.py:94-154`
- Modify: `docker/coordinator/main.py:77-135`
- Modify: `docker/docker-compose.yml`

**Interfaces:**
- Produces: every node/coordinator container now reads a `SCENARIO` env var (default `full_run` if unset, mirroring existing `os.environ.get(..., default)` patterns already used in these files) and namespaces its own `ENERGY_DB`/`LOG_FILE` paths accordingly, wiping only its own scenario's files on startup.

This task is config/wiring — no new pure logic to unit test (Tasks 2-4 already covered the logic these paths feed into). Verification is a `docker compose config` dry run plus the manual end-to-end pass in Task 10.

- [ ] **Step 1: Add `LOG_FILE` wiring to `docker/node/main.py`**

In `build_runner()` (lines 94-154), after the existing `energy_db = os.environ["ENERGY_DB"]` (line 99) and its `unlink` (line 105), add:
```python
    log_file = os.environ.get("LOG_FILE")
    Path(energy_db).unlink(missing_ok=True)
    if log_file:
        Path(log_file).unlink(missing_ok=True)
```
(Replace the existing single-line `Path(energy_db).unlink(missing_ok=True)` with the block above — same effect for `energy_db`, plus wiping the log file, so a scenario restart never appends to a previous run's log.)

Pass it through in the `NodeRunner(...)` construction (after Task 2's `shadow_node=shadow_node,` line): add `log_file=log_file,`.

- [ ] **Step 2: Add `LOG_FILE` wiring to `docker/coordinator/main.py`**

In `main()` (lines 77-135), after the existing `status_path`/unlink block (lines 95-101), add:
```python
    log_file = os.environ.get("LOG_FILE")
    Path(energy_db).unlink(missing_ok=True)
    Path(status_path).unlink(missing_ok=True)
    if log_file:
        Path(log_file).unlink(missing_ok=True)
```
Pass it through in the `CoordinatorRunner(...)` construction (line 103): add `log_file=log_file,` after `status_path=status_path,`.

- [ ] **Step 3: Rewrite `docker/docker-compose.yml`**

Replace the entire file with:
```yaml
# docker/docker-compose.yml
name: crop-mesh

services:
  coordinator:
    build:
      context: ..
      dockerfile: docker/coordinator/Dockerfile
    image: crop-mesh-coordinator:latest
    depends_on: [node_0, node_1, node_2]
    ports:
      - "9000:9000"
    environment:
      CONFIG_PATH: /config/config.yaml
      SCENARIO: ${SCENARIO:-full_run}
      ENERGY_DB: /energy/${SCENARIO:-full_run}/merged.db
      LOG_FILE: /energy/${SCENARIO:-full_run}/coordinator.log
    volumes:
      - ${HOST_PROJECT_ROOT}/config.yaml:/config/config.yaml:ro
      - ${HOST_PROJECT_ROOT}/outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_0:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    image: crop-mesh-node:latest
    environment:
      NODE_ID: node_0
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_0
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      SCENARIO: ${SCENARIO:-full_run}
      ENERGY_DB: /energy/${SCENARIO:-full_run}/node_0.db
      LOG_FILE: /energy/${SCENARIO:-full_run}/node_0.log
    volumes:
      - ${HOST_PROJECT_ROOT}/config.yaml:/config/config.yaml:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/node_0:/data/node_0:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/probe:/data/probe:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/classes.json:/data/classes.json:ro
      - ${HOST_PROJECT_ROOT}/outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_1:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    image: crop-mesh-node:latest
    environment:
      NODE_ID: node_1
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_1
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      SCENARIO: ${SCENARIO:-full_run}
      ENERGY_DB: /energy/${SCENARIO:-full_run}/node_1.db
      LOG_FILE: /energy/${SCENARIO:-full_run}/node_1.log
    volumes:
      - ${HOST_PROJECT_ROOT}/config.yaml:/config/config.yaml:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/node_1:/data/node_1:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/probe:/data/probe:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/classes.json:/data/classes.json:ro
      - ${HOST_PROJECT_ROOT}/outputs/docker_mesh/energy:/energy
    networks: [mesh]

  node_2:
    build:
      context: ..
      dockerfile: docker/node/Dockerfile
    image: crop-mesh-node:latest
    environment:
      NODE_ID: node_2
      CONFIG_PATH: /config/config.yaml
      DATA_ROOT: /data/node_2
      PROBE_ROOT: /data/probe
      CLASSES_JSON: /data/classes.json
      SCENARIO: ${SCENARIO:-full_run}
      ENERGY_DB: /energy/${SCENARIO:-full_run}/node_2.db
      LOG_FILE: /energy/${SCENARIO:-full_run}/node_2.log
    volumes:
      - ${HOST_PROJECT_ROOT}/config.yaml:/config/config.yaml:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/node_2:/data/node_2:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/probe:/data/probe:ro
      - ${HOST_PROJECT_ROOT}/data/docker_mesh/classes.json:/data/classes.json:ro
      - ${HOST_PROJECT_ROOT}/outputs/docker_mesh/energy:/energy
    networks: [mesh]

  controller:
    build:
      context: ..
      dockerfile: docker/controller/Dockerfile
    image: crop-mesh-controller:latest
    ports:
      - "9100:9100"
    environment:
      HOST_PROJECT_ROOT: ${HOST_PROJECT_ROOT}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ${HOST_PROJECT_ROOT}:/workspace
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
      NODE_URL_TEMPLATE: http://{node_id}:8000
      COORDINATOR_EVENTS_URL: http://coordinator:9000/events
      CONTROLLER_URL: http://controller:9100
      ENERGY_DIR: /energy
      CONFIG_PATH: /config/config.yaml
      REFRESH_S: "3"
    volumes:
      - ${HOST_PROJECT_ROOT}/config.yaml:/config/config.yaml:ro
      - ${HOST_PROJECT_ROOT}/outputs/docker_mesh/energy:/energy:ro
    networks: [mesh]

networks:
  mesh: {}
```
Note: `controller` is intentionally not in `dashboard`'s `depends_on` — the dashboard must render even if the controller is briefly unreachable (it already treats every HTTP call as best-effort via `httpx`/`try`/`except`), and `controller` has no dependency on the other services being healthy to itself start.

- [ ] **Step 4: Verify the compose file parses and interpolates correctly**

Run (with a placeholder value, from the `docker/` directory):
```bash
HOST_PROJECT_ROOT=/tmp/fake docker compose config --quiet
```
Expected: exits 0 with no output (`--quiet` suppresses the rendered YAML on success; a non-zero exit or YAML error means a syntax mistake in the rewrite).

- [ ] **Step 5: Commit**

```bash
git add docker/node/main.py docker/coordinator/main.py docker/docker-compose.yml
git commit -m "feat: namespace container energy/log paths by SCENARIO, add controller service and HOST_PROJECT_ROOT"
```

---

### Task 6: ControllerRunner state machine

**Files:**
- Create: `docker/controller/controller_runner.py`
- Test: `tests/test_controller_runner.py`

**Interfaces:**
- Consumes: nothing from other tasks — pure, dependency-injected logic.
- Produces: `ControllerRunner(start_scenario: Callable[[str], None], stop_scenario: Callable[[], None])` with methods `.start(scenario: str) -> dict`, `.stop() -> dict`, `.status() -> dict`. `start_scenario`/`stop_scenario` are expected to raise on failure (any exception type); `ControllerRunner` catches `Exception`, not a specific type. Task 7 supplies the real callables.

- [ ] **Step 1: Write the failing test**

Create `tests/test_controller_runner.py`:
```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docker" / "controller"))

import pytest
from controller_runner import ControllerRunner  # noqa: E402


def _make_runner(start_scenario=None, stop_scenario=None):
    return ControllerRunner(
        start_scenario=start_scenario or (lambda scenario: None),
        stop_scenario=stop_scenario or (lambda: None),
    )


def test_initial_state_is_idle():
    runner = _make_runner()
    assert runner.status() == {
        "running_scenario": None, "state": "idle", "started_at": None, "error_detail": None,
    }


def test_start_moves_to_running_and_records_scenario():
    runner = _make_runner()
    result = runner.start("full_run")
    assert result["state"] == "running"
    assert result["running_scenario"] == "full_run"
    assert result["started_at"] is not None


def test_start_rejects_unknown_scenario():
    runner = _make_runner()
    with pytest.raises(ValueError):
        runner.start("not_a_real_scenario")
    assert runner.status()["state"] == "idle"  # rejected before any state change


def test_start_stops_the_previously_running_scenario_first():
    calls = []
    runner = _make_runner(
        start_scenario=lambda s: calls.append(("start", s)),
        stop_scenario=lambda: calls.append(("stop",)),
    )
    runner.start("full_run")
    runner.start("disconnection")
    assert calls == [("start", "full_run"), ("stop",), ("start", "disconnection")]
    assert runner.status()["running_scenario"] == "disconnection"


def test_stop_when_idle_is_a_noop():
    calls = []
    runner = _make_runner(stop_scenario=lambda: calls.append("stop"))
    result = runner.stop()
    assert result["state"] == "idle"
    assert calls == []


def test_start_failure_moves_to_error_state_with_detail():
    def failing_start(scenario):
        raise RuntimeError("docker compose exploded")

    runner = _make_runner(start_scenario=failing_start)
    with pytest.raises(RuntimeError):
        runner.start("full_run")
    status = runner.status()
    assert status["state"] == "error"
    assert "docker compose exploded" in status["error_detail"]


def test_stop_is_callable_again_after_an_error_and_recovers_to_idle():
    calls = []
    runner = _make_runner(
        start_scenario=lambda s: (_ for _ in ()).throw(RuntimeError("boom")),
        stop_scenario=lambda: calls.append("stop"),
    )
    with pytest.raises(RuntimeError):
        runner.start("full_run")
    assert runner.status()["state"] == "error"

    result = runner.stop()
    assert calls == ["stop"]
    assert result["state"] == "idle"
    assert result["error_detail"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_controller_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'controller_runner'`

- [ ] **Step 3: Implement `ControllerRunner`**

Create `docker/controller/controller_runner.py`:
```python
"""Pure start/stop/mutual-exclusion state machine for the scenario
control plane. Docker/subprocess calls are injected as plain callables
(mirroring docker/node/node_runner.py's fetch_all_knowledge and
docker/coordinator/coordinator_runner.py's post_all) so this class is
unit-testable with fakes -- see docker/controller/main.py for the real
docker-compose-via-subprocess wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

VALID_SCENARIOS = {"full_run", "class_addition", "disconnection", "distribution_shift"}

# Raises on failure; returns nothing on success.
StartScenario = Callable[[str], None]
StopScenario = Callable[[], None]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ControllerRunner:
    start_scenario: StartScenario
    stop_scenario: StopScenario
    state: str = "idle"  # idle | starting | running | stopping | error
    running_scenario: str | None = None
    started_at: str | None = None
    error_detail: str | None = None

    def status(self) -> dict:
        return {
            "running_scenario": self.running_scenario,
            "state": self.state,
            "started_at": self.started_at,
            "error_detail": self.error_detail,
        }

    def start(self, scenario: str) -> dict:
        if scenario not in VALID_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}, must be one of {sorted(VALID_SCENARIOS)}")
        if self.state in ("starting", "stopping"):
            raise RuntimeError(f"cannot start while controller is {self.state}")
        if self.state in ("running", "error"):
            self._do_stop()
        self.state = "starting"
        try:
            self.start_scenario(scenario)
        except Exception as exc:
            self.state = "error"
            self.error_detail = str(exc)
            raise
        self.state = "running"
        self.running_scenario = scenario
        self.started_at = _now_iso()
        self.error_detail = None
        return self.status()

    def stop(self) -> dict:
        if self.state in ("starting", "stopping"):
            raise RuntimeError(f"cannot stop while controller is {self.state}")
        if self.state == "idle":
            return self.status()
        self._do_stop()
        return self.status()

    def _do_stop(self) -> None:
        self.state = "stopping"
        try:
            self.stop_scenario()
        except Exception as exc:
            self.state = "error"
            self.error_detail = str(exc)
            raise
        self.state = "idle"
        self.running_scenario = None
        self.started_at = None
        self.error_detail = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_controller_runner.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add docker/controller/controller_runner.py tests/test_controller_runner.py
git commit -m "feat: add ControllerRunner start/stop/mutual-exclusion state machine"
```

---

### Task 7: Controller FastAPI service + Docker wiring

**Files:**
- Create: `docker/controller/main.py`
- Create: `docker/controller/Dockerfile`
- Create: `docker/controller/requirements.txt`

**Interfaces:**
- Consumes: `ControllerRunner` (Task 6).
- Produces: `GET /status`, `POST /start {"scenario": str}`, `POST /stop` — the HTTP contract Task 9's dashboard calls.

This task wraps Task 6's already-tested logic in FastAPI and wires the two real callables; there is no new pure logic to unit-test, so verification is manual (build + curl), done as part of Task 10's end-to-end pass. Keep the subprocess logic in a small, separately-named function so it stays easy to read/patch later, even though it isn't unit-tested here.

- [ ] **Step 1: Create `docker/controller/requirements.txt`**
```
fastapi>=0.115
uvicorn>=0.30
```

- [ ] **Step 2: Create `docker/controller/main.py`**
```python
# docker/controller/main.py
"""FastAPI wrapper around ControllerRunner. Talks to the *host* Docker
daemon over the mounted socket by shelling out to `docker compose`
against the SAME docker-compose.yml the user brought the stack up with
originally (bind-mounted read-only at /workspace) -- never a
docker-compose.yml baked into this image, so there is exactly one
source of truth for the topology. All volume paths in that file use
${HOST_PROJECT_ROOT} (absolute), never a relative `..`, because a
`docker compose` process running inside a container resolves relative
paths against ITS OWN filesystem view, then hands the daemon a path
that means something different on the real host -- absolute paths
sidestep that mismatch entirely.
"""

from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI, HTTPException

from controller_runner import ControllerRunner

COMPOSE_FILE = "/workspace/docker/docker-compose.yml"
MANAGED_SERVICES = ["coordinator", "node_0", "node_1", "node_2"]
COMPOSE_TIMEOUT_S = 120


def _run_compose(args: list[str], env_overrides: dict) -> None:
    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, *args],
        env=env, capture_output=True, text=True, timeout=COMPOSE_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed: {result.stderr.strip()}")


def start_scenario(scenario: str) -> None:
    _run_compose(["up", "-d", "--force-recreate", *MANAGED_SERVICES], {"SCENARIO": scenario})


def stop_scenario() -> None:
    _run_compose(["stop", *MANAGED_SERVICES], {})


runner = ControllerRunner(start_scenario=start_scenario, stop_scenario=stop_scenario)
app = FastAPI()


@app.get("/status")
def status():
    return runner.status()


@app.post("/start")
def start(body: dict):
    if "scenario" not in body:
        raise HTTPException(status_code=400, detail="missing required field: 'scenario'")
    try:
        return runner.start(body["scenario"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/stop")
def stop():
    try:
        return runner.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9100)
```

- [ ] **Step 3: Create `docker/controller/Dockerfile`**
```dockerfile
# docker/controller/Dockerfile
# docker:27-cli ships the `docker` CLI and the `compose` plugin, needed to
# drive the host daemon over the mounted socket -- this image never runs
# its OWN daemon, only the client talking out through the socket.
FROM docker:27-cli

RUN apk add --no-cache python3 py3-pip

WORKDIR /app
COPY docker/controller/requirements.txt .
# --break-system-packages: Alpine's system python3 is PEP-668
# externally-managed; this is a throwaway container image, not a shared
# host Python, so overriding that guard here is safe.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY docker/controller/main.py docker/controller/controller_runner.py ./

EXPOSE 9100
CMD ["python3", "main.py"]
```

- [ ] **Step 4: Verify the image builds**

Run (from the repo root):
```bash
docker build -f docker/controller/Dockerfile -t crop-mesh-controller:latest .
```
Expected: build succeeds (exit 0).

- [ ] **Step 5: Commit**

```bash
git add docker/controller/main.py docker/controller/Dockerfile docker/controller/requirements.txt
git commit -m "feat: add controller FastAPI service wrapping docker compose start/stop"
```

---

### Task 8: Dashboard data helpers — scenario paths, fairness disclosure, export schema

**Files:**
- Modify: `docker/dashboard/data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Produces:
  - `SCENARIOS = ["full_run", "class_addition", "disconnection", "distribution_shift"]`
  - `scenario_paths(energy_dir: str, scenario: str) -> dict` → `{"merged_db", "node_dbs" (dict per node_id), "status_path", "log_paths" (dict per node_id + "coordinator")}`
  - `read_log_file(path: str) -> list[dict]`
  - `build_fairness_disclosure(rows: list, cfg) -> dict`
  - `build_export_payload(scenario: str, rows: list, transfers: list, status: dict | None, cfg) -> dict` (the unified schema from spec §6)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_dashboard_data.py`:
```python
def test_scenario_paths_builds_per_node_and_shared_paths():
    paths = scenario_paths("/energy", "disconnection", num_nodes=2)
    assert paths["merged_db"] == "/energy/disconnection/merged.db"
    assert paths["node_dbs"] == {
        "node_0": "/energy/disconnection/node_0.db",
        "node_1": "/energy/disconnection/node_1.db",
    }
    assert paths["status_path"] == "/energy/disconnection/status.json"
    assert paths["log_paths"]["node_0"] == "/energy/disconnection/node_0.log"
    assert paths["log_paths"]["coordinator"] == "/energy/disconnection/coordinator.log"


def test_read_log_file_returns_empty_list_when_file_missing(tmp_path):
    assert read_log_file(str(tmp_path / "missing.log")) == []


def test_read_log_file_parses_one_json_object_per_line(tmp_path):
    path = tmp_path / "node_0.log"
    path.write_text('{"ts": 1.0, "stage": "a", "message": "x"}\n{"ts": 2.0, "stage": "b", "message": "y"}\n')
    entries = read_log_file(str(path))
    assert [e["message"] for e in entries] == ["x", "y"]


def test_build_fairness_disclosure_reports_macro_avg_and_worst_node():
    rows = [
        {"node_id": "node_0", "round_idx": 0, "crop_accuracy": 0.80, "disease_accuracy": 0.70,
         "baseline_crop_accuracy": 0.60, "baseline_disease_accuracy": 0.50},
        {"node_id": "node_1", "round_idx": 0, "crop_accuracy": 0.55, "disease_accuracy": 0.55,
         "baseline_crop_accuracy": 0.50, "baseline_disease_accuracy": 0.50},
    ]
    cfg = {
        "training.local_epochs_per_round": 1, "training.distill_epochs_per_round": 1,
        "training.lr": 0.001, "training.distill_lr": 0.0005,
        "data.non_iid_strategy": "manual", "data.test_fraction": 0.15,
    }
    table = build_fairness_disclosure(rows, cfg)
    assert table["node_count"] == 2
    # node_0 delta: crop +0.20, disease +0.20 -> avg 0.20; node_1: crop +0.05, disease +0.05 -> avg 0.05
    assert table["macro_avg_and_worst_node"]["macro_avg"] == pytest.approx(0.125)
    assert table["macro_avg_and_worst_node"]["worst_node"] == pytest.approx(0.05)
    assert table["per_node_scores"]["node_0"]["mesh"]["crop_accuracy"] == 0.80
    assert table["per_node_scores"]["node_0"]["baseline"]["crop_accuracy"] == 0.60
    assert "delta_g_formula" in table and table["delta_g_formula"]


def test_build_export_payload_includes_fairness_disclosure_and_scenario_name():
    rows = [{"node_id": "node_0", "round_idx": 0, "crop_accuracy": 0.8, "disease_accuracy": 0.7,
             "baseline_crop_accuracy": 0.6, "baseline_disease_accuracy": 0.5}]
    cfg = {"training.local_epochs_per_round": 1, "training.distill_epochs_per_round": 1,
           "training.lr": 0.001, "training.distill_lr": 0.0005,
           "data.non_iid_strategy": "manual", "data.test_fraction": 0.15}
    payload = build_export_payload("disconnection", rows, transfers=[], status=None, cfg=cfg)
    assert payload["scenario"] == "disconnection"
    assert payload["complete"] is False
    assert "fairness_disclosure" in payload
    assert payload["nodes"]["node_0"] == rows
```
(Add `import pytest` to the top of `tests/test_dashboard_data.py` if not already present, and add `scenario_paths`, `read_log_file`, `build_fairness_disclosure`, `build_export_payload` to the existing `from data import (...)` block.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'scenario_paths'`

- [ ] **Step 3: Implement**

Add to `docker/dashboard/data.py`:
```python
SCENARIOS = ["full_run", "class_addition", "disconnection", "distribution_shift"]


def scenario_paths(energy_dir: str, scenario: str, num_nodes: int = 3) -> dict:
    """All namespaced paths for one scenario's data, rooted at energy_dir
    (e.g. "/energy"). Mirrors exactly what docker-compose.yml's
    ${SCENARIO}-templated ENERGY_DB/LOG_FILE env vars point node/coordinator
    containers at, so the dashboard reads from the same place they write to.
    """
    base = f"{energy_dir}/{scenario}"
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    return {
        "merged_db": f"{base}/merged.db",
        "node_dbs": {node_id: f"{base}/{node_id}.db" for node_id in node_ids},
        "status_path": f"{base}/status.json",
        "log_paths": {**{node_id: f"{base}/{node_id}.log" for node_id in node_ids}, "coordinator": f"{base}/coordinator.log"},
    }


def read_log_file(path: str) -> list:
    """Reads a JSONL activity log written by NodeRunner/CoordinatorRunner's
    _log(). Returns [] if the file doesn't exist yet (scenario never run) --
    same "no rows" convention as sqlite_store.read_all for a missing db.
    """
    if not Path(path).exists():
        return []
    entries = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _cfg_get(cfg, key: str, default=None):
    """cfg may be a real src.config.Config or a plain dict keyed by dotted
    path (what the unit tests pass) -- both expose a compatible
    .get(key, default) method, so no branching is needed.
    """
    return cfg.get(key, default)


def build_fairness_disclosure(rows: list, cfg) -> dict:
    """Appendix A.1's Collaboration-Gain Fairness Disclosure Table, computed
    from the last round each node reported. ΔG per node/metric is
    mesh_accuracy - baseline_accuracy; macro_avg is the mean of every
    node's own (crop+disease)-averaged ΔG, worst_node is the minimum of
    those per-node averages -- reported separately per §4.3's "macro
    average and the worst-node score (not just the best or the overall
    average)".
    """
    node_ids = sorted({r["node_id"] for r in rows})
    per_node_scores = {}
    per_node_deltas = []
    for node_id in node_ids:
        node_rows = [r for r in rows if r["node_id"] == node_id]
        last = max(node_rows, key=lambda r: r["round_idx"])
        mesh = {"crop_accuracy": last.get("crop_accuracy"), "disease_accuracy": last.get("disease_accuracy")}
        baseline = {
            "crop_accuracy": last.get("baseline_crop_accuracy"),
            "disease_accuracy": last.get("baseline_disease_accuracy"),
        }
        per_node_scores[node_id] = {"mesh": mesh, "baseline": baseline}
        deltas = [mesh[m] - baseline[m] for m in ("crop_accuracy", "disease_accuracy")
                  if mesh[m] is not None and baseline[m] is not None]
        if deltas:
            per_node_deltas.append(sum(deltas) / len(deltas))

    macro_avg = sum(per_node_deltas) / len(per_node_deltas) if per_node_deltas else None
    worst_node = min(per_node_deltas) if per_node_deltas else None

    local_epochs = _cfg_get(cfg, "training.local_epochs_per_round")
    lr = _cfg_get(cfg, "training.lr")
    distill_epochs = _cfg_get(cfg, "training.distill_epochs_per_round")
    distill_lr = _cfg_get(cfg, "training.distill_lr")

    return {
        "task_metric": "crop_accuracy, disease_accuracy",
        "node_count": len(node_ids),
        "data_split": f"per-node held-out test split, test_fraction={_cfg_get(cfg, 'data.test_fraction')}",
        "non_iid_description": f"non_iid_strategy={_cfg_get(cfg, 'data.non_iid_strategy')}",
        "test_set_scope": "local test (per-node held-out split; test samples never enter any training set)",
        "local_only_budget": {"local_epochs_per_round": local_epochs, "lr": lr},
        "collective_budget": {
            "local_epochs_per_round": local_epochs, "lr": lr,
            "distill_epochs_per_round": distill_epochs, "distill_lr": distill_lr,
        },
        "per_node_scores": per_node_scores,
        "macro_avg_and_worst_node": {"macro_avg": macro_avg, "worst_node": worst_node},
        "delta_g_formula": (
            "per-node delta = mean(mesh_accuracy - baseline_accuracy) over {crop_accuracy, disease_accuracy}, "
            "using each node's own last completed round; macro_avg is the mean across nodes, "
            "worst_node is the minimum across nodes"
        ),
    }


def build_export_payload(scenario: str, rows: list, transfers: list, status: dict | None, cfg) -> dict:
    """The single export schema shared by every tab (spec §6) -- extends
    the existing final_result.json shape (status/nodes/knowledge_transfers)
    with the fairness disclosure table. `complete` mirrors is_run_complete
    so a partial/crashed run's export is never mistaken for a finished one.
    """
    base = build_final_result_payload(rows, transfers, status)
    return {
        "scenario": scenario,
        "target_node": None,
        "perturbation_applied": False,
        "complete": is_run_complete(status),
        **base,
        "fairness_disclosure": build_fairness_disclosure(rows, cfg),
    }
```
Add `from pathlib import Path` to the top-of-file imports (alongside the existing `import json`, `import time`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dashboard_data.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add docker/dashboard/data.py tests/test_dashboard_data.py
git commit -m "feat: add scenario-path, log-file, and fairness-disclosure helpers to dashboard data"
```

---

### Task 9: Dashboard tabs — Start/Stop, collapsed panels, per-scenario rendering

**Files:**
- Modify: `docker/dashboard/app.py`

**Interfaces:**
- Consumes: `data.SCENARIOS`, `data.scenario_paths`, `data.read_log_file`, `data.build_export_payload` (Task 8); controller's `GET /status`, `POST /start`, `POST /stop` (Task 7).

This task is Streamlit UI code with no meaningful unit-testable logic of its own (all the logic it calls is already tested in Task 8); verification is the manual end-to-end pass in Task 10.

- [ ] **Step 1: Add controller/config wiring and scenario labels**

In `docker/dashboard/app.py`, after the existing env var block (around line 50), add:
```python
CONTROLLER_URL = os.environ.get("CONTROLLER_URL", "http://controller:9100")
ENERGY_DIR = os.environ.get("ENERGY_DIR", "/energy")
CONFIG_PATH = os.environ.get("CONFIG_PATH")

SCENARIO_LABELS = {
    "full_run": "Full Run",
    "class_addition": "Class Addition",
    "disconnection": "Disconnection",
    "distribution_shift": "Distribution Shift",
}
```
Add the new imports at the top:
```python
from src.config import Config
from data import (
    build_export_payload,
    build_final_result_payload,
    build_log_lines,
    build_status_rows,
    is_run_complete,
    merge_transfer_rows,
    read_log_file,
    rows_for_node,
    scenario_paths,
    to_json_str,
    SCENARIOS,
)
```

- [ ] **Step 2: Add controller HTTP helpers**

Add near the existing `_poll_json`/`_poll_health` helpers:
```python
def _controller_status() -> dict:
    try:
        resp = httpx.get(f"{CONTROLLER_URL}/status", timeout=3.0)
        return resp.json() if resp.status_code == 200 else {"state": "unreachable", "running_scenario": None, "error_detail": None}
    except httpx.HTTPError:
        return {"state": "unreachable", "running_scenario": None, "error_detail": None}


def _controller_start(scenario: str) -> None:
    try:
        resp = httpx.post(f"{CONTROLLER_URL}/start", json={"scenario": scenario}, timeout=10.0)
        if resp.status_code >= 400:
            st.error(f"Start failed: {resp.json().get('detail', resp.text)}")
    except httpx.HTTPError as exc:
        st.error(f"Could not reach controller: {exc}")


def _controller_stop() -> None:
    try:
        resp = httpx.post(f"{CONTROLLER_URL}/stop", timeout=10.0)
        if resp.status_code >= 400:
            st.error(f"Stop failed: {resp.json().get('detail', resp.text)}")
    except httpx.HTTPError as exc:
        st.error(f"Could not reach controller: {exc}")
```

- [ ] **Step 3: Extract the existing single-scenario body into `render_tab`**

Replace the body of `render()` from `st.subheader("Node activity")` (line 196) down to (but not including) the trailing `time.sleep(REFRESH_S)` (line 292) with a call to a new per-tab function, and define that function above `render()`:
```python
def _render_tab(scenario: str, controller_status: dict) -> None:
    label = SCENARIO_LABELS[scenario]
    paths = scenario_paths(ENERGY_DIR, scenario, num_nodes=NUM_NODES)

    running_here = controller_status.get("running_scenario") == scenario
    busy_elsewhere = controller_status.get("state") in ("starting", "running", "stopping") and not running_here
    state = controller_status.get("state", "unreachable")

    header_col, status_col = st.columns([5, 2])
    header_col.subheader(label)
    status_col.caption(f"state: {state}" + (f" ({controller_status['error_detail']})" if controller_status.get("error_detail") else ""))

    start_col, stop_col = st.columns([1, 1])
    if start_col.button("Start", key=f"start_{scenario}", disabled=busy_elsewhere or state in ("starting", "stopping")):
        _controller_start(scenario)
        st.rerun()
    if stop_col.button("Stop", key=f"stop_{scenario}", disabled=not running_here or state in ("starting", "stopping")):
        _controller_stop()
        st.rerun()

    st.caption("Node activity")
    cols = st.columns(NUM_NODES)
    for col, node_id in zip(cols, [f"node_{i}" for i in range(NUM_NODES)]):
        with col:
            with st.expander(node_id, expanded=False):
                components.html(
                    _log_panel_html(read_log_file(paths["log_paths"][node_id]), panel_key=f"{scenario}_{node_id}"),
                    height=280, scrolling=False,
                )
    with st.expander("coordinator", expanded=False):
        components.html(
            _log_panel_html(read_log_file(paths["log_paths"]["coordinator"]), panel_key=f"{scenario}_coordinator"),
            height=280, scrolling=False,
        )

    rows = read_all(paths["merged_db"])
    with st.expander("Per-round results", expanded=False):
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("No rows yet.")

    cfg = Config.load(CONFIG_PATH)
    with st.expander("Collaboration-Gain Fairness Disclosure Table (Appendix A.1)", expanded=False):
        if rows:
            st.json(build_fairness_disclosure(rows, cfg))
        else:
            st.write("No data yet.")

    status = None
    if Path(paths["status_path"]).exists():
        try:
            status = json.loads(Path(paths["status_path"]).read_text())
        except (json.JSONDecodeError, OSError):
            status = None

    with st.expander("Summary", expanded=False):
        if is_run_complete(status):
            st.success(f"All {status['num_rounds']} round(s) complete at {status['completed_at']}.")
        else:
            st.info("Run still in progress or not started.")

    transfers = merge_transfer_rows([read_transfers(path) for path in paths["node_dbs"].values()])
    st.download_button(
        f"Download {scenario}_result.json",
        data=to_json_str(build_export_payload(scenario, rows, transfers, status, cfg)),
        file_name=f"{scenario}_result.json",
        mime="application/json",
        key=f"export_{scenario}",
    )
```
`from pathlib import Path` is already present at the top of the file (used for `STATUS_PATH`); add `build_fairness_disclosure` to the `from data import (...)` block added in Step 1.

Note: per-node/coordinator panels are each their own top-level `st.expander`, never nested inside another expander — Streamlit does not allow nesting expanders (it raises at render time), so the "Node activity" grouping above is a plain `st.caption`, not a wrapping expander.

- [ ] **Step 4: Wire tabs into `render()`**

Replace lines 196-291 of `render()` (from `st.subheader("Node activity")` through the end of the `else:` for "Run still in progress") with:
```python
    controller_status = _controller_status()
    tabs = st.tabs([SCENARIO_LABELS[s] for s in SCENARIOS])
    for tab, scenario in zip(tabs, SCENARIOS):
        with tab:
            _render_tab(scenario, controller_status)
```
Everything before this in `render()` (title, global Status row via `_poll_health()`) stays unchanged — matching the existing screenshot layout where Status is global and the tabs replace the single "Node activity"/"Round metrics"/"Final results" flow.

- [ ] **Step 5: Manual verification**

Run: `python -m py_compile docker/dashboard/app.py docker/dashboard/data.py`
Expected: no output (syntax is valid). Full behavioral verification happens in Task 10's end-to-end pass, since this task has no headless test harness for Streamlit itself.

- [ ] **Step 6: Commit**

```bash
git add docker/dashboard/app.py
git commit -m "feat: render 4 scenario tabs with Start/Stop, collapsed panels, and per-tab export"
```

---

### Task 10: Execution guide + manual end-to-end verification

**Files:**
- Modify: `docs/docker_mesh_execution_guide.md`

**Interfaces:** none — documentation and a manual verification checklist.

- [ ] **Step 1: Document `HOST_PROJECT_ROOT` and the controller**

Add a new section to `docs/docker_mesh_execution_guide.md` (after the existing setup steps), explaining:
- `HOST_PROJECT_ROOT` must be set to this repo's absolute path as the Docker daemon itself would resolve it (create `docker/.env` with `HOST_PROJECT_ROOT=<absolute path>` — Compose auto-loads a `.env` file next to the compose file it's run with).
- The one-time bring-up command is unchanged (`docker compose up -d --build`), run from `docker/`, with `HOST_PROJECT_ROOT` set — this creates all 6 services (`coordinator`, `node_0/1/2`, `controller`, `dashboard`) once, with `SCENARIO` defaulting to `full_run`.
- After that, all scenario switching happens through the dashboard's tabs (Start/Stop), or directly via `curl -X POST http://localhost:9100/start -d '{"scenario": "disconnection"}' -H 'Content-Type: application/json'` / `curl -X POST http://localhost:9100/stop`.
- Only one scenario ever runs at a time; starting a new one always stops whichever is running first and begins from a clean slate (existing per-container startup wipe logic, now scoped to `/energy/{scenario}/`).

- [ ] **Step 2: Manual end-to-end verification checklist**

Add this checklist to the guide, and run through it once before considering this plan done (per spec §8 — Axis C requires the scenario demo be judge-triggerable):
```markdown
## Manual verification (run once after implementing the control plane)

1. `docker compose up -d --build` (with `HOST_PROJECT_ROOT` set) — all 6 containers reach a healthy/running state.
2. Open the dashboard at http://localhost:8501 — 4 tabs are visible: Full Run, Class Addition, Disconnection, Distribution Shift.
3. Click Start on the "Full Run" tab. Confirm: the controller's state moves idle -> starting -> running (visible in the tab's state caption), node/coordinator log panels (expand them) begin showing activity, and `outputs/docker_mesh/energy/full_run/` appears on disk.
4. While Full Run is still going, click Start on the "Disconnection" tab. Confirm: Full Run's containers stop (its state caption shows the tab is no longer running), Disconnection's containers start fresh, and `outputs/docker_mesh/energy/full_run/` is left untouched (Full Run's last data is still viewable in its own tab, un-overwritten) while `outputs/docker_mesh/energy/disconnection/` starts fresh.
5. Let a run reach completion (or stop it early) and click "Download {scenario}_result.json" — confirm the downloaded file has `scenario`, `nodes`, `knowledge_transfers`, `fairness_disclosure`, and `complete` keys, and that `fairness_disclosure.per_node_scores` has one entry per node with both `mesh` and `baseline` accuracy figures.
6. Click Start on the same tab a second time after it finished — confirm the per-round results table restarts from round 0 (clean slate), not appending to the previous run's rows.
```

- [ ] **Step 3: Commit**

```bash
git add docs/docker_mesh_execution_guide.md
git commit -m "docs: document HOST_PROJECT_ROOT, controller usage, and manual verification for scenario tabs"
```

---

## Handoff to Sub-project 2

Once all 10 tasks pass, every tab runs the identical mesh+shadow loop (only the scenario *label* differs — `perturbation_applied` stays `false` and `events` stays empty for all 4 tabs, per spec §9). Sub-project 2's job is to make the 3 non-full_run tabs actually apply their perturbation at the configured round, which the design doc's §9 already scopes.
