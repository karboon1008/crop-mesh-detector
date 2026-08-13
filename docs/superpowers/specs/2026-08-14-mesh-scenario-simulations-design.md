# Mesh Scenario Simulations — Design

Status: approved for implementation planning
Date: 2026-08-14

## 1. Motivation

The grading rubric's Axis C ("scenario-driven demonstration") requires the
code to actively run at least one of: dynamically adding a new class at
runtime, simulating node disconnection/failure/reconnection, or injecting
distribution shift — and show the mesh still copes. Today the project only
addresses non-IID data via partitioning (`src/data/plantvillage.py`), which
satisfies the general "at least one challenge" bar but not this specific,
scored requirement — there is no code path that disconnects a node, adds a
class at runtime, or injects a distribution shift.

This spec covers building all three scenarios, sharing common
infrastructure, so the write-up can demonstrate maximum rubric coverage
rather than the minimum.

## 2. Goals / non-goals

Goals:
- Demonstrate, with reproducible runs and structured output, that the
  decentralised mesh (prototype + probe-logit knowledge distillation,
  `src/federated/`) continues to function and recovers faster than a
  no-exchange baseline under three disruptions: node disconnection/
  reconnection, a new class appearing at runtime, and a local distribution
  shift.
- Reuse existing infrastructure (`src.train.build_dataloaders`,
  `MeshSimulator`, `Node`, `src/evaluate.py`'s collaboration-gain metrics)
  rather than duplicating it.
- Keep the change to already-tested core files (`src/federated/node.py`,
  `src/federated/mesh.py`) minimal and backward-compatible.

Non-goals:
- No real network/transport layer. The mesh remains a synchronous,
  fully-connected, in-process simulation, as it is today — "disconnection"
  is simulated by excluding a node from the broadcast/aggregation step, not
  by modelling actual network partitions.
- No changes to the aggregation algorithms (`src/federated/aggregation.py`)
  or model architectures (`src/models/`).
- No UI/visualization — output is JSON + a human-readable `.log` file only.

## 3. Core extension: `src/federated/node.py` and `mesh.py`

- `Node.__init__` gains `active: bool = True`. Default value preserves all
  existing behaviour and existing tests unchanged.
- `MeshSimulator.run_round` changes:
  1. `local_train` — unconditional for **all** nodes, regardless of
     `active`. A disconnected node keeps training on its own local data
     while offline (it drifts, but isn't frozen).
  2. `compute_knowledge` — only called for nodes where `node.active` is
     `True`. An inactive node's payload never enters the
     `payloads` dict, so it is never broadcast to peers.
  3. Aggregation + `distill` — the per-node loop skips any node where
     `node.active` is `False` entirely (no consensus received, no
     distillation attempted). Active nodes are unaffected: they aggregate
     over whichever peers contributed a payload this round (already-existing
     "empty peer_payloads -> skip" behaviour is unchanged for the
     single-remaining-node edge case).
  4. `evaluate` — unconditional for all nodes every round, active or not.
     This is what produces the dip-and-recovery curve in the output.
- `RoundLog` gains one new field: `active_nodes: list[str] = field(default_factory=list)`.
  Additive; existing `dataclasses.asdict(round_log)` dumps in `src/train.py`
  keep working, they just get one more key.

## 4. Package layout: `src/scenarios/`

```
src/scenarios/
  __init__.py
  harness.py             # shared round-driver, event log, JSON+log writer
  disconnection.py       # python -m src.scenarios.disconnection
  class_addition.py      # python -m src.scenarios.class_addition
  distribution_shift.py  # python -m src.scenarios.distribution_shift
```

Each scenario script is independently runnable via `python -m
src.scenarios.<name> [--config path]`. Each:

1. Loads `Config`, loads the dataset, and calls the existing
   `src.train.build_dataloaders(cfg, dataset)` to get `probe_loader` and
   `node_loaders` — the exact same starting shards a normal training run
   would use. No partitioning logic is duplicated.
2. Builds **two independent node sets** from those same starting loaders,
   each with freshly-initialized models of the configured architecture
   (first entry in `models.architectures`, or `--arch` override):
   - `baseline_nodes`: plain `Node` objects, never wrapped in a
     `MeshSimulator` — each round they only call `local_train` +
     `evaluate`. This is the "went it alone" counterfactual.
   - `mesh_nodes`: wrapped in a `MeshSimulator`, run through
     `run_round` each round exactly as `src.train.run_mesh` does.
3. Applies the scenario's perturbation hook (below) to the same target
   node index in **both** sets at the same scheduled round(s), so the
   two curves are directly comparable.
4. Drives `scenarios.rounds` rounds via `harness.run_scenario`, which
   calls, each round: perturbation hook (if scheduled this round) → each
   baseline node's `local_train`+`evaluate` → `mesh.run_round` → records a
   `ScenarioRoundRecord` (round index, baseline evals, mesh evals,
   `compute_collaboration_gain(mesh_evals, baseline_evals)`, and any
   `ScenarioEvent`s fired this round).
5. Calls `harness.write_scenario_report(...)` to persist output.

### 4.1 `harness.py` contents

```python
@dataclass
class ScenarioEvent:
    round_idx: int
    event_type: str      # "disconnect" | "reconnect" | "class_added" | "shift_applied"
    node_id: str
    details: dict

@dataclass
class ScenarioRoundRecord:
    round_idx: int
    baseline_eval: dict[str, dict[str, float]]   # node_id -> {"crop_accuracy", "disease_accuracy"}
    mesh_eval: dict[str, dict[str, float]]
    collaboration_gain: dict                      # output of compute_collaboration_gain
    events: list[ScenarioEvent]

def run_scenario(
    baseline_nodes: list[Node],
    mesh: MeshSimulator,
    num_rounds: int,
    perturbation_hook: Callable[[int, list[Node], MeshSimulator], list[ScenarioEvent]],
    round_kwargs: dict,
) -> list[ScenarioRoundRecord]: ...

def write_scenario_report(
    output_dir: Path,
    scenario_name: str,
    target_node_id: str,
    config_snapshot: dict,
    records: list[ScenarioRoundRecord],
) -> None:
    """Writes outputs/scenarios/{scenario_name}.json and
    outputs/scenarios/{scenario_name}.log. The JSON includes a computed
    `recovery_round` for the target node: the first round index at/after
    the disruption's end where its mesh crop_accuracy and disease_accuracy
    are both within a fixed tolerance (5 percentage points) of its
    pre-disruption value, or null if it never recovers within the run —
    reported for both the mesh curve and the baseline curve so the
    contrast is explicit.
    """
```

`perturbation_hook(round_idx, nodes, mesh_or_none)` is called once per
round for the baseline set (`mesh_or_none=None`) and once for the mesh set
(`mesh_or_none=mesh`), returning any `ScenarioEvent`s it fired that round
(empty list on rounds where nothing happens). Each scenario module supplies
its own hook closure built from its config block.

## 5. Per-scenario perturbation hooks

### 5.1 Disconnection (`src/scenarios/disconnection.py`)

- Config: `target_node`, `disconnect_round`, `reconnect_round`.
- Hook: at `disconnect_round`, sets `nodes[target_idx].active = False`,
  emits `ScenarioEvent("disconnect", ...)`. At `reconnect_round`, sets it
  back to `True`, emits `ScenarioEvent("reconnect", ...)`. Applied to both
  baseline and mesh node sets identically — for baseline this is a
  structural no-op (baseline never used `active`), which is itself the
  point: it shows the baseline curve unaffected by connectivity while the
  mesh curve dips and recovers.

### 5.2 Class addition (`src/scenarios/class_addition.py`)

- Config: `target_node`, `source_crop`, `inject_round`, `reserve_fraction`.
- Validation at startup: `source_crop` must exist in
  `dataset.labels.crop_classes`, and under `data.manual_node_crops` must be
  assigned to a node **other than** `target_node` (raise `ValueError`
  otherwise — mirrors the existing validation style in
  `_manual_partition`).
- Before either node set is built: from the *source* node's train/test
  index lists (as produced by `build_dataloaders`), carve out
  `reserve_fraction` of the indices belonging to `source_crop` (identified
  via `dataset.labels.class_to_crop_disease`), and remove them from what
  will become the source node's loaders. This is a one-time, index-only
  computation shared by both node sets — it produces a plain list of
  reserved indices, plus the reduced source-node index lists, both reused
  when constructing the source node's loaders for `baseline_nodes` and for
  `mesh_nodes`. The underlying images are already on disk (confirmed
  present, e.g. `Tomato` already has ~16k images across 10 disease
  subfolders) and nothing is duplicated across nodes: the source node
  simply starts the run with slightly fewer of its own images, and the
  withheld slice is not visible to anyone until injected.
- Hook: at `inject_round`, rebuilds *that node set's own* target node
  `train_loader`/`test_loader` as new `DataLoader`s over `ConcatDataset`s
  of (existing target `Subset`, reserved-indices `Subset`), and reassigns
  `node.train_loader`/`node.test_loader` in place on that node object. The
  hook runs once for `baseline_nodes` and once for `mesh_nodes` (per the
  harness's per-set hook invocation, §4.1), each building its own separate
  `DataLoader` instances from the same underlying reserved-index list — no
  loader or dataset object is shared between the two sets. Emits
  `ScenarioEvent("class_added", ..., details={"crop": source_crop, "n_injected": ...})`.

### 5.3 Distribution shift (`src/scenarios/distribution_shift.py`)

- Config: `target_node`, `shift_round`, `corruption`, `severity`.
- New small wrapper in this module:
  ```python
  class CorruptedDataset(Dataset):
      """Wraps a dataset, applying a severity-scaled brightness shift +
      Gaussian blur + additive noise to the already-normalized image
      tensor, simulating a degraded camera/lighting condition."""
      def __init__(self, base: Dataset, severity: float): ...
      def __len__(self): return len(self.base)
      def __getitem__(self, idx):
          image, crop, disease = self.base[idx]
          return _corrupt(image, self.severity), crop, disease
  ```
  `severity=0.0` must be a no-op (identity), verified by a unit test.
- Hook: at `shift_round`, wraps the target node's `train_loader.dataset`
  and `test_loader.dataset` in `CorruptedDataset` and rebuilds both
  `DataLoader`s (same `batch_size`/`shuffle` as the originals). Applied to
  both baseline and mesh target nodes identically, from that round onward
  (models the farm's camera/environment degrading permanently, not a
  one-off blip). Emits `ScenarioEvent("shift_applied", ..., details={"corruption": ..., "severity": ...})`.
- No external data or assets are needed — corruption is a synthetic tensor
  transform computed on the fly at `__getitem__` time.

## 6. Config additions (`config.yaml`)

```yaml
scenarios:
  rounds: 6
  disconnection:
    target_node: "node_1"
    disconnect_round: 2
    reconnect_round: 4
  class_addition:
    target_node: "node_0"
    source_crop: "Tomato"       # must belong to a different node in manual_node_crops
    inject_round: 2
    reserve_fraction: 0.3
  distribution_shift:
    target_node: "node_2"
    shift_round: 2
    corruption: "brightness_blur_noise"
    severity: 0.5
```

Read via the existing generic `Config.get("scenarios.disconnection.target_node", ...)`
dotted-key accessor — no new config-loading code required.

## 7. Output

Per scenario run, under `outputs/scenarios/`:

- `{name}.json` — config snapshot used for the run, the full list of
  `ScenarioRoundRecord`s (baseline eval, mesh eval, collaboration gain,
  events — all per round), and a top-level summary block:
  `{"target_node": ..., "recovery_round_mesh": int|null, "recovery_round_baseline": int|null}`.
- `{name}.log` — the same events and the summary, in human-readable lines,
  written via the stdlib `logging` module as the run progresses (not
  reconstructed after the fact), so a live `tail -f` during a demo shows
  progress.

This is additive to the existing `outputs/` layout (`results_*.json`,
`round_logs_*.json`, `sustainability_report.*`) and does not change any of
it.

## 8. Testing

- Refactor: move the `synthetic_dataset` fixture (and the small node-shard
  building helper it's used with) from `tests/test_pipeline.py` into
  `tests/conftest.py`, so `tests/test_scenarios.py` can reuse it without
  duplicating the synthetic-folder-tree setup. This is the one non-scenario
  change the plan includes, and it's needed for the new tests, not a
  drive-by cleanup.
- New `tests/test_scenarios.py`:
  - `Node(active=False)` default-True sanity check, and a `MeshSimulator.run_round`
    test with one of 3 nodes inactive: assert its `node_id` is absent from
    the internal payload pool that peers aggregate over (indirectly, via
    peers' `per_node_distill_loss` still populated and the inactive node's
    own `per_node_distill_loss` entry absent), and that `RoundLog.active_nodes`
    reflects it.
  - Disconnection hook unit test: toggles at the right rounds, no-ops
    outside the window.
  - Class-addition carve-out unit test: reserved indices are disjoint from
    the remaining source-node indices, and the target node's loader length
    increases by the expected count after `inject_round`.
  - `CorruptedDataset` unit test: `severity=0.0` returns bit-identical
    tensors; `severity>0` returns a different tensor.
  - One short (2-3 round) end-to-end smoke test per scenario on the
    synthetic dataset fixture: runs the scenario script's core function,
    asserts the JSON output file exists, parses as valid JSON, contains the
    expected `event_type`s at the expected rounds, and has well-formed
    per-round eval dicts (same shape assertions as the existing
    `test_end_to_end_mesh_round_beats_no_exchange_smoke`).

## 9. Error handling

Validated once at scenario startup (not inside the round loop), raising
`ValueError` with a message in the same style as
`_manual_partition`'s existing validation:
- `target_node` must be a valid index/id among the built nodes.
- `class_addition.source_crop` must exist in the dataset and be assigned
  (in `manual_node_crops`) to a node other than `target_node`.
- All configured round indices (`disconnect_round`, `reconnect_round`,
  `inject_round`, `shift_round`) must be within `[0, scenarios.rounds)`,
  and `disconnect_round < reconnect_round`.

## 10. Risks / assumptions

- Assumes `data.non_iid_strategy` is `"manual"` (as currently configured)
  for `class_addition`, since it relies on `manual_node_crops` to find a
  crop owned by a different node. If the strategy is changed to
  `by_crop`/`by_disease`/`dirichlet`, the "different node" check needs the
  equivalent crop→node mapping recomputed from whatever `partition_nodes`
  produced — out of scope for this spec; `class_addition.py` will raise a
  clear error if `non_iid_strategy != "manual"` rather than guess.
- Three scenarios plus a shared harness is a larger surface than the
  minimum rubric bar (one scenario). If time is tight, `disconnection.py`
  alone (plus the core extension and harness) already satisfies the rubric
  requirement; `class_addition.py`/`distribution_shift.py` can be dropped
  without touching anything already built.
