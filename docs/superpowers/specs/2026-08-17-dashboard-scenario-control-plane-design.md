# Dashboard Scenario Tabs + Start/Stop Control Plane — Design

Status: draft, awaiting review
Date: 2026-08-17

## 1. Motivation

Today's Docker mesh (`docker/docker-compose.yml`: `coordinator`, `node_0/1/2`,
`dashboard`) runs exactly one topology — the plain collaborative mesh, no
scenario concept, no baseline arm. The Streamlit dashboard
(`docker/dashboard/app.py`) is explicitly read-only: it polls each
node/coordinator's `/log` endpoint and reads SQLite energy dbs, but "has no
Docker socket access... it cannot trigger a run" (per its own module
docstring). Starting/stopping the stack is a manual host-side step.

Separately, `src/scenarios/{disconnection,class_addition,distribution_shift}.py`
already implement three challenge scenarios end-to-end, but as standalone,
single-process Python scripts: each builds two independent in-memory node
sets (a `local_only` baseline set and a mesh set) and drives them side by
side, round by round, in one process. None of this touches Docker.

This spec is **Sub-project 1** of a two-part effort to bring these together:
a tabbed dashboard (baseline full run + 3 scenarios), each tab independently
start/stoppable, with mutual exclusion so only one scenario's containers ever
run at a time (avoiding the resource blow-up of running all 4 concurrently).

Sub-project 1 also picks up a requirement that surfaced while scoping this
against the competition rulebook (`docs/Official Problem Statement_0.pdf`,
Cambridge Edge AI Innovation for Sustainability Challenge 2026):

- §6.1.3 (Minimum Evidence Requirements) mandates a comparison against a
  `local_only` baseline for every run reporting collaboration gain, or an
  explicit justification for why one can't be provided. None of our
  scenarios have such a justification available.
- Axis C's core sub-criterion is `ΔG = Score(Collective) − Score(local_only)`,
  which must ship with the Collaboration-Gain Fairness Disclosure Table
  (Appendix A.1) — impossible to fill in without a real `local_only` arm.

Because of this, **every** tab (including the plain baseline full run, not
just the 3 scenario tabs) needs a `local_only` shadow-model arm computed
inside the Docker containers. Sub-project 1 therefore includes the shadow
model mechanism itself; only the scenario-specific *perturbations*
(disconnect/reconnect, class injection, distribution corruption) are deferred
to Sub-project 2.

## 2. Goals / non-goals

Goals:
- A 4-tab dashboard (Full Run, Class Addition, Disconnection, Distribution
  Shift), each with its own Start/Stop control, log panels, results table,
  fairness-disclosure table, and JSON export — all sourced from one unified
  per-scenario data schema.
- A control-plane service with the sole authority to start/stop the
  `coordinator`/`node_0/1/2` containers, enforcing "only one scenario running
  at a time" and "starting a scenario always begins from a clean slate."
- Every tab's containers compute a `local_only` (shadow) baseline arm
  alongside the mesh arm, every round, so every tab's own export is
  independently sufficient evidence for §6.1.3 and the Appendix A.1 table.
- Per-scenario data/log namespacing so a stopped tab's last results remain
  visible while a different tab is running.

Non-goals (deferred to Sub-project 2, not forgotten):
- Actually applying scenario perturbations (disconnect/reconnect a node,
  inject a class mid-run, corrupt a node's distribution) inside the
  containers. Until Sub-project 2 lands, the 3 scenario tabs run the
  identical mesh+shadow loop the Full Run tab does, just under a different
  scenario label (`perturbation_applied: false` in the export).
- Any change to `aggregation.py`, model architectures, or the existing
  single-process `src/scenarios/*.py` scripts (left untouched — they remain
  valid, independent evidence regardless of this work).
- Multi-scenario concurrent execution of any kind. The resource constraint is
  a hard design input, not a nice-to-have.

## 3. Control plane

A new `controller` service joins the compose stack (6th container). It is
the only service with the Docker socket mounted — the Streamlit `dashboard`
container stays fully read-only/no-socket, as it is today.

Endpoints:
- `POST /start {scenario}` — stops whatever is currently running (if
  anything), then brings up `coordinator` + `node_0/1/2` fresh with
  `SCENARIO=<name>` set.
- `POST /stop` — stops `coordinator` + `node_0/1/2` only; `dashboard` and
  `controller` stay up throughout.
- `GET /status` — `{running_scenario: str|None, state, started_at,
  error_detail}`.

State machine: `idle → starting → running → stopping → idle`, or `→ error` on
a Docker failure. The dashboard polls `/status` and disables Start/Stop
buttons mid-transition, so a double-click can't race two starts. "Clean
slate" needs no special wipe logic in the controller: coordinator/nodes
already wipe their own db/status files on startup, so stop-then-start is
sufficient. An `error` state requires a manual "Reset" action in the UI — no
silent auto-retry that could leave two scenarios' containers running at once.

## 4. Data model & namespacing

Each scenario gets its own subdirectory under the shared `/energy` volume:
`/energy/{scenario}/{merged.db, node_0.db, node_1.db, node_2.db,
status.json}`, plus a per-scenario log file for node/coordinator activity.
`SCENARIO` becomes a required env var on `coordinator`/`node_0/1/2`; the
existing path-building code in `node_runner.py`/coordinator `main.py` reads
it to pick the subdirectory (a targeted change, not a rewrite).

This is what lets a stopped tab's results survive while a different tab runs
— each tab only ever touches its own subdirectory — and it's exactly what
gets wiped on a clean-slate restart (only the directory for the scenario
being (re)started, never another tab's).

`round_metrics` (`src/energy/sqlite_store.py`) gains four new nullable
columns: `baseline_crop_accuracy`, `baseline_disease_accuracy`,
`baseline_energy_kwh`, `baseline_duration_s`. These are populated by each
node's new shadow-model step, reusing `upsert_row`'s existing merge pattern
rather than adding a parallel table. The existing `active` column doubles as
the disconnect flag once Sub-project 2 wires in perturbations.

Node/coordinator log lines are written to the per-scenario log file (in
addition to the existing in-memory list) as they're generated, so the
dashboard can render a tab's logs whether its containers are currently
running (tail the file) or stopped (just read it) — the log panel no longer
depends on polling a live `/log` HTTP endpoint that only exists while that
tab's containers are up.

### Shadow-model execution order (per node, per round)

Within each node container, per round, the shadow (`local_only`) step and the
mesh step run **sequentially**, shadow first, mirroring `src/scenarios/harness.py`'s
existing order (baseline arm fully computed, then the mesh arm) — not
concurrently. Two reasons:
- **Energy measurement validity**: energy is measured by wall-clock duration
  inside `ComputeEnergyTracker.track(...)` context managers. Concurrent
  execution on the same CPU-only hardcoded device would create contention
  that inflates both measured durations unpredictably, corrupting the Axis A
  energy figures.
- **Fairness-table honesty**: `local_only_budget` vs `collective_budget` in
  the Appendix A.1 table needs to state the same epochs/lr/tuning effort for
  both arms — sequential execution with the same round_kwargs is what makes
  that claim true, exactly as it already is in the validated single-process
  scripts.

The 3 node containers still run concurrently with each other (as today); only
the shadow-vs-mesh ordering *within* one node's process is sequential. All 3
nodes get their own shadow model — `local_only` means every node trained in
isolation, feeding `macro_avg_and_worst_node` in the fairness table. The
per-round cost is roughly 2x wall-clock time per node, not additional
containers.

## 5. Dashboard UI — tab layout

Each of the 4 tabs (Full Run, Class Addition, Disconnection, Distribution
Shift) has an identical layout, differing only in the scenario name it drives
and (once Sub-project 2 lands) which perturbation config applies:

1. **Header row**: Start/Stop button + live state badge (from the
   controller's `/status`) + last-run timestamp. Start is disabled while
   another tab is `running`/`starting`/`stopping`; clicking it always stops
   whatever's active first, per §3's mutual-exclusion rule.
2. **Node activity logs** (collapsed by default): the existing three-column
   node_0/node_1/node_2 + coordinator log panels, same stage-color scheme
   (`local_train` orange, `round_gather`/`distill` green/purple), reading
   from that tab's namespaced log file.
3. **Per-round results table** (collapsed by default): one row per round —
   `active_nodes`, `baseline_eval`/`mesh_eval` (crop/disease accuracy, macro +
   worst-node), `collaboration_gain`, `total_bytes_exchanged`,
   `baseline_compute_energy_kwh`/`mesh_compute_energy_kwh`,
   `communication_energy_j`, and (once Sub-project 2 lands) `events`.
4. **Collaboration-Gain Fairness Disclosure Table** (Appendix A.1, collapsed
   by default), rendered directly rather than only exported: `task_metric`,
   `node_count`, `data_split`, `non_iid_description`, `test_set_scope`,
   `local_only_budget`/`collective_budget`, `per_node_scores`,
   `macro_avg_and_worst_node`, `delta_g_formula`.
5. **Summary panel** (collapsed by default): `recovery_round_mesh` /
   `recovery_round_baseline` (null for Full Run — no disruption to recover
   from), `sustainability` (energy split, `gain_per_joule`).
6. **Export JSON button**: dumps the exact data backing sections 2–5, so what
   a judge sees on screen and what they download always match.

## 6. Export schema

One schema, reused by all 4 tabs:

```json
{
  "scenario": "full_run | class_addition | disconnection | distribution_shift",
  "target_node": null,
  "perturbation_applied": false,
  "complete": true,
  "status": { "...": "unchanged from today's final_result.json" },
  "nodes": {
    "node_0": [
      {
        "round_idx": 0,
        "crop_accuracy": 0.0, "disease_accuracy": 0.0,
        "energy_kwh": 0.0, "duration_s": 0.0, "active": true,
        "baseline_crop_accuracy": 0.0, "baseline_disease_accuracy": 0.0,
        "baseline_energy_kwh": 0.0, "baseline_duration_s": 0.0
      }
    ]
  },
  "knowledge_transfers": [
    { "round_idx": 0, "from_node": "node_0", "to_node": "node_1", "size_bytes": 0, "fetched_at": "..." }
  ],
  "rounds": [
    {
      "round_idx": 0,
      "baseline_eval": {}, "mesh_eval": {}, "collaboration_gain": {},
      "events": [],
      "active_nodes": [], "total_bytes_exchanged": 0,
      "baseline_compute_energy_kwh": 0.0, "mesh_compute_energy_kwh": 0.0,
      "communication_energy_j": 0.0, "compute_energy_method": "..."
    }
  ],
  "summary": {
    "recovery_round_mesh": null, "recovery_round_baseline": null,
    "sustainability": {
      "total_baseline_compute_energy_kwh": 0.0,
      "total_mesh_compute_energy_kwh": 0.0,
      "total_communication_energy_j": 0.0,
      "gain_per_joule": {}, "compute_energy_method": "..."
    }
  },
  "fairness_disclosure": {
    "task_metric": "...", "node_count": 3, "data_split": "...",
    "non_iid_description": "...", "test_set_scope": "...",
    "local_only_budget": {}, "collective_budget": {},
    "per_node_scores": {}, "macro_avg_and_worst_node": {},
    "delta_g_formula": "..."
  }
}
```

`status`/`nodes`/`knowledge_transfers` are today's existing dashboard fields
(extended with the four new baseline columns); `rounds`/`summary` come from
the `src/scenarios/*.py` report shape; `fairness_disclosure` is new. The
collapsed UI panels in §5 render slices of this same object — there is no
separate "display" representation to keep in sync with the export.

## 7. Error handling

- **Controller Docker failures** (container won't stop/start): state machine
  moves to `error`, surfaced in the UI with the failure reason; requires a
  manual "Reset," never a silent auto-retry that could double-run containers.
- **Node/coordinator crash mid-round**: `status.json` is left unwritten for
  that scenario; its export gets `"complete": false` rather than silently
  reporting a finished-looking but truncated run.
- **Double-click / concurrent Start**: rejected while the state machine is
  mid-transition (`starting`/`stopping`), not a race between two starts.

## 8. Testing

- Controller start/stop/mutual-exclusion state machine: unit-tested against a
  mocked Docker client, no real containers required.
- `tests/test_dashboard_data.py`-style unit tests extended for the new
  per-scenario log/results/fairness-table parsing helpers.
- Manual end-to-end pass before calling this done: `docker compose up`, click
  through all 4 tabs' Start/Stop, confirm mutual exclusion and clean-slate
  restart behave as designed — required because Axis C explicitly demands
  the scenario demonstration be "triggerable by judges following the run
  instructions."
- `docs/docker_mesh_execution_guide.md` updated for the new `controller`
  service, `SCENARIO` env var, and per-scenario directory layout.

## 9. Handoff to Sub-project 2

Sub-project 2 adds, on top of this foundation:
- New coordinator/node HTTP endpoints so the coordinator can trigger the
  actual perturbations (disconnect/reconnect a target node, inject a class,
  corrupt a distribution) at the scheduled round, driven by the existing
  `config.yaml` `scenarios:` block — mirroring the mutation logic already in
  `src/scenarios/{disconnection,class_addition,distribution_shift}.py`'s
  hooks, just invoked over HTTP instead of direct object access.
- Flipping `perturbation_applied: true` and populating `events` once a
  scenario tab's containers actually apply a perturbation.
- Extending `docker_mesh_execution_guide.md` and the Exchange Artefact Table
  narrative for the newly-real disruption behavior.
