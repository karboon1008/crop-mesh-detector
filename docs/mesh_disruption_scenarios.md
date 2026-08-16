# Mesh Disruption Scenarios: Disconnection, Class Addition, and Distribution Shift

This document explains the three scenario-driven demonstrations built under
[`src/scenarios/`](../src/scenarios/), why they exist, what each one does,
how to run them, and the results from the most recent real-dataset run.

## Why these exist

The rest of the project (`src/train.py`) only exercises **non-IID data**
as a mesh "challenge" (via `data.non_iid_strategy` partitioning). These
three scripts add the other three commonly-required challenge types —
**node disconnection/reconnection**, **a new class appearing at runtime**,
and **a local distribution shift** — and prove, on the real PlantVillage
dataset, that the decentralised mesh (prototype + probe-logit knowledge
distillation, `src/federated/`) keeps working, and recovers, under each
one. Every run compares the mesh against a **no-exchange baseline**
(the same nodes, same disruption, but never sharing knowledge) so the
mesh's benefit is a direct, like-for-like number, not just a description.

## Shared building blocks

| Piece | File | What it adds |
|---|---|---|
| `Node.active` flag | [`src/federated/node.py`](../src/federated/node.py) | Marks a node connected (`True`, default) or disconnected (`False`). |
| Round filtering | [`src/federated/mesh.py`](../src/federated/mesh.py) | `MeshSimulator.run_round` still trains and evaluates every node every round, but a disconnected node's knowledge never enters the exchange pool and it never receives/distils a peer consensus. |
| Scenario harness | [`src/scenarios/harness.py`](../src/scenarios/harness.py) | Runs a baseline node set and a mesh node set side by side under an identical disruption schedule, computes the collaboration gain each round, and writes the JSON report. |

All three scenario scripts are independently runnable and follow the same
pattern: load the real dataset → build two nodesets (baseline, mesh) →
run N rounds, applying the scenario's disruption at a configured round →
write `outputs/scenarios/{scenario_name}.json`.

---

## 1. Disconnection

**Simulates:** a farm node drops off the mesh mid-run (e.g. connectivity
loss) and later reconnects.

**What the script does** ([`src/scenarios/disconnection.py`](../src/scenarios/disconnection.py)):
at `disconnect_round`, the target node's `active` flag flips to `False` —
it keeps training on its own local data (it isn't frozen, just isolated),
but its knowledge is excluded from the broadcast/aggregation/distillation
steps. At `reconnect_round`, `active` flips back to `True` and it resumes
exchanging knowledge with its peers.

**How to run:**
```bash
python -m src.scenarios.disconnection [--config path] [--arch name]
```

**Config** (`config.yaml` → `scenarios.disconnection`):

| Key | Value used | Meaning |
|---|---|---|
| `target_node` | `node_1` | Node that disconnects |
| `disconnect_round` | `2` | Round it goes offline |
| `reconnect_round` | `4` | Round it comes back online |

**Result** (target node `node_1`, real PlantVillage run):

| Round | Active nodes | Bytes exchanged | Comm. energy (J) | Event | Mesh crop acc. | Baseline crop acc. |
|---|---|---|---|---|---|---|
| 0 | node_0, node_1, node_2 | 2,624,664 | 78.74 | — | 99.71% | 80.22% |
| 1 | node_0, node_1, node_2 | 2,624,664 | 78.74 | — | 99.61% | 98.02% |
| 2 | node_0, node_2 | 878,984 | 26.37 | **disconnect** | 98.69% | 99.47% |
| 3 | node_0, node_2 | 878,984 | 26.37 | — | 99.08% | 98.45% |
| 4 | node_0, node_1, node_2 | 2,624,664 | 78.74 | **reconnect** | 99.90% | 98.55% |
| 5 | node_0, node_1, node_2 | 2,624,664 | 78.74 | — | 99.90% | 99.13% |

**Recovery round:** mesh **4**, baseline **4**.

**Explanation:** the byte count (and the communication energy it drives,
at `config.yaml`'s `radio_energy_j_per_byte.wifi = 0.00003` J/byte) is the
hard evidence the disconnection is real — both drop by exactly the same
66% (2.62M bytes / 78.74 J down to 879K bytes / 26.37 J, only
`node_0`↔`node_2` exchange while `node_1` is out) and jump straight back
the round it reconnects. `node_1`'s own mesh-side accuracy dips slightly
at the disconnect round (99.61% → 98.69%) because it's now learning from
local data alone, then keeps climbing while still isolated (99.08% at
round 3) before jumping to 99.90% once reconnected. `node_1`'s baseline
accuracy shows no equivalent dip at round 2 (98.02% → 99.47%) — the
baseline arm never exchanges knowledge in the first place, so toggling
the mesh's `active` flag doesn't change its trajectory at all; both
"recovery rounds" land on round 4 only because the harness's recovery
check starts looking from `reconnect_round` onward, not because either
arm was struggling before then. The scenario's real evidence here is
that the mesh's communication spend is automatically suspended in
lockstep with the disruption (not a fixed tax paid regardless of who can
benefit from it) and that exclusion/re-inclusion is clean — no crash, no
corrupted state — even though this particular run doesn't show a faster
mesh recovery. This run's `summary.sustainability.gain_per_joule` =
{crop_accuracy: 2.52e-08, disease_accuracy: 4.91e-08} — the smallest of
the three scenarios, for two compounding reasons: this run's final-round
macro gain is itself the smallest of the three (0.60 points crop / 1.17
points disease, versus roughly 2–4 points for the other two scenarios),
and this run's total mesh energy spend (≈237 kJ, dominated by
≈0.0658 kWh of compute) is roughly 2.7x either other scenario's total.

---

## 2. Class Addition

**Simulates:** a crop that's currently only grown at one farm starts
being grown at a *different* farm mid-run (e.g. crop rotation) — that
node has never seen the crop before.

**What the script does** ([`src/scenarios/class_addition.py`](../src/scenarios/class_addition.py)):
before the run starts, it carves a `reserve_fraction` slice of the
`source_crop`'s samples **out of the source node's own shard** (no
duplication — the source node simply starts with slightly fewer of its
own images). At `inject_round`, that reserved slice is spliced into the
target node's train and test sets, so it suddenly has training signal
*and* test coverage for a crop it never had before.

**How to run:**
```bash
python -m src.scenarios.class_addition [--config path] [--arch name]
```

**Config** (`config.yaml` → `scenarios.class_addition`):

| Key | Value used | Meaning |
|---|---|---|
| `target_node` | `node_0` | Node that gains the new crop |
| `source_crop` | `Tomato` | Crop currently owned by `node_2` |
| `inject_round` | `2` | Round the crop appears at `node_0` |
| `reserve_fraction` | `0.3` | Fraction of `node_2`'s Tomato images carved out and injected |

**Result** (target node `node_0`, real PlantVillage run — 4,393 Tomato
samples injected at round 2):

| Round | Event | Comm. energy (J) | Mesh crop acc. | Baseline crop acc. |
|---|---|---|---|---|
| 0 | — | 78.74 | 94.22% | 70.67% |
| 1 | — | 78.74 | 98.98% | 90.64% |
| 2 | **class_added** (Tomato, 4,393 samples) | 80.95 | 99.01% | 87.46% |
| 3 | — | 80.95 | 99.90% | 91.06% |
| 4 | — | 80.95 | 99.90% | 93.19% |
| 5 | — | 80.95 | 99.80% | 94.77% |

**Recovery round:** mesh **2** (immediate), baseline **5**.

**Explanation:** this is still the clearest "mesh helps" result of the
three, and this run's exact numbers make the gap wider than before. The
recovery check requires *both* `crop_accuracy` and `disease_accuracy` to
be back within 5 points of their pre-injection (round 1) values — on the
mesh side, `node_0`'s crop accuracy barely moves (98.98% → 99.01%) and its
disease accuracy dips from 99.34% to 95.26% (a 4.08-point drop, just
inside tolerance), so the mesh clears the bar the very same round the
crop appears, because peers who already grow Tomato (`node_2`) contribute
that knowledge to the shared consensus, which `node_0` distils into its
own model immediately. The baseline (training on the new crop alone, no
peer knowledge) doesn't just drop harder on crop accuracy (90.64% →
87.46%) — its disease accuracy collapses from 95.39% to 81.88% (a
13.51-point drop) and only crosses back over the 5-point tolerance line
at round 5 (91.65%, a 3.74-point gap), after two intermediate rounds
still short of it (84.94% at round 3, 86.57% at round 4). So the mesh
recovers **3 rounds faster** than the baseline in this run — a direct,
quantified demonstration of the mesh's collaborative benefit, and a
larger one than the previous run showed.
`summary.sustainability.gain_per_joule` = {crop_accuracy: 2.16e-07,
disease_accuracy: 4.13e-07}.

---

## 3. Distribution Shift

**Simulates:** one farm's camera or lighting conditions degrade mid-run
(dust, a broken lens, a change of season) — its images become
progressively corrupted from a given round onward.

**What the script does** ([`src/scenarios/distribution_shift.py`](../src/scenarios/distribution_shift.py)):
at `shift_round`, the target node's train *and* test datasets are wrapped
in `CorruptedDataset`, which applies a severity-scaled brightness shift +
Gaussian blur + additive noise to every image from that round onward
(`severity=0` is a verified no-op; nothing is corrupted before the shift
round).

**How to run:**
```bash
python -m src.scenarios.distribution_shift [--config path] [--arch name]
```

**Config** (`config.yaml` → `scenarios.distribution_shift`):

| Key | Value used | Meaning |
|---|---|---|
| `target_node` | `node_2` | Node whose images degrade |
| `shift_round` | `2` | Round the corruption starts |
| `corruption` | `brightness_blur_noise` | Corruption type applied |
| `severity` | `0.5` | Corruption strength |

**Result** (target node `node_2`, real PlantVillage run):

| Round | Event | Comm. energy (J) | Mesh crop acc. | Baseline crop acc. |
|---|---|---|---|---|
| 0 | — | 78.74 | 99.49% | 95.30% |
| 1 | — | 78.74 | 99.74% | 98.68% |
| 2 | **shift_applied** (brightness_blur_noise, severity 0.5) | 78.74 | 99.84% | 99.02% |
| 3 | — | 78.74 | 99.56% | 98.61% |
| 4 | — | 78.74 | 99.72% | 99.30% |
| 5 | — | 78.74 | 99.70% | 97.19% |

**Recovery round:** mesh **2**, baseline **2**.

**Explanation:** `node_2`'s own local accuracy still shows no clean
before/after corruption dip in this run — both its mesh (99.74% →
99.84%) and baseline (98.68% → 99.02%) accuracy actually *rise* at the
round the corruption is applied, because `shift_round` corrupts
`node_2`'s train **and** test sets together: the model is trained and
evaluated on the same corrupted distribution from that round on, so there
is no clean holdout left to reveal a gap, and both "recovery rounds" land
on round 2 (the round the shift starts) rather than showing any lag.

That part of the old narrative for this scenario ("masked by near-ceiling
accuracy") no longer holds, though — this run's node sets were not
uniformly near-ceiling (`node_2`'s own baseline accuracy starts at only
95.30% in round 0), and the **macro** collaboration gain across all three
nodes stays real and non-trivial through to the last round: at round 5,
`mesh_macro` beats `baseline_macro` by 1.97 points on crop accuracy
(99.68% vs. 97.71%) and 3.62 points on disease accuracy (99.02% vs.
95.40%). Dividing that final-round gain by this run's total energy spend
gives `summary.sustainability.gain_per_joule` = {crop_accuracy: 2.27e-07,
disease_accuracy: 4.16e-07} — comparable to, and in fact slightly higher
than, Class Addition's (2.16e-07 / 4.13e-07), not weaker as the old
near-ceiling framing implied. What this run actually shows, honestly, is
narrower than that old framing: the *target node's own local trajectory*
is the wrong lens for seeing this disruption's effect, because the
corruption never creates a clean/dirty split for `node_2` to recover
from — the mesh's real, measurable benefit here only shows up once you
compare macro accuracy across the whole node set, not the disrupted
node in isolation. The run still proves the pipeline functions correctly
end-to-end: the corruption is applied exactly once, at the configured
round, to the configured node, with no crash.

---

## Overall summary

| Scenario | Target node | Disruption | Recovery — mesh | Recovery — baseline | Mesh advantage | gain_per_joule (crop / disease) |
|---|---|---|---|---|---|---|
| Disconnection | node_1 | Offline rounds 2–3, reconnects round 4 | round 4 | round 4 | Tolerates cleanly; no recovery-speed advantage shown this run | 2.52e-08 / 4.91e-08 |
| Class Addition | node_0 | New crop (Tomato) appears round 2 | **round 2** | round 5 | **3 rounds faster** | 2.16e-07 / 4.13e-07 |
| Distribution Shift | node_2 | Image corruption from round 2 | round 2 | round 2 | No recovery-speed gap at the node level (train+test corrupted together); real macro-level gain (+1.97pp crop / +3.62pp disease at the final round) | 2.27e-07 / 4.16e-07 |

`gain_per_joule` is each scenario's final-round macro collaboration gain
divided by the total energy (compute + communication) the mesh nodeset
spent across the whole run — see
[`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md) §1
and §4 for how this metric is defined and why Baseline B (local-only, zero
exchange) is the comparison point.

## Output files

Each run writes one JSON report to `outputs/scenarios/`:

```
outputs/scenarios/disconnection.json
outputs/scenarios/class_addition.json
outputs/scenarios/distribution_shift.json
```

Each file contains: the config used, a per-round record (baseline eval,
mesh eval, collaboration gain, events fired that round, byte counts, and
each round's `communication_energy_j` plus `baseline_compute_energy_kwh` /
`mesh_compute_energy_kwh`), and a `summary` block with
`recovery_round_mesh` / `recovery_round_baseline` (the first round
at/after the disruption where that node's accuracy is back within 5
points of its pre-disruption value, or `null` if it never recovers within
the run) and a `sustainability` sub-block (`total_baseline_compute_energy_kwh`,
`total_mesh_compute_energy_kwh`, `total_communication_energy_j`, and
`gain_per_joule` — the per-metric figures reported in each section above
and in the Overall summary table). This directory is git-ignored,
matching the rest of `outputs/` — re-run the commands above to regenerate
it.

All three runs above used a single architecture (`mobilenet_v3_small`,
the default) rather than the full 3-architecture sweep `src/train.py`
runs — the scenarios exist to prove the mesh survives disruption, not to
compare architectures, so one architecture is sufficient evidence; pass
`--arch <name>` to run a scenario against a different one.

For the full Axis A (Sustainability Impact) and Appendix E
(Sustainability-Measurement Guidance) rubric justification behind the
energy figures reported throughout this document, see
[`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md).
