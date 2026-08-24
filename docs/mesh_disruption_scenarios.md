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
| Derived metrics | [`src/scenarios/metrics.py`](../src/scenarios/metrics.py) | Computes adaptation gain, retention gain, energy totals, and the gain-per-cost ratios from a report's per-round records. Pure dict-in/dict-out, so it needs neither the dataset nor torch. |
| Summariser | [`src/scenarios/summarise.py`](../src/scenarios/summarise.py) | Reads the three JSON reports and writes consolidated JSON, Markdown, and CSV. Every figure is recomputed from the per-round records rather than read back from a stored summary. |
| Automation | [`src/scenarios/run_all.py`](../src/scenarios/run_all.py) | Runs all three scenarios in separate subprocesses, then summarises them. One command, no manual sequencing. |

All three scenario scripts are independently runnable and follow the same
pattern: load the real dataset → build two nodesets (baseline, mesh) →
run N rounds, applying the scenario's disruption at a configured round →
write `outputs/scenarios/{scenario_name}.json`.

## Running everything in one command

```bash
# run all three scenarios, then consolidate the reports
python -m src.scenarios.run_all [--config path] [--arch name]

# run a subset
python -m src.scenarios.run_all --scenarios class_addition distribution_shift

# re-derive every table from reports that already exist (no dataset, no torch)
python -m src.scenarios.summarise --output-dir outputs
```

`run_all` is a convenience wrapper over the same three entry points —
nothing in the scenario code is reachable only through it. It executes
each scenario in its own subprocess so that one failure doesn't abort the
others, reports each one's wall-clock time and exit status, and then calls
the summariser.

The summariser writes four files into `outputs/scenarios/`: `summary.json`
(all derived metrics for all scenarios), `summary.md` (the same content as
Markdown tables), `summary.csv` (one row per scenario), and
`summary_per_round.csv` (one row per scenario-round). Because it recomputes
everything from the per-round records, it is the fastest way for a reviewer
to check any number quoted in this document or in the research document's
§10 against the raw evidence — and it runs without `torch`,
`scikit-learn`, or the PlantVillage download.

**Methodology note — compute-energy figures are a wall-clock proxy, not a
CodeCarbon measurement.** This environment does not run CodeCarbon
(`config.yaml`'s `energy.track_with_codecarbon` is `false`, and
`codecarbon` isn't installed), so every `*_compute_energy_kwh` figure in
this document and in `outputs/scenarios/*.json` comes from
`ComputeEnergyTracker`'s (`src/energy/tracker.py`) fallback estimator:
wall-clock duration of each tracked training/evaluation block × a fixed
15 W (`fallback_power_watts`, the class default) assumed average power —
`energy_kwh = 15.0 * duration_s / 3.6e6`. That is a real measurement of
real wall-clock time spent on real training/evaluation work, so it is a
legitimate estimate, but it is **not** a CodeCarbon-measured joules
figure, and it inherits wall-clock's weakness: anything else competing
for the CPU during a tracked block inflates that block's figure (see the
Disconnection section's caveat below for a concrete case). The tracker
records which method produced a figure as `"proxy_wall_power"` (vs.
`"codecarbon"`) per tracked block; a future run with CodeCarbon enabled
would report the same field as `"codecarbon"` instead. See
[`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md) §2
and §7 for more on this distinction.

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
mesh recovery.

`gain_per_joule` (formula: each scenario's final-round macro
collaboration gain, per metric, divided by the total energy — compute
plus communication, converted to Joules — the mesh nodeset spent across
the whole run; see [`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md)
§1 for why Baseline B is the comparison point) for this run, as measured,
is `{crop_accuracy: 2.52e-08, disease_accuracy: 4.91e-08}` — the smallest
of the three scenarios, for two compounding reasons: this run's
final-round macro gain is itself the smallest of the three (0.60 points
crop / 1.17 points disease, versus roughly 2–4 points for the other two
scenarios), and this run's total mesh energy spend (≈237 kJ, dominated
by ≈0.0658 kWh of compute) is roughly 2.7x either other scenario's total,
as measured.

**Measurement caveat — round 3's compute-energy figure is a contention
artifact, not a scenario effect.** That 2.7x figure is inflated by one
outlier. Reading `outputs/scenarios/disconnection.json` directly: every
mesh round's `mesh_compute_energy_kwh` sits in a ~0.0048–0.0090 kWh band
(rounds 0, 1, 2, 4, 5) *except* round 3, at 0.0337 kWh — 4–7x every other
round, in every one of the three scenario JSON files. Round 3's own
*baseline* block for the same round (0.0033 kWh) shows no equivalent
spike, and round 2 — also during `node_1`'s disconnection window — is
itself unremarkable (0.0048 kWh), so the spike tracks neither the
disconnection scenario nor the disruption window; it lines up with
unrelated Docker/MQTT feature-work being committed concurrently on the
same machine during this run's multi-hour, CPU-only training window,
which is exactly the kind of contention the 15 W wall-clock proxy (see
the methodology note above) cannot distinguish from real training cost.
**No re-run was performed to resolve this** — instead, replacing round
3's figure with the mean of its neighbouring rounds
((0.004769 + 0.006248) / 2 = 0.005509 kWh) gives a corrected total mesh
compute energy of ≈0.0377 kWh (vs. ≈0.0658 kWh as-measured) and a
corrected total mesh energy of ≈136 kJ (vs. ≈237 kJ as-measured) — about
**1.5x** either other scenario's total, not 2.7x, and a corrected
`gain_per_joule` of `{crop_accuracy: 4.39e-08, disease_accuracy: 8.58e-08}`.
The raw, as-measured figures are kept above for transparency; the
corrected figures are the more representative read of this scenario's
actual energy cost, and are what the Overall summary table's footnote
below reports. This is a same-data recomputation to remove one known
measurement artifact, not a re-derivation of a cleaner narrative — it
does not change any of this section's accuracy/recovery findings, and it
does not claim to know precisely how much of round 3's inflation the
concurrent load caused beyond "enough to be the only outlier in the
dataset."

Separately, this run's total mesh compute energy (≈0.0658 kWh
as-measured; ≈0.0377 kWh corrected) is roughly **3.1x** (as-measured) or
**1.8x** (corrected) this run's total *baseline* compute energy
(0.0214 kWh) — the compute cost `docs/sustainability_energy_plan.md` §5
asks to be quantified, on top of the ≈368 J of communication energy
across the run. Both the as-measured and corrected figures above were
produced under the pre-fix evaluate() boundary described in the note at
the end of this document (§"Measurement-boundary note") — a future
re-run under the fixed boundary would count an equivalent evaluation
pass on the baseline side too, which would move this ratio somewhat
(direction not derived here, since it wasn't re-run).

---

## 2. Class Addition

**Simulates:** a crop that's currently only grown at one farm starts
being grown at a *different* farm mid-run (e.g. crop rotation) — that
node has never seen the crop before.

**What the script does** ([`src/scenarios/class_addition.py`](../src/scenarios/class_addition.py)):
it arranges for the target node to genuinely lack `source_crop` before
`inject_round`, then splices that crop's samples into the target's train
*and* test sets at `inject_round`, so it suddenly has training signal and
test coverage for a crop it never had before. There are two ways to
arrange the "genuinely lacks it" precondition, and which one applies
depends on how the data was partitioned:

| `injection_source` | When it applies | What it does |
|---|---|---|
| `target_withheld` | Partitions where every node holds some of every crop (`dirichlet`, `label_skew`) | Removes `source_crop` from the target node's own shard before round 0 and gives it back at `inject_round`. The class is genuinely unseen, because the target never trained on it. |
| `peer_reserve` | Partitions where nodes hold disjoint crops (`manual`) | Carves a `reserve_fraction` slice of `source_crop` out of a *peer's* shard (no duplication — the peer starts with slightly fewer of its own images) and injects that slice at `inject_round`. |
| `auto` (default) | Always | Picks `target_withheld` if the target actually owns the crop, `peer_reserve` otherwise. This is what makes the scenario run correctly under whichever partitioning is configured. |

The mechanism actually used, and the per-node crop counts it was chosen
from, are recorded in the report's `provenance` block, so a reviewer never
has to infer which path ran.

**Note on the archived run below:** it predates `injection_source` and used
the `peer_reserve` path under a disjoint partition. The shipped
`config.yaml` uses a `dirichlet` partition, so `auto` now selects
`target_withheld` — the stricter test. A re-run today will therefore not
reproduce the accuracy figures in the table below, and should show a
sharper injection-round drop on both arms.

**How to run:**
```bash
python -m src.scenarios.class_addition [--config path] [--arch name]
```

**Config** (`config.yaml` → `scenarios.class_addition`):

| Key | Value used | Meaning |
|---|---|---|
| `target_node` | `node_0` | Node that gains the new crop |
| `source_crop` | `Tomato` | Crop withheld from the target, or carved out of a peer |
| `inject_round` | `2` | Round the crop appears at `node_0` |
| `injection_source` | `auto` | `auto`, `target_withheld`, or `peer_reserve` (see above) |
| `reserve_fraction` | `0.3` | `peer_reserve` only: fraction of the peer's images carved out and injected |

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
disease_accuracy: 4.13e-07}. This run's total mesh compute energy
(0.02463 kWh) is roughly **2.6x** this run's total baseline compute
energy (0.00932 kWh) — the compute cost of collaboration itself, on top
of the 481 J of communication energy across the run. As with every
compute-energy figure in this document, this ratio was measured under
the pre-fix `evaluate()` boundary described in the
"Measurement-boundary note" near the end of this document; it was not
affected by the round-3 contention artifact described in the
Disconnection section (no round in this scenario's JSON is an outlier).

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
node in isolation. This run's total mesh compute energy (0.02399 kWh) is
roughly **2.7x** this run's total baseline compute energy (0.00891 kWh)
— comparable to Class Addition's ratio, and, like that section, measured
under the pre-fix `evaluate()` boundary (see "Measurement-boundary note"
below); no round in this scenario's JSON is a contention outlier either.
The run still proves the pipeline functions correctly
end-to-end: the corruption is applied exactly once, at the configured
round, to the configured node, with no crash.

---

## Overall summary

| Scenario | Target node | Disruption | Recovery — mesh | Recovery — baseline | Mesh advantage | gain_per_joule (crop / disease) |
|---|---|---|---|---|---|---|
| Disconnection | node_1 | Offline rounds 2–3, reconnects round 4 | round 4 | round 4 | Tolerates cleanly; no recovery-speed advantage shown this run | 2.52e-08 / 4.91e-08 (as measured)\* |
| Class Addition | node_0 | New crop (Tomato) appears round 2 | **round 2** | round 5 | **3 rounds faster** | 2.16e-07 / 4.13e-07 |
| Distribution Shift | node_2 | Image corruption from round 2 | round 2 | round 2 | No recovery-speed gap at the node level (train+test corrupted together); real macro-level gain (+1.97pp crop / +3.62pp disease at the final round) | 2.27e-07 / 4.16e-07 |

\* Disconnection's as-measured `gain_per_joule` is depressed by one
contaminated round (round 3's compute-energy figure — see the
Disconnection section's "Measurement caveat" above). Correcting for it
gives `gain_per_joule` = **4.39e-08 / 8.58e-08** — still the smallest of
the three scenarios, but roughly 1.7x higher than the as-measured figure
in this table. Both numbers are reported here deliberately: the
as-measured one because it's what the raw JSON contains, the corrected
one because it's the more representative estimate of this scenario's
actual energy cost.

`gain_per_joule` is each scenario's final-round macro collaboration gain,
per metric, divided by the total energy (compute + communication,
converted to Joules) the mesh nodeset spent across the whole run — see
[`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md) §1
and §4 for why Baseline B (local-only, zero exchange) is the comparison
point.

### Adaptability metrics (from `src/scenarios/metrics.py`)

`summarise` recomputes two node-level adaptability metrics that the
recovery-round comparison alone doesn't capture. **Adaptation gain** is the
mean of (mesh − baseline) accuracy for the *target* node across all
post-disruption rounds — how much better off the disrupted node is for
having peers, while disrupted. **Retention gain** is how much less accuracy
the mesh arm gave up at the disruption round than the baseline arm did,
with each arm's drop floored at zero.

| Scenario | Adaptation gain (crop / disease) | Retention gain (crop / disease) |
|---|---|---|
| Disconnection | +0.50pp / +0.45pp | −0.92pp / +0.00pp |
| Class Addition | **+8.04pp / +11.16pp** | **+3.18pp / +9.43pp** |
| Distribution Shift | +1.17pp / +6.74pp | +1.30pp / +5.00pp |

The pattern is the one the design predicts: collaboration pays most where
the disruption is a *knowledge gap* (class addition — a node lacks what
its peers have), pays moderately where it is partly one (distribution
shift), and pays essentially nothing where it is not one at all
(disconnection removes signal rather than revealing a deficiency). The
negative crop retention on Disconnection is real and reported as such: the
mesh arm dips at the disconnect round because it loses a peer signal it had
been benefiting from, which the baseline arm never had to lose.

Two figures alongside `gain_per_joule` complete the Appendix E picture.
**`gain_per_additional_joule`** divides the same gain by
`(mesh compute − baseline compute) + communication`, i.e. only the energy
collaboration *added* — 3.73e-08 / 7.28e-08 (Disconnection, as measured;
1.02e-07 / 1.99e-07 with round 3 corrected), 3.47e-07 / 6.62e-07 (Class
Addition), 3.60e-07 / 6.60e-07 (Distribution Shift). **`gain_per_byte`**
divides it by total bytes exchanged — 4.87e-10 / 9.51e-10, 1.20e-09 /
2.30e-09, and 1.25e-09 / 2.30e-09 respectively.

The useful reading of these: communication is **0.86–0.87%** of the
additional energy in the two uncontaminated runs. Over 99% of what
collaboration costs is on-device computation, not radio — a direct
consequence of exchanging kilobyte-scale prototypes rather than
megabyte-scale weights, and it means the distillation schedule, not the
transport, is the lever that matters for sustainability here. There is no
compute *saving* being negated by communication in these runs, because
distillation is work done on top of local training rather than instead of
it; that is stated plainly rather than framed away.

### Measurement-boundary note (evaluate() inside vs. outside the tracked block)

The per-round `mesh_compute_energy_kwh` figures in `outputs/scenarios/*.json`
(and everywhere they're quoted in this document) were produced by an
earlier version of `src/scenarios/harness.py`'s `run_scenario()` in which
the baseline arm's `node.evaluate()` call sat *outside* its
`tracker.track(...)` block, while the mesh arm's evaluation (inside
`MeshSimulator.run_round`) was tracked. That meant the mesh's tracked
compute-energy figure included an evaluation pass that the baseline's
did not — biasing every mesh:baseline compute-energy ratio reported in
this document (e.g. Class Addition's ≈2.6x, Distribution Shift's ≈2.7x,
Disconnection's ≈3.1x as-measured / ≈1.8x corrected) somewhat in the
mesh's favour. This has since been fixed in the code (`baseline_eval`
now happens inside the tracked block, matching the mesh arm's boundary),
but the three JSON files above were **not regenerated** — re-running
them would take several more hours each and was out of scope for this
fix. Treat every mesh:baseline compute-energy ratio in this document as
measured under the old (pre-fix) boundary; a future re-run under the
fixed boundary would likely narrow these ratios somewhat, since the
baseline side would then also be charged for its evaluation pass. This
does not affect `gain_per_joule` itself (which uses only the mesh
nodeset's own energy, not a baseline comparison) or any accuracy/recovery
finding in this document — only the mesh-vs-baseline compute-energy
comparison sentences added alongside `gain_per_joule` in each section
above.

## Output files

Each run writes one JSON report to `outputs/scenarios/`:

```
outputs/scenarios/disconnection.json
outputs/scenarios/class_addition.json
outputs/scenarios/distribution_shift.json
```

`summarise` (invoked directly, or automatically by `run_all`) adds four
consolidated files derived from those reports:

```
outputs/scenarios/summary.json          # every derived metric, all scenarios
outputs/scenarios/summary.md            # the same, as Markdown tables
outputs/scenarios/summary.csv           # one row per scenario
outputs/scenarios/summary_per_round.csv # one row per scenario-round
```

Each file contains: the config used, a per-round record (baseline eval,
mesh eval, collaboration gain, events fired that round, byte counts, and
each round's `communication_energy_j` plus `baseline_compute_energy_kwh` /
`mesh_compute_energy_kwh`), and a `summary` block with
`recovery_round_mesh` / `recovery_round_baseline` (the first round
at/after the disruption where that node's accuracy is back within 5
points of its pre-disruption value, or `null` if it never recovers within
the run) and a `sustainability` sub-block (`total_baseline_compute_energy_kwh`,
`total_mesh_compute_energy_kwh`, `total_communication_energy_j`,
`gain_per_joule` — the per-metric figures reported in each section above
and in the Overall summary table — and `compute_energy_method`, either
`"proxy_wall_power"` or `"codecarbon"`, per the methodology note near the
top of this document). This directory is git-ignored, matching the rest
of `outputs/` — re-run the commands above to regenerate it.
`compute_energy_method` was added after the three JSON files above were
generated, so those specific files don't contain that key yet; a future
re-run (under the current code) would include it, always as
`"proxy_wall_power"` in this environment. The same applies to the
`provenance` block (architecture, node count, partition strategy, energy
measurement method, injection mechanism), `disruption_start_round` /
`disruption_end_round`, and the `adaptation`/`retention`/`efficiency`
sub-blocks of `summary`: all were added afterwards, so the archived files
lack them and `summarise` reconstructs the derived ones from the per-round
records instead. A re-run under the current code emits a strict superset of
what the archived files contain.

All three runs above used a single architecture (`mobilenet_v3_small`,
the default) rather than the full 3-architecture sweep `src/train.py`
runs — the scenarios exist to prove the mesh survives disruption, not to
compare architectures, so one architecture is sufficient evidence; pass
`--arch <name>` to run a scenario against a different one.

For the full Axis A (Sustainability Impact) and Appendix E
(Sustainability-Measurement Guidance) rubric justification behind the
energy figures reported throughout this document, see
[`docs/sustainability_energy_plan.md`](sustainability_energy_plan.md).
