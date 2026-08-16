# Sustainability & Energy Plan: Appendix E / Axis A Coverage

This document plans how the three mesh-disruption scenarios in
[`mesh_disruption_scenarios.md`](mesh_disruption_scenarios.md) double as the
project's evidence for **Axis A: Sustainability Impact (25%)** and the
**Appendix E: Sustainability-Measurement Guidance** methodology in the
Cambridge Edge AI Innovation for Sustainability Challenge 2026 problem
statement. It is a planning document — it states what already exists, what
one additional pass of engineering would add, and the expected numbers and
justification for the research document. It does not itself change code.

## 1. Scope and baseline choice

**Baseline: B — local-only.** Axis A (§4.1) requires stating and justifying
one comparison baseline. Baseline A (centralised cloud) would require
building a new counterfactual — an estimate of cloud upload + cloud training
energy that doesn't exist yet. Baseline B is **already the control group**
every scenario script runs: each of the three scenarios (disconnection,
class addition, distribution shift) trains a baseline nodeset with *zero*
knowledge exchange, side-by-side with the mesh nodeset, under an identical
disruption schedule. Reusing it means the sustainability story rides on
evidence that already exists rather than a second experiment.

This also directly answers Appendix E.2's framing question: *does the
communication energy the mesh spends get paid back by the compute/accuracy
benefit it buys — especially under disruption, when the mesh has to work
hardest to recover?* `gain_per_joule` (Appendix A) is exactly the metric
Baseline B is built to support.

## 2. What the codebase already covers

| Piece | File | Appendix E / Axis A row it satisfies |
|---|---|---|
| `ComputeEnergyTracker` (CodeCarbon-measured, with a disclosed wall-power fallback) | [`src/energy/tracker.py`](../src/energy/tracker.py) | "Estimated energy" (E.1) — already wired into [`src/train.py`](../src/train.py) |
| `CommunicationCostEstimator` (bytes → J → gCO2e via `radio_energy_j_per_byte`) | [`src/energy/tracker.py`](../src/energy/tracker.py) | "Communication cost" (E.1); pitfall #3 in E.2 (communication energy must be counted) |
| Per-round byte-count evidence, already published | [`src/federated/mesh.py`](../src/federated/mesh.py) → [`mesh_disruption_scenarios.md`](mesh_disruption_scenarios.md) | Communication-cost dimension (Appendix A "Dimensions to present"), already measured, just not yet converted to Joules |
| K210 `.kmodel` export: `mobilenet_v3_small`, 4.88 MB, 1.55M params | [`k210_kmodel_export_findings.md`](k210_kmodel_export_findings.md) | Compute dimension (model size / param count), "recommended for all teams" regardless of baseline |
| K210 real-hardware energy-measurement method, fully specified | [`k210_riscv_c_sdk_benchmark.md`](k210_riscv_c_sdk_benchmark.md) | E.3 "High" credibility tier, once executed — not yet run |

None of this needs to be built — it is cited as existing evidence.

## 3. The one gap, stated plainly

The three scenario scripts (`src/scenarios/disconnection.py`,
`class_addition.py`, `distribution_shift.py`) do not currently call
`ComputeEnergyTracker` or `CommunicationCostEstimator` the way
`src/train.py` does. They produce accuracy and byte-count evidence but no
Joules/CO2e figure. A future engineering pass — not part of this plan —
would wire the same call pattern `train.py` already uses into
`harness.py`'s per-round loop, adding an `energy` block to each scenario's
JSON report and a "Comm. energy (J)" column to the existing per-round
markdown tables. Because this reuses an existing, tested class rather than
inventing new energy math, it is low-risk relative to the eight days
remaining before the 2026-08-24 submission deadline.

## 4. Per-scenario expected results

Computed now from the byte counts already published in
`mesh_disruption_scenarios.md`, using `config.yaml`'s
`radio_energy_j_per_byte.wifi = 0.00003` J/byte (the baseline nodeset never
exchanges anything, so its communication energy is always 0 J — that
contrast is the entire Axis A argument for Baseline B):

| Scenario | Normal comm. energy/round | During disruption | Expected mesh-vs-baseline story |
|---|---|---|---|
| Disconnection | 2,624,664 B → **78.7 J** | 878,984 B → **26.4 J** (−66%) | The mesh's communication spend is *automatically* suspended in proportion to how many peers can currently benefit from it — the byte/energy drop during the outage is itself evidence the cost tracks the opportunity, not a fixed tax |
| Class addition | 78.7 J/round throughout (unchanged — same node/round schedule) | — | **Strongest `gain_per_joule` story**: identical energy spend to the no-disruption case, but the mesh recovers 2 rounds faster than the baseline (round 2 vs. round 4) — a strictly better outcome for zero extra Joules |
| Distribution shift | 78.7 J/round throughout (unchanged) | — | **Weakest energy story, reported honestly**: since both node sets were already near-ceiling accuracy when the corruption hit, there is no visible gain to divide by the (unchanged) energy spend — this becomes the disclosed counter-case (Axis A sub-criterion A4), not a result to hide |

Compute energy is flagged as **not yet measured** — `config.yaml`'s
`energy.track_with_codecarbon` is `false` by default, so no scenario run has
produced a real compute-energy figure yet. The plan states this explicitly
as a disclosed limitation (Appendix E.2: "a missing term must be flagged as
a limitation... does not constitute non-compliance") rather than omitting
it silently.

## 5. Additional information to add beyond current evidence

- **Appendix A.1 Collaboration-Gain Fairness Disclosure Table**, filled out
  once per scenario. This is mandatory once ΔG is reported (the recovery-round
  comparisons already are a form of ΔG). The strong point to state:
  `local_only_budget` and `collective_budget` are identical in every
  scenario — same architecture, same rounds, same schedule — so there is no
  `fairness_exception_reason` to write; this is a clean, defensible
  disclosure rather than a caveat.
- **The compute cost collaboration itself adds.** Beyond local training,
  the mesh's probe-logit distillation step is an extra forward pass per
  round on top of what the baseline does. Quantifying this (small, since
  it's logits, not gradients or a second backward pass) directly answers
  Appendix E.2's warning that on-device training/update energy — "precisely
  the cost collaboration adds" — must not be ignored in favour of inference
  alone.
- **K210 measured inference energy as a separate, deployment-side term.**
  The simulation energy above is a *training-time* figure (what it costs to
  run the mesh); the K210 benchmark in `k210_riscv_c_sdk_benchmark.md` would
  give a *deployment-time* inference figure on real edge hardware. Stating
  both, and stating explicitly that they are different phases (not double
  counting one as the other), closes the exact gap Appendix E.2 warns about:
  "inference alone must not be counted" as a stand-in for training energy,
  and vice versa.
- **Memory-access energy**: explicitly flagged as unestimated (Appendix
  E.2 pitfall #2 — DRAM access energy is typically the larger share on
  memory-bound edge models) — one disclosed sentence, not a new measurement
  effort, given the time remaining.

## 6. Why this earns marks — rubric mapping

| Evidence above | Rubric item it satisfies |
|---|---|
| Baseline B, same schedule/architecture for both arms | **A1** — baseline reasonableness |
| Comm. energy measured; compute flagged as estimate-only; memory-access flagged as a limitation | **A2** — cost-inventory completeness (to the extent claimed, per E.2) |
| Class-addition's free 2-round gain vs. distribution-shift's honest null result | **A3** — soundness of the directional conclusion (shown, not just asserted, including the negative case) |
| Distribution-shift's near-ceiling caveat, memory-access gap, compute-energy gap all stated up front | **A4** — uncertainty and counter-case disclosure |
| Fairness table with no exception needed | **Appendix A.1** — mandatory ΔG disclosure |
| Reuses the same three scenario runs already required for Axis C's scenario-driven demonstration | **Axis C** bonus — one dataset now answers two axes, which is the efficient use of the time remaining before 2026-08-24 |

## 7. Status: executed

The engineering pass this plan describes in §3 has since been carried
out and the three scenarios re-run end-to-end on the real PlantVillage
dataset — the "expected results" in §4 (projected from byte counts alone,
with compute energy explicitly flagged as not yet measured) have been
superseded by actually-measured per-round communication energy **and**
compute energy for both arms of all three scenarios. See
[`docs/mesh_disruption_scenarios.md`](mesh_disruption_scenarios.md) for
the resulting per-round tables, each scenario's real `gain_per_joule`
figures, and the Overall summary table — including the distribution-shift
result, which this fresh run shows is not the "near-ceiling, no
measurable gain" case §4 anticipated (its `gain_per_joule` is comparable
to class addition's).
