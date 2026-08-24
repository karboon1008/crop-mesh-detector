# Corn Disease-Split Knowledge-Transfer Pipeline Design

Date: 2026-08-19
Status: Approved

## Purpose

Extend the node_1 validation recipe (`docs/node1_validation_tuning_and_results.md`,
`src/validation/`) into a **3-node, single-crop, disease-disjoint local
simulation**, then add a second stage that runs 1-5 rounds of the existing
federated knowledge-exchange mechanics (`src/federated/node.py`,
`src/federated/mesh.py` — prototypes + probe-set logits, unmodified) on top
of it, producing a per-round ONNX model and JSON report (energy, bytes,
accuracy) per node.

The goal is a **defensible demonstration of collaboration gain**: each node
trains on a disjoint subset of one crop's disease classes, so it starts
with zero exposure to the other nodes' diseases, and knowledge transfer is
the only way it can ever learn about them. This is deliberately the
class-disjoint non-IID label-skew setup already studied in the federated
learning literature cited in this repo (`src/evaluate.py`'s docstring — Zhu
et al., Q. Li et al.), rather than a per-crop-per-node split — see
"Rejected alternative" below for why that was tried first and dropped.

## Scope for this pass

- **Crop:** Corn only (`data/PlantVillage/Corn___*`), all 4 of its classes:
  `healthy` (1,162), `Common_rust` (1,192),
  `Cercospora_leaf_spot Gray_leaf_spot` (513), `Northern_Leaf_Blight` (985).
- **Node/disease assignment** (fixed, not config-driven round-robin):

  | Node | Disease (all images) | Healthy share |
  |---|---|---|
  | `node_0` | `Common_rust` (1,192) | disjoint 1/3 of the 1,162 `healthy` images |
  | `node_1` | `Cercospora_leaf_spot Gray_leaf_spot` (513) | disjoint 1/3 of `healthy` |
  | `node_2` | `Northern_Leaf_Blight` (985) | disjoint 1/3 of `healthy` |

  `node_1`/`node_2`'s disease pair is deliberately the exact confusion this
  session's own real run already documented: `Cercospora_leaf_spot Gray_leaf_spot`
  misclassified as `Northern_Leaf_Blight` 13-29 times when trained together
  on one node (`docs/node1_validation_tuning_and_results.md`). Splitting them
  onto separate nodes and testing whether knowledge transfer helps either
  side recognize the other's disease is a real, evidenced question, not a
  synthetic one.
- **Model:** `mobilenet_v3_small` only (matches the existing validation
  recipe).
- **Rounds:** stage 2 defaults to 2 rounds, configurable 1-5 via
  `--rounds`.

## Rejected alternative (recorded so it isn't re-litigated)

The first framing considered was one full crop per node (`node_0`=Apple,
`node_1`=Corn, `node_2`=Tomato, all of each crop's classes). This was
dropped because each node's *crop* head only ever sees one crop label as
ground truth — cross-entropy with a single ever-present class drives the
other crop logits down every step, so each node's crop head degenerates
toward an almost input-independent "it's always my crop" classifier after
15 epochs. The peer consensus a node then distills toward is built from two
*other* similarly-collapsed classifiers, which mainly transmits "the peers
agree this probably isn't my crop" rather than any real cross-crop
discriminative signal. The disease-disjoint, single-crop framing in this
spec avoids that confound entirely — there is no crop-identification task
at all (`num_crop_classes=1`), so every result is about disease
recognition, the thing this project actually cares about.

## Constraints / decisions (from user)

1. Do not modify `src/federated/node.py`, `src/federated/mesh.py`,
   `src/federated/aggregation.py`, `src/energy/tracker.py`,
   `src/data/plantvillage.py`, or `src/models/factory.py` — reuse their
   public functions/classes as-is, same rule this repo already followed for
   the node_1 validation pipeline.
2. Dataset build stays **in-memory** (filter `data/PlantVillage` sample
   indices at load time) — no physical per-node copy like
   `scripts/split_node_data.py` produces for `docker_mesh`.
3. Stage 2 rounds do **not** call `node.local_train()` separately —
   `node.distill()` alone (KD phase + local-supervised-and-prototype phase,
   already combined in one call) is the per-round update, continuing from
   the previous round's (or stage 1's, for round 1) model weights. Nothing
   resets to a fresh/random model between rounds.
4. Stage 1's output is a genuine, fully-converged local baseline per node
   (15 epochs, same recipe as node_1's validation pass) — stage 2 starts
   from there, it does not train from scratch.

## Known limitation (state this in the reported results, don't just imply it)

Prototype alignment (`_prototype_alignment_loss`, `node.py:264-282`) only
ever pulls a node's *own* local samples toward a peer prototype for a class
that node also has local samples for — the loop iterates over the node's
own `train_loader` only. Since `healthy` is the one class every node has
local samples of, prototype exchange meaningfully reinforces cross-node
agreement on `healthy` embeddings. It does **not** teach node_0 anything
about `Cercospora`'s or `Northern_Leaf_Blight`'s prototype directly, since
node_0 never has a local batch of that class to run the loss against. The
entire cross-node disease-recognition effect this experiment is measuring
comes from **probe-set logit distillation** (phase (a) of `distill()`),
not from prototypes. The stage-2 summary report must say this explicitly
rather than let the per-disease gain numbers be read as "prototype
exchange taught the model new diseases."

## Expected outcome (a prediction to check the real run against, not a target to force)

- **Round 0 (stage-1 baseline):** each node scores high (~90%+, consistent
  with `node1_validation_tuning_and_results.md`'s pattern) on its own
  disease and its own `healthy` share. On the other two nodes' diseases, a
  node's `disease_accuracy` should be near 0% — not just "low confidence,"
  actively wrong, because those output units never received a positive
  training signal.
- **Round 1:** some measurable movement is plausible — the KD signal comes
  from real Corn probe images spanning all 4 classes, so it's a
  genuinely informative channel here (unlike the rejected crop-per-node
  framing, where the peer signal itself was collapsed/uninformative).
  Expect a partial closing of the gap on the other nodes' diseases, not a
  jump to parity — one round of KD is a much weaker training signal than
  stage 1's 15 supervised epochs.
- **Round 2:** further movement in the same direction, likely smaller
  (diminishing returns is the typical distillation pattern).
- **Bytes/energy:** total bytes exchanged scale with rounds (roughly
  linear, `active_n - 1` broadcast per round, unchanged mechanic from
  `mesh.py:75`); per-round compute energy should be much smaller than
  stage 1's, since each round is one `distill()` call, not 15 epochs.

## Fairness-aligned local-only control arm (Appendix A.1 requirement)

Checked directly against `docs/Official Problem Statement_0.pdf`: Appendix
A.1's Collaboration-Gain Fairness Disclosure Table requires
`local_only_budget` and `collective_budget` to be aligned, or an explicit
`fairness_exception_reason` — "without it, G may not be treated as
high-confidence evidence." As originally spec'd, the knowledge-transfer
arm (stage 1 + N rounds of `distill()`) had strictly more total training
than the stage-1-only baseline it would be compared against — a real gap,
not just a documentation nicety, since extra gradient steps alone (with no
peer knowledge at all) could move a node's *own*-disease accuracy, and a
naive before/after comparison wouldn't isolate how much of that came from
collaboration vs. just more training time.

**Fix:** stage 2 runs a second, parallel arm per node — the **local-only
control**. Per round, alongside the knowledge-transfer node, a shadow
`Node` (same starting checkpoint, continuing from its own prior round's
state, never resetting) runs `node.local_train(epochs=training.distill_epochs_per_round,
lr=training.distill_lr)` — i.e. the same epoch count and learning rate
`distill()`'s local-supervised phase would use — but with **no exchange at
all**: no `compute_knowledge`, no aggregation, no KD phase. This aligns
the local-supervised training budget between the two arms round-for-round.
The one intentional, disclosed asymmetry is the KD phase itself (the
extra probe-set batches `distill()` runs) — that's not a fairness bug,
it's the mechanism being measured, and it's declared as such in
`fairness_exception_reason` rather than hidden.

The control arm is evaluated the same two ways as the knowledge-transfer
arm each round (local test set + cross-node union set) so both
`collaboration_gain_per_disease` (KT vs. round-0) and the Appendix-A.1
`macro_avg_and_worst_node` figures can be read against a budget-matched
local-only comparator, not just the un-extended stage-1 baseline.

This reuses `Node.local_train` (unchanged) and `training.distill_epochs_per_round`/
`training.distill_lr` (already in `config.yaml`, no new config key needed)
— only the orchestration in `run_knowledge_transfer.py` is new.

## Data layer

New module: `src/validation/corn_mesh_dataset.py`.

1. **Load once, full dataset:** `load_full_dataset(cfg.get("data.root"),
   image_size)` (reused, untouched) — this discovers all 38 PlantVillage
   classes via `ImageFolder`, same call every other pipeline stage in this
   repo already makes.
2. **Filter to Corn:** find `"Corn"`'s crop index in
   `dataset.labels.crop_classes`, then filter all sample indices to those
   whose `class_to_crop_disease` crop component matches it. This reuses the
   existing `dataset.labels` machinery exactly like `node1_dataset.py`'s
   `get_node1_indices` already does, just crop-scoped instead of
   `manual_node_crops`-scoped.
3. **Compact Corn-only label remap:** build a small `CornLabelMap`
   (`crop_classes=["Corn"]`, `disease_classes=["healthy", "Common_rust",
   "Cercospora_leaf_spot Gray_leaf_spot", "Northern_Leaf_Blight"]`, fixed
   order) and a `name_to_compact_disease_idx` dict. This is necessary
   because the full dataset's `disease_idx` values span all ~22
   PlantVillage-wide disease names — training a 4-class head against those
   raw indices would need `num_disease_classes≈22` with 18 permanently-dead
   output units, which contradicts the earlier decision to scope the head
   tightly. A thin wrapper `Dataset` (`CornDiseaseView`) wraps the filtered
   indices + the underlying `PlantVillageDataset`, and its `__getitem__`
   returns `(image, 0, compact_disease_idx)` — crop label is always `0`.
   Save this map once as `outputs/validation/corn_mesh/classes.json`
   (`{crop_classes, disease_classes, image_size}`), read by both stages.
4. **Disease assignment:** `Common_rust` → `node_0` in full, `Cercospora_...`
   → `node_1` in full, `Northern_Leaf_Blight` → `node_2` in full — no split
   needed, one node owns each disease's samples entirely.
5. **Healthy split (dedup-aware, 3-way, no cross-node duplicates):**
   - Compute a perceptual hash (reuse `hashing.average_hash`, unchanged)
     for every one of the 1,162 `healthy` images.
   - Union-find group at Hamming distance ≤ `corn_mesh.healthy_dedup_threshold`
     (config, default 5 — same default as the existing node_1 split) —
     reuses `node1_dataset.py`'s `group_duplicates`, which is already
     dataset-agnostic (takes indices + hashes + threshold, no node1-specific
     logic), so no change needed there.
   - Assign whole groups (never split a group across nodes) to the 3 node
     shares via greedy load-balancing — sort groups by size descending,
     assign each to whichever node share currently has the fewest images —
     targeting roughly 387 each. This is new code (`split_healthy_3way`),
     since `node1_dataset.py`'s existing split is 2-way (train/test), not
     3-way.
6. **Per-node train/test split:** for each node, combine its disease-class
   indices + its healthy share, then run the existing dedup-aware
   train/test split (`node1_dataset.py`'s `dedup_aware_split`, reused
   unchanged, generic over any index list) at `data.test_fraction` (0.15,
   reused from `config.yaml`).
7. **Shared public probe set:** carve from the combined Corn-only pool
   (all 4 classes, stratified) via `carve_public_probe_set` (reused,
   unchanged) at `data.probe_set_fraction` (reused). Identical across all 3
   nodes, used only for KD logits, consistent with the rest of this repo.
8. **Cross-node eval set:** the union of all 3 nodes' held-out test
   indices — built once in stage 2, used for the per-disease
   collaboration-gain measurement described above.

## Stage 1: per-node training (extends `src/validation/`)

New script: `src/validation/run_corn_pipeline.py`, generalizing
`run_pipeline.py`'s train→export→evaluate chaining (currently hardcoded to
`node_1`/`mobilenet_v3_small`) into a loop over `node_0`/`node_1`/`node_2`,
using `corn_mesh_dataset.py` in place of `node1_dataset.py`. The recipe
itself is unchanged from `train_mobilenet.py`/`export_onnx.py`/
`evaluate_onnx.py` (15 epochs, Adam + weight decay + cosine LR,
class-weighted disease loss, train-only augmentation, best-checkpoint by
held-out disease accuracy, ONNX export + PyTorch/ONNX parity check,
`report.json`) — those three modules are reused without modification, only
the dataset-scoping module changes.

```
python -m src.validation.run_corn_pipeline
python -m src.validation.run_corn_pipeline --stage train
python -m src.validation.run_corn_pipeline --stage export
python -m src.validation.run_corn_pipeline --stage evaluate
```

Output layout:

```
outputs/validation/corn_mesh/
    classes.json                              # shared CornLabelMap
    node_0_common_rust_mobilenet_v3_small/
        checkpoint.pt  training_log.json  classes.json  model.onnx  manifest.json  report.json
    node_1_cercospora_mobilenet_v3_small/     # same shape
    node_2_northern_leaf_blight_mobilenet_v3_small/   # same shape
```

Each node's own `classes.json` stores `{crop_classes, disease_classes,
image_size, train_idx, test_idx}` — same shape `run_pipeline.py` already
writes, just with the shared Corn-only label lists and this node's own
split.

## Stage 2: knowledge-transfer rounds

New script: `src/validation/run_knowledge_transfer.py`. Reuses
`src/federated/node.py` (`Node`, `KnowledgePayload`) and
`src/federated/mesh.py` (`aggregate_prototypes`/`aggregate_logits`,
existing `trimmed_mean`/`krum` methods) — no changes to either file.

```
python -m src.validation.run_knowledge_transfer --rounds 2
```

Per round:

1. Build 3 `Node` instances, loading each node's stage-1 `checkpoint.pt` as
   starting weights for round 1; round 2+ starts from the previous round's
   post-`distill()` in-memory model (never reloaded from disk mid-run).
2. Each node computes its `KnowledgePayload` (`compute_knowledge`, reused
   unchanged) from its **current** model against the shared probe loader —
   no `local_train()` call, per the constraint above.
3. For each node, aggregate the *other two* nodes' payloads
   (`aggregate_prototypes`/`aggregate_logits`, reused unchanged,
   `federated.aggregation`/`federated.trim_fraction`/`federated.krum_neighbors`
   from `config.yaml`, unchanged) into a consensus.
4. Each node calls `node.distill(...)` (reused unchanged) against that
   consensus — this single call both integrates peer knowledge (KD phase)
   and re-anchors to local ground truth (supervised + prototype phase).
5. Export each node's post-distill model to ONNX (reuses
   `export_onnx.py`'s `export_checkpoint`/`export_onnx`, unchanged).
6. Evaluate each node two ways: **local** (its own held-out test set,
   reuses `evaluate_onnx.py`'s `run_evaluation`/`build_report` unchanged)
   and **cross-node** (the union eval set from data-layer step 8, same
   `run_evaluation` call, different index list) — the cross-node report's
   `per_class_accuracy.disease` breakdown is what the per-disease gain
   table reads from.
7. Track compute energy per node per round (`ComputeEnergyTracker.track`,
   reused, one labelled block per node covering its `compute_knowledge` +
   `distill` calls) and communication bytes (`KnowledgePayload.size_bytes()`
   per node, `total_bytes_exchanged` following `mesh.py:75`'s existing
   `(active_n - 1)` broadcast-cost formula, fed into
   `CommunicationCostEstimator.estimate_all_radios`, reused unchanged).
8. Run the **local-only control** arm (previous section) for each node in
   parallel — same round index, own continuing model state, no exchange —
   and evaluate it the same two ways (local + cross-node).

Output layout:

```
outputs/validation/corn_mesh/knowledge_transfer/
    round_1/
        node_0/  model.onnx  manifest.json  report.json   # report.json: {"local": {...}, "cross_node": {...}}
        node_1/  ...
        node_2/  ...
        local_only_control/
            node_0/  checkpoint.pt  report.json           # same report.json shape, no ONNX export (not a deployment artifact)
            node_1/  ...
            node_2/  ...
        round_summary.json
    round_2/
        ... same shape ...
    knowledge_transfer_summary.json
```

### `round_summary.json` shape

```json
{
  "round": 1,
  "per_node_distill_loss": {
    "node_0": {"kd_loss": 0.0, "sup_loss": 0.0, "proto_loss": 0.0, "total_loss": 0.0}
  },
  "per_node_bytes_sent": {"node_0": 0},
  "total_bytes_exchanged": 0,
  "energy": {
    "per_node": {"node_0": {"duration_s": 0.0, "energy_kwh": 0.0, "method": "proxy_wall_power"}},
    "per_node_local_only_control": {"node_0": {"duration_s": 0.0, "energy_kwh": 0.0, "method": "proxy_wall_power"}},
    "total_compute_energy_kwh": 0.0,
    "total_duration_s": 0.0
  },
  "communication_estimate": {
    "wifi": {"radio": "wifi", "bytes": 0, "energy_kwh": 0.0, "co2_kg": 0.0}
  },
  "per_node_scores": {
    "node_0": {
      "collective": {"local": {"disease_accuracy": 0.0}, "cross_node": {"disease_accuracy": 0.0}},
      "local_only_control": {"local": {"disease_accuracy": 0.0}, "cross_node": {"disease_accuracy": 0.0}}
    }
  }
}
```

### `knowledge_transfer_summary.json` shape

```json
{
  "node_count": 3,
  "data_split": {
    "strategy": "disjoint disease-label skew within one crop (Corn)",
    "strength": "complete disjoint — each node has exactly one assigned disease class, with zero overlap with its peers'; only the shared healthy class is split (dedup-aware, ~1/3 each, no image duplicated across nodes)",
    "node_diseases": {"node_0": "Common_rust", "node_1": "Cercospora_leaf_spot Gray_leaf_spot", "node_2": "Northern_Leaf_Blight"}
  },
  "local_only_budget": {"epochs_per_round": 0, "lr": 0.0, "rounds": 2, "note": "training.distill_epochs_per_round/distill_lr, no exchange, aligned to the collective arm's local-supervised phase"},
  "collective_budget": {"distill_epochs_per_round": 0, "lr": 0.0, "kd_weight": 0.0, "proto_weight": 0.0, "rounds": 2},
  "fairness_exception_reason": "Local-supervised training budgets are aligned round-for-round between the collective and local-only-control arms (see 'Fairness-aligned local-only control arm'). The one intentional, disclosed asymmetry is the collective arm's extra KD phase over the shared probe set — that is the mechanism under test, not an unaligned budget.",
  "test_set_scope": "both — local (own node's held-out test set) and global held-out (union of all 3 nodes' held-out test sets); test samples never enter any training set",
  "rounds_run": 2,
  "cumulative_bytes_exchanged": 0,
  "cumulative_energy_kwh": 0.0,
  "delta_g_formula": "Score(collective, round_N) - Score(local_only_control, round_N), per metric per node",
  "per_node_scores": {
    "node_0": {"collective": {"disease_accuracy": 0.0}, "local_only_control": {"disease_accuracy": 0.0}}
  },
  "macro_avg_and_worst_node": {
    "macro_gain": {"disease_accuracy": 0.0},
    "worst_node_gain": {"disease_accuracy": 0.0}
  },
  "collaboration_gain_per_disease": {
    "node_0": {
      "Common_rust": {"round_0_accuracy": 0.0, "round_2_collective_accuracy": 0.0, "round_2_local_only_control_accuracy": 0.0, "gain_vs_round0": 0.0, "gain_vs_local_only_control": 0.0},
      "Cercospora_leaf_spot Gray_leaf_spot": {"round_0_accuracy": 0.0, "round_2_collective_accuracy": 0.0, "round_2_local_only_control_accuracy": 0.0, "gain_vs_round0": 0.0, "gain_vs_local_only_control": 0.0},
      "Northern_Leaf_Blight": {"round_0_accuracy": 0.0, "round_2_collective_accuracy": 0.0, "round_2_local_only_control_accuracy": 0.0, "gain_vs_round0": 0.0, "gain_vs_local_only_control": 0.0},
      "healthy": {"round_0_accuracy": 0.0, "round_2_collective_accuracy": 0.0, "round_2_local_only_control_accuracy": 0.0, "gain_vs_round0": 0.0, "gain_vs_local_only_control": 0.0}
    }
  },
  "limitation_note": "Cross-node disease-recognition gain above comes from probe-set logit distillation only, not prototype alignment — see design doc's Known Limitation section."
}
```

`macro_avg_and_worst_node` reuses `src/evaluate.py`'s existing
`compute_collaboration_gain` shape (fed the cross-node eval results,
`collective` vs. `local_only_control` rather than vs. stage-1 baseline),
following the Official Problem Statement's Appendix A.1 field names
directly (`node_count`, `data_split`, `local_only_budget`,
`collective_budget`, `test_set_scope`, `per_node_scores`,
`macro_avg_and_worst_node`, `delta_g_formula`,
`fairness_exception_reason`) so this file is directly usable as the
Collaboration-Gain Fairness Disclosure Table's source data, not just an
internal diagnostic.

## Config additions (`config.yaml`)

A new top-level section, additive only — nothing existing changes:

```yaml
corn_mesh:
  crop: "Corn"
  node_diseases:
    node_0: "Common_rust"
    node_1: "Cercospora_leaf_spot Gray_leaf_spot"
    node_2: "Northern_Leaf_Blight"
  healthy_dedup_threshold: 5
  rounds: 2
  output_dir: "outputs/validation/corn_mesh"
```

Everything else this pipeline needs (`data.image_size`, `data.seed`,
`data.test_fraction`, `data.probe_set_fraction` and its small-class
knobs, `training.lr`/`distill_lr`/`proto_weight`/`kd_weight`/
`kd_temperature`/`batch_size`, `federated.aggregation`/`trim_fraction`/
`krum_neighbors`, `energy.*`) is read from the existing sections, unchanged
— this pipeline is a new consumer of those knobs, not a reason to add
duplicates.

## Error handling

- Missing `data/PlantVillage` → reuse `load_full_dataset`'s existing
  `FileNotFoundError` (unchanged).
- `run_corn_pipeline.py` fails fast with a clear message if a later
  stage's required input (checkpoint, manifest, ONNX file) is missing —
  same pattern `run_pipeline.py` already uses.
- `run_knowledge_transfer.py` fails fast if stage 1's 3 checkpoints (and
  shared `classes.json`) aren't present yet.
- `--rounds` outside `[1, 5]` is a validation error at argument parsing,
  not a silent clamp.

## Testing / verification

No formal automated test suite is required for correctness of the
diagnostic *numbers* (this is exploratory, like the node_1 validation
pipeline before it) — but unlike that pipeline, this one has a genuine
correctness invariant worth a unit test: **the healthy 3-way split must
produce zero overlapping indices between any two nodes, and zero
duplicate-group members split across nodes.** Add this as a new test
alongside the existing `tests/` coverage. Beyond that, verification is
running both stages end to end and confirming:

- Stage 1: `report.json` per node shows high own-disease/own-healthy
  accuracy (sanity check against `node1_validation_tuning_and_results.md`'s
  numbers).
- Stage 2: each round produces all 3 nodes' `model.onnx` +
  `report.json` (local + cross_node) + `round_summary.json`, **and** the
  parallel `local_only_control/` reports; parity checks in the ONNX export
  pass; `knowledge_transfer_summary.json`'s `collaboration_gain_per_disease`
  table is well-formed and its round-0 entries per node show near-zero
  accuracy on the other two nodes' diseases (confirms the baseline
  collapse this design predicts) before checking whether the collective
  arm's round-2 numbers moved relative to the local-only-control arm's
  round-2 numbers (not just relative to round 0 — that's the
  budget-aligned comparison Appendix A.1 requires).
- Confirm the local-only-control arm actually consumed the same
  `distill_epochs_per_round`/`distill_lr` budget as the collective arm's
  local-supervised phase each round (a quick log/assert in
  `run_knowledge_transfer.py`, not just a docstring claim).

## Out of scope (this pass)

- `efficientnet_lite0` / `mobilevit_xxs` — `mobilenet_v3_small` only.
- Modifying `src/federated/node.py`'s round structure (e.g. combining the
  KD and local-supervised phases into one loss, as canonical FedMD/DS-FL
  do) — flagged as a pre-existing quirk in the current mesh core, but
  fixing it is a separate, broader change affecting `src/train.py`'s main
  sweep too, not scoped to this task.
- Physical on-disk dataset splitting (`data/corn_mesh/` folders) — stays
  in-memory, per the constraint above.
- Any change to `data.manual_node_crops`, `src/train.py`, or the main
  14-crop/3-node mesh sweep — this is a fully separate, additive
  experiment living under `src/validation/` and a new `corn_mesh:` config
  block.
- Non-Corn crops, or extending this same disease-split pattern to Apple/
  Tomato/etc. — a possible follow-up once this pipeline validates the
  approach on one crop.
