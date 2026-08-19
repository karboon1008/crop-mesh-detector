# Tomato Multi-Source Dirichlet Knowledge-Transfer Pipeline Design

Date: 2026-08-19
Status: Draft (awaiting user review)

## Purpose

Retest the corn disease-split knowledge-transfer pipeline's null result
(`docs/corn_disease_knowledge_transfer_results.md` — cross-node disease
accuracy stayed at exactly 0.000 every round, collective arm scored
marginally *worse* than the local-only control) under a fundamentally
different non-IID regime. The corn pipeline
(`src/validation/corn_mesh_dataset.py`) gives each node **zero** exposure
to its peers' diseases by construction (complete disjoint split, one
disease per node) — there is no statistical signal for probe-set
logit-distillation to transfer, so a null result is close to guaranteed
regardless of whether knowledge transfer "works."

This design instead follows `docs/frai-9-1751118.pdf`'s
("Federated learning with dynamic weighted aggregation for multi-crop
disease detection") non-IID methodology: a **Dirichlet-distributed
label-skew partition** (every node gets some exposure to every class, just
skewed proportions) over a **fixed, global train/test split** carved out
before any per-node partitioning. This is the paper's data-handling
protocol only — its AdaClass adaptive aggregation algorithm is explicitly
**out of scope**; this pass changes the *split*, not the *aggregation*
mechanism, which stays the existing `src/federated/` trimmed-mean/Krum +
KD/prototype-distillation machinery, unmodified.

Scope was narrowed mid-design (per user direction) from "apply Dirichlet
splitting somewhere in the mesh" to a concrete, real target: **Tomato**,
pooling all disease classes across **four real datasets already on disk**
— PlantVillage, PlantDoc, and both PlantWild versions (v1 and v2) — merged
into one canonical label space.

## Scope for this pass

- **Crop:** Tomato only, pooled across four sources:
  - `data/PlantVillage/Tomato___*` (10 classes, ImageFolder, existing loader)
  - `data/PlantDoc/{train,test}/Tomato*` (8 classes, no existing loader —
    new; PlantDoc's own train/test split is discarded, both folders pooled
    together since this pipeline defines its own global split)
  - `data/PlantWild/plantwild/plantwild/images/tomato *` (v1, 8 classes,
    no existing loader — new)
  - `data/PlantWild/plantwild_v2/plantwild_v2/tomato *` (v2, 7 classes, no
    `healthy`/leaf equivalent, no existing loader — new)
- **Node count / Dirichlet concentration:** 3 nodes, α = 0.3 (matches this
  repo's existing `data.num_nodes` default and the paper's 3-client
  setup — stronger per-node skew, but every class still gets a meaningful
  per-node count given the smallest canonical class has ~679 pooled
  images).
- **Model:** `mobilenet_v3_small` only (matches the node_1/corn validation
  recipe).
- **Rounds:** stage 2 defaults to 2 rounds, configurable, same convention
  as the corn pipeline.

## Canonical label taxonomy

PlantVillage's 10 Tomato classes are the canonical space (it's the only
source with all of them). The other three sources map onto it; gaps are
left as zero-contribution, not forced:

| Canonical class | PlantVillage | PlantDoc | PlantWild v1 | PlantWild v2 |
|---|---|---|---|---|
| `Tomato___Bacterial_spot` | 2,127 | `Tomato leaf bacterial spot` (110) | `tomato bacterial leaf spot` (280) | `tomato bacterial leaf spot` (110) |
| `Tomato___Early_blight` | 1,000 | `Tomato Early blight leaf` (88) | `tomato early blight` (346) | `tomato early blight` (187) |
| `Tomato___healthy` | 1,591 | `Tomato leaf` (63)† | `tomato leaf` (226)† | — (no equivalent) |
| `Tomato___Late_blight` | 1,909 | `Tomato leaf late blight` (111) | `tomato late blight` (295) | `tomato late blight` (163) |
| `Tomato___Leaf_Mold` | 952 | `Tomato mold leaf` (91) | `tomato leaf mold` (239) | `tomato leaf mold` (156) |
| `Tomato___Septoria_leaf_spot` | 1,771 | `Tomato Septoria leaf spot` (151) | `tomato septoria leaf spot` (220) | `tomato septoria leaf spot` (130) |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | `Tomato two spotted spider mites leaf` (2) | — | — |
| `Tomato___Target_Spot` | 1,404 | — | — | — |
| `Tomato___Tomato_mosaic_virus` | 373 | `Tomato leaf mosaic virus` (54) | `tomato mosaic virus` (189) | `tomato mosaic virus` (63) |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | `Tomato leaf yellow virus` (76) | `tomato yellow leaf curl virus` (171) | `tomato yellow leaf curl virus` (93) |

† `healthy` mapping needs a quick visual spot-check before implementation
— confirming PlantDoc's generic `"Tomato leaf"` and PlantWild v1's
`"tomato leaf"` folders are actually healthy leaves, not just
unlabeled/mixed images. Flagged as a pre-implementation verification
step, not assumed.

Merged total ≈ 21,774 raw images before dedup consolidation (real
post-dedup count will be lower). Smallest class `Tomato_mosaic_virus`
(679) to largest `Tomato_Yellow_Leaf_Curl_Virus` (5,697) is a ~8.4:1
imbalance — worse than the paper's CCMT tomato set (~4.7:1) but improved
from PlantVillage+PlantDoc alone (~12.7:1). Carries over the paper's two
imbalance mitigations, both already present in this repo's reused
training recipe (`train_mobilenet.py`): class-weighted cross-entropy loss
and train-only augmentation.

## Data layer

New modules:

1. **`src/data/plantdoc.py`** — scans `data/PlantDoc/{train,test}`,
   filters to the 8 Tomato-prefixed folders (both `train/` and `test/`
   pooled), returns `(path, source_class_name)` pairs. No transform logic
   here — image loading/transforms stay in the merged dataset wrapper.
2. **`src/data/plantwild.py`** — scans `data/PlantWild/plantwild/plantwild/images`
   (v1) and `data/PlantWild/plantwild_v2/plantwild_v2` (v2) for
   `tomato *`-prefixed folders, tagging each returned tuple with which
   version it came from (`(path, source_class_name, "v1"|"v2")`) — kept
   only for provenance/debugging, not used as a modeling axis per the
   merge decision below.
3. **`src/validation/tomato_mesh_dataset.py`** (new, mirrors
   `corn_mesh_dataset.py`'s role):
   - **Canonical label map** (`TomatoLabelMap`: `crop_classes=["Tomato"]`,
     `disease_classes=` the 10 canonical names above, fixed order) plus
     per-source `name_to_compact_disease_idx` dicts (one for PlantVillage's
     own `Tomato___*` folder names, one for PlantDoc's, one for PlantWild's
     — same compact index space, different source-name spellings feeding
     into it). `num_crop_classes=1`, avoiding the crop-identification
     confound the corn design's "Rejected alternative" section already
     documented (single crop in scope, no crop head to collapse).
   - **Load PlantVillage's Tomato subset** via the existing
     `load_full_dataset` + crop-index filter (same mechanic
     `corn_mesh_dataset.py` already uses for Corn, reused unmodified).
   - **Load PlantDoc's and PlantWild's (v1+v2) Tomato subsets** via the two
     new loaders above, remapped through `TomatoLabelMap`.
   - **Merge into one pooled index list**, each entry
     `(source, path_or_underlying_index, canonical_disease_idx)`.
   - **Global perceptual-hash dedup grouping** over the *entire* pooled
     set (reuses `hashing.average_hash` + `node1_dataset.py`'s
     `group_duplicates`, both already dataset-agnostic) — this is the
     control point for PlantWild v1/v2 potential image overlap: if v2 is a
     filtered subset of v1's underlying photos, perceptual hashing groups
     them together regardless of the differing filenames, and the group
     is treated as one unit for the split (never split across train/test,
     never double-counted).
   - **Known risk, explicitly carried over from `node1_dataset.py`'s
     documented bug** (`docs/node1_validation_tuning_and_results.md`):
     Hamming-≤5 grouping previously collapsed 97% of one visually-uniform
     class into a single group, skewing that class's actual test fraction
     from a configured 15% to ~72%. This design adds a **max-group-size
     cap** (new parameter, `tomato_mesh.dedup_max_group_size`) — any group
     larger than the cap is rejected/split apart rather than treated as
     one unit, on top of the existing Hamming-distance threshold. The cap
     defaults to `min(25, 5% of that canonical class's pooled count)` — a
     starting heuristic, not a validated value — and **must be re-checked
     against the real merged pool's actual hash-distance/group-size
     histogram during implementation** before the resulting split is
     trusted (this is a data-dependent calibration, not something
     decidable correctly on paper alone). Implementation must log the
     resulting group-size histogram so this is verifiable, and a unit test
     should assert no single group exceeds the configured cap.
   - **Global test split**: run the (now globally-scoped, not per-node)
     dedup-aware split once over the whole pooled+grouped index, at a new
     `tomato_mesh.test_fraction` key (default 0.20, matching the paper's
     CCMT tomato ratio) — kept as its own key rather than repurposing the
     shared `data.test_fraction` (0.15), so this pipeline's global-split
     fraction can't silently change the per-node split fraction the
     general mesh and corn pipelines already rely on. This is the one and
     only test set — held out before any per-node
     partitioning happens, evaluated centrally against every node's
     checkpoint. This is a deliberate, structural change from the corn
     pipeline's *per-node* dedup-aware split, made specifically because the
     per-node approach is the one with the known miscalibration bug, and
     because the paper's own protocol is a global split.
   - **Public probe set**: carved from the remaining training pool (not
     the test set) via the same stratified-by-class algorithm as
     `carve_public_probe_set`, adapted to operate on the merged
     multi-source index list rather than requiring a `PlantVillageDataset`
     object directly (existing `carve_public_probe_set` is not modified;
     this is a new adapter function reusing its algorithm).
   - **Dirichlet partition across 3 nodes**: the remaining training pool
     (post test-split, post probe-carve) is split via the same symmetric-
     Dirichlet-per-class algorithm as `plantvillage.py`'s
     `_dirichlet_partition` (α = 0.3), reimplemented in this module against
     the merged index/label arrays rather than modifying
     `plantvillage.py` itself — following this repo's existing precedent
     (the corn design's constraint #1: don't modify shared modules the
     general mesh pipeline depends on) to avoid any regression risk to
     `src/train.py`'s main sweep.
   - **Per-node + global diagnostics logged to `classes.json`**: each
     node's exact per-class training-sample counts, dominant class %, and
     Jensen-Shannon divergence from uniform (same diagnostic the paper
     reports in its Table 4) — needed both for sanity-checking the
     partition and for the collaboration-gain reporting below.

## Stage 1: per-node training

New script: `src/validation/run_tomato_pipeline.py`, generalizing the same
train→export→evaluate chaining the corn pipeline already generalized
(`train_mobilenet.py`/`export_onnx.py`/`evaluate_onnx.py`, reused
unmodified — 15 epochs, Adam + weight decay + cosine LR, class-weighted
disease loss, train-only augmentation, best-checkpoint by held-out disease
accuracy, ONNX export + parity check, `report.json`), looped over
`node_0`/`node_1`/`node_2`, consuming `tomato_mesh_dataset.py` in place of
`corn_mesh_dataset.py`.

One disclosed deviation from the paper: the paper uses **no early
stopping** (final communication round's model is taken unconditionally).
This design keeps `train_mobilenet.py`'s existing best-checkpoint-by-
held-out-accuracy selection unchanged, evaluated against the shared global
test set, for consistency with the already-reused, untouched training
module and the corn pipeline's precedent. Since the global test set is
never used for gradient updates, this is a "peek for model selection," not
a training-data leak — but it is a real, disclosed difference from the
paper's stricter protocol, worth flagging rather than silently picking a
side.

```
python -m src.validation.run_tomato_pipeline
python -m src.validation.run_tomato_pipeline --stage train
python -m src.validation.run_tomato_pipeline --stage export
python -m src.validation.run_tomato_pipeline --stage evaluate
```

Output layout:

```
outputs/validation/tomato_mesh/
    classes.json                       # shared TomatoLabelMap, global test_idx,
                                        # per-node partition diagnostics (JS divergence, dominant class %)
    node_0_mobilenet_v3_small/
        checkpoint.pt  training_log.json  classes.json  model.onnx  manifest.json  report.json
    node_1_mobilenet_v3_small/         # same shape
    node_2_mobilenet_v3_small/         # same shape
```

Each node's own `classes.json` stores `{crop_classes, disease_classes,
image_size, train_idx}` — no per-node `test_idx`, since the test set is
shared and lives only in the top-level `classes.json`.

## Stage 2: knowledge-transfer rounds

New script: `src/validation/run_tomato_knowledge_transfer.py` (kept
separate from the existing corn-specific `run_knowledge_transfer.py`
rather than overloading it — that file already has Corn-specific
defaults/fallbacks baked in). Reuses `src/federated/node.py`
(`Node`, `KnowledgePayload`) and `src/federated/mesh.py`
(`aggregate_prototypes`/`aggregate_logits`, `trimmed_mean`/`krum`) exactly
as the corn pipeline does — **no changes to either file, and no AdaClass**
— aggregation mechanism is explicitly out of scope for this pass.

```
python -m src.validation.run_tomato_knowledge_transfer --rounds 2
```

Same fairness-aligned local-only control arm as the corn design (a shadow
`Node` per round running `local_train()` with no exchange, budget-matched
to `distill()`'s local-supervised phase), for the same Appendix A.1
reason: isolating collaboration gain from "just more training steps."

### Evaluation — structurally simpler than corn's local/cross_node split

The corn pipeline evaluated each node two ways (its own held-out test set,
and the union of all 3 nodes' held-out test sets) because there were 3
separate per-node test sets. Here there is only **one** test set, shared
by construction, so every node's checkpoint (collective and
local-only-control arms) is evaluated exactly once per round against it —
`report.json` drops the `{"local": ..., "cross_node": ...}` split entirely
in favor of one flat evaluation result.

### Per-class collaboration-gain reporting — redefined for skewed, not disjoint, data

Corn's `collaboration_gain_per_disease` table made sense because each node
owned exactly one disease. Here, every node has some exposure to every
class (Dirichlet skew, not exclusion), so "the disease this node doesn't
have" doesn't exist as a category. Instead, for each node, classes are
labeled **low-representation** if that node's own Dirichlet-assigned
training count for that class falls in the bottom 25% of its own 10-class
distribution (a per-node, reporting-only descriptive label — this does
**not** feed back into aggregation weighting, unlike the paper's AdaClass,
which is out of scope). The gain table then reports, per node per class:
`round_0_accuracy`, `round_N_collective_accuracy`,
`round_N_local_only_control_accuracy`, `gain_vs_round0`,
`gain_vs_local_only_control` — same fields as corn's table, computed
against the one shared test set's per-class slice.

`knowledge_transfer_summary.json` keeps the same Appendix A.1-aligned
top-level shape as the corn pipeline (`node_count`, `data_split`,
`local_only_budget`, `collective_budget`, `test_set_scope`,
`per_node_scores`, `macro_avg_and_worst_node`, `delta_g_formula`,
`fairness_exception_reason`), with `data_split.strategy` updated to
describe the Dirichlet partition instead of the disjoint one, e.g.:

```json
"data_split": {
  "strategy": "Dirichlet label-skew partition (alpha=0.3) over a merged, deduplicated multi-source Tomato pool (PlantVillage, PlantDoc, PlantWild v1+v2), with a global test split carved out before partitioning",
  "strength": "every node has some training exposure to all 10 canonical disease classes; skew is proportional (Dirichlet-drawn), not exclusionary",
  "sources": ["PlantVillage", "PlantDoc", "PlantWild_v1", "PlantWild_v2"],
  "dirichlet_alpha": 0.3
}
```

## Expected outcome (a prediction to check the real run against)

- **Round 0 (stage-1 baseline):** every node should score reasonably on
  its well-represented classes and lower — but not the corn pipeline's
  "actively near-0%, never-seen-a-single-example" pattern — on its
  low-representation classes, since Dirichlet skew (unlike disjoint
  exclusion) still assigns *some* samples of every class to every node
  except in the unlucky tail of the draw. If any node's low-representation
  class does come out near-zero anyway, that's a real, inspectable
  possibility with α=0.3 and should be checked against that node's logged
  per-class sample count before concluding anything about the KD
  mechanism itself.
- **Round 1-2:** because peer nodes now have non-collapsed classifiers on
  the classes a given node is weak in (unlike corn's peers, whose "other"
  disease outputs were actively suppressed by cross-entropy training),
  the probe-set KD signal has real content to transfer this time. Expect
  partial closing of the low-representation-class gap, testing directly
  whether the corn null result was caused by complete disjointness rather
  than a fundamental limitation of the KD mechanism.

## Config additions (`config.yaml`)

Additive only:

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
  dedup_max_group_size: 25         # starting heuristic (also capped at 5% of a class's pooled count) — re-tune against real data, see Known Risk above
  rounds: 2
  output_dir: "outputs/validation/tomato_mesh"
```

Reused unchanged from existing sections: `data.root` (PlantVillage base,
for the Tomato-crop filter step), `data.image_size`, `data.seed`,
`data.probe_set_fraction` + its small-class knobs, `training.*`,
`federated.*`, `energy.*`. `data.test_fraction` itself is **not** reused
by this pipeline — see `tomato_mesh.test_fraction` above.

## Error handling

- Missing `data/PlantDoc` or `data/PlantWild/{plantwild,plantwild_v2}` →
  fail fast with a clear message naming the missing source, same pattern
  as the corn pipeline's existing checks — do not silently proceed with
  fewer sources than configured.
- `dedup_max_group_size` violation (a group larger than the cap survives
  grouping) → raise, don't warn-and-continue; this is the exact failure
  mode that went undetected in the node_1 Soybean case.
- Healthy-label mapping mismatch (visual spot-check in the taxonomy
  section above turns out wrong) → this is a pre-implementation
  verification task, not a runtime error path; flagged here so it isn't
  silently skipped.

## Testing / verification

- Unit test: the global dedup grouping never produces a group that spans
  the train/test boundary (same invariant class as corn's healthy-3-way-
  split test, adapted to the 2-way global split).
- Unit test: no configured `dedup_max_group_size` is exceeded in the
  grouped output.
- Unit test: Dirichlet partition's per-node index sets are pairwise
  disjoint and their union equals the full training pool (no sample
  assigned to zero or multiple nodes).
- End-to-end: Stage 1's `report.json` per node shows plausible accuracy
  patterns per the "Expected outcome" section; Stage 2's
  `knowledge_transfer_summary.json` is well-formed and its round-0
  per-class entries are inspected against each node's logged
  low-representation-class list before drawing any conclusion about
  whether collaboration gain appeared.

## Out of scope (this pass)

- AdaClass or any other change to the aggregation algorithm — stays
  trimmed-mean/Krum + KD/prototype distillation, unmodified. This pass is
  about the *data split*, not the *aggregation weighting*.
- `efficientnet_lite0` / `mobilevit_xxs` — `mobilenet_v3_small` only.
- Physical on-disk dataset splitting — stays in-memory, same as corn.
- Applying this same Dirichlet/multi-source pattern to other crops, or
  retrofitting it onto the general 14-crop/3-node mesh sweep
  (`src/train.py`, `data.manual_node_crops`) — a possible follow-up once
  this pipeline's results are in, not bundled into this pass.
- Modifying `src/data/plantvillage.py`, `src/validation/node1_dataset.py`,
  or `src/validation/corn_mesh_dataset.py` — all reused as read-only
  references for their algorithms, not edited.
