# Mesh Model/Crop Allocation Scenarios — Comparison & Recommendation

Date: 2026-08-18

## Why this document exists

Task 9's real run against node_1 (`docs/node1_validation_tuning_and_results.md`)
surfaced a concrete data-scarcity problem: Soybean is a single "healthy"-only
class with 5,090 real photos, and because those photos are so visually
uniform, the dedup-aware split's average-hash grouping collapsed most of
them into one cluster — leaving only 132 images actually usable for
training. This is a symptom of a broader question the user raised: **when
one node's crop mix includes a data-poor crop, what's the best way to
structure models/crops across the mesh's nodes so the federated knowledge
exchange (prototypes + probe-set logits, `src/federated/mesh.py`) actually
compensates for it?**

This document compares three node/model allocation scenarios, grounded in
(a) this repo's actual mesh mechanics, and (b) external research on
federated learning under model and data heterogeneity, and recommends one.

## Full data distribution across all three nodes (not just Soybean)

Re-checked directly against both data locations on disk:
`data/PlantVillage/` (the full, unpartitioned source — 38 crop/disease
folders) and `data/docker_mesh/` (the pre-split copy `scripts/split_node_data.py`
produces for the Docker-based mesh simulation, matching `config.yaml`'s
`manual_node_crops` and `probe_set_fraction`). Soybean was the specific
case that surfaced the problem, but it's worth seeing the whole picture
before deciding how to re-partition.

**`data/docker_mesh/` — per-node image counts** (this is what each node
actually trains on today):

| Node | Crops | Classes | Total images |
|---|---|---|---|
| `node_0` | Apple, Blueberry, Cherry, Peach, Raspberry | 10 | 9,115 |
| `node_1` | Corn, Potato, Soybean, Squash, Strawberry | 11 | 13,790 |
| `node_2` | Grape, Orange, Pepper (bell), Tomato | 17 | 28,685 |
| `probe` (shared, public) | all 14 crops | 37 | 2,715 |

This confirms `config.yaml`'s own comment
(`docker_mesh.data_dir` section, `config.yaml:83-85`) that node_2 is
"volume-imbalanced by design" — it's **~3.1x node_0's size and ~2.1x
node_1's size**, entirely because Tomato alone contributes 8 disease
classes and two of its classes (`Tomato_Yellow_Leaf_Curl_Virus` at 5,357
and `Bacterial_spot` at 2,127) are among the largest classes in the whole
dataset. node_2 isn't data-poor at all — if anything it's the node the
mesh should worry about *dominating* the shared probe-set logits it
contributes, not the node needing supplementation.

**Smallest class per node** (the "weakest link" inside each node's own
shard, not just Soybean):

| Node | Smallest class | Count | Note |
|---|---|---|---|
| `node_0` | `Apple___Cedar_apple_rust` | 266 | `Peach___healthy` (338) is also thin |
| `node_1` | `Potato___healthy` | **147** | Scarcer in absolute count than Soybean's 5,090 "healthy" images — but see below, Soybean's problem isn't count |
| `node_2` | `Tomato___Tomato_mosaic_virus` | 362 | `Grape___healthy` (401) also thin |

**The Soybean issue is structural, not (only) volume.** Every other crop
in `data/PlantVillage/` has at least one disease class alongside
`healthy`. Soybean is the *only* crop in the entire 38-class dataset with
**zero disease-labeled images at all** — `Soybean___healthy` is its one
and only class. This means the disease head has literally never seen a
single diseased-soybean example, no matter how the split or dedup
threshold is tuned — augmentation and better splitting can't manufacture
a disease signal that was never collected. This is exactly why the Cotton
and Soybean 2025 dataset flagged below matters specifically for Soybean:
it's the only source found that actually adds *disease* labels for this
crop, not just more `healthy` volume.

`Potato___healthy` (147 images, node_1) is a different, more ordinary kind
of scarcity — Potato does have two disease classes (`Early_blight`,
`Late_blight`, 1,000 each) already, so this is a class-imbalance problem
within an otherwise-fine crop, not a structural gap. That's the kind of
thing the existing per-class weighting (`compute_class_weights`,
`src/validation/train_mobilenet.py`) is already designed to help with;
Soybean's zero-disease-images gap is not something any amount of
in-repo reweighting can fix.

## The three scenarios (as posed)

1. **Heterogeneous models, disjoint multi-crop nodes** — node_1 runs
   model A on crops {A,B,C}, node_2 runs model B on crops {D,E,F}, node_3
   runs model C on crops {G,H,I}. Each node picks its own architecture.
2. **One shared model, disjoint multi-crop nodes** — every node runs the
   *same* architecture, but each still covers several crops (node_1: model
   A on {A,B,C}; node_2: model A on {D,E,F}; node_3: model A on {G,H,I}).
   This is the closest to **what this repo already does** — see below.
3. **One shared model, one crop per node, curated data** — every node runs
   the same architecture, each node covers exactly one crop, and that
   crop's data is specifically curated/vetted for quality ("the data are
   better for the model selected") rather than taken as-is.

## How this repo's mesh actually works today (the constraint that decides the answer)

`src/federated/node.py` and `src/federated/mesh.py` exchange **prototypes**
(mean penultimate-layer feature vectors per class) and **probe-set logits**
(soft predictions on a shared public probe set) — never raw weights,
gradients, or images (`KnowledgePayload`, `src/federated/node.py:22-37`).
Two structural requirements fall out of this design:

- **Prototype aggregation needs a shared embedding space.** `aggregate_prototypes`
  averages (or trims/Krums) feature vectors keyed by `(task, class_id)`
  across peers (`src/federated/aggregation.py`, used at
  `src/federated/mesh.py:87-92`). `MultiTaskNet`'s `embed_dim` is
  architecture-specific — `mobilenet_v3_small`, `efficientnet_lite0`, and
  `mobilevit_xxs` each produce a different pooled-feature width
  (`src/models/factory.py:64-76`, `_probe_embedding_dim`). **Averaging
  prototype vectors of different lengths is not defined** — there's no
  projection/adapter layer anywhere in this codebase to reconcile that.
- **Logit exchange needs a shared, consistently-indexed label space.**
  `GlobalLabelMap` (`src/data/plantvillage.py:60-108`) exists specifically
  so every node's locally-discovered classes map to the *same* global
  crop/disease indices — required for `consensus_crop_logits`/
  `consensus_disease_logits` to mean the same thing to every peer
  (`src/federated/mesh.py:93-104`). This already works today because every
  node's model head is sized to the full global label space
  (`num_crop_classes`/`num_disease_classes` computed once from the whole
  dataset in `src/train.py:165-166`), regardless of which crops a given
  node actually trains on.

**Consequence: Scenario 1 (heterogeneous models) breaks prototype exchange
as this codebase is written today.** Logit exchange alone would still
work (it only depends on matching label indices, not architecture), but
prototype alignment — half of the `distill()` loss
(`src/federated/node.py:213`, `proto_weight * proto_loss`) — would need to
be disabled or replaced with an architecture-agnostic substitute for
heterogeneous nodes. This isn't a hard blocker in the literature (see
below), but it is a real implementation gap, not a config change.

## What the research says

**Model-heterogeneous federated learning is an active, real research area**
— but it needs machinery this repo doesn't have yet. The first dedicated
survey on the topic (2024) frames it as "partial" vs. "complete"
heterogeneity and surveys the actual techniques used to make it work:
knowledge distillation, mutual learning, split learning, and shared
hypernetworks that adapt to arbitrary architectures ([A Survey on
Model-heterogeneous Federated Learning](https://www.computer.org/csdl/proceedings-article/bigdata/2024/10825769/23ykcvTJiLK),
[Heterogeneous Federated Learning: State-of-the-art and Research
Challenges](https://dl.acm.org/doi/10.1145/3625558), [Federated Learning
with Heterogeneous Architectures using Graph
HyperNetworks](https://arxiv.org/pdf/2201.08459)). None of these are
"drop-in" — they're each a research-grade component this project would
need to build, not a configuration toggle. The survey also notes system-
heterogeneity research (matching architecture to a node's hardware) is
*less mature* than data-heterogeneity research — relevant here since the
project's own energy/compute-budget angle (`src/energy/tracker.py`) is
exactly a system-heterogeneity motivation.

**Agriculture-specific federated learning papers consistently handle data
heterogeneity with a *shared* model, not per-client architectures.**
PCE-FL clusters farms with similar data distributions and uses federated
knowledge distillation to cut communication cost ~91% while holding 89.1%
accuracy under extreme heterogeneity
([PCE-FL](https://doi.org/10.3390/agriengineering8050182)). AGRIFOLD
reports 97.5% accuracy on leaf disease detection with a shared-model
federated setup ([AGRIFOLD](https://www.sciencedirect.com/science/article/pii/S0957417425019906)).
A privacy-preserving crop-disease FL framework using MobileNetV2 (the same
architecture family this repo already uses) reports ~96% accuracy under
non-IID conditions with one shared model
([IJET paper](https://ijetjournal.org/federated-learning-privacy-preserving-crop-disease-detection/)).
The common thread: **the field's answer to "some clients have less/worse
data" is smarter aggregation and clustering of a shared model, not
different models per client.** That directly favors Scenario 2 or 3 over
Scenario 1.

**Supplementary data for the Soybean gap specifically exists, but needs
vetting before use:**
- The **Cotton and Soybean Plant Leaf Dataset (2025)** — 5,200 diseased +
  healthy images collected with the Central Institute for Cotton Research
  ([Journal of Phytopathology](https://onlinelibrary.wiley.com/doi/10.1111/jph.70051))
  — is the most directly relevant: it targets exactly the crop this
  project is short on *disease* images for (this repo's Soybean folder is
  "healthy"-only; PlantVillage never had soybean disease classes at all).
- **LeafNet** — 186,000+ images across 22 crop species and 97 disease
  classes, explicitly built to address PlantVillage's known limitations
  ([IEEE DataPort](https://ieee-dataport.org/documents/leafnet-large-scale-dataset-training-image-text-models-leaf-disease-identification)) —
  is a much bigger lift (different label taxonomy to reconcile) but is the
  right long-term answer if this project ever wants broader crop/disease
  coverage than PlantVillage's 14 crops.
- A 2024 review of plant-disease datasets is worth reading before adopting
  any of these — it catalogs exactly the kind of dataset-quality pitfalls
  (background bias, class imbalance, near-duplicate images) this project's
  own Task 9 run just independently rediscovered
  ([Plant disease recognition datasets in the age of deep learning: challenges
  and opportunities](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466843/)).

**Important scope note:** integrating any of these means extending
`GlobalLabelMap`/`_parse_crop_disease` to a second data source with its own
folder-naming convention, and re-running the dedup-aware split across the
combined pool — this is new work, not a config change, and should get its
own spec before implementation.

## Scenario-by-scenario verdict

| | Scenario 1 (hetero models) | Scenario 2 (shared model, multi-crop/node) | Scenario 3 (shared model, 1 crop/node, curated data) |
|---|---|---|---|
| Works with this repo's mesh *today* | ❌ prototype exchange has no cross-architecture bridge | ✅ this is essentially the current design | ✅ same mechanics as Scenario 2, finer partition |
| Matches the agriculture-FL literature's actual approach | ❌ not how any of the reviewed papers handle heterogeneity | ✅ matches PCE-FL/AGRIFOLD's shared-model pattern | ✅ matches, plus curation echoes PCE-FL's clustering-by-similar-distribution idea |
| Data-scarcity mitigation for a crop like Soybean | Unchanged — still one node stuck with 5,090 near-identical "healthy" photos | Unchanged — scarcity is still hidden inside a multi-crop node's shard | **Directly addresses it** — a dedicated node makes the scarce crop's data quality/quantity visible and fixable (targeted augmentation, external data, tighter dedup tuning) instead of averaged away inside a larger shard |
| Collaboration-gain measurement (`src/evaluate.py`) | Confounded — `macro_average`/`worst_node` assume comparable per-node models; meaningless to compare a node's baseline-vs-mesh gain if the "baseline" architecture also changes | Clean — already what `compute_collaboration_gain` was built to measure | Clean, and **more diagnostic** — one crop per node means a low `worst_node_gain` unambiguously points at *that* crop, not a shard average hiding it |
| Engineering cost | High — needs a genuinely new heterogeneous-FL component (distillation bridge or hypernetwork) with no existing precedent in this codebase | None — already implemented, already swept in `src/train.py` for all 3 architectures | Low-medium — mostly a `manual_node_crops` reconfiguration (more, smaller nodes) plus new per-crop data-curation work |
| Communication cost | Same payload shape, but the *investment* to make it meaningful (distillation infrastructure) is the real cost, not bytes | Baseline (`CommunicationCostEstimator`, `src/energy/tracker.py`) | Same shape as Scenario 2; more nodes = more pairwise broadcast (`total_bytes_exchanged` in `MeshSimulator.run_round` scales with `active_n - 1` per node), a real but bounded overhead |

## Recommendation

**Scenario 3 (shared model, finer-grained one-or-few-crop nodes, with
curated/supplemented data for the scarce crop) is the best fit for this
project, with Scenario 2 as the practical fallback if re-partitioning nodes
is out of scope right now.**

Reasoning:

1. **Scenario 1 is a research project of its own**, not a configuration
   change — it requires building genuine cross-architecture knowledge
   transfer (distillation or a hypernetwork bridge) that doesn't exist in
   this codebase, and the agriculture-FL literature reviewed above doesn't
   actually validate that approach for this problem; every real-world
   agriculture FL paper found uses a shared model. Recommend **against**
   Scenario 1 unless the project's goal changes to explicitly researching
   heterogeneous-architecture FL itself.
2. **Scenario 2 is already what this repo runs** (`config.yaml`'s
   `manual_node_crops`, one architecture swept across all 3 nodes at a
   time, `src/train.py`). It's the correct floor to keep. But it doesn't
   *fix* the Soybean-style problem — a data-poor crop's scarcity is still
   diluted inside a multi-crop shard, and `compute_collaboration_gain`'s
   per-node numbers can't isolate it from that node's other, better-off
   crops.
3. **Scenario 3 directly targets the diagnosed problem.** Giving a scarce
   crop (Soybean) its own node makes its data quality/quantity an explicit,
   measurable target: you can apply the Cotton/Soybean 2025 dataset or
   similar supplementary sources to *that node specifically*, retune the
   dedup-aware split threshold for *that node's* image homogeneity, and
   read `worst_node_gain` as a direct answer to "did the mesh actually help
   the weak node," which is precisely the metric `src/evaluate.py` was
   built to report and precisely the question the user is asking
   ("expected... the knowledge of each node improve").
4. This also matches the PCE-FL pattern most closely of anything reviewed
   — clustering nodes by data characteristics and applying federated
   distillation to the weak cluster — just expressed here as "give the
   weak crop its own node" rather than post-hoc clustering, which is a
   simpler mechanism to implement in this repo's existing manual-partition
   scheme.

## Concrete next steps (not implemented by this document)

1. Re-partition `config.yaml`'s `manual_node_crops` so Soybean gets its own
   node rather than sharing with Corn/Potato/Strawberry/Squash — it's the
   clearest case (structurally zero-disease-images, not just low volume).
   `Potato___healthy` (147 images) is a separate, ordinary class-imbalance
   case that per-class weighting already partially addresses; it doesn't
   need its own node the way Soybean does. node_2 (28,685 images, ~3.1x
   node_0) doesn't need supplementation at all — if anything, watch that
   its probe-set logit contribution doesn't dominate the shared consensus.
2. Before adopting the Cotton/Soybean 2025 dataset or LeafNet, verify
   licensing/usage terms and check class-naming compatibility with
   `_parse_crop_disease`'s `Crop___Disease` convention
   (`src/data/plantvillage.py:36-49`) — this needs its own small spec, since
   it changes `GlobalLabelMap` construction.
3. Retune the dedup-aware split's Hamming threshold specifically for
   whichever crop ends up isolated on its own node (per
   `docs/node1_validation_tuning_and_results.md`'s open finding) before
   trusting that node's reported accuracy.
4. Re-run the mesh (`src/train.py`) with the re-partitioned nodes and
   compare `worst_node_gain` before/after — that's the metric that will
   actually show whether isolating the scarce crop let the mesh's
   knowledge exchange help it more than it could when buried in a
   multi-crop shard.

## Does Scenario 3 still meet the Official Problem Statement? (Cambridge Edge AI Challenge 2026)

Checked directly against `docs/Official Problem Statement_0.pdf`. Short
answer: **yes — Scenario 3 doesn't change the mesh's compliance posture at
all (same exchange mechanism as today), and it makes the story measurably
stronger on three of the four scored axes.** Detail below, axis by axis and
rule by rule.

### Core requirements (§2, §3) — unaffected either way

Scenario 3 only changes *which crops go to which node* (`config.yaml`'s
`manual_node_crops`) — it doesn't touch the exchange mechanism
(`KnowledgePayload`: prototypes + probe-set logits, never raw images) or
the model architecture. So every compliance property the mesh already has
carries over unchanged:

- **No raw data / no data tunnelling (§5.1)**: unaffected — still just
  prototypes and probe-set logits, same as today.
- **"Should handle at least one of: Non-IID, new classes, node failure,
  continual learning, catastrophic forgetting" (§2)** — this repo already
  demonstrates **three of the five** as live, runnable simulations
  (`src/scenarios/`): `disconnection.py` (node failure/reconnection),
  `class_addition.py` (new classes injected mid-run), `distribution_shift.py`
  (corruption injected mid-run), orchestrated by `harness.py`. Non-IID is
  the partition scheme itself (`manual_node_crops`). Scenario 3 doesn't
  remove any of this — it's an orthogonal change (which crops per node),
  so all three simulations keep working exactly as before.
- **Energy/compute/communication awareness (§2, Axis A)**: unaffected —
  `src/energy/tracker.py`'s `ComputeEnergyTracker` and
  `CommunicationCostEstimator` measure whatever partition is configured.

### Where Scenario 3 measurably helps the score

**Axis C: Technical Maturity (25%) — the collaboration-gain fairness table gets *cleaner*, not harder.**
§4.3's core sub-criterion is `ΔG = Score(Collective) − Score(local_only)`,
mandatorily accompanied by Appendix A.1's Fairness Disclosure Table, which
explicitly requires `macro_avg_and_worst_node` — **"the macro average and
the worst-node score (not just the best or the overall average)."** This is
*exactly* what `src/evaluate.py`'s `compute_collaboration_gain` already
computes (`macro_average`, `worst_node`, `per_node`). Scenario 3's
finer-grained, single-scarce-crop nodes make the `worst_node_gain` number
mean something specific and defensible — "did collaboration help the
Soybean node" — rather than an average across a multi-crop shard where a
strong crop can mask a weak one. A judge checking whether the comparison
is "clearly unfair" (the Low anchor in §4.5's table) will find a cleaner
story here, not a muddier one.

**Axis B: Ease of Integration & Feasibility (25%) — simpler and more realistic, not more complex.**
Scenario 3 requires **zero new dependencies or frameworks** — it's a
`manual_node_crops` reconfiguration in `config.yaml`, nothing else. It also
makes the existing "farm" framing (`config.yaml`'s own comments — "orchard
farm", "field-crop farm", "greenhouse/market-garden farm") more literal
and more believable: a real farm growing one or two crops is a more
realistic industrial integration story than one farm growing five unrelated
crops in a single non-IID shard. This directly serves §4.2's "industrial
applicability" and "honest identification of barriers" checks.

**Axis A: Sustainability Impact (25%) — a real, disclosable trade-off, argued rigorously instead of ignored.**
More, smaller nodes means more pairwise broadcast: `MeshSimulator.run_round`
sums every active node's payload size and multiplies by `(active_n - 1)`
(`src/federated/mesh.py:75`), so splitting one multi-crop node into several
single-crop nodes **increases total communication bytes**, not decreases
them. This is exactly the kind of trade-off §4.1's A4 sub-criterion wants
disclosed ("is the case where collaboration may not be worthwhile
discussed?") — and Baseline B's `gain_per_joule`/`gain_per_byte`
(Appendix A) is the right way to show whether isolating the scarce crop is
worth the extra communication it costs. **Recommendation: don't split all
14 crops into 14 singleton nodes** — that maximises this cost for no
benefit on the 13 crops that weren't data-poor to begin with. Instead, peel
off *only* the structurally scarce crop(s) (Soybean) into their own node
and leave the rest as multi-crop nodes (i.e. Scenario 3 applied surgically,
not uniformly) — this keeps the communication-cost increase small and
easy to justify while still getting the diagnostic and data-quality benefits
for the node that actually needs them.

**Axis D: Creativity (25%) — a genuine, defensible narrative, not just "we re-split the data."**
`class_addition.py`'s existing mechanism (inject a crop's data into a node
mid-round, per `config.yaml`'s `scenarios.class_addition` block) maps
almost exactly onto "we found the Soybean node had zero disease images,
sourced the 2025 Cotton and Soybean Leaf Disease dataset, and injected it
as a live mid-simulation class-addition event." That's a concrete,
judge-triggerable demonstration tying together the real root-cause finding
from this session's own validation work (`docs/node1_validation_tuning_and_results.md`),
external data sourcing, and an already-built scenario-simulation feature —
a distinctive, well-evidenced story rather than a generic FL demo.

### What Scenario 3 does *not* solve by itself — still needs doing regardless of scenario choice

- **The Exchange Artefact Table (§5.2) doesn't exist yet** — this is
  paperwork, not code, but it's mandatory for submission (§8.1, §6.1
  item 4) under any scenario. The artifacts are already well-defined
  (`KnowledgePayload.prototypes`/`crop_logits`/`disease_logits`,
  `size_bytes()`), so this is a documentation task, not new engineering.
- **The per-sample probe-set logits need an explicit `risk_reason`.**
  Appendix 5.2's table flags "per-sample... logits" as "assumed to be at
  least medium risk unless evidence to the contrary is provided." The
  defensible argument here is that the probe set is public and identical
  across every node (`carve_public_probe_set`), so responses to it can't
  reveal any node's *private* images — but the soft-label distillation
  signal itself is a known, accepted pattern in the cited literature
  (DS-FL, already referenced in `src/federated/node.py`'s docstrings) and
  should be argued as such explicitly, not assumed self-evidently safe.
- **The dedup-threshold miscalibration must be fixed before any of these
  numbers are submitted as evidence** — per
  `docs/node1_validation_tuning_and_results.md`, today's node_1 test split
  is ~72%, not ~15%, which would fail Appendix A.1's `data_split`
  disclosure requirement on its face if submitted as-is.

### Bottom line

Scenario 3 (applied surgically — one dedicated node for Soybean
specifically, not a full one-crop-per-node atomisation) still satisfies
every requirement in the Official Problem Statement, and strengthens the
argument on Axis B (simpler, more realistic integration) and Axis C (a
cleaner, more defensible `worst_node_gain` for the exact fairness table the
rules mandate), with Axis A requiring an honest disclosure of the added
communication cost rather than a free win, and Axis D benefiting from tying
the real Soybean data-gap finding to the mesh's existing class-addition
scenario as a concrete demo.

## Per-node recommendation: which crop(s) and disease(s) to focus on for robust predictions

Checked directly against every class count in `data/PlantVillage/` (38
classes across 14 crops). "Robust" here means three things, checked per
class, not just per crop: **(1)** enough images to train on (rule of thumb
used below: ≥ ~800-1000 is robust, ~400-800 is moderate/thin, < 400 is
weak), **(2)** the crop has *both* a `healthy` class and at least one
disease class present — a crop with only one label can never support real
diagnosis, it can only confirm "this looks like crop X," and **(3)**
reasonable balance between `healthy` and disease counts within the same
crop — a large disease class next to a tiny `healthy` class (or vice versa)
biases the model toward whichever side has more examples.

This is evaluated against the **current** `manual_node_crops` assignment
(`config.yaml`), not the surgical Scenario-3 re-split — see the note at the
end for how peeling off Soybean changes node_1's picture.

### node_0 — Apple, Blueberry, Cherry, Peach, Raspberry

| Crop | Classes (count) | Verdict |
|---|---|---|
| **Apple** | healthy 1,645; Apple_scab 630; Black_rot 621; Cedar_apple_rust 275 | **Best pick for node_0.** healthy is robust, scab/black_rot are moderate-robust and give real 3-way disease differentiation. Cedar_apple_rust (275) is the one weak spot — expect it to be the least reliable Apple prediction. |
| **Cherry** | Powdery_mildew 1,052; healthy 854 | **Second-best pick.** Both classes are robust *and* well-balanced (1,052 vs. 854) — one of the cleanest 2-class problems in the whole dataset. |
| **Peach** | Bacterial_spot 2,297; healthy 360 | Usable, but lopsided — 6.4x more disease images than healthy. Expect the model to be confident on Bacterial_spot and comparatively weak/overconfident-wrong on healthy Peach unless class-weighted. |
| **Blueberry** | healthy 1,502 (only class) | **Not diagnostic.** No disease label exists for Blueberry at all — the model can only confirm "this is a blueberry leaf," never detect disease on it. |
| **Raspberry** | healthy 371 (only class) | **Weakest in this node.** Single-class *and* thin (371). Same structural gap as Blueberry, with less data on top. |

**Recommendation for node_0: lead with Apple (healthy/scab/black_rot) and Cherry (healthy/powdery_mildew) as the crops you'd actually trust a disease prediction from; treat Peach's healthy class and Apple's Cedar_apple_rust as known weak spots; don't present Blueberry or Raspberry as "disease detection" at all — they can only do species/health confirmation.**

### node_1 — Corn, Potato, Soybean, Squash, Strawberry

| Crop | Classes (count) | Verdict |
|---|---|---|
| **Corn** | Common_rust 1,192; healthy 1,162; Northern_Leaf_Blight 985; Cercospora_leaf_spot_Gray_leaf_spot 513 | **Best pick for node_1 — the most complete crop in this node.** Three classes robust (985-1,192), one moderate (513). This is the node's flagship multi-disease target. **Known weak spot, already observed in this session's own Task 9 run:** Cercospora (513, the thinnest class) was the single largest confusion source, misclassified as Northern_Leaf_Blight 13-29 times — visually similar diseases, and the thinner class loses that fight. |
| **Potato** | Early_blight 1,000; Late_blight 1,000; healthy 152 | Disease pair is robust and perfectly balanced against each other (1,000/1,000). healthy is the weak spot (152) — same lopsided pattern as Peach, worse. |
| **Strawberry** | Leaf_scorch 1,109; healthy 456 | Usable, moderately lopsided (2.4x). Workable but healthy is the thinner side. |
| **Soybean** | healthy 5,090 (only class) | **Not diagnostic, despite huge volume.** Zero disease images exist for Soybean anywhere in this dataset — see `docs/node1_validation_tuning_and_results.md`'s finding. Volume doesn't fix a structural gap; this needs external disease data (the 2025 Cotton and Soybean dataset flagged above) before it can do anything but confirm "this is soybean." |
| **Squash** | Powdery_mildew 1,835 (only class) | **The mirror-image problem of Soybean/Blueberry/Raspberry**: Squash has *only* a disease label, no `healthy` counterpart at all. The model can never learn what a healthy squash leaf looks like, so it can't actually diagnose Squash either — it can only pattern-match "does this look like the mildew photos." |

**Recommendation for node_1: Corn (healthy/Common_rust/Northern_Leaf_Blight, with Cercospora flagged as needing more data or targeted augmentation) is the flagship; Potato's disease pair is reliable but its healthy class needs supplementation; Strawberry is workable; Soybean and Squash are both structurally single-class — in opposite directions — and neither should be presented as a real disease/healthy classifier without new data.**

### node_2 — Grape, Orange, Pepper (bell), Tomato

| Crop | Classes (count) | Verdict |
|---|---|---|
| **Tomato** | Tomato_Yellow_Leaf_Curl_Virus 5,357; Bacterial_spot 2,127; Late_blight 1,909; Septoria_leaf_spot 1,771; Spider_mites 1,676; healthy 1,591; Target_Spot 1,404; Early_blight 1,000; Leaf_Mold 952; Tomato_mosaic_virus 373 | **The single strongest crop in the entire dataset.** 9 of 10 classes are robust (≥950); this is the only crop with enough breadth (9 distinct diseases) *and* depth to build a genuinely comprehensive multi-disease classifier. Only Tomato_mosaic_virus (373) is thin. |
| **Pepper, bell** | healthy 1,478; Bacterial_spot 997 | **Cleanest 2-class problem in the whole dataset** — both classes robust and reasonably balanced (1.5x, not the 4-6x skew seen elsewhere). |
| **Grape** | Esca_(Black_Measles) 1,383; Black_rot 1,180; Leaf_blight_(Isariopsis) 1,076; healthy 423 | Three disease classes robust and give good differentiation; healthy (423) is the weak spot, same lopsided pattern as Peach/Potato. |
| **Orange** | Haunglongbing_(Citrus_greening) 5,507 (only class) | **Not diagnostic, despite the largest single-class count in the whole dataset.** Structurally identical problem to Squash — disease-only, no healthy counterpart, so it can't actually differentiate healthy from diseased Orange. |

**Recommendation for node_2: Tomato is the clear flagship — the most trustworthy, best-populated multi-disease target of the whole project — with Pepper bell as a clean secondary target. Grape's three diseases are solid but its healthy class needs supplementation. Orange, despite its huge volume, has the same structural blind spot as Squash and shouldn't be presented as a real classifier without a healthy-orange dataset added.**

### If Soybean is peeled into its own node (the surgical Scenario 3 from above)

node_1 becomes Corn/Potato/Squash/Strawberry, and Corn remains its clear
flagship — nothing changes there. The new Soybean-only node's honest status
is: **it can confirm "this is a soybean leaf" and nothing else**, until the
2025 Cotton and Soybean Leaf Disease dataset (or an equivalent source) is
integrated to give it real disease classes to learn and predict from. That
node's `local_only` and `collective` scores in the fairness table
(Appendix A.1) should be reported as a **species-confirmation baseline**,
not a disease-detection result, until that data gap is closed — reporting
it as a disease classifier without a disease class in its training data
would be exactly the kind of self-reported figure Appendix A.1 exists to
catch.

### One-line summary, per node

- **node_0:** trust Apple and Cherry; Peach usable with caveats; Blueberry/Raspberry are species-confirmation only.
- **node_1:** trust Corn (flag Cercospora); Potato's diseases are solid, its healthy class isn't; Soybean and Squash are species-confirmation only, for opposite reasons.
- **node_2:** trust Tomato (the project's best crop overall) and Pepper bell; Grape's diseases are solid, its healthy class isn't; Orange is species-confirmation only despite its huge volume.

## Sources

- `docs/Official Problem Statement_0.pdf` — Cambridge Edge AI Innovation for Sustainability Challenge 2026, official rules and scoring reference
- [A Survey on Model-heterogeneous Federated Learning: Problems, Methods, and Prospects](https://www.computer.org/csdl/proceedings-article/bigdata/2024/10825769/23ykcvTJiLK)
- [Heterogeneous Federated Learning: State-of-the-art and Research Challenges (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3625558)
- [Federated Learning with Heterogeneous Architectures using Graph HyperNetworks](https://arxiv.org/pdf/2201.08459)
- [PCE-FL: A Personalized, Clustered, and Communication-Efficient Federated Learning Framework for Robust Tomato Leaf Disease Detection](https://doi.org/10.3390/agriengineering8050182)
- [AGRIFOLD: AGRIculture Federated learning for Optimized Leaf disease Detection](https://www.sciencedirect.com/science/article/pii/S0957417425019906)
- [Federated Learning-Based Privacy-Preserving Crop Disease Detection Using MobileNetV2 And Grad-CAM](https://ijetjournal.org/federated-learning-privacy-preserving-crop-disease-detection/)
- [Cotton and Soybean Plant Leaf Dataset Generation for Multiclass Disease Classification (Journal of Phytopathology, 2025)](https://onlinelibrary.wiley.com/doi/10.1111/jph.70051)
- [LeafNet: A large-scale dataset for training image-text models in leaf disease identification (IEEE DataPort)](https://ieee-dataport.org/documents/leafnet-large-scale-dataset-training-image-text-models-leaf-disease-identification)
- [Plant disease recognition datasets in the age of deep learning: challenges and opportunities](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466843/)
