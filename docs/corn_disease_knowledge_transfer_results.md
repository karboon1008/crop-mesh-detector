# Corn Disease-Split Knowledge-Transfer — Real Run Results

Date: 2026-08-19
Run: `python -m src.validation.run_corn_pipeline` then `python -m src.validation.run_knowledge_transfer`, against the real `data/PlantVillage` dataset (CPU-only PyTorch, `.venv`).
Spec: `docs/superpowers/specs/2026-08-19-corn-disease-knowledge-transfer-design.md`
Outputs: `outputs/validation/corn_mesh/`

## Timing

| Stage | Wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | 12m 15s | 3 nodes × 15 epochs, train → export → evaluate |
| Stage 2 (knowledge transfer) | 8m 42s | 2 rounds × (3 nodes distill + 3 control nodes local_train + export + dual eval) |
| **Total** | **~20m 58s** | |

## Node / disease assignment

| Node | Disease (all images) | Healthy share |
|---|---|---|
| node_0 | Common_rust | disjoint ~1/3 of healthy |
| node_1 | Cercospora_leaf_spot Gray_leaf_spot | disjoint ~1/3 of healthy |
| node_2 | Northern_Leaf_Blight | disjoint ~1/3 of healthy |

Data-split strength (as disclosed in `knowledge_transfer_summary.json`): "complete disjoint — each node has exactly one assigned disease class, with zero overlap with its peers'; only the shared healthy class is split (dedup-aware, ~1/3 each, no image duplicated across nodes)."

## Stage 1 — per-node local baseline (own test set)

| Node | Test images | Crop accuracy | Disease accuracy | Per-class disease accuracy |
|---|---|---|---|---|
| node_0 | 225 | 1.000 | 1.000 | Common_rust: 1.000, healthy: 1.000 |
| node_1 | 128 | 1.000 | 1.000 | Cercospora_leaf_spot Gray_leaf_spot: 1.000, healthy: 1.000 |
| node_2 | 195 | 1.000 | 1.000 | Northern_Leaf_Blight: 1.000, healthy: 1.000 |

## Round 0 — cross-node baseline (before any knowledge transfer)

Each node's model evaluated on the union of all 3 nodes' held-out test sets, broken down by disease.

| Node | healthy | Common_rust | Cercospora_leaf_spot Gray_leaf_spot | Northern_Leaf_Blight |
|---|---|---|---|---|
| node_0 | 1.000 | **1.000** (own) | 0.000 | 0.000 |
| node_1 | 1.000 | 0.000 | **1.000** (own) | 0.000 |
| node_2 | 1.000 | 0.000 | 0.000 | **1.000** (own) |

Exactly as predicted: every node is perfect on its own disease and its healthy share, and *exactly* zero — not just low — on both other nodes' diseases.

## Collective (knowledge-transfer) arm — cross-node accuracy per round

| Node | Disease | Round 0 | Round 1 | Round 2 |
|---|---|---|---|---|
| node_0 | Common_rust (own) | 1.000 | 0.981 | 0.912 |
| node_0 | healthy | 1.000 | 1.000 | 1.000 |
| node_0 | Cercospora_leaf_spot Gray_leaf_spot | 0.000 | 0.000 | 0.000 |
| node_0 | Northern_Leaf_Blight | 0.000 | 0.000 | 0.000 |
| node_1 | Cercospora_leaf_spot Gray_leaf_spot (own) | 1.000 | 0.963 | 0.988 |
| node_1 | healthy | 1.000 | 0.993 | 1.000 |
| node_1 | Common_rust | 0.000 | 0.000 | 0.000 |
| node_1 | Northern_Leaf_Blight | 0.000 | 0.000 | 0.000 |
| node_2 | Northern_Leaf_Blight (own) | 1.000 | 0.993 | 1.000 |
| node_2 | healthy | 1.000 | 1.000 | 1.000 |
| node_2 | Common_rust | 0.000 | 0.000 | 0.000 |
| node_2 | Cercospora_leaf_spot Gray_leaf_spot | 0.000 | 0.000 | 0.000 |

**Finding:** cross-node accuracy on the *other* two nodes' diseases stayed at exactly 0.000 for every node, every round — no measurable knowledge transfer was detected. Own-disease accuracy drifted slightly (mostly downward) instead. `healthy` (the one shared, prototype-reinforced class) stayed robust at ~1.0 throughout, as the design's "Known Limitation" section anticipated.

## Final collaboration-gain table (round 0 vs. round 2, collective vs. fairness-aligned local-only control)

| Node | Disease | Round 0 | Round 2 (collective) | Round 2 (local-only control) | Gain vs. round 0 | Gain vs. control |
|---|---|---|---|---|---|---|
| node_0 | healthy | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| node_0 | Common_rust | 1.000 | 0.913 | 1.000 | **-0.088** | **-0.088** |
| node_0 | Cercospora_leaf_spot Gray_leaf_spot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_0 | Northern_Leaf_Blight | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_1 | healthy | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| node_1 | Common_rust | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_1 | Cercospora_leaf_spot Gray_leaf_spot | 1.000 | 0.988 | 1.000 | **-0.012** | **-0.012** |
| node_1 | Northern_Leaf_Blight | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_2 | healthy | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| node_2 | Common_rust | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_2 | Cercospora_leaf_spot Gray_leaf_spot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| node_2 | Northern_Leaf_Blight (own) | 1.000 | 1.000 | 0.974 | 0.000 | **+0.026** |

**Macro / worst-node gain** (`compute_collaboration_gain`, collective vs. local-only control, cross-node set):

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.000 | 0.000 |
| disease_accuracy | **-0.0067** | **-0.0018** |

Both slightly negative — on average, the collective arm did marginally *worse* than the budget-matched, no-exchange control arm.

## Communication / energy cost

| Round | Bytes exchanged | Bytes/node sent | Compute energy (cumulative, kWh) |
|---|---|---|---|
| 1 | 96,888 | 16,148 each | 0.000895 |
| 2 | 96,888 | 16,148 each | 0.001805 |
| **Total (cumulative)** | **193,776** | — | **0.001805** |

## Interpretation

- The round-0 baseline collapse is exactly as designed: complete-disjoint disease skew produces provably zero cross-node exposure before any exchange.
- **The central hypothesis — that probe-set logit distillation would teach a node about its peers' diseases — was not observed in this run.** Cross-node accuracy never moved off 0.000, for any node, any disease, either round.
- The only measurable effect of knowledge transfer was a **small, inconsistent erosion of own-disease accuracy** in 2 of 3 nodes (node_0, node_1), while node_2 held steady and even slightly outperformed its own control arm.
- `healthy` — the one class with real, shared local exposure across all nodes, reinforced by both the KD and prototype channels — stayed perfectly robust throughout, consistent with the design doc's "Known Limitation" section.
- This is a genuine null result on the paper's central claim at the default hyperparameters (`kd_weight=0.5`, `distill_epochs_per_round=1`, 2 rounds) — not a bug. Possible next steps: more rounds (up to 5, already supported via `--rounds`), a higher `kd_weight` or `distill_epochs_per_round`, or accepting and reporting the null result as-is, since the spec disclosed this risk before the run.
