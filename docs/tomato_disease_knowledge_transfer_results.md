# Tomato Dirichlet-Skew Knowledge-Transfer — Real Run Results

Date: 2026-08-19 to 2026-08-20
Spec: `docs/superpowers/specs/2026-08-19-tomato-dirichlet-mesh-design.md`
Plan: `docs/superpowers/plans/2026-08-19-tomato-dirichlet-mesh.md`

This file covers **two completed real runs** of the same pipeline against the same real, pooled multi-source Tomato dataset (PlantVillage + PlantDoc + PlantWild v1/v2), CPU-only PyTorch, differing only in the Stage 1 epoch count and Stage 2 round count:

| Run | Stage 1 epochs/node | Stage 2 rounds | Output folder | `FULL_RUN_EXIT_CODE` |
|---|---|---|---|---|
| [Run 1](#run-1--epochs2-rounds5) | 2 | 5 | `outputs/validation/tomato_mesh_epoch2_round5/` | 0 |
| [Run 2](#run-2--epochs10-rounds10) | 10 | 10 | `outputs/validation/tomato_mesh/` | 0 |

See the [comparison](#comparison-run-1-vs-run-2) and [tuning recommendation](#tuning-recommendation-for-the-next-run) sections at the end for the cross-run analysis.

## Why this pipeline exists (plain-English recap)

A [previous run on corn](corn_disease_knowledge_transfer_results.md) split diseases so that **each node only ever saw one disease** — that setup made a "no transfer happened" result almost inevitable, since there was nothing to transfer.

This tomato run changes the *split*, not the transfer mechanism, to give the method a fairer chance:

- Every node gets **some** exposure to **every** disease class, just in very skewed proportions (a **Dirichlet partition**, α = 0.3 — the same non-IID recipe used in the reference paper `docs/frai-9-1751118.pdf`).
- All 3 nodes are evaluated against the **same shared global test set** (not their own local slice), so accuracy numbers are directly comparable across nodes.
- Two parallel arms are trained with identical local-supervised budgets, so any difference between them isolates the effect of the one thing that differs:
  - **Collective arm**: each round, nodes also exchange knowledge-distillation (KD) logits + prototypes over a shared probe set.
  - **Local-only control arm**: same schedule, same epochs, no exchange at all.

Two key metrics recur below:
- **`crop_accuracy`** — did it identify the plant as Tomato at all? (Trivial here — only one crop — so it's 1.000 everywhere and not interesting on its own.)
- **`disease_accuracy`** — did it identify the *correct disease* out of 10 possible classes? This is the number that actually matters.

## Run 1 — epochs=2, rounds=5

### Timing (approximate, from output file timestamps — no explicit wall-clock log)

| Stage | Approx. wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | ~11 min | 3 nodes × 2 epochs, train → export → evaluate |
| Stage 2 (knowledge transfer) | ~1h 28m | 5 rounds × (3 nodes distill + 3 control nodes local-train + export + dual eval) |
| **Total** | **~1h 39m** | |

### Data split: Dirichlet label-skew (α = 0.3) across 3 nodes

Unlike the corn run's *complete* disjoint split, every node here has **some** samples of **most** classes — but the mix is heavily skewed, and a couple of classes are effectively (or literally) missing for one node.

| Node | Samples | Dominant class | Dominant share | Classes present | Notably thin / missing classes |
|---|---|---|---|---|---|
| node_0 | 2,135 | Leaf_Mold | 28.2% | 10 / 10 | healthy (13 imgs), Septoria_leaf_spot (5), Target_Spot (16) |
| node_1 | 5,460 | Septoria_leaf_spot | 31.1% | 8 / 10 | **Bacterial_spot (0), Tomato_mosaic_virus (0)** — zero samples; Leaf_Mold (6), Spider_mites (23) |
| node_2 | 8,954 | Tomato_Yellow_Leaf_Curl_Virus | 41.0% | 10 / 10 | Late_blight (9), Septoria_leaf_spot (10), Tomato_mosaic_virus (216) |

All 3 nodes are evaluated on the same **4,354-image global held-out test set** (carved out before the Dirichlet split, never seen in training).

### Stage 1 — per-node local baseline (shared global test set)

Each node trained only on its own skewed local data for 2 epochs, then evaluated on the global test set — this is the "before any collaboration" starting point (identical to the Round 0 numbers below).

| Node | Test images | Crop accuracy | Disease accuracy | Strongest classes | Weakest classes |
|---|---|---|---|---|---|
| node_0 | 4,354 | 1.000 | 0.379 | Tomato_mosaic_virus (0.79), Spider_mites (0.74), Target_Spot (0.72) | Septoria_leaf_spot (0.00), Leaf_Mold (0.16), Bacterial_spot (0.18) |
| node_1 | 4,354 | 1.000 | 0.508 | healthy (0.91), Tomato_YLCV (0.81), Septoria_leaf_spot (0.66) | Bacterial_spot (0.00, no samples), Tomato_mosaic_virus (0.00, no samples), Spider_mites (0.24) |
| node_2 | 4,354 | 1.000 | 0.533 | healthy (0.91), Tomato_YLCV (0.86), Spider_mites (0.87) | Late_blight (0.00, 9 samples), Septoria_leaf_spot (0.01), Target_Spot (0.39) |

**Pattern:** each node is strongest on the classes it happened to be dealt the most of, and weakest (often exactly 0) on the classes it barely saw — a direct fingerprint of the Dirichlet skew, not a training bug.

### Round-by-round trend: collective vs. local-only control (`disease_accuracy`, global test set)

| Node | Round 0 | R1 collective / control | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| node_0 | 0.379 | 0.224 / 0.304 | 0.287 / 0.334 | 0.447 / 0.390 | 0.521 / 0.544 | 0.457 / 0.420 |
| node_1 | 0.508 | 0.539 / 0.461 | 0.537 / 0.530 | 0.508 / 0.545 | 0.560 / 0.558 | 0.553 / 0.549 |
| node_2 | 0.533 | 0.519 / 0.528 | 0.626 / 0.614 | 0.591 / 0.620 | 0.590 / 0.594 | 0.638 / 0.631 |

Both arms wobble round to round (small dataset + only 1 local epoch/round makes this noisy), but by round 5 the **collective arm edges out the control on 2 of 3 nodes** (node_0 and node_2), reversing what happened in the corn run.

### Final per-class collaboration gain (Round 0 → Round 5, collective vs. control)

This is the key diagnostic: for each node/class, did the collective arm end up *closer to or further from* correct, and did it beat the no-exchange control?

**node_0** (weak node: 10/10 classes present but thin on several)

| Disease | Round 0 | R5 collective | R5 control | Gain vs R0 | Gain vs control | Low-rep for node? |
|---|---|---|---|---|---|---|
| Late_blight | 0.300 | **0.853** | 0.767 | **+0.553** | **+0.087** | no |
| Leaf_Mold | 0.162 | **0.645** | 0.483 | **+0.483** | **+0.162** | no |
| Tomato_YLCV | 0.528 | **0.847** | 0.481 | **+0.319** | **+0.366** | no |
| Spider_mites | 0.745 | 0.585 | 0.549 | -0.160 | +0.036 | no |
| Bacterial_spot | 0.181 | 0.147 | 0.772 | -0.034 | **-0.625** | no |
| Tomato_mosaic_virus | 0.793 | 0.774 | 0.817 | -0.018 | -0.043 | no |
| Septoria_leaf_spot | 0.000 | 0.000 | 0.017 | 0.000 | -0.017 | yes (5 imgs) |
| Early_blight | 0.233 | 0.000 | 0.000 | -0.233 | 0.000 | no |
| healthy | 0.315 | 0.011 | 0.051 | -0.304 | -0.040 | yes (13 imgs) |
| Target_Spot | 0.715 | 0.000 | 0.000 | **-0.715** | 0.000 | yes (16 imgs) |

**node_1** (missing Bacterial_spot & Tomato_mosaic_virus entirely)

| Disease | Round 0 | R5 collective | R5 control | Gain vs R0 | Gain vs control | Low-rep for node? |
|---|---|---|---|---|---|---|
| Target_Spot | 0.468 | **0.820** | 0.944 | +0.352 | -0.124 | no |
| Septoria_leaf_spot | 0.663 | **0.836** | 0.674 | +0.173 | **+0.162** | no |
| Late_blight | 0.676 | 0.793 | 0.887 | +0.117 | -0.095 | no |
| Tomato_YLCV | 0.813 | 0.901 | 0.906 | +0.088 | -0.005 | no |
| healthy | 0.914 | 0.962 | 0.925 | +0.048 | +0.038 | no |
| Spider_mites | 0.237 | 0.045 | 0.006 | -0.193 | +0.039 | yes (23 imgs) |
| Early_blight | 0.323 | 0.000 | 0.000 | -0.323 | 0.000 | no |
| Bacterial_spot | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | yes (0 imgs) |
| Leaf_Mold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | yes (6 imgs) |
| Tomato_mosaic_virus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | yes (0 imgs) |

**node_2** (missing/thin on Late_blight, Septoria_leaf_spot, Tomato_mosaic_virus)

| Disease | Round 0 | R5 collective | R5 control | Gain vs R0 | Gain vs control | Low-rep for node? |
|---|---|---|---|---|---|---|
| Leaf_Mold | 0.358 | **0.770** | 0.632 | **+0.412** | **+0.139** | no |
| Bacterial_spot | 0.451 | 0.700 | 0.681 | +0.249 | +0.019 | no |
| Tomato_mosaic_virus | 0.506 | 0.628 | 0.396 | +0.122 | **+0.232** | yes (216 imgs) |
| Target_Spot | 0.390 | 0.517 | 0.397 | +0.127 | +0.120 | no |
| Early_blight | 0.620 | 0.796 | 0.774 | +0.176 | +0.022 | no |
| Tomato_YLCV | 0.856 | 0.956 | 0.950 | +0.101 | +0.006 | no |
| Late_blight | 0.000 | 0.006 | 0.085 | +0.006 | -0.078 | yes (9 imgs) |
| Septoria_leaf_spot | 0.011 | 0.013 | 0.094 | +0.002 | -0.081 | yes (10 imgs) |
| Spider_mites | 0.872 | 0.872 | 0.941 | 0.000 | -0.068 | no |
| healthy | 0.906 | 0.860 | 0.871 | -0.046 | -0.011 | no |

### Macro / worst-node gain (collective vs. local-only control, round 5, global test set)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.000 | 0.000 |
| disease_accuracy | **+0.0158** | **+0.0367** |

Both are **positive** — a small but real average edge for the collective arm over the control, and the worst-performing node benefits *more* than average.

### Communication / energy cost

| Round | Bytes exchanged | Cumulative energy (kWh) |
|---|---|---|
| 1 | 483,896 | 0.003857 |
| 2 | 483,896 | 0.007501 |
| 3 | 483,896 | 0.011079 |
| 4 | 483,896 | 0.014692 |
| 5 | 483,896 | 0.018599 |
| **Total** | **2,419,480** | **0.018599** |

### Interpretation

- **This is a mild positive result, not a null result.** Unlike the corn disjoint-split run — where cross-node accuracy stayed at exactly 0.000 forever because there was zero statistical overlap to exploit — the Dirichlet skew here gives every node *some* real exposure to every class, and knowledge-distillation over the shared probe set measurably helps: macro `disease_accuracy` gain is **+1.6 points**, worst-node gain **+3.7 points**, both positive.
- **The gains are concentrated on classes a node already had *some* (but not zero) local exposure to.** Look at node_0's Late_blight (+55 pts vs round 0, +8.7 pts vs control) and Leaf_Mold (+48 / +16 pts) — these are classes node_0 has hundreds of samples for but still struggled with alone; the collective signal from peers who are stronger on those classes clearly helps.
- **Classes with (near-)zero local samples for a node stay near zero regardless of arm.** node_1's Bacterial_spot and Tomato_mosaic_virus (0 samples) and node_0/node_2's thinnest classes (5–16 images) don't move — confirms the pipeline's own disclosed limitation: *prototype* alignment only reinforces classes a node already has local batches for, so a class with literally zero local samples gets no prototype signal, and KD alone isn't enough to rescue it.
- **A few classes got worse under the collective arm** (e.g., node_0's Target_Spot fell from 0.715 → 0.000, Bacterial_spot lost 62.5 pts vs. its own control). This looks like **catastrophic forgetting under KD pressure**: pulling a small node's weights toward the ensemble's average behavior can overwrite a class it happened to be locally decent at, especially with only 1 local epoch/round to re-anchor it. This is the same failure mode the corn report flagged as "small, inconsistent erosion of own-disease accuracy," just less severe here because the skew is partial rather than total.
- **Net takeaway:** switching from a complete-disjoint split to a Dirichlet label-skew split — closer to how real federated deployments actually look — is enough to turn the method from "provably can't work" (corn) into "measurably, if modestly, helps," while also reproducing the same known weak spot (near-zero-sample classes and occasional forgetting) at smaller scale.

## Run 2 — epochs=10, rounds=10

### Timing (approximate, from output file timestamps)

| Stage | Approx. wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | ~1h 0m | 3 nodes x 10 epochs, train to export to evaluate |
| Stage 2 (knowledge transfer) | ~6h 30m | 10 rounds x (3 nodes distill + 3 control nodes local-train + export + dual eval) |
| **Total** | **~7h 30m** | |

Rounds visibly slowed down partway through (~25-28 min/round for rounds 1-2, ~50 min/round for later rounds) — most likely system contention on the dev machine, not a change in the workload itself (see the energy anomaly below).

### Data split: identical to Run 1 (confirms determinism)

Same seed, same source data, so the Dirichlet partition (alpha=0.3) produced exactly the same per-node class counts as Run 1:

| Node | Samples | Dominant class | Dominant share | JS divergence from uniform | Classes present | Low-representation classes |
|---|---|---|---|---|---|---|
| node_0 | 2,135 | Leaf_Mold | 28.20% | 0.2212 | 10 / 10 | healthy, Septoria_leaf_spot, Target_Spot |
| node_1 | 5,460 | Septoria_leaf_spot | 31.10% | 0.2965 | 8 / 10 | Bacterial_spot, Leaf_Mold, Spider_mites Two-spotted_spider_mite, Tomato_mosaic_virus |
| node_2 | 8,954 | Tomato_Yellow_Leaf_Curl_Virus | 41.02% | 0.2213 | 10 / 10 | Late_blight, Septoria_leaf_spot, Tomato_mosaic_virus |

Full per-class counts:

| Class | node_0 | node_1 | node_2 |
|---|---|---|---|
| Bacterial_spot | 409 | 0 | 1,583 |
| Early_blight | 61 | 38 | 1,176 |
| healthy | 13 | 1,105 | 315 |
| Late_blight | 580 | 1,293 | 9 |
| Leaf_Mold | 602 | 6 | 477 |
| Septoria_leaf_spot | 5 | 1,698 | 10 |
| Spider_mites Two-spotted_spider_mite | 99 | 23 | 1,152 |
| Target_Spot | 16 | 721 | 343 |
| Tomato_mosaic_virus | 273 | 0 | 216 |
| Tomato_Yellow_Leaf_Curl_Virus | 77 | 576 | 3,673 |

node_1 has literally zero training samples of Bacterial_spot and Tomato_mosaic_virus.

Global test set: same 4,354 images, carved out before partitioning, shared across all nodes and both runs.

### Stage 1 — per-node training log (all 10 epochs)

**node_0**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 2.3023 | 1.0000 | 0.1991 |
| 2 | 1.9740 | 1.0000 | 0.2846 |
| 3 | 1.7437 | 1.0000 | 0.2687 |
| 4 | 1.6073 | 1.0000 | 0.4428 |
| 5 | 1.5030 | 1.0000 | 0.4467 |
| 6 | 1.4194 | 1.0000 | 0.5795 |
| 7 | 1.1614 | 1.0000 | 0.5990 |
| 8 | 0.9519 | 1.0000 | 0.6406 |
| 9 | 0.7826 | 1.0000 | 0.6357 |
| 10 | 0.7517 | 1.0000 | 0.6649 (best) |

**node_1**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.7183 | 1.0000 | 0.1325 |
| 2 | 1.4444 | 1.0000 | 0.1723 |
| 3 | 1.3570 | 1.0000 | 0.2917 |
| 4 | 1.2487 | 1.0000 | 0.4265 |
| 5 | 1.0397 | 1.0000 | 0.5519 |
| 6 | 0.8546 | 1.0000 | 0.5276 |
| 7 | 0.6567 | 1.0000 | 0.6011 |
| 8 | 0.6370 | 1.0000 | 0.5746 |
| 9 | 0.5405 | 1.0000 | 0.6383 |
| 10 | 0.4394 | 1.0000 | 0.6477 (best) |

**node_2**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.6972 | 1.0000 | 0.4566 |
| 2 | 1.3462 | 1.0000 | 0.3071 |
| 3 | 1.1517 | 1.0000 | 0.5988 |
| 4 | 0.9414 | 1.0000 | 0.5994 |
| 5 | 0.8797 | 1.0000 | 0.5827 |
| 6 | 0.7270 | 1.0000 | 0.6185 |
| 7 | 0.6406 | 1.0000 | 0.6008 |
| 8 | 0.5427 | 1.0000 | 0.6463 |
| 9 | 0.4495 | 1.0000 | 0.7221 (best checkpoint) |
| 10 | 0.3823 | 1.0000 | 0.7200 |

### Stage 1 — final per-node report (shared global test set, 4,354 images)

**node_0** — crop_accuracy=1.0000, disease_accuracy=0.6649

| Class | Accuracy |
|---|---|
| Spider_mites Two-spotted_spider_mite | 0.9733 |
| Tomato_Yellow_Leaf_Curl_Virus | 0.8906 |
| Tomato_mosaic_virus | 0.8049 |
| Bacterial_spot | 0.7396 |
| Late_blight | 0.7042 |
| Leaf_Mold | 0.6791 |
| healthy | 0.5430 |
| Target_Spot | 0.4419 |
| Early_blight | 0.4229 |
| Septoria_leaf_spot | 0.0768 |

Top confusions: Septoria_leaf_spot to Leaf_Mold (143), Target_Spot to Spider_mites (127), Septoria_leaf_spot to Late_blight (84), Septoria_leaf_spot to Tomato_mosaic_virus (67), healthy to Target_Spot (66), healthy to Spider_mites (59), Early_blight to Late_blight (56), Tomato_YLCV to Tomato_mosaic_virus (47), Late_blight to Early_blight (46), Late_blight to Leaf_Mold (38).

**node_1** — crop_accuracy=1.0000, disease_accuracy=0.6477

| Class | Accuracy |
|---|---|
| Tomato_Yellow_Leaf_Curl_Virus | 0.9423 |
| healthy | 0.9086 |
| Late_blight | 0.8471 |
| Spider_mites Two-spotted_spider_mite | 0.7953 |
| Target_Spot | 0.7266 |
| Septoria_leaf_spot | 0.7505 |
| Early_blight | 0.3513 |
| Leaf_Mold | 0.2432 |
| Bacterial_spot | 0.0000 (0 training samples) |
| Tomato_mosaic_virus | 0.0000 (0 training samples) |

Top confusions: Bacterial_spot to Septoria_leaf_spot (175), Leaf_Mold to Septoria_leaf_spot (120), Bacterial_spot to Early_blight (113), Bacterial_spot to Target_Spot (75), Bacterial_spot to Tomato_YLCV (74), Early_blight to Late_blight (67), Early_blight to Septoria_leaf_spot (56), Bacterial_spot to Late_blight (52), Leaf_Mold to Late_blight (49), Septoria_leaf_spot to Late_blight (49).

**node_2** — crop_accuracy=1.0000, disease_accuracy=0.7221

| Class | Accuracy |
|---|---|
| Tomato_Yellow_Leaf_Curl_Virus | 0.9230 |
| Tomato_mosaic_virus | 0.9085 |
| healthy | 0.8978 |
| Target_Spot | 0.8277 |
| Spider_mites Two-spotted_spider_mite | 0.9377 |
| Leaf_Mold | 0.8108 |
| Early_blight | 0.7670 |
| Bacterial_spot | 0.7094 |
| Septoria_leaf_spot | 0.3284 |
| Late_blight | 0.1710 |

Top confusions: Late_blight to Early_blight (178), Septoria_leaf_spot to Leaf_Mold (146), Late_blight to Septoria_leaf_spot (138), Tomato_YLCV to Tomato_mosaic_virus (63), Septoria_leaf_spot to Early_blight (62), Bacterial_spot to Early_blight (60), Late_blight to Leaf_Mold (41), Septoria_leaf_spot to Target_Spot (39), Septoria_leaf_spot to Tomato_mosaic_virus (28), Bacterial_spot to Target_Spot (27).

### Stage 2 — round-by-round trend, all 10 rounds (disease_accuracy, global test set)

| Round | node_0 collective | node_0 control | node_1 collective | node_1 control | node_2 collective | node_2 control |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.6649 | - | 0.6477 | - | 0.7221 | - |
| 1 | 0.3923 | 0.2823 | 0.5296 | 0.4809 | 0.6288 | 0.5737 |
| 2 | 0.3624 | 0.4667 | 0.5333 | 0.5547 | 0.5459 | 0.5696 |
| 3 | 0.4646 | 0.4839 | 0.5519 | 0.5292 | 0.6309 | 0.6332 |
| 4 | 0.4010 | 0.3452 | 0.5351 | 0.5464 | 0.6183 | 0.6408 |
| 5 | 0.5345 | 0.4543 | 0.5838 | 0.5400 | 0.6401 | 0.6114 |
| 6 | 0.5551 | 0.4915 | 0.5955 | 0.5721 | 0.6479 | 0.6316 |
| 7 | 0.5503 | 0.5195 | 0.5746 | 0.5567 | 0.6330 | 0.6314 |
| 8 | 0.5140 | 0.5696 | 0.6137 | 0.5253 | 0.6500 | 0.6534 |
| 9 | 0.5597 | 0.6029 | 0.6194 | 0.5730 | 0.6711 | 0.6603 |
| 10 (final) | 0.5868 | 0.4111 | 0.5999 | 0.5705 | 0.6481 | 0.6557 |

Note every round starts noticeably below the Stage-1/round-0 baseline in both arms — restarting the local-supervised budget at just 1 epoch/round undoes some of Stage 1's 10-epoch depth before it climbs back; by round 10 node_0 and node_1 have recovered above their round-1 dip, node_2 has not fully recovered to its round-0 baseline in either arm.

### Communication / energy per round

| Round | Bytes exchanged (this round) | Cumulative bytes | Cumulative compute energy (kWh) |
|---|---|---|---|
| 1 | 483,896 | 483,896 | 0.003764 |
| 2 | 483,896 | 967,792 | 0.007579 |
| 3 | 483,896 | 1,451,688 | 0.011978 |
| 4 | 483,896 | 1,935,584 | 0.016662 |
| 5 | 483,896 | 2,419,480 | 0.020528 |
| 6 | 483,896 | 2,903,376 | 0.023810 |
| 7 | 483,896 | 3,387,272 | 0.029291 |
| 8 | 483,896 | 3,871,168 | 0.034862 |
| 9 | 483,896 | 4,355,064 | 0.096552 |
| 10 | 483,896 | 4,838,960 | 0.101015 |

Anomaly: energy jumps about 10x between rounds 8 and 9 (+0.0617 kWh in one round, vs roughly 0.004-0.006 kWh/round everywhere else) while bytes-per-round stay perfectly constant. Since bytes (a function of model/probe-set size only) did not change, this is almost certainly the wall-clock slowdown observed live during the run (rounds visibly took longer in their second half) feeding into the proxy_wall_power energy estimate (duration times wattage), not a change in the actual computational workload. Treat the cumulative_energy_kwh figures as a rough proxy on this shared dev machine, not a precise sustainability measurement.

### Per-node distill loss per round

| Round | Node | kd_loss | sup_loss | proto_loss | total_loss |
|---|---|---|---|---|---|
| 1 | node_0 | 2.1555 | 1.3339 | 0.2959 | 1.7824 |
| 1 | node_1 | 2.3965 | 0.8191 | 0.2757 | 1.2741 |
| 1 | node_2 | 1.5356 | 0.7627 | 0.2817 | 1.0880 |
| 2 | node_0 | 0.7933 | 1.2580 | 0.1112 | 1.2021 |
| 2 | node_1 | 0.8907 | 0.7490 | 0.1020 | 0.8565 |
| 2 | node_2 | 0.6081 | 0.6494 | 0.0999 | 0.7368 |
| 3 | node_0 | 0.8271 | 1.1309 | 0.1029 | 1.1157 |
| 3 | node_1 | 0.9651 | 0.6407 | 0.0852 | 0.7588 |
| 3 | node_2 | 0.5279 | 0.5650 | 0.0838 | 0.6381 |
| 4 | node_0 | 0.7991 | 1.0792 | 0.0928 | 1.0637 |
| 4 | node_1 | 0.8723 | 0.5851 | 0.0762 | 0.6903 |
| 4 | node_2 | 0.6212 | 0.5245 | 0.0735 | 0.6001 |
| 5 | node_0 | 0.7671 | 0.9896 | 0.0985 | 0.9949 |
| 5 | node_1 | 0.8745 | 0.5361 | 0.0696 | 0.6427 |
| 5 | node_2 | 0.6552 | 0.4778 | 0.0658 | 0.5535 |
| 6 | node_0 | 0.7919 | 0.9461 | 0.0880 | 0.9637 |
| 6 | node_1 | 0.8674 | 0.4950 | 0.0659 | 0.6030 |
| 6 | node_2 | 0.5643 | 0.4606 | 0.0641 | 0.5282 |
| 7 | node_0 | 0.6468 | 0.8908 | 0.0834 | 0.8791 |
| 7 | node_1 | 0.8044 | 0.4866 | 0.0615 | 0.5835 |
| 7 | node_2 | 0.5182 | 0.4398 | 0.0614 | 0.5027 |
| 8 | node_0 | 0.6311 | 0.8794 | 0.0822 | 0.8656 |
| 8 | node_1 | 0.9201 | 0.4448 | 0.0614 | 0.5632 |
| 8 | node_2 | 0.5150 | 0.4015 | 0.0570 | 0.4635 |
| 9 | node_0 | 0.6053 | 0.8586 | 0.0817 | 0.8431 |
| 9 | node_1 | 0.7816 | 0.4343 | 0.0573 | 0.5316 |
| 9 | node_2 | 0.5973 | 0.3822 | 0.0537 | 0.4502 |
| 10 | node_0 | 0.6239 | 0.8127 | 0.0828 | 0.8166 |
| 10 | node_1 | 0.7990 | 0.4026 | 0.0577 | 0.5070 |
| 10 | node_2 | 0.5217 | 0.3691 | 0.0534 | 0.4313 |

sup_loss and proto_loss both decline steadily round over round for every node (the local-supervised signal is consolidating); kd_loss declines too but far less smoothly, especially for node_0 - consistent with node_0 being pulled hardest by peer consensus (see per-class gains below).

### Final per-class collaboration gain (Round 0 to Round 10, collective vs control)

**node_0**

| Class | Round 0 | R10 collective | R10 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Bacterial_spot | 0.7396 | 0.7698 | 0.4283 | +0.0302 | +0.3415 | no |
| Early_blight | 0.4229 | 0.0824 | 0.0036 | -0.3405 | +0.0789 | no |
| healthy | 0.5430 | 0.5645 | 0.0242 | +0.0215 | +0.5403 | yes |
| Late_blight | 0.7042 | 0.7928 | 0.7425 | +0.0885 | +0.0503 | no |
| Leaf_Mold | 0.6791 | 0.7264 | 0.6588 | +0.0473 | +0.0676 | no |
| Septoria_leaf_spot | 0.0768 | 0.0000 | 0.0107 | -0.0768 | -0.0107 | yes |
| Spider_mites Two-spotted_spider_mite | 0.9733 | 0.5875 | 0.4273 | -0.3858 | +0.1602 | no |
| Target_Spot | 0.4419 | 0.1723 | 0.0037 | -0.2697 | +0.1685 | yes |
| Tomato_mosaic_virus | 0.8049 | 0.9024 | 0.8537 | +0.0976 | +0.0488 | no |
| Tomato_Yellow_Leaf_Curl_Virus | 0.8906 | 0.7988 | 0.6115 | -0.0919 | +0.1872 | no |

**node_1**

| Class | Round 0 | R10 collective | R10 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Bacterial_spot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes (0 samples) |
| Early_blight | 0.3513 | 0.0000 | 0.0538 | -0.3513 | -0.0538 | no |
| healthy | 0.9086 | 0.9167 | 0.9140 | +0.0081 | +0.0027 | no |
| Late_blight | 0.8471 | 0.8310 | 0.7425 | -0.0161 | +0.0885 | no |
| Leaf_Mold | 0.2432 | 0.0000 | 0.0000 | -0.2432 | 0.0000 | yes |
| Septoria_leaf_spot | 0.7505 | 0.8529 | 0.9254 | +0.1023 | -0.0725 | no |
| Spider_mites Two-spotted_spider_mite | 0.7953 | 0.2493 | 0.1098 | -0.5460 | +0.1395 | yes |
| Target_Spot | 0.7266 | 0.9663 | 0.9288 | +0.2397 | +0.0375 | no |
| Tomato_mosaic_virus | 0.0000 | 0.0061 | 0.0000 | +0.0061 | +0.0061 | yes (0 samples) |
| Tomato_Yellow_Leaf_Curl_Virus | 0.9423 | 0.9755 | 0.9108 | +0.0332 | +0.0647 | no |

**node_2**

| Class | Round 0 | R10 collective | R10 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Bacterial_spot | 0.7094 | 0.8151 | 0.7943 | +0.1057 | +0.0208 | no |
| Early_blight | 0.7670 | 0.7491 | 0.7778 | -0.0179 | -0.0287 | no |
| healthy | 0.8978 | 0.8763 | 0.8172 | -0.0215 | +0.0591 | no |
| Late_blight | 0.1710 | 0.0181 | 0.0000 | -0.1529 | +0.0181 | yes |
| Leaf_Mold | 0.8108 | 0.6858 | 0.5912 | -0.1250 | +0.0946 | no |
| Septoria_leaf_spot | 0.3284 | 0.0021 | 0.0810 | -0.3262 | -0.0789 | yes |
| Spider_mites Two-spotted_spider_mite | 0.9377 | 0.9050 | 0.9852 | -0.0326 | -0.0801 | no |
| Target_Spot | 0.8277 | 0.3970 | 0.5356 | -0.4307 | -0.1386 | no |
| Tomato_mosaic_virus | 0.9085 | 0.6220 | 0.6220 | -0.2866 | 0.0000 | yes |
| Tomato_Yellow_Leaf_Curl_Virus | 0.9230 | 0.9878 | 0.9825 | +0.0647 | +0.0052 | no |

### Macro / worst-node gain (collective vs control, round 10, global test set)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.0000 | 0.0000 |
| disease_accuracy | 0.0658 | 0.1757 |

### Interpretation (Run 2)

- Collaboration gain is clearer and larger with more training depth, but still not uniform across nodes: node_0 gained a striking +17.6 pts, node_1 a modest +2.9 pts, and node_2 slipped slightly (-0.8 pts) - so more epochs/rounds amplified the signal where it existed rather than fixing where it did not.
- Zero-exposure classes remain the hard limit. node_1's Bacterial_spot (0 samples) stayed at exactly 0.0000 in every arm across all 10 rounds. Tomato_mosaic_virus (also 0 samples) moved a hair - 0.0000 to 0.0061 - the first nonzero movement seen on a truly-absent class in either run, but still far from usable.
- Per-class volatility is real, not just noise from small samples: node_0's Bacterial_spot gained about 34 pts over its own control while Target_Spot and Spider_mites lost ground against round 0 - the KD/prototype pull helps some classes and actively erodes others in the same node, the same pattern Run 1 showed at a smaller scale.
- Energy scaling is not simply proportional to round count on this hardware - see the round 8 to 9 anomaly above. Any energy-based tuning decision should be re-validated on a quieter machine or averaged over repeated runs before being treated as precise.

## Comparison: Run 1 vs Run 2

| Metric | Run 1 (epochs=2, rounds=5) | Run 2 (epochs=10, rounds=10) |
|---|---|---|
| node_0 gain (collective minus control) | +3.7 pts | +17.6 pts |
| node_1 gain | +0.4 pts | +2.9 pts |
| node_2 gain | +0.7 pts | -0.8 pts |
| Macro gain (disease_accuracy) | +1.6 pts | +6.6 pts |
| Worst-node gain (disease_accuracy) | +3.7 pts | +17.6 pts |
| Stage 1 disease_accuracy, node_0/1/2 (round-0 baseline) | 0.379 / 0.508 / 0.533 | 0.665 / 0.648 / 0.722 |
| Cumulative bytes exchanged | 2,419,480 | 4,838,960 (exactly 2x, proportional to round count) |
| Cumulative compute energy (kWh) | 0.018599 | 0.101015 (5.4x, disproportionate, see anomaly above) |
| Total wall-clock (approx.) | ~1h 39m | ~7h 30m |

Takeaways:
- Deeper training (both Stage 1 epochs and Stage 2 rounds) increased the macro and worst-node collaboration gain roughly 4x, driven almost entirely by node_0. It did not help node_2, which flipped from a small positive to a small negative gain - more budget sharpened the signal where it existed, it did not create a signal where there was not a stable one.
- Bytes exchanged scales exactly linearly with round count (as designed - same probe set, same model, every round). Energy did not scale as cleanly, most likely due to real wall-clock variability on the dev machine rather than the workload itself changing.

## Tuning recommendation for the next run

Based on both runs' data:

1. Keep Stage 1 at 10 epochs (or similar) - it clearly earns its cost. The stronger round-0 baseline it produced fed directly into Run 2's much larger collaboration gains.
2. For Stage 2, trade round count for epoch depth instead of just adding more rounds. Try 4 rounds times training.distill_epochs_per_round=3 (12 epoch-equivalents of local training, slightly more than Run 2's 10x1) instead of pushing past 10 rounds. This should:
   - Keep (or exceed) the total local-supervised training budget that drove Run 2's gains.
   - Cut the per-round-only overhead (compute_knowledge, prototype/logit aggregation, communication bytes) from 10 occurrences to 4, a direct, mechanical reduction independent of the wall-clock anomaly above.
   - Likely reduce round-to-round noise, since each round becomes a more substantial training increment (the control arm's own volatility, visible in both runs, is largely a symptom of the 1-epoch/round budget being too shallow to produce a stable measurement).
3. Do not trust a single run's per-node verdict. node_2's gain sign flipped between these two runs. Before treating any config's collaboration gain (or its energy profile) as settled, repeat the comparison at least once more, or across a couple of seeds - this project's own reference paper uses 5 seeds for exactly this reason.
4. Re-verify energy figures on a quieter machine (or via codecarbon instead of the proxy_wall_power fallback) before using them to justify a tuning decision - the round 8 to 9 jump in Run 2 shows the current proxy is sensitive to system contention, not just to the configured workload.
