# Apple Dirichlet-Skew Knowledge-Transfer — Real Run Results

Date: 2026-08-22 to 2026-08-23
Spec/scripts: `src/validation/run_apple_pipeline.py` (Stage 1), `src/validation/run_apple_knowledge_transfer.py` (Stage 2), `src/validation/apple_mesh_dataset.py` — direct mirrors of the Tomato Dirichlet-mesh pipeline (`docs/tomato_disease_knowledge_transfer_results.md`), same aggregation mechanics (`run_knowledge_transfer.run_kt_round`), different crop/data split.
Against the real, pooled multi-source Apple dataset (PlantVillage + PlantDoc + PlantWild v1/v2), CPU-only PyTorch, `.venv`.

This file covers **four completed real runs** against the same real, pooled multi-source Apple dataset, differing in Dirichlet `alpha` (partition skew), the Stage 1/Stage 2 epoch/round budget, and (Run 3/4) the disease class set:

| Run | alpha | Classes | Stage 1 epochs/node | Stage 2 rounds x distill-epochs | Output folder |
|---|---|---|---|---|---|
| [Run 1](#run-1--alpha03-epochs20-rounds2-x-10) | 0.3 | 4 (incl. `Black_rot`) | 20 | 2 x 10 (20 epoch-equiv.) | `outputs/validation/apple_mesh/` |
| [Run 2](#run-2--alpha07-epochs4-rounds4-x-4) | 0.7 | 4 (incl. `Black_rot`) | 4 | 4 x 4 (16 epoch-equiv.) | `outputs/validation/apple_mesh_a0.7/` (overwritten by Run 3 — see below) |
| [Run 3](#run-3--alpha07-epochs20-rounds4-x-4-black_rot-excluded) | 0.7 | 3 (`Black_rot` excluded) | 20 | 4 x 4 (24 epoch-equiv.) | `outputs/validation/apple_mesh_a0.7/` |
| [Run 4](#run-4--alpha03-epochs20-rounds4-x-4-black_rot-excluded) | **0.3** | 3 (`Black_rot` excluded) | 20 | 4 x 4 (24 epoch-equiv.) | `outputs/validation/apple_mesh_a0.3/` |

**Run 3 reuses Run 2's output folder** (`outputs/validation/apple_mesh_a0.7/`), overwriting it — Run 2's raw checkpoints/reports no longer exist on disk; its numbers below are preserved only in this document. **Run 4 uses a new folder** (`outputs/validation/apple_mesh_a0.3/`) and does not touch Run 3's data.

See the [4-way comparison](#comparison-run-1-vs-run-2-vs-run-3-vs-run-4) and [tuning recommendation](#tuning-recommendation-for-the-next-run) sections at the end for the cross-run analysis. **Headline finding: Run 4 is a clean, alpha-isolated comparison against Run 3 — identical class set (3, `Black_rot` excluded), identical epoch/round budget (20 + 4x4), only `dirichlet_alpha` differs (0.3 vs 0.7) — and it shows the milder-skew run (Run 3, alpha=0.7) had a real, unambiguous positive collaboration signal (macro +4.38 pts, worst-node +4.92 pts) while the sharper-skew run (Run 4, alpha=0.3) came back essentially flat/null (macro -0.16 pts, worst-node -1.06 pts), matching Run 1/Run 2's pattern rather than Run 3's.** This is the first time in this series that two runs differ by exactly one variable, and it points toward alpha (skew severity), not the class-count change, as the more likely driver of Run 3's positive result.

## Pipeline notes / fixes made to enable Run 1 (carried into Run 2 unchanged)

Two real bugs were found and fixed while getting this pipeline to run for Run 1 (both are Windows-specific correctness fixes, not changes to model/aggregation behavior); both fixes are already in the code and applied to Run 2 as well:

1. **Windows `MAX_PATH` (260-char) crash in `apple_mesh_dataset.py`.** One PlantDoc Apple stock-photo filename (`data/PlantDoc/train/Apple leaf/apple-tree-branch-...-928225.jpg`) is long enough that its absolute path exceeds Windows' 260-character limit once combined with this checkout's path depth, causing `PIL.Image.open()` to raise `FileNotFoundError` even though the file exists. Fixed with a `\\?\` extended-length-path helper (`_win_long_path`) applied at the two `Image.open()` call sites in `apple_mesh_dataset.py`, plus the same fix in `evaluate_onnx.py` (which opens test-set images directly by path, bypassing the dataset's `__getitem__`).
2. **Stage 2 disclosure bug**: `build_apple_knowledge_transfer_summary()` was re-deriving `local_only_budget.epochs_per_round` / `collective_budget.distill_epochs_per_round` from `cfg.get("training.distill_epochs_per_round", 1)` instead of using the actual `--distill-epochs` CLI value the round actually ran with. This only affects the disclosure metadata in `knowledge_transfer_summary.json` (the real training already used the correct value — confirmed by round wall-clock time and loss curves); fixed by threading `distill_epochs` through as an explicit parameter. Run 1's `knowledge_transfer_summary.json` was regenerated from its already-computed round summaries (no need to re-run training) to pick up the fix; Run 2 generated its summary correctly from the start.

## Run 1 — alpha=0.3, epochs=20, rounds=2 x 10

Ran via:
```
python -m src.validation.run_apple_pipeline --epochs 20
python -m src.validation.run_apple_knowledge_transfer --rounds 2 --distill-epochs 10
```

### Why this configuration

The Tomato report's own [tuning recommendation](tomato_disease_knowledge_transfer_results.md#tuning-recommendation-for-the-next-run) argued for trading Stage-2 round *count* for epoch *depth* per round (fewer, deeper rounds instead of many shallow ones) to cut per-round overhead and reduce round-to-round noise from an under-trained control arm. This run follows that recommendation directly: Stage 1 keeps a deep local baseline (20 epochs/node), and Stage 2 uses only 2 rounds but at 10 local-supervised epochs/round each (20 epoch-equivalents total, budget-aligned between the collective and local-only-control arms).

### Timing (from output file timestamps)

| Stage | Wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | 37m 52s | 3 nodes x 20 epochs, train -> export -> evaluate |
| Stage 2 (knowledge transfer) | 1h 42m 3s | round-0 baseline + 2 rounds x (3 nodes distill + 3 control nodes local-train x10 epochs + export + eval) |
| **Total** | **~2h 19m 55s** | |

Per-node Stage 1 breakdown: node_0 (1,402 train images) 12m 38s, node_1 (723 images) 9m 8s, node_2 (1,782 images) 15m 54s. Per-round Stage 2 breakdown: round 1 51m 29s, round 2 49m 25s (round-0 baseline eval itself took ~1m 9s).

### Data split: Dirichlet label-skew (alpha=0.3) across 3 nodes, merged multi-source Apple pool

Sources: PlantVillage, PlantDoc, PlantWild v1, PlantWild v2. 4 canonical disease classes (`Apple_scab`, `Black_rot`, `Cedar_apple_rust`, `healthy`). Global test split (1,028 images) carved out before the Dirichlet partition, shared across all 3 nodes and both runs.

| Node | Samples | Dominant class | Dominant share | JS divergence from uniform | Classes present | Low-representation classes |
|---|---|---|---|---|---|---|
| node_0 | 1,402 | healthy | 77.5% | 0.2807 | 4 / 4 | Black_rot (1 image) |
| node_1 | 723 | Apple_scab | 94.5% | 0.4328 | 3 / 4 | Black_rot (0), healthy (10) |
| node_2 | 1,782 | Black_rot | 36.4% | 0.0941 | 4 / 4 | Apple_scab (38 images) |

Full per-class counts:

| Class | node_0 | node_1 | node_2 |
|---|---|---|---|
| Apple_scab | 243 | 683 | 38 |
| Black_rot | 1 | 0 | 649 |
| Cedar_apple_rust | 71 | 30 | 516 |
| healthy | 1,087 | 10 | 579 |

node_1 has literally **zero** training samples of `Black_rot` — the same "at least one node with zero exposure to a class" pattern the Tomato run's node_1 showed for `Bacterial_spot`/`Tomato_mosaic_virus`.

### Stage 1 — per-node training log (all 20 epochs)

**node_0**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.0033 | 1.0000 | 0.3288 |
| 2 | 0.8259 | 1.0000 | 0.5924 |
| 3 | 0.8234 | 1.0000 | 0.6051 |
| 4 | 0.7515 | 1.0000 | 0.5146 |
| 5 | 0.8306 | 1.0000 | 0.5749 |
| 6 | 0.7392 | 1.0000 | 0.5350 |
| 7 | 0.6107 | 1.0000 | 0.6634 |
| 8 | 0.6954 | 1.0000 | 0.6498 |
| 9 | 0.6319 | 1.0000 | 0.6352 |
| 10 | 0.5824 | 1.0000 | 0.6975 |
| 11 | 0.5196 | 1.0000 | 0.6566 |
| 12 | 0.4534 | 1.0000 | 0.6877 |
| 13 | 0.4379 | 1.0000 | 0.6955 |
| 14 | 0.3748 | 1.0000 | 0.7111 |
| 15 | 0.3369 | 1.0000 | 0.6994 |
| 16 | 0.3334 | 1.0000 | 0.7150 |
| 17 | 0.3279 | 1.0000 | 0.7043 |
| 18 | 0.3023 | 1.0000 | 0.7101 |
| 19 | 0.3021 | 1.0000 | **0.7208 (best)** |
| 20 | 0.2698 | 1.0000 | 0.7189 |

**node_1**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.4591 | 1.0000 | 0.3716 |
| 2 | 1.3176 | 1.0000 | 0.2636 |
| 3 | 1.0647 | 1.0000 | 0.2539 |
| 4 | 1.5584 | 1.0000 | 0.2422 |
| 5 | 1.0991 | 1.0000 | 0.2422 |
| 6 | 1.0791 | 1.0000 | 0.2957 |
| 7 | 0.9156 | 1.0000 | 0.3726 |
| 8 | 0.9618 | 1.0000 | 0.5185 |
| 9 | 0.9865 | 1.0000 | 0.4105 |
| 10 | 0.7575 | 1.0000 | 0.5214 |
| 11 | 0.9421 | 1.0000 | 0.4319 |
| 12 | 0.8043 | 1.0000 | 0.5088 |
| 13 | 0.7296 | 1.0000 | 0.4640 |
| 14 | 0.7172 | 1.0000 | **0.5467 (best)** |
| 15 | 0.6644 | 1.0000 | 0.5136 |
| 16 | 0.5583 | 1.0000 | 0.5379 |
| 17 | 0.4542 | 1.0000 | 0.5263 |
| 18 | 0.4701 | 1.0000 | 0.5243 |
| 19 | 0.4332 | 1.0000 | 0.5156 |
| 20 | 0.4296 | 1.0000 | 0.5224 |

**node_2**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.2587 | 1.0000 | 0.5282 |
| 2 | 1.0162 | 1.0000 | 0.5486 |
| 3 | 0.8669 | 1.0000 | 0.6177 |
| 4 | 0.7517 | 1.0000 | 0.6887 |
| 5 | 0.7992 | 1.0000 | 0.6498 |
| 6 | 0.8437 | 1.0000 | 0.6566 |
| 7 | 0.6362 | 1.0000 | 0.8035 |
| 8 | 0.6363 | 1.0000 | 0.7442 |
| 9 | 0.5665 | 1.0000 | 0.8064 |
| 10 | 0.4873 | 1.0000 | 0.7753 |
| 11 | 0.4840 | 1.0000 | 0.7967 |
| 12 | 0.4114 | 1.0000 | 0.7957 |
| 13 | 0.4560 | 1.0000 | 0.8054 |
| 14 | 0.3932 | 1.0000 | 0.7870 |
| 15 | 0.3428 | 1.0000 | 0.8025 |
| 16 | 0.3242 | 1.0000 | **0.8259 (best)** |
| 17 | 0.3107 | 1.0000 | 0.8103 |
| 18 | 0.2777 | 1.0000 | 0.8161 |
| 19 | 0.2601 | 1.0000 | 0.8132 |
| 20 | 0.2768 | 1.0000 | 0.8132 |

### Round 0 baseline (shared global test set, 1,028 images)

| Node | Crop accuracy | Disease accuracy | Apple_scab | Black_rot | Cedar_apple_rust | healthy |
|---|---|---|---|---|---|---|
| node_0 | 1.000 | 0.7208 | 0.8527 | 0.0570 | 0.8199 | 0.9087 |
| node_1 | 1.000 | 0.5467 | 0.6783 | 0.0000 | 0.5342 | 0.7236 |
| node_2 | 1.000 | 0.8259 | 0.7171 | 0.7565 | 0.8696 | 0.9087 |

Each node is weakest exactly on the class it was starved of by the Dirichlet draw — node_0 and node_1 both collapse to 0% on `Black_rot` (1 and 0 training images respectively), node_2 is comparatively weak on `Apple_scab` (only 38 images).

### Stage 2 — round-by-round trend (disease_accuracy, global test set)

| Round | node_0 collective | node_0 control | node_1 collective | node_1 control | node_2 collective | node_2 control |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.7208 | - | 0.5467 | - | 0.8259 | - |
| 1 | 0.6430 | 0.5924 | 0.2510 | 0.2510 | 0.6770 | 0.6673 |
| 2 (final) | 0.6440 | 0.6459 | 0.2510 | 0.2510 | 0.7179 | 0.7403 |

Every node drops noticeably below its Stage-1 round-0 baseline in **both** arms — restarting local-supervised optimization with a fresh Adam optimizer state at the start of Stage 2 undoes some of Stage 1's depth before either arm can re-climb. node_2 partially recovers toward its baseline by round 2 in both arms; node_0 and node_1 do not.

**node_1's collective and local-only-control arms produce bit-for-bit identical `disease_accuracy` and per-class accuracy in both rounds** (0.2510, 100% of predictions landing on `Apple_scab`, 0% everywhere else) — not a pipeline bug; node_1's shard is 94.5% `Apple_scab` with zero `Black_rot` samples, and both arms' models independently converged to the same degenerate always-predict-`Apple_scab` solution given that extreme skew plus only 723 training images.

### Final per-class collaboration gain (Round 0 -> Round 2)

**node_0**

| Class | Round 0 | R2 collective | R2 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8527 | 0.6938 | 0.8488 | -0.1589 | -0.1550 | no |
| Black_rot | 0.0570 | **0.1088** | 0.0000 | **+0.0518** | **+0.1088** | yes (1 img) |
| Cedar_apple_rust | 0.8199 | 0.3602 | 0.3043 | -0.4596 | +0.0559 | no |
| healthy | 0.9087 | 0.9712 | 0.9519 | +0.0625 | +0.0192 | no |

**node_1**

| Class | Round 0 | R2 collective | R2 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.6783 | 1.0000 | 1.0000 | +0.3217 | 0.0000 | no |
| Black_rot | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | yes (0 imgs) |
| Cedar_apple_rust | 0.5342 | 0.0000 | 0.0000 | -0.5342 | 0.0000 | no |
| healthy | 0.7236 | 0.0000 | 0.0000 | -0.7236 | 0.0000 | yes (10 imgs) |

**node_2**

| Class | Round 0 | R2 collective | R2 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.7171 | 0.3295 | 0.2946 | -0.3876 | +0.0349 | yes (38 imgs) |
| Black_rot | 0.7565 | 0.8135 | 0.8031 | +0.0570 | +0.0104 | no |
| Cedar_apple_rust | 0.8696 | 0.7578 | 0.9006 | -0.1118 | -0.1429 | no |
| healthy | 0.9087 | 0.8990 | 0.9255 | -0.0096 | -0.0264 | no |

### Macro / worst-node gain (collective vs. local-only control, round 2)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.0000 | 0.0000 |
| disease_accuracy | **-0.0081** | **0.0000** |

### Communication / energy per round

| Round | Bytes exchanged (this round) | Cumulative bytes | Cumulative compute energy (kWh) |
|---|---|---|---|
| 1 | 139,288 | 139,288 | 0.012538 |
| 2 | 139,288 | 278,576 | 0.024555 |

### Per-node distill loss per round

| Round | Node | kd_loss | sup_loss | proto_loss | total_loss |
|---|---|---|---|---|---|
| 1 | node_0 | 1.1199 | 0.3944 | 0.2416 | 0.6986 |
| 1 | node_1 | 2.9718 | 0.3257 | 0.3102 | 1.1595 |
| 1 | node_2 | 0.9641 | 0.5939 | 0.2384 | 0.8461 |
| 2 | node_0 | 0.3787 | 0.3278 | 0.0697 | 0.3951 |
| 2 | node_1 | 0.9104 | 0.2831 | 0.0661 | 0.4749 |
| 2 | node_2 | 0.7477 | 0.4565 | 0.0557 | 0.5368 |

### Interpretation (Run 1)

- Macro `disease_accuracy` gain is essentially flat (-0.8 pts) and worst-node gain is exactly 0.0000 — the collective arm neither meaningfully helps nor hurts relative to the aligned-budget local-only control at this budget.
- The clearest genuine class-level win: node_0's `Black_rot` (+10.9 pts vs. control, +5.2 pts vs. round 0) — a class node_0 had exactly 1 training image for, rescued by cross-node signal from node_2 (649 `Black_rot` images).
- True zero-sample classes stay unrecoverable: node_1's `Black_rot` (0 samples) stayed at exactly 0.0000 in every arm, every round.
- node_1's collapse to a single-class predictor is the dominant story of this run — both arms converged to the same degenerate always-`Apple_scab` solution, and Stage 2 made node_1 *worse* in both arms than its Stage-1-only baseline (0.5467 vs 0.2510 in both Stage-2 arms).

## Run 2 — alpha=0.7, epochs=4, rounds=4 x 4

Ran via:
```
python -m src.validation.run_apple_pipeline --epochs 4 --output-dir outputs/validation/apple_mesh_a0.7
python -m src.validation.run_apple_knowledge_transfer --rounds 4 --distill-epochs 4 --output-dir outputs/validation/apple_mesh_a0.7
```

### Why this configuration

Run 1's node_1 shard (94.5% `Apple_scab`, 0 `Black_rot` samples) produced a hard KD failure mode neither the collective nor control arm could escape. Before re-tuning the round/epoch schedule, the recommendation was to **fix the partition draw first**: `config.yaml`'s `apple_mesh.dirichlet_alpha` was loosened from 0.3 in two preview steps (0.5, then 0.7 — both computed and shown to the user *before* committing to a full run, since there is no `--alpha` CLI flag; `dirichlet_alpha` is config-only). At 0.5, node_1 still had 0 `Black_rot` samples (94.5% -> 75.5% dominance, an improvement, but the zero-sample class persisted). At **0.7**, every node has >=1 sample of every class for the first time, while all three nodes keep clear non-IID skew (no node close to uniform). Stage 1/2 budget was simultaneously lightened to 4 epochs / 4 rounds x 4 distill-epochs (16 epoch-equivalents) per the user's request — **note this is a second, independent change alongside alpha**, which matters for interpreting the comparison below.

### Dirichlet alpha preview (computed before running any training)

| Node | Metric | alpha=0.3 (Run 1) | alpha=0.5 (previewed, not run) | alpha=0.7 (Run 2, used) |
|---|---|---|---|---|
| node_0 | samples | 1,402 | 1,618 | 1,517 |
| | dominant | healthy 77.5% | healthy 59.3% | healthy 59.8% |
| | Black_rot count | 1 | 9 | 22 |
| | classes present | 4/4 | 4/4 | 4/4 |
| node_1 | samples | 723 | 820 | 872 |
| | dominant | Apple_scab 94.5% | Apple_scab 75.5% | Apple_scab 66.6% |
| | Black_rot count | **0** | **0** | **1** |
| | classes present | 3/4 | 3/4 | **4/4** |
| node_2 | samples | 1,782 | 1,469 | 1,518 |
| | dominant | Black_rot 36.4% | healthy 44.8% | healthy 43.2% |
| | Apple_scab count | 38 | 83 | 105 |
| | classes present | 4/4 | 4/4 | 4/4 |

### Timing (from output file timestamps)

| Stage | Wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | 6m 25s | 3 nodes x 4 epochs, train -> export -> evaluate |
| Stage 2 (knowledge transfer) | 1h 8m 45s | round-0 baseline + 4 rounds x (3 nodes distill + 3 control nodes local-train x4 epochs + export + eval) |
| **Total** | **~1h 15m 10s** | |

Per-node Stage 1 breakdown: node_0 2m 27s, node_1 1m 38s, node_2 2m 10s. Per-round Stage 2 breakdown: round-0 baseline ~1m 6s, round 1 17m 35s, round 2 18m 2s, round 3 16m 32s, round 4 15m 30s — markedly faster per round than Run 1 (4 distill-epochs/round vs. 10) despite running twice as many rounds, so total Stage 2 time was still ~33 minutes less than Run 1's.

### Data split: Dirichlet label-skew (alpha=0.7), same 3-node/4-class setup

| Node | Samples | Dominant class | Dominant share | JS divergence from uniform | Classes present | Low-representation classes |
|---|---|---|---|---|---|---|
| node_0 | 1,517 | healthy | 59.8% | 0.1501 | 4 / 4 | Black_rot (22 images) |
| node_1 | 872 | Apple_scab | 66.6% | 0.2068 | 4 / 4 | Black_rot (1 image) |
| node_2 | 1,518 | healthy | 43.2% | 0.1019 | 4 / 4 | Apple_scab (105 images) |

Full per-class counts:

| Class | node_0 | node_1 | node_2 |
|---|---|---|---|
| Apple_scab | 278 | 581 | 105 |
| Black_rot | 22 | 1 | 627 |
| Cedar_apple_rust | 310 | 177 | 130 |
| healthy | 907 | 113 | 656 |

Disclosed `data_split.strength` in `knowledge_transfer_summary.json`: *"every node has some training exposure to all 4 canonical disease classes; skew is proportional (Dirichlet-drawn), not exclusionary"* — the true zero-sample-class problem from Run 1 is gone.

### Stage 1 — per-node training log (all 4 epochs)

**node_0**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.2746 | 1.0000 | 0.5457 |
| 2 | 1.0672 | 1.0000 | 0.2043 |
| 3 | 0.8751 | 1.0000 | 0.4134 |
| 4 | 0.7262 | 1.0000 | **0.7665 (best)** |

**node_1**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.2287 | 1.0000 | 0.5185 |
| 2 | 0.8850 | 1.0000 | 0.5914 |
| 3 | 0.7546 | 1.0000 | 0.5399 |
| 4 | 0.5931 | 1.0000 | **0.6800 (best)** |

**node_2**

| Epoch | Train loss | Test crop acc | Test disease acc |
|---|---|---|---|
| 1 | 1.1741 | 1.0000 | 0.2938 |
| 2 | 0.8570 | 1.0000 | 0.7432 |
| 3 | 0.6472 | 1.0000 | 0.5156 |
| 4 | 0.5457 | 1.0000 | **0.8171 (best)** |

All three nodes happened to peak at epoch 4 (the last epoch) — disease accuracy is visibly noisy epoch-to-epoch at only 4 epochs (e.g. node_0 dips to 0.204 at epoch 2 before jumping to 0.767 at epoch 4), consistent with a cosine LR schedule tuned for a longer run and a small per-node dataset.

### Round 0 baseline (shared global test set, 1,028 images)

| Node | Crop accuracy | Disease accuracy | Apple_scab | Black_rot | Cedar_apple_rust | healthy |
|---|---|---|---|---|---|---|
| node_0 | 1.000 | 0.7665 | 0.6822 | 0.6477 | 0.7453 | 0.8822 |
| node_1 | 1.000 | 0.6800 | 0.8101 | 0.0000 | 0.8385 | 0.8534 |
| node_2 | 1.000 | 0.8171 | 0.7946 | 0.7150 | 0.8447 | 0.8678 |

Notably, **node_1's `Black_rot` accuracy is still 0.0000 at round 0** even though it now has 1 training image of it — a single sample is enough for KD/prototype exchange to have *something* real to align to (see gain table below), but not enough for node_1's own 4-epoch local training alone to learn the class.

### Stage 2 — round-by-round trend (disease_accuracy, global test set)

| Round | node_0 collective | node_0 control | node_1 collective | node_1 control | node_2 collective | node_2 control |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.7665 | - | 0.6800 | - | 0.8171 | - |
| 1 | 0.6313 | 0.5613 | 0.5895 | 0.6313 | 0.6741 | 0.6946 |
| 2 | 0.6994 | 0.5778 | 0.4698 | 0.6508 | 0.6693 | 0.7023 |
| 3 | 0.7014 | 0.6858 | 0.6819 | 0.6313 | 0.7471 | 0.6887 |
| 4 (final) | 0.6469 | 0.6547 | 0.5477 | 0.6566 | 0.7130 | 0.6693 |

Same pattern as Run 1: every node drops below its Stage-1 baseline in round 1 (optimizer reset), in both arms. Unlike Run 1, node_1 is **not** identical between arms any round — its collective and control scores diverge round to round (e.g. round 2: collective 0.4698 vs control 0.6508, a 18-point gap in the control's favor), and node_1's collective score is choppy (0.59 -> 0.47 -> 0.68 -> 0.55) rather than flat-lined, confirming the true zero-sample-class collapse from Run 1 is gone — but the collective arm is not clearly *better* for node_1 either; round 3 is its only round where collective beats control.

### Final per-class collaboration gain (Round 0 -> Round 4)

**node_0**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.6822 | 0.6085 | 0.5814 | -0.0736 | +0.0271 | no |
| Black_rot | 0.6477 | 0.1399 | 0.1088 | -0.5078 | +0.0311 | yes |
| Cedar_apple_rust | 0.7453 | 0.5155 | 0.6584 | -0.2298 | -0.1429 | no |
| healthy | 0.8822 | 0.9567 | 0.9519 | +0.0745 | +0.0048 | no |

**node_1**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8101 | 0.9419 | 0.8760 | +0.1318 | +0.0659 | no |
| Black_rot | 0.0000 | **0.0311** | 0.0000 | **+0.0311** | **+0.0311** | yes (1 img) |
| Cedar_apple_rust | 0.8385 | 0.5590 | 0.7516 | -0.2795 | -0.1925 | no |
| healthy | 0.8534 | 0.5385 | 0.7885 | -0.3149 | -0.2500 | no |

**node_2**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.7946 | 0.4574 | 0.3488 | -0.3372 | +0.1085 | yes |
| Black_rot | 0.7150 | **0.9275** | 0.8290 | **+0.2124** | **+0.0985** | no |
| Cedar_apple_rust | 0.8447 | 0.5404 | 0.2857 | -0.3043 | **+0.2547** | no |
| healthy | 0.8678 | 0.8389 | 0.9423 | -0.0288 | -0.1034 | no |

### Macro / worst-node gain (collective vs. local-only control, round 4)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.0000 | 0.0000 |
| disease_accuracy | **-0.0243** | **-0.1070** |

Both figures are **worse** than Run 1's (-0.0081 macro, 0.0000 worst-node) — see [Comparison](#comparison-run-1-vs-run-2) for why this is not read as "alpha=0.7 is worse than alpha=0.3."

### Communication / energy per round

| Round | Bytes exchanged (this round) | Cumulative bytes | Cumulative compute energy (kWh) |
|---|---|---|---|
| 1 | 147,480 | 147,480 | 0.004118 |
| 2 | 147,480 | 294,960 | 0.008355 |
| 3 | 147,480 | 442,440 | 0.012229 |
| 4 | 147,480 | 589,920 | 0.015833 |

Bytes/round are ~6% higher than Run 1's (147,480 vs 139,288) — a side effect of the alpha=0.7 partition producing slightly different per-class prototype counts per node, not a config change (prototype byte-size scales with how many distinct classes each node has nonzero samples for; alpha=0.7 gives every node all 4 classes, alpha=0.3 gave node_1 only 3).

### Per-node distill loss per round

| Round | Node | kd_loss | sup_loss | proto_loss | total_loss |
|---|---|---|---|---|---|
| 1 | node_0 | 1.0463 | 0.6948 | 0.1539 | 0.8725 |
| 1 | node_1 | 1.7759 | 0.8140 | 0.1625 | 1.1305 |
| 1 | node_2 | 0.7927 | 0.7442 | 0.1359 | 0.8695 |
| 2 | node_0 | 0.2983 | 0.5649 | 0.0575 | 0.5833 |
| 2 | node_1 | 0.7444 | 0.7024 | 0.0760 | 0.7718 |
| 2 | node_2 | 0.5096 | 0.6330 | 0.0705 | 0.6802 |
| 3 | node_0 | 0.3337 | 0.4977 | 0.0470 | 0.5192 |
| 3 | node_1 | 0.7928 | 0.6034 | 0.0517 | 0.6816 |
| 3 | node_2 | 0.4908 | 0.5451 | 0.0503 | 0.5827 |
| 4 | node_0 | 0.3607 | 0.4429 | 0.0449 | 0.4724 |
| 4 | node_1 | 0.6202 | 0.5509 | 0.0462 | 0.6015 |
| 4 | node_2 | 0.5350 | 0.4960 | 0.0528 | 0.5471 |

kd_loss drops sharply between round 1 and round 2 for every node (as in Run 1), then declines more gently through rounds 3-4; sup_loss/proto_loss decline steadily throughout.

### Interpretation (Run 2)

- **The structural fix worked exactly as intended**: no node has a true zero-sample class anymore, and node_1's `Black_rot` moved off its hard 0.0000 floor for the first time across both runs (+3.1 pts vs. both round 0 and its own control) — direct evidence that giving KD *something* (even 1 real local sample) to align a peer's stronger signal to is enough to escape the zero-sample dead end.
- **But the macro/worst-node picture got worse, not better.** Macro gain -2.4 pts (vs. Run 1's -0.8), worst-node gain -10.7 pts (vs. Run 1's 0.0). node_0's `Cedar_apple_rust` (-14.3 pts vs control) and node_1's `Cedar_apple_rust`/`healthy` (-19.3/-25.0 pts vs control) are the main drags — classes each node had *decent* local exposure to and the control arm handled them noticeably better than the collective arm did.
- **node_1 is still the weakest node, just in a different way.** Instead of Run 1's flat double-collapse (both arms identical, always-Apple_scab), Run 2's node_1 has real round-to-round volatility and a collective arm that underperforms its own control in 3 of 4 rounds — the KD pull is doing *something* different now (not degenerate), but it is not obviously helping this specific node's macro disease accuracy at this budget.
- **The clear per-class winners this run are node_2's `Black_rot` (+9.9 pts vs control) and `Cedar_apple_rust` (+25.5 pts vs control)** — node_2 is the best-resourced, most-balanced node, and continues to benefit from collaboration the way Run 1's node_2 partially did.

## Run 3 — alpha=0.7, epochs=20, rounds=4 x 4, `Black_rot` excluded

Ran via:
```
python -m src.validation.run_apple_pipeline --epochs 20 --output-dir outputs/validation/apple_mesh_a0.7
python -m src.validation.run_apple_knowledge_transfer --rounds 4 --distill-epochs 4 --output-dir outputs/validation/apple_mesh_a0.7
```

### Why this configuration

User-requested variant: keep Run 2's alpha (0.7) and round/epoch shape (4 rounds x 4 distill-epochs), but (a) go back to Run 1's deeper 20-epoch Stage 1 instead of Run 2's 4 epochs, and (b) **drop `Black_rot` from the label space entirely** — a 3-class problem (`Apple_scab`, `Cedar_apple_rust`, `healthy`) instead of 4. This is a **different, not stricter, isolation of variables** than the diagnosis doc's original recommendation (`docs/apple_kt_diagnosis_and_next_run_tuning.md` recommended re-running alpha=0.7 at Run 1's *exact* budget — 2x10, not 4x4 — while *keeping* `Black_rot`, arguing the pooled 650 `Black_rot` images aren't actually scarce and removing the class discards the run's one unambiguous positive result). Both changes (epochs 4->20, and dropping `Black_rot`) were applied together in this run, so — consistent with this document's own recurring caution — the result below should not be read as isolating either variable's effect on its own.

New config knob added to support this: `apple_mesh.exclude_diseases` (list, `config.yaml`) — `build_apple_label_map()` is called with `APPLE_DISEASE_ORDER` filtered by this list, and `_items_from_paths()` (PlantDoc/PlantWild loaders) now skips any canonical name not in the resulting label map instead of raising `KeyError`, mirroring `load_plantvillage_apple_items`'s existing skip behavior. `PlantDoc` has no `Black_rot` folder for Apple at all; `PlantWild` v1 does (`Apple_Black_rot`) and is the one source actually affected by the new skip logic.

### Dirichlet split preview (computed before training, `scripts/preview_apple_dirichlet_split.py`)

| Node | Samples | Dominant class | Share | JS divergence from uniform | Classes present |
|---|---|---|---|---|---|
| node_0 | 648 | Apple_scab | 42.6% | 0.0079 | 3/3 |
| node_1 | 950 | Apple_scab | 60.6% | 0.0696 | 3/3 |
| node_2 | 1,642 | healthy | 84.7% | 0.2101 | 3/3 |

Full per-class counts:

| Class | node_0 | node_1 | node_2 | pooled |
|---|---|---|---|---|
| Apple_scab | 276 | 576 | 104 | 956 |
| Cedar_apple_rust | 206 | 262 | 147 | 615 |
| healthy | 166 | 112 | 1,391 | 1,669 |

Global test set: 853 images (down from Run 1/2's 1,028, since `Black_rot` test images are gone too). Probe set: 170 images. Every node has >=1 sample of every remaining class (same property Run 2 established for the 4-class case). Note this is a **fresh Dirichlet draw**, not a re-shaping of Run 1/2's split — removing a class changes the pool composition and the random draw sequence even at the same seed/alpha, so per-node sample counts differ from Run 2's 4-class split (e.g. node_2: 1,518 -> 1,642).

### Timing (from output file timestamps)

| Stage | Wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | 24m 21s | 3 nodes x 20 epochs, train -> export -> evaluate |
| Stage 2 (knowledge transfer) | 1h 2m 54s | round-0 baseline + 4 rounds x (3 nodes distill + 3 control nodes local-train x4 epochs + export + eval) |
| **Total** | **~1h 27m 15s** | |

Per-node Stage 1 breakdown: node_0 5m 47s, node_1 7m 58s, node_2 10m 36s — faster than Run 1's 20-epoch run (12-16 min/node) since each node's pool shrank once `Black_rot` images were removed. Per-round Stage 2 breakdown: round-0 baseline ~1m 11s, round 1 11m 30s, round 2 15m 27s, round 3 17m 24s, round 4 17m 22s.

### Round 0 baseline (shared global test set, 853 images)

| Node | Crop accuracy | Disease accuracy | Apple_scab | Cedar_apple_rust | healthy |
|---|---|---|---|---|---|
| node_0 | 1.000 | 0.8804 | 0.8577 | 0.8466 | 0.9078 |
| node_1 | 1.000 | 0.8746 | 0.8277 | 0.8589 | 0.9102 |
| node_2 | 1.000 | 0.8406 | 0.7228 | 0.8344 | 0.9173 |

All three Stage-1 baselines are markedly higher than any 4-class run (Run 1: 0.721/0.547/0.826, Run 2: 0.767/0.680/0.817) — a 3-class problem is simply easier, and no node has a near-collapsed class dragging its average down.

### Stage 2 — round-by-round trend (disease_accuracy, global test set)

| Round | node_0 collective | node_0 control | node_1 collective | node_1 control | node_2 collective | node_2 control |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.8804 | - | 0.8746 | - | 0.8406 | - |
| 1 | 0.7843 | 0.7808 | 0.5064 | 0.6600 | 0.5018 | 0.5885 |
| 2 | 0.7796 | 0.7831 | 0.6424 | 0.7163 | 0.7351 | 0.5803 |
| 3 | 0.7890 | 0.7597 | 0.7292 | 0.7808 | 0.7866 | 0.6624 |
| 4 (final) | 0.7409 | 0.6917 | 0.8171 | 0.8113 | 0.7796 | 0.7034 |

Per-round collective-minus-control (macro across the 3 nodes): round 1 = -6.7 pts, round 2 = +4.0 pts, round 3 = +3.4 pts, round 4 = **+4.9 pts**. **Round 4 is the first round across all three Apple runs where the collective arm beats the local-only control for every single node simultaneously** (+4.9/+0.6/+7.6 pts) — Run 1 never had all 3 positive in the same round, and Run 2's node_1 was negative in 3 of its 4 rounds.

As in both previous runs, every node still drops well below its Stage-1 round-0 baseline in **both** arms (e.g. node_0: 0.880 -> 0.741 collective / 0.692 control) — the per-round Adam-optimizer-reset issue flagged after Run 1 is still present and still unfixed. The positive result here is relative (collective vs. its own control), not absolute (collective vs. Stage 1).

### Final per-class collaboration gain (Round 0 -> Round 4)

**node_0**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8577 | 0.7528 | 0.3670 | -0.1049 | **+0.3858** | no |
| Cedar_apple_rust | 0.8466 | 0.3558 | 0.9755 | -0.4908 | **-0.6196** | no |
| healthy | 0.9078 | 0.8818 | 0.7872 | -0.0260 | +0.0946 | yes |

**node_1**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8277 | 0.8614 | 0.8876 | +0.0337 | -0.0262 | no |
| Cedar_apple_rust | 0.8589 | 0.6871 | 0.6319 | -0.1718 | +0.0552 | no |
| healthy | 0.9102 | 0.8392 | 0.8322 | -0.0709 | +0.0071 | yes |

**node_2**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.7228 | 0.4831 | 0.3521 | -0.2397 | +0.1311 | yes |
| Cedar_apple_rust | 0.8344 | 0.8405 | 0.5644 | +0.0061 | **+0.2761** | no |
| healthy | 0.9173 | 0.9433 | 0.9787 | +0.0260 | -0.0355 | no |

### Macro / worst-node gain (collective vs. local-only control, round 4)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.0000 | 0.0000 |
| disease_accuracy | **+0.0438** | **+0.0492** |

Both figures are positive for the first time across all three runs — see [Interpretation](#interpretation-run-3) for why this should still be read cautiously.

### Communication / energy per round

| Round | Bytes exchanged (this round) | Cumulative bytes | Cumulative compute energy (kWh) |
|---|---|---|---|
| 1 | 114,624 | 114,624 | 0.002674 |
| 2 | 114,624 | 229,248 | 0.006257 |
| 3 | 114,624 | 343,872 | 0.010334 |
| 4 | 114,624 | 458,496 | 0.014400 |

Bytes/round are lower than both Run 1 (139,288) and Run 2 (147,480) — fewer classes means smaller per-node prototype payloads (3 class-mean vectors instead of 4) and a smaller probe set (170 vs 205 images).

### Per-node distill loss per round

| Round | Node | kd_loss | sup_loss | proto_loss | total_loss |
|---|---|---|---|---|---|
| 1 | node_0 | 1.1599 | 0.7910 | **19.5263** | 16.2989 |
| 1 | node_1 | 1.4656 | 0.7159 | **17.1061** | 15.3102 |
| 1 | node_2 | 1.4393 | 0.3442 | 0.2161 | 0.6437 |
| 2 | node_0 | 0.3012 | 0.6444 | 0.4457 | 0.9246 |
| 2 | node_1 | 0.4586 | 0.6154 | 0.6614 | 1.1511 |
| 2 | node_2 | 0.7196 | 0.3487 | 0.6568 | 0.9784 |
| 3 | node_0 | 0.2721 | 0.5470 | 0.1161 | 0.5811 |
| 3 | node_1 | 0.3580 | 0.5253 | 0.1207 | 0.6017 |
| 3 | node_2 | 0.3644 | 0.2791 | 0.0817 | 0.3612 |
| 4 | node_0 | 0.2948 | 0.4618 | 0.0708 | 0.4827 |
| 4 | node_1 | 0.3473 | 0.4764 | 0.0712 | 0.5169 |
| 4 | node_2 | 0.4670 | 0.2532 | 0.0485 | 0.3174 |

Notable anomaly not seen in Run 1/2: node_0 and node_1's `proto_loss` spikes to 19.5/17.1 in round 1 only (vs. node_2's 0.22), pulling their `total_loss` to ~16 — this settles to the same ~0.1-0.7 range as node_2 by round 2 and behaves normally for the rest of the run. Consistent with the 20-epoch Stage-1 backbones for node_0/node_1 having settled into a very different embedding scale than node_2's before the first round's peer-prototype alignment kicks in; not investigated further here.

### Interpretation (Run 3)

- **This is the first Apple run with an unambiguous net-positive macro AND worst-node gain** (+4.4 / +4.9 pts), and the first round (round 4) where every node's collective arm beats its own control arm simultaneously. Per the diagnosis doc's original hypothesis, giving every node real (if thin) exposure to every class — here helped further by shrinking the problem to 3 classes — does appear to give KD more to work with than Run 1's true zero-sample-class deadlock.
- **The gain is asymmetric and partly earned by the control arm's own instability, not purely by the collective arm's strength.** node_0's positive contribution comes almost entirely from `Apple_scab` (+38.6 pts vs. control) — but that's because the **control** arm collapsed to 36.7% on `Apple_scab` (a class node_0 has ample local data for, 276 images) while the collective arm merely held steady at 75.3%. Symmetrically, node_2's biggest win, `Cedar_apple_rust` (+27.6 pts vs control), is control collapsing to 56.4% while collective held 84.1%. In both cases the collective arm isn't dramatically *better* than Stage 1 — it is more *stable* round-to-round than a local-only arm that keeps re-initializing Adam every round on a small, skewed shard.
- **This run also has the sharpest single negative data point across all three runs**: node_0's `Cedar_apple_rust` collapses to 35.6% in the collective arm while its control arm holds 97.5% (a -62-pt gap) — the opposite pattern from the `Apple_scab` case on the very same node. KD pressure pulled node_0 away from a class it already knew well locally, likely because peer consensus (node_1's shard is 60.6% `Apple_scab`-dominant) biased the aggregated signal toward `Apple_scab` at `Cedar_apple_rust`'s expense. This is masked by disease_accuracy's sample-count weighting (node_0's test set has more `Apple_scab` images than `Cedar_apple_rust` ones) rather than resolved.
- **Every node still drops well below its Stage-1 baseline in both arms** — the per-round Adam-reset issue (flagged after Run 1, still unaddressed in code) is present here too; this run's positive result is about the collective arm losing *less* ground than the control arm during that shared degradation, not about collaboration recovering Stage-1-level performance.
- **Two variables changed from Run 2 at once** (epochs 4->20, `Black_rot` dropped) **and one from Run 1** (epochs 20 kept, but rounds/epochs-per-round 2x10 -> 4x4, alpha 0.3->0.7, class count 4->3) — this result cannot be cleanly attributed to any single change. The most defensible reading is: *at this specific alpha/budget/class-count combination*, collaboration nets out positive; it does not by itself establish that dropping `Black_rot` (or raising Stage-1 epochs, specifically) is what flipped the sign from Run 2's negative result.

## Run 4 — alpha=0.3, epochs=20, rounds=4 x 4, `Black_rot` excluded

Ran via:
```
python -m src.validation.run_apple_pipeline --epochs 20 --output-dir outputs/validation/apple_mesh_a0.3
python -m src.validation.run_apple_knowledge_transfer --rounds 4 --distill-epochs 4 --output-dir outputs/validation/apple_mesh_a0.3
```

### Why this configuration

User-requested: repeat Run 3's exact class set (3 classes, `Black_rot` excluded) and epoch/round budget (20-epoch Stage 1, 4 rounds x 4 distill-epochs), changing **only** `dirichlet_alpha` back to 0.3. Unlike every prior pair of runs in this series, this is a **true single-variable isolation** — Run 3 and Run 4 are identical in every other respect (class set, Stage 1 epochs, Stage 2 rounds/epochs, aggregation method, model, data sources). `apple_mesh.exclude_diseases: ["Black_rot"]` was left unchanged in `config.yaml`; only `dirichlet_alpha` (0.7 -> 0.3) and `output_dir` (-> `apple_mesh_a0.3`) changed.

### Dirichlet split preview (computed before training, `scripts/preview_apple_dirichlet_split.py`)

| Node | Samples | Dominant class | Share | JS divergence from uniform | Classes present |
|---|---|---|---|---|---|
| node_0 | 1,024 | healthy | 56.7% | 0.0410 | 3/3 |
| node_1 | 1,278 | Apple_scab | 53.1% | 0.0290 | 3/3 |
| node_2 | 938 | healthy | 82.6% | 0.2052 | 3/3 |

Full per-class counts:

| Class | node_0 | node_1 | node_2 | pooled |
|---|---|---|---|---|
| Apple_scab | 241 | 678 | 37 | 956 |
| Cedar_apple_rust | 202 | 287 | 126 | 615 |
| healthy | 581 | 313 | 775 | 1,669 |

Same global test set (853 images) and probe set (170 images) as Run 3 (same class set, same dedup/test-carve logic) — only the per-node Dirichlet draw differs. Every node has >=1 sample of every class; **no zero-sample-class problem at alpha=0.3 on the 3-class pool**, unlike the original 4-class alpha=0.3 finding (Run 1) that motivated loosening alpha in the first place — that degenerate shard was a property of `Black_rot` being the smallest globally-pooled class (650 images) in the 4-class pool, not an inherent property of alpha=0.3 itself. node_0/node_1 are noticeably more balanced here (JS-div 0.03-0.04) than Run 3's node_0/node_1 (0.008-0.070); node_2 remains the most skewed node in both runs (~0.20-0.21).

### Timing (from output file timestamps)

| Stage | Wall-clock | What ran |
|---|---|---|
| Stage 1 (per-node training) | 25m 36s | 3 nodes x 20 epochs, train -> export -> evaluate |
| Stage 2 (knowledge transfer) | 1h 19m 59s | round-0 baseline + 4 rounds x (3 nodes distill + 3 control nodes local-train x4 epochs + export + eval) |
| **Total** | **~1h 45m 35s** | |

Per-node Stage 1 breakdown: node_0 8m 10s, node_1 10m 34s, node_2 6m 52s — roughly proportional to this run's sample counts (1,024/1,278/938), which are more evenly spread across nodes than Run 3's (648/950/1,642), making this run's Stage 1 slightly slower overall despite an identical pooled total (3,240 training images in both runs). Per-round Stage 2 breakdown: round-0 baseline ~1m 10s, round 1 13m 32s, round 2 22m 2s, round 3 21m 32s, round 4 21m 43s — noticeably slower per round than Run 3's (11-17 min), consistent with node_1 here (1,278 images, the largest node) taking longer per local-train/distill pass than any single node in Run 3.

### Round 0 baseline (shared global test set, 853 images)

| Node | Crop accuracy | Disease accuracy | Apple_scab | Cedar_apple_rust | healthy |
|---|---|---|---|---|---|
| node_0 | 1.000 | 0.8828 | 0.7978 | 0.8650 | 0.9433 |
| node_1 | 1.000 | 0.8898 | 0.8464 | 0.8712 | 0.9243 |
| node_2 | 1.000 | 0.8124 | 0.8015 | 0.6687 | 0.8747 |

All three baselines are close to Run 3's (0.880/0.875/0.841) — expected, since Stage 1 epoch budget and class set are identical; the milder per-node skew here (vs. Run 3) doesn't materially change round-0 baseline quality.

### Stage 2 — round-by-round trend (disease_accuracy, global test set)

| Round | node_0 collective | node_0 control | node_1 collective | node_1 control | node_2 collective | node_2 control |
|---|---|---|---|---|---|---|
| 0 (baseline) | 0.8828 | - | 0.8898 | - | 0.8124 | - |
| 1 | 0.7948 | 0.8206 | 0.8312 | 0.5955 | 0.5780 | 0.5909 |
| 2 | 0.7808 | 0.7667 | 0.8359 | 0.7960 | 0.6143 | 0.6166 |
| 3 | 0.8242 | 0.7902 | 0.8218 | 0.8453 | 0.6729 | 0.6377 |
| 4 (final) | 0.8277 | 0.8113 | 0.8429 | 0.8535 | 0.6084 | 0.6190 |

Per-round collective-minus-control (macro across the 3 nodes): round 1 = **+6.6 pts** (driven almost entirely by node_1's control arm collapsing to 0.5955 that round), round 2 = +1.7 pts, round 3 = +1.5 pts, round 4 = **-0.2 pts**. Unlike Run 3, where the sign consolidated to positive-for-every-node by round 4, this run's early positive macro trend (rounds 1-3) **erodes back toward flat by round 4** as the control arms catch up — the opposite trajectory from Run 3's.

### Final per-class collaboration gain (Round 0 -> Round 4)

**node_0**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.7978 | 0.6554 | 0.6667 | -0.1423 | -0.0112 | no |
| Cedar_apple_rust | 0.8650 | 0.8957 | 0.6810 | +0.0307 | **+0.2147** | yes |
| healthy | 0.9433 | 0.9102 | 0.9527 | -0.0331 | -0.0426 | no |

**node_1**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8464 | 0.7640 | 0.7528 | -0.0824 | +0.0112 | no |
| Cedar_apple_rust | 0.8712 | 0.8098 | 0.8773 | -0.0613 | -0.0675 | yes |
| healthy | 0.9243 | 0.9054 | 0.9078 | -0.0189 | -0.0024 | no |

**node_2**

| Class | Round 0 | R4 collective | R4 control | Gain vs R0 | Gain vs control | Low-rep? |
|---|---|---|---|---|---|---|
| Apple_scab | 0.8015 | 0.0375 | **0.0000** | -0.7640 | +0.0375 | yes (37 imgs) |
| Cedar_apple_rust | 0.6687 | 0.5399 | 0.7607 | -0.1288 | **-0.2209** | no |
| healthy | 0.8747 | 0.9953 | 0.9551 | +0.1206 | +0.0402 | no |

Notable: node_2's `Apple_scab` (37 pooled images, its scarcest class) **collapses in both arms** by round 4 — control hits an exact 0.0000 floor, collective barely survives at 3.75%. This is the same near-zero-sample near-collapse pattern documented in Run 1's `Black_rot` case, just at a much smaller absolute scale (a thin-but-real class rather than a true zero-sample one); KD gave a marginal, not meaningful, rescue here.

### Macro / worst-node gain (collective vs. local-only control, round 4)

| Metric | Macro gain | Worst-node gain |
|---|---|---|
| crop_accuracy | 0.0000 | 0.0000 |
| disease_accuracy | **-0.0016** | **-0.0106** |

Essentially flat — statistically indistinguishable from zero given the noise already documented in this series, and a stark contrast to Run 3's clearly positive +0.0438 / +0.0492 at the identical class set and epoch/round budget.

### Communication / energy per round

| Round | Bytes exchanged (this round) | Cumulative bytes | Cumulative compute energy (kWh) |
|---|---|---|---|
| 1 | 114,624 | 114,624 | 0.003155 |
| 2 | 114,624 | 229,248 | 0.008301 |
| 3 | 114,624 | 343,872 | 0.013363 |
| 4 | 114,624 | 458,496 | 0.018456 |

Bytes/round are identical to Run 3 (114,624) — same class count and probe-set size, so payload size doesn't depend on alpha. Cumulative compute energy is higher than Run 3's (0.0185 vs 0.0144 kWh) purely because this run's rounds took longer wall-clock (node_1's larger shard), not because of anything alpha-specific.

### Per-node distill loss per round

| Round | Node | kd_loss | sup_loss | proto_loss | total_loss |
|---|---|---|---|---|---|
| 1 | node_0 | 0.8800 | 0.5670 | 0.1703 | 0.7578 |
| 1 | node_1 | 0.8357 | 0.5839 | 0.1819 | 0.7741 |
| 1 | node_2 | 1.9959 | 0.3933 | 0.1915 | 0.8039 |
| 2 | node_0 | 0.2302 | 0.4823 | 0.0903 | 0.5234 |
| 2 | node_1 | 0.3178 | 0.5103 | 0.0957 | 0.5717 |
| 2 | node_2 | 0.6541 | 0.2984 | 0.0621 | 0.4061 |
| 3 | node_0 | 0.2535 | 0.4288 | 0.0524 | 0.4485 |
| 3 | node_1 | 0.4300 | 0.4679 | 0.0638 | 0.5197 |
| 3 | node_2 | 0.5144 | 0.2793 | 0.0452 | 0.3540 |
| 4 | node_0 | 0.2172 | 0.3928 | 0.0391 | 0.4011 |
| 4 | node_1 | 0.2624 | 0.4067 | 0.0479 | 0.4318 |
| 4 | node_2 | 0.3664 | 0.2595 | 0.0328 | 0.3038 |

No round-1 `proto_loss` spike this time (unlike Run 3's 19.5/17.1 for node_0/node_1) — all three nodes' losses are in the same well-behaved range from round 1 onward. This supports the earlier guess that Run 3's spike was specific to that run's Stage-1-converged embedding scale, not a general artifact of the 20-epoch/3-class/4x4 configuration.

### Interpretation (Run 4)

- **This is the cleanest single-variable comparison in the series so far, and it points at alpha, not class count, as the likely driver of Run 3's positive result.** Everything except `dirichlet_alpha` (0.7 -> 0.3) is identical to Run 3, and the result reverted to essentially flat/null (macro -0.16 pts, worst-node -1.06 pts) — matching Run 1 and Run 2's pattern rather than Run 3's positive one.
- **The round-by-round trajectory is the opposite shape from Run 3's.** Run 3 started slightly negative (round 1) and consolidated to positive by round 4. Run 4 starts positive (round 1, +6.6 pts macro, mostly because node_1's *control* arm happened to collapse that round) and **erodes back to flat** by round 4 as the control arms catch up. Neither run shows a stable, monotonic trend — round-to-round volatility from the per-round Adam-reset issue remains the dominant noise source in both.
- **The one class-level near-failure in this run is node_2's `Apple_scab`** (37 pooled images) — both arms are near-zero by round 4 (control exactly 0.0, collective 3.75%), the same near-zero-sample pattern seen with `Black_rot` in Run 1, just smaller in magnitude. Milder alpha (0.7, Run 3) didn't need to solve this — Run 3's thinnest class (node_1's `Black_rot`... no, node_0's `Cedar_apple_rust` at worst — see Run 3's table) never fully collapsed the way this run's `Apple_scab` did, suggesting Run 4's *specific* draw concentrated `Apple_scab` scarcity onto node_2 harder than Run 3's draw concentrated any single class onto any single node.
- **No round-1 proto_loss anomaly this time**, unlike Run 3 — reinforces that anomaly as run-specific rather than a property of the general 3-class/20-epoch/4x4 configuration.
- **Bottom line for the thesis narrative:** across this 4-run series, the two runs sharing alpha=0.7 (Run 2, Run 3) both had richer round-to-round dynamics and (once the epoch/class confounds are controlled for, as in Run 3 vs Run 4) the only genuinely clean positive result; the two runs at alpha=0.3 (Run 1, Run 4) both came back flat/negative regardless of class count or epoch depth. This is the first evidence in the series consistent enough to tentatively attribute to alpha itself, though per this report's standing caution, it is still one seed per configuration — repeating Run 3 and Run 4 at a second seed is the natural next step before treating "alpha=0.7 collaborates better than alpha=0.3" as a settled finding.

## Comparison: Run 1 vs Run 2 vs Run 3 vs Run 4

| Metric | Run 1 (a=0.3, 4cls, 2x10) | Run 2 (a=0.7, 4cls, 4x4@4ep) | Run 3 (a=0.7, 3cls, 4x4@20ep) | Run 4 (a=0.3, 3cls, 4x4@20ep) |
|---|---|---|---|---|
| node_0 gain (final round) | -0.19 pts | -0.78 pts | **+4.92 pts** | +1.64 pts |
| node_1 gain | 0.00 pts | -10.89 pts | +0.58 pts | -1.06 pts |
| node_2 gain | -2.24 pts | +4.37 pts | **+7.62 pts** | -1.06 pts |
| Macro gain (disease_accuracy) | -0.81 pts | -2.43 pts | **+4.38 pts** | -0.16 pts |
| Worst-node gain (disease_accuracy) | 0.00 pts | -10.70 pts | **+4.92 pts** | -1.06 pts |
| Any node with a true 0-sample class? | Yes (node_1: Black_rot) | No | No | No |
| Stage 1 round-0 baseline disease_accuracy, node_0/1/2 | 0.721 / 0.547 / 0.826 | 0.767 / 0.680 / 0.817 | 0.880 / 0.875 / 0.841 | 0.883 / 0.890 / 0.812 |
| Cumulative bytes exchanged | 278,576 | 589,920 | 458,496 | 458,496 |
| Cumulative compute energy (kWh) | 0.024555 | 0.015833 | 0.014400 | 0.018456 |
| Total wall-clock | ~2h 20m | ~1h 15m | ~1h 27m | ~1h 46m |

**Run 3 vs Run 4 is the only pair in this series that isolates a single variable** (alpha, 0.7 vs 0.3 — everything else identical). It shows Run 3's positive result did not survive reverting to alpha=0.3, which is the strongest evidence so far that skew severity (not the class-count change, not epoch depth on its own) is what separates this series' one positive result from its three flat/negative ones.

## Tuning recommendation for the next run

1. **Replicate Run 3 vs Run 4 at a second Dirichlet seed** before treating "alpha=0.7 collaborates better than alpha=0.3 on this 3-class Apple problem" as a settled finding — it is currently one seed per alpha value, and this report has repeatedly cautioned against over-indexing on single-seed results.
2. **Still worth running: Run 3's configuration (alpha=0.7, epochs=20, 4x4) with `Black_rot` restored (4 classes)** — this isolates the class-count variable directly (Run 3 vs. this hypothetical Run 5, both alpha=0.7/epochs=20/4x4), the one variable Run 3 vs Run 4 didn't touch. Between this and (1), (1) is the higher-priority next step since it directly tests the finding this document now leads with.
3. **Address the per-round optimizer reset directly** — all four runs now show every node dropping well below its Stage-1 baseline in both arms at round 1, and both Run 3's positive result and Run 4's erosion-to-flat trajectory are shaped by this instability rather than by collaboration itself. This is the same code-level fix flagged after Run 1 (`_load_apple_node_from_checkpoint` / `node.local_train` currently re-initialize Adam state every round) and remains unimplemented across all four runs.
4. **The `Black_rot`-removal decision should still be treated as a scope decision, not just a tuning knob.** `docs/apple_kt_diagnosis_and_next_run_tuning.md` (§5) argued against removing it. Run 3/Run 4 together show the 3-class problem's own outcome now hinges more on alpha than on the class-count change itself — worth factoring into whether the thesis reports the 3-class or 4-class Apple problem going forward.
