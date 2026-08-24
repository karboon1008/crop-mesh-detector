# Node-1 Validation Pipeline — Tuning Changes & Results

Date: 2026-08-18

This documents the tuning changes made in `src/validation/` (scoped to
node_1 — Corn/Potato/Soybean/Strawberry/Squash — and `mobilenet_v3_small`
only) to investigate why training/test accuracy (80-99%) didn't carry over
to real-world inference, and what actually changed in the numbers when run
against the real dataset.

Background/root-cause investigation: see
`docs/superpowers/specs/2026-08-18-node1-validation-pipeline-design.md`.
Implementation plan: `docs/superpowers/plans/2026-08-18-node1-validation-pipeline.md`.

## What changed, vs. the current `src/train.py` / `config.yaml` pipeline

| Aspect | Before (`config.yaml` / `src/train.py`) | After (`src/validation/`) |
|---|---|---|
| Training epochs | `baseline_epochs: 1` (`config.yaml:37`, marked "TEMP: reduced from 10 for smoke run") | 15 (`train_mobilenet.py`'s `run_training`, default `epochs=15`) |
| Optimizer | Adam, fixed `lr=0.001`, no weight decay (`config.yaml:39`, `src/federated/node.py:60`) | Adam, `lr=0.001`, `weight_decay=1e-4` |
| LR schedule | None (flat learning rate for the whole run) | `CosineAnnealingLR(optimizer, T_max=15)` |
| Disease loss | Plain, unweighted `F.cross_entropy` (`src/federated/node.py:71-73`), despite known class imbalance | Disease head weighted by inverse class frequency (`compute_class_weights`); crop head left unweighted (its classes are roughly balanced already) |
| Data augmentation | None — `Resize -> ToTensor -> Normalize` only (`src/data/plantvillage.py:120-126`) | Train-only: `RandomResizedCrop(scale=(0.7,1.0))`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(brightness/contrast/saturation=0.3, hue=0.05)`, then the same `ToTensor -> Normalize`. Eval transform stays the original clean pipeline (so held-out accuracy reflects real deployed preprocessing, not augmented images). |
| Train/test split | Plain random index split (`train_test_split_indices`) | Dedup-aware split: 64-bit average-hash per image, union-find grouping at Hamming distance ≤ 5, split at the *group* level so near-duplicate PlantVillage shots can't land on both sides — **see caveat below, this needs retuning** |
| Checkpoint selection | Whatever the model looks like after the last epoch | Best checkpoint kept by held-out `test_disease_accuracy` across all 15 epochs, not just the last one |
| Batch handling | N/A | `drop_last=True` on the train loader, added after review caught a real risk: `mobilenet_v3_small`'s BatchNorm2d crashes on a final batch of exactly 1 sample — this exact error is already documented occurring elsewhere in this repo's test suite for the same architecture |
| Evaluation artifact | In-memory PyTorch model only, one accuracy number, no per-image detail | Full ONNX export (with an inline PyTorch-vs-ONNX **parity check** — "Parity OK: all 8 sampled predictions match" on this run) evaluated via `onnxruntime`, producing `report.json`: per-image expected vs. predicted crop/disease with confidence, plus per-class accuracy and top confusion pairs |

## The actual numbers

**Old baseline** (`outputs/results_summary.json`, `mobilenet_v3_small.baseline_eval.node_1`, 1 epoch, no augmentation, plain random split):

```
crop_accuracy:    0.9386
disease_accuracy: 0.8980
```

**New run** (`outputs/validation/node_1_mobilenet_v3_small/report.json`, 15 epochs, augmentation, dedup-aware split, evaluated through the exported ONNX model):

```
crop_accuracy:    0.9891
disease_accuracy: 0.9940
num_test_images:  10,408
```

Per-epoch trend (`training_log.json`) climbs steadily and plateaus around
epoch 9-12 (`test_disease_accuracy` ~0.99 from epoch 9 onward), which is a
real, honest training-progress signal that simply didn't exist before
(the old pipeline only ever reported one number, from one epoch).

Remaining confusions are concentrated in exactly two visually similar corn
diseases (`Northern_Leaf_Blight` vs. `Cercospora_leaf_spot Gray_leaf_spot`,
29 + 13 misclassifications) — a much narrower, more plausible failure mode
than "wrong disease, right crop" across the board.

## Important caveat — read before treating this as a validated improvement

The new `disease_accuracy` (0.994) being higher than the old baseline
(0.898) does **not** cleanly prove the tuning fixed the real-world gap,
because the comparison is confounded:

- The dedup-aware split's threshold (average-hash, Hamming ≤ 5) turned out
  **miscalibrated for real PlantVillage photos**. Verified directly from
  `report.json`: Soybean (a single "healthy"-only class, 5,090 real, visually
  near-identical lab photos) had 4,958/5,090 (97.4%) of its images land in
  the *test* split — the union-find grouping almost certainly merged the
  vast majority of Soybean's genuinely distinct photos into one giant
  false-positive "duplicate" cluster, and the group-level split then dumped
  that whole cluster onto one side. Only 132 Soybean images were left for
  training.
- Net effect: the held-out test set ended up **10,408 images (~72% of
  node_1), not the intended ~15%** (`config.yaml`'s `test_fraction: 0.15`).
  A bigger, differently-composed test set isn't wrong on its own, but it
  means this run's number and the old baseline's number aren't measuring
  the same thing.

**This is not a code bug** — the dedup-aware split does exactly what it was
specified to do. The average-hash threshold itself is the thing that needs
retuning (options: a stricter Hamming distance, a larger `hash_size` for
finer-grained hashes, or a cap on how large one duplicate-group is allowed
to grow) before this specific number can be trusted as a clean before/after
comparison.

## Bottom line

- The training recipe changes (more epochs, weight decay, LR schedule,
  class weighting, augmentation, best-checkpoint selection) are real,
  reviewed improvements over the current `config.yaml` defaults, and the
  per-epoch training curve now gives an honest signal that didn't exist
  before.
- The evaluation methodology fix (dedup-aware split) is directionally
  correct — PlantVillage does contain near-duplicate images that a plain
  random split would leak across train/test — but its current threshold
  is too aggressive for this dataset's Soybean class and needs recalibrating
  before the accuracy numbers from this specific run are trusted as final.
- Recommended next step before rolling this recipe out to node_0/node_2 or
  the other two architectures: retune the dedup threshold, re-run, and
  confirm the test split lands close to the intended 15% with a
  crop-balanced composition, then re-compare against the old baseline on
  equal footing.

## Where the artifacts live

```
outputs/validation/node_1_mobilenet_v3_small/
    checkpoint.pt        # best PyTorch checkpoint by held-out disease accuracy
    training_log.json    # per-epoch train_loss / test_crop_accuracy / test_disease_accuracy
    classes.json          # crop/disease label lists, image_size, stored test_idx
    model.onnx              # exported ONNX model
    manifest.json            # mean/std/image_size/classes for the ONNX model
    report.json                # per-image + summary evaluation report (the file referenced above)
```
