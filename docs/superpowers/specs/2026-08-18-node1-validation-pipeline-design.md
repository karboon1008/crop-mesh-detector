# Node-1 Validation Pipeline (train → test → ONNX → report) Design

Date: 2026-08-18
Status: Approved

## Purpose

Investigate why training/test accuracy (80-99%) does not carry over to
real-world inference, and build a small, self-contained pipeline that lets
us test fixes cheaply on one node / one architecture before rolling them out
to the full sweep (`src/train.py`, all 3 nodes, all 3 architectures).

Scope for this pass: **node_1 only** (`Corn, Potato, Soybean, Strawberry,
Squash`, per `config.yaml`'s `manual_node_crops`), **mobilenet_v3_small
only**.

## Root cause established prior to this design

A prior investigation (this session) traced the full train → inference
pipeline (`src/data/plantvillage.py`, `src/predict.py`,
`apps/crop_disease_detection/inference.py`, `pi/inference_service.py`,
`scripts/export_for_pi.py`) and found **no preprocessing bug** — resize,
color space, normalization, tensor layout, class-index mapping, and
softmax are all consistent everywhere. The user confirmed real-world
failures show **genuine confusion within the correct crop** (wrong disease,
right crop), which points at:

1. **Domain gap** — PlantVillage is a lab dataset (uniform background,
   single leaf, studio lighting); the model has never seen the variation a
   laptop/phone photo introduces.
2. **Zero data augmentation** in training (`src/data/plantvillage.py:120-126`
   — `Resize -> ToTensor -> Normalize` only).
3. **Optimistic evaluation methodology** — test accuracy is measured on a
   held-out split of the *same* lab-condition pool, via a plain random
   index split (`train_test_split_indices`,
   `src/data/plantvillage.py:338-345`) with no dedup against PlantVillage's
   known near-duplicate images, and each node's number only ever reflects
   its own narrow, fixed crop subset (`config.yaml:19-22`).
4. Contributing factors found while reviewing the training internals for
   this design: `config.yaml:37` sets `baseline_epochs: 1` (comment: "TEMP:
   reduced from 10 for smoke run"); `training.lr` is a single fixed value
   with no weight decay or LR schedule (`config.yaml:39-40`); disease loss
   is unweighted cross-entropy despite documented class imbalance
   (`src/federated/node.py:71-73`).

This pipeline exists to test remediation of (2) and (3) — plus the
above training-recipe issues — on a small scope before touching the main
training code.

## Constraints / decisions (from user)

1. **Do not modify** `src/data/plantvillage.py`, `src/train.py`,
   `src/federated/node.py`, or `src/models/factory.py`. Reuse their public
   functions/classes (`load_full_dataset`, `partition_nodes`, `build_model`,
   `Node`, etc.) by importing them; implement anything new (augmentation,
   dedup split, improved training recipe) in new files under
   `src/validation/`.
2. Node scope = node_1's crops as already defined in `config.yaml`'s
   `manual_node_crops`, obtained via the existing
   `partition_nodes(strategy="manual")` — same definition of "node_1" the
   real mesh pipeline uses, so results stay comparable later.
3. Model scope = `mobilenet_v3_small` only for this pass.
4. No requirement to reuse the Streamlit/Pi frontend inference code
   directly — the ONNX evaluation stage reimplements the (already-verified
   identical) preprocessing independently.
5. Output: a JSON report with per-image expected vs. predicted (crop +
   disease, with confidences) plus an aggregate summary.
6. `train_mobilenet.py` runs its own held-out evaluation (PyTorch, every
   epoch) for training-time signal / best-checkpoint selection; the ONNX
   stage re-evaluates the same held-out split through the exported model to
   produce the final report — same test split, two different jobs (fast
   per-epoch training signal vs. one-shot deployed-artifact correctness).

## Folder structure

```
src/validation/
  __init__.py
  node1_dataset.py    # node_1 scoping + dedup-aware split + augmented train transform
  train_mobilenet.py  # train mobilenet_v3_small with an improved recipe; per-epoch eval
  export_onnx.py       # export best checkpoint -> ONNX + manifest.json
  evaluate_onnx.py     # onnxruntime inference over held-out split -> report.json
  run_pipeline.py       # CLI entrypoint chaining all stages, --stage to run just one

outputs/validation/node_1_mobilenet_v3_small/
  checkpoint.pt         # best PyTorch checkpoint (by held-out disease_accuracy)
  classes.json           # crop_classes / disease_classes / image_size (node_1 scope)
  training_log.json      # per-epoch train_loss, test_crop_accuracy, test_disease_accuracy
  model.onnx              # exported model
  manifest.json           # mean/std/image_size/classes, mirrors scripts/export_for_pi.py's format
  report.json              # final per-image + summary evaluation report
```

## Data flow

1. **Load + scope** (`node1_dataset.py`): `load_full_dataset(cfg.data.root,
   cfg.data.image_size)` (reused, untouched), then
   `partition_nodes(dataset, remaining_idx, num_nodes=3,
   strategy="manual", ..., manual_node_crops=cfg.data.manual_node_crops)`
   (reused, untouched) → take `shards[1]` as node_1's index list. (The probe
   set carve-out is skipped — no mesh/distillation here, this is a
   local-only training run.)

2. **Dedup-aware split** (new code, `node1_dataset.py`): compute a 64-bit
   average-hash per image (grayscale, resize to 8x8, threshold against the
   mean pixel value; no new dependency — plain PIL/numpy). Union images
   whose hashes are within Hamming distance <= 5 into duplicate-groups
   (union-find). Split **at the group level** into train/test using
   `cfg.data.test_fraction` (0.15, reused from `config.yaml`), so
   near-duplicate PlantVillage shots can't land on both sides. This
   replaces `train_test_split_indices` only within this module — the
   original function in `src/data/plantvillage.py` is untouched.

3. **Two dataset instances, one root** (new code): build
   `PlantVillageDataset` twice against the same `data/PlantVillage` root —
   `train_ds` and `eval_ds`. Both are the unmodified class from
   `src/data/plantvillage.py`; after construction, `train_ds.transform` is
   **overwritten** (plain attribute assignment, no class edit) with an
   augmented pipeline:
   ```python
   transforms.Compose([
       transforms.Resize((image_size, image_size)),
       transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
       transforms.ToTensor(),
       transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
   ])
   ```
   `eval_ds.transform` is left as its original clean `Resize -> ToTensor ->
   Normalize` pipeline (needed so held-out accuracy reflects real deployed
   preprocessing, not augmented images). `Subset(train_ds, train_idx)` /
   `Subset(eval_ds, test_idx)` are built from the same dedup-aware split
   (both instances share identical `ImageFolder` ordering since they point
   at the same root).

4. **Train** (`train_mobilenet.py`): `build_model("mobilenet_v3_small",
   num_crop, num_disease, pretrained=True)` (reused, untouched). Improved
   recipe vs. current `config.yaml` defaults:
   - 15 epochs (vs. the current `baseline_epochs: 1` smoke-test setting),
     keeping the best-by-held-out-accuracy checkpoint rather than the last.
   - Adam with `weight_decay=1e-4` instead of none.
   - `CosineAnnealingLR` over the 15 epochs, starting from `lr=0.001`
     (same starting point as `config.yaml`, decayed instead of held fixed).
   - Disease-head cross-entropy weighted by inverse class frequency within
     node_1's training split (crop head left unweighted — its classes are
     roughly balanced by design already).
   - After every epoch: evaluate on `eval_ds`'s held-out subset (same
     accuracy computation pattern as `Node.evaluate()`), log
     `{epoch, train_loss, test_crop_accuracy, test_disease_accuracy}` to
     `training_log.json`, and keep the checkpoint with the best
     `test_disease_accuracy` seen so far.

5. **Export** (`export_onnx.py`): `torch.onnx.export` the best checkpoint,
   opset 17, static `(1, 3, image_size, image_size)` NCHW input, writing
   `manifest.json` with the same `mean`/`std`/`image_size`/`classes` shape
   used by `scripts/export_for_pi.py` and read by `pi/inference_service.py`.

6. **Evaluate the ONNX artifact** (`evaluate_onnx.py`): run
   `onnxruntime.InferenceSession` over every image in the held-out test
   split (same split used in step 4's final epoch), applying the *exact*
   preprocessing already verified identical to the frontend (`Resize ->
   RGB -> /255 -> normalize -> HWC->CHW -> batch dim -> float32`). For each
   image, softmax each head, take top-1 crop/disease + confidence, compare
   to expected labels, and write `report.json`.

## report.json shape

```json
{
  "summary": {
    "model": "mobilenet_v3_small",
    "node": "node_1",
    "num_test_images": 000,
    "crop_accuracy": 0.00,
    "disease_accuracy": 0.00,
    "per_class_accuracy": {
      "crop": {"Corn": 0.00, "Potato": 0.00, "...": 0.00},
      "disease": {"Corn___Common_rust": 0.00, "...": 0.00}
    },
    "top_confusions": [
      {"expected": "Corn___Common_rust", "predicted": "Corn___Northern_Leaf_Blight", "count": 0}
    ]
  },
  "results": [
    {
      "filename": "...",
      "expected_crop": "Corn", "expected_disease": "Common_rust",
      "predicted_crop": "Corn", "predicted_disease": "Northern_Leaf_Blight",
      "crop_confidence": 0.00, "disease_confidence": 0.00,
      "crop_correct": true, "disease_correct": false
    }
  ]
}
```

`top_confusions` is derived from `results` (grouped by `(expected_disease,
predicted_disease)` where `disease_correct == false`, sorted descending by
count, capped at e.g. top 10) — no separate tracking needed during
inference.

## Error handling

- Missing `data/PlantVillage` → reuse `load_full_dataset`'s existing
  `FileNotFoundError` (untouched).
- `run_pipeline.py` fails fast with a clear message if a later stage's
  required input file (checkpoint, manifest, ONNX file) is missing —
  e.g. running `--stage evaluate_onnx` before `--stage export_onnx` has run.
- `export_onnx.py` includes a PyTorch-vs-ONNX parity spot-check (same idea
  as `scripts/export_for_pi.py`'s `check_parity`, reimplemented locally):
  run a handful of held-out images through both the PyTorch checkpoint and
  the exported ONNX session, and print a warning if top-1 predictions
  disagree, so an export bug is caught before `evaluate_onnx.py` blames the
  training itself.

## Testing / verification

No formal automated test suite for this validation tooling — it's a
diagnostic pipeline, not shipped product code. Verification is running it
end to end (`python -m src.validation.run_pipeline`) against node_1 /
mobilenet_v3_small and confirming:
- `training_log.json` shows test accuracy trending sensibly across epochs
  (not just one data point as today).
- `report.json` is well-formed and its `summary.disease_accuracy` /
  `crop_accuracy` can be directly compared against the current
  `outputs/results_summary.json` node_1 baseline numbers to see whether
  augmentation + the dedup-aware split + the improved recipe move the
  needle (and in particular, whether the honest, deduped number is lower
  than today's 80-99%, which would confirm the leakage/optimism
  hypothesis).

## Out of scope (this pass)

- Editing `src/train.py` / `src/data/plantvillage.py` / `src/federated/
  node.py` / `src/models/factory.py` — reused as-is.
- node_0, node_2, or the other two architectures (`efficientnet_lite0`,
  `mobilevit_xxs`) — single node/architecture proof point first.
- Federated mesh / distillation training — local-only baseline-style
  training only.
- Reusing the Streamlit/Pi frontend code directly in the ONNX evaluation
  stage — preprocessing is reimplemented locally (already verified
  identical).
- Fine-tuning on real-world (non-lab) photos — a possible follow-up once
  this pipeline confirms how much of the gap augmentation + honest
  evaluation alone close.
