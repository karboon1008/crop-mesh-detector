# Step-by-Step Execution Guide

Exact commands, in order, with what you should see after each one. This
covers what's implemented in the codebase today (data → train → predict →
export to ONNX → Raspberry Pi). For what each script actually does
internally, see [docs/codebase_guide.md](codebase_guide.md); for the full Pi
hardware walkthrough, see
[docs/raspberry_pi_deployment.md](raspberry_pi_deployment.md).

All commands below assume your terminal's working directory is the repo
root (`crop-mesh-detector/`).

---

## Step 0 — Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Expected result:** no errors; `pip` finishes installing `torch`,
`torchvision`, `timm`, `codecarbon`, `onnx`, `onnxruntime`,
`tensorflow-datasets`, `pytest`, etc. Takes a few minutes — `torch` and
`tensorflow` are large downloads. Verify with:

```bash
python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```
which should print two version strings with no `ImportError`.

---

## Step 1 — Download the dataset

```bash
python scripts/download_plantvillage.py
```

**Expected result (first run):**
```
Loading 'plant_village' via TensorFlow Datasets (downloads on first run)...
Writing images: 100%|██████████████████| 54305/54305 [xx:xx<00:00, ... it/s]
Done. 54305 images across 38 classes written to <repo>/data/PlantVillage.
```
Downloads ~800MB (cached under `~/tensorflow_datasets/`) then re-encodes
every image as a `.jpg` under `data/PlantVillage/<Crop>___<Disease>/`. Takes
several minutes. Class/image counts may differ slightly by TFDS version, but
you should end up with `data/PlantVillage/` containing ~38 subfolders full of
`.jpg` files.

**Expected result (already downloaded):**
```
<repo>/data/PlantVillage already has data — nothing to do.
```

---

## Step 2 — (Optional, recommended) Run the test suite

```bash
pytest tests/ -v
```

**Expected result:**
```
tests/test_pipeline.py::test_label_parsing PASSED
tests/test_pipeline.py::test_partition_by_crop_is_disjoint_and_complete PASSED
tests/test_pipeline.py::test_partition_manual_assigns_named_crops_to_named_nodes PASSED
tests/test_pipeline.py::test_partition_manual_rejects_incomplete_crop_assignment PASSED
tests/test_pipeline.py::test_model_forward_shapes[mobilenet_v3_small] PASSED
tests/test_pipeline.py::test_model_forward_shapes[efficientnet_lite0] PASSED
tests/test_pipeline.py::test_model_forward_shapes[mobilevit_xxs] PASSED
tests/test_pipeline.py::test_end_to_end_mesh_round_beats_no_exchange_smoke PASSED
tests/test_pipeline.py::test_energy_and_communication_tracking PASSED

9 passed in XX.XXs
```
Runs entirely on synthetic (randomly generated) images — doesn't need Step 1
to have finished, and doesn't need a GPU. If any of these fail, don't move on
to Step 3 — fix the environment first.

---

## Step 3 — Train (the slow step)

**All 3 architectures in one run:**
```bash
python -m src.train --config config.yaml
```

**Expected console output (repeats per architecture):**
```
Loaded 54305 images, 14 crop classes, 21 disease classes.

=== Architecture: mobilenet_v3_small ===
-- baseline (local-only) --
-- mesh (prototype + logit exchange) --
  round 0: 187392 bytes exchanged
  round 1: 187392 bytes exchanged
  round 2: 187392 bytes exchanged
  round 3: 187392 bytes exchanged
  round 4: 187392 bytes exchanged
macro_gain: {'crop_accuracy': 0.0421, 'disease_accuracy': 0.0187}
worst_node_gain: {'crop_accuracy': 0.0103, 'disease_accuracy': -0.0056}

=== Architecture: efficientnet_lite0 ===
...

=== Architecture: mobilevit_xxs ===
...

Done. Results and sustainability report written to outputs/ (now covering 3 architecture(s): ['mobilenet_v3_small', 'efficientnet_lite0', 'mobilevit_xxs'])
```
The exact numbers (image/class counts, bytes, gains) will differ on your
machine/dataset version — the point is the *shape* of the output should
match this.

**Expected files after this finishes**, all under `outputs/`:
```
outputs/
├── checkpoints/
│   ├── classes.json
│   ├── mobilenet_v3_small/{node_0,node_1,node_2}.pt
│   ├── efficientnet_lite0/{node_0,node_1,node_2}.pt
│   └── mobilevit_xxs/{node_0,node_1,node_2}.pt
├── results_mobilenet_v3_small.json
├── results_efficientnet_lite0.json
├── results_mobilevit_xxs.json
├── results_summary.json
├── round_logs_mobilenet_v3_small.json   (+ the other 2 architectures)
├── run_state.json
├── sustainability_report.json
├── sustainability_report.md
└── emissions.csv                          (only if energy.track_with_codecarbon: true and CodeCarbon works on your machine)
```

**How long this takes:** depends entirely on your CPU/GPU and `config.yaml`'s
`training` section (rounds, epochs, batch size) — treat this as the step to
run overnight or on a GPU machine/Colab, not something to wait on
interactively.

**Splitting across sessions instead** (e.g. Colab free-tier GPU quota):
```bash
python -m src.train --config config.yaml --arch mobilenet_v3_small --fresh
python -m src.train --config config.yaml --arch efficientnet_lite0
python -m src.train --config config.yaml --arch mobilevit_xxs
```
Each command's console output looks like a single-architecture slice of the
block above. `--fresh` on the first call wipes any previous
`results_summary.json`/`run_state.json`; the next two calls merge into it
instead of overwriting — check `results_summary.json` after the third
command and you should see all 3 architecture keys present.

---

## Step 4 — (Optional) Batch inference on a folder of images

```bash
python -m src.predict --folder data/PlantVillage/Tomato___healthy --checkpoints-dir outputs/checkpoints
```

**Expected result:**
```
Auto-selected arch=efficientnet_lite0 node=node_1 (avg test accuracy 0.8734)
Found 1591 images (labeled by folder name).
Wrote 1591 predictions to outputs/predictions.csv
crop_accuracy: 0.9899  disease_accuracy: 0.9718
```
(The "Auto-selected..." line only appears if you *don't* pass `--arch`/
`--node` yourself.) `outputs/predictions.csv` will contain one row per image
with predicted crop/disease + confidence, and — because this example folder
is named like a PlantVillage class — `true_crop`/`true_disease`/
`crop_correct`/`disease_correct` columns too.

---

## Step 5 — Export to ONNX for the Raspberry Pi

```bash
pip install onnx onnxruntime      # already in requirements.txt
python scripts/export_for_pi.py --all
```

**Expected result:**
```
Exporting all 3 architecture(s): ['mobilenet_v3_small', 'efficientnet_lite0', 'mobilevit_xxs']

=== mobilenet_v3_small (best node: node_1, avg test accuracy 0.8532) ===
Exporting mobilenet_v3_small/node_1 to ONNX...
Quantizing to int8...
Checking PyTorch-vs-ONNX prediction parity on 8 samples from data/PlantVillage...
Parity OK: all 8 sampled predictions match.
Bundle ready at outputs/pi_export/mobilenet_v3_small/ (model.onnx + manifest.json).

=== efficientnet_lite0 (best node: node_1, avg test accuracy 0.8734) ===
...

=== mobilevit_xxs (best node: node_0, avg test accuracy 0.8410) ===
...

All exports done:
  mobilenet_v3_small   node=node_1   avg_accuracy=0.8532  ->  outputs/pi_export/mobilenet_v3_small/
  efficientnet_lite0   node=node_1   avg_accuracy=0.8734  ->  outputs/pi_export/efficientnet_lite0/
  mobilevit_xxs        node=node_0   avg_accuracy=0.8410  ->  outputs/pi_export/mobilevit_xxs/

Copy whichever bundle(s) you want to the Pi — see docs/raspberry_pi_deployment.md.
```

**Expected files:**
```
outputs/pi_export/
├── mobilenet_v3_small/{model.onnx, manifest.json}
├── efficientnet_lite0/{model.onnx, manifest.json}
└── mobilevit_xxs/{model.onnx, manifest.json}
```
If you see a `WARNING: N/8 samples disagree between PyTorch and the
quantized ONNX model` line instead of `Parity OK`, the export still
succeeded, but that architecture/node's quantized model doesn't fully agree
with the original — worth trying a different `--node` for that architecture
before deploying it.

**Single-model instead of all three:**
```bash
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0
```
**Expected result:** same per-architecture block as above, but only one, and
the final line reads `Done. Copy the whole outputs/pi_export/ folder to the
Pi.` (no per-architecture subfolders this time — everything lands directly
in `outputs/pi_export/`).

---

## Step 6 — Deploy to the Raspberry Pi

This step happens *on the Pi*, not your dev machine. Full walkthrough
(flashing the OS, camera setup, systemd service) is in
[docs/raspberry_pi_deployment.md](raspberry_pi_deployment.md). Short version:

```bash
# on the Pi, after copying pi/requirements-pi.txt and a pi_export/ folder over
python3 -m venv .venv && source .venv/bin/activate
pip install -r pi/requirements-pi.txt

python inference_service.py --model-dir pi_export --camera file --image test.jpg --once
```

**Expected result (smoke test, no live camera):**
```
Loaded efficientnet_lite0/node_1 — 14 crop classes, 21 disease classes.
['2026-08-13T10:15:32.481203+00:00', 'Tomato', 0.9821, 'Bacterial_spot', 0.7734, 42.6]
```
That printed row is `[timestamp, predicted_crop, crop_confidence,
predicted_disease, disease_confidence, latency_ms]` — the same row also gets
appended to `predictions_log.csv`. `latency_ms` here is your real per-image
inference time on that Pi's CPU.

**Live camera** (drop `--once`, swap `--camera file --image ...` for
`--camera opencv` or `--camera picamera2`): runs forever, printing/logging
one row every `--interval` seconds until you stop it (Ctrl+C).

---

## Quick troubleshooting map

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: PlantVillage data not found at data/PlantVillage` | Run Step 1 first, or check `data.root` in `config.yaml` |
| `pytest` fails before you've even run Step 1 | Environment/dependency issue — fix before training, don't skip ahead |
| `src.train` runs but `emissions.csv` never appears | `energy.track_with_codecarbon` is `false` in `config.yaml`, or CodeCarbon couldn't access hardware counters — the wall-clock proxy still ran, just without that file |
| `export_for_pi.py` says a `data/PlantVillage` parity check was skipped | That's fine — it just means `data/PlantVillage` isn't present on the machine you're exporting from (e.g. exporting from a laptop that never ran Step 1); export itself still succeeded |
| `KeyError` / `ValueError` reading `results_summary.json` in Step 4/5 | Step 3 hasn't produced results yet for the architecture you're asking for — check `outputs/results_summary.json`'s keys |
