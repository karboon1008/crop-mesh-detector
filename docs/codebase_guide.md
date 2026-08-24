# Codebase Guide

A map of this repository: what every folder is for, what each Python script
does, where the ONNX conversion actually happens, and the exact command
sequence to go from a fresh checkout to a running Raspberry Pi demo.

This is a **simulation** of a decentralised "mesh" of farms that each train
their own crop/disease classifier and only ever exchange small derived
knowledge (class-embedding prototypes + soft logits on a shared public probe
set) — never raw images, labels, gradients, or model weights. See the
[README](../README.md) for the conceptual background; this doc is about the
code itself.

---

## 1. Folder-by-folder

```
crop-mesh-detector/
├── config.yaml              # every tunable (data, model, training, energy) — see §2
├── requirements.txt          # dev-machine/Colab dependencies (training + ONNX export)
├── data/                     # PlantVillage images land here (git-ignored, only .gitkeep tracked)
├── notebooks/                # Colab notebook for running a training sweep on free-tier GPU
├── outputs/                  # everything a training/export run produces (git-ignored)
├── scripts/                  # one-off CLI utilities: dataset download, ONNX export
├── src/                       # the actual library code (importable as `src.*`)
│   ├── config.py              # YAML config loader
│   ├── data/                  # dataset loading + non-IID partitioning into simulated farms
│   ├── models/                # model factory (3 backbones, shared-backbone/two-head design)
│   ├── federated/              # the mesh itself: one node, aggregation rules, round orchestration
│   ├── energy/                 # compute + communication energy/CO2e accounting
│   ├── train.py                # main CLI entry point (baseline + mesh training)
│   ├── evaluate.py             # collaboration-gain metrics
│   ├── predict.py              # batch inference with a trained checkpoint
│   └── model_selection.py      # shared "which checkpoint is best?" logic
├── pi/                         # code that runs ON the Raspberry Pi itself
│   ├── inference_service.py     # camera -> ONNX model -> CSV log, no training stack needed
│   └── requirements-pi.txt      # Pi-only deps (onnxruntime, numpy, Pillow, no torch)
├── tests/
│   └── test_pipeline.py         # end-to-end smoke test on synthetic (fake) images
└── docs/
    ├── raspberry_pi_deployment.md  # step-by-step Pi deployment walkthrough
    └── codebase_guide.md            # this file
```

### `data/`
Empty except for `.gitkeep` until you run `scripts/download_plantvillage.py`,
which populates `data/PlantVillage/<Crop>___<Disease>/*.jpg`. Never committed
— it's a few hundred MB to a few GB of images.

### `notebooks/`
`crop_mesh_detector_colab.ipynb` — runs `src.train` on Google Colab's
free-tier GPU, one architecture at a time (see `--arch` in §3), so a full
3-architecture sweep can be split across several sessions without hitting
Colab's usage limits.

### `outputs/`
Everything training/export produces, all git-ignored except `.gitkeep`:
- `checkpoints/<arch>/<node_id>.pt` — trained PyTorch weights, one file per
  (architecture, simulated farm node)
- `checkpoints/classes.json` — crop/disease class name lists + image size,
  needed to rebuild a model and interpret its output indices
- `results_<arch>.json`, `results_summary.json` — accuracy, model size,
  collaboration gain, bytes exchanged
- `round_logs_<arch>.json` — per-round training/distillation losses
- `emissions.csv` — CodeCarbon's raw measurement log
- `run_state.json` — running totals across possibly-separate training
  invocations (so splitting a sweep across sessions still produces one
  combined sustainability report)
- `sustainability_report.json` / `.md` — the combined energy/CO2e report
- `pi_export/<arch>/{model.onnx, manifest.json}` — **this is where the ONNX
  models live** (see §4 below)

### `scripts/`
Standalone CLI utilities, run directly with `python scripts/<name>.py`
(not as `-m src.something` modules):
- `download_plantvillage.py` — fetches the dataset
- `export_for_pi.py` — **converts trained checkpoints to ONNX** (see §4)

### `pi/`
The *only* code meant to run on the Raspberry Pi. Deliberately has zero
dependency on `torch`/`torchvision`/`timm` — just `onnxruntime`, `numpy`,
`Pillow`, and optionally `opencv-python-headless` for a USB webcam
(`picamera2` ships with Raspberry Pi OS already).

### `tests/`
`pytest tests/ -v` generates a tiny synthetic image set on the fly (random
noise images under class-named folders) and runs the whole pipeline against
it — no PlantVillage download needed. Good smoke test before committing to a
real, slow training run.

---

## 2. `config.yaml` — the one file that controls everything

`src/config.py` loads it into a `Config` object with dotted-key access
(`cfg.get("training.rounds")`). Nothing else in `src/` hardcodes an
experiment parameter — to change node count, architectures, non-IID split
strategy, aggregation rule, or energy assumptions, edit this file, not code.

Key sections: `data` (dataset root, image size, node count, probe-set
fraction, non-IID strategy), `models` (which of the 3 architectures to run),
`training` (epochs, rounds, learning rates, loss weights), `federated`
(aggregation rule: trimmed-mean or Krum), `energy` (CodeCarbon on/off, grid
carbon intensity, per-byte radio energy assumptions), `output` (where results
land).

---

## 3. What each Python script does

### `src/config.py`
Thin YAML-to-dict wrapper (`Config.load()`, `Config.get("a.b.c", default)`).
No logic beyond that.

### `src/data/plantvillage.py`
- `_parse_crop_disease()` — splits a PlantVillage folder name like
  `Tomato___Bacterial_spot` into `("Tomato", "Bacterial_spot")`.
- `PlantVillageDataset` — wraps `torchvision.ImageFolder`; each sample
  returns `(image_tensor, crop_label, disease_label)` instead of a single
  label, since every image is trained against two heads.
- `carve_public_probe_set()` — splits off a small, identical-for-everyone
  slice used only for logit exchange, never for training.
- `partition_nodes()` — splits the rest into N non-IID shards using one of
  four strategies (`by_crop`, `by_disease`, `dirichlet`, `manual`) —
  simulating farms that each see a different regional mix of crops/diseases.
  `manual` (the current `config.yaml` default) assigns named crops to named
  nodes explicitly (e.g. an "orchard farm" node growing only
  Apple/Cherry/Peach/Blueberry/Raspberry).
- `train_test_split_indices()` — per-node train/test split.

### `src/models/factory.py`
- `ARCH_TO_TIMM` — maps the 3 config-friendly names
  (`mobilenet_v3_small`, `efficientnet_lite0`, `mobilevit_xxs`) to their
  `timm` model names.
- `MultiTaskNet` — one shared backbone + two `nn.Linear` heads (crop,
  disease). `forward(..., return_features=True)` also returns the pooled
  embedding, needed for prototype computation.
- `build_model()` — constructs a `MultiTaskNet` for a given architecture.
- `count_parameters()`, `model_size_mb()` — used for the size/accuracy
  comparison in results.
- `quantize_dynamic()` — a PyTorch-side INT8 dynamic quantization helper
  (linear heads only); this is a *demonstration* utility, not what's used
  for the actual Pi export (that's ONNX Runtime's quantizer — see §4).

### `src/federated/aggregation.py`
Pure math, no I/O: `aggregate_vectors()` implements trimmed-mean and Krum
over a list of same-shaped tensors from peers. `aggregate_prototypes()` and
`aggregate_logits()` are typed wrappers around it for the two kinds of
knowledge a node exchanges.

### `src/federated/node.py`
One simulated farm:
- `local_train()` — supervised training on the node's own private shard only.
- `compute_prototypes()` — mean feature-embedding per crop/disease class,
  computed from the node's own data.
- `compute_probe_logits()` — soft predictions on the shared public probe set.
- `compute_knowledge()` — bundles both into a `KnowledgePayload`
  (`size_bytes()` on that payload is what feeds the communication-energy
  estimate).
- `distill()` — trains towards the *peer consensus* (KL-divergence on probe
  logits + prototype-alignment MSE), never towards another node's raw data.
- `evaluate()` — crop/disease accuracy on the node's own held-out test split.

### `src/federated/mesh.py`
`MeshSimulator.run_round()` orchestrates one exchange round across every
node: local train → each node computes its `KnowledgePayload` → simulate a
fully-connected broadcast (every payload reaches every peer once — an upper
bound on real bandwidth) → each node aggregates what it *received* (robustly,
excluding its own payload) → each node distils towards that consensus →
evaluate. `run()` just calls `run_round()` N times.

### `src/energy/tracker.py`
- `ComputeEnergyTracker` — wraps CodeCarbon around a labelled block of code
  (`with tracker.track("label"): ...`); falls back to a wall-clock ×
  assumed-power proxy if CodeCarbon is disabled/unavailable.
- `CommunicationCostEstimator` — converts a byte count into estimated
  transmit energy/CO2e for Wi-Fi/cellular/LoRa, using `config.yaml`'s
  per-byte radio energy figures.
- `write_sustainability_report()` — combines both into
  `outputs/sustainability_report.json` + a human-readable `.md` narrative.

### `src/train.py` — main CLI entry point
For each configured architecture: trains an aligned-compute-budget
**baseline** (`run_baseline()`, no exchange at all) and the **mesh**
(`run_mesh()`, prototype+logit exchange over several rounds), computes the
collaboration gain, saves PyTorch checkpoints to
`outputs/checkpoints/<arch>/<node_id>.pt`, and writes
`outputs/results_<arch>.json` + merges into `outputs/results_summary.json`.
At the end, writes the sustainability report. Supports `--arch` to restrict
one run to a single architecture (results merge into any existing
`results_summary.json` rather than overwriting it) and `--fresh` to discard
prior results.

### `src/evaluate.py`
Pure metrics, no I/O: `macro_average()`, `worst_node()`, and
`compute_collaboration_gain()` (mesh accuracy − baseline accuracy, both as a
macro-average across nodes and a worst-node score, so a win can't hide behind
only helping already-strong nodes).

### `src/model_selection.py`
Shared logic used by both `src/predict.py` and `scripts/export_for_pi.py`:
`pick_best_arch_node()` / `pick_best_node_for_arch()` read
`outputs/results_summary.json` and return whichever (architecture, node)
scored highest on average crop/disease test accuracy. `list_architectures()`
lists every architecture present so far (useful when a sweep was split
across several separate `--arch` runs).

### `src/predict.py`
Batch inference with a **PyTorch** checkpoint (not the exported ONNX model —
for that, see `pi/inference_service.py`). Auto-picks the best-scoring
(architecture, node) unless `--arch`/`--node` are given. If the input folder
is laid out one-subfolder-per-class like PlantVillage itself, it parses true
labels from folder names and reports accuracy; otherwise it just writes
predictions to a CSV.
```
python -m src.predict --folder path/to/images --checkpoints-dir outputs/checkpoints
```

### `scripts/download_plantvillage.py`
Fetches PlantVillage via TensorFlow Datasets' `plant_village` catalog entry
(no Kaggle account needed) and re-encodes every image as a real `.jpg` under
`data/PlantVillage/<Crop>___<Disease>/`. Includes a fallback that swaps in a
direct S3 URL if Mendeley's own endpoint 403s (common on Colab/cloud IPs).
Run once, ahead of training; a no-op if `data/PlantVillage/` already has data.

### `scripts/export_for_pi.py` — **the ONNX conversion step**
See §4 — this is the file that actually produces `.onnx` files.

### `pi/inference_service.py`
Runs **on the Pi**. Loads `model.onnx` + `manifest.json` via
`onnxruntime.InferenceSession`, reads frames from one of three camera
backends (`FileSource` for smoke-testing with a static image, `OpenCVSource`
for a USB webcam, `Picamera2Source` for the official Camera Module),
preprocesses with the exact mean/std/image-size from the manifest, runs
inference, and appends each prediction (crop, disease, confidences, latency)
to `predictions_log.csv`. No PyTorch/timm/torchvision on this side at all.

### `tests/test_pipeline.py`
Builds a tiny synthetic PlantVillage-like folder tree (random-noise JPEGs)
and exercises: label parsing, both partition strategies used in CI (`by_crop`,
`manual`), a forward pass through all 3 architectures, one full mesh round
end-to-end, and the energy/communication tracker — all without needing the
real dataset or a GPU.

---

## 4. Where the ONNX conversion happens

**`scripts/export_for_pi.py`** is the entire ONNX pipeline, in one file.
Nothing in `src/` touches ONNX — this script is the boundary between the
training stack (PyTorch/timm) and the deployment stack (ONNX Runtime only).

Flow, per (architecture, node) being exported:

1. **Load a trained checkpoint.** Rebuilds the `MultiTaskNet` with
   `build_model()` and loads `outputs/checkpoints/<arch>/<node_id>.pt` — the
   file `src/train.py` saved after mesh training.
2. **Export to FP32 ONNX** (`export_onnx()` in `scripts/export_for_pi.py:38`):
   `torch.onnx.export(model, dummy_input, ..., dynamo=False, opset_version=17)`
   to a temporary `model_fp32.onnx`. `dynamo=False` deliberately forces
   PyTorch's older TorchScript-based exporter — the newer torch.export/dynamo
   exporter has shape-inference bugs with this model. Opset 17 is required
   because `mobilevit_xxs`'s attention op needs ≥14.
3. **Quantize to INT8** (`scripts/export_for_pi.py:107`): calls
   `onnxruntime.quantization.quantize_dynamic()`, restricted to
   `op_types_to_quantize=["MatMul", "Gemm"]` (i.e. only the linear heads —
   dynamically quantizing the conv backbone produces `ConvInteger` ops many
   `onnxruntime` builds can't run on CPU). Output is `model.onnx`; the FP32
   intermediate file is deleted.
4. **Write `manifest.json`** alongside it — architecture name, node id, crop
   and disease class name lists, image size, and the ImageNet
   mean/std used for preprocessing. This is what `pi/inference_service.py`
   reads to know how to preprocess images and decode class indices.
5. **Parity check** — if `data/PlantVillage` exists locally, runs a handful
   of real images through both the original PyTorch model and the new
   quantized ONNX session and compares top-1 predictions
   (`check_parity()`), warning if any disagree, *before* you ever copy
   anything to hardware.

Final artefact layout:
```
outputs/pi_export/
├── mobilenet_v3_small/{model.onnx, manifest.json}
├── efficientnet_lite0/{model.onnx, manifest.json}
└── mobilevit_xxs/{model.onnx, manifest.json}
```
(single-model mode — no `--all` — writes directly to `outputs/pi_export/`
instead of one subfolder per architecture).

For the full Pi hardware setup after this point (flashing the OS, copying
the bundle over, running with a live camera, systemd autostart), see
[`docs/raspberry_pi_deployment.md`](raspberry_pi_deployment.md) — this
section only covers *where the conversion happens and why it's shaped this
way*.

---

## 5. Step-by-step: running everything from scratch

### 5.1 Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2 Get the dataset
```bash
python scripts/download_plantvillage.py
```
First run downloads ~800MB and re-encodes ~54,000 images to
`data/PlantVillage/<Crop>___<Disease>/*.jpg` — takes a few minutes. No-op on
subsequent runs (skips if `data/PlantVillage/` already has files).

### 5.3 (Optional) run the test suite first
```bash
pytest tests/ -v
```
Uses synthetic images, no dataset or GPU required — confirms the pipeline
works before committing to a real training run.

### 5.4 Train
All architectures from `config.yaml` in one run:
```bash
python -m src.train --config config.yaml
```
Or split across sessions (e.g. Colab free-tier GPU quota), one architecture
at a time — results accumulate into the same `outputs/results_summary.json`:
```bash
python -m src.train --config config.yaml --arch mobilenet_v3_small --fresh   # start new sweep
python -m src.train --config config.yaml --arch efficientnet_lite0           # merges in
python -m src.train --config config.yaml --arch mobilevit_xxs                # merges in
```
Adjust node count, non-IID strategy, epochs/rounds, aggregation rule, or
energy assumptions in `config.yaml` before running, not via flags.

Produces, under `outputs/`: `checkpoints/<arch>/<node_id>.pt`,
`results_<arch>.json`, `results_summary.json`, `round_logs_<arch>.json`,
`emissions.csv`, `sustainability_report.json`/`.md`.

### 5.5 (Optional) batch inference with a PyTorch checkpoint
```bash
python -m src.predict --folder path/to/images --checkpoints-dir outputs/checkpoints
```
Auto-selects the best-scoring (architecture, node) unless you pass
`--arch`/`--node`. Writes `outputs/predictions.csv`.

### 5.6 Export to ONNX for the Raspberry Pi
```bash
pip install onnx onnxruntime      # already in requirements.txt
python scripts/export_for_pi.py --all      # every architecture, one subfolder each
# or a single one:
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0
```
Produces `outputs/pi_export/<arch>/{model.onnx, manifest.json}` (see §4).
Runs a parity check against real images automatically if
`data/PlantVillage` is present.

### 5.7 Deploy to the Raspberry Pi
Full walkthrough (flashing the OS, copying files, camera setup, systemd
service) is in
[`docs/raspberry_pi_deployment.md`](raspberry_pi_deployment.md). Short
version:
```bash
# on the Pi
python3 -m venv .venv && source .venv/bin/activate
pip install -r pi/requirements-pi.txt

# copy from dev machine
scp -r outputs/pi_export/mobilenet_v3_small pi@<pi-ip>:~/crop-mesh-detector/pi_export
scp pi/inference_service.py pi@<pi-ip>:~/crop-mesh-detector/

# smoke test (no camera needed)
python inference_service.py --model-dir pi_export --camera file --image test.jpg --once

# live camera
python inference_service.py --model-dir pi_export --camera opencv --interval 5       # USB webcam
python inference_service.py --model-dir pi_export --camera picamera2 --interval 5    # Pi Camera Module
```

### Quick reference: which command needs which environment
| Step | Command | Runs on | Needs torch/timm? |
|---|---|---|---|
| Download data | `scripts/download_plantvillage.py` | dev machine / Colab | no (needs tensorflow-datasets) |
| Train | `src.train` | dev machine / Colab (GPU helps) | yes |
| Batch predict | `src.predict` | dev machine / Colab | yes |
| Export ONNX | `scripts/export_for_pi.py` | dev machine / Colab | yes (+ onnx/onnxruntime) |
| Live inference | `pi/inference_service.py` | Raspberry Pi | **no** — onnxruntime only |
