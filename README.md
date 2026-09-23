# Crop-Mesh-Detector

A **computer-simulation-only** implementation of **HiveMind**'s decentralised
knowledge-mesh design (the accompanying research document), adapted to
**PlantVillage** for joint **crop-type**
and **crop-disease** detection across six simulated farm nodes. No hardware,
radios, or real farms are involved — every "node" is a Python object holding
its own private data shard, running in a single process on your machine.

Three lightweight backbones are trained and compared: **MobileNetV3-Small**,
**EfficientNet-Lite0**, and **MobileViT-XXS**.

## Team

**Team name:** HiveMind

**Lead representative:** Wong E Chern

**Other team members:** Yap Kar Boon, Muhammad Aiman bin Rosli, Dr Chang Siow Wee, Leo Yung Lynn

**University affiliation:** University of Bristol (MScR Student)

## Licence

MIT — see [`LICENSE`](LICENSE). Pre-existing third-party dependencies (PyTorch,
timm, CodeCarbon, etc.) retain their own licences; see the research document's
Pre-Existing IP Disclosure section for the full list.

## The core constraint this project satisfies

> Nodes must never exchange raw data **or model weights/gradients** — only
> small, non-invertible knowledge artefacts.

Concretely, each simulated node only ever sends two things to its peers:

1. **Class prototypes** — the mean feature-embedding vector per crop-type
   and per disease class, computed on the node's own private data (a few
   hundred floats per class, not an image).
2. **Public-probe-set logits** — soft predictions on a small, shared,
   non-private image set, so heterogeneous model sizes can still teach each
   other (FedProto / DS-FL style distillation).

No image, label distribution, gradient, or weight tensor ever leaves a node.
Each node aggregates what its **peers** sent it with a Byzantine-robust rule
(trimmed mean or Krum) and distils its own model towards that consensus —
there is no central aggregating server (that's what makes this a *mesh*
rather than a federated-server setup).

## Project structure

```
crop-mesh-detector/
├── config.yaml                  # every tunable lives here (see the `continual:` section)
├── requirements.txt
├── src/
│   ├── config.py                 # tiny YAML config loader
│   ├── data/
│   │   ├── plantvillage.py       # dataset loading + preprocessing, crop/disease label
│   │   │                         # parsing, probe carve, non-IID partition into nodes
│   │   └── splits.py             # probe / node shard / continual batch / train-test splits
│   ├── models/
│   │   └── factory.py            # MobileNetV3-Small / EfficientNet-Lite0 / MobileViT-XXS
│   │                             # with a shared backbone + two heads (crop, disease)
│   ├── federated/
│   │   ├── aggregation.py        # trimmed-mean / Krum robust aggregation
│   │   ├── node.py               # one simulated farm: train, extract knowledge, distil, eval
│   │   ├── knowledge_store.py    # the ONE shared knowledge database (SQLite)
│   │   ├── continual.py          # per-batch continual mesh (EMA teacher/learner roles)
│   │   └── mesh.py               # fixed-round all-to-all mesh (used by the scenarios)
│   ├── energy/
│   │   └── tracker.py            # CodeCarbon compute-energy tracking +
│   │                             # communication-cost estimator + sustainability report
│   ├── train.py                  # CLI entry point: the whole continual run
│   ├── evaluate.py               # metric helpers
│   └── scenarios/                # mesh simulation scenarios (disconnection,
│                                 # class addition, distribution shift)
├── scripts/
│   ├── download_plantvillage.py   # fetches PlantVillage into data/PlantVillage/
│   └── check_energy_measurement.py # pre-flight check: is CodeCarbon really
│                                   # reading hardware counters on this machine?
├── tests/                          # synthetic-image test suite (no download needed)
└── outputs/                        # results, emissions.csv, sustainability_report.*
```

## How it works, end to end

There is no separate local-only stage. Data arrives in batches, one batch
per trigger, the way it would in the field:

```bash
python -m src.train                 # start: set up the stream and run batch 0 (20,000 images)
python -m src.train --next-batch    # a new batch of 3,000 images arrives and is run on the saved models
python -m src.train --next-batch    # ...and another 3,000, until PlantVillage runs out
python -m src.train --reset         # throw everything away and start again from batch 0
```

Between triggers, everything is saved under `outputs/continual/`: which
images arrived in which batch, every node's model and optimizer, the EMAs,
and the knowledge database. Each trigger picks up exactly where the last
one stopped.

### Data preprocessing and splits

Set up once, on the first run:

1. **Load PlantVillage** (every class folder, ~54k images) and parse each
   folder name (e.g. `Tomato___Bacterial_spot`) into two labels: crop type
   (`Tomato`) and disease (`Bacterial_spot`).
2. **Ownership**: every image is assigned to the node (farm) that would
   photograph it. The split is non-IID: each node gets a different class mix
   (Dirichlet label skew by default).

Then every batch, only when it is triggered:

3. **Draw the batch**: a **stratified** sample (every class in
   proportion) of the images no earlier batch used:
   `continual.first_batch_size` (20,000) for batch 0,
   `continual.next_batch_size` (3,000) for each `--next-batch`.
4. **Probe slice**: a stratified 5% of this batch (`data.probe_set_fraction`,
   e.g. 150 of 3,000) is added to the public probe set shared by all nodes.
   The probe set is **cumulative**: batch *b*'s probe set is every batch's
   slice from 0 to *b*, in order.
5. **Deliver** the other 95% of the batch to the nodes that own the images.
6. **Each node splits its own arrivals** into private train/test,
   85% / 15% (`data.test_fraction`), stratified per class.
7. **Preprocessing** happens as images are read: train images get
   augmentation (random resized crop, flips, rotation, colour jitter);
   test/probe images only get resize + ImageNet normalisation.

A node that receives fewer than 2 train images or no test image in a batch
sits that batch out, keeping its model, EMA, and database entry.

**Probe logits across batches**: a teacher uploads logits for the probe
set as it stands when it uploads. An entry left over from an earlier batch
therefore covers only the probe images up to that batch. Because the probe
set only grows by appending, those images are the first part of today's
probe set, in the same order. A learner distils on the probe images that
*every* entry it retrieved covers: the probe set as of the oldest entry.

### Batch 0 (every node teaches and learns)

1. **Local training** — private train batch → backbone → feature → linear
   heads → logits → cross-entropy → backward → weights updated.
2. **Pre-distill evaluation** on the batch's private test set.
3. **Knowledge extraction**: prototypes (mean feature per class over the
   private train set) and probe logits (logits per image of the probe set so far).
4. **Upload** each node's prototypes + probe logits to the one shared
   database, labelled with the node id.
5. Every node is a teacher, so every node's knowledge is available to all others.
6. **Aggregation** of peers' prototypes and probe logits with trimmed mean
   or Krum (`federated.aggregation`), **excluding the node's own entry**.
7. **Distillation** towards that consensus.
8. **Post-distill evaluation** on the same private test set as step 2.
9. **Record** whether post beats pre, bytes uploaded/downloaded, and
   compute energy per phase.

### Later batches (each `--next-batch`, continuing from the saved models)

1. Local training on the new batch's private train set (a flat
   `training.local_epochs_per_round`; only batch 0 scales epochs up for
   small nodes).
2. Pre-distill evaluation on the new batch's private test set.
3. Update an EMA of the pre-distill metric (`continual.ema_metric`,
   `ema_alpha`):
   - **EMA rose vs. the previous batch → teacher**: extract knowledge and
     upload it, replacing the node's previous entry in the database.
   - **EMA < `ema_threshold` (default 0.8) → learner**: retrieve every other
     node's latest entry, aggregate (own knowledge excluded), distil.
   - A node can be both (improving but still weak) or neither (dipped but
     still above threshold — it just keeps its locally trained model).
   All teachers upload before any learner retrieves.
4. Post-distill evaluation on the same test set as step 2.
5. Record post vs. pre, bytes exchanged, and energy consumed.

**Sustainability accounting**: compute energy is measured with CodeCarbon
(hardware power counters where available, otherwise a documented
wall-clock proxy) for each phase of each node's batch; communication
energy is *estimated* from the exact bytes uploaded to and downloaded from
the knowledge database, using published per-byte radio energy figures; both
are converted to CO2e using one configurable grid-carbon-intensity factor.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data

```bash
python scripts/download_plantvillage.py
```

This uses TensorFlow Datasets' built-in `plant_village` catalog entry —
no Kaggle account or API token needed. The first run downloads ~800MB
(cached by TFDS under `~/tensorflow_datasets/`) and then writes every
image back out as a plain `.jpg` into `data/PlantVillage/<Crop>___<Disease>/`,
so the result is identical to a manual download: real, browsable image
files, one folder per class. Re-encoding ~54,000 images takes a few
minutes — this is meant to be run once, ahead of training.

### Run

On a new machine or cluster (e.g. a fresh Isambard/Slurm allocation), check
CodeCarbon can actually measure hardware energy there before you rely on the
numbers — run this inside the job allocation itself:

```bash
python scripts/check_energy_measurement.py
```

Then train:

```bash
python -m src.train --config config.yaml                # batch 0
python -m src.train --config config.yaml --next-batch   # each further batch
```

To run the mesh simulation scenarios (each writes a JSON report to
`outputs/scenarios/`):

```bash
python -m src.scenarios.disconnection [--config path] [--arch name]
python -m src.scenarios.class_addition [--config path] [--arch name]
python -m src.scenarios.distribution_shift [--config path] [--arch name]
```

### Hyperparameter tuning

`scripts/tune_hyperparams.py` runs a Bayesian (Optuna TPE) search over the
core training hyperparameters (`lr`, `distill_lr`, `proto_weight`,
`kd_weight`, `kd_temperature`) for one architecture. Each trial is a full
`python -m src.train` subprocess against its own scratch output directory
under `outputs_tuning/trials/`, scored on two objectives — maximize the
nodes' final post-distill test accuracy, minimize total compute energy (kWh) — so the result is a
Pareto front of trials rather than one "best" config:

```bash
python -m scripts.tune_hyperparams --n-trials 30 --arch efficientnet_lite0
```

Progress is checkpointed to a SQLite study db under `outputs_tuning/`, so a
re-run with the same `--study-name` resumes instead of starting over. The
Pareto-optimal trials (accuracy, energy, and the params that produced them)
are printed at the end and written to `outputs_tuning/pareto_front.json`.
Each trial's own `outputs_tuning/trials/trial_XXXX/config.yaml` records the
exact config used, so any trial can be re-run standalone with
`python -m src.train --config outputs_tuning/trials/trial_XXXX/config.yaml`.

### Try it out: classify an image or use your laptop's camera

Once a checkpoint exists in `outputs/checkpoints/` (from `python -m src.train`),
`src/infer.py` gives quick, ad-hoc predictions — crop type, disease, and each
one's confidence — without needing a labeled folder like `src/predict.py`
expects:

```bash
# Classify one or more specific image files
python -m src.infer --images path/to/leaf1.jpg path/to/leaf2.jpg

# Take a snapshot with the laptop's webcam and classify it
# (opens a preview window — press SPACE to capture, 'q' to cancel)
python -m src.infer --camera

# Continuously classify the webcam feed (prediction overlaid live, 'q' to quit)
python -m src.infer --camera --live
```

Add `--output path/to/results.csv` to any of the above to also save the
predictions to disk.

Everything — which architectures to run, node count, non-IID strategy,
epochs/rounds, aggregation rule, radio energy assumptions, grid carbon
intensity — is controlled from `config.yaml`. Results land in `outputs/`:

- `continual/stream.json` — which images arrived in which batch, at which node
- `continual/<architecture>/knowledge.db` — the shared knowledge database
  (latest entry per node + upload/retrieval logs with byte sizes)
- `continual/<architecture>/batch_summary.csv` — one row per (batch, node):
  role, EMA, pre/post-distill accuracy, improved?, bytes up/down, energy
- `continual/<architecture>/batch_logs.json` — the same, in full detail
- `results_<architecture>.json` — per-architecture summary (distillations
  that improved, mean distillation gain, bytes, energy, final per-node eval)
- `results_summary.json` — all architectures combined
- `checkpoints/<architecture>/<node>.pt` — each node's model after the last batch
- `emissions.csv` — CodeCarbon's raw per-block measurement log
- `sustainability_report.json` / `.md` — the combined compute +
  communication energy/CO2e picture and the "was it worth it" narrative

### Test

```bash
pytest tests/ -v
```

The test suite generates a tiny synthetic image set on the fly (no
PlantVillage download needed) and runs the full pipeline — label parsing,
non-IID partitioning, model forward passes for all three architectures,
one full mesh round, and the energy/communication accounting — to verify
everything actually works before you spend time on a full training run.

## A note on CodeCarbon and geolocation

CodeCarbon can fall back to IP-based geolocation to pick a grid factor for
its own internal CO2 estimate. This project does **not** use that internal
estimate: `write_sustainability_report()` only takes CodeCarbon's measured,
hardware-based **energy** figure (kWh, location-independent) and applies
the grid factor you set in `config.yaml` (`energy.grid_carbon_intensity_gco2_per_kwh`),
so the reported carbon figure is reproducible regardless of where you run
this.

### Verifying real energy measurement on a new machine (e.g. an HPC cluster)

CodeCarbon needs access to real hardware power counters — RAPL for CPU,
NVML for GPU — to actually *measure* energy. When it can't reach them (common
on shared HPC nodes, containerised allocations, or restricted permissions) it
silently degrades to its own constant-TDP-times-load estimate instead of
failing loudly, and the pipeline's own log still tags that block `"method":
"codecarbon"` — so the only way to tell measured from guessed is to check.

Before a real run on a new machine or cluster partition (e.g. a Slurm
allocation on a supercomputer such as Isambard), run this **inside the actual
job allocation**, not the login node — power counters are per-node, and a
login node's access often doesn't match what a compute node grants:

```bash
python scripts/check_energy_measurement.py
```

It prints what CodeCarbon detected and exits `0` only if CPU energy is
genuinely hardware-measured; otherwise it exits `2` and explains why (e.g. no
RAPL access), and separately flags if a GPU is visible to PyTorch but not to
CodeCarbon (GPU energy would then be missing from the report entirely, not
just estimated).

If it comes back unmeasured, either chase RAPL/NVML permissions for that
partition, or set `energy.fallback_power_watts` in `config.yaml` to a
realistic figure for that node type — the default (15W) models a laptop CPU
and will badly undercount a GPU-class HPC node.

## Design choices worth knowing about

- **Shared backbone, two heads** (crop type + disease), not two separate
  models — halves the on-device footprint for the same accuracy target.
- **Batch-norm statistics are kept local** to each node and never exchanged
  (FedBN) — they encode camera/lighting bias, which is exactly the kind of
  thing that shouldn't need to travel between farms.
- **Communication byte count assumes a fully-connected broadcast** (every
  node's payload reaches every other node once per round) — this is an
  upper bound; a real gossip relay with partial connectivity would use
  less bandwidth than what's reported here. See `MeshSimulator.run_round`
  in [`src/federated/mesh.py`](src/federated/mesh.py).
- **`efficientnet_lite0` and `mobilevit_xxs` map to timm's `tf_efficientnet_lite0`
  and `mobilevit_xxs`** respectively — chosen because they have pretrained
  weights available in `timm`, keeping all three architectures on the same
  loading interface (see [`src/models/factory.py`](src/models/factory.py)).
