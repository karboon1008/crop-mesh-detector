# Crop-Mesh-Detector

A **computer-simulation-only** implementation of the decentralised knowledge-mesh
design from the accompanying *WheatMesh* report (Solution A), adapted to the
public **PlantVillage** dataset for joint **crop-type** and **crop-disease**
detection. No hardware, radios, or real farms are involved — every "node" is a
Python object holding its own private data shard, running in a single process
on your machine.

Three lightweight backbones are trained and compared: **MobileNetV3-Small**,
**EfficientNet-Lite0**, and **MobileViT-XXS**.

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
├── config.yaml                  # every tunable lives here
├── requirements.txt
├── src/
│   ├── config.py                 # tiny YAML config loader
│   ├── data/
│   │   └── plantvillage.py       # dataset loading, crop/disease label parsing,
│   │                             # non-IID partition into nodes, public probe set
│   ├── models/
│   │   └── factory.py            # MobileNetV3-Small / EfficientNet-Lite0 / MobileViT-XXS
│   │                             # with a shared backbone + two heads (crop, disease)
│   ├── federated/
│   │   ├── aggregation.py        # trimmed-mean / Krum robust aggregation
│   │   ├── node.py                # one simulated farm: train, extract knowledge, distil, eval
│   │   └── mesh.py                # orchestrates rounds across all nodes (no central server)
│   ├── energy/
│   │   └── tracker.py            # CodeCarbon compute-energy tracking +
│   │                             # communication-cost estimator + sustainability report
│   ├── train.py                   # CLI entry point
│   ├── evaluate.py                # collaboration-gain / worst-node metrics
│   └── scenarios/                  # mesh simulation scenarios (disconnection,
│                                   # class addition, distribution shift) sharing
│                                   # a common round-driver harness
├── scripts/
│   ├── download_plantvillage.py   # fetches PlantVillage into data/PlantVillage/
│   └── check_energy_measurement.py # pre-flight check: is CodeCarbon really
│                                   # reading hardware counters on this machine?
├── tests/
│   └── test_pipeline.py           # end-to-end smoke test on synthetic images
└── outputs/                        # results, emissions.csv, sustainability_report.*
    └── scenarios/                  # per-scenario JSON reports (disconnection.json,
                                    # class_addition.json, distribution_shift.json)
```

## How it works, end to end

1. **Load PlantVillage** and parse each class folder name (e.g.
   `Tomato___Bacterial_spot`) into two labels: crop type (`Tomato`) and
   disease (`Bacterial_spot`).
2. **Carve out a public probe set** (a small, shared, non-private slice of
   the data) — used only for logit exchange, never for training.
3. **Partition the rest into N non-IID node shards** (by crop, by disease,
   or Dirichlet label-skew — configurable), simulating farms that each see
   a different regional mix of crops/diseases.
4. For each architecture, run two experiments so the mesh's benefit can be
   measured, not assumed:
   - **Baseline**: every node trains alone, no exchange at all (aligned
     compute budget — same total epochs as the mesh run).
   - **Mesh**: for several rounds, each node trains locally, computes its
     knowledge payload (prototypes + probe logits), broadcasts it, robustly
     aggregates what it *receives* from peers, and distils towards that
     consensus.
5. **Collaboration gain** = mesh accuracy − baseline accuracy, reported as
   both a macro-average across nodes and a worst-node score (so the mesh
   can't claim a win by only helping the already-strong nodes).
6. **Sustainability accounting**: compute energy is measured with
   CodeCarbon (hardware power counters where available, otherwise a
   documented wall-clock proxy), communication energy is *estimated* from
   the exact byte count exchanged using published per-byte radio energy
   figures, and both are converted to CO2e using one stated, configurable
   grid-carbon-intensity factor — so the "communication energy should not
   erase compute savings" claim is checked numerically, not asserted.

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
python -m src.train --config config.yaml
.venv/bin/python -m src.train --config config.yaml
```

To run the mesh simulation scenarios (each writes a JSON report to
`outputs/scenarios/`):

```bash
python -m src.scenarios.disconnection [--config path] [--arch name]
python -m src.scenarios.class_addition [--config path] [--arch name]
python -m src.scenarios.distribution_shift [--config path] [--arch name]
```

Everything — which architectures to run, node count, non-IID strategy,
epochs/rounds, aggregation rule, radio energy assumptions, grid carbon
intensity — is controlled from `config.yaml`. Results land in `outputs/`:

- `results_<architecture>.json` — per-architecture accuracy, model size,
  collaboration gain, bytes exchanged
- `results_summary.json` — all architectures combined
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
