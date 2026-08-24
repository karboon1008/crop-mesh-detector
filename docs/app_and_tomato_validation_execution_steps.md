# Crop Disease Detection App + Tomato Validation Pipeline — Execution Steps

Exact, ordered commands for two independent things in this repo:

1. Running the `apps/crop_disease_detection` Streamlit app (native Python, and via Docker).
2. Running the Tomato Dirichlet-mesh validation pipeline (`src/validation/run_tomato_pipeline.py`
   Stage 1, then `src/validation/run_tomato_knowledge_transfer.py` Stage 2).

All commands assume the terminal's working directory is the repo root
(`crop-mesh-detector/`) unless stated otherwise.

---

## Part 1 — Run the app

### Option A — Native (no Docker)

1. Activate the existing venv:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
2. Install the app's dependencies (first time, or whenever
   `apps/crop_disease_detection/requirements.txt` changes):
   ```powershell
   pip install -r apps\crop_disease_detection\requirements.txt
   ```
3. Confirm the model exists — the app hard-requires it and will `st.stop()`
   without it: `apps/model/model.onnx`.
4. Launch from the **repo root** (the app does `sys.path.insert` assuming
   that CWD — see `apps/crop_disease_detection/app.py` line 19):
   ```powershell
   streamlit run apps\crop_disease_detection\app.py
   ```
5. Streamlit opens `http://localhost:8501`. Grant camera permission, click
   "Capture crop image," confirm a colour-coded result (🟢 Healthy /
   🔴 Diseased / 🟠 Uncertain) appears. Each successful classification logs
   to `apps/crop_disease_detection/data/detections.db`.
6. Stop with `Ctrl+C`.

### Option B — Docker (standalone compose file, recommended for a clean run)

The app ships its own `docker-compose.yml` scoped to just this app (separate
from `docker/docker-compose.yml`'s mesh-simulation stack).

1. From `apps/crop_disease_detection/`:
   ```powershell
   cd apps\crop_disease_detection
   docker compose up --build
   ```
2. Open `http://localhost:8502` (compose maps host port 8502 → container
   port 8501).
3. Bind mounts keep state on the host, not in the container:
   - `../model` → `/app/apps/model` (swap `apps/model/model.onnx` on the
     host + `docker compose restart` to deploy an updated model)
   - `./data` → `/app/apps/crop_disease_detection/data` (`detections.db`
     persists here)
4. Stop with `Ctrl+C`, or `docker compose down` from the same directory.

### Option C — Docker (plain `docker build` / `docker run`, no compose)

Build from the **repo root** — the Dockerfile needs `apps/` at the top of
the build context:

```powershell
docker build -f apps\crop_disease_detection\Dockerfile -t crop-disease-detection .
```

Run, bind-mounting the model and DB onto the host:

```powershell
docker run --rm -p 8502:8501 `
  -v "${PWD}\apps\model:/app/apps/model" `
  -v "${PWD}\apps\crop_disease_detection\data:/app/apps/crop_disease_detection/data" `
  crop-disease-detection
```

(The Dockerfile's own comments note a Git-Bash-specific `pwd -W` quirk on
Windows for this same volume syntax — irrelevant in PowerShell, where
`${PWD}` already resolves to a Windows-style path.)

Open `http://localhost:8502`. Stop with `Ctrl+C` (add `-d` beforehand to run
detached, then `docker stop <container>` to stop it).

### Troubleshooting — "Model ... outputs don't match the expected 14-crop / 21-disease classifier shape"

**Rule to remember:** the Dockerfile only bind-mounts `apps/model/` and
`apps/crop_disease_detection/data/` (see Option B step 3 above). The app's
`.py` files are `COPY`'d into the image at **build time** and are NOT live —
a code change to `apps/crop_disease_detection/*.py` (e.g. `inference.py`'s
manifest-loading logic) is invisible to a running container until the image
is rebuilt.

| You changed... | Command needed | Why |
|---|---|---|
| `apps/model/model.onnx` / `manifest.json` | `docker compose restart` | bind-mounted, read live from host |
| `apps/crop_disease_detection/*.py` | `docker compose up --build` (or `up --build -d`) | baked into the image; `restart` reuses the same old image and will NOT pick up the change |

If you see this error (or any other symptom of the app behaving like an
older version of the code), diagnose it like this before assuming the model
files are wrong:

1. Confirm the model/manifest on disk actually agree with each other:
   ```powershell
   cat apps\model\manifest.json
   python -c "import onnxruntime as ort; s = ort.InferenceSession('apps/model/model.onnx'); print([o.shape for o in s.get_outputs()])"
   ```
   `manifest.json`'s `len(crop_classes)`/`len(disease_classes)` must match
   the ONNX model's two output shapes exactly.
2. Check what code the *running container* actually has (not what's on disk
   in the repo — these can silently diverge, which is exactly this bug):
   ```powershell
   docker exec <container_name> grep -c "_load_manifest" /app/apps/crop_disease_detection/inference.py
   ```
   `0` means the container is running a stale, pre-manifest-support image —
   rebuild it (`docker compose up --build`, from `apps/crop_disease_detection/`).
3. After rebuilding, verify end-to-end without needing the browser:
   ```powershell
   docker exec <container_name> python -c "import sys; sys.path.insert(0,'/app'); from apps.crop_disease_detection import inference; s = inference.create_session(inference.MODEL_PATH); print('OK', inference.CROP_CLASSES, inference.DISEASE_CLASSES)"
   ```

---

## Part 2 — Tomato validation pipeline

### Stage 1 — `run_tomato_pipeline.py` (per-node train → export → evaluate)

1. Same venv active (`.venv\Scripts\Activate.ps1`), repo root as CWD.
2. Confirm the tomato source data referenced in `config.yaml` (`tomato_mesh`
   block) exists: `data/PlantDoc`, `data/PlantWild/plantwild/plantwild/images`,
   `data/PlantWild/plantwild_v2/plantwild_v2`, plus PlantVillage. Prepare/
   download these first if missing.
3. Run all 3 stages (train → export → evaluate) for all 3 nodes:
   ```powershell
   python -m src.validation.run_tomato_pipeline --epochs 10
   ```
   - `--epochs` defaults to 2 if omitted; the tuning writeup
     (`docs/tomato_disease_knowledge_transfer_results.md`) recommends 10 for
     a stronger baseline.
   - `--output-dir` defaults to `outputs/validation/tomato_mesh`.
   - `--stage train|export|evaluate` runs one stage only, but they must run
     in that order and each depends on the previous stage's output already
     existing in the same `--output-dir`.
4. Expect ~1h total at 10 epochs (3 nodes × 10 epochs, CPU), or ~11 min at
   `--epochs 2`, per the reference runs.
5. Verify outputs before Stage 2:
   ```powershell
   dir outputs\validation\tomato_mesh\classes.json
   dir outputs\validation\tomato_mesh\node_0_mobilenet_v3_small\checkpoint.pt
   dir outputs\validation\tomato_mesh\node_0_mobilenet_v3_small\model.onnx
   ```

### Stage 2 — `run_tomato_knowledge_transfer.py` (knowledge-transfer rounds)

1. Must point at the **same** `--output-dir` used in Stage 1 — the script
   validates that Stage 1's persisted `test_idx`/`train_idx` still match a
   fresh recompute of the data split, and raises if config or source data
   drifted between the two runs.
2. Run it:
   ```powershell
   python -m src.validation.run_tomato_knowledge_transfer --rounds 10 --output-dir outputs\validation\tomato_mesh
   ```
   - `--rounds` must be 1–20 (defaults to `config.yaml`'s
     `tomato_mesh.rounds: 5` if omitted).
   - `--batch-size` defaults to 4.
3. This is the expensive stage: ~1h28m for 5 rounds, ~6h30m for 10 rounds on
   the reference machine. It prints `=== round 0 baseline ===`, then
   `=== knowledge-transfer round N/rounds ===` per round.
4. Check final outputs:
   ```powershell
   dir outputs\validation\tomato_mesh\knowledge_transfer\round_0_baseline.json
   dir outputs\validation\tomato_mesh\knowledge_transfer\round_10\round_summary.json
   dir outputs\validation\tomato_mesh\knowledge_transfer\knowledge_transfer_summary.json
   ```
   The last file is the Appendix-A.1-style disclosure summary (macro gain,
   per-class collaboration gain, energy/communication cost) — same shape as
   `docs/tomato_disease_knowledge_transfer_results.md`.

**Note:** Stage 2 has no `--stage`/resume flag — if interrupted, rerun the
full `--rounds N` from the start.
