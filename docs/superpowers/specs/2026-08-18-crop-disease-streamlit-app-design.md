# Crop Disease Detection — Streamlit App Design

Date: 2026-08-18
Status: Approved

## Purpose

Build a local, single-page Streamlit app that captures a photo from the
user's laptop/PC webcam, runs it through the repo's existing crop/disease
ONNX classifier, shows a colour-coded result, and logs every detection with
a timestamp to a local SQLite database.

## Constraints / requirements (from user)

1. All app code lives under `apps/`.
2. The ONNX model lives under `apps/model/` and must be swappable — replacing
   the file and restarting the app deploys the new model, no code change
   needed for a same-shape model.
3. Single page: camera capture on the left, result on the right, with a
   colour indicator for predicted crop and predicted disease.
4. Streamlit framework; results saved with a datetime to a local SQLite DB.
5. App title: "Crop Disease Detection".
6. Work directly in `apps/`, not inside `.worktrees/`.

## Existing context this design builds on

- The trained model is a two-headed classifier (shared backbone, separate
  `crop_head` / `disease_head` linear layers) — see `src/models/factory.py`.
  It outputs `crop_logits` and `disease_logits` independently; there is no
  cross-consistency logic between the two heads anywhere in the existing
  codebase (`src/predict.py` picks top-1 per head via softmax+argmax).
- `apps/model/model.onnx` already exists (placed by the user). Verified
  directly via `onnx.load`:
  - input `image`: float32 `(1, 3, 160, 160)`
  - output `crop_logits`: `(*, 14)`
  - output `disease_logits`: `(*, 21)`
  This matches `outputs/checkpoints/classes.json` / the `outputs/pi_export/*/manifest.json`
  bundles: 14 crop classes, 21 disease classes (including `"healthy"`),
  `image_size=160`, ImageNet `mean=[0.485,0.456,0.406]` / `std=[0.229,0.224,0.225]`.
- Per user decision, classes/image_size/mean/std are **hardcoded** in the
  app rather than read from a `manifest.json` (the model bundle format used
  elsewhere in the repo pairs the two). This is a deliberate simplification:
  swapping in a same-shaped model (any of the three architectures in
  `outputs/pi_export/`, since they all share these classes/image_size) works
  with no code change; swapping in a model with a different class count or
  input size will require updating the hardcoded constants in `inference.py`.
- `streamlit>=1.61` and `onnxruntime`, `numpy`, `Pillow` are already
  installed in the repo's main `.venv` (confirmed via
  `python -c "import streamlit"` and `docker/dashboard/requirements.txt`
  already depending on `streamlit>=1.38`). `streamlit` is not yet listed in
  the root `requirements.txt` — this design adds it there.

## Folder structure

```
apps/
  model/
    model.onnx                 # already present; swap this file + restart the app to deploy an update
  crop_disease_detection/
    app.py                      # Streamlit entrypoint: page config, layout, wiring
    inference.py                 # ONNX Runtime session load + preprocessing + predict()
    db.py                        # sqlite3 helpers: init_db(), save_detection(), get_recent()
    data/
      detections.db               # created automatically on first run; gitignored
```

`apps/model/` sits as a sibling of the app folder (not nested inside it) so
future apps under `apps/` could share the same model directory if needed.

## Page layout

- `st.set_page_config(page_title="Crop Disease Detection", layout="wide")`.
- Title: "Crop Disease Detection".
- Two columns:
  - **Left**: `st.camera_input(...)`. Streamlit's built-in snapshot widget —
    user clicks "Take Photo" in-browser; no continuous video feed, no extra
    dependency (`streamlit-webrtc` not needed).
  - **Right**: result panel.
    - Before any capture: neutral placeholder text ("Take a photo to see
      results").
    - After a capture: a colour-coded box (green / red / amber — see below)
      showing predicted crop + its confidence, predicted disease + its
      confidence, and the tier label.
- **Below both columns**: "Recent detections" — an `st.dataframe` of the
  last 20 rows from SQLite (datetime, crop, crop_confidence, disease,
  disease_confidence, tier), most recent first, re-queried on every rerun.

## Detection flow

Fires automatically on every photo capture (no separate "Save" button —
per user decision, one click = one logged detection):

1. Decode the captured JPEG bytes (PIL) → convert to RGB → resize to
   160×160 (bilinear) → scale to `[0,1]` → normalize with ImageNet
   mean/std → transpose HWC→CHW → add batch dim → `(1, 3, 160, 160)`
   float32. This mirrors the exact preprocessing used in training/export
   (`src/data/plantvillage.py`, `scripts/export_for_pi.py`).
2. Run the ONNX Runtime session → `crop_logits`, `disease_logits`.
3. Softmax each head independently; take top-1 label + confidence per head
   (same approach as `src/predict.py`).
4. Classify into a tier:
   - `crop_confidence < 0.60` OR `disease_confidence < 0.60` →
     **Amber — Uncertain**
   - else, `disease_label == "healthy"` → **Green — Healthy**
   - else → **Red — Diseased: `<disease_label>`**
5. Render the result box in the right column per the tier's colour.
6. Insert one row into `detections.db`:
   `(id, captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier)`.
   `captured_at` is stored as an ISO-8601 local timestamp
   (`datetime.now().isoformat()`).

No image bytes are stored — per user decision, the DB holds text/numeric
results only.

## SQLite schema

```sql
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    predicted_crop TEXT NOT NULL,
    crop_confidence REAL NOT NULL,
    predicted_disease TEXT NOT NULL,
    disease_confidence REAL NOT NULL,
    tier TEXT NOT NULL
);
```

`db.py` exposes:
- `init_db(path) -> None` — creates the table if absent.
- `save_detection(path, captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier) -> None`
- `get_recent(path, limit=20) -> list[dict]`

## Model loading / swap behaviour

- `inference.py` loads the ONNX Runtime session once via
  `st.cache_resource`, keyed only on the fixed file path
  `apps/model/model.onnx` — i.e. cached for the lifetime of the running
  Streamlit process, not hot-reloaded while the server keeps running. This
  matches the user's stated requirement literally: "when rerun the apps it
  will deploy the updated one" — replace the file, then restart
  `streamlit run apps/crop_disease_detection/app.py`.
- Class name lists, `image_size=160`, and ImageNet mean/std are module-level
  constants in `inference.py` (see "Existing context" above for the exact
  values), not read from any manifest file.

## Error handling

- `apps/model/model.onnx` missing at startup → `st.error(...)` +
  `st.stop()` with a message telling the user where to put the file.
- Model present but shape-incompatible (e.g. someone swaps in a model with
  a different class count/input size) → the `onnxruntime.InferenceSession`
  construction or the first `session.run()` call raises; caught and
  surfaced via `st.error(...)` explaining the expected input
  `(1, 3, 160, 160)` and outputs `(*, 14)` / `(*, 21)`.
- Captured photo fails to decode → `st.error(...)`, no DB write, no crash.

## Testing

Streamlit UI interaction itself isn't practically unit-testable, but the
pure logic underneath it is. Add `tests/test_crop_disease_app.py` covering:
- Preprocessing function: given a synthetic PIL image, output has shape
  `(1, 3, 160, 160)`, dtype float32, and values in the expected normalized
  range.
- Tier classification function: given synthetic
  `(crop_confidence, disease_confidence, disease_label)` combinations,
  returns the correct tier for boundary cases (exactly 0.60, just above,
  just below, healthy vs. not).
- `db.py` round-trip: `init_db` + `save_detection` + `get_recent` against a
  temp SQLite file path, asserting the row comes back with the right
  fields and ordering (most recent first).

Manual verification (not automatable): run
`streamlit run apps/crop_disease_detection/app.py`, capture a real leaf
photo, confirm the result panel and a matching new DB row appear; capture a
deliberately blurry/blank photo and confirm the amber "Uncertain" tier
appears instead of a confident green/red.

## Out of scope (explicitly, per user decisions during brainstorming)

- Continuous/live camera feed (`streamlit-webrtc`) — snapshot-on-click only.
- `manifest.json`-driven class/image-size configuration — hardcoded instead.
- Storing captured images (file or blob) in the DB or on disk.
- Manual "Save result" button — every capture auto-saves.
- Cross-consistency checks between the crop and disease heads (e.g.
  rejecting a disease name that doesn't apply to the predicted crop) — the
  app surfaces the two independent head predictions as-is, same as the
  existing `src/predict.py`.
