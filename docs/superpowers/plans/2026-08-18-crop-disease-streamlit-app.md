# Crop Disease Detection Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page Streamlit app under `apps/crop_disease_detection/` that classifies a webcam snapshot with the existing ONNX crop/disease model, shows a colour-coded result, and logs every detection to a local SQLite DB.

**Architecture:** Three small modules with clear boundaries: `db.py` (pure SQLite persistence, no Streamlit/ML dependency), `inference.py` (pure ONNX Runtime preprocessing/prediction/tier logic, no Streamlit dependency — fully unit-testable without a Streamlit runtime), and `app.py` (the Streamlit page itself, which wires the other two together and is verified manually since UI interaction isn't practically unit-testable).

**Tech Stack:** Streamlit (already installed, `1.61.1`), ONNX Runtime, NumPy, Pillow, sqlite3 (stdlib).

**Spec:** `docs/superpowers/specs/2026-08-18-crop-disease-streamlit-app-design.md`

## Global Constraints

- All code lives under `apps/` in the main working tree — never touch `.worktrees/`.
- The model file is `apps/model/model.onnx` (already present). Verified shape: input `image` is float32 `(1, 3, 160, 160)`; outputs are `crop_logits (*, 14)` and `disease_logits (*, 21)`.
- Crop/disease class names, `image_size=160`, and ImageNet mean/std are **hardcoded** in `inference.py` — no `manifest.json`. Order must exactly match `outputs/checkpoints/classes.json` since it's the model's class index order.
- Confidence threshold for the "uncertain" tier is `0.60`, strict less-than (`< 0.60` is uncertain; exactly `0.60` counts as confident).
- Tier rule: `uncertain` if either head's confidence is below threshold; else `healthy` if `disease_label == "healthy"`; else `diseased`.
- No image bytes are stored anywhere — DB rows are text/numeric only (`captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier`).
- Detection is auto-saved once per photo capture, not once per Streamlit rerun — `st.camera_input` keeps returning the same object across unrelated reruns, so app.py must dedupe on the photo's `file_id` (see Task 3).
- Page title is exactly `"Crop Disease Detection"`.
- `streamlit` is not yet in the root `requirements.txt` even though it's installed — Task 3 adds it there (`streamlit>=1.38`, matching the floor already used in `docker/dashboard/requirements.txt`).

---

### Task 1: SQLite persistence layer

**Files:**
- Create: `apps/__init__.py` (empty — makes `apps` a regular package, same pattern as `src/__init__.py`)
- Create: `apps/crop_disease_detection/__init__.py` (empty)
- Create: `apps/crop_disease_detection/db.py`
- Create: `tests/test_crop_disease_app.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `db.init_db(db_path: Path) -> None` — creates the `detections` table if it doesn't exist, creating parent directories as needed.
  - `db.save_detection(db_path: Path, captured_at: str, predicted_crop: str, crop_confidence: float, predicted_disease: str, disease_confidence: float, tier: str) -> None`
  - `db.get_recent(db_path: Path, limit: int = 20) -> list[dict]` — each dict has keys `captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier`, most recently inserted row first.

- [ ] **Step 1: Create empty package markers**

Create `apps/__init__.py` with empty content (0 bytes), and `apps/crop_disease_detection/__init__.py` with empty content (0 bytes). These make `apps.crop_disease_detection` importable as `from apps.crop_disease_detection import db` from `tests/`, the same way `from src.data.plantvillage import ...` already works via `src/__init__.py`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_crop_disease_app.py`:

```python
"""Tests for the pure logic behind the Crop Disease Detection Streamlit app
(apps/crop_disease_detection/) -- the Streamlit UI itself (app.py) isn't
unit-tested here; see docs/superpowers/plans/2026-08-18-crop-disease-streamlit-app.md
for the manual verification checklist that covers it.
"""
from __future__ import annotations

import pytest

from apps.crop_disease_detection import db


# --- db.py ---

def test_save_and_get_recent_round_trip(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    db.save_detection(db_path, "2026-08-18T10:00:00", "Tomato", 0.91, "Early_blight", 0.83, "diseased")
    db.save_detection(db_path, "2026-08-18T10:01:00", "Apple", 0.95, "healthy", 0.88, "healthy")

    rows = db.get_recent(db_path, limit=20)

    assert len(rows) == 2
    # Most recently inserted row comes back first.
    assert rows[0]["predicted_crop"] == "Apple"
    assert rows[0]["predicted_disease"] == "healthy"
    assert rows[0]["tier"] == "healthy"
    assert rows[1]["predicted_crop"] == "Tomato"
    assert rows[1]["crop_confidence"] == pytest.approx(0.91)


def test_get_recent_respects_limit(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    for i in range(5):
        db.save_detection(db_path, f"2026-08-18T10:0{i}:00", "Tomato", 0.9, "healthy", 0.9, "healthy")

    rows = db.get_recent(db_path, limit=3)

    assert len(rows) == 3


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "detections.db"
    db.init_db(db_path)
    db.save_detection(db_path, "2026-08-18T10:00:00", "Tomato", 0.9, "healthy", 0.9, "healthy")

    db.init_db(db_path)  # calling again must not wipe existing rows
    rows = db.get_recent(db_path)

    assert len(rows) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_crop_disease_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apps.crop_disease_detection.db'` (or similar import error) — `db.py` doesn't exist yet.

- [ ] **Step 4: Implement `db.py`**

Create `apps/crop_disease_detection/db.py`:

```python
"""SQLite persistence for logged crop/disease detections. Pure stdlib —
no Streamlit or ML dependency here, so this is unit-testable in isolation.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                predicted_crop TEXT NOT NULL,
                crop_confidence REAL NOT NULL,
                predicted_disease TEXT NOT NULL,
                disease_confidence REAL NOT NULL,
                tier TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_detection(
    db_path: Path,
    captured_at: str,
    predicted_crop: str,
    crop_confidence: float,
    predicted_disease: str,
    disease_confidence: float,
    tier: str,
) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO detections
                (captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent(db_path: Path, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT captured_at, predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier
            FROM detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_crop_disease_app.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Ignore the runtime DB file**

Add to `.gitignore` (after the existing `outputs/*` block):

```
apps/crop_disease_detection/data/
```

- [ ] **Step 7: Commit**

```bash
git add apps/__init__.py apps/crop_disease_detection/__init__.py apps/crop_disease_detection/db.py tests/test_crop_disease_app.py .gitignore
git commit -m "feat: add SQLite persistence layer for crop disease detections"
```

---

### Task 2: Inference logic (preprocessing, ONNX Runtime, tier classification)

**Files:**
- Create: `apps/crop_disease_detection/inference.py`
- Modify: `tests/test_crop_disease_app.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent module).
- Produces:
  - `inference.MODEL_PATH: Path` — resolves to `apps/model/model.onnx`.
  - `inference.IMAGE_SIZE: int` — `160`.
  - `inference.CROP_CLASSES: list[str]` — 14 names, in model-index order.
  - `inference.DISEASE_CLASSES: list[str]` — 21 names, in model-index order.
  - `inference.CONFIDENCE_THRESHOLD: float` — `0.60`.
  - `inference.preprocess(image: PIL.Image.Image) -> np.ndarray` — shape `(1, 3, 160, 160)`, dtype `float32`.
  - `inference.classify_tier(crop_confidence: float, disease_confidence: float, disease_label: str) -> str` — one of `"healthy"`, `"diseased"`, `"uncertain"`.
  - `inference.create_session(model_path: Path) -> onnxruntime.InferenceSession` — raises `FileNotFoundError` if `model_path` doesn't exist, `ValueError` if its shapes don't match this app's classes.
  - `inference.predict(session: onnxruntime.InferenceSession, image: PIL.Image.Image) -> dict` — keys `predicted_crop, crop_confidence, predicted_disease, disease_confidence, tier`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_crop_disease_app.py` (add these imports to the top-of-file import block, then add the test functions below the existing DB tests):

```python
import numpy as np
from PIL import Image

from apps.crop_disease_detection import inference
```

```python
# --- inference.py: preprocess ---

def test_preprocess_output_shape_and_dtype():
    image = Image.new("RGB", (320, 240), color=(100, 150, 200))

    result = inference.preprocess(image)

    assert result.shape == (1, 3, inference.IMAGE_SIZE, inference.IMAGE_SIZE)
    assert result.dtype == np.float32


def test_preprocess_normalizes_values():
    # A flat white image should map, per channel, to (1.0 - mean) / std.
    image = Image.new("RGB", (160, 160), color=(255, 255, 255))

    result = inference.preprocess(image)

    expected = (np.array([1.0, 1.0, 1.0], dtype=np.float32) - inference.MEAN) / inference.STD
    for c in range(3):
        assert result[0, c].mean() == pytest.approx(float(expected[c]), abs=1e-4)


# --- inference.py: classify_tier ---

@pytest.mark.parametrize(
    "crop_conf, disease_conf, disease_label, expected_tier",
    [
        (0.90, 0.90, "healthy", "healthy"),
        (0.90, 0.90, "Early_blight", "diseased"),
        (0.59, 0.90, "healthy", "uncertain"),
        (0.90, 0.59, "healthy", "uncertain"),
        (0.60, 0.60, "healthy", "healthy"),  # exactly at threshold counts as confident
    ],
)
def test_classify_tier(crop_conf, disease_conf, disease_label, expected_tier):
    assert inference.classify_tier(crop_conf, disease_conf, disease_label) == expected_tier


# --- inference.py: create_session ---

def test_create_session_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inference.create_session(tmp_path / "does_not_exist.onnx")


# --- inference.py: predict, end-to-end against the real bundled model ---

def test_predict_end_to_end_with_bundled_model():
    session = inference.create_session(inference.MODEL_PATH)
    image = Image.new("RGB", (200, 200), color=(80, 120, 60))

    result = inference.predict(session, image)

    assert result["predicted_crop"] in inference.CROP_CLASSES
    assert result["predicted_disease"] in inference.DISEASE_CLASSES
    assert 0.0 <= result["crop_confidence"] <= 1.0
    assert 0.0 <= result["disease_confidence"] <= 1.0
    assert result["tier"] in {"healthy", "diseased", "uncertain"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_crop_disease_app.py -v`
Expected: the new tests FAIL with `ModuleNotFoundError: No module named 'apps.crop_disease_detection.inference'`; the Task 1 DB tests still PASS.

- [ ] **Step 3: Implement `inference.py`**

Create `apps/crop_disease_detection/inference.py`:

```python
"""Pure prediction logic for the Crop Disease Detection app: ONNX Runtime
session loading/validation, image preprocessing, and the crop/disease
softmax + confidence-tier classification. No Streamlit import here so this
module is unit-testable without a Streamlit runtime -- see app.py for the
Streamlit-specific caching wrapper around create_session().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.onnx"

IMAGE_SIZE = 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLD = 0.60

# Order matches outputs/checkpoints/classes.json -- position is the model's
# class index, so these lists must stay in exactly this order.
CROP_CLASSES = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
    "Pepper,_bell", "Potato", "Raspberry", "Soybean", "Squash",
    "Strawberry", "Tomato",
]
DISEASE_CLASSES = [
    "Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy",
    "Powdery_mildew", "Cercospora_leaf_spot Gray_leaf_spot", "Common_rust",
    "Northern_Leaf_Blight", "Esca_(Black_Measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Haunglongbing_(Citrus_greening)", "Bacterial_spot", "Early_blight",
    "Late_blight", "Leaf_scorch", "Leaf_Mold", "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite", "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus",
]


def preprocess(image: Image.Image) -> np.ndarray:
    """RGB PIL image -> (1, 3, IMAGE_SIZE, IMAGE_SIZE) float32, normalized
    identically to training (resize -> [0,1] -> ImageNet mean/std -> CHW).
    """
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def classify_tier(crop_confidence: float, disease_confidence: float, disease_label: str) -> str:
    if crop_confidence < CONFIDENCE_THRESHOLD or disease_confidence < CONFIDENCE_THRESHOLD:
        return "uncertain"
    return "healthy" if disease_label == "healthy" else "diseased"


def create_session(model_path: Path) -> ort.InferenceSession:
    """Loads and validates the ONNX model at model_path. Raises
    FileNotFoundError if it's missing, ValueError if its input/output shapes
    don't match this app's hardcoded classes/image size.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Place an ONNX model there with input "
            f"'image' shaped (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}) and outputs "
            f"'crop_logits' ({len(CROP_CLASSES)} classes) + 'disease_logits' "
            f"({len(DISEASE_CLASSES)} classes)."
        )
    session = ort.InferenceSession(str(model_path))
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_input_shape = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    if len(inputs) != 1 or list(inputs[0].shape) != expected_input_shape:
        actual = inputs[0].shape if inputs else None
        raise ValueError(f"Model at {model_path} has input shape {actual}; expected {expected_input_shape}.")
    if (
        len(outputs) != 2
        or outputs[0].shape[-1] != len(CROP_CLASSES)
        or outputs[1].shape[-1] != len(DISEASE_CLASSES)
    ):
        raise ValueError(
            f"Model at {model_path} outputs don't match the expected "
            f"{len(CROP_CLASSES)}-crop / {len(DISEASE_CLASSES)}-disease classifier shape."
        )
    return session


def predict(session: ort.InferenceSession, image: Image.Image) -> dict:
    batch = preprocess(image)
    input_name = session.get_inputs()[0].name
    crop_logits, disease_logits = session.run(None, {input_name: batch})
    crop_probs = _softmax(crop_logits)
    disease_probs = _softmax(disease_logits)
    crop_idx = int(np.argmax(crop_probs, axis=1)[0])
    disease_idx = int(np.argmax(disease_probs, axis=1)[0])
    crop_confidence = float(crop_probs[0, crop_idx])
    disease_confidence = float(disease_probs[0, disease_idx])
    disease_label = DISEASE_CLASSES[disease_idx]
    return {
        "predicted_crop": CROP_CLASSES[crop_idx],
        "crop_confidence": crop_confidence,
        "predicted_disease": disease_label,
        "disease_confidence": disease_confidence,
        "tier": classify_tier(crop_confidence, disease_confidence, disease_label),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_crop_disease_app.py -v`
Expected: PASS (all tests, Task 1's and Task 2's together)

- [ ] **Step 5: Commit**

```bash
git add apps/crop_disease_detection/inference.py tests/test_crop_disease_app.py
git commit -m "feat: add ONNX inference and confidence-tier logic for crop disease detection"
```

---

### Task 3: Streamlit page

**Files:**
- Create: `apps/crop_disease_detection/app.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `db.init_db`, `db.save_detection`, `db.get_recent` (Task 1); `inference.MODEL_PATH`, `inference.create_session`, `inference.predict` (Task 2).
- Produces: the runnable app itself — no other task depends on it.

- [ ] **Step 1: Add the `streamlit` dependency**

In `requirements.txt`, add this line in the "Core ML stack" section (it's already installed in `.venv`, but wasn't declared at the repo root — only `docker/dashboard/requirements.txt` had it):

```
streamlit>=1.38       # apps/crop_disease_detection — webcam capture + result UI
```

- [ ] **Step 2: Implement `app.py`**

Create `apps/crop_disease_detection/app.py`:

```python
"""Crop Disease Detection -- single-page Streamlit app. Captures a photo
from the browser's webcam, classifies it with the ONNX model in
apps/model/model.onnx, shows a colour-coded result, and logs every capture
to a local SQLite DB (apps/crop_disease_detection/data/detections.db).

Run:
    streamlit run apps/crop_disease_detection/app.py

To deploy an updated model: replace apps/model/model.onnx (same input/output
shape as documented in inference.py), then restart this command.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from apps.crop_disease_detection import db, inference

DB_PATH = Path(__file__).resolve().parent / "data" / "detections.db"

TIER_STYLE = {
    "healthy": ("#2e7d32", "🟢 Healthy"),
    "diseased": ("#c62828", "🔴 Diseased"),
    "uncertain": ("#ef6c00", "🟠 Uncertain"),
}

st.set_page_config(page_title="Crop Disease Detection", layout="wide")
st.title("Crop Disease Detection")

db.init_db(DB_PATH)


@st.cache_resource
def get_session():
    return inference.create_session(inference.MODEL_PATH)


try:
    session = get_session()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

left, right = st.columns(2)

with left:
    photo = st.camera_input("Capture crop image")

with right:
    st.subheader("Result")
    if photo is None:
        st.info("Take a photo to see results.")
    else:
        # st.camera_input keeps returning the same object across unrelated
        # reruns (e.g. widget interactions elsewhere on the page) -- without
        # this file_id check, every rerun would re-run inference and insert
        # a duplicate DB row for the same capture.
        if st.session_state.get("last_photo_id") != photo.file_id:
            try:
                image = Image.open(photo)
                result = inference.predict(session, image)
            except Exception as e:
                st.error(f"Could not run detection on this photo: {e}")
                result = None
            else:
                captured_at = datetime.now().isoformat(timespec="seconds")
                db.save_detection(
                    DB_PATH,
                    captured_at,
                    result["predicted_crop"],
                    result["crop_confidence"],
                    result["predicted_disease"],
                    result["disease_confidence"],
                    result["tier"],
                )
            st.session_state["last_photo_id"] = photo.file_id
            st.session_state["last_result"] = result

        result = st.session_state.get("last_result")
        if result is not None:
            color, label = TIER_STYLE[result["tier"]]
            st.markdown(
                f"""
                <div style="padding:1.2em;border-radius:0.5em;background-color:{color};color:white;">
                    <h3 style="margin-top:0;">{label}</h3>
                    <p><b>Predicted crop:</b> {result['predicted_crop']}
                        ({result['crop_confidence'] * 100:.1f}%)</p>
                    <p><b>Predicted disease:</b> {result['predicted_disease']}
                        ({result['disease_confidence'] * 100:.1f}%)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.subheader("Recent detections")
rows = db.get_recent(DB_PATH, limit=20)
if rows:
    st.dataframe(rows, use_container_width=True)
else:
    st.caption("No detections logged yet.")
```

- [ ] **Step 3: Launch the app**

Run: `streamlit run apps/crop_disease_detection/app.py`
Expected: browser opens to a page titled "Crop Disease Detection" with a camera capture widget on the left, a "Take a photo to see results." placeholder on the right, and an empty "Recent detections" section below.

- [ ] **Step 4: Manually verify a confident detection**

Click "Take Photo" and capture a real leaf (or any clear, well-lit image). Confirm:
- The right column shows a green or red box (not amber) with a predicted crop, predicted disease, and both confidence percentages.
- The "Recent detections" table below now shows exactly one new row with a `captured_at` timestamp matching now.

- [ ] **Step 5: Manually verify the uncertain tier**

Cover the camera lens (or point it at a blank wall) and take another photo. Confirm the right column shows the amber "🟠 Uncertain" box, and a second row appears in "Recent detections" with `tier = uncertain`.

- [ ] **Step 6: Manually verify no duplicate rows on unrelated reruns**

With the last photo still showing, interact with the "Recent detections" table (e.g. click a column header to sort, if supported) or resize the browser window. Confirm the row count in the table does NOT increase — the same capture must not be re-logged.

- [ ] **Step 7: Manually verify the missing-model error path**

Stop the app (Ctrl+C). Temporarily rename `apps/model/model.onnx` to `apps/model/model.onnx.bak`. Run `streamlit run apps/crop_disease_detection/app.py` again. Confirm the page shows a red `st.error` message referencing the missing model path and does not render the camera/result columns. Rename the file back to `apps/model/model.onnx` and confirm a fresh `streamlit run` works normally again.

- [ ] **Step 8: Commit**

```bash
git add apps/crop_disease_detection/app.py requirements.txt
git commit -m "feat: add Crop Disease Detection Streamlit page"
```

---

## Post-implementation checklist

- [ ] `pytest tests/test_crop_disease_app.py -v` passes in full.
- [ ] `streamlit run apps/crop_disease_detection/app.py` runs from a clean checkout with no extra setup beyond the existing `.venv`.
- [ ] `apps/model/model.onnx` can be swapped for another same-shaped bundle (e.g. copy `outputs/pi_export/efficientnet_lite0/model_fp32.onnx` over it, renamed to `model.onnx`) and, after restarting the app, produces predictions using the new weights.
