"""Crop Disease Detection -- single-page Streamlit app. Captures a photo
from the browser's webcam, classifies it with the ONNX model in
apps/model/model.onnx, shows a colour-coded result, and logs each
successfully classified capture to a local SQLite DB
(apps/crop_disease_detection/data/detections.db).

Run:
    streamlit run apps/crop_disease_detection/app.py

To deploy an updated model: replace apps/model/model.onnx (same input/output
shape as documented in inference.py), then restart this command.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
                # Don't update last_photo_id/last_result here: leaving them
                # unchanged means the guard above stays true on every
                # following rerun of this same failed photo, so inference is
                # retried (and this st.error keeps re-rendering the failure)
                # instead of going silently blank, until a new photo arrives.
                st.error(f"Could not run detection on this photo: {e}")
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
    st.dataframe(rows)
else:
    st.caption("No detections logged yet.")
