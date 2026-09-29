"""HiveMind crop-disease demo -- upload a leaf photo (or use the webcam) and the
HiveMind-trained ONNX model predicts the crop and the disease.

Run locally:   streamlit run app.py        -> http://localhost:8501
Run in Docker: docker compose up --build   -> http://localhost:8502
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

import db
import inference
import leaf_detection
from live_camera import live_camera

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "detections.db"

TIER_STYLE = {
    "healthy": ("#2e7d32", "Healthy"),
    "diseased": ("#c62828", "Diseased"),
    "uncertain": ("#ef6c00", "Uncertain"),
}


def pretty(label: str) -> str:
    return label.replace("_", " ").replace(",", "").strip().capitalize()


st.set_page_config(page_title="HiveMind Crop Disease Detection", layout="wide")
st.title("HiveMind: Crop Disease Detection")
db.init_db(DB_PATH)


ARCH_NAMES = {"efficientnet_lite0": "EfficientNet-Lite0", "mobilenet_v3_small": "MobileNetV3-Small",
              "mobilevit_xxs": "MobileViT-XXS"}
# every farm's final model from the HiveMind continual run, with its held-out accuracy
# (crop and disease both right, 25 unseen images per class x 38 classes)
MODELS = json.loads((HERE / "models" / "index.json").read_text())


@st.cache_resource
def get_session(path: str):
    return inference.create_session(HERE / path)


@st.cache_resource
def get_detector():
    return leaf_detection.load_detector()


detector = get_detector()


with st.sidebar:
    st.header("Model")
    best = max(MODELS, key=lambda m: m["heldout_acc"])
    archs = list(ARCH_NAMES)
    arch = st.selectbox("Architecture", archs, index=archs.index(best["arch"]),
                        format_func=lambda a: f"{ARCH_NAMES[a]} (best farm {100 * max(m['heldout_acc'] for m in MODELS if m['arch'] == a):.0f}%)")
    farms = [m for m in MODELS if m["arch"] == arch]
    default = max(range(len(farms)), key=lambda i: farms[i]["heldout_acc"])
    chosen = st.selectbox("Farm", farms, index=default,
                          format_func=lambda m: f"Farm {m['node'].split('_')[1]}: {100 * m['heldout_acc']:.1f}% held-out, {m['size_mb']} MB")
    st.markdown(
        f"""
- **Model:** {ARCH_NAMES[arch]}, farm {chosen['node'].split('_')[1]} of 6
  (HiveMind continual run, `kb-fresh-01`, 13 rounds)
- **Held-out accuracy:** {100 * chosen['heldout_acc']:.1f}% (crop and disease both right)
- **Classes:** 14 crops, 21 disease states
- **Size:** {chosen['size_mb']} MB (ONNX, int8 heads), runs on CPU
"""
    )
    st.caption("Each farm trained only on its own photos; farms shared class prototypes and "
               "probe-set predictions, never images or weights.")

    st.header("Leaf detection")
    if detector is None:
        find_leaves = False
        st.caption("Off: no leaf detector in models/detector/. Without it, the whole (zoomed) photo "
                   "is classified as one leaf. See README.md to add one.")
    else:
        find_leaves = st.toggle("Find every leaf first (two-stage)", value=True,
                                help="Stage 1 finds each leaf; stage 2 classifies each crop with the farm model.")
        leaf_threshold = st.slider("Leaf confidence threshold", 0.1, 0.9, detector.default_threshold, 0.05,
                                   disabled=not find_leaves)
        box_style = st.radio("Boxes show", ["detector", "diagnosis"], horizontal=True, disabled=not find_leaves,
                             format_func={"detector": "Leaf + score", "diagnosis": "Result number"}.get,
                             help="Leaf + score: red boxes labelled like 'leaf 0.86'. "
                                  "Result number: numbered to match the cards, coloured healthy/diseased/uncertain.")
        map_text = f", mAP@0.5 {100 * detector.map_50:.1f}% on held-out field photos" if detector.map_50 is not None else ""
        st.caption(f"Stage 1: SSDLite-MobileNetV3 leaf detector (COCO-initialised, fine-tuned on leaf "
                   f"boxes{map_text}). It is shared by every farm and not part of the mesh.")

try:
    session = get_session(chosen["path"])
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

left, right = st.columns(2)
with left:
    upload_tab, camera_tab = st.tabs(["Upload a photo", "Use the camera"])
    with upload_tab:
        uploaded = st.file_uploader("Leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    with camera_tab:
        if find_leaves:
            # Live view: the leaf detector runs in the browser on every frame; "Capture & diagnose"
            # sends the sharp full-resolution frame here for the two-stage diagnosis.
            captured = live_camera(detector, leaf_threshold)
            st.caption("Hold the camera so the leaves are in view; the red boxes show what the "
                       "detector finds. Then press Capture & diagnose.")
        else:
            captured = st.camera_input("Capture a leaf")
    # Diagnose whichever photo is newest: a fresh capture replaces an earlier upload and vice versa.
    for source, item in (("upload", uploaded), ("camera", captured)):
        item_id = getattr(item, "file_id", None) if item is not None else None
        if item_id is not None and st.session_state.get(f"last_{source}_id") != item_id:
            st.session_state[f"last_{source}_id"] = item_id
            st.session_state["photo_source"] = source
    newest = captured if st.session_state.get("photo_source") == "camera" else uploaded
    photo = newest or uploaded or captured
    if photo is not None and find_leaves:
        image = Image.open(photo).convert("RGB")
        zoom = None
        leaves = leaf_detection.detect_leaves(detector, image, leaf_threshold)
        if not leaves:
            st.warning("No leaf found at this threshold, so the whole photo is classified as one leaf. "
                       "Try a lower threshold, or turn leaf detection off and use the zoom.")
        photo_slot = st.empty()  # filled once each leaf's result (and box colour) is known
    elif photo is not None:
        # The model was trained on PlantVillage photos: one leaf filling the frame on a plain
        # background. A webcam shot is mostly background, which the model misreads (e.g. as
        # strawberry leaf scorch), so zoom into the centre where the leaf is.
        zoom = st.slider("Zoom to the centre (the leaf should fill the box)", 0.3, 1.0,
                         1.0 if uploaded else 0.5, 0.05)
        image = Image.open(photo).convert("RGB")
        w, h = image.size
        side = int(min(w, h) * zoom)
        image = image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        st.image(image, caption="What the model sees", width="stretch")

with right:
    st.subheader("Result")
    if photo is None:
        st.info("Upload a leaf photo (or take one) to see the prediction.")
    elif find_leaves:
        photo_id = f"{getattr(photo, 'file_id', photo.name)}@leaves@{leaf_threshold}@{chosen['path']}"
        if st.session_state.get("last_photo_id") != photo_id:
            try:
                targets = [leaf_detection.crop_leaf(image, leaf) for leaf in leaves] or [image]
                results = [inference.predict(session, target) for target in targets]
            except Exception as e:
                st.error(f"Could not run detection on this photo: {e}")
                st.session_state.pop("last_leaf_results", None)  # never show another photo's results
            else:
                captured_at = datetime.now().isoformat(timespec="seconds")
                for result in results:  # one logged detection per leaf
                    db.save_detection(
                        DB_PATH, captured_at,
                        result["predicted_crop"], result["crop_confidence"],
                        result["predicted_disease"], result["disease_confidence"], result["tier"],
                    )
                st.session_state["last_photo_id"] = photo_id
                st.session_state["last_leaf_results"] = (targets, results)

        targets, results = st.session_state.get("last_leaf_results", ([], []))
        if results:
            if leaves:
                found = f"{len(leaves)} leaf found" if len(leaves) == 1 else f"{len(leaves)} leaves found"
                note = "numbers match the results" if box_style == "diagnosis" else "results below are in the same order, best score first"
                photo_slot.image(leaf_detection.draw_leaves(image, leaves, [r["tier"] for r in results], box_style),
                                 caption=f"{found} ({note})", width="stretch")
            else:
                photo_slot.image(image, caption="What the model sees (no leaf found)", width="stretch")
            counts = {tier: sum(r["tier"] == tier for r in results) for tier in TIER_STYLE}
            st.markdown(" · ".join(f"**{n}** {TIER_STYLE[t][1].lower()}" for t, n in counts.items() if n))
            for number, (target, result) in enumerate(zip(targets, results), start=1):
                color, label = TIER_STYLE[result["tier"]]
                thumb, card = st.columns([1, 3])
                thumb.image(target, caption=f"Leaf {number}" if leaves else "Whole photo", width="stretch")
                score = f" · leaf {leaves[number - 1].score:.2f}" if leaves else ""
                card.markdown(
                    f"""
                    <div style="padding:0.8em;border-radius:0.5em;background-color:{color};color:white;margin-bottom:0.6em;">
                        <b style="font-size:1.1em;">{number}. {label}</b>{score}<br>
                        <b>Crop:</b> {pretty(result['predicted_crop'])} ({result['crop_confidence'] * 100:.1f}%)<br>
                        <b>Condition:</b> {pretty(result['predicted_disease'])} ({result['disease_confidence'] * 100:.1f}%)
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        photo_id = f"{getattr(photo, 'file_id', photo.name)}@{zoom}@{chosen['path']}"
        if st.session_state.get("last_photo_id") != photo_id:
            try:
                result = inference.predict(session, image)
            except Exception as e:
                st.error(f"Could not run detection on this photo: {e}")
            else:
                db.save_detection(
                    DB_PATH, datetime.now().isoformat(timespec="seconds"),
                    result["predicted_crop"], result["crop_confidence"],
                    result["predicted_disease"], result["disease_confidence"], result["tier"],
                )
                st.session_state["last_photo_id"] = photo_id
                st.session_state["last_result"] = result

        result = st.session_state.get("last_result")
        if result is not None:
            color, label = TIER_STYLE[result["tier"]]
            st.markdown(
                f"""
                <div style="padding:1.2em;border-radius:0.5em;background-color:{color};color:white;">
                    <h2 style="margin-top:0;color:white;">{label}</h2>
                    <p style="font-size:1.15em;"><b>Crop:</b> {pretty(result['predicted_crop'])}
                        ({result['crop_confidence'] * 100:.1f}%)</p>
                    <p style="font-size:1.15em;"><b>Condition:</b> {pretty(result['predicted_disease'])}
                        ({result['disease_confidence'] * 100:.1f}%)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**Top 3 conditions**")
            for name, prob in result["top_diseases"]:
                st.progress(prob, text=f"{pretty(name)}: {prob * 100:.1f}%")

st.subheader("Recent detections")
rows = db.get_recent(DB_PATH, limit=10)
if rows:
    st.dataframe(rows, width="stretch")
else:
    st.caption("No detections logged yet.")
