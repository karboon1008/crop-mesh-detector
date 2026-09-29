"""HiveMind crop-disease demo -- pick a sample, upload a leaf photo or use the webcam,
and the HiveMind-trained ONNX model predicts the crop and the disease.

Run locally:   streamlit run app.py        -> http://localhost:8501
Run in Docker: docker compose up --build   -> http://localhost:8502
"""
from __future__ import annotations

import html
import io
import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

import db
import farm_server
import inference
import leaf_detection
import phone_alerts
from live_camera import live_camera

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "detections.db"
SAMPLE_DIR = HERE / "sample_images"
# Leaf-free photos collected for retraining the leaf detector. data/ is the Docker volume, so
# they land in apps/hivemind_demo/data/negatives/ on the host.
NEGATIVES_DIR = HERE / "data" / "negatives"

TIER_STYLE = {
    "healthy": ("#2e7d32", "✅", "Healthy"),
    "diseased": ("#c62828", "⚠️", "Diseased"),
    "uncertain": ("#ef6c00", "❔", "Uncertain"),
}
TIER_LABEL = {tier: f"{icon} {label}" for tier, (_, icon, label) in TIER_STYLE.items()}

# Labels whose raw class name reads badly once underscores are replaced.
DISPLAY_NAMES = {
    "Pepper,_bell": "Bell pepper",
    "Cercospora_leaf_spot Gray_leaf_spot": "Cercospora leaf spot (gray leaf spot)",
    "Spider_mites Two-spotted_spider_mite": "Spider mites (two-spotted)",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (citrus greening)",
    "Esca_(Black_Measles)": "Esca (black measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis leaf spot)",
}

# The 38 crop/condition pairs in PlantVillage. The model predicts crop and condition with
# two separate heads, so it can answer a pair that never occurs (e.g. Apple + Common rust):
# that means the heads disagree, and the answer is shown as uncertain rather than trusted.
KNOWN_PAIRS = {
    "Apple": {"Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"},
    "Blueberry": {"healthy"},
    "Cherry": {"Powdery_mildew", "healthy"},
    "Corn": {"Cercospora_leaf_spot Gray_leaf_spot", "Common_rust", "Northern_Leaf_Blight", "healthy"},
    "Grape": {"Black_rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "healthy"},
    "Orange": {"Haunglongbing_(Citrus_greening)"},
    "Peach": {"Bacterial_spot", "healthy"},
    "Pepper,_bell": {"Bacterial_spot", "healthy"},
    "Potato": {"Early_blight", "Late_blight", "healthy"},
    "Raspberry": {"healthy"},
    "Soybean": {"healthy"},
    "Squash": {"Powdery_mildew"},
    "Strawberry": {"Leaf_scorch", "healthy"},
    "Tomato": {"Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
               "Spider_mites Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
               "Tomato_mosaic_virus", "healthy"},
}

ARCH_NAMES = {"efficientnet_lite0": "EfficientNet-Lite0", "mobilenet_v3_small": "MobileNetV3-Small",
              "mobilevit_xxs": "MobileViT-XXS"}
# every farm's final model from the HiveMind continual run, with its held-out accuracy
# (crop and disease both right, 25 unseen images per class x 38 classes)
MODELS = json.loads((HERE / "models" / "index.json").read_text())
BEST = max(MODELS, key=lambda m: m["heldout_acc"])

SOURCES = {"sample": "🖼️ Sample photo", "upload": "📁 Upload", "camera": "📷 Camera"}


def pretty(label: str) -> str:
    if label in DISPLAY_NAMES:
        return DISPLAY_NAMES[label]
    return label.replace("_", " ").replace(",", "").strip().capitalize()


def farm_no(model: dict) -> str:
    return model["node"].split("_")[1]


def model_name(model: dict) -> str:
    return f"{ARCH_NAMES[model['arch']]} · farm {farm_no(model)}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@st.cache_data
def load_samples() -> list[dict]:
    """The bundled held-out photos, with the true crop/condition parsed from the file name
    (e.g. diseased_pepperbell_bacterial_spot_1.jpg -> Pepper,_bell + Bacterial_spot)."""
    samples = []
    for path in sorted(SAMPLE_DIR.glob("*.jpg")):
        _, _, rest = path.stem.partition("_")
        rest, _, number = rest.rpartition("_")
        crop = next((c for c in inference.CROP_CLASSES if _norm(rest).startswith(_norm(c))), None)
        if crop is None:
            continue
        condition_key = _norm(rest)[len(_norm(crop)):]
        condition = next((d for d in inference.DISEASE_CLASSES if _norm(d) == condition_key), None)
        if condition is None:
            continue
        samples.append({"path": path, "crop": crop, "disease": condition, "number": number})
    return samples


def sample_label(sample: dict) -> str:
    state = "Healthy" if sample["disease"] == "healthy" else pretty(sample["disease"])
    return f"{pretty(sample['crop'])} · {state} (#{sample['number']})"


@st.cache_resource
def get_session(path: str):
    return inference.create_session(HERE / path)


@st.cache_resource
def get_detector():
    return leaf_detection.load_detector()


def diagnose(session, image: Image.Image) -> dict:
    """inference.predict, plus a check that the crop/condition pair can exist at all."""
    result = inference.predict(session, image)
    known = KNOWN_PAIRS.get(result["predicted_crop"])
    result["pair_known"] = known is None or result["predicted_disease"] in known
    if not result["pair_known"]:
        result["tier"] = "uncertain"
    return result


def uncertain_reason(result: dict) -> str | None:
    if not result["pair_known"]:
        return (f"The model contradicts itself: {pretty(result['predicted_crop']).lower()} with "
                f"{pretty(result['predicted_disease']).lower()} never appears in its training data.")
    if result["tier"] == "uncertain":
        return (f"Low confidence: the crop or condition score is below "
                f"{inference.CONFIDENCE_THRESHOLD:.0%}.")
    return None


def result_card(result: dict, heading: str = "", big: bool = True) -> None:
    color, icon, label = TIER_STYLE[result["tier"]]
    condition = "No disease found" if result["predicted_disease"] == "healthy" else pretty(result["predicted_disease"])
    st.markdown(
        f"""
        <div style="padding:{'1.1em 1.3em' if big else '0.7em 1em'};border-radius:0.6em;background-color:{color};color:white;margin-bottom:0.6em;">
            <div style="font-size:{'1.8em' if big else '1.2em'};font-weight:700;line-height:1.2;">{html.escape(heading)}{icon} {label}</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.3em 2em;margin-top:0.4em;">
                <div><div style="opacity:0.85;font-size:0.8em;">Crop</div>
                    <b>{html.escape(pretty(result['predicted_crop']))}</b> · {result['crop_confidence']:.0%}</div>
                <div><div style="opacity:0.85;font-size:0.8em;">Condition</div>
                    <b>{html.escape(condition)}</b> · {result['disease_confidence']:.0%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top3(result: dict) -> None:
    crops_col, diseases_col = st.columns(2)
    with crops_col:
        st.markdown("**Top 3 crops**")
        for name, prob in result["top_crops"]:
            st.progress(prob, text=f"{pretty(name)}: {prob:.1%}")
    with diseases_col:
        st.markdown("**Top 3 conditions**")
        for name, prob in result["top_diseases"]:
            st.progress(prob, text=f"{pretty(name)}: {prob:.1%}")


st.set_page_config(page_title="HiveMind Crop Disease Detection", page_icon="🌿", layout="wide")
db.init_db(DB_PATH)
detector = get_detector()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    archs = list(ARCH_NAMES)
    arch = st.selectbox(
        "Architecture", archs, index=archs.index(BEST["arch"]),
        format_func=lambda a: f"{ARCH_NAMES[a]} (up to {max(m['heldout_acc'] for m in MODELS if m['arch'] == a):.0%})",
        help="The three network types HiveMind trained. Percentages are the best farm's held-out accuracy.",
    )
    farms = sorted((m for m in MODELS if m["arch"] == arch), key=lambda m: m["node"])
    default = max(range(len(farms)), key=lambda i: farms[i]["heldout_acc"])
    chosen = st.selectbox(
        "Farm", farms, index=default,
        format_func=lambda m: f"Farm {farm_no(m)}: {m['heldout_acc']:.1%}"
                              + ("  ★ recommended" if m["path"] == BEST["path"] else ""),
        help="Each farm's final model. The percentage is its held-out accuracy "
             "(crop and condition both right on unseen photos).",
    )
    if chosen["path"] != BEST["path"]:
        st.caption(f"★ The most accurate model is {model_name(BEST)} ({BEST['heldout_acc']:.1%}).")
    st.markdown(f"**Held-out accuracy:** {chosen['heldout_acc']:.1%}  \n**Size:** {chosen['size_mb']} MB")

    st.header("Leaf detection")
    if detector is None:
        find_leaves = False
        st.caption("Off: no leaf detector in models/detector/. Without it, the whole (zoomed) photo "
                   "is classified as one leaf. See README.md to add one.")
    else:
        find_leaves = st.toggle("Find every leaf first (two-stage)", value=True,
                                help="Stage 1 finds each leaf; stage 2 classifies each crop with the farm model. "
                                     "Turn it off to classify the whole photo as one leaf, with a zoom slider.")
        leaf_threshold = st.slider("Leaf confidence threshold", 0.1, 0.9, detector.default_threshold, 0.05,
                                   disabled=not find_leaves,
                                   help="Lower it if leaves are missed, raise it if background is boxed.")
        colour_check = st.toggle("Only plant-coloured boxes", value=True, disabled=not find_leaves,
                                 help="Skips boxes whose pixels are mostly not leaf-coloured (cups, pets, sky, "
                                      "walls). It can't reliably tell faces from leaves; retraining the detector "
                                      "with no-leaf photos fixes that (see README).")
        min_plant = leaf_detection.MIN_PLANT_FRACTION if colour_check else 0.0
        box_style = st.radio("Boxes show", ["detector", "diagnosis"], horizontal=True, disabled=not find_leaves,
                             format_func={"detector": "Leaf + score", "diagnosis": "Result number"}.get,
                             help="Leaf + score: red boxes labelled like 'leaf 0.86'. "
                                  "Result number: numbered to match the cards, coloured healthy/diseased/uncertain.")

    st.header("Phone alerts")
    alerts_on = st.toggle("Alert my phone when a disease is found", value=True,
                          help="Sends a push notification through the free ntfy app, with the diagnosis, "
                               "the field's coordinates, a Navigate button and the photo. "
                               "Uncertain results never send an alert.")
    ntfy_server = phone_alerts.default_server()
    with st.expander("Alert settings"):
        ntfy_topic = st.text_input("ntfy topic", phone_alerts.default_topic(HERE / "data"),
                                   help="Anyone who knows this name can read the alerts, so keep it long and random.")
        default_name, default_lat, default_lon = phone_alerts.default_field()
        field_name = st.text_input("Field name", default_name)
        lat_col, lon_col = st.columns(2)
        field_lat = lat_col.number_input("Latitude", -90.0, 90.0, default_lat, format="%.5f")
        field_lon = lon_col.number_input("Longitude", -180.0, 180.0, default_lon, format="%.5f")
        st.caption(f"**On the phone:** install the free **ntfy** app, tap **+** and subscribe to "
                   f"`{ntfy_topic}` (server {ntfy_server.removeprefix('https://')}). Allow notifications.")
        server = farm_server.config()
        st.caption(f"Also sent to the **HiveMind Farmer** app through the alerts server at `{server[0]}` "
                   f"(as field camera `{server[1]}`)." if server else
                   "The HiveMind Farmer app isn't connected: set ALERTS_URL (docker-compose.yml does).")
        if st.button("Send a test alert", disabled=not ntfy_topic):
            try:
                phone_alerts.send(ntfy_server, ntfy_topic, "HiveMind test alert",
                                  f"Phone alerts work. Field: {field_name} ({field_lat:.5f}, {field_lon:.5f})",
                                  field_lat, field_lon, field_name)
                st.success("Test alert sent. Check the phone.")
            except Exception as e:
                st.error(f"Could not send: {e}")
    if alerts_on and not ntfy_topic:
        st.warning("Alerts are on but the ntfy topic is empty. Set one under *Alert settings*.")

    collect_mode = False
    if detector is not None:
        st.header("Retraining photos")
        collect_mode = st.toggle(
            "Collect no-leaf photos", value=False,
            help="Saves webcam frames with NO leaves in them (people, faces, hands, the room) for "
                 "retraining the leaf detector so it stops boxing them. See README.md.")
        if collect_mode:
            collect_target = int(st.number_input("Photos to collect", 50, 3000, 400, 50))
            collect_folder = st.text_input("Folder name", "room").strip() or "room"
            collect_folder = "".join(c for c in collect_folder if c.isalnum() or c in "-_") or "room"
            collect_dir = NEGATIVES_DIR / collect_folder
            st.caption("Open **📷 Camera** to start collecting.")

    with st.expander("Tips for a good result"):
        st.markdown(
            """
- Photograph **one leaf** that fills the frame, ideally on plain white paper.
- With leaf detection off, use the **Zoom** slider until the leaf fills the
  *What the model sees* box.
- The farm models learned from PlantVillage lab photos. Field or internet photos with busy
  backgrounds are often misread.
- **Most reliable (≥ 88%):** apple (healthy, cedar rust, black rot), tomato (healthy,
  yellow leaf curl, mosaic virus), peach and bell pepper (healthy, bacterial spot),
  blueberry healthy, squash powdery mildew, orange citrus greening, corn common rust.
- **Weak:** corn northern leaf blight, grape healthy, and corn/potato/strawberry healthy.
"""
        )


def send_phone_alert(results: list[dict], image: Image.Image, leaf_images: list[Image.Image]) -> None:
    """For a diagnosed photo with at least one diseased leaf: one ntfy push, and each diseased
    leaf reported to the alerts server so it appears in the HiveMind Farmer app.
    """
    if not alerts_on or not any(r["tier"] == "diseased" for r in results):
        return
    server = farm_server.config()
    if server:
        try:
            farm_server.report(*server, results, leaf_images, field_lat, field_lon)
        except Exception as e:
            st.warning(f"Not sent to the HiveMind Farmer app (alerts server {server[0]}): {e}")
    alert = phone_alerts.build_alert(results, field_name, field_lat, field_lon)
    if alert is None or not ntfy_topic:
        return
    photo = image.convert("RGB")
    photo.thumbnail((1280, 1280))
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=85)
    try:
        phone_alerts.send(ntfy_server, ntfy_topic, *alert, field_lat, field_lon, field_name, buf.getvalue())
        st.toast("Phone alert sent", icon="📱")
    except Exception as e:  # no internet etc.: the diagnosis still shows, only the push is lost
        st.warning(f"Phone alert not sent: {e}")


try:
    session = get_session(chosen["path"])
except (FileNotFoundError, ValueError) as e:
    st.error(f"The selected model could not be loaded: {e}")
    st.stop()

# ---------------------------------------------------------------- header
st.title("🌿 HiveMind: Crop Disease Detection")
st.caption("**1.** Choose a leaf photo → **2.** check what the model sees → **3.** read the result. "
           "Every detection is logged at the bottom of the page.")

left, right = st.columns([1, 1], gap="large")

# ---------------------------------------------------------------- 1. input
image = None      # the photo (or its centre crop, with leaf detection off)
photo_key = None  # identifies the photo, so it is diagnosed, logged and alerted only once
expected = None   # the true label, for sample photos
leaves: list = []
zoom = None
with left:
    st.subheader("1 · Choose a leaf photo")
    source = st.segmented_control("Photo source", list(SOURCES), default="sample", format_func=SOURCES.get,
                                  key="source", label_visibility="collapsed", required=True) or "sample"

    photo = None
    if source == "sample":
        samples = load_samples()
        if not samples:
            st.info("No sample photos found in `sample_images/`. Upload a photo or use the camera instead.")
        sample = st.selectbox(
            "Sample photo", samples, index=None, format_func=sample_label,
            placeholder=f"Pick one of {len(samples)} held-out photos…",
            help="Photos no farm ever trained on. The true label is known, so you can check the model.",
        )
        if sample is not None:
            photo, photo_key, expected = sample["path"], f"sample:{sample['path'].name}", sample
    elif source == "upload":
        photo = st.file_uploader("Leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if photo is not None:
            photo_key = f"upload:{photo.file_id}"
    elif collect_mode:
        collect_dir.mkdir(parents=True, exist_ok=True)
        saved = len(list(collect_dir.glob("*.jpg")))
        st.info("**Collecting no-leaf photos.** Keep all leaves and plants out of view. After "
                "Start collecting (5 s countdown) a photo is saved every 0.5 s: move your face "
                "near and far, hold up empty hands, bring in other people, show green clothes, "
                "the desk, walls and windows. Red boxes you see now are the false alarms "
                "these photos will fix.")
        captured = live_camera(detector, leaf_threshold, min_plant_fraction=min_plant, collect=True,
                               collected=saved, collect_target=collect_target, key="collect_camera")
        if captured is not None and captured.collect and saved < collect_target \
                and st.session_state.get("last_collect_id") != captured.file_id:
            st.session_state["last_collect_id"] = captured.file_id
            (collect_dir / f"neg-{captured.file_id}.jpg").write_bytes(captured.getvalue())
            saved += 1
        st.progress(min(saved / collect_target, 1.0),
                    text=f"{saved} of {collect_target} photos saved in "
                         f"`apps/hivemind_demo/data/negatives/{collect_folder}/`")
        st.caption("Delete any that show a plant, then retrain (README.md). Collected frames are for "
                   "retraining, not diagnosis: turn off *Collect no-leaf photos* to diagnose again.")
    elif find_leaves:
        # Live view: the leaf detector runs in the browser on every frame; "Capture & diagnose"
        # sends the sharp full-resolution frame here for the two-stage diagnosis.
        photo = live_camera(detector, leaf_threshold, min_plant_fraction=min_plant)
        st.caption("Hold the camera so the leaves are in view; the red boxes show what the "
                   "detector finds. Then press **Capture & diagnose**.")
        if photo is not None:
            photo_key = f"camera:{photo.file_id}"
    else:
        photo = st.camera_input("Hold the leaf close to the camera, then take the photo")
        if photo is not None:
            photo_key = f"camera:{photo.file_id}"

    if photo is not None:
        try:
            full = Image.open(photo).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be read as an image. Please use a JPG or PNG photo.")
            photo_key = None
        else:
            if find_leaves:
                image = full
                leaves = leaf_detection.detect_leaves(detector, image, leaf_threshold, min_plant_fraction=min_plant)
                if not leaves:
                    st.warning("No leaf found at this threshold, so the whole photo is classified as one leaf. "
                               "Try a lower threshold, or turn leaf detection off and use the zoom.")
                photo_slot = st.empty()  # filled once each leaf's result (and box colour) is known
            else:
                # The model was trained on PlantVillage photos: one leaf filling the frame on a plain
                # background. A webcam shot is mostly background, which the model misreads (e.g. as
                # strawberry leaf scorch), so zoom into the centre where the leaf is.
                zoom = st.slider(
                    "Zoom to the centre", 0.3, 1.0, 0.5 if source == "camera" else 1.0, 0.05,
                    key=f"zoom:{photo_key}", format="%.2f×",
                    help="1.00× uses the whole photo. Lower values crop to the centre. "
                         "Adjust until the leaf fills the box below.",
                )
                w, h = full.size
                side = int(min(w, h) * zoom)
                image = full.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
                st.image(image, caption="What the model sees (square centre crop)", width="stretch")
                if side < inference.IMAGE_SIZE:
                    st.warning(f"The crop is only {side}×{side} px, smaller than the model's "
                               f"{inference.IMAGE_SIZE}×{inference.IMAGE_SIZE} input. Zoom out or use a "
                               "sharper photo for a more reliable result.")

# ---------------------------------------------------------------- 2. result
with right:
    st.subheader("2 · Result")
    diagnosis = None
    if image is None:
        if not (collect_mode and source == "camera"):
            st.info("Choose a sample, upload a photo or use the camera. The prediction appears here.")
    else:
        mode = f"leaves@{leaf_threshold}@{min_plant}" if find_leaves else f"zoom@{zoom}"
        run_key = f"{photo_key}@{mode}@{chosen['path']}"
        # only run (and log, and alert) once per photo, settings and model, not on every rerun
        if st.session_state.get("diagnosis", {}).get("key") != run_key:
            st.session_state.pop("diagnosis", None)  # never show another photo's results
            try:
                with st.spinner("Analysing the leaf…"):
                    targets = [leaf_detection.crop_leaf(image, leaf) for leaf in leaves] or [image]
                    results = [diagnose(session, target) for target in targets]
            except Exception as e:
                st.error(f"Could not run detection on this photo: {e}")
            else:
                captured_at = datetime.now().isoformat(timespec="seconds")
                for result in results:  # one logged detection per leaf
                    db.save_detection(
                        DB_PATH, captured_at,
                        result["predicted_crop"], result["crop_confidence"],
                        result["predicted_disease"], result["disease_confidence"], result["tier"],
                        model_name(chosen),
                    )
                st.session_state["diagnosis"] = {"key": run_key, "targets": targets, "results": results}
                alert_photo = (leaf_detection.draw_leaves(image, leaves, [r["tier"] for r in results], box_style)
                               if leaves else image)
                send_phone_alert(results, alert_photo, targets)
        diagnosis = st.session_state.get("diagnosis")

    if diagnosis is not None:
        targets, results = diagnosis["targets"], diagnosis["results"]
        if find_leaves:
            if leaves:
                if len(leaves) == 1:
                    caption = "1 leaf found"
                else:
                    note = ("numbers match the results" if box_style == "diagnosis"
                            else "results are in the same order, best score first")
                    caption = f"{len(leaves)} leaves found ({note})"
                photo_slot.image(leaf_detection.draw_leaves(image, leaves, [r["tier"] for r in results], box_style),
                                 caption=caption, width="stretch")
            else:
                photo_slot.image(image, caption="What the model sees (no leaf found)", width="stretch")

        if len(results) == 1:
            result_card(results[0])
            if reason := uncertain_reason(results[0]):
                st.warning(f"**Treat this answer as a guess.** {reason} Try a closer, better-lit photo "
                           "of one leaf on a plain background, or another model.")
        else:
            counts = {tier: sum(r["tier"] == tier for r in results) for tier in TIER_STYLE}
            st.markdown(" · ".join(f"**{n}** {TIER_STYLE[t][2].lower()}" for t, n in counts.items() if n))
            for number, (target, result) in enumerate(zip(targets, results), start=1):
                thumb, card = st.columns([1, 3])
                thumb.image(target, caption=f"Leaf {number} · score {leaves[number - 1].score:.2f}", width="stretch")
                with card:
                    result_card(result, heading=f"{number}. ", big=False)
                    if reason := uncertain_reason(result):
                        st.caption(f"❔ {reason}")
                    with st.expander("Top 3 crops and conditions"):
                        top3(result)

        if expected is not None:
            first = results[0]
            right_crop = first["predicted_crop"] == expected["crop"]
            right_disease = first["predicted_disease"] == expected["disease"]
            truth = f"{pretty(expected['crop'])} · {pretty(expected['disease'])}"
            which = "Leaf 1 is" if len(results) > 1 else "This sample is"
            if right_crop and right_disease:
                st.success(f"**Correct.** {which} {truth}.")
            else:
                wrong = " and ".join(p for p, ok in (("crop", right_crop), ("condition", right_disease)) if not ok)
                st.error(f"**Wrong {wrong}.** {which} really {truth}.")

        if len(results) == 1:
            top3(results[0])
        st.caption(f"Predicted by {model_name(chosen)}. Results below {inference.CONFIDENCE_THRESHOLD:.0%} "
                   "confidence are marked uncertain. This is a research demo, not a replacement for an agronomist.")

# ---------------------------------------------------------------- history
st.divider()
rows = db.get_recent(DB_PATH, limit=20)
head, action = st.columns([4, 1], vertical_alignment="bottom")
head.subheader("Recent detections")
if rows:
    with action.popover("Clear history", icon="🗑️", width="stretch"):
        st.write("Delete every logged detection? This cannot be undone.")
        if st.button("Delete all", type="primary"):
            db.clear_detections(DB_PATH)
            st.rerun()
    table = [
        {
            "Time": datetime.fromisoformat(r["captured_at"]),
            "Result": TIER_LABEL.get(r["tier"], r["tier"]),
            "Crop": pretty(r["predicted_crop"]),
            "Crop conf.": r["crop_confidence"],
            "Condition": pretty(r["predicted_disease"]),
            "Condition conf.": r["disease_confidence"],
            "Model": r["model"] or "—",
        }
        for r in rows
    ]
    percent = {"min_value": 0.0, "max_value": 1.0, "format": "percent"}
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="D MMM, HH:mm:ss"),
            "Crop conf.": st.column_config.ProgressColumn(**percent),
            "Condition conf.": st.column_config.ProgressColumn(**percent),
        },
    )
    st.caption(f"The latest {len(rows)} detections, newest first. A photo with several leaves logs one row per leaf.")
else:
    st.caption("No detections logged yet. Your results will appear here.")
