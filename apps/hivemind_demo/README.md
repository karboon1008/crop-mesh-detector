# HiveMind crop-disease demo

Pick a bundled sample, upload a leaf photo or use the camera; the HiveMind-trained model
predicts the crop and whether the leaf is healthy or diseased (and which disease).

## Using the dashboard

1. **Choose a leaf photo** with *Sample photo*, *Upload* or *Camera*. Samples are held-out
   photos with a known label, so the app says whether the model got them right.
2. **Check what the model sees:** the detected leaf boxes, or (with leaf detection off) the
   centre crop, adjusted with the zoom slider until the leaf fills the box.
3. **Read the result:** Healthy, Diseased or Uncertain, with the top 3 crops and conditions.
   A result is **Uncertain** when the crop or condition confidence is below 60%, or when
   the predicted crop/condition pair never occurs in PlantVillage (e.g. apple + common rust),
   which means the model's two outputs disagree. Uncertain results never send a phone alert.

Every detection is logged in *Recent detections* with the model that made it. Use
*Clear history* before a demo to start fresh.

## Run it

**Docker** (after installing Docker Desktop):

```bash
cd apps/hivemind_demo
docker compose up --build
```

Then open <http://localhost:8502>. Stop it with `Ctrl+C`, then `docker compose down`.

**Without Docker** (the backup, already tested on this laptop): double-click `run_demo.bat`, or

```bash
cd apps/hivemind_demo
python -m streamlit run app.py
```

Then open <http://localhost:8501>.

## The models

Choose them in the sidebar. There are 18 models: the final model of each of the 6 farms for each of
the 3 architectures, from the HiveMind continual run (`kb-fresh-01`, commit 03477e7; PlantVillage,
13 rounds). All are ONNX with int8 linear heads, cover 14 crops and 21 disease states, and run on CPU.

Held-out accuracy means crop and disease both right, on 25 unseen test images per class across all 38
classes, scored with the app's own preprocessing. The default is the best model, EfficientNet-Lite0
farm 0.

| Architecture | farm 0 | farm 1 | farm 2 | farm 3 | farm 4 | farm 5 | Size |
|---|---|---|---|---|---|---|---|
| EfficientNet-Lite0 | 75.4% | 69.3% | 62.5% | 56.5% | 64.1% | 61.6% | 13.6 MB |
| MobileNetV3-Small | 51.6% | 64.9% | 56.1% | 52.7% | 61.3% | 59.3% | 6.1 MB |
| MobileViT-XXS | 71.7% | 60.0% | 58.0% | 59.7% | 62.5% | 57.1% | 2.6 MB |

Each farm mostly saw its own mix of crops, so farm models differ a lot on crops they rarely saw.
The sample images were checked on the default model only.

## Two-stage mode: find every leaf, then classify each one

The farm models only ever saw lab photos of one leaf on a plain background. For a field
photo with several leaves, the app can run a **leaf detector first** (stage 1): it boxes
every leaf, crops each one to a padded square, and the chosen farm model classifies each
crop (stage 2). The result shows the photo with numbered boxes coloured healthy / diseased /
uncertain, and one result card per leaf. Each leaf is logged as its own detection.

The detector is SSDLite320-MobileNetV3: COCO-pretrained weights, fine-tuned with a single
"leaf" class. It is shared by every farm and is **not** part of the mesh, so nothing about
the HiveMind knowledge sharing changes.

**Included.** `models/detector/` ships the trained detector: fine-tuned on the PlantDoc
object-detection set (2,344 training photos), with **mAP@0.5 89.0%** (precision 85%, recall 81% at
the 0.5 threshold) on its 236 held-out field photos. It takes ~10-20 ms per photo on a laptop
CPU. The same files are in `outputs/pi_export/` for the Raspberry Pi. Without them, the sidebar
says leaf detection is off, and the app works exactly as before (centre-zoom, one leaf).

**Retraining it.** From the repo root (`scripts/download_plantdoc_od.py` fetches the dataset;
any Pascal VOC folder of leaf boxes works, and every box counts as "leaf"):

```bash
pip install -r requirements.txt   # the repo's training requirements (torch, torchvision)
python scripts/download_plantdoc_od.py
python -m src.detection.train_detector --train-dir <PlantDoc>/TRAIN --test-dir <PlantDoc>/TEST
python scripts/export_detector_for_pi.py --output-dir apps/hivemind_demo/models/detector
```

Training starts from torchvision's COCO weights (downloaded on first run), keeps the epoch
with the best mAP@0.5 on `--test-dir`, and writes `outputs/detector/detector_metrics.json`.
The export writes both files straight into the app; its mAP@0.5 and score threshold appear
in the sidebar. Restart the app (or `docker compose up --build`) to pick it up.

**Controls** (sidebar, only shown once a detector is present):
- **Find every leaf first**: switch between two-stage and the original single-leaf view.
- **Leaf confidence threshold**: lower it if leaves are missed, raise it if background is
  boxed. If nothing passes, the whole photo is classified as one leaf, with a warning.

## Stopping false "leaf" boxes on people, faces and objects

Every PlantDoc training photo contains leaves, so the shipped detector never learned what is
*not* a leaf. It tends to box the main object of any photo: on 17 leaf-free test photos it
boxed 15 (88%), including a face (0.59), a cup (0.96) and a cat (0.99).

**Built in now: the plant-colour check** (sidebar toggle *Only plant-coloured boxes*, on by
default). It drops boxes whose pixels are mostly not vegetation-coloured. That cut the leaf-free
false alarms to 12% while keeping every sample leaf, but it **cannot tell faces from leaves**.

**The real fix: retrain with "no leaf" photos** (people, faces, hands, the room). They need no
labelling.

**All in Docker (no Python on your computer):**
1. **Collect:** in the demo app's sidebar, turn on **Retraining photos > Collect no-leaf photos**.
   Open **Use the camera**, press **Start collecting** and move around with no leaves in view.
   The photos land in `apps/hivemind_demo/data/negatives/<folder>/` on your computer.
2. **Retrain and export:** follow [`docker/detector-training/README.md`](../../docker/detector-training/README.md).

**Or with Python installed**, about 15 minutes on a laptop CPU, from the repo root:

```bash
# 1. Collect ~300-500 frames with NO leaves or plants in view: move around, faces near and far,
#    several people, green clothes, the desk and walls. Do it in the demo room if you can.
python scripts/capture_negatives.py --out data/negatives/room --count 400
#    (any other leaf-free photos can go in data/negatives/ too; delete frames showing a plant)

# 2. How bad is the current model on them? (the "before" number)
python scripts/check_false_alarms.py data/negatives

# 3. Fine-tune the existing detector with the negatives (15% are held out to score false alarms)
python -m src.detection.train_detector --train-dir data/PlantDoc-OD/TRAIN --test-dir data/PlantDoc-OD/TEST \
    --negatives-dir data/negatives --init-checkpoint outputs/detector/leaf_ssdlite.pt \
    --checkpoint outputs/detector/leaf_ssdlite_neg.pt --epochs 10

# 4. Export it into the app (and for the Pi), then check the "after" number
python scripts/export_detector_for_pi.py --checkpoint outputs/detector/leaf_ssdlite_neg.pt \
    --output-dir apps/hivemind_demo/models/detector
python scripts/check_false_alarms.py data/negatives
```

Each epoch prints mAP@0.5 and the false-alarm rate on the held-out negatives. The kept epoch is
the one with the best mAP@0.5 x (1 - false-alarm rate). Photos you then use to test the demo
must not be in the negatives folder, or the "after" number will look better than it is.

## Phone alerts (ntfy, free)

When a photo or camera capture has a **diseased** leaf, the dashboard sends a push notification to
your phone through **ntfy** (https://ntfy.sh), a free push service. No Apple developer account is
needed, and it works with the phone locked. The notification has:
- the diagnosis of each diseased leaf,
- the field's name and coordinates, and tapping it opens Apple Maps there,
- a **Navigate to field** button,
- the photo with the leaves boxed.

**Set it up once:**
1. On the phone, install **ntfy** (free, App Store / Google Play).
2. On the dashboard, the sidebar's **Phone alerts** section shows a topic like
   `hivemind-farm-3fa9c1b2d4`. In the ntfy app, tap **+**, enter that topic (server
   `ntfy.sh`), and allow notifications.
3. Press **Send a test alert**. The phone should buzz within a second or two.

Set the field's name and coordinates in the same section (default: a demo greenhouse in Cameron
Highlands), or in `docker-compose.yml` via `FIELD_NAME`, `FIELD_LAT`, `FIELD_LON`. The topic is
created once and kept in `data/ntfy_topic.txt`; set `NTFY_TOPIC` to fix it. Anyone who knows the
topic can read the alerts on ntfy.sh, so keep it long and random. The dashboard needs internet
access to send.

## HiveMind Farmer iPhone app

`docker compose up --build` also starts the **farmer alerts server** (port 8080), the backend of
the iPhone app in `mobile/ios/`. Each diseased leaf the dashboard finds is reported to it as field
camera `node_0`, with the field's coordinates and the leaf photo, and shows up in the app.

1. Run the app in the Xcode **iPhone Simulator** (see `mobile/ios/README.md`).
2. In the app: **Settings > Demo mode off**. The defaults are already the Docker setup: server
   `http://localhost:8080`, farm code `demo-farm-token`. Tap **Connect**.
3. Allow notifications (**Settings > Turn on disease alerts**).
4. Diagnose a diseased leaf on the dashboard. Within ~5 s the app shows it in **Alerts** and on
   the **Map**, and pops a **HiveMind Farmer** notification. Tap it for the location, photo and
   treatment steps.

The app checks the server every 5 s while it is open. Without a paid Apple developer account it
can't be woken when closed, so for a locked phone use the ntfy alerts above. On a real iPhone,
set the server to your Mac's network address, e.g. `http://192.168.1.23:8080`.

## Live camera

With leaf detection on, **Use the camera** is a live view: the leaf detector runs in the browser
on every frame and draws red `leaf 0.86` boxes as you move the camera. **Capture & diagnose**
sends the sharp, full-resolution frame through the same two-stage diagnosis as an uploaded photo.
**Switch camera** toggles front/back on phones and tablets.

- **Where it runs:** detection happens in the browser (onnxruntime-web, bundled in
  `live_camera_web/vendor/`, no internet needed). It works the same under Docker: no video is
  streamed to the container.
- **Secure page:** browsers only allow the camera on a secure page. Open the demo at
  `http://localhost:8501` / `:8502` on the same machine, not at a network address.
- **Background blur:** if the laptop blurs the camera background, the leaves behind you vanish
  for the detector. The page asks the browser to switch blur off, and shows a warning with where
  to turn it off when the system forces it (Windows: Settings > Bluetooth & devices > Cameras or
  Studio Effects; Mac: Control Centre > Video Effects > Portrait/Background off).
- **Fallback:** with leaf detection switched off, the tab falls back to the plain snapshot camera
  with centre-zoom.

## Demo tips

- **Two-stage mode:** show a photo with several leaves and point out the numbered boxes and
  per-leaf verdicts. The per-leaf crops still include some background, so the
  PlantVillage-trained classifier is less reliable on them than on lab photos. Rehearse with
  the photos you plan to show.

- **Using the camera:** hold the leaf close so it fills the frame, ideally against plain
  white paper. Then move the **"Zoom to the centre"** slider until the leaf fills the
  "What the model sees" box. A small leaf with lots of desk around it gets misread,
  usually as strawberry leaf scorch. The model only saw full-frame lab photos, so it
  never learned to ignore the background.

- `sample_images/` holds 26 held-out photos, never trained on by any farm, shown under
  *Sample photo* (the label is read from the file name: `healthy_*.jpg`, `diseased_*.jpg`).
  The default model gets all 26 right, with leaf detection on or off.
- Reliable classes (>= 88% on held-out images):
  - Apple: healthy, cedar rust, black rot
  - Tomato: healthy, yellow leaf curl virus, mosaic virus
  - Peach: healthy, bacterial spot
  - Pepper: healthy, bacterial spot
  - Blueberry healthy, squash powdery mildew, orange citrus greening
  - Corn common rust
- Avoid live: corn northern leaf blight (0%), grape healthy (12%),
  corn/potato/strawberry healthy (24-28%).
- The model was trained on PlantVillage lab photos: a single leaf on a plain
  background. Random internet or field photos (cluttered backgrounds) are a
  different domain and will often be wrong. Use PlantVillage-style images.
