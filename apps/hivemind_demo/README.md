# HiveMind crop-disease demo

Upload a leaf photo in the browser; the HiveMind-trained model predicts the crop and
whether the leaf is healthy or diseased (and which disease).

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

## Demo tips

- **Using the camera:** hold the leaf close so it fills the frame, ideally against plain
  white paper. Then move the **"Zoom to the centre"** slider until the leaf fills the
  "What the model sees" box. A small leaf with lots of desk around it gets misread,
  usually as strawberry leaf scorch. The model only saw full-frame lab photos, so it
  never learned to ignore the background.

- `sample_images/` holds 26 held-out photos, never trained on by any farm. Each was
  checked to be predicted correctly: `healthy_*.jpg` and `diseased_*.jpg`.
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
