# Leaf-detector training in Docker

Retrain the leaf detector with no Python on your computer. Everything runs in a CPU container;
the repo is mounted into it, so datasets, checkpoints and the exported ONNX stay on your disk.
Run every command **from the repo root**. The first build downloads PyTorch (~1 GB).

```bash
# 0. Build the image (once)
docker compose -f docker/detector-training/docker-compose.yml build

# 1. PlantDoc leaf boxes -> data/PlantDoc-OD/ (skip if you already have them)
docker compose -f docker/detector-training/docker-compose.yml run --rm train \
    python scripts/download_plantdoc_od.py

# 2. "Before": how often the current detector boxes your no-leaf photos
#    (collect them in the demo app: sidebar > Collect no-leaf photos)
docker compose -f docker/detector-training/docker-compose.yml run --rm train \
    python scripts/check_false_alarms.py apps/hivemind_demo/data/negatives

# 3. Retrain with the negatives. With your earlier checkpoint, fine-tune it (fast):
docker compose -f docker/detector-training/docker-compose.yml run --rm train \
    python -m src.detection.train_detector --train-dir data/PlantDoc-OD/TRAIN --test-dir data/PlantDoc-OD/TEST \
    --negatives-dir apps/hivemind_demo/data/negatives \
    --init-checkpoint outputs/detector/leaf_ssdlite.pt --checkpoint outputs/detector/leaf_ssdlite_neg.pt --epochs 10
#    Without it, train from COCO weights instead (drop --init-checkpoint, use --epochs 30).

# 4. Export into the demo app, then the "after" number
docker compose -f docker/detector-training/docker-compose.yml run --rm train \
    python scripts/export_detector_for_pi.py --checkpoint outputs/detector/leaf_ssdlite_neg.pt \
    --output-dir apps/hivemind_demo/models/detector
docker compose -f docker/detector-training/docker-compose.yml run --rm train \
    python scripts/check_false_alarms.py apps/hivemind_demo/data/negatives
```

Then rebuild the demo: `cd apps/hivemind_demo && docker compose up --build`.

Webcam capture with `scripts/capture_negatives.py` does not work inside Docker on Windows or Mac
(Docker Desktop can't pass the camera through). Use the demo app's **Collect no-leaf photos**
mode instead: it uses the camera through the browser.
