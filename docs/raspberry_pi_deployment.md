# Deploying to a Raspberry Pi

Runs the best-scoring trained model on a Pi's CPU, classifying images from a
camera in near-real-time — no training stack (torch/torchvision/timm) needed
on the Pi itself.

## 1. Train and export (dev machine or Colab, wherever training ran)

`python -m src.train` supports training one architecture at a time via
`--arch` — useful for splitting a full 3-architecture sweep across several
shorter runs (e.g. to stay under Colab's free-tier GPU usage limits; see the
Colab notebook, which does exactly this). Each run's results merge into
`outputs/results_summary.json` rather than overwriting it:

```bash
python -m src.train --config config.yaml --arch mobilenet_v3_small --fresh  # start a new sweep
python -m src.train --config config.yaml --arch efficientnet_lite0          # merges in
python -m src.train --config config.yaml --arch mobilevit_xxs               # merges in
```

(Omit `--arch` to train every architecture listed in `config.yaml` in one
run instead, if you're not fighting a GPU quota.)

After that, `scripts/export_for_pi.py` converts trained checkpoints to
ONNX + int8-quantized bundles a Pi can run:

```bash
pip install onnx onnxruntime   # already in requirements.txt
python scripts/export_for_pi.py --all
```

`--all` exports **every** architecture present in `results_summary.json`
(each using its own best-scoring node) into its own subfolder:

```
outputs/pi_export/
├── mobilenet_v3_small/{model.onnx, manifest.json}
├── efficientnet_lite0/{model.onnx, manifest.json}
└── mobilevit_xxs/{model.onnx, manifest.json}
```

Each `model.onnx` is a few MB to a few tens of MB; `manifest.json` has the
class names, image size, and normalization constants. Copy whichever
subfolder(s) you want to try onto the Pi and compare real latency/accuracy
there — you don't have to commit to one in advance.

For a single specific model instead of all three:

```bash
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0
```

(Omitting both `--arch`/`--node`/`--all` picks the single best-scoring
architecture+node overall.)

Every export runs a parity check against a handful of real images from
`data/PlantVillage` (if present) to confirm the quantized ONNX model agrees
with the original PyTorch model's predictions before you ever touch hardware.

**If you ran training on Colab**: run the export step there too, right
after training, then download only the small `outputs/pi_export/` folder(s)
instead of the full `outputs/` — the dataset and checkpoints don't need to
leave Colab at all.

## 2. Set up the Pi

- Flash **Raspberry Pi OS (64-bit)** with Raspberry Pi Imager.
- If using the official Camera Module: `sudo raspi-config` → Interface Options
  → enable Camera → reboot.
- Create a venv and install the Pi-only dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pi/requirements-pi.txt
```

(`picamera2` for the Camera Module ships with Raspberry Pi OS already —
you only need `pi/requirements-pi.txt`'s `opencv-python-headless` if you're
using a USB webcam instead.)

## 3. Copy the bundle(s) + script onto the Pi

With `--all` you'll have one subfolder per architecture — copy whichever
one(s) you want to try (start with just one to keep step 4 simple):

```bash
scp -r outputs/pi_export/mobilenet_v3_small pi@<pi-ip>:~/crop-mesh-detector/pi_export
scp pi/inference_service.py pi@<pi-ip>:~/crop-mesh-detector/
```

## 4. Smoke test — no camera yet

Confirms onnxruntime + preprocessing + the manifest all work on real ARM
hardware before adding a camera into the mix. Copy any test image onto the
Pi first:

```bash
python inference_service.py --model-dir pi_export --camera file --image test.jpg --once
```

Should print one prediction row and exit. Note the `latency_ms` — this is
your real per-image inference time on this Pi, and tells you whether the
`--interval` in step 5 is reasonable, or whether you should re-export with a
faster architecture (see step 1).

## 5. Run with a live camera

USB webcam:

```bash
python inference_service.py --model-dir pi_export --camera opencv --interval 5
```

Official Camera Module:

```bash
python inference_service.py --model-dir pi_export --camera picamera2 --interval 5
```

Runs forever, classifying one frame every `--interval` seconds, printing each
prediction and appending it to `predictions_log.csv` (timestamp, predicted
crop, confidence, predicted disease, confidence, latency).

## 6. Optional: run on boot with systemd

```ini
# /etc/systemd/system/crop-detector.service
[Unit]
Description=crop-mesh-detector inference service
After=network.target

[Service]
ExecStart=/home/pi/crop-mesh-detector/.venv/bin/python /home/pi/crop-mesh-detector/inference_service.py --model-dir /home/pi/crop-mesh-detector/pi_export --camera picamera2 --interval 5
WorkingDirectory=/home/pi/crop-mesh-detector
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now crop-detector.service
sudo journalctl -u crop-detector -f   # watch live output
```

## Later: adding a hardware accelerator

This setup targets CPU-only Pi 4/5. If you later add a **Coral USB
Accelerator**, `model.onnx` would need converting to a fully int8-quantized
TFLite model and compiling with `edgetpu_compiler` — a different export path
than `scripts/export_for_pi.py`. If you add a **Pi 5 AI Kit (Hailo-8L)**,
the ONNX model would go through Hailo's own compiler to a `.hef` file
instead. Both are worth revisiting only once you know actual CPU-only
latency from step 4 and decide it's not fast enough.
