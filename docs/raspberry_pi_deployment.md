# Deploying to a Raspberry Pi

Runs the best-scoring trained model on a Pi's CPU, classifying images from a
camera in near-real-time — no training stack (torch/torchvision/timm) needed
on the Pi itself.

## 1. Export the model (dev machine or Colab, wherever training ran)

After a completed `python -m src.train` run:

```bash
pip install onnxruntime   # already in requirements.txt
python scripts/export_for_pi.py
```

This picks whichever `(architecture, node)` scored the highest average
crop+disease test accuracy in `outputs/results_summary.json`, converts it to
ONNX, quantizes it to int8, and writes a small self-contained bundle to
`outputs/pi_export/`:

- `model.onnx` — a few MB to a few tens of MB, depending on architecture
- `manifest.json` — class names, image size, normalization constants

It also runs a parity check against a handful of real images from
`data/PlantVillage` (if present) to confirm the quantized ONNX model agrees
with the original PyTorch model's predictions before you ever touch hardware.

If the auto-picked architecture turns out too slow on the Pi in step 4, redo
this with an explicit override, e.g.:

```bash
python scripts/export_for_pi.py --arch mobilenet_v3_small --node node_0
```

(Check `outputs/results_summary.json` for available node IDs per architecture.)

**If you ran training on Colab**: run this export step there too, right
after training, then download only the small `outputs/pi_export/` folder
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

## 3. Copy the bundle + script onto the Pi

From your dev machine:

```bash
scp -r outputs/pi_export pi@<pi-ip>:~/crop-mesh-detector/
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
