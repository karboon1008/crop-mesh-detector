"""Exports the stage-1 leaf detector (src/detection/) to ONNX for the
Raspberry Pi, next to the classifier bundle from scripts/export_for_pi.py:

    python scripts/export_detector_for_pi.py \
        --checkpoint outputs/detector/leaf_ssdlite.pt --output-dir outputs/pi_export

Writes detector.onnx + detector_manifest.json into --output-dir. The Pi
(pi/inference_service.py --detector) resizes each frame to 320x320,
feeds it in [0, 1] (the model normalises internally), and gets back
boxes (in 320x320 pixels), scores and labels with NMS already applied —
so the Pi needs no torch, only onnxruntime.

The graph is kept in fp32: SSDLite is small (~3.4M parameters), and
onnxruntime's dynamic int8 quantisation only covers MatMul/Gemm, which
this conv-only network barely has.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.leaf_detector import DETECTOR_IMAGE_SIZE, LEAF_LABEL, load_detector  # noqa: E402


def export_detector_onnx(model: torch.nn.Module, output_path: Path) -> None:
    model.eval()
    dummy = torch.rand(3, DETECTOR_IMAGE_SIZE, DETECTOR_IMAGE_SIZE)
    kwargs = dict(
        input_names=["image"],
        output_names=["boxes", "scores", "labels"],
        # The number of detections varies per image.
        dynamic_axes={"boxes": {0: "num_detections"}, "scores": {0: "num_detections"},
                      "labels": {0: "num_detections"}},
        opset_version=17,
    )
    # torchvision detection models take a list of CHW images; the legacy
    # TorchScript exporter is the one torchvision validates them against.
    try:
        torch.onnx.export(model, ([dummy],), str(output_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, ([dummy],), str(output_path), **kwargs)


def check_parity(model: torch.nn.Module, onnx_path: Path, num_samples: int = 3) -> float:
    """Largest absolute score difference between PyTorch and ONNX Runtime
    on a few random frames (compared over the top detections both return).
    """
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path))
    worst = 0.0
    with torch.no_grad():
        for seed in range(num_samples):
            torch.manual_seed(seed)
            image = torch.rand(3, DETECTOR_IMAGE_SIZE, DETECTOR_IMAGE_SIZE)
            torch_scores = model([image])[0]["scores"].numpy()
            _, onnx_scores, _ = session.run(None, {"image": image.numpy()})
            n = min(len(torch_scores), len(onnx_scores), 10)
            if n:
                worst = max(worst, float(np.abs(np.sort(torch_scores)[::-1][:n] - np.sort(onnx_scores)[::-1][:n]).max()))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/detector/leaf_ssdlite.pt")
    parser.add_argument("--output-dir", default="outputs/pi_export")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Default score cut-off written to the manifest (defaults to the checkpoint's own)")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = load_detector(Path(args.checkpoint))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "detector.onnx"

    print(f"Exporting {args.checkpoint} to {onnx_path}...")
    export_detector_onnx(model, onnx_path)
    worst = check_parity(model, onnx_path)
    print(f"PyTorch vs ONNX max score difference on random frames: {worst:.2e}")

    score_threshold = args.score_threshold
    if score_threshold is None:
        score_threshold = checkpoint.get("metadata", {}).get("score_threshold", 0.5)
    manifest = {
        "model": "ssdlite320_mobilenet_v3_large",
        "image_size": DETECTOR_IMAGE_SIZE,
        "input_range": "[0, 1] RGB, CHW, no mean/std (normalised inside the graph)",
        "leaf_label": LEAF_LABEL,
        "score_threshold": score_threshold,
        "metrics": checkpoint.get("metadata", {}).get("metrics"),
    }
    (output_dir / "detector_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {onnx_path} + detector_manifest.json. Copy them to the Pi next to model.onnx.")


if __name__ == "__main__":
    main()
