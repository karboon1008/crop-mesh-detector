"""Sends a fake edge-node detection to a running alerts service, so the
iPhone notification can be demonstrated live without a camera in a field:

    python -m services.alerts.simulate                           # tomato late blight at node_0
    python -m services.alerts.simulate --node node_2 --crop "Pepper,_bell" --disease Bacterial_spot
    python -m services.alerts.simulate --image leaf.jpg --jitter-m 15

Uses the demo node keys from config.example.json unless --node-key is given.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import urllib.request
from pathlib import Path

from services.alerts.app import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--config", default=None, help="Config to read node keys/positions from")
    parser.add_argument("--node", default="node_0")
    parser.add_argument("--node-key", default=None)
    parser.add_argument("--crop", default="Tomato")
    parser.add_argument("--disease", default="Late_blight")
    parser.add_argument("--confidence", type=float, default=0.91)
    parser.add_argument("--image", default=None, help="JPEG of the leaf to attach as evidence")
    parser.add_argument("--jitter-m", type=float, default=0.0,
                        help="Move the reported position up to this many metres from the node (a plant, not the pole)")
    args = parser.parse_args()

    config = load_config(args.config)
    node = config["nodes"][args.node]
    body = {
        "node_id": args.node, "crop": args.crop, "disease": args.disease,
        "crop_confidence": args.confidence, "disease_confidence": args.confidence,
    }
    if args.jitter_m and node.get("latitude") is not None:
        angle, dist = random.uniform(0, 2 * math.pi), random.uniform(0, args.jitter_m)
        body["latitude"] = node["latitude"] + dist * math.cos(angle) / 111_320
        body["longitude"] = node["longitude"] + dist * math.sin(angle) / (111_320 * math.cos(math.radians(node["latitude"])))
    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        image.thumbnail((480, 480))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        body["image_jpeg_b64"] = base64.b64encode(buf.getvalue()).decode()

    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/api/detections", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-Node-Key": args.node_key or node["key"]},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        print(json.dumps(json.loads(response.read()), indent=2))


if __name__ == "__main__":
    main()
