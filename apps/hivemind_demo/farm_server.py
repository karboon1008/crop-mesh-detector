"""Reports the dashboard's disease detections to the farmer alerts server (services/alerts/),
the backend of the HiveMind Farmer iPhone app. The dashboard acts as one field camera
(node_0 in services/alerts/config.example.json): each diseased leaf is POSTed to
/api/detections with the field's coordinates and the leaf photo, the server merges repeats
into one alert per disease, and the app picks it up.

Configured from the environment (docker-compose.yml): ALERTS_URL, ALERTS_NODE_ID,
ALERTS_NODE_KEY. With ALERTS_URL unset or empty, nothing is sent.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

from PIL import Image


def config() -> tuple[str, str, str] | None:
    url = os.environ.get("ALERTS_URL", "").strip().rstrip("/")
    if not url:
        return None
    return url, os.environ.get("ALERTS_NODE_ID", "node_0"), os.environ.get("ALERTS_NODE_KEY", "demo-node-0-key")


def _jpeg_b64(image: Image.Image) -> str:
    photo = image.convert("RGB")
    photo.thumbnail((480, 480))  # small evidence photo: the leaf, not the whole scene
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def report(url: str, node_id: str, node_key: str, results: list[dict], leaf_images: list[Image.Image],
           latitude: float, longitude: float, timeout: float = 5.0) -> list[dict]:
    """POSTs every diseased leaf; returns the server's replies (alert_id, new_alert, ...).
    Raises on network or HTTP errors.
    """
    replies = []
    for result, leaf in zip(results, leaf_images):
        if result["tier"] != "diseased":
            continue
        body = {
            "node_id": node_id, "crop": result["predicted_crop"], "disease": result["predicted_disease"],
            "crop_confidence": round(float(result["crop_confidence"]), 4),
            "disease_confidence": round(float(result["disease_confidence"]), 4),
            "latitude": latitude, "longitude": longitude, "image_jpeg_b64": _jpeg_b64(leaf),
        }
        request = urllib.request.Request(
            f"{url}/api/detections", data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "X-Node-Key": node_key},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            replies.append(json.loads(response.read()))
    return replies
