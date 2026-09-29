"""Phone alerts through ntfy (https://ntfy.sh), a free, open-source push service.

When the dashboard finds a diseased leaf, it publishes one message to an ntfy topic. Every
phone subscribed to that topic in the free ntfy app (iPhone or Android) gets a real push
notification, even when locked, with no Apple developer account: the ntfy app's own push
permission carries it. The notification shows the diagnosis and the field's coordinates,
opens Apple Maps at the field when tapped, has a "Navigate" button, and carries the photo
with the leaves boxed.

Anyone who knows a topic name can read it on the public ntfy.sh server, so the default
topic is a long random name, kept in data/ntfy_topic.txt so it survives restarts. Settings
can come from the environment (docker-compose.yml): NTFY_SERVER, NTFY_TOPIC, FIELD_NAME,
FIELD_LAT, FIELD_LON.
"""
from __future__ import annotations

import base64
import os
import secrets
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_SERVER = "https://ntfy.sh"
# Demo field: Tanah Rata, Cameron Highlands (same as services/alerts/config.example.json).
DEFAULT_FIELD = ("Tomato greenhouse A", 4.47212, 101.37913)


def default_topic(data_dir: Path) -> str:
    """$NTFY_TOPIC, else a random topic created once and kept in data_dir/ntfy_topic.txt."""
    if os.environ.get("NTFY_TOPIC"):
        return os.environ["NTFY_TOPIC"].strip()
    path = Path(data_dir) / "ntfy_topic.txt"
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    topic = f"hivemind-farm-{secrets.token_hex(5)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(topic)
    return topic


def default_server() -> str:
    return os.environ.get("NTFY_SERVER", DEFAULT_SERVER).rstrip("/")


def default_field() -> tuple[str, float, float]:
    name, lat, lon = DEFAULT_FIELD
    return (os.environ.get("FIELD_NAME", name), float(os.environ.get("FIELD_LAT", lat)),
            float(os.environ.get("FIELD_LON", lon)))


def maps_url(latitude: float, longitude: float, label: str) -> str:
    # Commas are escaped: ntfy's Actions header uses commas as separators.
    query = urllib.parse.urlencode({"ll": f"{latitude:.6f},{longitude:.6f}", "q": label})
    return "https://maps.apple.com/?" + query.replace(",", "%2C")


def pretty(label: str) -> str:
    return label.replace("_", " ").replace(",", "").strip()


def build_alert(results: list[dict], field_name: str, latitude: float, longitude: float) -> tuple[str, str] | None:
    """(title, message) for the diseased leaves in `results`, or None if none is diseased."""
    diseased = [r for r in results if r["tier"] == "diseased"]
    if not diseased:
        return None
    first = diseased[0]
    what = f"{pretty(first['predicted_crop'])} {pretty(first['predicted_disease']).lower()}"
    title = f"Disease detected: {what}"
    if len(diseased) > 1:
        title += f" (+{len(diseased) - 1} more)"
    lines = [f"{i}. {pretty(r['predicted_crop'])}: {pretty(r['predicted_disease'])} "
             f"({r['disease_confidence'] * 100:.0f}%)" for i, r in enumerate(diseased, start=1)]
    lines.append(f"Where: {field_name} ({latitude:.5f}, {longitude:.5f})")
    lines.append("Tap to open the location in Maps.")
    return title, "\n".join(lines)


def _header(value: str) -> str:
    """Plain ASCII as is; anything else (é, ·, emoji) as an RFC 2047 encoded word, which ntfy
    decodes. Raw Latin-1 bytes would be misread, since ntfy expects UTF-8.
    """
    if value.isascii():
        return value
    return "=?UTF-8?B?" + base64.b64encode(value.encode()).decode() + "?="


def send(server: str, topic: str, title: str, message: str, latitude: float, longitude: float,
         field_name: str, image_jpeg: bytes | None = None, timeout: float = 10.0) -> None:
    """Publishes one notification. With an image, the photo is the request body and the
    text goes in headers (ntfy's attachment upload); raises on network or HTTP errors.
    """
    click = maps_url(latitude, longitude, field_name)
    headers = {
        "Title": _header(title),
        # A header can't hold line breaks; ntfy turns a literal "\n" in the Message header into one.
        "Message": _header(message.replace("\n", "\\n")),
        "Tags": "warning,seedling",
        "Priority": "high",
        "Click": click,
        "Actions": f"view, Navigate to field, {click}",
    }
    url = f"{server.rstrip('/')}/{urllib.parse.quote(topic, safe='')}"
    if image_jpeg:
        headers["Filename"] = "leaves.jpg"
        request = urllib.request.Request(url, data=image_jpeg, headers=headers, method="PUT")
    else:
        request = urllib.request.Request(url, data=message.encode(), headers={
            k: v for k, v in headers.items() if k != "Message"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()
