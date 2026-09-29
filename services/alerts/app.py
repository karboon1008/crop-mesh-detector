"""Farmer alerts service: the bridge between the HiveMind edge nodes and the
farmer's iPhone app (mobile/ios/).

  edge node (Pi camera)  --POST /api/detections-->  this service  --APNs-->  iPhone
                          --POST /api/nodes/heartbeat-->            <--REST--  (alerts, map, nodes)

What it adds on top of a raw "disease seen" message:
- Location: every alert carries coordinates, from the detection itself (a
  node with GPS) or else the node's surveyed position in the config.
- De-duplication: a node that keeps seeing the same disease updates one open
  alert (count, confidence, last seen) instead of paging the farmer every
  frame. A reminder is pushed at most every `renotify_hours`.
- Neighbour warnings: farms with a node within `nearby_radius_km` of a new
  outbreak get a "risk nearby" notice. Only the disease, a rounded (~1 km)
  location and the distance are shared, never the other farm's identity or
  exact field, in the same spirit as the mesh never sharing raw data.
- Feedback: the farmer marks an alert treated or a false alarm. False
  alarms are the labels a later continual-learning batch can learn from.

Run:  ALERTS_CONFIG=services/alerts/config.json uvicorn --factory services.alerts.app:build_app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from services.alerts.push import INVALID_TOKEN, build_sender
from services.alerts.store import ALL_STATUSES, OPEN_STATUSES, Store

HERE = Path(__file__).resolve().parent
DEFAULT_POLICY = {
    "min_confidence": 0.6,  # both crop and disease confidence must reach this
    "merge_window_hours": 24,  # repeat detections within this window update the same alert
    "renotify_hours": 6,  # "still detected" reminder at most this often
    "nearby_radius_km": 5.0,  # neighbour farms within this distance get a risk notice
    "offline_after_minutes": 30,  # a node silent this long shows as offline
}
MAX_IMAGE_BYTES = 1_000_000


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str | None) -> datetime:
    """Client timestamp -> UTC, never later than now (a node with a drifting
    clock must not push its alerts to the top forever).
    """
    if not value:
        return now_utc()
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return min(dt.astimezone(timezone.utc), now_utc())


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def pretty(label: str) -> str:
    return label.replace(",", "").replace("_", " ").strip()


# --- request bodies ---

class Heartbeat(BaseModel):
    node_id: str
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    battery_pct: float | None = Field(None, ge=0, le=100)


class DetectionIn(BaseModel):
    node_id: str
    crop: str
    disease: str
    crop_confidence: float = Field(ge=0, le=1)
    disease_confidence: float = Field(ge=0, le=1)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    captured_at: str | None = None
    image_jpeg_b64: str | None = None  # small evidence photo (the leaf crop), shown in the app


class StatusUpdate(BaseModel):
    status: str
    note: str | None = Field(None, max_length=500)


class DeviceIn(BaseModel):
    token: str = Field(min_length=8, max_length=200)
    platform: str = "ios"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("ALERTS_CONFIG", HERE / "config.example.json"))
    config = json.loads(Path(path).read_text())
    config["policy"] = {**DEFAULT_POLICY, **config.get("policy", {})}
    return config


def create_app(config: dict, db_path: str | os.PathLike | None = None, sender=None) -> FastAPI:
    policy = config["policy"]
    store = Store(Path(db_path or config.get("db_path", HERE / "data" / "alerts.db")))
    sender = sender or build_sender(config.get("apns"))
    farms: dict = config["farms"]
    nodes_cfg: dict = config["nodes"]
    token_to_farm = {token: farm_id for farm_id, farm in farms.items() for token in farm.get("app_tokens", [])}

    for node_id, node in nodes_cfg.items():
        store.upsert_node(node_id, node["farm_id"], node.get("name", node_id), node.get("latitude"), node.get("longitude"))

    app = FastAPI(title="HiveMind farmer alerts", version="1.0")
    app.state.store = store
    app.state.sender = sender

    # --- auth ---

    def farm_from_token(authorization: str | None = Header(None)) -> str:
        """The iOS app sends `Authorization: Bearer <farm app token>`."""
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Missing bearer token")
        farm_id = token_to_farm.get(authorization.split(" ", 1)[1].strip())
        if farm_id is None:
            raise HTTPException(401, "Unknown token")
        return farm_id

    def node_from_key(node_id: str, key: str | None) -> dict:
        node = nodes_cfg.get(node_id)
        if node is None or not key or key != node.get("key"):
            raise HTTPException(401, "Unknown node or wrong X-Node-Key")
        return node

    # --- serialisation ---

    def node_view(node: dict) -> dict:
        last_seen = node.get("last_seen")
        online = bool(last_seen) and now_utc() - datetime.fromisoformat(last_seen) < timedelta(
            minutes=policy["offline_after_minutes"])
        return {
            "node_id": node["node_id"], "name": node["name"], "latitude": node["latitude"],
            "longitude": node["longitude"], "last_seen": last_seen, "battery_pct": node["battery_pct"],
            "online": online,
        }

    def alert_view(alert: dict, with_detections: bool = False) -> dict:
        node = store.get_node(alert["node_id"]) or {}
        view = {
            "id": alert["id"], "node_id": alert["node_id"], "field_name": node.get("name", alert["node_id"]),
            "crop": alert["crop"], "disease": alert["disease"], "status": alert["status"],
            "confidence": round(alert["max_confidence"], 4), "latitude": alert["latitude"],
            "longitude": alert["longitude"], "first_seen": alert["first_seen"], "last_seen": alert["last_seen"],
            "detection_count": alert["detection_count"], "note": alert["note"],
            "image_path": f"/api/alerts/{alert['id']}/image" if alert["has_image"] else None,
        }
        if with_detections:
            view["detections"] = store.list_detections(alert["id"])
        return view

    # --- notifications ---

    def push_to_farm(farm_id: str, kind: str, payload: dict, alert_id: int | None, collapse_id: str | None) -> int:
        sent_at = iso(now_utc())
        delivered = 0
        for token in store.device_tokens(farm_id):
            status = sender.send(token, payload, collapse_id)
            store.log_push(farm_id, token, alert_id, kind, payload, status, sent_at)
            if status == INVALID_TOKEN:
                store.remove_device(token)
            else:
                delivered += 1
        return delivered

    def notify_alert(alert_id: int, farm_id: str, reminder: bool) -> int:
        alert = store.get_alert(alert_id, farm_id)
        node = store.get_node(alert["node_id"])
        what = f"{pretty(alert['crop'])} {pretty(alert['disease']).lower()}"
        where = f"{node['name']} ({alert['latitude']:.5f}, {alert['longitude']:.5f})"
        if reminder:
            title = f"Still detected: {what}"
            body = f"{alert['detection_count']} detections at {where}. Tap to see where."
        else:
            title = f"Disease detected: {what}"
            body = f"{where}, {alert['max_confidence'] * 100:.0f}% confidence. Tap to see where."
        payload = {
            "aps": {"alert": {"title": title, "body": body}, "sound": "default",
                    "category": "DISEASE_ALERT", "thread-id": f"alert-{alert_id}"},
            "kind": "disease_alert", "alert_id": alert_id,
            "latitude": alert["latitude"], "longitude": alert["longitude"],
        }
        count = push_to_farm(farm_id, "reminder" if reminder else "disease_alert", payload, alert_id, f"alert-{alert_id}")
        store.mark_notified(alert_id, iso(now_utc()))
        return count

    def notify_neighbours(alert_id: int, farm_id: str) -> int:
        alert = store.get_alert(alert_id, farm_id)
        delivered = 0
        for other_farm in farms:
            if other_farm == farm_id:
                continue
            distances = [
                haversine_km(alert["latitude"], alert["longitude"], n["latitude"], n["longitude"])
                for n in store.list_nodes(other_farm) if n["latitude"] is not None
            ]
            if not distances or min(distances) > policy["nearby_radius_km"]:
                continue
            distance = max(0.5, round(min(distances) * 2) / 2)  # to the nearest 0.5 km, never "0 km"
            payload = {
                "aps": {"alert": {"title": f"Risk nearby: {pretty(alert['disease']).lower()}",
                                  "body": f"Found on {pretty(alert['crop']).lower()} about {distance:g} km from your "
                                          f"fields. Check your {pretty(alert['crop']).lower()} plants."},
                        "sound": "default", "category": "NEARBY_RISK"},
                "kind": "nearby_risk", "crop": alert["crop"], "disease": alert["disease"],
                "latitude": round(alert["latitude"], 2), "longitude": round(alert["longitude"], 2),
                "distance_km": distance,
            }
            delivered += push_to_farm(other_farm, "nearby_risk", payload, None, None)
        return delivered

    # --- node endpoints ---

    @app.get("/api/health")
    def health():
        return {"ok": True, "push": type(sender).__name__}

    @app.post("/api/nodes/heartbeat")
    def heartbeat(body: Heartbeat, x_node_key: str | None = Header(None)):
        node_from_key(body.node_id, x_node_key)
        store.heartbeat(body.node_id, iso(now_utc()), body.latitude, body.longitude, body.battery_pct)
        return {"ok": True}

    @app.post("/api/detections")
    def report_detection(body: DetectionIn, x_node_key: str | None = Header(None)):
        node_cfg = node_from_key(body.node_id, x_node_key)
        farm_id = node_cfg["farm_id"]
        store.heartbeat(body.node_id, iso(now_utc()), battery_pct=None)

        confidence = min(body.crop_confidence, body.disease_confidence)
        if body.disease.lower() == "healthy":
            return {"accepted": False, "reason": "healthy"}
        if confidence < policy["min_confidence"]:
            return {"accepted": False, "reason": "below_min_confidence"}

        node = store.get_node(body.node_id)
        latitude = body.latitude if body.latitude is not None else node["latitude"]
        longitude = body.longitude if body.longitude is not None else node["longitude"]
        if latitude is None or longitude is None:
            raise HTTPException(422, "No location: send latitude/longitude or configure the node's position")

        image = None
        if body.image_jpeg_b64:
            try:
                image = base64.b64decode(body.image_jpeg_b64, validate=True)
            except ValueError:
                raise HTTPException(422, "image_jpeg_b64 is not valid base64")
            if len(image) > MAX_IMAGE_BYTES or not image.startswith(b"\xff\xd8"):
                raise HTTPException(422, "image must be a JPEG under 1 MB")

        seen = parse_time(body.captured_at)
        seen_at = iso(seen)
        existing = store.find_open_alert(
            farm_id, body.node_id, body.crop, body.disease, iso(seen - timedelta(hours=policy["merge_window_hours"])))
        if existing is None:
            alert_id = store.create_alert(farm_id, body.node_id, body.crop, body.disease, confidence,
                                          latitude, longitude, seen_at, image)
        else:
            alert_id = existing["id"]
            store.merge_into_alert(alert_id, confidence, latitude, longitude, seen_at, image)
        store.add_detection(alert_id, seen_at, body.crop_confidence, body.disease_confidence, latitude, longitude)

        notified = neighbours = 0
        if existing is None:
            notified = notify_alert(alert_id, farm_id, reminder=False)
            neighbours = notify_neighbours(alert_id, farm_id)
        else:
            last = existing["last_notified"]
            if last is None or now_utc() - datetime.fromisoformat(last) >= timedelta(hours=policy["renotify_hours"]):
                notified = notify_alert(alert_id, farm_id, reminder=True)
        return {"accepted": True, "alert_id": alert_id, "new_alert": existing is None,
                "devices_notified": notified, "neighbour_devices_notified": neighbours}

    # --- app endpoints ---

    @app.get("/api/me")
    def me(farm_id: str = Depends(farm_from_token)):
        return {"farm_id": farm_id, "farm_name": farms[farm_id].get("name", farm_id), "policy": policy}

    @app.get("/api/alerts")
    def list_alerts(status: str = "open", farm_id: str = Depends(farm_from_token)):
        statuses = {"open": OPEN_STATUSES, "closed": ("treated", "false_alarm"), "all": ALL_STATUSES}.get(status)
        if statuses is None:
            raise HTTPException(422, "status must be open, closed or all")
        return [alert_view(a) for a in store.list_alerts(farm_id, statuses)]

    @app.get("/api/alerts/{alert_id}")
    def get_alert(alert_id: int, farm_id: str = Depends(farm_from_token)):
        alert = store.get_alert(alert_id, farm_id)
        if alert is None:
            raise HTTPException(404, "No such alert")
        return alert_view(alert, with_detections=True)

    @app.get("/api/alerts/{alert_id}/image")
    def get_alert_image(alert_id: int, farm_id: str = Depends(farm_from_token)):
        image = store.get_alert_image(alert_id, farm_id)
        if not image:
            raise HTTPException(404, "No image")
        return Response(image, media_type="image/jpeg")

    @app.patch("/api/alerts/{alert_id}")
    def update_alert(alert_id: int, body: StatusUpdate, farm_id: str = Depends(farm_from_token)):
        if body.status not in ALL_STATUSES:
            raise HTTPException(422, f"status must be one of {ALL_STATUSES}")
        if not store.update_alert_status(alert_id, farm_id, body.status, body.note, iso(now_utc())):
            raise HTTPException(404, "No such alert")
        return alert_view(store.get_alert(alert_id, farm_id))

    @app.get("/api/nodes")
    def list_nodes(farm_id: str = Depends(farm_from_token)):
        return [node_view(n) for n in store.list_nodes(farm_id)]

    @app.get("/api/nearby-risks")
    def nearby_risks(farm_id: str = Depends(farm_from_token)):
        """Other farms' open outbreaks near this farm's nodes: disease, crop,
        rounded location and distance only.
        """
        own = [n for n in store.list_nodes(farm_id) if n["latitude"] is not None]
        risks = []
        for alert in store.list_open_alerts_excluding_farm(farm_id):
            if not own:
                break
            distance = min(haversine_km(alert["latitude"], alert["longitude"], n["latitude"], n["longitude"]) for n in own)
            if distance <= policy["nearby_radius_km"]:
                risks.append({
                    "crop": alert["crop"], "disease": alert["disease"],
                    "latitude": round(alert["latitude"], 2), "longitude": round(alert["longitude"], 2),
                    "distance_km": max(0.5, round(distance * 2) / 2), "last_seen": alert["last_seen"],
                })
        return sorted(risks, key=lambda r: r["distance_km"])

    @app.post("/api/devices")
    def register_device(body: DeviceIn, farm_id: str = Depends(farm_from_token)):
        store.register_device(body.token, farm_id, body.platform, iso(now_utc()))
        return {"ok": True}

    @app.delete("/api/devices/{token}")
    def unregister_device(token: str, farm_id: str = Depends(farm_from_token)):
        store.remove_device(token, farm_id)
        return {"ok": True}

    @app.get("/api/push-log")
    def push_log(farm_id: str = Depends(farm_from_token)):
        """Notifications sent (or, in dry-run mode, that would have been sent) to this farm."""
        return store.push_log(farm_id)

    return app


def build_app() -> FastAPI:
    """uvicorn --factory entry point: config from $ALERTS_CONFIG (default: the demo config)."""
    return create_app(load_config())
