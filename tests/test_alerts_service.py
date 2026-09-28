"""Farmer alerts service (services/alerts/): detection intake, alert merging,
push notifications (dry-run and APNs), neighbour warnings, app endpoints."""
from __future__ import annotations

import base64
import io
import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.alerts import app as alerts_app
from services.alerts.app import create_app, load_config
from services.alerts.push import APNsConfig, APNsSender, DRY_RUN, INVALID_TOKEN, SENT

CONFIG_PATH = Path(__file__).resolve().parent.parent / "services" / "alerts" / "config.example.json"
FARM = {"Authorization": "Bearer demo-farm-token"}
NEIGHBOUR = {"Authorization": "Bearer demo-neighbour-token"}
NODE0 = {"X-Node-Key": "demo-node-0-key"}


class RecordingSender:
    def __init__(self, status=DRY_RUN):
        self.sent = []
        self.status = status

    def send(self, token, payload, collapse_id=None):
        self.sent.append((token, payload, collapse_id))
        return self.status


@pytest.fixture
def sender():
    return RecordingSender()


@pytest.fixture
def client(tmp_path, sender):
    app = create_app(load_config(CONFIG_PATH), tmp_path / "alerts.db", sender)
    with TestClient(app) as c:
        c.post("/api/devices", json={"token": "iphone-farmer-1"}, headers=FARM)
        c.post("/api/devices", json={"token": "iphone-neighbour"}, headers=NEIGHBOUR)
        yield c


def detection(**overrides):
    body = {"node_id": "node_0", "crop": "Tomato", "disease": "Late_blight",
            "crop_confidence": 0.95, "disease_confidence": 0.88}
    return {**body, **overrides}


def jpeg_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (40, 160, 40)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_new_detection_creates_alert_with_node_location_and_pushes(client, sender):
    r = client.post("/api/detections", json=detection(), headers=NODE0).json()
    assert r["accepted"] and r["new_alert"] and r["devices_notified"] == 1

    farmer_pushes = [p for t, p, _ in sender.sent if t == "iphone-farmer-1"]
    assert len(farmer_pushes) == 1
    push = farmer_pushes[0]
    assert push["kind"] == "disease_alert" and push["alert_id"] == r["alert_id"]
    assert (push["latitude"], push["longitude"]) == (4.47212, 101.37913)  # node_0's configured position
    assert push["aps"]["alert"]["title"] == "Disease detected: Tomato late blight"
    assert "Tomato greenhouse A" in push["aps"]["alert"]["body"] and "88%" in push["aps"]["alert"]["body"]
    assert push["aps"]["category"] == "DISEASE_ALERT"

    alert = client.get(f"/api/alerts/{r['alert_id']}", headers=FARM).json()
    assert alert["field_name"] == "Tomato greenhouse A" and alert["status"] == "new"
    assert alert["confidence"] == 0.88 and len(alert["detections"]) == 1


def test_detection_gps_overrides_node_position(client):
    r = client.post("/api/detections", json=detection(latitude=4.4725, longitude=101.3795), headers=NODE0).json()
    alert = client.get(f"/api/alerts/{r['alert_id']}", headers=FARM).json()
    assert (alert["latitude"], alert["longitude"]) == (4.4725, 101.3795)


def test_repeat_detections_merge_and_do_not_spam(client, sender):
    first = client.post("/api/detections", json=detection(), headers=NODE0).json()
    second = client.post("/api/detections", json=detection(disease_confidence=0.97), headers=NODE0).json()
    assert second["alert_id"] == first["alert_id"] and not second["new_alert"]
    assert second["devices_notified"] == 0  # within renotify_hours
    assert len([1 for t, *_ in sender.sent if t == "iphone-farmer-1"]) == 1

    alert = client.get(f"/api/alerts/{first['alert_id']}", headers=FARM).json()
    assert alert["detection_count"] == 2 and alert["confidence"] == 0.95  # min(crop, disease) of the best report


def test_reminder_after_renotify_window(client, sender, monkeypatch):
    first = client.post("/api/detections", json=detection(), headers=NODE0).json()
    real_now = alerts_app.now_utc
    monkeypatch.setattr(alerts_app, "now_utc", lambda: real_now() + timedelta(hours=7))
    second = client.post("/api/detections", json=detection(), headers=NODE0).json()
    assert second["alert_id"] == first["alert_id"] and second["devices_notified"] == 1
    reminder = [p for t, p, _ in sender.sent if t == "iphone-farmer-1"][-1]
    assert reminder["aps"]["alert"]["title"].startswith("Still detected")


def test_treated_alert_recurrence_opens_new_alert(client):
    first = client.post("/api/detections", json=detection(), headers=NODE0).json()
    client.patch(f"/api/alerts/{first['alert_id']}", json={"status": "treated", "note": "Sprayed copper"}, headers=FARM)
    again = client.post("/api/detections", json=detection(), headers=NODE0).json()
    assert again["new_alert"] and again["alert_id"] != first["alert_id"]
    closed = client.get("/api/alerts?status=closed", headers=FARM).json()
    assert [a["note"] for a in closed] == ["Sprayed copper"]


def test_healthy_and_low_confidence_are_ignored(client, sender):
    assert client.post("/api/detections", json=detection(disease="healthy"), headers=NODE0).json() == {
        "accepted": False, "reason": "healthy"}
    low = client.post("/api/detections", json=detection(crop_confidence=0.4), headers=NODE0).json()
    assert low == {"accepted": False, "reason": "below_min_confidence"}
    assert sender.sent == [] and client.get("/api/alerts", headers=FARM).json() == []


def test_neighbour_farm_gets_rounded_risk_notice_only(client, sender):
    client.post("/api/detections", json=detection(), headers=NODE0)
    neighbour = [p for t, p, _ in sender.sent if t == "iphone-neighbour"]
    assert len(neighbour) == 1
    notice = neighbour[0]
    assert notice["kind"] == "nearby_risk" and "alert_id" not in notice
    assert (notice["latitude"], notice["longitude"]) == (4.47, 101.38)
    assert notice["distance_km"] == 2.5  # ~2.3 km, to the nearest 0.5
    assert "Tanah Rata" not in json.dumps(notice) and "greenhouse" not in json.dumps(notice)

    risks = client.get("/api/nearby-risks", headers=NEIGHBOUR).json()
    assert risks == [{"crop": "Tomato", "disease": "Late_blight", "latitude": 4.47, "longitude": 101.38,
                      "distance_km": 2.5, "last_seen": risks[0]["last_seen"]}]
    assert client.get("/api/alerts", headers=NEIGHBOUR).json() == []  # never the other farm's alerts


def test_auth_is_enforced(client):
    assert client.post("/api/detections", json=detection(), headers={"X-Node-Key": "wrong"}).status_code == 401
    assert client.post("/api/detections", json=detection(node_id="node_3"), headers=NODE0).status_code == 401
    assert client.get("/api/alerts").status_code == 401
    assert client.get("/api/alerts", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = client.post("/api/detections", json=detection(), headers=NODE0).json()
    assert client.get(f"/api/alerts/{r['alert_id']}", headers=NEIGHBOUR).status_code == 404


def test_image_evidence_round_trip_and_validation(client):
    r = client.post("/api/detections", json=detection(image_jpeg_b64=jpeg_b64()), headers=NODE0).json()
    alert = client.get(f"/api/alerts/{r['alert_id']}", headers=FARM).json()
    assert alert["image_path"] == f"/api/alerts/{r['alert_id']}/image"
    image = client.get(alert["image_path"], headers=FARM)
    assert image.headers["content-type"] == "image/jpeg" and image.content.startswith(b"\xff\xd8")

    not_jpeg = base64.b64encode(b"GIF89a....").decode()
    assert client.post("/api/detections", json=detection(image_jpeg_b64=not_jpeg), headers=NODE0).status_code == 422


def test_nodes_heartbeat_and_online_state(client, monkeypatch):
    client.post("/api/nodes/heartbeat", json={"node_id": "node_0", "battery_pct": 76}, headers=NODE0)
    nodes = {n["node_id"]: n for n in client.get("/api/nodes", headers=FARM).json()}
    assert set(nodes) == {"node_0", "node_1", "node_2"}  # only this farm's
    assert nodes["node_0"]["online"] and nodes["node_0"]["battery_pct"] == 76
    assert not nodes["node_1"]["online"]

    real_now = alerts_app.now_utc
    monkeypatch.setattr(alerts_app, "now_utc", lambda: real_now() + timedelta(minutes=45))
    assert not client.get("/api/nodes", headers=FARM).json()[0]["online"]


def test_invalid_device_token_is_forgotten(tmp_path):
    sender = RecordingSender(status=INVALID_TOKEN)
    with TestClient(create_app(load_config(CONFIG_PATH), tmp_path / "a.db", sender)) as c:
        c.post("/api/devices", json={"token": "stale-token-123"}, headers=FARM)
        c.post("/api/detections", json=detection(), headers=NODE0)
        assert c.app.state.store.device_tokens("farm_tanah_rata") == []
        assert c.get("/api/push-log", headers=FARM).json()[0]["status"] == INVALID_TOKEN


def test_apns_sender_request_shape(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    import jwt

    key = ec.generate_private_key(ec.SECP256R1())
    key_path = tmp_path / "AuthKey_TEST.p8"
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))

    class FakeClient:
        def __init__(self):
            self.calls = []

        def post(self, url, json, headers):
            self.calls.append((url, json, headers))
            status = 410 if "gone" in url else 200
            return type("R", (), {"status_code": status, "text": ""})()

    client = FakeClient()
    sender = APNsSender(APNsConfig("TEAM123", "KEY123", str(key_path), "org.hivemind.cropguard", True), client)
    assert sender.send("abc123", {"aps": {}}, "alert-1") == SENT
    assert sender.send("gone", {"aps": {}}) == INVALID_TOKEN

    url, _, headers = client.calls[0]
    assert url == "https://api.sandbox.push.apple.com/3/device/abc123"
    assert headers["apns-topic"] == "org.hivemind.cropguard" and headers["apns-collapse-id"] == "alert-1"
    token = headers["authorization"].split()[1]
    assert jwt.get_unverified_header(token)["kid"] == "KEY123"
    assert jwt.decode(token, key.public_key(), algorithms=["ES256"])["iss"] == "TEAM123"


def test_pi_reporter_filters_queues_and_retries_in_order():
    pytest.importorskip("onnxruntime")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pi_inference_service", Path(__file__).resolve().parent.parent / "pi" / "inference_service.py")
    pi_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi_service)

    reporter = pi_service.AlertReporter("http://alerts", "node_0", "k", latitude=4.1, longitude=101.2)
    posts, online = [], {"up": False}

    def fake_post(path, body):
        if online["up"]:
            posts.append((path, body))
        return online["up"]

    reporter._post = fake_post
    leaf = Image.new("RGB", (800, 600), (40, 160, 40))
    reporter.report("Tomato", 0.9, "healthy", 0.99, leaf, "t0")  # healthy: never reported
    reporter.report("Tomato", 0.5, "Late_blight", 0.9, leaf, "t1")  # below 0.6: never reported
    reporter.report("Tomato", 0.9, "Late_blight", 0.8, leaf, "t2")
    reporter.report("Potato", 0.9, "Early_blight", 0.7, leaf, "t3")
    reporter.flush()  # offline: both kept
    assert len(reporter.queue) == 2 and posts == []

    online["up"] = True
    reporter.flush()
    assert [b["captured_at"] for _, b in posts] == ["t2", "t3"] and not reporter.queue
    body = posts[0][1]
    assert (body["latitude"], body["longitude"]) == (4.1, 101.2)
    thumb = Image.open(io.BytesIO(base64.b64decode(body["image_jpeg_b64"])))
    assert max(thumb.size) == 480


def test_pi_reporter_payload_is_accepted_by_service(client):
    """The Pi's report body is exactly what POST /api/detections accepts."""
    import importlib.util

    pytest.importorskip("onnxruntime")
    spec = importlib.util.spec_from_file_location(
        "pi_inference_service", Path(__file__).resolve().parent.parent / "pi" / "inference_service.py")
    pi_service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi_service)

    reporter = pi_service.AlertReporter("http://testserver", "node_0", "demo-node-0-key")
    reporter._post = lambda path, body: client.post(path, json=body, headers={"X-Node-Key": reporter.node_key}).status_code == 200
    reporter.report("Tomato", 0.9, "Late_blight", 0.8, Image.new("RGB", (64, 64)), "2026-09-28T10:00:00Z")
    reporter.flush()
    reporter.maybe_heartbeat()
    [alert] = client.get("/api/alerts", headers=FARM).json()
    assert alert["image_path"] and alert["last_seen"] == "2026-09-28T10:00:00+00:00"
    assert (alert["latitude"], alert["longitude"]) == (4.47212, 101.37913)  # service falls back to the node's position
