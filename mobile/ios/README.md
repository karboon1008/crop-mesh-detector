# HiveMind Farmer (iOS)

SwiftUI app for farmers. It gets a push notification when a HiveMind field camera finds a
disease, shows exactly where the plant is, and guides the farmer to it. Design notes and
suggested next features: [`docs/farmer_mobile_app.md`](../../docs/farmer_mobile_app.md).

```
Pi field camera ──POST /api/detections──▶ services/alerts ──APNs push──▶ iPhone
 (pi/inference_service.py --report-url)    (FastAPI + SQLite)  ◀──REST──  (this app)
```

## What it does

| Tab | |
|---|---|
| **Home** | The farm at a glance. Satellite map showing the estimated area each open disease covers (red: new, orange: seen; the outline around its detections plus 12 m, or a circle that grows with how often it was seen) with its size, and each camera's outline, with chips for cameras online, lowest battery, open alerts and nearby outbreaks. Tap a camera for its card. Crop chips with open alerts. **Hive intelligence:** outbreaks on nearby farms with the risk to yours, and 7 days of outbreak activity on your farm and around it. **Learning together:** your confirmed and false-alarm reports, which train the next model round while photos stay on the farm. Last activities. |
| **Alerts** | Open alerts, most urgent first (new before seen, then by disease urgency). When your cameras have found a disease, those alerts come first and outbreaks reported by nearby farms follow; with none, nearby outbreaks lead. Resolved alerts below. The **Sort** button arranges the lists by detection time, action needed or nearest (distance from you to the plant; nearby farms by their distance), with the order either way round. Pull to refresh. |
| **Alert detail** | Map pin on the plant (satellite view), coordinates in decimal and degrees/minutes/seconds, distance and direction from you, **Navigate** (Apple Maps walking directions), copy or share the location with a worker, the leaf photo, detection history, a treatment guide that works offline, and **I've treated it** / **False alarm** buttons with an optional note. |
| **Map** | All alerts, field cameras (blue online, grey offline) and nearby outbreaks (orange ~1 km circles). Tap a pin to open its alert. |
| **Fields** | Each camera's online state, last contact, battery, open alerts. |
| **Settings** | Demo mode, server address and farm code, notification permission, and **Simulate a detection** in demo mode. |

**Notifications:**
- **Tap:** opens that alert.
- **Navigate to plant** (long‑press): starts Apple Maps directions without opening the app.
- **Mark as seen:** marks the alert seen from the lock screen.
- **Grouping:** repeat detections of the same outbreak are merged by the server, and at most one reminder is sent every 6 h, so farmers aren't spammed.

**Offline first:** the last data is cached on the phone and shown with an "Offline" banner. Status
changes made without signal are queued and sent later. The treatment guide is bundled with the app.

## Build

Needs a Mac with Xcode 15 or later (iOS 17 SDK).

```bash
brew install xcodegen
cd mobile/ios
xcodegen generate
open HiveMindFarmer.xcodeproj
```

In Xcode, select the **HiveMindFarmer** target > Signing & Capabilities, and choose your team.
A free Apple ID is enough. Change the bundle id (`org.hivemind.farmer`) if it's taken. Then run
on an iPhone or the simulator.

**With a free Apple ID, on a real iPhone:**
- **First run:** turn on Settings > Privacy & Security > **Developer Mode** (the phone restarts),
  and after installing, trust yourself under Settings > General > VPN & Device Management.
- **7‑day expiry:** the app stops opening after 7 days. Plug the phone in and press Run again
  to renew it, so do it the day before a demo.
- **Push:** the project ships without the push entitlement, because free accounts can't sign
  it. The demo's simulated notifications don't need it. Real APNs needs the paid account; see
  *Real push notifications* below.

> This app was written without access to a Mac, so it has not been compiled yet. Expect to
> fix the odd compiler error on the first build.

## Demo for judges (no server needed)

1. Launch the app. **Demo mode** is on by default: a sample farm in Cameron Highlands with 3
   open alerts, 3 field cameras (one offline, low battery) and one outbreak on a neighbouring farm.
2. Settings > **Turn on disease alerts** > Allow.
3. Settings > **Simulate a detection (in 5 s)**, then lock the phone. The alert arrives like a
   real one ("Disease detected: Tomato tomato yellow leaf curl virus").
4. Tap it: the app opens on the plant's location. Show **Navigate**, the treatment guide, and **I've treated it**.

## Connected to the dashboard (Docker)

Start the dashboard with `cd apps/hivemind_demo && docker compose up --build`: that also starts the
alerts server on port 8080. In the app (Simulator), set **Settings > Demo mode off**, keep the
defaults (`http://localhost:8080`, farm code `demo-farm-token`) and tap **Connect**. Every disease
the dashboard finds then shows up in the app within ~5 s, with a notification. While open, the app
checks the server every 5 s and notifies for alerts it hasn't announced yet, with no Apple push
account needed.

## Run it against the real alerts service

On the laptop (same Wi-Fi as the phone), from the repo root:

```bash
pip install fastapi uvicorn "httpx[http2]" "pyjwt[crypto]"
ALERTS_CONFIG=services/alerts/config.example.json \
  uvicorn --factory services.alerts.app:build_app --host 0.0.0.0 --port 8080
```

In the app: Settings > turn Demo mode **off**, set the server address to `http://<laptop-ip>:8080`,
enter the farm code `demo-farm-token`, then tap **Connect**. Then trigger a detection as if a
field camera saw it:

```bash
python -m services.alerts.simulate --node node_0 --crop Tomato --disease Late_blight --image leaf.jpg
```

Or use a real camera. On the Pi:

```bash
python pi/inference_service.py --model-dir pi_export --detector pi_export/detector.onnx --camera opencv \
  --report-url http://<laptop-ip>:8080 --node-id node_0 --node-key demo-node-0-key --lat 4.47212 --lon 101.37913
```

### Real push notifications (APNs)

Without APNs credentials the service runs in dry-run mode. Every notification it would send is
listed at `GET /api/push-log` (send the farm token as the bearer token), and the app still shows
new alerts when refreshed. To deliver real pushes:

1. **Create a key:** in the Apple Developer portal > Keys, create a key with *Apple Push
   Notifications service* enabled and download `AuthKey_<KEYID>.p8`.
2. **Configure the service:** copy `services/alerts/config.example.json` to `config.json`,
   **change every token and key**, and add:
   ```json
   "apns": {"team_id": "ABCDE12345", "key_id": "KEYID", "key_path": "services/alerts/AuthKey_KEYID.p8",
            "bundle_id": "org.hivemind.farmer", "use_sandbox": true}
   ```
   `use_sandbox: true` is for builds run from Xcode. TestFlight and App Store builds need `false`,
   and the entitlement set to `production`.
3. **Enable push in the project:** add `CODE_SIGN_ENTITLEMENTS: push/HiveMindFarmer.entitlements`
   and the `remote-notification` background mode to the target in `project.yml` (see the comment
   there), and run `xcodegen generate` again.
4. **Run the app:** on a real iPhone (the simulator can't receive APNs pushes) with Demo mode off.
   The phone registers its token with the server automatically.
