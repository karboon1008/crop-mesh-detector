# Farmer mobile app: design and feature proposals

Goal: when a HiveMind field camera finds a disease, the farmer knows within seconds, knows
**exactly where** the plant is, and knows **what to do first**, even with poor signal in the field.

## System design

```
 Field (per farm)                         Farm server / cloud                 Farmer
 ┌──────────────────────────┐   HTTPS    ┌──────────────────────────┐  APNs  ┌──────────────┐
 │ Pi camera node           │──────────▶ │ services/alerts          │──────▶ │ iPhone app   │
 │ leaf detector → classifier│ detections │  • merge repeat detections│        │ mobile/ios   │
 │ GPS or surveyed position │ heartbeats │  • cooldown reminders     │ ◀──────│              │
 │ outbox while offline     │            │  • neighbour risk (≈1 km) │  REST  │ offline cache│
 └──────────────────────────┘            │  • farmer feedback        │        └──────────────┘
          ▲                              └────────────┬─────────────┘
          │ knowledge mesh (unchanged: prototypes + probe logits, never images/weights)
          └───────────── other farms' nodes ◀──────────┘ false-alarm labels → next continual batch
```

**Why a small alerts service instead of pushing straight from the Pi:**
- **Signing key:** APNs needs a signing key, and it shouldn't sit on every field device.
- **Knowing where to send:** someone has to track which phones belong to which farm.
- **One alert per outbreak:** a camera sees the same sick plant in every frame. Without merging,
  the farmer gets dozens of pushes an hour and turns notifications off.
- **Warnings across farms:** a Pi can't warn neighbouring farms. The service can, sharing only
  a rounded location.

**Where the coordinates come from:**
- **Node GPS:** if the node has a GPS module, it sends the position with each detection.
- **Surveyed position:** otherwise the service uses the node's position from its config. The
  surveyed position is accurate to a few metres, which is enough to find a greenhouse row.
- **Later, per plant:** see *Precise plant position* below.

**Privacy stays consistent with the mesh:**
- **Neighbour notices:** only the disease, the crop, a location rounded to about 1 km, and the
  distance, to the nearest 0.5 km.
- **Never shared:** the other farm's name, its fields, or photos.

## What is built (v1)

| Area | Feature | Where |
|---|---|---|
| Alerts | Push on a new outbreak, deduplicated per node, crop and disease; a reminder at most every 6 h; a new alert if it comes back after being treated | `services/alerts/app.py` |
| Location | Map pin, decimal and degrees/minutes/seconds coordinates, distance and direction from the farmer, Apple Maps walking directions, share the location with a worker | `AlertDetailView.swift` |
| Notification actions | "Navigate to plant" and "Mark as seen" straight from the lock screen | `Notifications.swift` |
| What to do | Offline treatment guide for all 20 disease classes the model detects, with urgency, how it spreads, what to look for, first steps and prevention. It always ends with "check with your extension officer" | `treatments.json` |
| Feedback loop | Treated / false alarm with a note. False alarms become labels for the next continual‑learning batch | `PATCH /api/alerts/{id}` |
| Neighbour warning | "Late blight about 2.5 km away" push, plus orange circles on the map | `notify_neighbours`, `FarmMapView` |
| Device health | Camera online/offline, last contact, battery; low battery is flagged | `NodesView.swift`, heartbeat |
| Offline first | Cached data, "offline" banner, queued status changes; the Pi queues reports while it has no signal | `AlertStore.swift`, `AlertReporter` |
| Demo | Built‑in sample farm and a simulated notification, so no server is needed | `DemoData.swift` |

## Recommended next features

Ranked by value to a real farmer divided by effort.

1. **Phone number sign‑in and roles (owner / worker).**
   - **Why:** today a farm shares one code.
   - **What:** each person logs in with an SMS one‑time code. Owners see everything. Workers get
     alerts for their assigned plots and can mark them treated. Every action records who did it.
2. **SMS / WhatsApp fallback.**
   - **Why:** many farm workers have basic phones or no data plan.
   - **What:** send the same alert as text with a map link (e.g. through Twilio) when no app
     device confirms it within N minutes.
3. **Scouting mode on the phone.**
   - **What:** convert the classifier and leaf detector to Core ML and run them on the iPhone
     with the camera, fully offline. The farmer checks nearby plants after an alert and
     confirms or rejects it on the spot.
   - **Payoff:** those confirmations are high‑quality labels for the mesh.
4. **Weather‑based risk forecast.**
   - **Why:** late blight, downy mildew and similar diseases follow humidity, leaf wetness and
     temperature.
   - **What:** pull a local forecast, or read a humidity sensor on the node, and warn "High
     blight risk for the next 48 h: scout tomatoes" before anything is detected.
5. **Treatment log and spray records.**
   - **What:** record the product, dose, date and who applied it, and show the harvest
     interval ("safe to harvest after 14 Oct").
   - **Payoff:** needed for Good Agricultural Practice certification (e.g. MyGAP in Malaysia) and
     export buyers. It also measures whether treatment worked (did the alert recur?).
6. **Precise plant position.**
   - **What:** with a fixed camera, combine the leaf box position with the camera's heading and
     field of view to estimate the plant's position (roughly ±2 m). Alternatively, show a
     "plant is in row 7, 3rd from the left" description taken from a field layout.
7. **Outbreak heatmap and timeline.**
   - **What:** see how an outbreak spreads over days across plots, and which camera saw it first.
   - **Payoff:** helps decide where to put cameras and when to quarantine a plot.
8. **Tasks and assignment.**
   - **What:** "Remove infected plants in greenhouse A" is assigned to a worker, with a due
     time and a photo as proof when done. It ties into roles (1) and the treatment log (5).
9. **Multiple languages and voice.**
   - **What:** Bahasa Melayu, Chinese and Tamil for Malaysian farms, with large text and
     read‑aloud alerts for low‑literacy users. The guide JSON is already keyed by disease, so
     translations drop in.
10. **Extension officer and cooperative view.**
    - **What:** an anonymised regional dashboard of outbreaks, built on the same rounding as
      neighbour warnings, for agricultural officers to plan interventions.
    - **Payoff:** matches the mesh's "learn together, share no raw data" story.
11. **Quiet hours and priorities.**
    - **What:** "Act today" diseases always alert. Lower urgency waits for the morning summary.
    - **Uses:** Time Sensitive and Critical Alerts on iOS for high‑urgency outbreaks, where approved.
12. **Market and yield impact estimate.**
    - **What:** show the value at risk ("~RM 1,200 of tomatoes in greenhouse A") to help farmers
      prioritise. It needs the crop area and price inputs.

## Production hardening checklist

- **Transport:** HTTPS only. Remove `NSAllowsArbitraryLoads` from the app, and put the service
  behind TLS (e.g. Caddy or nginx).
- **Credentials:** per‑device node keys, stored on the Pi outside the repo; rotate the demo keys.
  App tokens are replaced by real accounts (feature 1).
- **Database:** SQLite is fine for one farm or cooperative. Use PostgreSQL once several farms
  report at once.
- **Push certificates:** separate sandbox and production APNs settings, and monitor the push
  log for failures.
- **Retention:** keep evidence photos for a limited period (e.g. 90 days), and let farmers
  delete their data.
