"""One-off script to render docker/architecture.png -- a diagram of the
Docker mesh's current architecture (coordinator as scheduler-only,
peer-to-peer knowledge exchange between nodes, dashboard reading each
node's own db). Not part of the pipeline; re-run manually if the
architecture changes:

    python scripts/render_docker_architecture.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 1180
BG = (255, 255, 255)
INK = (30, 34, 40)
BLUE = (39, 98, 197)
GREEN = (30, 140, 80)
PURPLE = (130, 71, 191)
GREY = (110, 118, 128)
LIGHT = (247, 249, 251)

FONT_DIR = Path("C:/Windows/Fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


f_title = _font("segoeuib.ttf", 30)
f_h2 = _font("segoeuib.ttf", 19)
f_box_title = _font("segoeuib.ttf", 19)
f_body = _font("segoeui.ttf", 15)
f_small = _font("segoeui.ttf", 14)
f_label = _font("segoeuii.ttf", 14)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def rrect(xy, radius, outline, width=2, fill=None):
    d.rounded_rectangle(xy, radius=radius, outline=outline, width=width, fill=fill)


def centered_text(cx, y, text, font, fill=INK):
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    d.text((cx - w / 2, y), text, font=font, fill=fill)


def multiline(cx, y, lines, font, fill=INK, line_h=18):
    for i, line in enumerate(lines):
        centered_text(cx, y + i * line_h, line, font, fill=fill)


def seg(p1, p2, color, width=2, dash=False):
    x1, y1 = p1
    x2, y2 = p2
    if not dash:
        d.line([p1, p2], fill=color, width=width)
        return
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(int(length // 9), 1)
    for i in range(0, n, 2):
        t0, t1 = i / n, min((i + 1) / n, 1)
        d.line([(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)], fill=color, width=width)


def head(p_from, p_to, color, width=2, size=9):
    x1, y1 = p_from
    x2, y2 = p_to
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (0.5, -0.5):
        hx = x2 - size * math.cos(ang - da)
        hy = y2 - size * math.sin(ang - da)
        d.line([(x2, y2), (hx, hy)], fill=color, width=width)


def label_box(cx, cy, text, font=f_label, fill=INK):
    bbox = d.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.rectangle([cx - w / 2 - 5, cy - h / 2 - 3, cx + w / 2 + 5, cy + h / 2 + 5], fill=(255, 255, 255))
    d.text((cx - w / 2, cy - h / 2 - 2), text, font=font, fill=fill)


# ---- Title ----
centered_text(W / 2, 18, "Crop Mesh \u2014 Docker Architecture", f_title)
centered_text(W / 2, 56, "5 containers on one Docker \"mesh\" network  \u2014  knowledge exchange is peer-to-peer; the coordinator only schedules rounds", f_body, fill=GREY)

# ---- Outer host box ----
rrect((30, 92, W - 30, H - 30), 16, GREY, width=2)
d.text((50, 102), "Host machine (Docker Desktop)", font=f_h2, fill=GREY)

# ---- Docker network box ----
net_top, net_bottom = 140, 800
rrect((60, net_top, W - 60, net_bottom), 14, BLUE, width=2)
d.text((80, net_top + 10), "docker network: mesh", font=f_h2, fill=BLUE)

# ---- Coordinator + Dashboard boxes ----
coord_box = (560, 195, 860, 330)
rrect(coord_box, 10, INK, width=2, fill=LIGHT)
coord_cx = (coord_box[0] + coord_box[2]) / 2
centered_text(coord_cx, coord_box[1] + 10, "coordinator", f_box_title)
multiline(coord_cx, coord_box[1] + 38, [
    "port 9000",
    "sequences /round/start -> /round/gather",
    "owns NO round_metrics data",
    "only writes status.json",
    "exposes /events + /log",
], f_small, line_h=17)

dash_box = (1000, 195, 1300, 330)
rrect(dash_box, 10, PURPLE, width=2, fill=LIGHT)
dash_cx = (dash_box[0] + dash_box[2]) / 2
centered_text(dash_cx, dash_box[1] + 10, "dashboard", f_box_title)
multiline(dash_cx, dash_box[1] + 38, [
    "port 8501 (Streamlit)",
    "read-only \u2014 cannot trigger a run",
    "polls each node's /health, /log",
    "reads each node's OWN db directly",
    "combines all 3 itself (no merged db)",
], f_small, line_h=17)

# ---- Node boxes ----
node_top = 500
node_w, node_h = 300, 210
node_xs = [110, 610, 1110]
node_ids = ["node_0", "node_1", "node_2"]
node_cxs = [x + node_w / 2 for x in node_xs]
for x, cx, nid in zip(node_xs, node_cxs, node_ids):
    box = (x, node_top, x + node_w, node_top + node_h)
    rrect(box, 10, GREEN, width=2, fill=LIGHT)
    centered_text(cx, node_top + 10, nid, f_box_title)
    multiline(cx, node_top + 38, [
        "port 8000 (internal only)",
        "own read-only data shard",
        "local_train -> compute_knowledge",
        "fetches peers' knowledge itself",
        "aggregates locally",
        "(trimmed_mean / krum)",
        "distill -> evaluate",
        f"writes {nid}.db itself",
    ], f_small, line_h=17)

# ---- Control bus: coordinator -> bus -> each node (orthogonal, solid black) ----
bus_y = 400
seg((coord_cx, coord_box[3]), (coord_cx, bus_y), INK, 2)
seg((node_cxs[0], bus_y), (node_cxs[2], bus_y), INK, 2)
for cx in node_cxs:
    seg((cx, bus_y), (cx, node_top - 4), INK, 2)
    head((cx, node_top - 20), (cx, node_top - 4), INK, 2)
label_box(coord_cx, bus_y - 16, "/round/start, /round/gather", font=f_label, fill=INK)
label_box(coord_cx, bus_y + 16, "(scheduling only \u2014 never reads/aggregates knowledge)", font=f_small, fill=GREY)

# ---- Peer-to-peer knowledge exchange band (below node boxes, dashed green) ----
p2p_y = node_top + node_h + 40
for cx in node_cxs:
    seg((cx, node_top + node_h), (cx, p2p_y), GREEN, 2, dash=True)
seg((node_cxs[0], p2p_y), (node_cxs[2], p2p_y), GREEN, 2, dash=True)
# double-headed arrows between adjacent nodes
head((node_cxs[0] + 30, p2p_y), (node_cxs[0], p2p_y), GREEN, 2)
head((node_cxs[1] - 30, p2p_y), (node_cxs[1], p2p_y), GREEN, 2)
head((node_cxs[1] + 30, p2p_y), (node_cxs[1], p2p_y), GREEN, 2)
head((node_cxs[2] - 30, p2p_y), (node_cxs[2], p2p_y), GREEN, 2)
label_box((node_cxs[0] + node_cxs[2]) / 2, p2p_y + 22, "GET /knowledge/{round}  \u2014  direct peer-to-peer, coordinator never involved", font=f_label, fill=GREEN)

# ---- Dashboard read-only polling bus (right edge, dashed purple) ----
# Routed entirely below/beside both boxes so it never crosses their text.
dash_cx_bottom = (dash_box[0] + dash_box[2]) / 2
bus2_y = 360
right_x = W - 70
node_mid_y = node_top + node_h / 2

seg((dash_cx_bottom, dash_box[3]), (dash_cx_bottom, bus2_y), PURPLE, 2, dash=True)
seg((coord_box[2], bus2_y), (dash_cx_bottom, bus2_y), PURPLE, 2, dash=True)
seg((coord_box[2], bus2_y), (coord_box[2], coord_box[1] + 90), PURPLE, 2, dash=True)
head((coord_box[2] + 14, coord_box[1] + 90), (coord_box[2], coord_box[1] + 90), PURPLE, 2)
label_box((coord_box[2] + dash_cx_bottom) / 2, bus2_y + 18, "reads /events, /log", font=f_label, fill=PURPLE)

seg((dash_cx_bottom, bus2_y), (right_x, bus2_y), PURPLE, 2, dash=True)
seg((right_x, bus2_y), (right_x, node_mid_y), PURPLE, 2, dash=True)
seg((right_x, node_mid_y), (node_xs[2] + node_w + 4, node_mid_y), PURPLE, 2, dash=True)
head((node_xs[2] + node_w + 14, node_mid_y), (node_xs[2] + node_w, node_mid_y), PURPLE, 2)
label_cx = right_x - 130
label_box(label_cx, 424, "polls every node's /health, /log", font=f_label, fill=PURPLE)
label_box(label_cx, 448, "+ reads its db directly (read-only \u2014", font=f_label, fill=PURPLE)
label_box(label_cx, 472, "same for node_0 / node_1, omitted for clarity)", font=f_label, fill=PURPLE)

# ---- Host filesystem / bind mounts ----
fs_top = net_bottom + 30
rrect((60, fs_top, W - 60, H - 60), 14, GREY, width=2)
d.text((80, fs_top + 10), "host bind mounts (read-only unless noted)", font=f_h2, fill=GREY)

mounts = [
    ("config.yaml", "-> /config/config.yaml (ro)", "all 5 services"),
    ("data/docker_mesh/{node_0,node_1,node_2,probe,classes.json}", "-> /data/... (ro)", "node_0 / node_1 / node_2 (own shard only)"),
    ("outputs/docker_mesh/energy/", "each node writes its OWN node_X.db; coordinator writes only status.json", "dashboard mounts this dir read-only and combines all 3 dbs itself"),
]
my = fs_top + 50
for path, detail, who in mounts:
    d.ellipse([80, my + 6, 90, my + 16], fill=GREY)
    d.text((100, my), path, font=f_box_title, fill=INK)
    d.text((100, my + 24), detail, font=f_small, fill=GREY)
    d.text((100, my + 46), f"used by: {who}", font=f_small, fill=BLUE)
    my += 74

# ---- Legend ----
legend_y = H - 42
seg((80, legend_y), (120, legend_y), INK, 2)
d.text((128, legend_y - 8), "control / scheduling", font=f_small, fill=INK)
seg((360, legend_y), (400, legend_y), GREEN, 2, dash=True)
d.text((408, legend_y - 8), "peer-to-peer knowledge exchange", font=f_small, fill=GREEN)
seg((760, legend_y), (800, legend_y), PURPLE, 2, dash=True)
d.text((808, legend_y - 8), "read-only polling (dashboard)", font=f_small, fill=PURPLE)

out_path = Path(__file__).resolve().parent.parent / "docker" / "architecture.png"
img.save(out_path)
print(f"Wrote {out_path}")
