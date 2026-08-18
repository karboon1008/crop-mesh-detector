# docker/dashboard/app.py
"""Read-only observability dashboard: polls each node's /health and /log
directly, polls the coordinator's /events and /log for a live request/stage
log, and reads the merged + per-node SQLite metrics. No Docker socket
access, and it only ever issues GET requests -- it cannot trigger a run.
Excluded from the sustainability accounting scope (see spec §8): its own
CPU usage is outside every ComputeEnergyTracker scope in the node
containers.
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from data import (
    build_final_result_payload,
    build_log_lines,
    build_status_rows,
    is_run_complete,
    merge_transfer_rows,
    rows_for_node,
    to_json_str,
)
from src.energy.sqlite_store import read_all, read_transfers

NUM_NODES = int(os.environ.get("NUM_NODES", "3"))
# Default reproduces the Docker-network behaviour exactly; overridable with the
# same named {node_id}/{index} placeholders as the coordinator's
# NODE_URL_TEMPLATE, so the dashboard can also point at nodes running on local
# ports outside Docker.
NODE_URL_TEMPLATE = os.environ.get("NODE_URL_TEMPLATE", "http://{node_id}:8000")
NODE_BASE_URLS = {
    f"node_{i}": NODE_URL_TEMPLATE.format(node_id=f"node_{i}", index=i) for i in range(NUM_NODES)
}
COORDINATOR_EVENTS_URL = os.environ.get("COORDINATOR_EVENTS_URL", "http://coordinator:9000/events")
COORDINATOR_LOG_URL = COORDINATOR_EVENTS_URL.rsplit("/", 1)[0] + "/log"
MERGED_DB = os.environ.get("MERGED_DB", "/energy/merged.db")
REFRESH_S = float(os.environ.get("REFRESH_S", "3"))

# Each node writes its own db alongside merged.db (see docker-compose.yml's
# per-node ENERGY_DB), so no extra env var is needed to find them.
NODE_DB_PATHS = {node_id: str(Path(MERGED_DB).parent / f"{node_id}.db") for node_id in NODE_BASE_URLS}
STATUS_PATH = Path(MERGED_DB).parent / "status.json"

CHART_METRICS = [
    ("energy_kwh", "Compute energy per round"),
    ("duration_s", "Round duration per round"),
    ("knowledge_bytes_sent", "Estimated knowledge bytes sent per round"),
    ("crop_accuracy", "Crop accuracy per round"),
    ("disease_accuracy", "Disease accuracy per round"),
]

STAGE_COLORS = {
    "round_start": "#4fc3f7",
    "local_train": "#ffb74d",
    "baseline_local_train": "#e57373",
    "round_gather": "#ba68c8",
    "distill_kd": "#81c784",
    "distill_sup": "#81c784",
}
DEFAULT_STAGE_COLOR = "#90a4ae"

HEADER_CSS = """
<style>
@keyframes blink-live { 0%, 100% { opacity: 1; } 50% { opacity: 0.15; } }
.live-dot {
    height: 12px; width: 12px; border-radius: 50%; background: #ff3b3b;
    display: inline-block; margin-right: 8px; animation: blink-live 1.2s infinite;
}
.live-badge {
    display: flex; align-items: center; justify-content: flex-end;
    height: 100%; font-weight: 700; color: #ff3b3b; letter-spacing: 0.05em;
}
@keyframes blink-online { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
@keyframes blink-offline { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.status-row { display: flex; gap: 28px; align-items: center; flex-wrap: wrap; padding: 4px 0 12px; }
.status-item { display: flex; align-items: center; font-weight: 600; font-size: 0.95rem; }
.status-dot { height: 11px; width: 11px; border-radius: 50%; display: inline-block; margin-right: 8px; }
.status-dot.online { background: #3fb950; animation: blink-online 1.4s infinite; }
.status-dot.offline { background: #f85149; animation: blink-offline 1.4s infinite; }
</style>
"""

LOG_PANEL_CSS = """
<style>
body { margin: 0; background: transparent; }
.log-panel {
    height: 260px; overflow-y: auto; background: #0d1117; color: #c9d1d9;
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.8rem; line-height: 1.55; padding: 10px 12px; border: 1px solid #262b36;
    border-radius: 6px; box-sizing: border-box;
}
.log-line { white-space: pre-wrap; word-break: break-word; }
.log-time { color: #6e7681; margin-right: 6px; }
.log-stage { font-weight: 600; margin-right: 6px; }
.log-empty { color: #6e7681; font-style: italic; }
</style>
"""


def _log_panel_html(log_entries: list, panel_key: str) -> str:
    lines = build_log_lines(log_entries)
    body_parts = []
    for row in lines:
        time_str = html.escape(row["time"])
        message = html.escape(row.get("message", ""))
        stage = row.get("stage")
        if stage:
            color = STAGE_COLORS.get(stage, DEFAULT_STAGE_COLOR)
            stage_html = f'<span class="log-stage" style="color:{color}">{html.escape(stage)}</span>'
        else:
            stage_html = ""
        body_parts.append(
            f'<div class="log-line"><span class="log-time">[{time_str}]</span>{stage_html}{message}</div>'
        )
    body = "\n".join(body_parts) if body_parts else '<div class="log-empty">No activity yet.</div>'
    return f"""
{LOG_PANEL_CSS}
<div class="log-panel" id="log-{panel_key}">{body}</div>
<script>
  var el = document.getElementById("log-{panel_key}");
  if (el) {{ el.scrollTop = el.scrollHeight; }}
</script>
"""


def _poll_health() -> dict:
    health = {}
    for node_id, base_url in NODE_BASE_URLS.items():
        try:
            resp = httpx.get(f"{base_url}/health", timeout=3.0)
            health[node_id] = resp.status_code == 200
        except httpx.HTTPError:
            health[node_id] = False
    return health


def _poll_json(url: str) -> list:
    try:
        resp = httpx.get(url, timeout=3.0)
        return resp.json() if resp.status_code == 200 else []
    except httpx.HTTPError:
        return []


def _read_status() -> dict | None:
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _render_activity_panel(title: str, log_entries: list) -> None:
    with st.expander(title, expanded=True):
        components.html(_log_panel_html(log_entries, panel_key=title), height=280, scrolling=False)


def _status_row_html(status_rows: list) -> str:
    items = []
    for row in status_rows:
        state = "online" if row["online"] else "offline"
        items.append(
            f'<div class="status-item"><span class="status-dot {state}"></span>'
            f'{html.escape(row["node_id"])} — {state}</div>'
        )
    return f'<div class="status-row">{"".join(items)}</div>'


def render() -> None:
    st.set_page_config(page_title="Crop Mesh Dashboard", layout="wide")
    st.markdown(HEADER_CSS, unsafe_allow_html=True)

    header_col, badge_col = st.columns([6, 1])
    header_col.title("Crop Mesh Dashboard")
    badge_col.markdown(
        '<div class="live-badge"><span class="live-dot"></span>LIVE</div>', unsafe_allow_html=True
    )

    st.subheader("Status")
    st.markdown(_status_row_html(build_status_rows(_poll_health())), unsafe_allow_html=True)

    st.subheader("Node activity")
    st.caption("What each node is doing right now — refreshes every few seconds.")
    cols = st.columns(len(NODE_BASE_URLS))
    for col, node_id in zip(cols, sorted(NODE_BASE_URLS)):
        with col:
            _render_activity_panel(node_id, _poll_json(f"{NODE_BASE_URLS[node_id]}/log"))
    _render_activity_panel("coordinator", _poll_json(COORDINATOR_LOG_URL))

    rows = read_all(MERGED_DB)
    with st.expander("Charts", expanded=True):
        if rows:
            df = pd.DataFrame(rows)
            for metric, chart_title in CHART_METRICS:
                if metric not in df.columns or df[metric].dropna().empty:
                    continue
                fig = px.line(
                    df.dropna(subset=[metric]),
                    x="round_idx",
                    y=metric,
                    color="node_id",
                    markers=True,
                    title=chart_title,
                )
                fig.update_layout(xaxis_title="Round", yaxis_title=metric, legend_title="Node")
                fig.update_xaxes(dtick=1)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data yet — charts will appear once a round finishes.")

    st.subheader("Round metrics (merged.db)")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.write("No rows yet.")

    st.subheader("Knowledge transfers")
    st.caption("Every completed peer-to-peer GET /knowledge/{round} pull, with actual payload size.")
    transfers = merge_transfer_rows([read_transfers(path) for path in NODE_DB_PATHS.values()])
    if transfers:
        st.dataframe(pd.DataFrame(transfers), use_container_width=True)
    else:
        st.write("No transfers yet.")

    st.subheader("Coordinator event log")
    st.dataframe(pd.DataFrame(list(reversed(_poll_json(COORDINATOR_EVENTS_URL)))), use_container_width=True)

    status = _read_status()
    st.header("Final results")
    if is_run_complete(status):
        st.success(f"All {status['num_rounds']} round(s) complete at {status['completed_at']}.")
        final_df = pd.DataFrame(rows)
        last_round = final_df["round_idx"].max()
        final_round_df = final_df[final_df["round_idx"] == last_round]

        st.dataframe(final_round_df, use_container_width=True)

        acc_cols = [c for c in ["crop_accuracy", "disease_accuracy"] if c in final_round_df.columns]
        if acc_cols:
            acc_long = final_round_df.melt(
                id_vars="node_id", value_vars=acc_cols, var_name="metric", value_name="accuracy"
            )
            fig = px.bar(
                acc_long, x="node_id", y="accuracy", color="metric", barmode="group",
                title="Final accuracy per node",
            )
            fig.update_layout(xaxis_title="Node", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)

        if "energy_kwh" in final_df.columns:
            energy_by_node = final_df.groupby("node_id")["energy_kwh"].sum().reset_index()
            fig = px.bar(
                energy_by_node, x="node_id", y="energy_kwh",
                title="Total compute energy per node (all rounds)",
            )
            fig.update_layout(xaxis_title="Node", yaxis_title="Energy (kWh)")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Export results")
        node_ids = sorted({r["node_id"] for r in rows})
        export_cols = st.columns(len(node_ids) + 1)
        for col, node_id in zip(export_cols, node_ids):
            col.download_button(
                f"Download {node_id}.json",
                data=to_json_str(rows_for_node(rows, node_id)),
                file_name=f"{node_id}_result.json",
                mime="application/json",
            )
        export_cols[-1].download_button(
            "Download final_result.json",
            data=to_json_str(build_final_result_payload(rows, transfers, status)),
            file_name="final_result.json",
            mime="application/json",
        )
    else:
        st.info("Run still in progress — final results will appear here once all rounds complete.")

    time.sleep(REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    render()
