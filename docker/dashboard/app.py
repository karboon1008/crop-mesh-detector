# docker/dashboard/app.py
"""Read-only observability dashboard: polls each node's /health directly,
polls the coordinator's /events for a live request log, and reads the
merged SQLite metrics. No Docker socket access, and it only ever issues
GET requests -- it cannot trigger a run. Excluded from the sustainability
accounting scope (see spec §8): its own CPU usage is outside every
ComputeEnergyTracker scope in the node containers.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import pandas as pd
import streamlit as st

from data import build_status_rows
from src.energy.sqlite_store import read_all

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
MERGED_DB = os.environ.get("MERGED_DB", "/energy/merged.db")
REFRESH_S = float(os.environ.get("REFRESH_S", "3"))


def _poll_health() -> dict:
    health = {}
    for node_id, base_url in NODE_BASE_URLS.items():
        try:
            resp = httpx.get(f"{base_url}/health", timeout=3.0)
            health[node_id] = resp.status_code == 200
        except httpx.HTTPError:
            health[node_id] = False
    return health


def _poll_events() -> list:
    try:
        resp = httpx.get(COORDINATOR_EVENTS_URL, timeout=3.0)
        return resp.json() if resp.status_code == 200 else []
    except httpx.HTTPError:
        return []


def render() -> None:
    st.set_page_config(page_title="Mesh dashboard", layout="wide")
    st.title("Docker/HTTP mesh — live status")

    st.subheader("Node status")
    st.dataframe(pd.DataFrame(build_status_rows(_poll_health())), use_container_width=True)

    st.subheader("Round metrics (merged.db)")
    rows = read_all(MERGED_DB)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        for metric in ["energy_kwh", "knowledge_bytes_sent", "crop_accuracy"]:
            if metric in df.columns:
                pivot = df.pivot_table(index="round_idx", columns="node_id", values=metric)
                st.line_chart(pivot)
    else:
        st.write("No rows yet.")

    st.subheader("Coordinator event log")
    st.dataframe(pd.DataFrame(list(reversed(_poll_events()))), use_container_width=True)

    time.sleep(REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    render()
