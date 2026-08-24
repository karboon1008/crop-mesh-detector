# docker/coordinator/main.py
"""Entry point for the round coordinator. Fan-out to nodes uses
asyncio.gather over httpx.AsyncClient for concurrency -- sequential calls
would force nodes to train one at a time instead of in parallel. Exposes
a tiny /events endpoint purely for the dashboard to poll -- it records
only which path was called and what status came back, never touching
knowledge content. The coordinator owns no round_metrics data at all (see
coordinator_runner.py) -- the only thing it ever writes to disk is a
small status.json completion marker so the dashboard knows the run has
finished; every node's own energy/accuracy/communication numbers live
exclusively in that node's own db.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx
import uvicorn
from fastapi import FastAPI

from src.config import Config
from coordinator_runner import CoordinatorRunner

events: list = []
MAX_EVENTS = 200


async def _post_one(client: httpx.AsyncClient, node_id: str, url: str, body: dict, timeout: float):
    try:
        resp = await client.post(url, json=body, timeout=timeout)
        events.append({"ts": time.time(), "node_id": node_id, "path": url, "status": resp.status_code})
        events[:] = events[-MAX_EVENTS:]
        if resp.status_code != 200:
            return node_id, None
        return node_id, resp.json()
    except (httpx.HTTPError, ValueError):
        # ValueError covers json.JSONDecodeError -- a node returning HTTP 200
        # with a non-JSON body must not crash the whole gather() fan-out, it
        # should just be excluded from this round like any other failure.
        events.append({"ts": time.time(), "node_id": node_id, "path": url, "status": "error"})
        events[:] = events[-MAX_EVENTS:]
        return node_id, None


async def _post_all_async(node_base_urls: dict, node_ids: list, path: str, body: dict, timeout: float) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_post_one(client, n, f"{node_base_urls[n]}{path}", body, timeout) for n in node_ids)
        )
    return dict(results)


def make_post_all(node_base_urls: dict, timeout: float):
    def post_all(node_ids: list, path: str, body: dict) -> dict:
        if not node_ids:
            return {}
        return asyncio.run(_post_all_async(node_base_urls, node_ids, path, body, timeout))

    return post_all


def make_health_check(node_base_urls: dict):
    def health_check(node_id: str) -> bool:
        try:
            resp = httpx.get(f"{node_base_urls[node_id]}/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    return health_check


def main() -> None:
    cfg = Config.load(os.environ.get("CONFIG_PATH"))
    num_nodes = cfg.get("data.num_nodes", 3)
    expected_nodes = [f"node_{i}" for i in range(num_nodes)]
    # Default reproduces the Docker-network behaviour exactly (service name as
    # hostname, port 8000), so docker-compose.yml needs no change. Overridable
    # so peers can live at other hosts/ports without editing this file.
    # Placeholders are named, not positional, so the env var's value is
    # self-documenting: {node_id} is e.g. "node_1", {index} is e.g. 1. {index}
    # is what makes the no-Docker workflow expressible -- three node processes
    # on three local ports is NODE_URL_TEMPLATE="http://127.0.0.1:810{index}"
    # (each node process picking up the matching port via its own PORT var).
    node_url_template = os.environ.get("NODE_URL_TEMPLATE", "http://{node_id}:8000")
    node_base_urls = {
        n: node_url_template.format(node_id=n, index=i) for i, n in enumerate(expected_nodes)
    }
    round_timeout_s = cfg.get("docker_mesh.round_timeout_s", 300)
    status_path = os.environ["STATUS_PATH"]
    # Every container start is a fresh run, not a resume -- wipe the
    # previous run's completion marker so the dashboard can't show a stale
    # "final results" from before this restart. This is the only path the
    # coordinator ever writes.
    Path(status_path).unlink(missing_ok=True)

    runner = CoordinatorRunner(
        expected_nodes=expected_nodes,
        node_base_urls=node_base_urls,
        num_rounds=cfg.get("training.rounds", 5),
        round_timeout_s=round_timeout_s,
        post_all=make_post_all(node_base_urls, round_timeout_s),
        health_check=make_health_check(node_base_urls),
        status_path=status_path,
    )

    events_app = FastAPI()

    @events_app.get("/events")
    def get_events():
        return events[-MAX_EVENTS:]

    @events_app.get("/log")
    def get_log():
        return runner.activity_log[-200:]

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(events_app, host="0.0.0.0", port=9000), daemon=True
    )
    server_thread.start()

    print(f"[coordinator] waiting for {expected_nodes} to come online...")
    runner.wait_until_all_online()
    print("[coordinator] all nodes online, starting round loop")
    runner.run_all_rounds()
    print("[coordinator] all rounds complete")

    threading.Event().wait()  # idle -- no re-run trigger yet, see spec §12


if __name__ == "__main__":
    main()
