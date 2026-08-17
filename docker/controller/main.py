# docker/controller/main.py
"""FastAPI wrapper around ControllerRunner. Talks to the *host* Docker
daemon over the mounted socket by shelling out to `docker compose`
against the SAME docker-compose.yml the user brought the stack up with
originally (bind-mounted read-only at /workspace) -- never a
docker-compose.yml baked into this image, so there is exactly one
source of truth for the topology. All volume paths in that file use
${HOST_PROJECT_ROOT} (absolute), never a relative `..`, because a
`docker compose` process running inside a container resolves relative
paths against ITS OWN filesystem view, then hands the daemon a path
that means something different on the real host -- absolute paths
sidestep that mismatch entirely.
"""

from __future__ import annotations

import os
import subprocess

from fastapi import FastAPI, HTTPException

from controller_runner import ControllerRunner

COMPOSE_FILE = "/workspace/docker/docker-compose.yml"
MANAGED_SERVICES = ["coordinator", "node_0", "node_1", "node_2"]
COMPOSE_TIMEOUT_S = 120


def _run_compose(args: list[str], env_overrides: dict) -> None:
    env = {**os.environ, **env_overrides}
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, *args],
            env=env, capture_output=True, text=True, timeout=COMPOSE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker compose {' '.join(args)} timed out after {COMPOSE_TIMEOUT_S}s"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed: {result.stderr.strip()}")


def start_scenario(scenario: str) -> None:
    _run_compose(["up", "-d", "--force-recreate", *MANAGED_SERVICES], {"SCENARIO": scenario})


def stop_scenario() -> None:
    _run_compose(["stop", *MANAGED_SERVICES], {})


runner = ControllerRunner(start_scenario=start_scenario, stop_scenario=stop_scenario)
app = FastAPI()


@app.get("/status")
def status():
    return runner.status()


@app.post("/start")
def start(body: dict):
    if "scenario" not in body:
        raise HTTPException(status_code=400, detail="missing required field: 'scenario'")
    try:
        return runner.start(body["scenario"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/stop")
def stop():
    try:
        return runner.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9100)
