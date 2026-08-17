# Docker Mesh Execution Guide

## Setup

### Step 1: Configure `HOST_PROJECT_ROOT`

The Docker Compose setup requires the `HOST_PROJECT_ROOT` environment variable to be set to the absolute path of this repository (as the Docker daemon on your machine would resolve it). This is used to mount the configuration, data, and output directories into the containers.

Create a `.env` file in the `docker/` directory with the following content:

```bash
HOST_PROJECT_ROOT=/path/to/crop-mesh-detector/.worktrees/dashboard-scenario-control-plane
```

On Linux/macOS:
```bash
cd docker
echo "HOST_PROJECT_ROOT=$(cd .. && pwd)" > .env
```

On Windows (PowerShell):
```powershell
cd docker
"HOST_PROJECT_ROOT=$(Convert-Path ..)" | Out-File -FilePath .env -Encoding UTF8
```

Docker Compose will automatically load the `.env` file when running commands from the `docker/` directory.

### Step 2: Initial Bring-Up

Run the following command from the `docker/` directory to bring up all 6 services:

```bash
cd docker
docker compose up -d --build
```

This creates:
- 1 **coordinator**: Orchestrates the training loop across nodes
- 3 **nodes** (node_0, node_1, node_2): Perform federated learning with local datasets
- 1 **controller**: Manages scenario transitions (start/stop)
- 1 **dashboard**: Streamlit UI at http://localhost:8501 showing real-time training progress and results

The `SCENARIO` environment variable defaults to `full_run`. All containers are created with this initial scenario label, but you may switch scenarios immediately via the dashboard or API without restarting compose.

### Step 3: Scenario Switching

After the initial `docker compose up -d --build`, all scenario switching happens via one of two methods:

#### Method 1: Dashboard UI (Recommended)

1. Open http://localhost:8501 in your browser
2. You will see 4 tabs: **Full Run**, **Class Addition**, **Disconnection**, and **Distribution Shift**
3. Each tab shows:
   - Current controller state (idle, starting, running, stopped)
   - Node and coordinator logs (expandable panels)
   - Per-round results table (populated once a run is in progress)
   - A Start button (begins the scenario) and a Download button (when completed)
4. Click **Start** on any tab to begin that scenario
5. Click **Stop** on the tab to halt it

#### Method 2: Direct API

Send HTTP requests to the controller's REST API at `http://localhost:9100`:

**Check status:**
```bash
curl http://localhost:9100/status
```

**Start a scenario:**
```bash
curl -X POST http://localhost:9100/start \
  -H 'Content-Type: application/json' \
  -d '{"scenario": "full_run"}'
```

Supported scenario names: `full_run`, `class_addition`, `disconnection`, `distribution_shift`

**Stop the current scenario:**
```bash
curl -X POST http://localhost:9100/stop
```

### Important Behavior

- **Only one scenario runs at a time.** Starting a new scenario automatically stops any currently running scenario.
- **Clean slate per scenario.** When a new scenario starts, all node containers and the coordinator are restarted fresh. Previous state is wiped.
- **Data isolation.** Each scenario's results are saved to a separate directory: `outputs/docker_mesh/energy/{scenario}/`. Previous runs' data is preserved.
- **No container restart needed.** The entire workflow—initial bring-up, scenario switching, switching again—happens without restarting the compose stack.

## Manual Verification (run once after implementing the control plane)

1. **Initial bring-up succeeds and all 6 containers are running.**
   - Run: `docker compose up -d --build` (from `docker/` with `HOST_PROJECT_ROOT` set)
   - Expected: All 6 containers start without errors. Run `docker compose ps` and confirm all show a "Running" or "Up" state.
   - Note: Node containers may show "unhealthy" or crash-loop if the training data directories (`data/docker_mesh/node_0`, etc.) do not exist. This is expected in environments without the full training dataset — see caveat below.

2. **Dashboard is reachable and displays 4 tabs.**
   - Run: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501`
   - Expected: HTTP 200 response
   - Or open http://localhost:8501 in a browser — you should see 4 tabs: Full Run, Class Addition, Disconnection, Distribution Shift

3. **Controller's `/status` endpoint responds.**
   - Run: `curl http://localhost:9100/status`
   - Expected: JSON response with controller state and current scenario information

4. **Controller accepts `POST /start` with a scenario payload.**
   - Run: `curl -X POST http://localhost:9100/start -H 'Content-Type: application/json' -d '{"scenario": "full_run"}'`
   - Expected: HTTP 200 response; controller begins launching the scenario's nodes and coordinator

5. **Controller accepts `POST /stop` and stops the scenario.**
   - Run: `curl -X POST http://localhost:9100/stop`
   - Expected: HTTP 200 response; containers stop gracefully

6. **Cleanup:**
   - Run: `docker compose down`
   - Expected: All containers stop and are removed

### Caveat: Training Data Not Available in All Environments

Steps in the checklist that require a completed training round (e.g., "let a run reach completion," "verify per-node accuracy figures in the downloaded JSON") require the training dataset to exist at `data/docker_mesh/node_0/`, `data/docker_mesh/node_1/`, `data/docker_mesh/node_2/`, and `data/docker_mesh/probe/`. If these directories do not exist in your environment, the node containers will crash or remain unhealthy; this is **not a bug**—it is expected behavior when no data is available.

To run the full checklist including training completion, ensure the training dataset is present:
- `data/docker_mesh/node_0/`: Training dataset for node 0
- `data/docker_mesh/node_1/`: Training dataset for node 1
- `data/docker_mesh/node_2/`: Training dataset for node 2
- `data/docker_mesh/probe/`: Probe dataset for evaluation
- `data/docker_mesh/classes.json`: Class labels (required by all nodes)

If you only have the code repository without the dataset, the control plane (dashboard, controller, scenario switching API) can still be verified via steps 1-5 above.
