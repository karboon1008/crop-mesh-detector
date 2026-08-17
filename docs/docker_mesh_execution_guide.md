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

**Data-Dependent Items:** Items marked **[REQUIRES DATA]** need the training dataset to exist at `data/docker_mesh/node_0/`, `data/docker_mesh/node_1/`, `data/docker_mesh/node_2/`, `data/docker_mesh/probe/`, and `data/docker_mesh/classes.json`. Items without this marker are pure UI/wiring checks that work in any environment.

1. `docker compose up -d --build` (with `HOST_PROJECT_ROOT` set) — all 6 containers reach a healthy/running state.

2. Open the dashboard at http://localhost:8501 — 4 tabs are visible: Full Run, Class Addition, Disconnection, Distribution Shift.

3. **[REQUIRES DATA]** Click Start on the "Full Run" tab. Confirm: the controller's state moves idle -> starting -> running (visible in the tab's state caption), node/coordinator log panels (expand them) begin showing activity, and `outputs/docker_mesh/energy/full_run/` appears on disk.

4. **[REQUIRES DATA]** While Full Run is still going, click Start on the "Disconnection" tab. Confirm: Full Run's containers stop (its state caption shows the tab is no longer running), Disconnection's containers start fresh, and `outputs/docker_mesh/energy/full_run/` is left untouched (Full Run's last data is still viewable in its own tab, un-overwritten) while `outputs/docker_mesh/energy/disconnection/` starts fresh.

5. **[REQUIRES DATA]** Let a run reach completion (or stop it early) and click "Download {scenario}_result.json" — confirm the downloaded file has `scenario`, `nodes`, `knowledge_transfers`, `fairness_disclosure`, and `complete` keys, and that `fairness_disclosure.per_node_scores` has one entry per node with both `mesh` and `baseline` accuracy figures.

6. **[REQUIRES DATA]** Click Start on the same tab a second time after it finished — confirm the per-round results table restarts from round 0 (clean slate), not appending to the previous run's rows.

### Bounded Verification: Infrastructure and Control-Plane Wiring (No Training Data Required)

If you do not have the training dataset under `data/docker_mesh/`, you can still verify the infrastructure and control-plane wiring using these curl-based checks:

1. **Initial bring-up and container creation:**
   - Run: `docker compose up -d --build` (from `docker/` with `HOST_PROJECT_ROOT` set)
   - Expected: All 6 containers are created. Run `docker compose ps -a` and confirm all are present.
   - Note: Node containers will exit with error code 1 if training data is missing — this is expected and not a bug.

2. **Dashboard HTTP endpoint:**
   - Run: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501`
   - Expected: HTTP 200 response (dashboard is reachable)

3. **Controller `/status` endpoint:**
   - Run: `curl http://localhost:9100/status`
   - Expected: JSON response such as `{"running_scenario":null,"state":"idle","started_at":null,"error_detail":null}`

4. **Controller `/start` endpoint:**
   - Run: `curl -X POST http://localhost:9100/start -H 'Content-Type: application/json' -d '{"scenario": "full_run"}'`
   - Expected: Controller responds (HTTP 200) and attempts to start containers; this tests the API wiring even if container startup fails due to missing data.

5. **Controller `/stop` endpoint:**
   - Run: `curl -X POST http://localhost:9100/stop`
   - Expected: Controller responds with JSON indicating idle state.

6. **Cleanup:**
   - Run: `docker compose down`
   - Expected: All containers stop and are removed, network cleaned up.
