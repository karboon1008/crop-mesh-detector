# Multi-Agent Innovation Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a shared ARM64 environment on `b6dj.aip2.isambard`, then dispatch three parallel research forks (communication, personalization, edge footprint), each seeking a mechanism that beats both FedAvg and D-PSGD outright on real, measured data.

**Architecture:** Tasks 1–4 are bounded, fully-specified infrastructure work (Isambard environment bootstrap) executed directly, once, shared by all three agents. Tasks 5–7 each dispatch one `Agent` tool call (`subagent_type: "fork"`, `isolation: "worktree"`) with a complete, self-contained directive prompt — the fork's own internal work (reasoning, implementation, validation) is necessarily open-ended and is scoped by the spec's §5 protocol and §7 report template, not by pre-written code, since the mechanism each agent invents is unknown until it reasons about it. Task 8 is the joint review checkpoint.

**Tech Stack:** Python 3.11 (`cray-python` module), PyTorch/timm for `aarch64`+CUDA 12.6, SLURM (`workq` partition, `brics.b6dj` account), existing HiveMind training infra (`src/train_local.py`, `src/federated/*`, `src/energy/tracker.py`).

**Spec:** `docs/superpowers/specs/2026-08-30-multi-agent-innovation-search-design.md`

## Global Constraints

- Never touch `~/work/crop-mesh-detector` (the real Isambard checkout) on any cluster.
- Never `git push` to `github.com/karboon1008/crop-mesh-detector`.
- Never modify anything already reported in the research document without flagging it as a proposed change for review.
- Primary compute: `b6dj.aip2.isambard` (account `brics.b6dj`, partition `workq`, `--gres=gpu:1`). Fallback: `b36bl.macs3.isambard` (account `brics.b36bl`, partition `ampere`, `--gres=gpu:1`) if `b6dj` setup fails for a given agent.
- Full-scale validation directly (3 architectures, N=6, 8 rounds) — no staged cheap-check-first phase.
- Multi-seed (3 seeds) re-run required before any agent reports an outright win as confirmed rather than preliminary.
- Every agent produces a structured report regardless of outcome (win, partial, or rigorous negative result) — no silent stops.

---

### Task 1: Verify `aarch64` PyTorch/CUDA works on `b6dj`

**Files:** none (Isambard-side verification only)

**Interfaces:**
- Produces: a confirmed, working `python3` + `pip` combination on `b6dj` that Task 2 builds the venv from.

- [ ] **Step 1: Check available Python and CUDA modules**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "module avail 2>&1 | grep -i 'python\|cuda'"
```

Expected: `cray-python/3.11.7` and `cuda/12.6` (or similar) listed.

- [ ] **Step 2: Load modules and check pip's view of the platform**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "module load cray-python/3.11.7 cuda/12.6 2>&1; python3 -c 'import platform; print(platform.machine(), platform.python_version())'"
```

Expected: `aarch64 3.11.7`.

- [ ] **Step 3: Test a GPU allocation directly (not the login node)**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "srun --account=brics.b6dj --partition=workq --gres=gpu:1 --time=00:05:00 nvidia-smi 2>&1"
```

Expected: real GPU info for an NVIDIA GH200 device.

---

### Task 2: Build the shared venv and install PyTorch for `aarch64`

**Files:** none (Isambard-side; creates `~/work/hivemind-agents-base/.venv` on `b6dj`)

**Interfaces:**
- Consumes: verified module set from Task 1.
- Produces: `~/work/hivemind-agents-base/.venv` on `b6dj`, activatable, with `torch.cuda.is_available() == True` inside a GPU allocation.

- [ ] **Step 1: Create the base directory and venv**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "mkdir -p ~/work/hivemind-agents-base && module load cray-python/3.11.7 && python3 -m venv ~/work/hivemind-agents-base/.venv"
```

- [ ] **Step 2: Install PyTorch + torchvision for CUDA 12.6 on aarch64**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "source ~/work/hivemind-agents-base/.venv/bin/activate && pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -30"
```

If this specific index URL 404s or resolves x86_64 wheels (PyTorch's aarch64+CUDA wheel availability varies by release), fall back to plain `pip install torch torchvision` (PyPI now ships `manylinux_2_28_aarch64` wheels for recent torch releases) and record which one actually worked in the Task 3 report.

- [ ] **Step 3: Verify CUDA is visible inside a real GPU allocation**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "srun --account=brics.b6dj --partition=workq --gres=gpu:1 --time=00:05:00 bash -c 'source ~/work/hivemind-agents-base/.venv/bin/activate && python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
```

Expected: a torch version string, `True`, and a GH200-identifying device name.

- [ ] **Step 4: Install the rest of HiveMind's dependencies into the same venv**

```powershell
scp -F "$env:USERPROFILE\.ssh\config_clifton" "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\requirements.txt" b6dj.aip2.isambard:~/work/hivemind-agents-base/requirements.txt
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "source ~/work/hivemind-agents-base/.venv/bin/activate && pip install -r ~/work/hivemind-agents-base/requirements.txt 2>&1 | tail -40"
```

Note: `torch`/`torchvision` are already installed from Step 2 — if `requirements.txt` pins a specific CUDA build that conflicts, let the already-installed aarch64 build win (edit the installed requirements list on the fly with `pip install -r requirements.txt --no-deps` for the torch lines if pip tries to downgrade/replace them).

---

### Task 3: Transfer the dataset and smoke-test a real training step

**Files:** none (Isambard-side)

**Interfaces:**
- Consumes: the venv from Task 2.
- Produces: `~/work/hivemind-agents-base/data/{PlantVillage,PlantDoc}` on `b6dj`, and one confirmed real training step run, proving the bootstrap is usable before any agent relies on it.

- [ ] **Step 1: Transfer the dataset from the working b36bl checkout**

`b6dj` and `b36bl` are separate Isambard projects with separate filesystems — transfer via this local machine as the relay (both ends only need standard scp, no direct b36bl→b6dj path is assumed to exist).

```powershell
scp -F "$env:USERPROFILE\.ssh\config_clifton" -r b36bl.macs3.isambard:~/work/crop-mesh-detector/data/PlantVillage "C:\Users\yk25302\AppData\Local\Temp\claude\C--Users-yk25302-downloads-Seai-hivemind\cc185b6b-b0b0-4e8e-ad72-255e8bfc9a82\scratchpad\PlantVillage_transfer"
scp -F "$env:USERPROFILE\.ssh\config_clifton" -r b36bl.macs3.isambard:~/work/crop-mesh-detector/data/PlantDoc "C:\Users\yk25302\AppData\Local\Temp\claude\C--Users-yk25302-downloads-Seai-hivemind\cc185b6b-b0b0-4e8e-ad72-255e8bfc9a82\scratchpad\PlantDoc_transfer"
```

If this dataset is large enough to make the b36bl→local→b6dj double-hop impractical (check size first with `ssh b36bl.macs3.isambard "du -sh ~/work/crop-mesh-detector/data"`), re-run `scripts/download_plantvillage.py` and the PlantDoc download step directly on `b6dj` instead of transferring, using the same commands the original `b36bl` setup used (check `README.md`'s Setup section for the exact commands).

- [ ] **Step 2: Push the dataset to b6dj**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "mkdir -p ~/work/hivemind-agents-base/data"
scp -F "$env:USERPROFILE\.ssh\config_clifton" -r "C:\Users\yk25302\AppData\Local\Temp\claude\C--Users-yk25302-downloads-Seai-hivemind\cc185b6b-b0b0-4e8e-ad72-255e8bfc9a82\scratchpad\PlantVillage_transfer" b6dj.aip2.isambard:~/work/hivemind-agents-base/data/PlantVillage
scp -F "$env:USERPROFILE\.ssh\config_clifton" -r "C:\Users\yk25302\AppData\Local\Temp\claude\C--Users-yk25302-downloads-Seai-hivemind\cc185b6b-b0b0-4e8e-ad72-255e8bfc9a82\scratchpad\PlantDoc_transfer" b6dj.aip2.isambard:~/work/hivemind-agents-base/data/PlantDoc
```

- [ ] **Step 3: Copy the current source tree (same files already transferred for the D-PSGD run)**

```powershell
scp -F "$env:USERPROFILE\.ssh\config_clifton" -r "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\src" "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\scripts" "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\tests" "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\config.yaml" b6dj.aip2.isambard:~/work/hivemind-agents-base/
```

- [ ] **Step 4: Run a real, minimal training step as a smoke test**

```powershell
ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "cd ~/work/hivemind-agents-base && srun --account=brics.b6dj --partition=workq --gres=gpu:1 --time=00:15:00 bash -c 'source .venv/bin/activate && python -m src.train_local --config config.yaml --arch mobilevit_xxs --fresh' 2>&1 | tail -40"
```

Expected: the same per-node `local pair_accuracy=...` output style seen on the `b36bl` run — proves the full stack (data loading, model build, GPU training, energy tracking) works end to end on this new cluster before any agent's budget is spent on it.

- [ ] **Step 5: Record the outcome**

Note in the plan's tracking (or a short `~/work/hivemind-agents-base/BOOTSTRAP_NOTES.md` on `b6dj`) which install path from Task 2 Step 2 actually worked, and confirm Step 4 succeeded. This is what Tasks 5–7 point their agents at.

---

### Task 4: Create isolated per-agent directories

**Files:** none (Isambard-side)

**Interfaces:**
- Consumes: the bootstrapped `~/work/hivemind-agents-base` from Tasks 1–3.
- Produces: three independent Isambard working directories, one per agent, each with its own copy of the venv-pointing setup and data symlink, so agents cannot collide.

- [ ] **Step 1: Create the three directories, each with its own source copy and shared-venv/data symlinks**

```powershell
foreach ($dir in @("crop-mesh-agentA-comm","crop-mesh-agentB-personalization","crop-mesh-agentC-edge")) {
  ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "mkdir -p ~/work/$dir && cp -r ~/work/hivemind-agents-base/src ~/work/hivemind-agents-base/scripts ~/work/hivemind-agents-base/tests ~/work/hivemind-agents-base/config.yaml ~/work/$dir/ && ln -s ~/work/hivemind-agents-base/data ~/work/$dir/data && ln -s ~/work/hivemind-agents-base/.venv ~/work/$dir/.venv"
}
```

- [ ] **Step 2: Verify all three are independently usable**

```powershell
foreach ($dir in @("crop-mesh-agentA-comm","crop-mesh-agentB-personalization","crop-mesh-agentC-edge")) {
  ssh -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard "ls ~/work/$dir/src/train_dpsgd.py ~/work/$dir/data/PlantVillage ~/work/$dir/.venv/bin/python 2>&1"
}
```

Expected: no "No such file" errors for any of the three.

---

### Task 5: Dispatch Agent A (communication)

**Files:** none locally beyond the Agent tool call itself; the agent will create files inside its own worktree and `~/work/crop-mesh-agentA-comm` on `b6dj`.

**Interfaces:**
- Consumes: the bootstrapped `b6dj` environment (Task 4), the spec (`docs/superpowers/specs/2026-08-30-multi-agent-innovation-search-design.md`), and this project's real measured baselines (mesh/FedAvg/D-PSGD bytes tables in spec §4.1).
- Produces: a report file (path of the agent's choosing, but must be named and pointed out explicitly in its final message), plus any code/results backing it, inside `~/work/crop-mesh-agentA-comm` on `b6dj`.

- [ ] **Step 1: Call the Agent tool with `subagent_type: "fork"`, `isolation: "worktree"`, using this exact prompt**

```
You are Agent A in the HiveMind multi-agent innovation search. Read
docs/superpowers/specs/2026-08-30-multi-agent-innovation-search-design.md
in the crop-mesh-detector repo (you have full context from this
conversation already, but treat that spec file as the authoritative,
current version of your mandate).

Your dimension: communication (§4.1 of the spec). Your baselines to
beat are in that section's table. Your cross-disciplinary seeds are
radio astronomy interferometry and epidemiology — engage with both
seriously before narrowing to a mechanism; formalize any analogy
rigorously or explicitly discard it and fall back to first-principles
FL reasoning.

Follow the common protocol in spec §5 exactly: reason before building,
implement by reusing this project's existing infra (Node,
aggregation.py, ComputeEnergyTracker — follow the pattern
src/train_dpsgd.py already set), add a smoke test on synthetic data
before any real run, then validate at full scale (3 architectures,
N=6, 8 rounds, fresh stage-1 warm start) on b6dj.aip2.isambard using
~/work/crop-mesh-agentA-comm (account brics.b6dj, partition workq,
--gres=gpu:1; fall back to b36bl.macs3.isambard, account brics.b36bl,
partition ampere, if b6dj proves unusable for your workload). If your
result looks like it beats both FedAvg and D-PSGD outright on bytes
exchanged, re-run it across 3 seeds before calling it a confirmed win
rather than preliminary.

Hard rules: never touch ~/work/crop-mesh-detector on either cluster,
never git push to github.com/karboon1008/crop-mesh-detector, never
modify anything already reported in the research document.

Write your report using the exact template in spec §7 (Mechanism,
Hypothesis, Results table, Verdict, Honest limitations, Suggested next
step) to a file inside your isolated worktree, and also leave a copy
on b6dj under ~/work/crop-mesh-agentA-comm/REPORT.md. Report
regardless of outcome — a rigorous negative result, clearly explained,
is a valid and valuable deliverable. In your final message to me,
state the verdict in one sentence and the exact path to your report.
```

- [ ] **Step 2: Note the returned agent name/id for later review in Task 8**

---

### Task 6: Dispatch Agent B (personalization)

**Files:** none locally beyond the Agent tool call itself; the agent will create files inside its own worktree and `~/work/crop-mesh-agentB-personalization` on `b6dj`.

**Interfaces:**
- Consumes: same as Task 5, but spec §4.2 for mandate/baselines/seeds.
- Produces: a report file inside `~/work/crop-mesh-agentB-personalization` on `b6dj`, following the same §7 template.

- [ ] **Step 1: Call the Agent tool with `subagent_type: "fork"`, `isolation: "worktree"`, using this exact prompt**

```
You are Agent B in the HiveMind multi-agent innovation search. Read
docs/superpowers/specs/2026-08-30-multi-agent-innovation-search-design.md
in the crop-mesh-detector repo (you have full context from this
conversation already, but treat that spec file as the authoritative,
current version of your mandate).

Your dimension: ML performance via personalization (§4.2 of the
spec). Your target is per-node accuracy (each node's own local
distribution, not global-test macro) beating both FedAvg's and
D-PSGD's per-node results (outputs_dpsgd_baseline/{fedavg,dpsgd}_results_<arch>.json,
step1_to_step2_gain.macro_gain and per_node_scores, from the crop-mesh-detector
local checkout). The centralized-pooled-data baseline does not exist yet
in this project — you will need to produce it yourself (all 6 nodes'
data pooled into one training run, no partitioning) as the ceiling
reference, and report honestly whether your mechanism clears it
per-node, even though clearing it is a stretch goal, not the primary
bar. Your cross-disciplinary seeds are immunology (innate vs. adaptive
immunity, antibody diversity, immune memory) and finance (portfolio
theory's risk-adjusted diversification, and federated fraud-detection
consortiums in banking) — engage with both seriously before narrowing
to a mechanism; formalize any analogy rigorously or explicitly discard
it and fall back to first-principles FL reasoning.

Follow the common protocol in spec §5 exactly: reason before building,
implement by reusing this project's existing infra (Node,
aggregation.py, ComputeEnergyTracker — follow the pattern
src/train_dpsgd.py already set), add a smoke test on synthetic data
before any real run, then validate at full scale (3 architectures,
N=6, 8 rounds, fresh stage-1 warm start) on b6dj.aip2.isambard using
~/work/crop-mesh-agentB-personalization (account brics.b6dj, partition
workq, --gres=gpu:1; fall back to b36bl.macs3.isambard, account
brics.b36bl, partition ampere, if b6dj proves unusable for your
workload). If your result looks like it beats both FedAvg and D-PSGD
outright on per-node accuracy, re-run it across 3 seeds before calling
it a confirmed win rather than preliminary.

Hard rules: never touch ~/work/crop-mesh-detector on either cluster,
never git push to github.com/karboon1008/crop-mesh-detector, never
modify anything already reported in the research document.

Write your report using the exact template in spec §7 (Mechanism,
Hypothesis, Results table, Verdict, Honest limitations, Suggested next
step) to a file inside your isolated worktree, and also leave a copy
on b6dj under ~/work/crop-mesh-agentB-personalization/REPORT.md.
Report regardless of outcome — a rigorous negative result, clearly
explained, is a valid and valuable deliverable. In your final message
to me, state the verdict in one sentence and the exact path to your
report.
```

- [ ] **Step 2: Note the returned agent name/id for later review in Task 8**

---

### Task 7: Dispatch Agent C (edge footprint)

**Files:** none locally beyond the Agent tool call itself; the agent will create files inside its own worktree and `~/work/crop-mesh-agentC-edge` on `b6dj`.

**Interfaces:**
- Consumes: same as Task 5, but spec §4.3 for mandate/baselines/seeds.
- Produces: a report file inside `~/work/crop-mesh-agentC-edge` on `b6dj`, following the same §7 template.

- [ ] **Step 1: Call the Agent tool with `subagent_type: "fork"`, `isolation: "worktree"`, using this exact prompt**

```
You are Agent C in the HiveMind multi-agent innovation search. Read
docs/superpowers/specs/2026-08-30-multi-agent-innovation-search-design.md
in the crop-mesh-detector repo (you have full context from this
conversation already, but treat that spec file as the authoritative,
current version of your mandate).

Your dimension: edge footprint — memory, compute, and time (§4.3 of
the spec). This is explicitly the hardest of the three: the model
itself is identical size across mesh/FedAvg/D-PSGD (same architecture,
same parameter count), so any real advantage must come from something
structural — what's held in memory during a round, forward/backward
pass count, a fundamentally different computation graph — not the
model. A credible, well-argued negative result ("no structural
advantage exists here, and here is why") is an acceptable and valued
outcome for you specifically, more so than for Agents A or B — do not
force a fabricated win. Your target, if one exists, is peak memory,
wall-clock duration, and compute energy per round beating both
FedAvg's and D-PSGD's measured figures (emissions.csv, run_state.json
in outputs_dpsgd_baseline/ in the crop-mesh-detector local checkout).
Your cross-disciplinary seeds are neuroscience (synaptic pruning —
structure and sparsity emerging from training dynamics rather than
applied post-hoc; treat lottery-ticket-style literature as a baseline
to differ from, not a seed) and thermodynamics (entropy/free-energy
minimization as a formal objective for an energy-cost term, rather
than treating energy as a metric measured after the fact) — engage
with both seriously before narrowing to a mechanism; formalize any
analogy rigorously or explicitly discard it and fall back to
first-principles reasoning.

Follow the common protocol in spec §5 exactly: reason before building,
implement by reusing this project's existing infra (Node,
aggregation.py, ComputeEnergyTracker — follow the pattern
src/train_dpsgd.py already set), add a smoke test on synthetic data
before any real run, then validate at full scale (3 architectures,
N=6, 8 rounds, fresh stage-1 warm start) on b6dj.aip2.isambard using
~/work/crop-mesh-agentC-edge (account brics.b6dj, partition workq,
--gres=gpu:1; fall back to b36bl.macs3.isambard, account brics.b36bl,
partition ampere, if b6dj proves unusable for your workload). If your
result looks like it beats both FedAvg and D-PSGD outright on any of
memory/compute/time, re-run it across 3 seeds before calling it a
confirmed win rather than preliminary.

Hard rules: never touch ~/work/crop-mesh-detector on either cluster,
never git push to github.com/karboon1008/crop-mesh-detector, never
modify anything already reported in the research document.

Write your report using the exact template in spec §7 (Mechanism,
Hypothesis, Results table, Verdict, Honest limitations, Suggested next
step) to a file inside your isolated worktree, and also leave a copy
on b6dj under ~/work/crop-mesh-agentC-edge/REPORT.md. Report
regardless of outcome — a rigorous negative result, clearly explained,
is a valid and valuable deliverable. In your final message to me,
state the verdict in one sentence and the exact path to your report.
```

- [ ] **Step 2: Note the returned agent name/id for later review in Task 8**

---

### Task 8: Joint review checkpoint

**Files:** none — this is a review/decision task, not a code task.

**Interfaces:**
- Consumes: all three agents' `REPORT.md` files and final-message verdicts.
- Produces: a joint decision, made with the user, on each agent's verdict and any follow-up (integration pass, documented exploration, or discard).

- [ ] **Step 1: Read all three reports (fetch via scp if only on `b6dj`, not yet local)**

```powershell
foreach ($dir in @("crop-mesh-agentA-comm","crop-mesh-agentB-personalization","crop-mesh-agentC-edge")) {
  scp -F "$env:USERPROFILE\.ssh\config_clifton" b6dj.aip2.isambard:~/work/$dir/REPORT.md "C:\Users\yk25302\downloads\Seai_hivemind\crop-mesh-detector\outputs_dpsgd_baseline\$($dir)_REPORT.md"
}
```

- [ ] **Step 2: Present a summary comparison (all three verdicts, headline numbers vs. FedAvg/D-PSGD) to the user, and jointly decide next steps per §8 of the spec**

No code changes happen in this step without explicit user sign-off — this is the checkpoint the spec's §8 requires before anything touches the real submission.

---

## Self-Review Notes

- **Spec coverage:** §4.1/4.2/4.3 (agent mandates) → Tasks 5/6/7 prompts. §5 (common protocol) → embedded verbatim into each prompt. §6 (execution mechanism, bootstrap) → Tasks 1–4. §7 (report template) → embedded verbatim into each prompt. §8 (joint review) → Task 8. §9/§10 (risks, success criteria) are context carried by the spec itself, not separate tasks — correctly so, they're not actions.
- **Placeholder scan:** the PyTorch install step (Task 2, Step 2) has an explicit fallback path rather than a single unverified command, because aarch64+CUDA wheel availability cannot be confirmed without live verification — this is a real engineering contingency, not a "TBD".
- **Type/interface consistency:** all three agent directories, accounts, and partitions are named identically across Tasks 4, 5, 6, 7 (`brics.b6dj`/`workq`/`--gres=gpu:1`, fallback `brics.b36bl`/`ampere`).
