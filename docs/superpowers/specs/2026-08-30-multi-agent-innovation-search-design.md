# Multi-Agent Innovation Search: Design

**Date:** 2026-08-30
**Status:** Approved for spec review; not yet implemented
**Owner:** Wong E Chern (HiveMind team), planned with Claude

## 1. Motivation

HiveMind's mesh (§3 of the research document) already has a clear, defensible position against the two most relevant baselines:

- **vs. FedAvg** (client-server, full weights): 27.8–51.1× fewer bytes exchanged, at the cost of lower raw accuracy and 3.0–3.7× more compute energy than local-only (§6.6 of the research doc).
- **vs. D-PSGD** (Lian et al., 2017 — decentralized, full weights, peer-to-peer): confirmed via a real Isambard run (`crop-mesh-detector-dpsgd/`, 2026-08-26) that D-PSGD converges to essentially the same accuracy as FedAvg (as theory predicts under full connectivity and identical mixing weights) while costing 2.5× FedAvg's bytes and 70–128× our mesh's bytes.

Both of these are **trade-off** wins: cheaper *at the cost of* something else (accuracy, compute). No dimension exists today where HiveMind beats *both* FedAvg and D-PSGD outright, with nothing given up in exchange — and no result yet approaches the centralized-pooled-data ceiling on any measure.

The working hypothesis, from discussion, is that a recombination of already-published FL techniques (which is what HiveMind currently is — Prototypical Networks + Hinton distillation + Blanchard Byzantine-robust aggregation + FedRS masking, per §12 of the research doc) is unlikely to produce an outright win of this kind. An outright win plausibly requires a mechanism that isn't a straightforward combination of existing federated-learning literature.

## 2. Goal

Find, implement, and rigorously validate a mechanism — for at least one of three dimensions — that beats **both** FedAvg and D-PSGD outright (not a trade-off), using real measured data, not formulas or projections. Stretch goal, principled path named explicitly (§4.2): beat the centralized-pooled-data ceiling on **per-node** performance under non-IID data.

## 3. Non-goals

- Not required to beat every baseline on every axis — a genuine outright win on *one* dimension, honestly reported, is the target.
- Not required to succeed. A rigorous negative result (a mechanism tried, validated, and shown not to win, with a clear explanation why) is an acceptable and valuable outcome per the deliverable rule (§7).
- Not touching the actual submitted repository (`github.com/karboon1008/crop-mesh-detector`) or its `main` branch. Everything here happens in isolated local worktrees and isolated Isambard directories.
- Not a redesign of HiveMind's existing, already-reported mesh. That system and its results (§1–§14 of the research document) stand as-is; this is a separate, parallel exploration.

## 4. The three agents

Each agent works independently, in parallel, on one dimension. No shared state between them during exploration.

### 4.1 Agent A — Communication

**Mandate:** find a mechanism that reduces total bytes exchanged below what our current mesh already achieves, while still beating FedAvg and D-PSGD's bytes by construction (trivially true if it beats our mesh, since our mesh already beats both) — the actual bar is beating our *own* mesh's bytes without giving back accuracy that drops below both FedAvg's and D-PSGD's floor.

**Baselines to beat** (8-round sweep, N=6, real numbers — see §6 for full provenance):

| | Our mesh (bytes) | FedAvg (bytes) | D-PSGD (bytes) |
|---|---|---|---|
| mobilenet_v3_small | 21,230,400 | 590,728,704 | 1,476,821,760 |
| efficientnet_lite0 | 25,531,200 | 1,304,305,152 | 3,260,762,880 |
| mobilevit_xxs | 9,403,200 | 367,658,496 | 919,146,240 |

**Cross-disciplinary seeds** (starting points to reason from, not a checklist — formalize rigorously or discard, don't decorate):
- *Radio astronomy interferometry*: many low-SNR telescopes combine only correlated summaries, never raw signal, into one high-SNR image. Structurally close to our own prototype/logit exchange — is there a more information-theoretically efficient summary statistic than a class-mean embedding?
- *Epidemiology*: disease/rumor-spread models give rigorous convergence bounds for information propagating over partial-connectivity graphs. This is the theoretical grounding the gossip-mesh idea lacked (it was deleted from the presentation figure for exactly this reason, 2026-08-28) — could finally validate a partial-connectivity topology properly, with an accuracy-cost measurement, not an assumption.

### 4.2 Agent B — ML performance via personalization

**Mandate:** beat FedAvg's and D-PSGD's **per-node** accuracy (not global-test macro — each node's own local distribution), and report honestly whether it also clears the centralized-pooled-data ceiling per-node. This is the one principled path to a "beats centralized" claim: a personalized model can beat a single pooled global model on an individual node's own distribution under non-IID data, even though it structurally cannot beat it on the global average.

**Baselines to beat**: per-node local pair_accuracy after collaboration, for FedAvg and D-PSGD (`outputs_dpsgd_baseline/{fedavg,dpsgd}_results_<arch>.json`, `step1_to_step2_gain.macro_gain` and `per_node_scores`). A centralized-pooled baseline does not yet exist and would need to be produced (all 6 nodes' data pooled into one training run, no partitioning) as the ceiling reference.

**Cross-disciplinary seeds:**
- *Immunology*: innate vs. adaptive immunity — a shared, general first line of defense plus a personalized, locally-adapted memory response. A different lens on "shared backbone + personalized head" than standard meta-learning/personalization-layer literature.
- *Finance*: portfolio theory's risk-adjusted diversification, and federated fraud-detection consortiums in banking (a real, existing industrial precedent for federated learning among competing parties) — could inform a personalized weighting/trust scheme beyond the current flat volume-weighting.

### 4.3 Agent C — Edge footprint (memory / compute / time)

**Mandate:** propose and validate a genuinely new mechanism or architecture optimized for on-device footprint — not measure the existing mesh's footprint. Flagged as the hardest of the three: model size is currently identical across mesh/FedAvg/D-PSGD (same architecture, same parameter count), so any real advantage must come from something structural — what's held in memory during a round, forward/backward pass count, a fundamentally different computation graph — not the model itself. A credible negative result ("no structural advantage exists here, and here is why") is an acceptable outcome for this agent specifically, more so than for A or B.

**Baselines to beat**: peak memory, wall-clock duration, and compute energy per round (`emissions.csv`, `run_state.json` — same measurement infra as the rest of this project), for FedAvg and D-PSGD.

**Cross-disciplinary seeds:**
- *Neuroscience*: synaptic pruning — structure and sparsity emerging from training dynamics, rather than applied post-hoc (as opposed to the lottery-ticket-style literature, which is FL-adjacent and should be treated as a baseline to differ from, not a seed).
- *Thermodynamics*: entropy / free-energy minimization as a formal objective for an energy-cost term, rather than treating energy as a metric measured after the fact.

## 5. Common protocol (all three agents)

1. **Reason before building.** Engage seriously with both assigned fields before narrowing to a mechanism. If neither leads anywhere rigorous, say so explicitly and fall back to first-principles FL reasoning — report that dead end honestly rather than silently abandoning it.
2. **Implement, reusing existing infrastructure.** `src/federated/node.py` (`Node`), `src/federated/aggregation.py`, `src/energy/tracker.py` (`ComputeEnergyTracker`), following the pattern `src/train_dpsgd.py` already established (reuse `federated_average`, don't reimplement; mirror `train_fedavg.py`'s CLI/output-schema conventions). Add a smoke test on synthetic data (pattern: `tests/test_dpsgd_smoke.py`) before any real run.
3. **Validate at full scale.** 3 architectures (mobilenet_v3_small, efficientnet_lite0, mobilevit_xxs), N=6, 8 rounds, fresh stage-1 warm start — matching the protocol already used for the FedAvg/D-PSGD comparison (`crop-mesh-detector-dpsgd/run_dpsgd_baseline.sh` as the template).
4. **Multi-seed for apparent wins.** If a result looks like it beats both baselines outright on the target dimension, re-run it across 3 seeds before reporting it as a win. A single-seed result stays labeled "preliminary" in the report, not a claim.
5. **Report regardless of outcome** (§7).

## 6. Execution mechanism

- **3 parallel `Agent` tool calls**, `subagent_type: "fork"` (inherits this entire project's context automatically — no separate briefing document needed), `isolation: "worktree"` (each fork gets an isolated local git worktree, preventing collisions between agents' local scratch files).
- **Isambard compute**: `b6dj.aip2.isambard` (NVIDIA GH200, ARM64, account `brics.b6dj`, partition `workq`, `gres=gpu:4`/node — confirmed live via a real `srun` on 2026-08-30) is the primary resource, given the generous node-hour allocation. `b36bl.macs3.isambard` (the existing, proven x86_64/A100 environment, already has the dataset and a working venv) is the fallback if `b6dj` setup proves unreliable for a given agent.
- **Shared `b6dj` bootstrap, done once before any agent launches** (by the planner, not redundantly by each agent): fresh Python/PyTorch environment for `aarch64`, PlantVillage/PlantDoc dataset transferred, smoke-tested with a trivial training step. Each agent gets its own isolated working directory on `b6dj` (e.g. `~/work/crop-mesh-detector-agentA-comm`), cloned/copied from this bootstrapped base, mirroring the pattern already used for `crop-mesh-detector-dpsgd` on `b36bl`.
- **Hard rule, all agents**: never touch `~/work/crop-mesh-detector` (the real checkout) on either cluster, never `git push` to `github.com/karboon1008/crop-mesh-detector`, never modify anything already reported in the research document without flagging it as a proposed change for review.

## 7. Report template (per agent, regardless of outcome)

1. **Mechanism** — what was built and why, including which cross-disciplinary seed (if any) led there, and how any analogy was formalized rigorously.
2. **Hypothesis** — what beating FedAvg/D-PSGD on this dimension would require, stated before results were known.
3. **Results table** — real numbers, this mechanism vs. our existing mesh vs. real FedAvg vs. real D-PSGD, same format as §6.6 of the research document.
4. **Verdict** — outright win / partial (trade-off) / no win, stated plainly.
5. **Honest limitations** — what wasn't tested, what could invalidate the result, single-seed vs. multi-seed status.
6. **Suggested next step** — what a follow-up would need to do.

## 8. After the agents report back

A joint review (planner + user) of all three reports before anything is treated as a real result:

- Decide, per report, whether the verdict (win/partial/no-win) holds up to scrutiny.
- For any outright win: decide whether it's worth a follow-up integration pass into the main mesh, or stays documented as a disclosed exploration.
- Nothing from this process gets folded into the actual submission (research document, code repo, presentation) without that explicit review step.

## 9. Risks

- **Isambard is a shared resource** even with generous allocation on `b6dj` — 3 concurrent full-scale runs, plus multi-seed re-runs for any apparent win, is meaningfully more compute than any single run so far in this project.
- **`b6dj` is unproven for this workload** — ARM64 PyTorch/CUDA compatibility for these specific model architectures (timm backbones) has not yet been confirmed; the bootstrap step (§6) exists specifically to surface this before any agent's budget is spent on it.
- **Worst case is wasted compute and time, not a broken deliverable** — nothing here touches the submitted repository or the already-submitted preliminary-round package.

## 10. Success criteria for this exploration

- Minimum: three honest, structured reports, each either a validated win, a validated partial result, or a rigorous negative result with a clear explanation — no agent stops silently or reports without real data behind it.
- Stretch: at least one dimension with a real, multi-seed-confirmed outright win over both FedAvg and D-PSGD.
- Aspirational: a personalization result (Agent B) that also clears the centralized-pooled ceiling per-node, which would be the strongest possible answer to the Challenge's own central question (Official Problem Statement §1) and the most publication-worthy outcome of the three.
