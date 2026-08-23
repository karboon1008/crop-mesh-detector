# MEMORY — Apple Dirichlet-Mesh Pipeline: Full Run Log

Date: 2026-08-22
Scope: what was already in the repo, what was found/fixed, and the exact
commands used to run the Apple Dirichlet-mesh validation pipeline
end-to-end (Stage 1 -> Stage 2), on Windows, using the existing `.venv`.
Companion results doc: `docs/apple_disease_knowledge_transfer_results.md`.

This file is a **runbook / journal**, not a design spec — it exists so a
future session (human or agent) can either (a) reproduce this exact run,
or (b) understand what was already checked off before spending time
re-verifying it.

---

## 1. Starting state (nothing needed to be built from scratch)

The Apple pipeline scripts already existed in the repo as **untracked**
files (not yet committed), written as direct mirrors of the Tomato
Dirichlet-mesh pipeline:

- `src/validation/apple_mesh_dataset.py` — merges PlantVillage + PlantDoc +
  PlantWild v1/v2 Apple images into one canonical label space
  (`Apple_scab`, `Black_rot`, `Cedar_apple_rust`, `healthy`), carves a
  global dedup-aware test split, then Dirichlet-partitions (alpha=0.3) the
  rest across 3 nodes.
- `src/validation/run_apple_pipeline.py` — Stage 1: per-node train ->
  export -> evaluate CLI (mirrors `run_tomato_pipeline.py`).
- `src/validation/run_apple_knowledge_transfer.py` — Stage 2:
  knowledge-transfer rounds + local-only-control arm on top of Stage 1's
  checkpoints (mirrors `run_tomato_knowledge_transfer.py`; reuses
  `run_knowledge_transfer.run_kt_round` for the actual aggregation math —
  only the data split differs from Tomato/Corn, not the algorithm).
- `config.yaml`'s `apple_mesh:` block (num_nodes=3, dirichlet_alpha=0.3,
  test_fraction=0.20, dedup_threshold=5, dedup_max_group_size=25, rounds=2,
  output_dir=`outputs/validation/apple_mesh`).
- Apple loader functions already added to `src/data/plantdoc.py`
  (`load_plantdoc_apple_paths`) and `src/data/plantwild.py`
  (`load_plantwild_v1_apple_paths`, `load_plantwild_v2_apple_paths`).

So the actual task was: **verify it works, run it for real (Stage 1 at 20
epochs/node, Stage 2 at 2 rounds x 10 distill-epochs/round), fix whatever
breaks, and produce a results doc** — not write the pipeline itself.

Source data confirmed already present locally (no downloads needed):
`data/PlantVillage/Apple___*`, `data/PlantDoc/{train,test}/Apple*`,
`data/PlantWild/plantwild/plantwild/images/Apple_*`,
`data/PlantWild/plantwild_v2/plantwild_v2/Apple_*`.

## 2. Pre-flight sanity check (before committing to a multi-hour run)

Ran a standalone data-loading smoke test first, since Stage 1/2 together
take hours on CPU and any data-loading bug should fail fast, not after
20+ minutes of training:

```powershell
.venv\Scripts\python.exe -c "
from src.config import Config
from src.validation.apple_mesh_dataset import prepare_apple_mesh_data
cfg = Config.load()
d = prepare_apple_mesh_data(cfg)
print(len(d.train_base), len(d.test_idx), len(d.probe_idx))
for nid, v in sorted(d.per_node.items()):
    print(nid, len(v['train_idx']))
"
```

This is exactly what surfaced Bug #1 below on the first attempt.

## 3. Bugs found and fixed

### Bug #1 — Windows `MAX_PATH` (260-char) crash in data loading

**Symptom:** the sanity check above raised
`FileNotFoundError: [Errno 2] No such file or directory:
'data\PlantDoc\train\Apple leaf\apple-tree-branch-...-928225.jpg'`
even though the file genuinely exists (confirmed with `ls`).

**Root cause:** one PlantDoc Apple stock-photo has an unusually long
filename. This checkout's path
(`C:\myCamb\crop-mesh-detector\crop-mesh-detector\...`, itself already ~47
chars, doubled up because the outer and inner folders share the same
name) plus `data\PlantDoc\train\Apple leaf\<237-char filename>.jpg` adds up
to **285 characters**, past Windows' classic 260-char `MAX_PATH` limit.
Confirmed via `reg.exe query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" //v LongPathsEnabled`
→ `0x0` (long-path support disabled at the OS level, the Windows default).

**Fix (code, not a system/registry change):** added a small
`_win_long_path()` helper that prepends the `\\?\` extended-length-path
prefix to absolute paths on Windows before calling `PIL.Image.open()` —
this bypasses `MAX_PATH` without needing admin rights or a registry edit
(which would have required elevation and would only fix it for *this*
machine, not any other checkout depth). No-op on non-Windows / already
UNC-prefixed paths.

Applied at every place that opens a raw image path from the Apple merged
dataset:
- `src/validation/apple_mesh_dataset.py`: `compute_merged_image_hashes()`
  and `AppleMergedDataset.__getitem__()`.
- `src/validation/evaluate_onnx.py`: `run_evaluation()` — this one opens
  test-set images directly by path (bypassing the dataset's
  `__getitem__`), so it needed the same fix independently, or Stage 1's
  `evaluate` stage / Stage 2's per-round eval would have hit the same
  crash the moment that specific image landed in a test/eval split.

Not applied to `tomato_mesh_dataset.py` / `node1_dataset.py` / `predict.py`
(same `Image.open()` pattern exists there too, but none of them hit this
in practice for this task's data — left alone to keep the change scoped).

### Bug #2 — Stage 2 disclosure metadata ignored the `--distill-epochs` CLI override

**Symptom:** after a full ~1h42m Stage 2 run with
`--distill-epochs 10`, the generated `knowledge_transfer_summary.json`
said `"epochs_per_round": 1` / `"distill_epochs_per_round": 1` — wrong,
and misleading for an Appendix-A.1-style disclosure artifact.

**Root cause:** `build_apple_knowledge_transfer_summary()` re-derived
those two fields from `cfg.get("training.distill_epochs_per_round", 1)`
(the config *default*) instead of using the actual `distill_epochs` value
`main()` had already resolved from `args.distill_epochs or cfg default`.
The real training loop (`run_apple_round_with_io` → `run_kt_round`) was
never affected — it correctly used 10 epochs/round the whole time
(confirmed independently by round wall-clock time: ~50 min/round is
consistent with 10 local-supervised epochs over each node's full shard,
not 1). Only the post-hoc summary metadata was wrong.

**Fix:** added `distill_epochs: int` as an explicit parameter to
`build_apple_knowledge_transfer_summary()` and threaded the real value
into it from `main()`, instead of re-deriving from `cfg`.

**Recovery (no re-training needed):** wrote a one-off script that
reloads `cfg` + recomputes `prepare_apple_mesh_data(cfg)`, reads the
already-persisted `round_0_baseline.json` / `round_1/round_summary.json`
/ `round_2/round_summary.json`, and calls the now-fixed
`build_apple_knowledge_transfer_summary(..., distill_epochs=10)` to
regenerate just `knowledge_transfer_summary.json` — the expensive part
(training) was already done and correct; only the summary-building step
needed to re-run, which takes seconds. The scratch script was deleted
after use (`_tmp_regen_apple_kt_summary.py` — user has since removed it;
it is not part of the permanent codebase).

The same latent bug exists in `run_tomato_knowledge_transfer.py` /
`run_knowledge_transfer.py`'s equivalent function (same
re-derive-from-cfg pattern) but was **not** fixed there — it's
inconsequential for any Tomato run that didn't pass a
`--distill-epochs` override, and fixing/regenerating Tomato's published
results doc was out of scope for this task. Flagging here in case a
future Tomato re-run with a non-default `--distill-epochs` is planned.

## 4. Exact commands run, in order

All from the repo root, using the existing `.venv` (no new dependencies
needed — `torch==2.13.0+cpu`, CPU-only, `torch.cuda.is_available()` is
`False` on this machine).

```powershell
# 1. Sanity check (see section 2) — run once, before Stage 1.

# 2. Stage 1 — per-node train -> export -> evaluate, 20 epochs/node
.venv\Scripts\python.exe -m src.validation.run_apple_pipeline --epochs 20 `
    > outputs\validation\apple_mesh\stage1_run.log 2>&1

# 3. Verify Stage 1 outputs before starting Stage 2 (must all exist):
#    outputs\validation\apple_mesh\classes.json
#    outputs\validation\apple_mesh\node_{0,1,2}_mobilenet_v3_small\{checkpoint.pt,model.onnx,manifest.json,report.json,training_log.json,classes.json}

# 4. Stage 2 — round-0 baseline + 2 knowledge-transfer rounds, 10 local epochs/round
.venv\Scripts\python.exe -m src.validation.run_apple_knowledge_transfer `
    --rounds 2 --distill-epochs 10 `
    > outputs\validation\apple_mesh\stage2_run.log 2>&1

# 5. Verify Stage 2 outputs (must all exist):
#    outputs\validation\apple_mesh\knowledge_transfer\round_0_baseline.json
#    outputs\validation\apple_mesh\knowledge_transfer\round_{1,2}\round_summary.json
#    outputs\validation\apple_mesh\knowledge_transfer\round_{1,2}\node_{0,1,2}\{checkpoint.pt,model.onnx,manifest.json,report.json,local_only_control\report.json}
#    outputs\validation\apple_mesh\knowledge_transfer\knowledge_transfer_summary.json

# 6. (one-off, after fixing Bug #2) Regenerate knowledge_transfer_summary.json
#    from already-computed round data, without re-running training — see
#    section 3, Bug #2 "Recovery". Script was deleted after use.
```

In this sandbox, background processes were launched with
`nohup ... > logfile 2>&1 &` (git-bash), and progress was polled by
periodically re-reading `stage{1,2}_run.log` and `find`-listing the
output tree — **not** with `sleep`+`get_output`, which was unreliable in
this environment (see section 6).

## 5. Timing observed (reference machine, CPU-only)

| Stage | Wall-clock | Notes |
|---|---|---|
| Stage 1 total | 37m 52s | node_0 (1,402 imgs) 12m38s, node_1 (723 imgs) 9m8s, node_2 (1,782 imgs) 15m54s |
| Stage 2 round-0 baseline | ~1m 9s | eval-only, 3 already-exported models |
| Stage 2 round 1 | 51m 29s | 3 collective distill (10 epochs + KD over 205-image probe set) + 3 control local_train (10 epochs) + 6 export/eval |
| Stage 2 round 2 | 49m 25s | same shape as round 1 |
| **End-to-end total** | **~2h 20m** | Stage 1 start to Stage 2 finish |

Use this as a planning reference for similarly-sized crops/configs on
this machine (roughly: Stage 1 minutes ≈ epochs × total-train-images /
~2200 images-per-minute; Stage 2 minutes/round ≈ distill_epochs × similar
rate, since KD-phase probe images are a small addition to each node's own
train-set pass).

## 6. Sandbox tooling notes (for whoever picks this up next)

- `sleep N && <cmd>` combined with `get_output`/blocking waits was
  **unreliable** in this sandbox for `N` beyond ~10s — calls would
  report "still running" long past when `N` seconds had genuinely
  elapsed, or would hang. Direct one-shot `exec` calls (no `sleep`)
  checking file timestamps/log contents worked reliably and instantly.
- For long (multi-hour) background training, delegating monitoring to a
  background `subagent_explore` with instructions to loop 80-150+
  read-only checks per invocation, then resuming it repeatedly, worked
  as a practical way to let real wall-clock time pass between check-ins
  without burning the main session's own tool-call budget on tight
  polling loops.
- The most reliable single "is it done yet" signal for this pipeline is
  file existence, not log content — both `run_apple_pipeline.py` and
  `run_apple_knowledge_transfer.py` print almost nothing to stdout
  mid-stage (one line per node/round, nothing per-epoch), so a long
  silent gap in the `.log` file is normal, not a hang. Watch for
  `report.json` (Stage 1, per node) / `round_summary.json` (Stage 2, per
  round) / `knowledge_transfer_summary.json` (Stage 2, final) instead.

## 7. Outputs produced

- `outputs/validation/apple_mesh/` — full Stage 1 + Stage 2 artifact
  tree (checkpoints, ONNX exports, manifests, reports, training/round
  logs, run logs). Gitignored (`outputs/*` in `.gitignore`), like all
  other crops' outputs — present on disk, not committed.
- `docs/apple_disease_knowledge_transfer_results.md` — full write-up
  (timing, data split, per-epoch training logs, round-by-round trend,
  per-class collaboration gain tables, macro/worst-node gain, energy,
  interpretation), same shape as `docs/tomato_disease_knowledge_transfer_results.md`
  and `docs/corn_disease_knowledge_transfer_results.md`.
- Code changes (all still uncommitted as of this writing):
  `src/validation/apple_mesh_dataset.py` (long-path fix),
  `src/validation/evaluate_onnx.py` (long-path fix),
  `src/validation/run_apple_knowledge_transfer.py` (disclosure-metadata fix).

## 8. Headline result (see the results doc for full detail)

Unlike the Tomato Dirichlet run (small positive macro collaboration
gain), this Apple run's macro `disease_accuracy` gain was ~flat/slightly
negative (-0.8 pts) and worst-node gain was exactly 0.0 at 2 rounds x 10
distill-epochs/round. The Dirichlet alpha=0.3 draw for Apple happened to
be more extreme than Tomato's (one node, node_1, landed 94.5% on a single
class with zero samples of another), which both arms converged to
identically — a property of this specific 3-node draw, not a general
verdict on the method for Apple. One clear per-class win: node_0's
`Black_rot` (1 training image) gained +10.9 pts vs. its local-only
control, showing cross-node signal can still rescue a near-zero-sample
class even when the macro picture is flat.

## 9. Follow-up run — alpha=0.7, lighter budget (2026-08-23)

After reviewing Run 1's headline result, the user asked for a second run
following this tuning recommendation (their words, in order of expected
impact): (1) fix the partition draw (higher alpha) before touching the
epoch/round schedule, (2) if keeping the exact partition, consider
trading round count for epoch depth or vice versa, (3) consider not
resetting Adam optimizer state between Stage-2 rounds (code-level, NOT
implemented — flagged for a future session), (4) don't over-index on one
run/seed. The user picked option (1): loosen `apple_mesh.dirichlet_alpha`
(config-only, no `--alpha` CLI flag) and use a lighter Stage 1/2 budget
(`--epochs 4`, `--rounds 4 --distill-epochs 4`), saving to a new output
dir so Run 1's results stay intact.

**Before running anything**, per the user's explicit request, the new
Dirichlet split was previewed in-memory (no training, no file writes) by
loading `Config`, overriding `cfg._data['apple_mesh']['dirichlet_alpha']`
directly in Python (bypassing `config.yaml` for the preview only), and
calling `prepare_apple_mesh_data(cfg)` to print
`partition_diagnostics` per node. This was done twice — alpha=0.5, then
alpha=0.7 — before the user picked 0.7, because **alpha=0.5 still left
node_1 with 0 `Black_rot` samples** (94.5% -> 75.5% dominance, an
improvement, but the structural zero-sample-class problem persisted).
Only alpha=0.7 gave every node >=1 sample of every class. `config.yaml`
was only edited to the final chosen value (0.7), never actually run at
0.5.

Exact commands for this run (output dir named `apple_mesh_a0.7` to match
the actual alpha used, per the user's explicit preference when the two
diverged from their original `apple_mesh_a0.5` request):

```powershell
# config.yaml: apple_mesh.dirichlet_alpha changed 0.3 -> 0.7 (comment explains why)

.venv\Scripts\python.exe -m src.validation.run_apple_pipeline --epochs 4 `
    --output-dir outputs\validation\apple_mesh_a0.7 > outputs\validation\apple_mesh_a0.7\stage1_run.log 2>&1

.venv\Scripts\python.exe -m src.validation.run_apple_knowledge_transfer `
    --rounds 4 --distill-epochs 4 --output-dir outputs\validation\apple_mesh_a0.7 `
    > outputs\validation\apple_mesh_a0.7\stage2_run.log 2>&1
```

Timing: Stage 1 6m25s, Stage 2 1h8m45s (round-0 baseline ~1m6s, then
~17-18min/round for rounds 1-2, ~15-16min/round for rounds 3-4) — total
~1h15m, much faster than Run 1's ~2h20m thanks to the lighter budget.
No bugs hit this time (both Run 1 fixes already in place); no code
changes were needed for this run.

**Result — the fix worked, the macro number did not improve:** alpha=0.7
eliminated the true zero-sample-class problem entirely (every node has
>=1 sample of every class; node_1's `Black_rot` accuracy moved off its
hard 0.0000 floor for the first time). But Run 2's macro
`disease_accuracy` gain (-2.4 pts) and worst-node gain (-10.7 pts) were
both *worse* than Run 1's (-0.8 pts / 0.0 pts). This is flagged in the
results doc as **confounded, not a clean verdict on alpha** — the user's
requested change bundled a higher alpha together with a much lighter
epoch/round budget (4 epochs/4x4 rounds vs. 20 epochs/2x10 rounds) in
the same run, so the regression could be the alpha, the budget (paying
the per-round Adam-reset penalty 4 times instead of 2, with less time
per round to recover each time), or both. The results doc's [Tuning
recommendation](apple_disease_knowledge_transfer_results.md#tuning-recommendation-for-the-next-run)
section spells out the follow-up needed to isolate which one:
re-run alpha=0.7 at Run 1's original 20-epoch/2x10 budget for a clean
matched-budget comparison.

Output: `outputs/validation/apple_mesh_a0.7/` (same artifact shape as
Run 1's `outputs/validation/apple_mesh/`).
`docs/apple_disease_knowledge_transfer_results.md` was restructured to
hold both runs side by side (Run 1 / Run 2 / Comparison / Tuning
recommendation sections), following the same pattern
`docs/tomato_disease_knowledge_transfer_results.md` already used for its
own two runs.
