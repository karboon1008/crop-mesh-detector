# Apple Knowledge-Transfer Diagnosis & Next-Run Tuning

Date: 2026-08-22
Companion to: `docs/apple_disease_knowledge_transfer_results.md` (the run being diagnosed),
`docs/tomato_disease_knowledge_transfer_results.md` (cross-crop comparison),
`docs/raspberry_pi_deployment.md` (deployment-target context).

**Update (2026-08-23) — the follow-up run happened, see
`docs/apple_disease_knowledge_transfer_results.md`'s "Run 2" section.**
The user picked recommendation #1 below (fix the partition draw) and, after
previewing both, chose **alpha=0.7** (not the 0.5 top pick below — 0.5 was
previewed and still left node_1 with 0 `Black_rot` samples, so 0.7 was
used instead) at the lighter `4 epochs x 4 rounds` budget from section 6.
**Section 1's root-cause diagnosis held up**: alpha=0.7 did eliminate the
true zero-sample-class collapse (node_1's `Black_rot` moved off its hard
0.0000 floor for the first time). **But the macro-gain prediction did
not pan out**: Run 2's macro gain (-2.4 pts) and worst-node gain (-10.7
pts) came out *worse* than the original alpha=0.3 run's (-0.8 pts / 0.0
pts), not better as the "mirrors Tomato's less-extreme draw" reasoning
in section 6 expected. The results doc's Run 2 interpretation section
argues this is most likely because alpha and the epoch/round budget were
changed together in the same run (recommendation #2's lighter schedule
was layered on top of #1's partition fix rather than tested separately),
so the regression can't be cleanly attributed to alpha alone — a matched-
budget re-run (alpha=0.7 at the original 20-epoch/2x10 schedule) is the
next step to isolate which variable actually drove the result.

**Update (2026-08-23, second follow-up) — a third run happened, against
this note's own §5 recommendation.** The user explicitly requested
excluding `Black_rot` (3-class problem: `Apple_scab`, `Cedar_apple_rust`,
`healthy`), kept alpha=0.7, went back to a 20-epoch Stage 1 (not Run 2's
4), and kept the 4x4 Stage-2 schedule. See `docs/apple_disease_knowledge_transfer_results.md`'s
"Run 3" section: this is the first Apple run with an unambiguous
net-positive macro gain (+4.4 pts) and worst-node gain (+4.9 pts) — but
two variables changed from Run 2 at once (epochs 4->20, `Black_rot`
dropped) and the isolated recommendation below (re-run alpha=0.7 at Run
1's exact 20-epoch/2x10 budget, keeping `Black_rot`) still has not been
done. Section 5's core argument — that removing `Black_rot` discards the
series' one unambiguous positive class-level result up to that point —
is still valid; Run 3 supersedes it with a new, stronger, but
not-directly-comparable positive result on an easier 3-class problem,
not a refutation of it.

**Update (2026-08-23, third follow-up) — a fourth run isolated alpha
directly.** Run 4 repeated Run 3's exact configuration (3-class,
20-epoch Stage 1, 4x4 Stage 2) with only `dirichlet_alpha` changed back
to 0.3. Result: macro gain -0.16 pts, worst-node gain -1.06 pts —
essentially flat, reverting to the Run 1/Run 2 pattern rather than Run
3's positive one. This is the first true single-variable comparison in
the series and points at alpha (skew severity), not the `Black_rot`
removal, as the more likely driver of Run 3's positive result. See
`docs/apple_disease_knowledge_transfer_results.md`'s "Run 4" section and
the updated 4-way comparison table.

This is a diagnostic/analysis note, not a new run — it explains *why* the
Apple Dirichlet-skew run (`--epochs 20` Stage 1, `--rounds 2 --distill-epochs 10`
Stage 2) came back flat/null instead of mildly positive like Tomato, and
records concrete tuning recommendations for the next run, including a
lighter `4 epochs x 4 rounds` configuration.

## 1. Why the result was flat/null — root cause is data skew, not epoch count

| Cause | Evidence |
|---|---|
| **node_1's shard is degenerate**: 94.5% `Apple_scab`, 0 `Black_rot`, only 723 images | Both collective *and* local-only-control arms converge to the identical always-predict-`Apple_scab` model (`disease_accuracy` = 0.2510, bit-for-bit identical every round). There is no statistical overlap for KD to exploit. |
| **Stage-2 optimizer reset erases Stage-1 depth** | Every node drops *below* its round-0 baseline in **both** arms at round 1 (e.g. node_0: 0.7208 -> 0.6430/0.5924). This is `_load_apple_node_from_checkpoint` restarting Adam from scratch each Stage-2 round, not a KD side effect. |
| **Only 2 rounds isn't enough to recover from that reset** | kd_loss drops sharply round1->round2 for every node (node_1: 2.97 -> 0.91), showing the models are still consolidating, not converged, when the run stops. |
| **This specific Dirichlet draw (alpha=0.3, 3 nodes) is harsher than Tomato's** | node_1's 94.5% single-class dominance vs. Tomato's most-skewed node at 41%. Same method, worse partition draw. |

**Conclusion: it is not "too much epoch on the KD."** 10 distill-epochs/round
is exactly what the Tomato report's own tuning recommendation suggested
(trade Stage-2 round *count* for epoch *depth*). The Apple run followed
that advice correctly — it simply landed a worse partition draw. Epoch
depth is not what broke this result; the extreme label skew is.

## 2. Raspberry Pi angle

Per `docs/raspberry_pi_deployment.md`, the intended architecture is:
**train on a dev machine/Colab -> export ONNX/int8 -> Pi only runs
inference** ("no training stack (torch/torchvision/timm) needed on the Pi
itself"). The Stage-1/Stage-2 mesh pipeline analyzed here is a
**simulation** of 3 federated nodes run on a dev CPU — it is not currently
wired to run on physical Raspberry Pi hardware.

If real on-device federated training on physical Pis is ever wanted:
- Stage 1 took 10-16 min/node for 20 epochs on the dev CPU used for this
  run. A Pi 4/5 CPU is typically several times slower for this class of
  workload, so expect Stage 1 in the tens-of-minutes-to-hours/node range,
  and Stage 2 (1h42m on dev CPU) could stretch to several hours.
- Practical mitigations if that's the goal: freeze the backbone and only
  fine-tune the classifier head on-device (much cheaper backward pass),
  or keep training on a dev machine/Colab as `raspberry_pi_deployment.md`
  already assumes and use the Pi purely for deployed inference.
- Reducing epochs for Pi feasibility is a **separate lever** from the
  "is 2 rounds x 10 epochs good tuning" question — one is about matching
  a compute budget, the other is about whether collaboration produces a
  measurable gain at all.

## 3. Round-by-round gain, computed explicitly (collective minus control)

| Round | node_0 gain | node_1 gain | node_2 gain | **Macro gain** |
|---|---|---|---|---|
| 1 | 0.6430 - 0.5924 = **+0.0506** | 0.2510 - 0.2510 = 0 | 0.6770 - 0.6673 = **+0.0097** | **+0.0201** (+2.0 pts) |
| 2 (final) | 0.6440 - 0.6459 = **-0.0019** | 0.2510 - 0.2510 = 0 | 0.7179 - 0.7403 = **-0.0224** | **-0.0081** (-0.8 pts, matches the published macro gain) |

**Key finding not spelled out in the original results doc:** round 1 was
mildly *positive* (+2.0 pts macro), and round 2 flipped it *negative*
(-0.8 pts). Going from 1 -> 2 rounds **reversed the sign** of the
collective edge — the control arm's own local-supervised fit caught up
and overtook the collective arm by round 2 for node_0 and node_2. This
is the opposite of "more rounds always help," and is direct evidence
against simply adding round count without also addressing the partition
skew or the per-round optimizer reset.

## 4. Should epochs be reduced, or rounds increased?

- **Do not reduce Stage-1 epochs.** The Tomato ablation (Run 1: 2
  epochs/5 rounds vs. Run 2: 10 epochs/10 rounds) showed deeper Stage-1
  training produces a stronger round-0 baseline that directly translates
  into larger collaboration gains (macro gain 1.6 -> 6.6 pts). Cutting
  Stage-1 depth would likely weaken baselines across the board.
- **Rounds is the more interesting lever, but it's not a simple "more is
  better."** Section 3 above shows round 2 undid round 1's edge for this
  run. Recommended options, in order of expected impact:
  1. **Fix the partition draw first.** Re-run with a different Dirichlet
     seed, or a less extreme `alpha` (e.g. 0.5-1.0 instead of 0.3), so no
     node lands at >90% one class with zero samples of another. This is
     the dominant variable — no epoch/round tuning fixes a structurally
     degenerate shard like node_1's.
  2. **If keeping this exact partition**, trade toward more, shallower
     rounds (e.g. 4-5 rounds x 4-5 epochs, similar total epoch-budget) to
     smooth the per-round optimizer-reset noise and see whether node_0 /
     node_2 keep recovering past round 2 (Tomato Run 2 showed nodes dip
     hard early and only fully recover by round 9-10).
  3. **Consider not resetting Adam state between rounds** (code-level fix
     in `_load_apple_node_from_checkpoint` / `node.local_train`) — this
     is what causes every node to dip below its Stage-1 baseline at
     round 1 in both arms, which is pure overhead unrelated to whether
     KD itself helps.
  4. Don't over-index on a single run/seed — the original results doc's
     own caution applies: repeat across seeds before treating any
     verdict as settled.

## 5. Do not remove `Black_rot` — it's not actually scarce, and it's the one clear win

Pooled across all 3 nodes, `Black_rot` totals **1 + 0 + 649 = 650
images** — comparable to the other classes' pooled totals (`Apple_scab`
964, `Cedar_apple_rust` 617, `healthy` 1,676). It only *looks* rare
because the Dirichlet split concentrated nearly all of it onto node_2 and
starved node_0/node_1 down to 1/0 samples — a property of this specific
partition draw, not of the underlying dataset.

That skew is actually the **best evidence in this run that KD works**:
node_0 went from 0.057 -> 0.109 disease-accuracy on `Black_rot` (+10.9
pts vs. its own local-only control, +5.2 pts vs. round 0) with exactly
**1** training image, purely from node_2's peer signal. Removing
`Black_rot`:
- Eliminates the only unambiguous positive class-level result in the run.
- Does nothing to fix node_1's collapse (driven by `Apple_scab`
  dominance / `healthy` scarcity, unrelated to `Black_rot`).
- Discards a class that is well-represented in the pooled data just to
  paper over a symptom of the partition, not the data itself.

**Recommendation: keep `Black_rot`; re-seed/loosen the Dirichlet split
instead.**

## 6. Alpha recommendations for a lighter `4 epochs x 4 rounds` run

Config knobs: `apple_mesh.dirichlet_alpha` in `config.yaml` (currently
`0.3`, config-only — no `--alpha` CLI flag on `run_apple_pipeline.py` /
`run_apple_knowledge_transfer.py`). Stage 1 epochs via `--epochs 4`,
Stage 2 via `--rounds 4 --distill-epochs 4` (16 epoch-equivalents total,
notably less than this run's 20 — less room to recover from a bad
partition draw or the per-round optimizer reset, which makes the
partition choice even more important than usual).

| # | Alpha | Rationale |
|---|---|---|
| 1 | **0.5** (top pick for this budget) | Mild loosening from 0.3. Keeps each node clearly non-IID (a real dominant class per node) but makes a 94.5%-single-class-with-zero-`Black_rot` node much less likely. With only 16 epoch-equivalents, the partition needs to give KD something recoverable to act on rather than spending the whole budget on a degenerate shard — mirrors why Tomato's less-extreme draw (worst node 41% dominant, no true zero-sample collapse) produced a real positive signal at a smaller budget than Apple's 20. |
| 2 | **1.0** | Moderate skew — every node should end up with all 4 classes present, no literal zero-sample classes. Useful as a "clean" control to isolate whether the 4x4 schedule itself is workable, before reintroducing harsher skew. If macro gain is still flat/negative at alpha=1.0, the bottleneck is the 4x4 budget itself, not the partition. |
| 3 | **0.3, different `data.seed`** | Don't loosen skew at all — re-run at 0.3 with a different seed (or 2-3 seeds) and average, per the original results doc's own caution about not trusting a single run's per-node verdict. Tests whether alpha=0.3 itself is fine and this was just an unlucky draw, without changing benchmark difficulty or breaking comparability with the existing Tomato/Corn/Apple alpha=0.3 reports. |

**Suggested next step:** run with **alpha=0.5** first (best chance of a
non-degenerate signal at this shorter budget), then validate with a
second seed at the same alpha (Recommendation 3's logic) before trusting
the result.
