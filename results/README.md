# Results

Finished continual-stream runs, copied back from Isambard as committed evidence.
Model checkpoints (`*.pt`) and hive databases (`*.db`) are left out; everything
else each run wrote is kept (per-batch logs, pooled-test logs, per-arch results,
sustainability reports, CodeCarbon `emissions.csv`, trigger logs).

Runs are grouped by cluster because compute-energy figures are hardware-specific:
A100 numbers and GH200 numbers are not comparable with each other.

Each run folder is named `<setting>_<strategy>_s<seed>` and holds
`continual/<arch>/{batch_summary.csv,batch_logs.json,pooled_log.json}` for
`mobilenet_v3_small`, `efficientnet_lite0` and `mobilevit_xxs`.

| Folder | Cluster | Code | Contents |
|---|---|---|---|
| `b36bl_a100/baselines/` | Isambard `b36bl.macs3` (A100) | `993ebf0` + the control-group code in commit `d01728a` | n2 seed 42: `continual` (HiveMind), `fedavg`, `fedavg_gated`, `dpsgd`, `dpsgd_gated`, `local_extra`; n4 seed 42: `continual` |
| `b36bl_a100/ablation/` | same | as above, plus: skip the KD pass when both KD weights are 0 | n2 seed 42: `logits_only` (`proto_weight: 0`), `protos_only` (`kd_weight = crop_kd_weight = 0`) |
| `b6dj_gh200/baselines/` | Isambard-AI `b6dj.aip2` (GH200) | same control-group code | n2 seeds 43/44 (runs finished so far); n6 seeds 42-44, all six strategies — **incomplete**: the n6 jobs ended early (seed 42 reached batch 7, seed 43 batch 5, seed 44 batch 4, of 0-12), so only compare n6 runs over the batches they share |

Settings (`configs/`): `n2` = node_0 Tomato, node_1 Grape; `n4` = 2 Tomato + 2 Grape
nodes (PlantVillage + PlantDoc + PlantWild); `n6` = all PlantVillage crops, Dirichlet split.

Strategies: `continual` = our knowledge hive; `fedavg`/`dpsgd` = standard weight
exchange; `*_gated` = the same, using our EMA upload/retrieve gates;
`local_extra` = no exchange, the same extra local epochs our distillation step
contains (control for "is it just more training?").

`scripts/analyze_baselines.py <runs dir> <setting>` summarises a setting.
