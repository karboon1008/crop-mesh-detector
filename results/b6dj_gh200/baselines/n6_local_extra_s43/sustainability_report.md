# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.851359 kWh (106.420 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00000000 kWh (0.00000 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (468 tracked blocks, all invocations): 0.851359 kWh (204.112 g CO2e), 217.2 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 48,
    "num_improved": 44,
    "mean_distill_gain": 0.2086287837615268
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 7,
    "num_improved": 7,
    "mean_distill_gain": 0.07161347543128056
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 35,
    "num_improved": 31,
    "mean_distill_gain": 0.13136604360009377
  }
}

Grid carbon intensity used: 125 gCO2/kWh.