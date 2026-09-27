# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.681558 kWh (85.195 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00989688 kWh (1.23711 g CO2e)
- Communication is **1.43%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (546 tracked blocks, all invocations): 0.681558 kWh (172.688 g CO2e), 175.3 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 48,
    "num_improved": 26,
    "mean_distill_gain": -0.08276554104291635
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 7,
    "num_improved": 3,
    "mean_distill_gain": -0.0010417922658127546
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 28,
    "num_improved": 14,
    "mean_distill_gain": 0.04151267316097062
  }
}

Grid carbon intensity used: 125 gCO2/kWh.