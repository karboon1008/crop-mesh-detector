# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.722512 kWh (90.314 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01071812 kWh (1.33976 g CO2e)
- Communication is **1.46%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (488 tracked blocks, all invocations): 0.722512 kWh (173.629 g CO2e), 184.5 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 8,
    "mean_distill_gain": -0.21253060842198834
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 7,
    "num_improved": 3,
    "mean_distill_gain": -0.04475463447826338
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 29,
    "num_improved": 11,
    "mean_distill_gain": -0.05112431661391156
  }
}

Grid carbon intensity used: 125 gCO2/kWh.