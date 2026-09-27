# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.691133 kWh (86.392 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00061390 kWh (0.07674 g CO2e)
- Communication is **0.09%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (506 tracked blocks, all invocations): 0.691133 kWh (166.238 g CO2e), 182.7 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 40,
    "mean_distill_gain": 0.2413365344780262
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 14,
    "num_improved": 14,
    "mean_distill_gain": 0.13067538784639424
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 36,
    "num_improved": 34,
    "mean_distill_gain": 0.15303224295652398
  }
}

Grid carbon intensity used: 125 gCO2/kWh.