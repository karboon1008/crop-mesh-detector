# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.064550 kWh (8.069 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00002968 kWh (0.00371 g CO2e)
- Communication is **0.05%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (426 tracked blocks, all invocations): 0.064550 kWh (15.336 g CO2e), 56.8 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 31,
    "num_improved": 28,
    "mean_distill_gain": 0.08428283803773086
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": 0.0032258064516129115
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 3,
    "num_improved": 3,
    "mean_distill_gain": 0.15608539816113445
  }
}

Grid carbon intensity used: 125 gCO2/kWh.