# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.633055 kWh (79.132 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00002380 kWh (0.00298 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (393 tracked blocks, all invocations): 0.633055 kWh (152.506 g CO2e), 166.5 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 17,
    "num_improved": 17,
    "mean_distill_gain": 0.12463139533513264
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": 0.007066052227342523
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 3,
    "num_improved": 3,
    "mean_distill_gain": 0.09002426590084374
  }
}

Grid carbon intensity used: 125 gCO2/kWh.