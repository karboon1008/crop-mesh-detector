# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.348760 kWh (43.595 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00000000 kWh (0.00000 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (338 tracked blocks, all invocations): 0.348760 kWh (82.862 g CO2e), 99.3 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 24,
    "num_improved": 23,
    "mean_distill_gain": 0.12623533343415339
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 2,
    "mean_distill_gain": 0.02549923195084486
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 5,
    "num_improved": 5,
    "mean_distill_gain": 0.09248914713150336
  }
}

Grid carbon intensity used: 125 gCO2/kWh.