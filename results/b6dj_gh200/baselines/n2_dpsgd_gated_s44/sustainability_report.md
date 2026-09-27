# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.590916 kWh (73.864 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00529683 kWh (0.66210 g CO2e)
- Communication is **0.89%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (498 tracked blocks, all invocations): 0.590916 kWh (140.349 g CO2e), 155.1 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 45,
    "num_improved": 13,
    "mean_distill_gain": -0.349618483626708
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 0,
    "mean_distill_gain": -0.48156682027649766
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 24,
    "num_improved": 4,
    "mean_distill_gain": -0.45033467649175923
  }
}

Grid carbon intensity used: 125 gCO2/kWh.