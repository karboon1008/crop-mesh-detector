# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.061619 kWh (7.702 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00003102 kWh (0.00388 g CO2e)
- Communication is **0.05%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (433 tracked blocks, all invocations): 0.061619 kWh (14.640 g CO2e), 55.1 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 29,
    "num_improved": 23,
    "mean_distill_gain": 0.11730649188886143
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": 0.007526881720430145
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 5,
    "num_improved": 4,
    "mean_distill_gain": 0.13920255949854055
  }
}

Grid carbon intensity used: 125 gCO2/kWh.