# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.969244 kWh (121.156 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00002838 kWh (0.00355 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (412 tracked blocks, all invocations): 0.969244 kWh (232.533 g CO2e), 233.0 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 23,
    "num_improved": 23,
    "mean_distill_gain": 0.1702537030422877
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 2,
    "mean_distill_gain": 0.01766513056835639
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 2,
    "mean_distill_gain": 0.06820276497695854
  }
}

Grid carbon intensity used: 125 gCO2/kWh.