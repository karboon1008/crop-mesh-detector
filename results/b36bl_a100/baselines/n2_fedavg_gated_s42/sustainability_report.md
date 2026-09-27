# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.050779 kWh (6.347 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00772890 kWh (0.96611 g CO2e)
- Communication is **13.21%** of total energy — a significant share of the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (454 tracked blocks, all invocations): 0.050779 kWh (12.064 g CO2e), 50.6 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 36,
    "num_improved": 12,
    "mean_distill_gain": -0.28332687641500975
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": -0.48940092165898613
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 9,
    "num_improved": 2,
    "mean_distill_gain": -0.36346757496827625
  }
}

Grid carbon intensity used: 125 gCO2/kWh.