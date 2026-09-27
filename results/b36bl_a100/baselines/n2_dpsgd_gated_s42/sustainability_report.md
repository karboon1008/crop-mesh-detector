# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.059375 kWh (7.422 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00545137 kWh (0.68142 g CO2e)
- Communication is **8.41%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (439 tracked blocks, all invocations): 0.059375 kWh (14.107 g CO2e), 55.0 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 36,
    "num_improved": 11,
    "mean_distill_gain": -0.31701613022285013
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 0,
    "mean_distill_gain": -0.49370199692780337
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": -0.4617511520737327
  }
}

Grid carbon intensity used: 125 gCO2/kWh.