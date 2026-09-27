# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.755790 kWh (94.474 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01198061 kWh (1.49758 g CO2e)
- Communication is **1.56%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (450 tracked blocks, all invocations): 0.755790 kWh (195.245 g CO2e), 190.3 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 11,
    "mean_distill_gain": -0.09904911915281361
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 11,
    "mean_distill_gain": -0.007221430243167489
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 12,
    "mean_distill_gain": -0.014384646175295038
  }
}

Grid carbon intensity used: 125 gCO2/kWh.