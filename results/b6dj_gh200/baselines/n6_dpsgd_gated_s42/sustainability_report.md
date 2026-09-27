# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.647006 kWh (80.876 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01389744 kWh (1.73718 g CO2e)
- Communication is **2.10%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (615 tracked blocks, all invocations): 0.647006 kWh (152.548 g CO2e), 176.9 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 9,
    "num_distillations": 54,
    "num_improved": 24,
    "mean_distill_gain": -0.08782593226001373
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 9,
    "num_distillations": 6,
    "num_improved": 1,
    "mean_distill_gain": -0.03977322668812172
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 9,
    "num_distillations": 33,
    "num_improved": 19,
    "mean_distill_gain": 0.0319835390767681
  }
}

Grid carbon intensity used: 125 gCO2/kWh.