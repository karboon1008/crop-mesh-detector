# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.566584 kWh (70.823 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01198061 kWh (1.49758 g CO2e)
- Communication is **2.07%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (450 tracked blocks, all invocations): 0.566584 kWh (143.449 g CO2e), 148.6 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 15,
    "mean_distill_gain": -0.0019322433539806264
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 14,
    "mean_distill_gain": -0.006088322690603483
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 14,
    "mean_distill_gain": -0.022806989249305486
  }
}

Grid carbon intensity used: 125 gCO2/kWh.