# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.706561 kWh (88.320 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01437673 kWh (1.79709 g CO2e)
- Communication is **1.99%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (540 tracked blocks, all invocations): 0.706561 kWh (169.565 g CO2e), 192.0 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 19,
    "mean_distill_gain": -0.023433501072684937
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 14,
    "mean_distill_gain": -0.01895937801890851
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 18,
    "mean_distill_gain": 0.0060966827337501145
  }
}

Grid carbon intensity used: 125 gCO2/kWh.