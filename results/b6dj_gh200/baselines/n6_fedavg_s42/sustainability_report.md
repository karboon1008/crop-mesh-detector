# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.624175 kWh (78.022 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01198061 kWh (1.49758 g CO2e)
- Communication is **1.88%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (450 tracked blocks, all invocations): 0.624175 kWh (155.716 g CO2e), 172.7 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 20,
    "mean_distill_gain": 0.1069366653423734
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 16,
    "mean_distill_gain": 0.006268688314340246
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 18,
    "mean_distill_gain": 0.018833489433814656
  }
}

Grid carbon intensity used: 125 gCO2/kWh.