# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.063379 kWh (7.922 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00002892 kWh (0.00362 g CO2e)
- Communication is **0.05%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (417 tracked blocks, all invocations): 0.063379 kWh (15.058 g CO2e), 55.8 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 25,
    "num_improved": 25,
    "mean_distill_gain": 0.12921818251140688
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": 0.022580645161290325
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 2,
    "mean_distill_gain": 0.026267281105990803
  }
}

Grid carbon intensity used: 125 gCO2/kWh.