# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.926740 kWh (115.842 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01042101 kWh (1.30263 g CO2e)
- Communication is **1.11%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (517 tracked blocks, all invocations): 0.926740 kWh (220.075 g CO2e), 233.5 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 24,
    "mean_distill_gain": -0.03564672259908875
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 8,
    "num_improved": 3,
    "mean_distill_gain": -0.04302618957846055
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 36,
    "num_improved": 17,
    "mean_distill_gain": 0.014322295249170487
  }
}

Grid carbon intensity used: 125 gCO2/kWh.