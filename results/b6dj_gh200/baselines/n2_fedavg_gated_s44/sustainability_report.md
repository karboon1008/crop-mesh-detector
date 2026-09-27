# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.808568 kWh (101.071 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00862583 kWh (1.07823 g CO2e)
- Communication is **1.06%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (497 tracked blocks, all invocations): 0.808568 kWh (195.558 g CO2e), 204.2 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 45,
    "num_improved": 16,
    "mean_distill_gain": -0.32612222605201924
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 0,
    "mean_distill_gain": -0.48586789554531484
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 25,
    "num_improved": 2,
    "mean_distill_gain": -0.4195446210586463
  }
}

Grid carbon intensity used: 125 gCO2/kWh.