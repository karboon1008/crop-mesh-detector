# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.571174 kWh (71.397 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00803467 kWh (1.00433 g CO2e)
- Communication is **1.39%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (451 tracked blocks, all invocations): 0.571174 kWh (152.157 g CO2e), 148.6 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 35,
    "num_improved": 9,
    "mean_distill_gain": -0.2731921055784009
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 0,
    "mean_distill_gain": -0.4772657450076805
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 7,
    "num_improved": 6,
    "mean_distill_gain": -0.043344687103452864
  }
}

Grid carbon intensity used: 125 gCO2/kWh.