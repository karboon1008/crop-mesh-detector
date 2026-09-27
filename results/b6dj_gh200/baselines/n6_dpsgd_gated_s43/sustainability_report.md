# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.730789 kWh (91.349 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01051397 kWh (1.31425 g CO2e)
- Communication is **1.42%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (484 tracked blocks, all invocations): 0.730789 kWh (180.618 g CO2e), 185.7 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 17,
    "mean_distill_gain": -0.05609704996090952
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 6,
    "num_improved": 0,
    "mean_distill_gain": -0.09853434226966817
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 25,
    "num_improved": 8,
    "mean_distill_gain": -0.02847485148891351
  }
}

Grid carbon intensity used: 125 gCO2/kWh.