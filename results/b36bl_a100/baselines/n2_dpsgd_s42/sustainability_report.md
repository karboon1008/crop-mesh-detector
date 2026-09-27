# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.064222 kWh (8.028 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00910457 kWh (1.13807 g CO2e)
- Communication is **12.42%** of total energy — a significant share of the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (690 tracked blocks, all invocations): 0.064222 kWh (15.258 g CO2e), 70.1 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 46,
    "num_improved": 14,
    "mean_distill_gain": -0.3906260481343513
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 46,
    "num_improved": 15,
    "mean_distill_gain": -0.4277353990361617
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 46,
    "num_improved": 12,
    "mean_distill_gain": -0.33687670471071374
  }
}

Grid carbon intensity used: 125 gCO2/kWh.