# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.783220 kWh (97.902 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00000000 kWh (0.00000 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (428 tracked blocks, all invocations): 0.783220 kWh (186.084 g CO2e), 202.0 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 39,
    "mean_distill_gain": 0.2278083532773381
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 15,
    "num_improved": 14,
    "mean_distill_gain": 0.09127256867043576
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 31,
    "num_improved": 29,
    "mean_distill_gain": 0.12949327967461474
  }
}

Grid carbon intensity used: 125 gCO2/kWh.