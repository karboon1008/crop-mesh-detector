# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.821875 kWh (102.734 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00955013 kWh (1.19377 g CO2e)
- Communication is **1.15%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (504 tracked blocks, all invocations): 0.821875 kWh (195.275 g CO2e), 216.2 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 42,
    "num_improved": 16,
    "mean_distill_gain": -0.12555034748413108
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 6,
    "num_improved": 1,
    "mean_distill_gain": -0.08845738636114975
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 7,
    "num_distillations": 38,
    "num_improved": 6,
    "mean_distill_gain": -0.12525941999201165
  }
}

Grid carbon intensity used: 125 gCO2/kWh.