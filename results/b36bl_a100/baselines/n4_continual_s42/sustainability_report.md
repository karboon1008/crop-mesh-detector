# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.094339 kWh (11.792 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00012312 kWh (0.01539 g CO2e)
- Communication is **0.13%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (984 tracked blocks, all invocations): 0.094339 kWh (22.414 g CO2e), 101.9 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 78,
    "num_improved": 56,
    "mean_distill_gain": 0.12077402052672073
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 8,
    "num_improved": 6,
    "mean_distill_gain": -0.03144419122367689
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 45,
    "num_improved": 40,
    "mean_distill_gain": 0.12946974311078036
  }
}

Grid carbon intensity used: 125 gCO2/kWh.