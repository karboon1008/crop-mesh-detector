# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.811782 kWh (101.473 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00000000 kWh (0.00000 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (476 tracked blocks, all invocations): 0.811782 kWh (199.012 g CO2e), 205.5 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 48,
    "num_improved": 45,
    "mean_distill_gain": 0.23861822838810773
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 7,
    "num_improved": 7,
    "mean_distill_gain": 0.10358819786380272
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 8,
    "num_distillations": 39,
    "num_improved": 37,
    "mean_distill_gain": 0.11812566127534227
  }
}

Grid carbon intensity used: 125 gCO2/kWh.