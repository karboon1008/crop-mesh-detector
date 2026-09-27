# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.848915 kWh (106.114 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00048201 kWh (0.06025 g CO2e)
- Communication is **0.06%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (379 tracked blocks, all invocations): 0.848915 kWh (204.084 g CO2e), 205.8 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 29,
    "mean_distill_gain": 0.25842085426956307
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 8,
    "num_improved": 8,
    "mean_distill_gain": 0.09061022155147634
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 29,
    "num_improved": 26,
    "mean_distill_gain": 0.1327292917052521
  }
}

Grid carbon intensity used: 125 gCO2/kWh.