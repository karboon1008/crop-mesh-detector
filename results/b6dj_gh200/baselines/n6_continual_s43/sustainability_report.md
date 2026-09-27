# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.555163 kWh (69.395 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00044113 kWh (0.05514 g CO2e)
- Communication is **0.08%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (356 tracked blocks, all invocations): 0.555163 kWh (131.626 g CO2e), 150.4 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 30,
    "num_improved": 28,
    "mean_distill_gain": 0.24733808130308932
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 6,
    "num_improved": 6,
    "mean_distill_gain": 0.07050463097533415
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 5,
    "num_distillations": 24,
    "num_improved": 24,
    "mean_distill_gain": 0.15851019816769737
  }
}

Grid carbon intensity used: 125 gCO2/kWh.