# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.554074 kWh (69.259 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.00000000 kWh (0.00000 g CO2e)
- Communication is **0.00%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (350 tracked blocks, all invocations): 0.554074 kWh (142.558 g CO2e), 141.1 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 27,
    "num_improved": 25,
    "mean_distill_gain": 0.12183184533257256
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 2,
    "num_improved": 1,
    "mean_distill_gain": 0.013978494623655913
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 23,
    "num_distillations": 8,
    "num_improved": 8,
    "mean_distill_gain": 0.1165364322447071
  }
}

Grid carbon intensity used: 125 gCO2/kWh.