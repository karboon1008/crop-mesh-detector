# Sustainability report

- Compute energy measurement basis: **at least one block used a constant-TDP estimate, not a hardware measurement** — see per-block `hardware_sources` in the JSON report
- Measured/estimated **compute energy**: 0.933050 kWh (116.631 g CO2e)
- Estimated **communication energy** (Wi-Fi, prototypes + probe logits only): 0.01437673 kWh (1.79709 g CO2e)
- Communication is **1.52%** of total energy — well within the 'communication should not erase compute savings' target.
- **Real sweep total from emissions.csv** (540 tracked blocks, all invocations): 0.933050 kWh (227.588 g CO2e), 238.6 min.

## Collaboration gain vs. energy spent

{
  "mobilenet_v3_small": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 20,
    "mean_distill_gain": 0.052485071602053755
  },
  "efficientnet_lite0": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 17,
    "mean_distill_gain": -0.0035523430112670007
  },
  "mobilevit_xxs": {
    "metric": "pair_accuracy",
    "num_batches": 6,
    "num_distillations": 36,
    "num_improved": 18,
    "mean_distill_gain": 0.012873222738701501
  }
}

Grid carbon intensity used: 125 gCO2/kWh.