# Research Document Structure — Cambridge Edge AI Innovation for Sustainability Challenge 2026

Source: `docs/Official Problem Statement_0.pdf` (extracted and read in full — Vision, Problem
Statement, Core Challenges, the 4 Scoring Axes, Knowledge-Exchange Rules, Judging Principles,
Format/Timeline, Deliverables, and all 7 appendices). This file is a reference outline for writing the
mandatory **research document** (preliminary-round deliverable #1 of 3, per §8.1), structured so
every mandatory disclosure/table and all four scored axes (A/B/C/D, 25% each) are covered and
easy for a judge to find.

Cross-references below (e.g. "§5.2", "Appendix A.1") point to sections of the official PDF.

## Recommended section order (flow rationale)

Mirrors how judges are told to read a submission (§6, §8.1): vision -> mechanism -> evidence it
actually ran -> cost/fairness disclosures -> the four scored axes each in their own clearly labelled
section -> disclosures/IP -> references. Labelling Axis A/B/C/D sections explicitly (rather than
scattering the content) matters because judging is axis-by-axis (25/25/25/25) — a submission
where a judge can't quickly find "the Axis B section" scores worse on clarity even with identical
content.

### 0. Front matter
- Team name/members, UK-university affiliation of the lead representative (§11)
- Licence statement (required in deliverable #3 per §8.1, but worth stating up front too)
- 1-paragraph executive summary: node count, scenario, headline ΔG, headline sustainability
  number — judges skim this first

### 1. Vision & Problem Framing (§1–§2)
- Which application scenario was picked (§2.1) and why — image or non-image; if non-image,
  explicitly justify the modality choice (mandatory when deviating from the default image scenario)
- Which Core Challenge(s) from §3 are targeted (Non-IID data / new classes / node failure /
  continual learning / catastrophic forgetting) — pick at least one explicitly, since Axis C's
  "scenario-driven demonstration" is graded against whatever is claimed here

### 2. System Architecture
- Architecture diagram (nodes, model, exchange path) — explicitly required in §8.1
- Node roles, model choice, self-learning form (supervised / self-supervised / self-reinforcement —
  §2 permits RL-derived knowledge exchange too; mention if used)

### 3. Knowledge-Exchange Design (§5.1–§5.3 — core of Axis C/D novelty)
- What is exchanged (prototypes / logits / deltas / etc.), why this representation was chosen, and
  the aggregation/robustness mechanism (e.g. trimmed-mean, Krum)
- **Exchange Artefact Table** (§5.2 — mandatory). Required fields per artefact:
  `artifact_name`, `purpose`, `direction`, `frequency`, `size_per_message`, `granularity`,
  `contains_metadata`, `reconstruction_risk`, `risk_reason`, `privacy_measure` (△), `notes` (△).
  Any row with `reconstruction_risk = low` **must** have a `risk_reason`, or it is treated as
  not-low by default (§5.2).
- **Model-Initialisation Disclosure** (§5.3 — mandatory): shared pretrained source + licence;
  explicit statement that it was not trained on the union of nodes' private data and was not
  specifically pre-targeted at the Challenge's domain/task (pre-training purity). If public
  pretraining data closely resembles a node's private data, disclose the possible effect on the
  local_only baseline and on ΔG.

### 4. Experimental Setup
- Data, non-IID construction method and its strength (e.g. Dirichlet alpha, disjoint-by-class),
  train/val/test split, random seed, per-node sample counts
- Baselines run: local_only (mandatory per §6.1 item 3, or a written justification for why one
  cannot be provided), plus any ablations/control groups

### 5. Collaboration-Gain Results (Axis C core sub-criterion)
- ΔG_local and/or ΔG_best; report **both** the macro-average and the **worst-node** score, not
  just the best or overall average
- **Collaboration-Gain Fairness Disclosure Table** (Appendix A.1 — mandatory whenever ΔG is
  reported). Required fields: `task_metric`, `node_count`, `data_split`, `non_iid_description`,
  `test_set_scope`, `local_only_budget`, `collective_budget`, `per_node_scores`,
  `delta_g_formula`. If local_only and collective budgets are not aligned, a
  `fairness_exception_reason` is mandatory — otherwise ΔG may not be treated as
  high-confidence evidence.
- Scenario-driven demonstration evidence: new-class injection, node disconnect/reconnect,
  distribution shift — must be triggerable by judges following run instructions, or backed by an
  automation script / screen recording (§4.3, §8.1)

### 6. Sustainability Impact — Axis A (25%)
Give this its own major section, not a footnote.
- State the chosen baseline explicitly and justify it:
  - **Baseline A | Centralised cloud** — compares communication volume/energy saved by not
    uploading raw data; may be a pure estimate/offline simulation (must not be used as the
    submitted system's initialisation weights or workflow — see §5.3).
  - **Baseline B | Local-only** — same baseline as Axis C's local_only; focuses on efficiency
    (`gain_per_joule`, `gain_per_byte`) since collaboration adds communication cost.
- Cost inventory per Appendix E — cover to the extent the claim requires:
  - **Compute energy**, including on-device training/updates (backprop, fine-tuning), not just
    inference — collaboration's added cost is disproportionately here
  - **Memory-access energy** (the "Memory Wall") — do not rely on FLOPs x J/MAC alone; DRAM
    access can be ~100x a MAC's energy
  - **Communication energy** — bytes exchanged x J/byte for the radio used x rounds/frequency;
    show whether communication energy negates the compute energy collaboration saves
  - Do not take TDP directly as actual power; state utilisation factor, voltage/frequency/
    quantisation precision assumptions
- State an **Estimation-Credibility grade** (Low / Medium / High, per Appendix E.3) and disclose
  assumptions/uncertainty
- **A4**: discuss counter-cases — when might collaboration NOT be worth it?
- Environmental-benefit narrative if used as a scoring basis: deployment scale, time horizon,
  source of conversion factors, uncertainty

### 7. Ease of Integration & Feasibility — Axis B (25%)
- Integration path into an existing Edge AI pipeline, prerequisites, effort estimate
- Dependency list (common, clearly versioned, not closed/obscure)
- At least one concrete industrial mapping + adoption path
- Honest barriers/open-problems section — do not skip; over-optimistic barrier identification is
  explicitly penalised in the scoring anchors (§4.5)
- Inter-node interface/boundary documentation (designed to be easy to connect to later)

### 8. Technical Maturity — Axis C (25%, ties back to Section 3/5 above)
- System stability & error-handling notes
- Reproducibility mapped to the **three-tier evidence levels** (§8.2):
  - L1: smoke test runs the multi-node flow but doesn't reproduce the document's figures
  - L2: key small-scale results re-runnable in a general environment
  - L3: main results re-runnable, or complete hardware-measurement evidence provided
  - State which tier is claimed and why (L1 alone does not count as full reproduction; hardware-
    constrained teams that provide reviewable evidence are not penalised for falling short of L3)
- Link back to Section 5's ΔG table and ablations here so this section is self-contained for a judge
  scoring Axis C alone

### 9. Creativity — Axis D (25%)
- Originality of the knowledge representation, cross-domain imagination, future extensibility, and
  a distinctive interpretation of "collective intelligence" — no mandatory table for this axis, but it
  should not be an afterthought paragraph

### 10. Academic & Industry Alignment / Future Work (§9–§10)
- First-principles discussion of the communication/compute/energy trade-offs made
- Continual-learning capability; emergent-collaborative-value explanation (why the collective
  result beats a single node or a naive concatenation)
- Post-challenge collaboration intent (optional)

### 11. Pre-existing IP & Licensing Disclosure (§5.3 + Appendix F)
- Any pre-existing code/libraries/datasets/models used, with licence-compliance statement

### 12. References (Appendix D)
- Format is free, but key method/source papers must be listed. Suggested references from the
  official document if relevant to the approach:
  1. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*
  2. Hinton et al., *Distilling the Knowledge in a Neural Network*
  3. Bilen et al., *Towards Universal Object Detection by Domain Attention*
  4. Kirkpatrick et al., *Overcoming Catastrophic Forgetting in Neural Networks*
  5. Snell et al., *Prototypical Networks for Few-shot Learning*
  6. Lian et al., *Can Decentralized Algorithms Outperform Centralized Algorithms?*
  7. Kairouz et al., *Advances and Open Problems in Federated Learning*

### Appendices (keep heavy tables out of the main flow so it reads well)
- Full Exchange Artefact Table, full ΔG Fairness Table, extended per-round logs/loss curves,
  additional ablations

## Don't-miss checklist (§6.1 Minimum Evidence Requirements + §8.1 mandatory deliverable list)

1. Node count, roles, exchange timing, and exchanged content stated explicitly
2. Machine-generated execution evidence (JSON/CSV exchange summary, logs, checkpoint
   metadata, or recording) — not prose alone
3. local_only baseline comparison, or explicit justification for why one cannot be provided
4. Complete Exchange Artefact Table (§5.2)
5. At least one sustainability metric (energy / communication / compute) — a rigorous estimate is
   acceptable, no measurement hardware required
6. ΔG Fairness Disclosure Table whenever any ΔG number appears (Appendix A.1)
7. Scenario-driven demo trigger instructions
8. Reproducibility run instructions with dependency versions + an entry point
9. Model-initialisation source + licence disclosure (§5.3)
10. References list (Appendix D)

## Mapping to this repo's existing artefacts

This project (Apple/Tomato/Corn Dirichlet-mesh federated pipelines) already has strong raw
material for several sections above:

| Document section | Existing project artefact |
|---|---|
| §3 Knowledge-Exchange Design / Exchange Artefact Table | `src/federated/node.py` (`KnowledgePayload`: prototypes + probe-set logits, never raw data/weights/gradients), `src/federated/aggregation.py` (trimmed-mean / Krum) |
| §4 Experimental Setup | `src/validation/apple_mesh_dataset.py`, `tomato_mesh_dataset.py`, `corn_mesh_dataset.py` (Dirichlet partition / disjoint-label partition, dedup-aware global test split) |
| §5 Collaboration-Gain Results / Fairness Table | `src/validation/run_apple_knowledge_transfer.py` (`build_apple_knowledge_transfer_summary`), `run_knowledge_transfer.py`, and `docs/apple_disease_knowledge_transfer_results.md` / `tomato_disease_knowledge_transfer_results.md` / `corn_disease_knowledge_transfer_results.md` |
| §6 Sustainability / Axis A cost inventory | `src/energy/tracker.py` (`ComputeEnergyTracker`, `CommunicationCostEstimator`), `docs/sustainability_energy_plan.md` |
| §8 Reproducibility tiers | `docs/execution_guide.md`, `docs/app_and_tomato_validation_execution_steps.md` |
| Scenario-driven demonstration | `docs/mesh_disruption_scenarios.md`, `docs/mesh_model_crop_scenario_comparison.md` |

Next suggested step: draft the actual Exchange Artefact Table and Collaboration-Gain Fairness
Table populated from the existing Apple/Tomato/Corn run docs.
