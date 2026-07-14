---
build:
  render: never
  list: never
---

# Receipts for *Can a Jacobian Lens Detect SAE Steering?*

Copied analysis tables from the public release at
[`tdj28/llm_selfref_pre`](https://github.com/tdj28/llm_selfref_pre)
(`data/sae_jlens_audit/confirmatory_v1_20260711/analysis/`).

| File | Contents |
|---|---|
| `analysis_summary.json` | Trial counts, primary detector metrics, claim boundary |
| `detector_metrics.csv` | AUROC / AUPRC / TPR@1%FPR by task and readout |
| `static_direction_scores.csv` | Static J / identity / random-J direction projections |
| `pursuit_summary.csv` | Sparse token-direction pursuit by `k` |
| `paired_semantic_effects.csv` | Layerwise target-minus-matched J effects |
| `paired_reference_metrics.csv` | Paired-reference fixed-score metrics |
| `paired_reference_feature_metrics.csv` | Per-feature paired-reference AUROC |
| `paired_reference_feature_transport_controls.csv` | Per-feature paired AUROC for identity and random-J transports (derived) |

`paired_reference_feature_transport_controls.csv` is **not** a copy of a
release file: it is derived post hoc for this post from the released
`paired_results/` shards, reusing the release's own
`analyze_sae_jlens_paired_reference.py` scoring functions (known-sign score,
20,000-replicate template-cluster bootstrap). The jacobian rows reproduce the
released `paired_reference_feature_metrics.csv` exactly (replication gate),
and the identity / random-J rows extend the same computation to the transport
controls.

Figures in the parent directory are the six release PNGs from
`.../figures/sae_jlens_*.png`.
