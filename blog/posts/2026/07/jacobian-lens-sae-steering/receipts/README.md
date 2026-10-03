---
build:
  render: never
  list: never
---

# Receipts for *Can a Jacobian Lens Detect SAE Steering?*

This folder holds the analysis tables behind the numbers in the post, so a
reader can check them without cloning the repository. Seven of the eight files
are byte-identical copies of
[`data/sae_jlens_audit/confirmatory_v1_20260711/analysis/`](https://github.com/tdj28/llm_selfref_pre/tree/c071aa4d737d72818f0774ca389c159b5da67dc1/data/sae_jlens_audit/confirmatory_v1_20260711/analysis)
in `tdj28/llm_selfref_pre` at result commit `c071aa4`. The eighth,
`paired_reference_feature_transport_controls.csv`, was computed after the fact
for this post and is not a release file; see below. The Status column marks
which files hold the prospectively frozen analysis and which are post-run
sensitivities.

| File | What it holds | Status |
|---|---|---|
| `analysis_summary.json` | Trial counts, bootstrap replicates, the primary post-state detector metrics, the primary layer-65 paired semantic effects, and the release's claim-boundary text | Prospectively frozen analysis; copied |
| `detector_metrics.csv` | Post-state-only detector, no clean reference: AUROC / AUPRC / Brier / TPR@1%FPR for the any-intervention and target-attribution tasks, for the J-lens, identity, raw-residual-norm and five random-J readouts. The confirmatory endpoint, J-lens target-attribution AUROC 0.4998, is in this file | Prospectively frozen analysis; copied |
| `static_direction_scores.csv` | Static, direction-only fingerprints: each target, matched-SAE-control and isotropic-control direction projected through the frozen J-lens, the identity lens and the five random-J lenses, with the deception-minus-unrelated score, lexicon-group scores, norms and kurtosis | Prospectively frozen analysis; copied |
| `pursuit_summary.csv` | Sparse nonnegative token-direction pursuit: the share of each direction's squared norm explained by `k` = 5, 10, 16 or 25 token directions, for targets, matched SAE controls and isotropic controls | Prospectively frozen analysis; copied |
| `paired_semantic_effects.csv` | Layer-by-layer change in the frozen deception-minus-unrelated score, target minus dose-matched SAE control, against the same clean prefix, at three readout positions for amplification and suppression; rows for the J-lens, identity and five random-J transports | Prospectively frozen stronger-access sensitivity, not a confirmatory endpoint; copied |
| `paired_reference_metrics.csv` | Attribution AUROC / AUPRC / TPR@1%FPR when the auditor has a matched clean reference, aggregated over features: known-sign and unknown-sign (absolute) scores, for the J-lens, identity and five random-J transports | Post-run sensitivity under the dated amendment; copied |
| `paired_reference_feature_metrics.csv` | The same known-sign clean-reference metrics per target feature, J-lens transport only | Post-run sensitivity under the dated amendment; copied |
| `paired_reference_feature_transport_controls.csv` | The same per-feature known-sign metrics for the J-lens, identity and five random-J transports | Derived for this post, not a release file; see below |

"Dated amendment" means
[`docs/SAE_JLENS_POSTRUN_AMENDMENT_20260711.md`](https://github.com/tdj28/llm_selfref_pre/blob/c071aa4d737d72818f0774ca389c159b5da67dc1/docs/SAE_JLENS_POSTRUN_AMENDMENT_20260711.md),
written after the primary result was opened. None of the three
`paired_reference_*` files is the confirmatory post-state detector.

## The one derived file

`paired_reference_feature_transport_controls.csv` is not in the release. It
was computed for this post after the release was published.

- Inputs: the released
  [`paired_results/`](https://github.com/tdj28/llm_selfref_pre/tree/c071aa4d737d72818f0774ca389c159b5da67dc1/data/sae_jlens_audit/confirmatory_v1_20260711/paired_results)
  shards at `c071aa4`.
- Code: the scoring functions of the release's own
  `experiments/exp2_sae/analyze_sae_jlens_paired_reference.py`. That script
  writes per-feature rows for the J-lens alone; the rows for the other
  transports come from a small wrapper around its `score_rows` / `summarize`
  functions that is not included here, so they cannot be regenerated from
  these receipts alone. The score is the known-sign score: the paired change
  in the deception-minus-unrelated logit score, multiplied by the requested
  intervention sign. Intervals come from the same 20,000-replicate bootstrap
  over the 51 template families.
- Check: the `jacobian` rows match the released
  `paired_reference_feature_metrics.csv` exactly, which shows the rerun
  reproduces the release before anything new is added.
- What is new: the same computation for the `identity` transport and the five
  `random_j_*` transports, per feature. The release reports those transports
  only in aggregate, in `paired_reference_metrics.csv`.
- Status: a post-run sensitivity, like the two released `paired_reference_*`
  files: it assumes a matched clean reference and a known intervention sign.
  It is an automated recomputation made for this post, not a release artifact.

## Figures

The six `sae_jlens_*.png` files in the parent directory are byte-identical
copies of the v1 release figures,
[`data/sae_jlens_audit/confirmatory_v1_20260711/figures/`](https://github.com/tdj28/llm_selfref_pre/tree/c071aa4d737d72818f0774ca389c159b5da67dc1/data/sae_jlens_audit/confirmatory_v1_20260711/figures)
at `c071aa4`. The three `sae_jlens_v2_*.png` files are byte-identical copies
from the separate v2 release,
[`data/sae_jlens_audit/confirmatory_v2_20260712/post_failure/figures/`](https://github.com/tdj28/llm_selfref_pre/tree/478a10dd0670eee47fc151882560482ae79fc790/data/sae_jlens_audit/confirmatory_v2_20260712/post_failure/figures)
at `478a10d`; that study failed its frozen replay gate, so everything those
figures show is exploratory post-outcome calculation, as the post says where it
uses them. The `.svg` files and `og-card.png` are diagrams drawn for the post,
not release files.

## What is not here

The v2 analysis tables are not mirrored in this folder, only its three
figures. Nothing here is an independent check: the copied files are the
release's own outputs, and the derived file reuses the release's code on the
release's data. To verify, diff each copied file against its pinned path above,
and rerun `analyze_sae_jlens_paired_reference.py` at `c071aa4` on the pinned
`paired_results/` shards; the two released `paired_reference_*.csv` files come
out directly, and the derived file follows from running the script's
per-feature summary over each of the seven transports.
