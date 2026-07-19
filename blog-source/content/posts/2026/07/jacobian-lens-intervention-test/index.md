---
title: "A Linear Nudge, a Nonlinear Wake"
slug: "jacobian-lens-intervention-test"
date: 2026-07-14
lastmod: 2026-07-18
aliases: ["/posts/jacobian-lens-intervention-test/"]
tags: ["AI", "LLM", "machine-learning", "interpretability", "sparse-autoencoders", "jacobian-lens", "reproducibility"]
author: Timothy Jones
lead: |
  A transformer repeatedly transforms a high-dimensional hidden state as
  information moves through its layers. If that state is deliberately changed
  halfway through the network, do the remaining layers carry the change forward
  as a local linear model predicts, or bend it into a different trajectory? The
  answer matters for interpreting what activation steering does inside a
  language model.
summary: "We followed controlled hidden-state edits through the final 29 blocks of Llama 3.3 70B. The delivered edits were dose-linear and a fixed corpus-average Jacobian preserved that scaling, while the model's actual downstream response did not."
key_result: |
  Against a frozen first-order benchmark on Llama 3.3 70B, the split was
  complete: 24 of 24 delivered edits stayed dose-linear over the prespecified
  2%/3%/4% panel, their fixed corpus-average Jacobian references stayed inside
  the same bounds, and 0 of 24 actual final-state responses did. The released
  Jacobian beat five random-map controls but did not clear the stronger
  added-value-over-identity margin at the primary layer-50 comparison (logit
  advantage 0.011, below the frozen 0.02 threshold). Linear edit in, nonlinear
  hidden-state wake out, for these generic directions only.
toc: false
draft: true
---

{{< panel "info" >}}
**AI-use disclosure.** Generative-AI tools helped implement, audit, execute,
interpret, visualize, review, and draft this study. The author selected the
research question, authorized the compute, has inspected the artifacts, and is
responsible for the final text and claims. This is an independent,
non-peer-reviewed Research Note. Verify numbers against the released receipts
before relying on them.
{{< /panel >}}

{{< panel "info" >}}
**Abstract.** In this note, activation steering means additive
residual-stream intervention: a controlled vector added to a model's
{{< refterm "residual-stream" "residual stream" >}}. That is one common
steering primitive, not a definition of the whole field. A
{{< refterm "jacobian-lens" "Jacobian lens" >}} is a corpus-average first-order
map from that mid-network state toward later residuals and vocabulary
dispositions. Before attributing a downstream effect to the meaning of a
steering direction, requested-versus-realized fidelity under low precision,
linear transport, and nonlinear model dynamics need to be separated.

**(1)** In Llama 3.3 70B Instruct, we applied three unselected Gaussian directions
after block 50 across eight prompts, varied dose from `0.5%` to `30%` of
{{< refterm "rms" "residual RMS" >}}, and followed the signed central response
through the remaining 29 blocks. Protocol, plan, and decision rules were frozen
before outcomes
([`a084caa`](https://github.com/tdj28/llm_selfref_pre/commit/a084caafc2ec27860044d80d3b33912f656fd08a)).

**(2)** On the frozen `2%`/`3%`/`4%` panel, every one of the 24 prompt ×
direction cases kept a dose-linear *source* edit, while every *final*
hidden-state response failed **both** gates: direction drifted
(\(c_{\min}\) fell to `0.778`–`0.848`, below `0.95`) and dose-normalized size
mismatched (\(d_{\max}\) rose to `0.572`–`0.703`, above `0.15`). The fixed-J
projection of the realized edits stayed inside the same bounds as an algebraic
consistency reference, not as a claim that J predicted the model.

**(3)** Requested-versus-realized edit fidelity under
{{< refterm "bf16" "BF16" >}} casting failed the frozen rule below `2%` and
passed from `2%` through `30%` (the protocol's short name for this check is
"delivery"). Median final-state {{< refterm "rms" "RMS" >}} gain was `1.82x`
at `2%` and tapered to `1.48x` at `30%` over this fixed census.

**(4)** At the primary layer-50 comparison, released J beat five
{{< refterm "random-j" "random-J" >}} maps but did not clear
added-value-over-identity (logit advantage `0.011`, residual advantage slightly
negative).

**Takeaway.** Linear edit in, nonlinear hidden-state wake out, for these
generic directions on this model. The result motivates an SAE-specific dose
scan; it does not establish an SAE, behavioral, or consciousness claim.
{{< /panel >}}

Study status: **complete** (pre-outcome freeze
[`a084caa`](https://github.com/tdj28/llm_selfref_pre/commit/a084caafc2ec27860044d80d3b33912f656fd08a);
audited result release
[`fde24e9`](https://github.com/tdj28/llm_selfref_pre/commit/fde24e93770859bf0ec848b91eb0564d550a641d)).
Experiment package:
[`experiments/consciousness_sae_signed_dose_scan`](https://github.com/tdj28/llm_selfref_pre/tree/main/experiments/consciousness_sae_signed_dose_scan).
Shipping table, sample records, and hashes are in the
[appendix](#appendix-release-inventory).

## The Question

Activation steering changes a model from the inside. In the additive form
studied here, the researcher adds a controlled vector to the hidden state
carried between transformer blocks instead of changing the prompt; contrastive
activation addition is the canonical modern example
([Rimsky et al., 2024](#ref-rimsky-2024)). The field also includes adaptive,
feature-targeted, and geometry-preserving variants, which this note does not
test. The added vector can encode a proposed semantic direction, such as a
sparse-autoencoder feature, or it can be deliberately generic, as in the
experiment reported here.

A Jacobian is a local derivative: the high-dimensional analogue of a tangent
line. A {{< refterm "jacobian-lens" "Jacobian lens" >}}
([Gurnee et al., 2026](#ref-gurnee-2026)) turns that idea into a reusable
reference by averaging input-output Jacobians over many prompts and token
positions. The result is one fixed linear map for each source layer: a
corpus-average first-order approximation, not the exact Jacobian of the prompt
under study. Gurnee et al. built the lens to surface verbalizable,
vocabulary-oriented content; this note repurposes it as a transport benchmark
for arbitrary residual perturbations, a different job from the one it was fit
for. Whether averaged first-order maps carry useful structure is itself
setting-dependent: relation-decoding work finds that mean Jacobian-based affine
maps decode some relations well and others poorly
([Hernandez et al., 2024](#ref-hernandez-2024)).

This distinction matters because a downstream change can have several causes.
The requested intervention may not land accurately in low-precision arithmetic.
A well-delivered edit may propagate as the linear map predicts. Or the model's
remaining nonlinear blocks may reshape it. Before attributing a downstream
effect to the meaning of a steering direction, these possibilities need to be
separated.

### Prior work, contribution, and non-claims

**Prior work.** Gurnee et al. introduce the Jacobian lens and connect mid-layer
directions to later residual and vocabulary geometry
([2026](#ref-gurnee-2026)). Public fitting tooling and Llama maps come from
Anthropic's reference implementation ([2026](#ref-anthropic-jlens)) and
Neuronpedia's Llama release ([2026](#ref-neuronpedia-jlens)). Recent dose and
steering work reports that intervention strength need not grow effects
monotonically ([Taimeskhanov, Vaiter, and Garreau, 2026](#ref-taimeskhanov-2026))
and that stronger SAE refusal steering can coincide with collateral
over-refusal ([O'Brien et al., 2025](#ref-obrien-2025)). An earlier Praxagent
note asks whether SAE steering leaves a detectable J-space fingerprint under
matched access models ([Jones, 2026](#ref-sae-steering)). This note does not
invent the lens or the general fact that neural networks are nonlinear.

**This note's contribution.**

- a prospectively frozen, target-blind signed dose scan on Llama 3.3 70B that
  retains requested edit, BF16-realized edit, fixed-J reference, and actual
  state separately at every dose and depth;
- a complete 24-cell census on the frozen `2%`/`3%`/`4%` linearity panel, with
  identity and five random-J controls on the primary predictive comparison;
- a public ledger in
  [`tdj28/llm_selfref_pre`](https://github.com/tdj28/llm_selfref_pre/tree/main/experiments/consciousness_sae_signed_dose_scan)
  with freeze and audited-release commits separated.

**Not claimed.**

- that SAE decoder directions, as a class, are nonlinear downstream;
- that any specific SAE feature has its published semantic interpretation;
- that generated behavior, deception, or consciousness-related vocabulary
  changed;
- that the corpus-average J outperforms simply carrying the residual change
  forward (it did not clear that frozen margin here).

## Design in Brief

We tested this in Llama 3.3 70B Instruct. Immediately after transformer block
50, we applied three unselected stress-test directions to eight prompts, varied
their magnitude from `0.5%` to `30%` of {{< refterm "rms" "residual RMS" >}},
and followed the response through the remaining 29 blocks. The directions were
independent Gaussian vectors normalized to unit {{< refterm "rms" "RMS" >}},
fixed without semantic or SAE-based selection. They test the mechanics of
perturbation transport rather than any concept.

Each direction was applied at positive and negative dose. Taking half their
difference isolates the intervention-aligned response (the part that reverses
with sign), while the midpoint relative to clean is retained as a common-mode
diagnostic. The paired signs form one curve, giving 24 prompt × direction
curves rather than 48 independent observations. Exact construction,
provenance, dose grid, and pairing equations are in
[Appendix A](#appendix-a-exact-test).

At every dose, the analysis retained four quantities: the edit *requested* in
high precision; the edit *realized* after casting into
{{< refterm "bf16" "BF16" >}} activations (see
[requested-versus-realized fidelity](#requested-versus-realized-edit-fidelity-under-bf16));
the downstream change from the fixed Jacobian map; and the downstream change
the model actually produced. Separating requested from realized keeps
low-precision rounding from being mistaken for nonlinear model dynamics.

The test fixed this comparison and its pass/fail criteria before any outcomes
were examined, making the result falsifiable without pretending to measure a
field-wide prior. Once a small edit landed faithfully, a local-linear account
would survive the `2%`/`3%`/`4%` diagnostic only if the dose-normalized final
response retained nearly the same direction and scale. The design did not
assume either outcome; it decided in advance what passing and failing would
mean.

## The Linear Edit and the Nonlinear Wake

### What “linear” means in this note

Imagine a dimmer switch. If the response is **dose-linear**, turning the dose
from `2%` to `4%` should roughly double the size of the change while leaving
its direction alone. In vector language: the hidden-state response at dose
\(b\) should look like \(b\cdot v\) for one fixed direction \(v\).

**Nonlinear**, here, means that rule fails in a measured way: as dose changes,
the response rotates (direction drifts) or its size stops tracking the dose
(amplification or compression). We are not using “nonlinear” as a vibe word for
“the network has nonlinear layers.” Transformers are nonlinear by construction.
The question is whether *this* dose–response curve stayed close enough to a
straight line through the origin on a small, prechosen panel.

A **cell** is one prompt paired with one generic direction: 8 prompts × 3
directions = 24 cells. For each cell we look at the central signed contrast
across dose (plus minus minus, halved), so the 24 cells are 24 curves, not 48
independent signed runs.

We apply the same linearity test to three quantities in each cell:

1. **Source:** the edit that actually landed after block 50.
2. **Fixed J:** that same realized edit pushed through the corpus-average
   Jacobian map (an algebraic straight-line reference).
3. **Actual final:** the model’s real residual change after block 79.

### How we scored pass versus fail

The frozen panel is \(B=\{0.02,0.03,0.04\}\) (`2%`, `3%`, `4%`), with `3%` as
the anchor. That panel is the first three-dose window where every source edit
already cleared the requested-versus-realized fidelity checks; Figure 2 shows
that boundary.

For a response vector \(x_b\) at dose \(b\), form the dose-normalized vector

\[
q_b = \frac{x_b}{s_b}.
\]

For the source edit, \(s_b\) is the requested dose fraction. For J and the
actual final state, \(s_b\) is the {{< refterm "rms" "RMS" >}} fraction of the
realized source edit (so we ask whether the downstream map preserves the
dose-normalized shape of what landed). If the response were exactly proportional
to dose, every \(q_b\) would be the same vector.

Two statistics then ask whether the \(q_b\) stay near the `3%` anchor
\(q_{0.03}\):

\[
c_{\min}
=
\min_{b\in B}\cos(q_b,q_{0.03}),
\qquad
d_{\max}
=
\max_{b\in B}
\frac{\operatorname{RMS}(q_b-q_{0.03})}
{\operatorname{RMS}(q_{0.03})}.
\]

- \(c_{\min}\) is directional agreement (1 means identical direction).
- \(d_{\max}\) is relative size mismatch after dose normalization (0 means
  identical scale and shape). The implementation calls \(d_{\max}\) “slope
  discrepancy.”

A cell **passes** for a quantity when both frozen gates hold:

\[
c_{\min}\ge 0.95
\quad\text{and}\quad
d_{\max}\le 0.15.
\]

Fail either gate and that quantity is scored **nonlinear** on this panel.
Exact construction of \(x_b\) (signed branches, common-mode diagnostic) is in
[Appendix A](#appendix-a-exact-test).

```python
import numpy as np

B = (0.02, 0.03, 0.04)
ANCHOR = 0.03

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def dose_linearity_gates(
    x_by_dose: dict[float, np.ndarray],
    s_by_dose: dict[float, float],
    *,
    cosine_min: float = 0.95,
    slope_max: float = 0.15,
) -> tuple[float, float, str]:
    """Return (c_min, d_max, 'pass'|'fail') for one cell and one quantity."""
    q = {b: x_by_dose[b] / s_by_dose[b] for b in B}
    q_anchor = q[ANCHOR]
    c_min = min(cosine(q[b], q_anchor) for b in B)
    d_max = max(rms(q[b] - q_anchor) / rms(q_anchor) for b in B)
    status = "pass" if c_min >= cosine_min and d_max <= slope_max else "fail"
    return c_min, d_max, status

# Example call for one prompt × direction cell:
# c_min, d_max, status = dose_linearity_gates(source_x, source_s)
```

{{< panel "info" >}}
**Why normalize by dose first?** Without dividing by \(s_b\), a perfectly
linear response \(x_b=b\,v\) would still change length across the panel, and a
naive cosine between raw \(x_{0.02}\) and \(x_{0.04}\) would not isolate the
failure mode we care about. Dose normalization asks whether the *per-unit-dose*
vector stayed put.
{{< /panel >}}

### What the census found

**Against that frozen first-order benchmark, the split was complete: 24 of 24
source edits passed both gates, 24 of 24 fixed-J references also passed, and
0 of 24 actual final-state responses did.**

| Quantity | \(c_{\min}\) (need \(\ge 0.95\)) | \(d_{\max}\) (need \(\le 0.15\)) | Cells passing |
|---|---:|---:|---:|
| Realized source edit | `0.995`–`0.996` | `0.089`–`0.101` | 24 / 24 |
| Fixed J of that edit | `0.993`–`0.996` | `0.090`–`0.117` | 24 / 24 |
| Actual final state | `0.778`–`0.848` | `0.572`–`0.703` | 0 / 24 |

So the finals did not fail by a hair on one metric. In every cell they failed
**both** rules: the dose-normalized response rotated away from the `3%` anchor
(cosine well below `0.95`) *and* its per-unit-dose size/shape drifted far past
the `0.15` ceiling. Source and fixed-J curves, by contrast, sat near cosine
`1` with only about `0.1` relative discrepancy.

In plain English: at the intervention site the delivered edit behaved like a
dimmer. After 29 more blocks, turning the dose from `2%` to `4%` no longer
gave “same arrow, twice as long.” The arrow bent, and the length stopped
tracking dose. The fixed Jacobian cannot invent that failure by itself: a
fixed linear map would preserve exact proportionality, and here the
approximately proportional source curves also stayed inside both numerical
bounds after projection. The curvature therefore sits in the model’s mapping
from the realized post-block-50 edit to the post-block-79 residual, not in
dose-dependent mangling of the source edit. Fixed J is an algebraic and
implementation control here; it is not evidence that J predicted the model
accurately. Predictive accuracy is tested separately in Figure 4.

<p class="figure-note">Table: frozen-panel linearity census ranges across the 24 cells. Provenance: <a href="fig-1-linearity-census.receipt.json">fig-1-linearity-census.receipt.json</a> · <a href="provenance.json"><code>provenance.json</code></a>.</p>

Figure 1 shows the complete census rather than an average. Its three horizontal
categories are the realized source edit, J applied to that realized edit, and
the actual final state. The left panel plots \(c_{\min}\) against the `0.95`
floor; the right plots \(d_{\max}\) against the `0.15` ceiling. Each connecting
line is one prespecified prompt × direction cell.

![Two paired dot plots show all 24 frozen prompt-by-direction cells over the 2%, 3%, and 4% dose panel, using 3% as the anchor. In the left plot, minimum cosine to the 3% anchor clusters between 0.993 and 0.996 for the realized source edit and J of the realized edit, above the 0.95 threshold, then drops to 0.778–0.848 for the actual final state, below threshold in all 24 cells. In the right plot, maximum dose-normalized RMS discrepancy from the 3% anchor is 0.089–0.117 for the first two quantities, below the 0.15 ceiling, but 0.572–0.703 for the actual final state, above threshold in every cell. Faint gray lines connect the same cell across the three quantities. Exact proportionality is preserved by a fixed linear map; here the approximately proportional delivered edits also remain within both numerical bounds after projection, while the downstream model response does not.](fig-1-linearity-census.svg)

<p class="figure-note">Figure 1: source linearity versus actual downstream curvature. Each line is one of 24 prompt × direction central-contrast curves over the prespecified <code>2%</code>/<code>3%</code>/<code>4%</code> panel. In these data, the fixed linear operator left the near-proportional delivered curves within both numerical bounds; that panel is a consistency reference, not a test of predictive accuracy. The model's actual final hidden states satisfy neither criterion in any cell. Dashed lines mark thresholds fixed before outcomes. Provenance: <a href="fig-1-linearity-census.receipt.json">receipt</a> · <a href="plot_signed_dose_scan_results.py"><code>plot_signed_dose_scan_results.py</code></a> · <a href="provenance.json"><code>provenance.json</code></a>. Verify (needs the on-request audited summary): <code>python3 plot_signed_dose_scan_results.py --summary CALIBRATION_SUMMARY.json --output-dir . --verify --post index.md</code>.</p>

## Requested-versus-realized edit fidelity under BF16

“Delivery” is **not** a standard ML term. It is this study’s short name for a
simple fidelity check: does the intervention vector we *requested* match the
vector that was *realized* in the residual after casting into the model’s
number format?

That format here is {{< refterm "bf16" "BF16" >}} (bfloat16): a 16-bit
floating-point type with FP32-like range but much coarser precision, so many
float32 values round to the same BF16 number. For bit layouts, casting demos,
and accumulation swamping, see the Deep Dive
[BF16, FP16, and FP32: Precision, Range, Swamping, and Determinism]({{< relref "knowledge-base/deep-dives/bf16-fp16-fp32/index.md" >}}).
The protocol builds the requested edit in float32 along a
unit-{{< refterm "rms" "RMS" >}} direction, then casts it into the BF16
residual hook. Write \(e^{\mathrm{req}}\) for the intended vector and
\(e^{\mathrm{real}}\) for the change actually present after the hook. If
rounding were negligible, those two would nearly match.

We score that match with the same two geometric ideas as linearity, now
comparing request to realization at a single dose:

\[
\cos\!\big(e^{\mathrm{req}},e^{\mathrm{real}}\big)
\ge 0.995,
\qquad
\frac{\operatorname{RMS}\big(e^{\mathrm{real}}-e^{\mathrm{req}}\big)}
{\operatorname{RMS}\big(e^{\mathrm{req}}\big)}
\le 0.10.
\]

The full gate also checks both signed branches, their central contrast, and the
common-mode response. Fail any piece and that cell fails the fidelity check at
that dose.

```python
# Uses the same rms() / cosine helpers as the linearity sketch above.
def requested_realized_ok(
    e_req, e_real, *, cos_min=0.995, rel_rmse_max=0.10
) -> bool:
    return (
        cosine(e_req, e_real) >= cos_min
        and rms(e_real - e_req) / rms(e_req) <= rel_rmse_max
    )
```

Why this matters: if a tiny requested dose mostly rounds away, a “nonlinear”
downstream curve could just be noise from a mangled source edit. The fidelity
check asks whether the source edit is faithful enough to blame the model for
what happens next.

### What we observed

At very small magnitudes, that rounding is not negligible. In this exact model,
hook, prompt panel, and set of generic directions, the three smallest dose
levels did not meet the predefined fidelity bounds. All 24 cells failed the
full paired-branch rule at `0.5%` and `1%`, and 18 of 24 did so at `1.5%`.
Every cell passed from `2%` through `30%`.

At `2%`, requested-versus-realized cosine had a median of `0.997`, while median
relative RMSE was `0.076`. Passing does not mean requested and realized edits
were bit-identical. It does show that the final-state nonlinearity on the
`2%`/`3%`/`4%` panel cannot be blamed on the low-dose fidelity failure seen
below `2%`. The boundary is consistent with BF16 rounding, but the experiment does not
isolate rounding from other implementation details such as the casting path,
kernel ordering, or fused operations. It is an empirical property of this
implementation and panel, not a universal BF16 limit; a datatype and hook-path
ablation (FP32, FP16, alternative hook implementations) would be the direct
mechanism test.

![Three aligned plots show the low-dose requested-versus-realized fidelity transition under BF16 execution in this model and intervention setup, across requested doses from 0.5% to 30%. Directional cosine rises sharply from a median 0.961 at 0.5% toward 1.0 and clears its 0.995 threshold by the eligible panel. Relative RMSE falls from a median 0.287 at 0.5% through its 0.10 ceiling near 1.5–2% and continues toward zero. Pale bands show the minimum-to-maximum range across the fixed 24-cell census, not uncertainty. The bottom strip reports the full paired-branch fidelity rule: 24 failed cells at 0.5%, 24 at 1%, 18 at 1.5%, and zero from 2% through 30%.](fig-3-bf16-delivery-floor.svg)

<p class="figure-note">Figure 2: low-dose requested-versus-realized fidelity under this BF16 setup (file stem <code>fig-3-bf16-delivery-floor</code>; “delivery” is the protocol nickname). The upper curves show the central signed estimate; the bottom strip applies the complete rule to both signed branches, their central contrast, and their common-mode response. Pale envelopes are the observed minimum and maximum across the fixed census (all 24 prespecified prompt × direction cells), not confidence intervals. Provenance: <a href="fig-3-bf16-delivery-floor.receipt.json">receipt</a> · <a href="plot_signed_dose_scan_results.py"><code>plot_signed_dose_scan_results.py</code></a> · <a href="provenance.json"><code>provenance.json</code></a>.</p>

## Watch the Wake Develop Across Depth

At each state, gain is the root-mean-square magnitude of the central signed
response, \((h_{+}-h_{-})/2\), divided by the RMS magnitude of the realized
central source edit. It is `1x` at state 50 by construction. A gain of `1.8x`,
for example, means the downstream central response is 80% larger in RMS than
the edit that landed. Median gain generally accumulated through later blocks,
although individual trajectories were not uniformly monotone. Every tested
trajectory ended above `1x`.

The median final-state gain was `1.82x` at `2%`, `1.66x` at `3%`, `1.60x` at
`4%`, `1.55x` at `8%`, and `1.48x` at `30%`. Relative amplification was
generally larger at lower eligible doses (doses where every cell passed the
requested-versus-realized checks), though the complete half-point dose series
was not perfectly monotone.

Figure 3 makes the depth and dose axes explicit. The upper heatmap runs dose
horizontally and post-block state vertically; the lower panel shows final-state
gain across dose for every one of the 24 cells, with their median emphasized.

![A two-panel dose-by-depth view of the complete downstream response. The upper heatmap places requested dose from 0.5% to 30% on the horizontal axis and states 50 through 79 on the vertical axis. Every state-50 cell begins at gain 1 because gain is normalized to the realized edit. Median gain generally accumulates in later blocks, reaching its largest values at the smallest doses and generally tapering as dose rises. A dashed vertical line marks 2%, the first dose where every cell met the frozen delivery criteria. The lower panel shows all 24 final-state trajectories as pale lines and their median as a dark line. The median is 1.82 times at 2%, 1.66 at 3%, 1.60 at 4%, 1.55 at 8%, and 1.48 at 30%. The 0.5–1.5% region is shaded because its delivery gate failed.](fig-2-dose-depth-arc.svg)

<p class="figure-note">Figure 3: the measured hidden-state response from the intervention point to the final block (file stem <code>fig-2-dose-depth-arc</code>). The heatmap uses fixed-panel medians (the median across the same prespecified 24 cells at each dose and layer); the lower panel retains every trajectory. These are descriptive census summaries of exactly these 24 cells, not population estimates. Provenance: <a href="fig-2-dose-depth-arc.receipt.json">receipt</a> · <a href="plot_signed_dose_scan_results.py"><code>plot_signed_dose_scan_results.py</code></a> · <a href="provenance.json"><code>provenance.json</code></a>.</p>

## Did the Jacobian Lens Predict the Wake?

Cross-dose curvature and predictive usefulness at one dose are logically
independent: a response can bend as dose changes while a first-order map still
captures useful structure at a fixed dose. Figure 1 used J only as a linear
algebra reference. Predictive accuracy requires a different test: compare J's
prediction with the final response and ask whether it beats cheap alternatives.
We used two: identity, which carries the observed source-state change forward
unchanged (a geometry-free transport baseline, not a vocabulary readout such
as a {{< refterm "logit-lens" "logit lens" >}} or tuned lens
([Belrose et al., 2023](#ref-belrose-2023))), and five prespecified
{{< refterm "random-j" "random-J" >}} controls, which apply seeded sign and
coordinate permutations to the released map.

The layer-50 source state was selected in advance for the primary comparison.
On this fixed `3%` panel, the released J cleared the absolute thresholds and
decisively beat the strongest of five random maps in both metrics. For
fixed-token logit correlation (agreement between predicted and observed score
changes for 2,048 prespecified vocabulary tokens), real J scored `0.344`; its
advantage over the strongest random map was `0.286`, with a 95% fixed-panel
stability lower bound of `0.271`. In residual space, the corresponding advantage
was `0.337`, with a lower bound of `0.321`.

Identity was the harder baseline. J's logit advantage over identity was only
`0.011`, with a lower bound of `-0.0015`, below the predefined `0.02` margin.
Its residual-space advantage was slightly negative (`-0.0015`). No tested
source-state layer cleared the added-value-over-identity margin in both spaces.
The bounded conclusion is therefore precise: the released map contains
structure beyond these five randomizations, but it did not show added value
over simply carrying the residual change forward unchanged.

This comparison cannot separate two explanations for that null: that
first-order transport itself adds little for these directions, or that corpus
averaging washes out prompt-local structure that a per-prompt Jacobian would
capture. A prompt-specific Jacobian baseline, run on the same cells, is the
natural next control and was not part of this study.

For Figure 4, each horizontal position has a different source-state layer but
the same target. At source layer \(\ell\), both \(J_\ell\) and identity receive
the observed signed residual change after block \(\ell\) and predict the
post-block-79 change. Only layer 50 is the intervention site. Layers 51–78 are
shorter-horizon diagnostics after the model has already transformed the edit.

![Four profiles compare predictions from source-state layers 50 through 78 to the same post-block-79 target. The left column shows that absolute released-J fixed-token logit correlation and residual cosine both exceed their frozen thresholds at every source layer and rise as the source approaches the target. The right column shows released J minus identity. Logit advantage generally rises above 0.02 after early layers, but residual advantage remains below the 0.02 margin at 28 of 29 source layers. Its sole residual pass at layer 78 does not coincide with a logit pass. At source layer 50, selected in advance for the primary comparison, logit correlation is 0.344 but J minus identity is only 0.011 with a lower bound below zero; residual J minus identity is slightly negative. No source layer satisfies the identity condition for both metrics. Shaded bands are 20,000-replicate prompt-resampling stability intervals over eight fixed prompts, not population confidence intervals.](fig-4-j-versus-identity.svg)

<p class="figure-note">Figure 4: prediction from successive source states to one fixed target. For each source layer \(\ell=50,\ldots,78\), \(J_\ell\) and identity receive the observed central residual change after block \(\ell\) and are compared with the same post-block-79 target. Layer 50 was selected in advance for the primary comparison; layers 51–78 are descriptive shorter-horizon diagnostics, not additional intervention sites. Bands are fixed-panel stability intervals obtained by resampling the eight prompts and do not support population-level inference. Provenance: <a href="fig-4-j-versus-identity.receipt.json">receipt</a> · <a href="plot_signed_dose_scan_results.py"><code>plot_signed_dose_scan_results.py</code></a> · <a href="provenance.json"><code>provenance.json</code></a>.</p>

## Implications for SAE Intervention Studies

The measurement design transfers to an SAE-specific study; the dose boundary
does not. A direct extension would replace the generic vectors with target SAE,
matched-SAE, and norm-matched generic directions, then rerun the
requested-versus-realized fidelity and common-mode checks for every vector.
The `2%`/`3%`/`4%` panel is only a
candidate calibration starting point. Any semantic or behavioral extension
would also need its own controls and never-intervened counterfactual branch.
The generic-direction anomaly motivates that experiment; it does not confirm
an SAE mechanism. The target features, prompts, dose panel, semantic endpoints,
and analysis rules would need to be specified before their outcomes are
inspected.

Two cheaper follow-ups would sharpen the present result before any semantic
extension: the datatype and hook-path ablation named above, which would turn
"consistent with BF16" into a mechanism test, and a small public notebook that
exports the requested edit, realized edit, fixed-J projection, and actual
final residual for one prompt, one direction, and one layer at each dose, so
the study's central decomposition can be audited in minutes rather than hours.

## Interpretation

Five conclusions follow from the experiment:

- this setup has an observed low-dose requested-versus-realized fidelity
  boundary under BF16 casting, with universal passage beginning at `2%` for
  the tested generic directions;
- above that boundary, the realized edit remains dose-linear over the primary
  `2%`/`3%`/`4%` panel, and the approximately proportional curves empirically
  remain within both bounds after fixed-J projection;
- the actual final residual state changes direction and scale enough to fall
  outside both predefined linearity bounds in every cell;
- the full arc shows a dose-dependent transformation generally accumulating
  through the remaining blocks; and
- the released J beats five randomized maps but does not clear the stronger
  added-value-over-identity rule at any tested source-state layer.

These conclusions apply to the 24 tested prompt × direction combinations and to
hidden-state dynamics. The experiment did not test generated behavior, SAE
features, deception, or consciousness-related vocabulary.

The depth arc localizes where the first-order description stops being adequate;
it does not identify the mechanism that bends the trajectory. Mechanistic
explanations suggested by this pattern are hypotheses for a new study whose
mechanism test is specified before data inspection.

Taken together, the result is both sharp and bounded: **linear edit in,
nonlinear hidden-state wake out.** The corpus-average Jacobian sees nonrandom
structure in that wake, but it does not outperform the simplest strong
baseline: carrying the residual change forward unchanged.

## Reproducibility And Artifact Ledger

Compact map for readers who already know what they want. Sample records and the
teaching inventory are in the [appendix](#appendix-release-inventory).

| Artifact | Link |
|---|---|
| Experiment package | [`experiments/consciousness_sae_signed_dose_scan`](https://github.com/tdj28/llm_selfref_pre/tree/main/experiments/consciousness_sae_signed_dose_scan) |
| Pre-outcome freeze | [`a084caa`](https://github.com/tdj28/llm_selfref_pre/commit/a084caafc2ec27860044d80d3b33912f656fd08a) |
| Audited result release | [`fde24e9`](https://github.com/tdj28/llm_selfref_pre/commit/fde24e93770859bf0ec848b91eb0564d550a641d) |
| Frozen plan directory | [`dose_scan_v1_plan_20260716`](https://github.com/tdj28/llm_selfref_pre/tree/a084caafc2ec27860044d80d3b33912f656fd08a/data/consciousness_sae_signed_dose_scan/dose_scan_v1_plan_20260716) |
| Compact replication record | [`…/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9`](https://github.com/tdj28/llm_selfref_pre/tree/fde24e93770859bf0ec848b91eb0564d550a641d/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9) |
| Protocol | [`docs/…/PROTOCOL.md`](https://github.com/tdj28/llm_selfref_pre/blob/fde24e93770859bf0ec848b91eb0564d550a641d/docs/consciousness_sae_signed_dose_scan/PROTOCOL.md) |
| Result summary | [`RESULT_SUMMARY.md`](https://github.com/tdj28/llm_selfref_pre/blob/fde24e93770859bf0ec848b91eb0564d550a641d/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9/RESULT_SUMMARY.md) |
| Figure generator (this post bundle) | [`plot_signed_dose_scan_results.py`](plot_signed_dose_scan_results.py) (SHA-256 `480c4b9e…`; committed upstream with the figure package at [`f5e906e`](https://github.com/tdj28/llm_selfref_pre/tree/f5e906e1737bc71bf20b642af1d698018eec82fe/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9/figures)) |
| Post-wide number manifest | [`provenance.json`](provenance.json) |
| Llama 3.3 70B Instruct | revision [`6f6073b…`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/6f6073b423013f6a7d4d9f39144961bfbfbc386b/README.md) (gated) |
| Neuronpedia J-lens config | revision [`a4114d7…`](https://huggingface.co/neuronpedia/jacobian-lens/blob/a4114d7752d11eb546e6cf372213d7e75526d3a1/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml) |

<p class="figure-note">Table: compact ledger. Freeze and release commits are separated on purpose. The 2.23&nbsp;GB raw residual tree stays off-repository; its inventory SHA-256 is in the appendix.</p>

---

## Technical Appendices

<h3 id="appendix-release-inventory">Appendix: release inventory</h3>

{{< panel "warning" >}}
**Study status: complete.** The design was frozen in git
([`a084caa`](https://github.com/tdj28/llm_selfref_pre/commit/a084caafc2ec27860044d80d3b33912f656fd08a))
*before* outcomes. The audited public package is
[`fde24e9`](https://github.com/tdj28/llm_selfref_pre/commit/fde24e93770859bf0ec848b91eb0564d550a641d)
in [`tdj28/llm_selfref_pre`](https://github.com/tdj28/llm_selfref_pre). C9
recomputed the audit from the unchanged raw tree with zero fresh model
forwards.
{{< /panel >}}

| What we shipped | In plain English | For specialists | License / access |
|---|---|---|---|
| Experiment code and gates | Scripts that build, validate, run, and audit the scan | [`experiments/consciousness_sae_signed_dose_scan`](https://github.com/tdj28/llm_selfref_pre/tree/main/experiments/consciousness_sae_signed_dose_scan) | open (repo license) |
| Frozen plan | The pre-outcome protocol bytes | [`dose_scan_v1_plan_20260716`](https://github.com/tdj28/llm_selfref_pre/tree/a084caafc2ec27860044d80d3b33912f656fd08a/data/consciousness_sae_signed_dose_scan/dose_scan_v1_plan_20260716); plan manifest field `plan_manifest_sha256` `79810742…63c51d` | open |
| Compact audit package | What a reader needs without the raw tensors | [`…/audit-recovery-c9`](https://github.com/tdj28/llm_selfref_pre/tree/fde24e93770859bf0ec848b91eb0564d550a641d/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9); audit status `pass` | open |
| Audited summary (26.8 MB) | Machine-readable census behind the figures | SHA-256 `b490b101c112f774ae7bffc9c54294a70b91c874c4cee78eddaa7890446c08f6` | on-request (too large for git; bound by `PUBLICATION_COMPLETE.json`) |
| Raw residual tree (2.23 GB) | Full archived arcs for independent replay | inventory SHA-256 `9fc8c8ffe5f8d34f8ddba863c11fff7370ef3644c1fad0139f96e39a4c8fbfc0` | withheld / on-request (volume retention; not in git) |
| Empirical figures | The four plots in this note | per-figure `.receipt.json` in this post bundle; index [`figure-receipts-index.json`](figure-receipts-index.json) | open (bundled here) |
| Llama 3.3 70B Instruct | Base weights for the run | revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b` | gated (Llama Community License) |
| Neuronpedia J-lens maps | Corpus-average \(J_\ell\) used as reference | revision `a4114d7752d11eb546e6cf372213d7e75526d3a1` | publicly released (see HF card) |

<p class="figure-note">Table: what "complete" means for this release. Middle column is plain language; right columns carry specialist pins and access conditions.</p>

### Open a record: compact publication marker

**Plain English.** The public package seals that the audit finished from the
unchanged raw tree and records the hashes a skeptic should match.

**Technical.** `PUBLICATION_COMPLETE.json` in the compact replication record:
`status` = `complete_atomic_audit_only_recovery`;
`summary_file_sha256` = `b490b101…08f6`;
`audit_receipt_sha256` = `176e2f7e…ee1`;
run id `signed-dose-a084caa-wl8obvtuq0ax8t-v2`.

### Open a record: linearity census receipt

**Plain English.** Figure 1's numbers are not typed by hand. A receipt lists
every plotted range and failure count.

**Technical.** [`fig-1-linearity-census.receipt.json`](fig-1-linearity-census.receipt.json):
`actual_final_failure_count` = 24;
`realized_source_failure_count` = 0;
`j_of_realized_failure_count` = 0;
source summary SHA-256 `b490b101…08f6`.

<h3 id="appendix-a-exact-test">Appendix A: Exact Test</h3>

The experiment used [Meta's Llama 3.3 70B Instruct
checkpoint](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/6f6073b423013f6a7d4d9f39144961bfbfbc386b/README.md)
at revision `6f6073b…` and [Neuronpedia's Llama-specific Jacobian-lens
release](https://huggingface.co/neuronpedia/jacobian-lens/blob/a4114d7752d11eb546e6cf372213d7e75526d3a1/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml)
at revision `a4114d7…`. Neuronpedia generated the maps with [Anthropic's
reference method](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e),
averaging derivatives over 125 completed WikiText prompts of at most 128
tokens. They were not estimated from this study's prompts or outcomes.

The three stress-test directions were independent standard-normal vectors in
the 8,192-dimensional residual space, normalized to unit
{{< refterm "rms" "RMS" >}} and fixed by predeclared seeds. There was no
semantic or SAE-based selection, rejection for outcome behavior, or post-hoc
sign flip. Dose ran from `0.5%` to `30%` of {{< refterm "rms" "residual RMS" >}}
in `0.5`-percentage-point increments. The prompts, directions,
primary dose panel, and decision rules were fixed before outcomes were opened.

For prompt \(p\), direction \(v\), and magnitude \(b\), the two signed branches
produce the realized central source edit

\[
e_b = \frac{h^{\mathrm{post50}}_{+b}-h^{\mathrm{post50}}_{-b}}{2}.
\]

The midpoint relative to the clean branch was retained as a separate
common-mode diagnostic. The two signed branches therefore form one paired
measurement. A linearity cell is one prompt × direction curve across dose,
giving 24 cells rather than 48.

For a realized layer-50 edit \(e_b\), the fixed lens supplies

\[
\Delta h_{79}^{J} = J_{50}\,e_b.
\]

The released \(J_\ell\) is an `8192 × 8192` corpus- and position-averaged
input-output Jacobian from the post-block-\(\ell\) residual to the post-block-79
residual. It has no prompt-specific conditioning, intercept, or centering term.
The row-vector implementation is \(e_bJ_{50}^{\mathsf T}\). Because the same
linear operator is used at every dose, exact source proportionality would be
preserved automatically. The realized curves were only approximately
proportional, so remaining within the numerical bounds after J projection is an
empirical consistency check. Figure 4 separately evaluates prediction against
the observed target state and against identity and random-map controls.

{{< panel "info" >}}
**Notation.** \(J_{50}\) is a matrix multiply, not a nonlinear function of the
prompt. Exact proportionality of source edits would be preserved by any fixed
linear map; the interesting failure is the model's actual final state.
{{< /panel >}}

The prespecified primary panel (the doses and decision rules committed before the
results were examined) compared the dose set
\(B=\{0.02,0.03,0.04\}\), with `3%` as the anchor. For a response vector
\(x_b\), define the dose-normalized vector \(q_b=x_b/s_b\). For the realized
source, \(s_b\) is the requested BF16 edit's residual-RMS fraction; for J and
the actual final response, \(s_b\) is the realized source edit's residual-RMS
fraction. The two implemented statistics were

\[
c_{\min}=\min_{b\in B}\cos(q_b,q_{0.03}),
\qquad
d_{\max}=\max_{b\in B}
\frac{\operatorname{RMS}(q_b-q_{0.03})}
{\operatorname{RMS}(q_{0.03})}.
\]

Each prompt-direction cell required \(c_{\min}\ge 0.95\) (the dose-normalized
vector kept nearly the same direction as the `3%` anchor) and
\(d_{\max}\le 0.15\) (its component-wise RMS deviation from the anchor stayed
small). These are vector-valued dose-response checks; "slope discrepancy" is
the implementation's name for \(d_{\max}\).

Realized-source and J-projected curves had minimum cosine between `0.993` and
`0.996` and maximum slope discrepancy between `0.089` and `0.117`. Actual-final
curves had minimum cosine between `0.778` and `0.848` and maximum slope
discrepancy between `0.572` and `0.703`.

Delivery used stricter source-fidelity criteria (checks on whether the edit
that landed matched the edit requested): requested-versus-realized cosine at
least `0.995` and relative RMSE at most `0.10`. The primary panel was
prospectively defined (written into the plan before outcomes were opened) and
begins at the first dose where all 24 cells met both requested-versus-realized
criteria.

Figure 4's bands are 20,000-replicate prompt-resampling stability intervals
(repeated reweightings of the same eight prompts to show sensitivity within
this panel). They are not confidence intervals for a broader prompt population.
Each Figure 4 map takes a different post-block source state \(\ell=50,\ldots,78\)
to the same post-block-79 target. Only block 50 is the intervention site.
Figure 2's and Figure 3's ranges are descriptive minima, maxima, or medians over
the complete fixed census and contain no inferential uncertainty (they report
the tested cases rather than estimate unseen ones).

<h3 id="appendix-b-pilot-study">Appendix B: Pilot Study and Motivation for the Signed-Dose Scan</h3>

An earlier target-blind pilot established that the released maps were loaded
and oriented correctly and that the clean readout could distinguish its
prespecified semantic and Yes/No controls. It did not meet the prespecified
generic-intervention criterion, so it did not support a claim that J
outperformed identity. More importantly for the present design, that pilot did
not preserve the edit that BF16 arithmetic actually realized.
Requested-versus-realized error and downstream curvature therefore remained
confounded.

The signed-dose scan was built to resolve that one ambiguity by retaining the
requested edit, realized edit, J reference, and actual state separately at every
dose. It does not retroactively change the pilot's threshold result. The full
gate table, statistics, hashes, and execution history remain in the [pilot
replication record](https://github.com/tdj28/llm_selfref_pre/tree/main/data/consciousness_readout_validation/pilot_v1_result_20260714_r15).

<h3 id="appendix-c-scope-and-prior-work">Appendix C: Scope Detail</h3>

Neither completed study measured downstream dose response for an SAE decoder
direction (the vector associated with one learned sparse-autoencoder feature).
The signed-dose result is evidence of nonlinear downstream model
response to these generic residual directions. It is not evidence that SAE
interventions as a class are nonlinear, that a specific SAE feature has its
published interpretation, or that consciousness-related vocabulary changed.

The contribution of this experiment is not the general observation that neural
networks are nonlinear. It is the point-by-point separation of requested edit,
realized edit, a fixed corpus-average J reference, and actual state across a
dose-by-depth scan whose primary panel and decision rules were set before the
outcomes were examined. Literature that motivates a dose scan or collateral-effect
measurement is cited in the early prior-work block and listed below; those
papers are study-specific reports, not evidence for the result here.

<h3 id="appendix-d-audit-and-provenance">Appendix D: Audit and Provenance</h3>

The replication record identifies the study as
`consciousness_sae_signed_dose_scan_v1` and the model run as
`signed-dose-a084caa-wl8obvtuq0ax8t-v2`. Its independent audit status is
`pass`; the audit was recomputed from the unchanged raw-data tree without new
model inference. The 2.23 GB raw dataset remains off-repository. Its complete
file-inventory SHA-256 is
`9fc8c8ffe5f8d34f8ddba863c11fff7370ef3644c1fad0139f96e39a4c8fbfc0`,
and the audited-summary SHA-256 is
`b490b101c112f774ae7bffc9c54294a70b91c874c4cee78eddaa7890446c08f6`.
The [compact replication
record](https://github.com/tdj28/llm_selfref_pre/tree/fde24e93770859bf0ec848b91eb0564d550a641d/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9)
contains the protocol pointers, result summary, audit receipts, and, as of
commit
[`f5e906e`](https://github.com/tdj28/llm_selfref_pre/commit/f5e906e1737bc71bf20b642af1d698018eec82fe),
the committed figure package.

All four empirical figures were generated programmatically from that audited
summary with Matplotlib. The generator validates source identity, row counts,
and zero-target guards; emits SVG, PDF, and 300-DPI PNG; and writes a separate
JSON receipt containing data selection, transformations, derived values, alt
text, and output hashes. Verification regenerated all 12 images byte-for-byte
and matched the article's alt text to the receipts. The generator ships
in this post bundle as
[`plot_signed_dose_scan_results.py`](plot_signed_dose_scan_results.py)
(SHA-256 `480c4b9ec2d9ea464119a9336053f5bb18838049274d7c623687f282047c25aa`).
It was not yet present at result commit `fde24e9`; that named gap was closed
by committing the script, the four figure sets (SVG/PDF/PNG), the per-figure
receipts, and the receipts index into the compact replication record at
[`f5e906e`](https://github.com/tdj28/llm_selfref_pre/tree/f5e906e1737bc71bf20b642af1d698018eec82fe/docs/consciousness_sae_signed_dose_scan/results/signed-dose-a084caa-wl8obvtuq0ax8t-v2-audit-recovery-c9/figures),
after re-verifying every file hash against the receipt provenance blocks and
`figure-receipts-index.json`. The experiment repo at `f5e906e` or later can
therefore be treated as a complete figure provenance root.

AI-assisted editorial review improved exposition but was not treated as
scientific verification. Two bounded `gpt-5.6-sol` Pro passes were kept
separate: the scientific-integrity review received the draft plus a compact
evidence card (review SHA-256
`2f25a07bfde8e7df7bbb59c3bbc0e13908052d6df6a3ed1c4709fdec38e89645`;
cost `$0.7671`), while the zero-context reader review received only the
scientifically corrected draft (review SHA-256
`48fd3d83a2e02495d4ec15ac5fb1f7838dab9bb675a2e59b464bfa42206e84ee`;
cost `$0.6345`). Neither received raw tensors or row-level data.

**Figure resources.** [Figure 1 receipt](fig-1-linearity-census.receipt.json) ·
[PDF](fig-1-linearity-census.pdf) · [PNG](fig-1-linearity-census.png)  
[Figure 2 receipt](fig-3-bf16-delivery-floor.receipt.json) ·
[PDF](fig-3-bf16-delivery-floor.pdf) · [PNG](fig-3-bf16-delivery-floor.png)  
[Figure 3 receipt](fig-2-dose-depth-arc.receipt.json) ·
[PDF](fig-2-dose-depth-arc.pdf) · [PNG](fig-2-dose-depth-arc.png)  
[Figure 4 receipt](fig-4-j-versus-identity.receipt.json) ·
[PDF](fig-4-j-versus-identity.pdf) · [PNG](fig-4-j-versus-identity.png)

## References

- <a id="ref-gurnee-2026"></a>Gurnee et al. (2026), [*Verbalizable Representations Form a Global Workspace
  in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html),
  *Transformer Circuits Thread*, published 6 July 2026; the Jacobian-lens
  construction is in [Methods](https://transformer-circuits.pub/2026/workspace/index.html#methods-jlens).
- <a id="ref-anthropic-jlens"></a>Anthropic, [Jacobian-lens reference
  implementation](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e),
  pinned commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (Apache License 2.0).
- <a id="ref-neuronpedia-jlens"></a>Neuronpedia, [Llama 3.3 70B Jacobian-lens
  configuration](https://huggingface.co/neuronpedia/jacobian-lens/blob/a4114d7752d11eb546e6cf372213d7e75526d3a1/llama3.3-70b-it/jlens/Salesforce-wikitext/config.yaml),
  pinned revision `a4114d7752d11eb546e6cf372213d7e75526d3a1`.
- <a id="ref-meta-llama-70b"></a>Meta, [Llama 3.3 70B Instruct model
  card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/6f6073b423013f6a7d4d9f39144961bfbfbc386b/README.md),
  pinned revision `6f6073b423013f6a7d4d9f39144961bfbfbc386b` (Llama Community License; gated weights).
- <a id="ref-belrose-2023"></a>Belrose et al. (2023), [*Eliciting Latent Predictions from Transformers
  with the Tuned Lens*](https://arxiv.org/abs/2303.08112), arXiv:2303.08112;
  accessed 18 July 2026.
- <a id="ref-hernandez-2024"></a>Hernandez et al. (2024), [*Linearity of Relation Decoding in Transformer
  Language Models*](https://arxiv.org/abs/2308.09124), ICLR 2024;
  arXiv:2308.09124; accessed 18 July 2026.
- <a id="ref-rimsky-2024"></a>Rimsky et al. (2024), [*Steering Llama 2 via Contrastive Activation
  Addition*](https://aclanthology.org/2024.acl-long.828/), ACL 2024,
  doi:10.18653/v1/2024.acl-long.828; accessed 18 July 2026.
- <a id="ref-taimeskhanov-2026"></a>Taimeskhanov, Vaiter, and Garreau (2026), [*Towards Understanding Steering
  Strength*](https://arxiv.org/abs/2602.02712), arXiv:2602.02712v2, 8 July
  2026; accepted at ICML 2026; accessed 17 July 2026.
- <a id="ref-obrien-2025"></a>O'Brien et al. (2025), [*Steering Language Model Refusal with Sparse
  Autoencoders*](https://icml.cc/virtual/2025/49605), ICML 2025 Actionable
  Interpretability Workshop; [arXiv:2411.11296v2](https://arxiv.org/abs/2411.11296),
  22 May 2025; accessed 17 July 2026.
- <a id="ref-sae-steering"></a>Jones, T. (2026), [*Can a Jacobian Lens Detect SAE
  Steering?*](https://praxagent.ai/blog/posts/jacobian-lens-sae-steering/),
  Praxagent Research Note.
