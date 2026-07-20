---
title: "Data leakage"
slug: "data-leakage"
summary: "An accidental flow of information across an evaluation boundary that makes held-out performance look more trustworthy than it is."
og_image: "data-leakage-split-first.png"
og_image_alt: "A protected evaluation splits first and fits preprocessing only on training rows."
draft: false
pro_reviewed: true
---

**Data leakage** occurs when information that should be unavailable during
model training influences the training process or the choices used to evaluate
it. In a train-test evaluation, the test set is meant to imitate genuinely
unseen observations. If test measurements help choose imputation values,
feature scales, selected features, or model settings, that evaluation boundary
has leaked.

Leakage does not require exposing test labels. Test feature values alone can
leak when they influence learned preprocessing.

{{< reference-figure
  src="data-leakage-split-first.svg"
  alt="The leaking workflow fits preprocessing on all rows before splitting, while the protected workflow splits first and fits imputation, scaling, principal component analysis, and the model using training rows only."
  caption="Top: fitting learned preprocessing on all rows lets the future test rows affect the values and directions used during training. Bottom: split independent units first, fit every data-adaptive step on training rows, and only transform the untouched test rows with the already fitted pipeline. Fixed parsing and declared unit conversions are not shown because they do not learn from the observed sample."
>}}

## A small example

Suppose the observed training widths are \(2.9\) and \(3.1\) millimeters, with
one training width missing. Their median is \(3.0\). The future test widths are
\(3.7\) and \(3.9\).

If the analyst combines both subsets before [imputation]({{< relref
"imputation.md" >}}), the median of all four observed values is \(3.4\). The
missing training cell is then filled with \(3.4\), so the test measurements have
changed the training table. That is leakage even though no test label was used.
The numbers are intentionally small toy values, not an empirical estimate of
the size of leakage in a real study.

## The protected sequence

1. Define the independent unit and split raw units into training and test sets.
2. Fit data-adaptive preparation, such as imputation and [standard
   scaling]({{< relref "knowledge-base/glossary/standard-scaling/index.md" >}}), on training rows only.
3. Fit feature selection, [principal component analysis]({{< relref
   "principal-component-analysis.md" >}}), and the prediction model on the
   training data only.
4. Apply the already fitted transformations to the test rows.
5. Evaluate once on those held-out rows using a metric chosen for the task.

Cross-validation repeats this boundary. Each fold needs its own preprocessing
fit, which is why software pipelines are useful.

Not every operation before a split leaks. A fixed file parser, documented unit
conversion, or validity rule declared without inspecting outcomes may be
applied consistently. The risky steps are those that learn values, directions,
thresholds, or choices from the observed data.

## Descriptive PCA is a different goal

Using every row for a purely descriptive PCA is not automatically leakage. If
the stated question is "what patterns appear in these exact observations?",
then the full dataset is the object being described and there is no held-out
performance claim.

The problem begins when an analyst reuses globally imputed, globally scaled, or
globally fitted PCA features and then claims that a later test split measures
performance on unseen data. For that claim, the split must happen before any
data-adaptive preprocessing is fitted.

## What preventing leakage does not establish

- A clean split does not repair biased sampling, mislabeled rows, dependence
  between related samples, or distribution shift.
- Leakage-free accuracy is not automatically biological understanding or a
  causal result.
- Randomly splitting rows is not enough when several rows come from the same
  organism, patient, plot, plate, or time series. Related rows may need to stay
  in the same split.
- Repeatedly checking the test result and changing the analysis in response can
  turn the test set into another training signal.

See also: [imputation]({{< relref "knowledge-base/glossary/imputation/index.md" >}}), [standard scaling]({{<
relref "knowledge-base/glossary/standard-scaling/index.md" >}}), and [principal component analysis]({{<
relref "knowledge-base/glossary/principal-component-analysis/index.md" >}}).
