---
title: "Imputation"
slug: "imputation"
summary: "A documented rule for filling missing values so an analysis can use a complete table without pretending the missing measurements were recovered."
og_image: "imputation-is-a-replacement.png"
og_image_alt: "A missing value is replaced with a training-column median while its missingness remains explicit."
draft: false
pro_reviewed: true
---

**Imputation** fills a missing cell with a value produced by a stated rule. For
example, median imputation replaces a missing value with the median of the
observed values in the same feature column.

A missing value may appear as a blank, `NA`, or `NaN`, which means "not a
number" in many data tools. Missing is not the same as zero. Imputation makes a
table complete enough for a method that cannot accept missing cells, but it
does not recover the measurement that was never observed.

{{< reference-figure
  src="imputation-is-a-replacement.svg"
  alt="A toy feature column with one missing value is inspected, a median of 11 is learned from observed training values, and the blank is filled while retaining a missing-value flag."
  caption="The observed toy training values 9, 10, 12, and 14 have median 11, so median imputation inserts 11 into the missing cell. The filled table is computationally complete, but 11 is a replacement rule, not a recovered measurement. Keeping a separate flag can preserve the fact that the original cell was missing, although whether that flag belongs in a model is another design choice."
>}}

## A checkable median example

The median of \(9,10,12,14\) is halfway between the two middle values, so it is
\(11\). This dependency-free Python applies that rule to a toy training column:

```python
from statistics import median

lengths = [9.0, 10.0, None, 12.0, 14.0]
observed = [value for value in lengths if value is not None]
replacement = median(observed)
filled = [replacement if value is None else value for value in lengths]

assert replacement == 11.0
assert filled == [9.0, 10.0, 11.0, 12.0, 14.0]
```

The code is easy. The scientific decision is harder: whether 11 is a defensible
replacement depends on why the value is missing and how the result will be
used.

## Investigate before filling

Ask these questions before choosing an imputation rule:

- Which features and biological or technical groups contain missing cells?
- Was the value lost randomly, or could absence reflect detection limits,
  sample quality, treatment, instrument, or operator?
- Does a single median make sense for every group and batch?
- How much of the table would be replaced?
- Does the interpretation survive another defensible missing-data treatment?

Deleting incomplete rows is also a missing-data decision. It can bias a result
when the incomplete rows differ systematically from the retained rows.

## What single imputation changes

Replacing several cells with the same median often reduces a feature's
[variance]({{< relref "knowledge-base/glossary/standard-deviation/index.md" >}}) and can alter correlations
with other features. A downstream method treats the replacements as ordinary
numbers unless the analysis explicitly represents their uncertainty. Multiple
imputation is a broader approach that creates several plausible completed
datasets, but it also requires assumptions and a rule for combining results.

For held-out prediction, fit the imputation rule on training rows only. Using
the test rows to choose a median creates [data leakage]({{< relref
"data-leakage.md" >}}). In a purely descriptive full-dataset analysis, using
all rows can match the stated goal, but the replacement and its effect must
still be reported.

## What imputation does not establish

- It does not prove what the missing value would have been.
- It does not make systematic missingness harmless.
- It does not correct a unit error, impossible value, duplicate row, or wrong
  label.
- It does not remove the need for a sensitivity analysis or uncertainty
  statement.

See also: [data leakage]({{< relref "knowledge-base/glossary/data-leakage/index.md" >}}), [standard
scaling]({{< relref "knowledge-base/glossary/standard-scaling/index.md" >}}), and [principal component
analysis]({{< relref "knowledge-base/glossary/principal-component-analysis/index.md" >}}).
