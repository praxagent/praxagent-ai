---
title: "Standard deviation"
slug: "standard-deviation"
summary: "A measure of how far values in one numerical feature typically spread around their mean."
og_image: "standard-deviation-same-mean.png"
og_image_alt: "Two toy feature columns share a mean but have different standard deviations."
draft: false
pro_reviewed: true
---

The **standard deviation** describes the spread of one numerical feature around
its mean. A small standard deviation means the values stay relatively close to
the mean. A larger one means they commonly sit farther away.

Standard deviation uses the same units as the feature. If root length is
measured in millimeters, its standard deviation is also measured in
millimeters.

{{< reference-figure
  src="standard-deviation-same-mean.svg"
  alt="Two toy feature columns have the same mean of 10, but the second column has values farther from 10 and therefore a larger standard deviation."
  caption="Both toy columns have mean 10 and use the same horizontal scale. The values 9, 9, 10, and 12 have population standard deviation 1.22, while 6, 9, 11, and 14 have population standard deviation 2.92. Standard deviation responds to distance from the mean, not to the location of the mean itself. These values illustrate the calculation and are not biological measurements."
>}}

## The population calculation

Suppose one feature contains \(n\) values. Let \(x_i\) be value \(i\), let
\(\mu\) be the mean of all \(n\) values, and let \(\sigma\) be their population
standard deviation. Then

\[
\sigma
= \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2}.
\]

Read the formula as a recipe:

1. subtract the mean from each value;
2. square each difference so negative and positive differences do not cancel;
3. average the squared differences; and
4. take the square root to return to the feature's original units.

The average squared difference in step 3 is the **variance**. Standard deviation
is its square root.

## Worked example

For the values \(8,9,10,13\), the mean is \(10\). Their squared distances from
the mean are \(4,1,0,9\), which sum to \(14\). Under the population convention,

\[
\sigma=\sqrt{14/4}=\sqrt{3.5}\approx1.87.
\]

This does not mean every value is exactly 1.87 units from the mean. It provides
one summary of the whole set's spread.

## Population and sample conventions

The formula above divides by \(n\). It describes the values supplied to the
calculation and is the convention used by scikit-learn's `StandardScaler`.
When data are treated as a sample used to estimate the spread of a larger
population, many programs instead divide by \(n-1\). For the four-value example,
that sample estimate is \(\sqrt{14/3}\approx2.16\).

Neither number is a typographical mistake. A report or code review should state
the convention, especially when reproducing an exact result.

## What standard deviation does not establish

- It is not the range between the minimum and maximum.
- It is not a confidence interval or a measure of uncertainty in the mean.
- It does not show whether the distribution is symmetric or has several
  clusters.
- It can be strongly affected by an outlier because distances are squared.
- Comparing raw standard deviations across features with different units can
  be misleading.

See also: [standard scaling]({{< relref "knowledge-base/glossary/standard-scaling/index.md" >}}), [Pearson
correlation]({{< relref "knowledge-base/glossary/pearson-correlation/index.md" >}}), and [principal component
analysis]({{< relref "knowledge-base/glossary/principal-component-analysis/index.md" >}}).
