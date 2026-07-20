---
title: "Principal component analysis (PCA)"
slug: "principal-component-analysis"
summary: "A linear dimensionality-reduction method that replaces numerical feature columns with new perpendicular axes ordered by how much variance they capture."
og_image: "pca-points-new-axes.png"
og_image_alt: "The same observations are described with original feature axes and new principal-component axes."
draft: false
pro_reviewed: true
---

**Principal component analysis (PCA)** turns several numerical feature columns
into a new set of coordinate axes. The first new axis follows the direction in
which the observations are most spread out. That spread is **variation**;
**variance** is one numerical way to measure it by averaging squared distances
from the mean. Each later axis is perpendicular to the earlier axes and
captures as much remaining variance as it can.

PCA is a form of **dimensionality reduction**. A table with seven feature
columns has seven dimensions. Plotting only its first two principal components
gives a two-dimensional summary. That summary is easier to inspect, but it can
leave variation out. The explained-variance ratio reports how much of the
dataset's total variance the displayed components retain.

For a complete biological example with real wheat-kernel measurements, inline
Python, and interpretation guidance, use the [full PCA Deep
Dive]({{< relref "knowledge-base/deep-dives/principal-component-analysis/index.md" >}}).

{{< reference-figure
  src="pca-points-new-axes.svg"
  alt="The same five observations appear before and after principal component analysis while the coordinate axes rotate to follow the long and short directions of their spread."
  caption="The observations do not move relative to one another. PCA replaces the original feature axes with principal component 1, which follows the greatest spread, and a perpendicular principal component 2, which follows the next greatest spread. This two-feature toy geometry explains the coordinate change; real analyses can start with many more features and can lose information when only two components are displayed."
>}}

## What the calculation does

Think of a data table with one observation per row and one measurement per
feature column. PCA then:

1. centers each feature by subtracting its mean;
2. optionally uses [standard scaling]({{< relref "knowledge-base/glossary/standard-scaling/index.md" >}})
   when the scientific question calls for comparable feature scales;
3. finds a unit-length direction through feature space with the greatest
   [variance]({{< relref "knowledge-base/glossary/standard-deviation/index.md" >}});
4. projects every observation onto that direction to produce its first
   principal-component score; and
5. repeats the search for perpendicular directions that capture successively
   less of the remaining variance.

A **score** is one observation's coordinate on a principal component. A
**principal-axis coefficient** is the weight assigned to one original feature
when constructing that component. Reading scores and coefficients together
connects a point on the map back to the original measurements.

## A two-feature score

Suppose a toy observation has standardized coordinates \(x=(1.25,0.50)\). Its
first principal-axis direction is \(v=(0.80,0.60)\). The direction has length
one because \(0.80^2+0.60^2=1\). The observation's PC1 score is the weighted
sum

\[
x\mathbin{\cdot}v
=(1.25\times0.80)+(0.50\times0.60)
=1.30.
\]

The dot symbol means multiply matching coordinates and add the products. The
numbers are a teaching example, not fitted coefficients from a dataset. In a
real analysis, software learns the direction from all rows.

## Choices that change the map

- **Rows and features:** adding or removing observations or feature columns
  changes the matrix PCA summarizes.
- **Units and scaling:** an unscaled feature with large numerical variance can
  dominate even when its units, rather than its biology, created that scale.
- **Missing values:** most PCA implementations require a complete matrix, so
  exclusions or [imputation]({{< relref "knowledge-base/glossary/imputation/index.md" >}}) become part of
  the analysis.
- **Displayed components:** a two-component plot can hide differences that
  lie mainly in later components.
- **Signs:** multiplying one component's coefficients and every corresponding
  score by \(-1\) mirrors the axis without changing the PCA result.

## What PCA does not establish

- A cluster is not automatically a biological class or a statistically
  significant group difference.
- Separation does not show that a measured feature caused the separation.
- Explained variance is not prediction accuracy or the percentage of biology
  explained.
- A descriptive full-dataset PCA does not demonstrate performance on new
  observations. That requires a held-out evaluation protected from [data
  leakage]({{< relref "knowledge-base/glossary/data-leakage/index.md" >}}).

See also: [standard deviation]({{< relref "knowledge-base/glossary/standard-deviation/index.md" >}}),
[standard scaling]({{< relref "knowledge-base/glossary/standard-scaling/index.md" >}}), and [Pearson
correlation]({{< relref "knowledge-base/glossary/pearson-correlation/index.md" >}}).
