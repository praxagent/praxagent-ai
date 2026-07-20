---
title: "L2 normalization"
slug: "l2-normalization"
summary: "Rescaling a nonzero vector to Euclidean length one while preserving its direction."
og_image: "l2-normalization-unit-circle.png"
og_image_alt: "Two vectors on the same ray land at the same point after L2 normalization."
draft: false
pro_reviewed: true
---

**L2 normalization** rescales a nonzero vector so its Euclidean length is 1 while preserving its direction. The name comes from the **L2 norm**, another name for Euclidean length.

For a vector \(x=(x_1,\ldots,x_d)\) with \(d\) coordinates, its L2 norm is

\[
\lVert x\rVert_2
= \sqrt{\sum_{j=1}^{d}x_j^2}.
\]

Here \(j\) identifies a coordinate and \(x_j\) is its value. If \(x\ne 0\), its L2-normalized vector is

\[
\hat{x}=\frac{x}{\lVert x\rVert_2}.
\]

The hat in \(\hat{x}\) labels the normalized version of \(x\).

{{< reference-figure
  src="l2-normalization-unit-circle.svg"
  alt="Two vectors of different lengths that point in the same direction land on the same point of a unit circle after L2 normalization."
  caption="The vectors \((3,4)\) and \((6,8)\) have lengths 5 and 10 but point along the same ray. Dividing each vector by its own length maps both to \((0.6,0.8)\) on the unit circle. L2 normalization removes overall length, not direction. The circle and coordinates are a two-dimensional teaching example."
>}}

## Worked example

For \(x=(3,4)\),

\[
\lVert x\rVert_2=\sqrt{3^2+4^2}=5,
\qquad
\hat{x}=\left(\frac{3}{5},\frac{4}{5}\right)=(0.6,0.8).
\]

The normalized vector has length

\[
\sqrt{0.6^2+0.8^2}=1.
\]

```python
import math

def l2_normalize(values):
    length = math.sqrt(sum(value * value for value in values))
    if length == 0:
        raise ValueError("the zero vector has no direction to normalize")
    return tuple(value / length for value in values)

assert l2_normalize((3.0, 4.0)) == (0.6, 0.8)
```

## The zero-vector boundary

The zero vector has norm 0, so dividing it by its norm would divide by zero. A pipeline must choose an explicit behavior, such as raising an error, leaving the vector unchanged, or adding a small stabilizing constant. These choices are not mathematically identical. A zero pooled embedding may also signal an upstream masking or model problem worth investigating.

L2 normalization differs from [root mean square (RMS)]({{< relref "knowledge-base/glossary/rms/index.md" >}}) scaling. For \(d\) coordinates, \(\operatorname{RMS}(x)=\lVert x\rVert_2/\sqrt{d}\). A unit-L2 vector has RMS \(1/\sqrt{d}\), while a unit-RMS vector has L2 norm \(\sqrt{d}\).

## Why retrieval systems use it

For nonzero vectors, the dot product of their L2-normalized versions equals their [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}). Normalization therefore lets a vector database use a dot-product search while comparing directions. It also discards vector magnitude, which may be helpful or may remove meaningful information depending on how the encoder was trained.

## What L2 normalization does not establish

- A unit-length vector is not a probability distribution; its coordinates can be negative and need not sum to 1.
- Equal directions do not prove equal text meaning or biological function.
- Normalization does not center features, standardize columns, or repair an inappropriate embedding model.

See also: [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}), [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}), [root mean square]({{< relref "knowledge-base/glossary/rms/index.md" >}}).
