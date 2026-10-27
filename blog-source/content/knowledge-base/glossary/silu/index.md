---
title: "SiLU"
slug: "silu"
summary: "The activation that multiplies a number by its own sigmoid, smooth through zero."
draft: false
pro_reviewed: true
og_image: "silu-card.png"
og_image_alt: "The SiLU curve passes smoothly through the origin, unlike a rectified linear unit, which has a corner there."
---

**SiLU** (sigmoid linear unit), also called swish, multiplies a number by the sigmoid of itself:

\[
\mathrm{SiLU}(x)=x\,\sigma(x)=\frac{x}{1+e^{-x}}.
\]

The sigmoid is smooth, and so is this product. In particular there is no corner at \(0\): \(\mathrm{SiLU}(0)=0\) and the slope there is \(1/2\). The dense models studied in the [note on chaos in a language model's feedback loop]({{< relref "posts/2026/10/chaos-bifurcations-language-model-feedback/index.md" >}}) use SiLU inside a gated feed-forward block. That block computes two vectors from the same input, applies SiLU to one of them coordinate by coordinate, and multiplies the result, coordinate by coordinate, with the other.

{{< reference-figure
  src="silu.svg"
  alt="A smooth SiLU curve crosses the origin with slope one half. A rectified linear unit is flat on the left and has a corner at zero."
  caption="The curve is the graph of x times the sigmoid of x, not a measured activation from a training run. Marked values: SiLU(0) = 0, SiLU(1) is about 0.731, and SiLU(-1) is about -0.269. The rectified linear unit is drawn only as a contrast: it is zero for negative inputs and has a corner at zero."
>}}

## Worked example

\[
\begin{aligned}
\mathrm{SiLU}(0) &= 0,\\
\mathrm{SiLU}(1) &= \frac{1}{1+e^{-1}} \approx 0.7311,\\
\mathrm{SiLU}(-1) &= \frac{-1}{1+e^{1}} \approx -0.2689.
\end{aligned}
\]

Negative inputs are not clipped to zero. They are shrunk toward zero, because the sigmoid of a large negative number is small. For such inputs the code below uses the equal form \(x\,e^{x}/(1+e^{x})\), because \(e^{-x}\) would overflow.

```python
from math import exp

def silu(x: float) -> float:
    """Return SiLU for a finite scalar input."""
    if x >= 0.0:
        return x / (1.0 + exp(-x))
    ex = exp(x)
    return x * ex / (1.0 + ex)
```

SiLU is a scalar function applied coordinate by coordinate inside a block. It is not a next-token probability, and applying it does not by itself produce a vocabulary distribution.

See also: [feed-forward network]({{< relref "knowledge-base/glossary/feed-forward-network/index.md" >}}), [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}).
