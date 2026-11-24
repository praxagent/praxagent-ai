---
title: "Kaplan–Yorke dimension"
slug: "kaplan-yorke-dimension"
summary: "An estimate of a chaotic attractor's dimension from its Lyapunov exponents: the largest dimension in which small volumes do not shrink in the long run, plus the fraction of the next direction needed to make them shrink."
draft: true
pro_reviewed: true
og_image: "kaplan-yorke-dimension-card.png"
og_image_alt: "The Hénon attractor, a thin folded band of points, beside the running sum of its two Lyapunov exponents, 0.42 and then minus 1.20, which crosses zero at 1.26."
---

The set that a chaotic motion settles onto, its attractor, can be thinner than any surface and yet more than a curve: its dimension can be a fraction. The **Kaplan–Yorke dimension** estimates that number from the [Lyapunov exponents]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}) \(\Lambda_1\ge\Lambda_2\ge\dots\ge\Lambda_d\) of a motion in \(d\) dimensions, the long-run rates at which small displacements grow or shrink along successive directions. Write \(S_k=\Lambda_1+\dots+\Lambda_k\) for the sum of the \(k\) largest, with \(S_0=0\), and let \(j\) be the largest \(k\) from 0 to \(d\) with \(S_k\ge0\). Then

\[
D_{\mathrm{KY}}=
\begin{cases}
j+\dfrac{S_j}{\lvert\Lambda_{j+1}\rvert}, & j\lt d,\\
d, & j=d.
\end{cases}
\]

The sum \(S_k\) is the long-run growth rate of a small \(k\)-dimensional volume of displacements in generic directions, and no \(k\)-dimensional volume grows faster. So \(j\) is the largest dimension in which such volumes do not shrink, which can exceed the number of positive exponents. The fraction is the share \(f\) of the next, contracting direction that brings the rate to zero, \(S_j+f\,\Lambda_{j+1}=0\): drawn against \(k\) with the points joined by straight lines, the running sum crosses zero exactly at \(D_{\mathrm{KY}}\). If \(j=d\), the running sum never turns negative and the estimate is the full dimension \(d\). For an orbit that settles on a fixed point or a cycle whose [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), the eigenvalues that scale small displacements on each pass around it, all have modulus below 1, every exponent is negative, so \(j=0\) and the estimate is 0, the dimension of a finite set of points. A multiplier of modulus exactly 1 gives a zero exponent, and the estimate can then be positive although the cycle is still a finite set.

{{< reference-figure
  src="kaplan-yorke-dimension.png"
  alt="Left: 20,000 points of the Hénon attractor, a thin band folded into nested arches. Right: the running sum of the map's two Lyapunov exponents, 0, then 0.42, then minus 1.20, joined by straight lines that cross zero at 1.26, marked with a diamond."
  caption="Toy example, no language model. (a) The attractor of the Hénon map with a = 1.4 and b = 0.3. (b) The running sum of its two Lyapunov exponents, computed along the orbit with Benettin's method: evolve two small displacements by the map's derivative, make them perpendicular and of unit length again after every step, and average the logarithms of their stretch factors. The first exponent is about 0.42, and the two must add to ln 0.3 ≈ −1.204, the natural logarithm of the factor 0.3 by which every step shrinks areas. The line between them crosses zero at 1 + 0.42/1.62 ≈ 1.26, the Kaplan–Yorke estimate. The estimate comes from growth rates, not from measuring the set's geometry."
>}}

## Worked example

The Hénon map \(F(x,y)=(1-1.4x^{2}+y,\;0.3x)\) is a standard chaotic map of the plane. Its [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}), the matrix of first partial derivatives,

\[
DF(x,y)=\begin{bmatrix}-2.8x & 1\\ 0.3 & 0\end{bmatrix},
\]

has determinant \(-0.3\) at every point. So every step shrinks areas by a factor of 0.3, and the two exponents, whose sum \(S_2\) is the growth rate of areas, must add up to \(\ln0.3\approx-1.204\) per step, where \(\ln\) is the natural logarithm. Measured along a long orbit, \(\Lambda_1\approx0.42\), so \(\Lambda_2\approx-1.62\). Then \(j=1\) and \(D_{\mathrm{KY}}=1+0.42/1.62\approx1.26\): more than a curve, much less than an area.

With more exponents the same rule applies. For \(0.10,\,0.04,\,-0.02,\,-0.15,\,-0.30\) the running sums are \(0.10,\,0.14,\,0.12,\,-0.03,\,-0.33\), so \(j=3\) and \(D_{\mathrm{KY}}=3+0.12/0.15=3.8\), although only two exponents are positive.

```python
import math


def kaplan_yorke(exponents):
    """Kaplan-Yorke dimension from a complete Lyapunov spectrum sorted from largest to smallest."""
    values = [float(v) for v in exponents]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("need one or more finite exponents")
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError("exponents must be sorted from largest to smallest")
    total = 0.0
    for j, value in enumerate(values):
        if total + value < 0:
            return j + total / abs(value)
        total += value
    return float(len(values))
```

`round(kaplan_yorke([0.10, 0.04, -0.02, -0.15, -0.30]), 6)` returns `3.8` and `round(kaplan_yorke([0.42, -1.62]), 2)` returns `1.26`. A map that preserves area has exponents summing to zero, and `kaplan_yorke([0.5, -0.5])` returns `2.0`.

## What the estimate does not establish

The Kaplan–Yorke conjecture proposes that the estimate equals the information dimension of the attractor's natural measure, the long-run share of time an orbit spends in each region; that dimension says how fast the information needed to name the small cell holding the orbit grows as the cells shrink. The conjecture is proved for some classes of systems, not in general. The estimate is built from growth rates rather than from the set's geometry, and it can differ from dimensions of the set itself, such as the box-counting dimension, which says how fast the number of small boxes needed to cover the set grows as the boxes shrink. Computed from a finite orbit, the exponents are finite-time estimates, and so is the dimension. The formula uses only the exponents up to the one that first makes the running sum negative, so a few leading exponents suffice when that happens among them. When it does not, all one learns is that the estimate is at least the number of exponents computed; the helper above, which assumes a complete spectrum, would return only that lower bound. And a chaotic transient, a long wander that eventually settles, can show positive exponents and a positive estimate over a window even when no chaotic attractor exists.

The [field guide to the dynamics of looped transformers]({{< relref "posts/2026/11/looped-transformer-dynamics/index.md" >}}) applies this estimate to looped transformers.

See also: [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}), [Oseledets theorem]({{< relref "knowledge-base/glossary/oseledets-theorem/index.md" >}}), [period-doubling bifurcation]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}), [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}).
