---
title: "Feigenbaum constants"
slug: "feigenbaum-constant"
summary: "Two universal numbers of period-doubling cascades in smooth one-hump maps with a quadratic top: the gaps between doublings shrink by δ ≈ 4.669, and distances on the cycle near the top of the hump by α ≈ 2.503."
draft: true
pro_reviewed: true
og_image: "feigenbaum-constant-card.png"
og_image_alt: "Ratios of successive gaps between period doublings of the logistic map for doubling index n = 2 to 7: 4.7514, 4.6563, 4.6682, 4.6687, 4.6691 and 4.6692, settling onto a dashed line at delta = 4.6692."
---

The **Feigenbaum constants** describe how fast a period-doubling cascade converges. In a cascade, a stable cycle's period doubles again and again, 2, 4, 8, 16 and so on, as a parameter changes. If \(r_n\) is the parameter value at which a cycle of period \(2^{n}\) is born, the gaps between successive onsets shrink geometrically, with ratio

\[
\delta=\lim_{n\to\infty}\frac{r_n-r_{n-1}}{r_{n+1}-r_n}=4.669201609\ldots
\]

where each gap is measured between two consecutive [period doublings]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}).

A second constant, \(\alpha=2.502907875\ldots\), scales distances in the state instead of the parameter. For a map \(F\) whose graph has a single hump, such as the logistic map \(x\mapsto r\,x(1-x)\), let \(c\) be its critical point, the top of the hump (\(c=1/2\) for the logistic map), and let \(s_n\) be the parameter at which \(c\) itself lies on the stable cycle of period \(2^{n}\). At \(s_n\) the cycle point nearest \(c\) is half a period away, at signed distance \(d_n=F^{2^{n-1}}(c)-c\), where \(F^{k}\) means \(k\) applications of \(F\). The ratio \(d_n/d_{n+1}\) tends to \(-\alpha\): each doubling shrinks that distance by a factor of about 2.5029 and moves the nearest point to the other side of \(c\). Some texts fold the sign into the constant and write \(\alpha=-2.5029\ldots\). Because the top is quadratic, the map squares small distances from \(c\), so the gap between the closest pair of cycle points, \(F(c)\) and the point half a period from it, shrinks by a factor of about \(\alpha^{2}\approx6.26\) per doubling.

Mitchell Feigenbaum found that both numbers are the same for every generic one-parameter family of smooth one-hump maps that runs through a cascade, provided the top is quadratic, falling away like the square of the distance from \(c\). Examples are the logistic map and the sine map \(x\mapsto r\sin(\pi x)\). Within that class the numbers are **universal**, and a top flatter than quadratic gives a different class with different constants ([Feigenbaum, 1978](https://doi.org/10.1007/BF01020332)). Here generic rules out contrived exceptions; for instance, measuring the parameter on a scale that is infinitely stretched at the accumulation point \(r_\infty=\lim_{n\to\infty}r_n\) changes the gap ratio. Systems in many dimensions that contract strongly in all but one direction near their cascade can share the same values ([Collet, Eckmann and Koch, 1981](https://doi.org/10.1007/BF01008475)).

{{< reference-figure
  src="feigenbaum-constant.png"
  alt="Ratios of successive gaps between period doublings of the logistic map, (r(n) - r(n-1)) / (r(n+1) - r(n)) for n = 2 to 7: 4.7514, 4.6563, 4.6682, 4.6687, 4.6691 and 4.6692, approaching a dashed line at delta = 4.6692."
  caption="Toy example, no language model. The first two onsets of the logistic map's cascade are exact, r(1) = 3 and r(2) = 1 + sqrt(6). Each later one was found in 60-digit arithmetic by solving two equations together: one puts a point on the cycle that is about to double, and one sets that cycle's multiplier, the product of the map's slopes around it, to -1. The known value of delta only supplies the starting guess for each search. The point at n is the ratio (r(n) - r(n-1)) / (r(n+1) - r(n)), for n = 2 to 7. The ratios approach Feigenbaum's delta (dashed) within a few doublings. Agreement to a fraction of a percent at finite order is numerical agreement, not a proof that a system belongs to the class."
>}}

## Worked example

The onsets \(r_1\) to \(r_6\) of the logistic cascade, from the birth of period 2 to the birth of period 64, are about 3, 3.449489743, 3.544090360, 3.564407266, 3.568759420 and 3.569691610. The first gap is \(3.449489743-3=0.449489743\) and the second \(3.544090360-3.449489743=0.094600617\), so the first ratio, the \(n=2\) term of the formula for \(\delta\), is \(0.449489743/0.094600617\approx4.7514\). The next ratios are about 4.6563, 4.6682 and 4.6687, closing in on \(\delta\).

```python
def gap_ratios(onsets):
    """Ratios (r[n] - r[n-1]) / (r[n+1] - r[n]) of successive gaps between doubling onsets."""
    if len(onsets) < 3:
        raise ValueError("need at least three onsets")
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    if any(g <= 0 for g in gaps):
        raise ValueError("onsets must increase")
    return [g0 / g1 for g0, g1 in zip(gaps, gaps[1:])]
```

`[round(x, 4) for x in gap_ratios([3, 3.449489743, 3.544090360, 3.564407266, 3.568759420, 3.569691610])]` returns `[4.7514, 4.6563, 4.6682, 4.6687]`.

## What a matching ratio does not prove

Ratios close to \(\delta\) at finite order are numerical agreement with the universality class, not a proof of membership: other mechanisms can produce similar numbers over a few doublings, and onsets read from finite runs rather than computed carry their own bias. The constants also say nothing about what a system computes, only about the geometry of its cascade.

The [field guide to the dynamics of looped transformers]({{< relref "posts/2026/11/looped-transformer-dynamics/index.md" >}}) compares a cascade inside a trained looped model with \(\delta\).

See also: [period-doubling bifurcation]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}), [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}), [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}).
