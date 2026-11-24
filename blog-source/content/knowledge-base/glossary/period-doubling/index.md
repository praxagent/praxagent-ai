---
title: "Period-doubling bifurcation"
slug: "period-doubling"
summary: "The change at which a fixed point or cycle loses stability through a multiplier of -1 and a cycle of twice the period branches off; repeated doublings can cascade into chaos."
draft: true
pro_reviewed: true
og_image: "period-doubling-card.png"
og_image_alt: "Bifurcation diagram of the logistic map for r from 2.8 to 4: one branch splits into two at r = 3, into four near 3.4495 and into eight near 3.5441, then dissolves into a chaotic band past the accumulation near 3.5699."
---

A **period-doubling bifurcation**, also called a *flip*, happens when a fixed point or cycle of an [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}) loses stability because one of its [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}) passes through \(-1\) as a parameter changes, while any others stay below 1 in absolute value. A fixed point is a state the map leaves unchanged; a cycle of period \(m\) is a set of \(m\) states the map visits in turn; a multiplier is a factor by which the map multiplies a small displacement along one direction over one full trip around the cycle, \(m\) steps (one step for a fixed point). Just past the crossing, a small displacement in that direction flips sign and grows with every trip.

In the generic case, where the multiplier crosses \(-1\) at nonzero speed and a combination of the map's second and third derivatives is nonzero there, a cycle of twice the period branches off at the crossing. The old fixed point or cycle does not split; it persists, now unstable. In the *supercritical* case the new cycle is stable, is born at the crossing close to the old state, and nearby orbits settle on it; in the *subcritical* case it is unstable and exists only before the crossing, and orbits leave the neighborhood instead. Repeated doublings, each after a smaller change of the parameter than the last, form a **cascade**: periods 2, 4, 8, 16 and so on, accumulating at a finite parameter value beyond which the motion can be chaotic.

For a fixed point \(x^\ast(r)\) of a one-dimensional map \(F\) with parameter \(r\), the multiplier is the slope \(\mu(r)=F'(x^\ast(r))\), and the flip happens at the parameter \(r_1\) where it reaches \(-1\). The new cycle is a pair of states \(p\) and \(q\) that the map swaps, and its multiplier is the product of the slopes along it, the cycle's [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}):

\[
\mu(r_1)=-1,
\qquad
F(p)=q,\quad F(q)=p,
\qquad
\mu_2(r)=F'(p)\,F'(q).
\]

The two-point cycle doubles in turn where \(\mu_2\) reaches \(-1\), and so on down the cascade.

{{< reference-figure
  src="period-doubling.png"
  alt="Bifurcation diagram of the logistic map for r from 2.8 to 4: long-run values of x form one branch that splits into two at r = 3, into four near 3.4495 and into eight near 3.5441, then dissolve into a dense band past about 3.5699, with pale windows of order inside the band. Dashed lines mark the first three doublings and a solid line the accumulation point."
  caption="Toy example, no language model. For each of 1,200 values of r, the logistic map was run from x = 0.5 for 1,500 steps and the next 160 values were plotted. Dashed lines mark the computed births of periods 2, 4 and 8, where a cycle's multiplier reaches -1; the solid line marks the accumulation of the cascade, extrapolated from the last two of six computed onsets by letting each later gap shrink by Feigenbaum's factor 4.669. The pale bands inside the chaotic region are windows where a cycle attracts again. The plot shows where orbits from one start end up: not every attractor, and none of the unstable cycles, such as each old cycle after it doubles."
>}}

Finite runs place doublings badly. Near an onset the cycle that is losing stability contracts very slowly, and just past it an orbit can linger near the now-unstable cycle for a long time before it escapes, so the point where a run appears to change depends on how long it was run. Computing each onset directly, as the parameter at which a cycle's multiplier equals \(-1\), avoids that bias.

## Worked example

For the logistic map \(F(x)=r\,x(1-x)\) on \(0\le x\le1\), which keeps \(x\) in that interval when \(0\le r\le4\), the nonzero fixed point \(1-1/r\) has multiplier \(2-r\), which reaches \(-1\) at \(r=3\). For \(3\lt r\le4\) the two-point cycle is

\[
p,\,q=\frac{(r+1)\mp\sqrt{(r+1)(r-3)}}{2r},
\]

and its multiplier simplifies to \(\mu_2=4+2r-r^{2}\). That reaches \(-1\) when \(r^{2}-2r-5=0\), at \(r=1+\sqrt{6}\approx3.449490\), the birth of period 4. At \(r=3.2\) the cycle is \(p=0.513045\), \(q=0.799455\), with \(\mu_2=0.16\): it attracts.

```python
import math

def logistic_two_cycle(r):
    """Points and multiplier of the logistic map's period-2 cycle, for 3 < r <= 4."""
    if isinstance(r, bool) or not isinstance(r, (int, float)) or not 3 < r <= 4:
        raise ValueError("this example covers 3 < r <= 4")
    root = math.sqrt((r + 1) * (r - 3))
    p, q = ((r + 1) - root) / (2 * r), ((r + 1) + root) / (2 * r)
    multiplier = r * (1 - 2 * p) * r * (1 - 2 * q)
    return p, q, multiplier
```

`[round(v, 6) for v in logistic_two_cycle(3.2)]` returns `[0.513045, 0.799455, 0.16]`.

## What a doubling does not establish

One doubling is not chaos, and it does not imply a cascade: a cycle can also lose stability through a multiplier of \(+1\) or a complex pair, and many routes end before period 4. A cascade seen in finite runs needs its onsets computed before the ratios of its gaps are compared with the [Feigenbaum constants]({{< relref "knowledge-base/glossary/feigenbaum-constant/index.md" >}}).

The [field guide to the dynamics of looped transformers]({{< relref "posts/2026/11/looped-transformer-dynamics/index.md" >}}) follows a cascade inside a trained looped model.

See also: [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [Feigenbaum constants]({{< relref "knowledge-base/glossary/feigenbaum-constant/index.md" >}}), [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}), [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}).
