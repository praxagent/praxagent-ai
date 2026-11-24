---
title: "Iterated map"
slug: "iterated-map"
summary: "A rule applied again and again to its own output; the sequence it produces can settle, approach a repeating cycle, or wander chaotically."
draft: false
pro_reviewed: true
og_image: "iterated-map-card.png"
og_image_alt: "Three cobweb diagrams of the logistic map: at r = 2.8 the path approaches a fixed point, at r = 3.2 it approaches a two-point cycle, and at r = 3.9 it shows no repeat in the 60 drawn steps."
---

An **iterated map** is a rule applied again and again to its own output. Start from a state \(x_0\), compute \(x_1=F(x_0)\), then \(x_2=F(x_1)\), and so on:

\[
x_{n+1}=F(x_n),\qquad n=0,1,2,\ldots
\]

where \(x_n\) is the state after \(n\) steps and \(F\) is the same function at every step. The state can be one number or a vector of millions. The sequence \(x_0,x_1,x_2,\ldots\) is the **orbit** of \(x_0\). A **fixed point** is a state the rule leaves unchanged, \(F(x^\ast)=x^\ast\). A **cycle of period \(p\)** is a set of \(p\) distinct states visited in turn: each satisfies \(F^{p}(x)=x\), where \(F^{p}\) means \(p\) applications of \(F\), and no smaller positive number of steps brings it back. The early part of an orbit, before it comes close to its long-run behavior, is the **transient**, and an **attractor** is a set that the rule maps onto itself and that orbits from all nearby starts approach. A rule often depends on a **parameter**, a setting held fixed while the state evolves, such as a temperature or a step size; changing the parameter changes the map.

In the long run a bounded orbit can approach a fixed point, approach a cycle, or keep moving without ever repeating. Such aperiodic motion is called **chaotic** when it is also sensitive to the start: a small enough difference between two starts grows exponentially on average, for as long as it stays small, at a rate given by a positive [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}). Definitions of chaos differ in detail, and aperiodic motion need not be chaotic: turning a point around a circle by an irrational fraction of a turn at every step never repeats, yet nearby points stay the same distance apart. An unbounded orbit can also escape to infinity.

Existence, attraction and uniqueness are three different questions. [Brouwer's fixed-point theorem]({{< relref "knowledge-base/glossary/brouwer-fixed-point/index.md" >}}) guarantees that a continuous map of a nonempty, compact, convex set of vectors with \(n\) real coordinates into itself has a fixed point, but not that orbits approach it; compact means closed and bounded, and convex means that the segment between any two points of the set stays in the set. An attracting fixed point need not be the only attractor: each attractor has its own [basin of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}), the set of starts whose orbits approach it. For a differentiable map, the [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}) of a fixed point are the slope of \(F\) there when the state is one number, or the eigenvalues of the [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}) of \(F\) there, its matrix of first derivatives, when the state is a vector. They decide attraction to first order: if every multiplier has magnitude below 1, the fixed point attracts all states close enough to it; if any exceeds 1, some nearby states move away; otherwise, with a multiplier of magnitude exactly 1, the answer depends on higher-order terms.

{{< reference-figure
  src="iterated-map.png"
  alt="Three cobweb diagrams of the logistic map for r = 2.8, 3.2 and 3.9. At 2.8 the path spirals in to the crossing of the curve and the diagonal, landing on alternate sides of it; at 3.2 it approaches a square loop between two points; at 3.9 it fills the plot with lines that do not close within the 60 drawn steps."
  caption="Toy example, no language model. Each panel plots the logistic map x(n+1) = r x(n) (1 - x(n)) (blue curve) and the diagonal x(n+1) = x(n) (dashed), and follows one orbit from x = 0.1 as a cobweb (brown): vertically to the curve, horizontally to the diagonal, repeated. At r = 2.8 the cobweb spirals in to the fixed point 0.642857 (circle), landing on alternate sides of it as the slope of −0.8 there predicts. At r = 3.2 it approaches a two-point cycle, 0.513045 and 0.799455 (circles). At r = 3.9 it does not close within the 60 drawn steps. A finite drawing cannot by itself prove that an orbit is chaotic."
>}}

## Worked example

For \(0\le r\le4\) the logistic map \(F(x)=r\,x(1-x)\) sends \(0\le x\le1\) into itself; \(r\) is its one parameter. Besides the fixed point 0 it has, for \(r>1\), the fixed point \(x^\ast=1-1/r\), the other solution of \(x=r\,x(1-x)\). At \(r=2.8\) that is \(x^\ast=0.642857\ldots\), and the slope there is \(F'(x^\ast)=r(1-2x^\ast)=2-r=-0.8\): to first order, a small displacement from \(x^\ast\) is multiplied by \(-0.8\) on every step, flipping sign and shrinking, so nearby orbits spiral in. At \(r=3.2\) the slope is \(-1.2\), the fixed point repels, and orbits approach a two-point cycle instead, 0.513045 and 0.799455 to six decimals, unless they land exactly on a fixed point; check that \(3.2\times0.513045\times0.486955\approx0.799455\). At \(r=3.9\) the orbit from 0.1 wanders: after 1,000 discarded steps, the average of \(\ln\lvert F'(x_n)\rvert\) over the next 200,000 steps, with \(\ln\) the natural logarithm, is about 0.496 per step, a positive numerical estimate of the Lyapunov exponent.

```python
def orbit(f, x0, steps):
    """Return [x0, f(x0), f(f(x0)), ...] with `steps` applications of f."""
    if not callable(f):
        raise ValueError("f must be callable")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    xs = [x0]
    for _ in range(steps):
        xs.append(f(xs[-1]))
    return xs
```

`orbit(lambda x: x / 2 + 1, 0.0, 4)` returns `[0.0, 1.0, 1.5, 1.75, 1.875]`: the map \(x\mapsto x/2+1\) has the fixed point 2, and each step halves the distance to it.

## What a finite run does not show

A finite orbit cannot by itself tell a long transient from an attractor, a long cycle from chaos, or a slow approach from a stall. In finite-precision arithmetic every step is rounded, so a computed orbit is a numerical observation; a chaotic one soon differs from the exact orbit of the same start, even when its statistics remain meaningful.

In a [looped transformer]({{< relref "knowledge-base/glossary/looped-transformer/index.md" >}}) with its input held fixed, one pass of the shared core is an iterated map of the whole internal state.

See also: [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}), [basin of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}), [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [period-doubling bifurcation]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}).
