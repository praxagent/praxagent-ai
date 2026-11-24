---
title: "Multiplier (fixed-point stability)"
slug: "multiplier"
summary: "An eigenvalue of a map's Jacobian at a fixed point: the factor by which a small displacement along one direction is multiplied on every step."
draft: false
pro_reviewed: true
og_image: "multiplier-card.png"
og_image_alt: "The complex plane with the unit circle shaded: toy multipliers lie inside it, so the fixed point attracts, and three arrows show the ways a multiplier can leave: through minus 1, through plus 1, or as a complex pair elsewhere on the circle."
---

A **multiplier** of a fixed point is a factor by which a map multiplies a small displacement from that point on every step. Take a differentiable [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}) \(F\) of real vectors with a fixed point \(x^\ast=F(x^\ast)\), and write \(\delta_n=x_n-x^\ast\) for the displacement after \(n\) steps. To first order the displacement evolves by the **[Jacobian matrix]({{< relref "knowledge-base/glossary/jacobian/index.md" >}})** \(J=DF(x^\ast)\), the matrix of first partial derivatives of \(F\) at \(x^\ast\):

\[
\delta_{n+1}\approx J\,\delta_n,
\qquad
J\,v=\mu\,v\;\Longrightarrow\;J^{n}v=\mu^{n}v .
\]

Here \(\mu\) is an eigenvalue of \(J\) and \(v\) a matching eigenvector, so a displacement along \(v\) is multiplied by \(\mu\) on every step. The eigenvalues of \(J\) are the fixed point's multipliers. A real multiplier scales displacements along its direction, and a negative one also flips their sign on every step. Because \(J\) is real, complex multipliers come in conjugate pairs \(\mu=\lvert\mu\rvert\,e^{\pm i\theta}\), where \(i\) is the imaginary unit, \(i^{2}=-1\), the modulus \(\lvert\mu\rvert\) is the distance of \(\mu\) from 0, and \(\theta\) is its angle from the positive real axis. Such a pair acts on a plane of displacements; in suitable coordinates on that plane, each step turns a displacement by \(\theta\) and scales it by \(\lvert\mu\rvert\).

If every multiplier lies inside the unit circle, \(\lvert\mu\rvert<1\), the fixed point attracts every state close enough to it. If any has \(\lvert\mu\rvert>1\), the fixed point is unstable, because some small displacements grow. It repels every other nearby state when all its multipliers lie outside the circle, and it is a *saddle*, attracting small displacements along some directions and repelling them along others, when some lie inside and some outside (see [stable manifold]({{< relref "knowledge-base/glossary/stable-manifold/index.md" >}})). When none lies outside but some lie on the circle, \(\lvert\mu\rvert=1\), the linear test does not decide, and the nonlinear terms of the map do. The largest modulus is the **spectral radius** \(\rho(J)\). To first order, a typical small displacement eventually shrinks or grows by about that factor per step; one with no component along the directions of the largest-modulus multipliers changes at the rate of the smaller ones. For a cycle, a set of states that the map visits in turn, the multipliers are those of its [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}) matrix.

As a parameter of the map changes, multipliers can leave the unit circle in three characteristic ways. In the usual case, when certain nondegeneracy conditions hold, each gives a different *bifurcation*, a qualitative change in the nearby orbits. Through \(-1\), small displacements flip sign as they grow, and a cycle of period two, a pair of states that the map swaps, is born or destroyed: a [period doubling]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}). Through \(+1\), the fixed point merges with another and both vanish: a fold. A complex pair crossing elsewhere creates or destroys an *invariant circle*, a closed curve that the map carries onto itself, on which the state turns by about \(\theta\) per step, an angle that need not be small: a Neimark–Sacker bifurcation. The nonlinear terms of the map enter those conditions and decide whether the new cycle or circle attracts.

{{< reference-figure
  src="multiplier.svg"
  alt="The complex plane, with a real and an imaginary axis, and the unit circle shaded. Four toy multipliers lie inside it: 0.5, -0.3 and the pair 0.4 plus or minus 0.5i. Arrows show three ways out of the circle: through -1, through +1, and as a complex pair; side cards name the usual bifurcation at each exit."
  caption="Toy values, not a measured spectrum. While every multiplier lies inside the unit circle, small nudges eventually die away. Each arrow marks a way out and the usual bifurcation there, not a guaranteed outcome. Through -1 (arrow 1), a nudge flips sign as it grows, and a cycle of period two is typically born or destroyed: a period doubling. Through +1 (arrow 2), the fixed point typically merges with another and both vanish: a fold. A complex pair leaving elsewhere (arrows 3) typically creates or destroys an invariant circle, on which the state turns per step by an angle that need not be small: a Neimark–Sacker bifurcation. Whether these outcomes occur, and whether a new cycle or circle attracts, depends on nonlinear terms that this linear picture does not show."
>}}

For a map with very many dimensions, the Jacobian is too large to form. The leading multipliers can still be found by an [Arnoldi eigensolver]({{< relref "knowledge-base/glossary/arnoldi-iteration/index.md" >}}), which needs only products \(Jv\) with chosen vectors; [forward-mode automatic differentiation]({{< relref "knowledge-base/glossary/forward-mode-automatic-differentiation/index.md" >}}) computes each product, exactly up to rounding, without building \(J\).

## Worked example

Take the toy Jacobian

\[
J=\begin{bmatrix}0.54 & -0.72\\ 0.72 & 0.54\end{bmatrix}.
\]

Its trace is 1.08 and its determinant is \(0.54^{2}+0.72^{2}=0.81\), so its multipliers solve \(\mu^{2}-1.08\,\mu+0.81=0\): \(\mu=0.54\pm0.72i\), with modulus \(\sqrt{0.81}=0.9\) and angle \(\arctan(0.72/0.54)\approx53.13^\circ\). This \(J\) is 0.9 times a rotation matrix, so the original coordinates are already suitable: each step turns every small displacement by about \(53^\circ\) and shrinks it to 90% of its length, and after 10 steps it keeps \(0.9^{10}\approx0.349\) of its original length.

```python
import cmath

def multipliers_2x2(a, b, c, d):
    """Eigenvalues of the 2-by-2 Jacobian [[a, b], [c, d]]."""
    entries = (a, b, c, d)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in entries):
        raise ValueError("entries must be real numbers")
    half_trace = (a + d) / 2
    root = cmath.sqrt(half_trace**2 - (a * d - b * c))
    return half_trace + root, half_trace - root
```

`multipliers_2x2(0.54, -0.72, 0.72, 0.54)` returns `((0.54+0.72j), (0.54-0.72j))`.

## What multipliers do not tell you

Multipliers describe only the linear response at one fixed point. They say nothing about states far away, about other fixed points, or about how large a disturbance grows before it decays: a matrix whose multipliers are all small can still stretch some vectors a great deal in one step (see [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}})). An eigensolver that returns only the leading multipliers can also miss the one that matters for a given question, such as the one nearest \(-1\).

See also: [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}), [stability edge and margin]({{< relref "knowledge-base/glossary/stability-edge/index.md" >}}), [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}), [basin of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}).
