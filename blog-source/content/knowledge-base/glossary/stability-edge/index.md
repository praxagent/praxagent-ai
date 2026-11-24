---
title: "Stability edge and margin"
slug: "stability-edge"
summary: "For an update that moves a state only part of the way to a map's output, the step size below which a fixed point attracts and above which it is unstable, and how far the usual step lies below it."
draft: false
pro_reviewed: true
og_image: "stability-edge-card.png"
og_image_alt: "Three nested disks that all touch the point 1 on the complex plane: the unit disk for step 1, a smaller dashed disk for step 1.25 and a still smaller one for step 2. A toy multiplier at minus 0.6 lies on the boundary of the step-1.25 disk, and one at 0.9 lies inside all three."
---

The **stability edge** of a fixed point, a state that a map leaves unchanged, is the step size that divides two regimes of a *relaxed* version of the map: at every shorter positive step the fixed point attracts nearby states, and at every longer step it is unstable. Its **margin** is how far the map's usual step lies below that edge. Take a differentiable map \(F\) of real vectors with a fixed point \(x^\ast=F(x^\ast)\), and write \(x_n\) for the state after \(n\) updates. The relaxed update moves the state only a fraction \(\lambda\) of the way from \(x_n\) to the map's output \(F(x_n)\), or beyond it when \(\lambda>1\):

\[
x_{n+1}=(1-\lambda)\,x_n+\lambda\,F(x_n).
\]

Here the step \(\lambda>0\) is one real number applied to the whole state, and \(\lambda=1\) is the map itself; shortening the step (\(\lambda<1\)) is called damping, and lengthening it (\(\lambda>1\)) over-relaxation. A fixed point of \(F\) stays a fixed point for every \(\lambda\ne0\). To first order, its stability is decided by its [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), the eigenvalues \(\mu\) of the Jacobian \(DF(x^\ast)\), the matrix of first partial derivatives of \(F\) at \(x^\ast\). The fixed point attracts when every multiplier lies inside the unit circle, \(\lvert\mu\rvert<1\), where the modulus \(\lvert\mu\rvert\) is the distance of \(\mu\) from 0 in the complex plane, and it is unstable when any lies outside. Relaxation turns the Jacobian into \((1-\lambda)I+\lambda\,DF(x^\ast)\), where \(I\) is the identity matrix. The eigenvectors stay the same, and each multiplier becomes

\[
\mu_\lambda=1-\lambda\,(1-\mu),
\]

so as \(\lambda\) changes, every multiplier moves along a straight line through the point 1. If \(\operatorname{Re}\mu<1\), where \(\operatorname{Re}\mu\) is the real part of \(\mu\), the relaxed multiplier lies strictly inside the unit circle exactly when

\[
0<\lambda<\lambda_c(\mu)=\frac{2\,(1-\operatorname{Re}\mu)}{\lvert1-\mu\rvert^{2}}=2\operatorname{Re}\frac{1}{1-\mu},
\]

where \(\lvert1-\mu\rvert\) is the distance from \(\mu\) to the point 1. If every multiplier has \(\operatorname{Re}\mu<1\), the edge of the fixed point is \(\lambda_c=\min_\mu\lambda_c(\mu)\), taken over all its multipliers: the smallest step at which some relaxed multiplier reaches the unit circle. The fixed point then attracts for every \(0<\lambda<\lambda_c\) and is unstable for every \(\lambda>\lambda_c\); at \(\lambda=\lambda_c\) itself the linear test does not decide. The margin of the unrelaxed map is \(m=\lambda_c-1\). For each multiplier,

\[
\lambda_c(\mu)-1=\frac{1-\lvert\mu\rvert^{2}}{\lvert1-\mu\rvert^{2}},
\]

so the margin is positive exactly when every multiplier lies inside the unit circle. If some multiplier has \(\operatorname{Re}\mu\ge1\), no positive step moves it inside the circle, and the fixed point has no edge.

Geometrically, the relaxed multiplier lies strictly inside the unit circle at step \(\lambda\) exactly when \(\mu\) lies strictly inside the circle of radius \(1/\lambda\) centered at \(1-1/\lambda\), a circle that passes through the point 1 and shrinks toward it as \(\lambda\) grows. The condition is the familiar step-size limit of gradient descent. Near a minimum with curvature (second derivative) \(h>0\), gradient descent with step size \(\eta\) maps a displacement \(x\) from the minimum to \(x-\eta\,h\,x\), which is stable for \(0<\eta<2/h\); for a real multiplier the edge \(2/(1-\mu)\) has the same form, with \(1-\mu\) in the role of \(h\) and \(\lambda\) in the role of \(\eta\).

{{< reference-figure
  src="stability-edge.svg"
  alt="Three nested disks that all touch the point 1 on the complex plane: the unit disk for step 1 (solid gray), a dashed disk of radius 0.8 for step 1.25 and a disk of radius 0.5 for step 2 (solid green). A toy multiplier at -0.6 lies on the boundary of the step-1.25 disk, and one at 0.9 lies inside all three; side cards explain each."
  caption="Toy values, not a measured spectrum. At a relaxed step lambda, a multiplier ends up strictly inside the unit circle exactly when it lies strictly inside a circle of radius 1/lambda that passes through the point 1: the unit circle at step 1, radius 0.8 at step 1.25 and radius 0.5 at step 2. The toy multiplier -0.6 sits on the step-1.25 circle, so its edge is 1.25: shorter steps keep it inside, and step 1.25 moves it to -1. The multiplier 0.9 stays inside all three, and its edge is 20. The edge belongs to one fixed point and says nothing about what motion follows past it."
>}}

## Worked example

Toy multipliers, no model. For \(\mu=-0.6\): \(\lambda_c=2(1.6)/1.6^{2}=1.25\), so \(m=0.25\), and at \(\lambda=1.25\) the multiplier becomes \(1-1.25\times1.6=-1\), on the circle. For \(\mu=0.9\): \(\lambda_c=2(0.1)/0.01=20\). For the complex pair \(\mu=0.5\pm0.5i\), where \(i\) is the imaginary unit, \(i^{2}=-1\), both members give \(\lambda_c=2(0.5)/(0.25+0.25)=2\). With all four present the edge is 1.25, set by \(-0.6\), even though 0.9 has the largest modulus and makes the approach slowest: here the multiplier that slows settling is not the one that sets the edge.

```python
def stability_edge(multipliers):
    """Smallest positive step at which some relaxed multiplier reaches the unit circle."""
    values = [complex(m) for m in multipliers]
    if not values:
        raise ValueError("need at least one multiplier")
    if any(m.real >= 1 for m in values):
        raise ValueError("a multiplier with real part >= 1 cannot be stabilized by relaxation")
    return min(2 * (1 / (1 - m)).real for m in values)
```

`stability_edge([-0.6, 0.9, 0.5 + 0.5j, 0.5 - 0.5j])` returns `1.25`.

## What the edge does not tell you

The edge is a statement about the linear stability of one fixed point. It does not say what motion follows past the edge, whether the relaxed loop started somewhere else ever reaches this fixed point, or whether a different step gives a better result for whatever the map computes. It is also only as good as the multipliers found: a minimum over a partial list can overestimate the true edge. A separate step for each part of the state can be analyzed part by part when the Jacobian is block triangular in those parts, with every block on one side of the diagonal zero: each diagonal block then has its own multipliers, relaxed by its own step, and its own edge. A causal model with one step per token is such a case, because each token's update depends only on itself and earlier tokens (see [block-triangular matrix]({{< relref "knowledge-base/glossary/block-triangular-matrix/index.md" >}})). Otherwise coupling between the parts can change the relaxed multipliers, so they must be computed for the whole state.

See also: [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [period-doubling bifurcation]({{< relref "knowledge-base/glossary/period-doubling/index.md" >}}), [block-triangular matrix]({{< relref "knowledge-base/glossary/block-triangular-matrix/index.md" >}}), [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}).
