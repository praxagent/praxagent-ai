---
title: "Jacobian"
slug: "jacobian"
summary: "The matrix of first partial derivatives of a map: how each output responds to a small change in each input, and so, where the map is differentiable, its best linear stand-in near one point."
draft: false
pro_reviewed: true
og_image: "jacobian-card.png"
og_image_alt: "A small circle of inputs around a point, its exact image under a toy map beside the Jacobian's linear prediction, and the error of that prediction shrinking like the square of the circle's radius."
---

The **Jacobian** of a map is the matrix of its first partial derivatives. It records how each output responds to a small change in each input, and so, where the map is differentiable, gives its best linear stand-in near one point. For a map \(F\) from \(n\) inputs to \(m\) outputs, evaluated at a point \(p\), it is the \(m\times n\) matrix

\[
J=DF(p),\qquad J_{ij}=\frac{\partial F_i}{\partial x_j}(p),
\]

where \(F_i\) is the \(i\)th output and \(x_j\) the \(j\)th input. If \(F\) is differentiable at \(p\), a small nudge \(d\) to the input changes the output by about \(J\,d\):

\[
F(p+d)=F(p)+J\,d+o\bigl(\lVert d\rVert\bigr),
\]

where \(\lVert d\rVert\) is the length of the nudge and \(o(\lVert d\rVert)\) stands for an error that shrinks faster than the nudge: divided by \(\lVert d\rVert\), it tends to zero as \(d\) does. Being differentiable at \(p\) means exactly that some matrix makes the error this small, and that matrix is \(J\). If \(F\) has continuous second derivatives near \(p\), as the toy map below does, the error is \(O(\lVert d\rVert^{2})\): at most a constant times \(\lVert d\rVert^{2}\) once the nudge is small enough. Column \(j\) of \(J\) is the response to a nudge of input \(j\) alone, and row \(i\) collects everything that moves output \(i\).

Often a computation needs only the product \(J\,v\) with a chosen vector \(v\), a **Jacobian-vector product**. [Forward-mode automatic differentiation]({{< relref "knowledge-base/glossary/forward-mode-automatic-differentiation/index.md" >}}) computes it exactly up to rounding, at a cost comparable to one more evaluation of \(F\), without ever forming \(J\). That matters when \(J\) is square with a million rows, as for a map that sends a state to the next state: an eigensolver such as [Arnoldi iteration]({{< relref "knowledge-base/glossary/arnoldi-iteration/index.md" >}}) approximates a few of its eigenvalues, the numbers \(\mu\) for which \(J\,w=\mu\,w\) for some nonzero vector \(w\), from such products alone.

{{< reference-figure
  src="jacobian.png"
  alt="Three panels for the toy map F(x, y) = (x squared plus y, x times y) near p = (2, 1). Left: twelve marked inputs on a circle of radius 0.3 around p. Middle: the exact image of that circle, a slightly bent oval, beside the Jacobian's prediction, a dashed ellipse, with short gray segments joining each exact point to its prediction. Right: on logarithmic axes, the largest error of the prediction on circles of radius 0.01 to 1 follows the line error equals r squared, below the dashed line for the nudge itself and meeting it at r = 1."
  caption="Toy example, no language model. (a) Twelve inputs on a circle of radius 0.3 around p = (2, 1). (b) Their exact images under F(x, y) = (x² + y, xy) (solid) and the Jacobian's prediction F(p) + J d (dashed); gray segments join each exact image to its prediction. (c) The largest error of the prediction on circles of radius r: exactly r² for this map, so it vanishes faster than the nudge itself (dashed), though by r = 1 it is as large as the nudge. A linear prediction this good near p need not be good far from it."
>}}

## Worked example

Take \(F(x,y)=(x^{2}+y,\;xy)\) at \(p=(2,1)\), where \(F(p)=(5,2)\). Differentiating each output by each input,

\[
J=\begin{bmatrix}2x & 1\\ y & x\end{bmatrix}_{(2,1)}=\begin{bmatrix}4 & 1\\ 1 & 2\end{bmatrix}.
\]

For the nudge \(d=(0.01,\,0)\) the Jacobian predicts the change \(J\,d=(0.04,\,0.01)\). The exact change is \(F(2.01,1)-F(2,1)=(5.0401-5,\;2.01-2)=(0.0401,\;0.01)\), so the prediction misses by \((0.0001,\,0)\). For this map the miss is exactly \((d_x^{2},\;d_x d_y)\), never larger than \(r^{2}\) for a nudge of length \(r\): halve the nudge and the error falls fourfold.

```python
def jacobian(f, x, h=1e-6):
    """Central-difference Jacobian of f at the point x: entry [i][j] is d f_i / d x_j."""
    if h <= 0:
        raise ValueError("h must be positive")
    point = [float(v) for v in x]
    columns = []
    for j in range(len(point)):
        up, down = list(point), list(point)
        up[j] += h
        down[j] -= h
        columns.append([(a - b) / (2 * h) for a, b in zip(f(up), f(down))])
    return [[column[i] for column in columns] for i in range(len(columns[0]))]
```

`[[round(v, 6) for v in row] for row in jacobian(lambda v: [v[0] ** 2 + v[1], v[0] * v[1]], [2, 1])]` returns `[[4.0, 1.0], [1.0, 2.0]]`.

## In an iterated map

For an [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}) \(x_{n+1}=F(x_n)\), differentiable at a fixed point \(x^\ast=F(x^\ast)\), the Jacobian \(J\) at \(x^\ast\) governs every small displacement \(\delta_n=x_n-x^\ast\) to first order, \(\delta_{n+1}\approx J\,\delta_n\). Its eigenvalues are the fixed point's [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}). If every multiplier \(\mu\) has absolute value \(\lvert\mu\rvert<1\), the point attracts all states close enough to it; if any has \(\lvert\mu\rvert>1\), some small displacements grow; if none has \(\lvert\mu\rvert>1\) but some has \(\lvert\mu\rvert=1\), the Jacobian alone cannot decide, and the nonlinear terms of \(F\) do. Along an orbit \(x_0,x_1,x_2,\ldots\) that is not at rest, the Jacobians at successive states multiply, the latest on the left: their product around a cycle, an orbit that repeats, is the cycle's [monodromy matrix]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}), and the long-run exponential growth rates of such products along an orbit, where those limits exist, are its [Lyapunov exponents]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}).

## What the Jacobian does not tell you

The Jacobian is local. It describes the map near one point, for nudges small enough that the nonlinear remainder is negligible next to the nudge; a larger change can respond quite differently, and the Jacobian at another point can be entirely different. The partial derivatives can even exist where the map is not differentiable, and there \(J\) is no stand-in at all: \(f(x,y)=xy/(x^{2}+y^{2})\), with \(f(0,0)=0\), has both partial derivatives zero at the origin, yet equals \(1/2\) at every other point of the diagonal \(x=y\). At a fixed point, the eigenvalues describe what many repeated steps do to a small nudge, not how far one step can stretch it: a [non-normal]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}) Jacobian with small eigenvalues can still amplify some nudges many times over. A finite-difference Jacobian is only as accurate as its step \(h\) allows: too large a step adds curvature, too small a step adds rounding error.

See also: [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [forward-mode automatic differentiation]({{< relref "knowledge-base/glossary/forward-mode-automatic-differentiation/index.md" >}}), [Arnoldi iteration]({{< relref "knowledge-base/glossary/arnoldi-iteration/index.md" >}}), [block-triangular matrix]({{< relref "knowledge-base/glossary/block-triangular-matrix/index.md" >}}), [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}).
