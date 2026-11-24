---
title: "Non-normal matrix"
slug: "non-normal-matrix"
summary: "A matrix that does not commute with its conjugate transpose; its eigenvalues can all be small while a single step still stretches some vectors."
draft: false
pro_reviewed: true
og_image: "non-normal-matrix-card.png"
og_image_alt: "Length of A to the power k times v against k for a toy 2-by-2 matrix whose eigenvalues are both 0.5: the length jumps from 1 to about 4 at the first step, stays near 4, then decays, while the eigenvalue rate 0.5 to the k falls steadily from the start."
---

A square matrix \(A\) is **normal** if it commutes with its conjugate transpose, \(AA^{*}=A^{*}A\), where \(A^{*}\) is the transpose of \(A\) with every entry replaced by its complex conjugate (for a real matrix, simply the transpose). Real symmetric matrices (equal to their transpose), real orthogonal ones (those that preserve length) and diagonal ones are normal. An eigenvector of \(A\) is a nonzero vector \(x\) that \(A\) only rescales, \(Ax=\mu x\), and the factor \(\mu\) is its eigenvalue. A normal matrix has a set of mutually orthogonal eigenvectors that spans the whole space, and the most it can stretch a vector in one step equals the largest modulus (absolute value) of its eigenvalues. A **non-normal** matrix loses these guarantees: its eigenvectors can be nearly parallel, or too few to span the space as in the example below, and one step can stretch some vectors far more than any eigenvalue suggests.

Two numbers separate the long run from a single step:

\[
\rho(A)=\max_i\lvert\mu_i\rvert,
\qquad
\sigma_{\max}(A)=\max_{v\ne0}\frac{\lVert Av\rVert}{\lVert v\rVert},
\qquad
\rho(A)\le\sigma_{\max}(A),
\]

where the \(\mu_i\) are the eigenvalues of \(A\), \(\rho(A)\) is its spectral radius, \(\lVert\cdot\rVert\) is ordinary Euclidean length, and \(\sigma_{\max}(A)\) is the largest singular value, the most \(A\) can stretch any vector in one step. The spectral radius sets the long-run rate: the largest stretch over \(k\) steps, \(\sigma_{\max}(A^{k})\), satisfies \(\sigma_{\max}(A^{k})^{1/k}\to\rho(A)\) as \(k\) grows (Gelfand's formula). For a normal matrix the two numbers are equal. For a non-normal one \(\sigma_{\max}\) can be much larger, and when \(\rho<1<\sigma_{\max}\) a disturbance can first grow, sometimes by a large factor, before it decays. This is called **transient growth**.

{{< reference-figure
  src="non-normal-matrix.png"
  alt="Length of A to the power k times v against k from 0 to 14 for A = [[0.5, 4], [0, 0.5]] and v = (0, 1): the brown curve jumps from 1 to about 4.03 at the first step, stays near 4 at step 2 and then decays toward zero, while a blue dashed curve 0.5 to the k falls steadily from 1. A gray dashed line at 4.06 marks the largest one-step stretch."
  caption="Toy example, no language model. The brown curve is the length of A^k v for A = [[0.5, 4], [0, 0.5]] and v = (0, 1); the blue dashed curve is 0.5^k, the decay the eigenvalues alone would suggest. The length jumps to about 4.03 at the first step, close to the largest one-step stretch of A, 4.06 (gray dashed line), stays near 4 for one more step and then decays, falling below 1 only at step 6. The eigenvalues still decide the long run: the length tends to zero."
>}}

## Worked example

Toy matrix, no model:

\[
A=\begin{bmatrix}0.5 & 4\\ 0 & 0.5\end{bmatrix},
\qquad
v=\begin{bmatrix}0\\ 1\end{bmatrix}.
\]

Both eigenvalues are 0.5, so \(\rho(A)=0.5\), but \(\sigma_{\max}(A)\approx4.0616\). Repeated application gives \(A^{k}v=\bigl(8k\cdot0.5^{k},\;0.5^{k}\bigr)\), whose lengths for \(k=0\) to 7 are 1, 4.031, 4.008, 3.003, 2.001, 1.250, 0.750 and 0.438. The eigenvalue rate alone, \(0.5^{k}\), would give 1, 0.5, 0.25 and so on. The length falls below its starting value only at step 6.

```python
import math

def lengths_of_powers(a, v, steps):
    """Euclidean lengths of v, Av, A^2 v, ... for a 2-by-2 matrix a."""
    if len(a) != 2 or any(len(row) != 2 for row in a) or len(v) != 2:
        raise ValueError("a must be 2-by-2 and v must have length 2")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    x, y = float(v[0]), float(v[1])
    out = [math.hypot(x, y)]
    for _ in range(steps):
        x, y = a[0][0] * x + a[0][1] * y, a[1][0] * x + a[1][1] * y
        out.append(math.hypot(x, y))
    return out
```

`[round(n, 3) for n in lengths_of_powers([[0.5, 4], [0, 0.5]], [0, 1], 7)]` returns `[1.0, 4.031, 4.008, 3.003, 2.001, 1.25, 0.75, 0.438]`.

## What non-normality does and does not imply

Non-normality is not instability. With \(\rho(A)<1\), every disturbance eventually decays, and some norm (a measure of vector length) adapted to \(A\), generally not the Euclidean one, shrinks every nonzero vector at every step. Nor does non-normality by itself mean growth: a non-normal matrix can have \(\sigma_{\max}<1\) and shrink every nonzero vector in one step. What fails is the inference from \(\rho<1\) to one-step Euclidean contraction, \(\sigma_{\max}<1\): convergence-rate and error bounds that need such contraction apply only once \(\sigma_{\max}<1\) has been checked directly. In many dimensions the transient factor can be large, and eigenvalues alone hide it. For an [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}), a rule applied again and again to its own output, the [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}) at a fixed point (a state the map leaves unchanged) is the matrix of first derivatives there, and its eigenvalues are the fixed point's [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}). So a fixed point can attract while one step still stretches some small disturbances.

See also: [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [stability edge and margin]({{< relref "knowledge-base/glossary/stability-edge/index.md" >}}), [block-triangular matrix]({{< relref "knowledge-base/glossary/block-triangular-matrix/index.md" >}}).
