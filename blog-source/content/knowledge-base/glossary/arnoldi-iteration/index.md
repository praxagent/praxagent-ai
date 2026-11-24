---
title: "Arnoldi iteration"
slug: "arnoldi-iteration"
summary: "A method that finds a few eigenvalues of a very large matrix using only its products with vectors, by building an orthonormal basis of the directions those products reach."
draft: true
pro_reviewed: false
og_image: "arnoldi-iteration-card.png"
og_image_alt: "Left: the eigenvalues of a 200 by 200 matrix in the complex plane, a bulk inside radius 0.55 and four outliers, with Ritz values after 10 and 40 steps landing on the outliers. Right: the distance from each outlier to its nearest Ritz value, with one curve for the complex-conjugate pair, falling geometrically to about ten digits by step 40."
---

**Arnoldi iteration** finds a few eigenvalues of an \(n\times n\) matrix \(A\) that is too large to write down, using nothing but products \(Av\) with chosen vectors. Take \(A\) real. Starting from a real vector \(q_1\) of unit length, it builds an orthonormal basis \(q_1,\dots,q_k\), unit vectors each perpendicular to all the others, of the *Krylov subspace*

\[
\mathcal{K}_k=\operatorname{span}\bigl\{q_1,\;Aq_1,\;A^{2}q_1,\;\dots,\;A^{k-1}q_1\bigr\},
\]

the directions that up to \(k-1\) applications of \(A\) reach from \(q_1\). Each step multiplies the newest basis vector by \(A\), subtracts from the product its components along every basis vector so far (the Gram–Schmidt process), and rescales what is left to unit length. The coefficients fill a small \(k\times k\) matrix \(H_k\) that is upper triangular apart from one line below the diagonal, a *Hessenberg* matrix, with

\[
A\,Q_k=Q_k H_k+h_{k+1,k}\,q_{k+1}e_k^{\mathsf T},
\]

where \(Q_k\) is the \(n\times k\) matrix with the basis vectors as columns, \(h_{k+1,k}\) is the length of the leftover part and \(q_{k+1}\) is that part rescaled, \(e_k=(0,\dots,0,1)\) has \(k\) entries, and \(\mathsf T\) marks a transpose. For a complex matrix the same steps run in complex arithmetic, with conjugate transposes in place of transposes.

The eigenvalues of \(H_k\), the **Ritz values**, approximate eigenvalues of \(A\). Those on the outer edge of the spectrum, the set of all eigenvalues of \(A\), usually converge first, fastest when they stand far from the rest. That suits a stability question, since a fixed point of a map attracts or repels according to the largest modulus (absolute value) among its [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}): the eigenvalues, at that point, of the map's [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}), its matrix of first partial derivatives. An eigenvalue that is near \(-1\) but not on the outer edge of the spectrum is not guaranteed to appear early.

The method never needs \(A\) itself, only a routine that returns \(Av\). For the Jacobian of a large map, [forward-mode automatic differentiation]({{< relref "knowledge-base/glossary/forward-mode-automatic-differentiation/index.md" >}}) supplies those products exactly up to rounding. Production solvers such as ARPACK, short for ARnoldi PACKage, restart implicitly: every so often they compress the basis to a few vectors that keep the wanted Ritz values and drop the others, then extend it again, so memory stays bounded while they converge.

{{< reference-figure
  src="arnoldi-iteration.png"
  alt="Left: the complex plane with the unit circle dashed and 200 true eigenvalues in gray, a bulk inside radius 0.55 and four outliers at minus 1.1, 0.95 and 0.75 plus or minus 0.55i; open blue circles mark Ritz values after 10 steps and red crosses after 40, both landing on the outliers while scattering through the bulk. Right: on a logarithmic axis, the distance from each outlier to its nearest Ritz value falls from about 1 to between 10 to the minus 10 and 10 to the minus 13 over 40 steps, in three curves with different line styles and markers: minus 1.1, 0.95, and one curve for the pair 0.75 plus or minus 0.55i."
  caption="Toy example, no language model: a 200 × 200 nonsymmetric matrix built with known eigenvalues, a bulk inside radius 0.55 and four outliers, −1.1, 0.95 and 0.75 ± 0.55i (i is the imaginary unit), probed only through products Av from a seeded random start. (a) After 10 steps the Ritz values (open circles) already sit near the four outliers; after 40 (crosses) the outliers are found to about ten digits or better, while the bulk is only loosely sketched. (b) Each outlier's error falls roughly geometrically, fastest for −1.1, the farthest from the bulk. The matrix and the start are real, so the Ritz values come in conjugate pairs and the two outliers 0.75 ± 0.55i share one curve. An eigenvalue inside the bulk would take far longer to resolve."
>}}

## Worked example

Take \(A=\operatorname{diag}(3,2,1)\) and \(q_1=(1,1,1)/\sqrt3\). The first step gives \(h_{11}=q_1^{\mathsf T}Aq_1=(3+2+1)/3=2\). What is left of \(Aq_1\) after removing its \(q_1\) part is \((1,0,-1)/\sqrt3\), of length \(h_{21}=\sqrt{2/3}\approx0.8165\), so \(q_2=(1,0,-1)/\sqrt2\). The second step gives \(h_{12}=q_1^{\mathsf T}Aq_2=2/\sqrt6\approx0.8165\) and \(h_{22}=q_2^{\mathsf T}Aq_2=(3+1)/2=2\). So

\[
H_2=\begin{bmatrix}2 & 0.8165\\ 0.8165 & 2\end{bmatrix},
\]

whose eigenvalues, \(2\pm0.8165\), are 2.8165 and 1.1835: after two products, already close to the extreme eigenvalues 3 and 1. In exact arithmetic a third step recovers 3, 2 and 1 exactly; in floating point it recovers them to rounding error.

The function below works in real arithmetic; a complex version needs complex arrays and the conjugating inner product `np.vdot`.

```python
import numpy as np

def arnoldi(matvec, v0, k):
    """k real Arnoldi steps from v0. Returns Q, with orthonormal columns, and H, whose leading k-by-k block
    holds the Hessenberg matrix; its eigenvalues are the Ritz values. Stops early, with a square H, if the
    new direction is negligible next to A q, so that the Krylov space has stopped growing to rounding
    accuracy. Even then a Ritz value is an approximation: check its residual."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    q = np.asarray(v0, dtype=float)
    if q.ndim != 1 or not np.any(q):
        raise ValueError("v0 must be a nonzero vector")
    Q = np.zeros((q.size, k + 1))
    H = np.zeros((k + 1, k))
    Q[:, 0] = q / np.linalg.norm(q)
    for j in range(k):
        w = np.asarray(matvec(Q[:, j]), dtype=float)
        scale = np.linalg.norm(w)
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ w
            w = w - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] <= 1e-12 * scale:
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = w / H[j + 1, j]
    return Q, H
```

`Q, H = arnoldi(lambda v: np.diag([3.0, 2.0, 1.0]) @ v, np.ones(3), 2)`, then `np.sort(np.linalg.eigvals(H[:2, :2]).real).round(4)` returns `array([1.1835, 2.8165])`. With `k=3` the function detects that the space has stopped growing and returns a square 3-by-3 `H`, whose eigenvalues match 1, 2 and 3 to within \(10^{-14}\).

## When a handful of steps is enough

In a stability calculation the matrix is often huge but its spectrum is lopsided: one or two multipliers matter and the rest are tiny. Arnoldi iteration then needs very few products. Each multiplication by \(A\) shrinks the components along the small eigenvalues far faster than the component along the large one, so after a few steps the Krylov subspace already contains the dominant eigenvector almost exactly, and \(H_k\) has a Ritz value sitting on its eigenvalue.

A toy example, with no language model involved: a \(500\times500\) nonsymmetric matrix built with one eigenvalue at \(-0.5\) and the other 499 drawn uniformly from \([-0.003,\,0.003]\), probed only through products \(Av\) from a seeded random start. For each number of steps \(k\), the table gives the Ritz value of largest modulus, \(\theta\), and the residual \(\lVert Ay-\theta y\rVert\) of its unit Ritz vector \(y\).

| Products \(k\) | Largest Ritz value \(\theta\) | Residual |
|---:|---:|---:|
| 1 | \(+0.000195\) | \(8.9\times10^{-3}\) |
| 2 | \(-0.476966\) | \(1.0\times10^{-1}\) |
| 3 | \(-0.499998\) | \(3.2\times10^{-4}\) |
| 4 | \(-0.500000\) | \(1.0\times10^{-6}\) |

Four products recover the eigenvalue to six digits, and the residual says so without access to the true answer. The matrix itself is used only to build the example and to check the residual; in a real problem the residual costs one more product. The \(k=2\) row shows why the residual matters: its Ritz value is already close, but the residual is the largest in the table, because the Ritz vector still carries a sizable part of the small-eigenvalue directions.

```python
rng = np.random.default_rng(1)
n = 500
eigs = np.concatenate([[-0.5], rng.uniform(-0.003, 0.003, n - 1)])
P = np.eye(n) + 0.3 * rng.standard_normal((n, n)) / np.sqrt(n)
A = P @ np.diag(eigs) @ np.linalg.inv(P)        # hidden; only A @ v is used below
v0 = rng.standard_normal(n)
for k in (1, 2, 3, 4):
    Q, H = arnoldi(lambda v: A @ v, v0, k)
    theta, Y = np.linalg.eig(H[:k, :k])
    i = np.argmax(abs(theta))
    y = Q[:, :k] @ Y[:, i].real
    y /= np.linalg.norm(y)
    print(k, f"{theta[i].real:+.6f}", f"{np.linalg.norm(A @ y - theta[i].real * y):.1e}")
```

With a gap this wide, plain *power iteration*, which repeatedly multiplies one vector by \(A\) and watches how much it stretches, would also find \(-0.5\) in a few products. Arnoldi iteration earns its keep because the same products also give the next few Ritz values, including complex-conjugate pairs that power iteration cannot separate, each with its own residual. A small residual certifies that \(\theta\) is an eigenvalue of \(A\); it does not certify that no larger eigenvalue was missed, which is the subject of the next section.

## What Ritz values do not tell you

Arnoldi iteration finds extreme and outlying eigenvalues first, and it may need many steps, or in practice never get there, for eigenvalues inside a dense cluster. A run that stops early can therefore miss the eigenvalue that matters: for a stability question, a missed multiplier near \(-1\) or outside the unit circle makes a state look safer than it is. To aim at a chosen point \(\sigma\), such as \(-1\), *shift-invert* Arnoldi runs the same steps on \((A-\sigma I)^{-1}\), with \(I\) the identity matrix: an eigenvalue \(\mu\) of \(A\) becomes \(1/(\mu-\sigma)\), largest for \(\mu\) nearest \(\sigma\), at the cost of solving a linear system at every step. For strongly [non-normal]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}) matrices, whose eigenvectors can be nearly parallel, Ritz values can wander before they settle and sit far from any eigenvalue for many steps. In floating point the basis slowly loses orthogonality, so practical codes reorthogonalize by repeating the subtraction. Before a Ritz value \(\theta\) is trusted, it should be confirmed by the residual \(\lVert Av-\theta v\rVert\) of its Ritz vector \(v=Q_k y\), where \(y\) is a unit eigenvector of \(H_k\) for \(\theta\).

The [field guide to the dynamics of looped transformers]({{< relref "posts/2026/11/looped-transformer-dynamics/index.md" >}}) finds the leading multipliers of looped transformers with an Arnoldi eigensolver fed by Jacobian-vector products.

See also: [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}), [forward-mode automatic differentiation]({{< relref "knowledge-base/glossary/forward-mode-automatic-differentiation/index.md" >}}), [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}).
