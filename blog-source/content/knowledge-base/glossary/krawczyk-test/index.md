---
title: "Krawczyk's test"
slug: "krawczyk-test"
summary: "An interval Newton test: if Krawczyk's operator maps a box strictly inside itself, the box contains exactly one solution."
draft: false
pro_reviewed: true
og_image: "krawczyk-test-card.png"
og_image_alt: "The interval from 1 to 2 is tested. Krawczyk's image, from 1.25 to about 1.583, sits strictly inside it, so the box contains one root."
---

**Krawczyk's test** is a way to prove that a system of equations has exactly one solution inside a box, using interval arithmetic rather than a rounded Newton iteration. A floating-point Newton iteration can return an accurate-looking point, but that alone does not prove that an exact solution exists nearby, or that it is the only one. Certified interval arithmetic replaces each number by an interval, using exact operations or rounding outward, so that every computed interval is guaranteed to contain the exact value it stands for. Those intervals are carried through the function and its derivative.

Let \(f\) map \(\mathbb{R}^n\) to \(\mathbb{R}^n\) and be continuously differentiable near a box \(X = X_1 \times \cdots \times X_n\), a product of closed intervals, one per coordinate. Pick a point \(x\) in \(X\), usually its center, and an \(n \times n\) matrix \(A\) close to the inverse of \(Df(x)\), the Jacobian matrix of partial derivatives of \(f\) at \(x\). Let \(I\) be the \(n \times n\) identity matrix, and let \(Df(X)\) be an interval matrix that contains \(Df(y)\) for every \(y\) in \(X\). Computed with interval operations, the Krawczyk operator is

\[
K(X)=x-Af(x)+\bigl(I-A\,Df(X)\bigr)(X-x).
\]

If \(K(X)\) lies strictly inside \(X\), then \(X\) contains exactly one root of \(f\).

The reason is a fixed-point argument. Write \(g(y)=y-Af(y)\), one Newton-like step with the fixed matrix \(A\); when \(A\) is invertible, the fixed points of \(g\) are exactly the roots of \(f\). For \(y\) in \(X\), \(g(y)=g(x)+(I-AM)(y-x)\) for a matrix \(M\) in \(Df(X)\), the average of the derivative along the segment from \(x\) to \(y\). So \(g(y)\) lies in \(K(X)\), and when \(K(X)\) sits inside \(X\), \(g\) maps the box into itself. The same enclosure bounds how much \(g\) can stretch. If every matrix in \(I-A\,Df(X)\) has norm below 1, \(g\) shrinks the distance between any two points of \(X\), \(A\) is invertible, and Banach's contraction-mapping theorem gives exactly one fixed point. Mapping the box strictly into itself is not enough on its own: a continuous map can do that and still cross the diagonal several times. In the centered form below, strict inclusion already forces the norm bound in a norm weighted by the box radii, and implementations commonly check both.

{{< reference-figure
  src="krawczyk-test.svg"
  alt="A number line shows the trial box from 1 to 2. A shorter interval from 1.25 to about 1.583, the Krawczyk image, lies strictly inside it."
  caption="One-dimensional worked example, f(x) = x squared minus 2 on the box [1, 2]. The midpoint is 1.5. Krawczyk's operator sends the box to [5/4, 19/12], about [1.25, 1.583], which sits strictly inside [1, 2]. The test therefore proves that this box contains exactly one root. The root is the square root of 2. The drawing is this arithmetic, not an orbit of a language model."
>}}

## Worked example

Let \(f(x)=x^2-2\) and \(X=[1,2]\). The midpoint is \(x=3/2\), \(f(3/2)=1/4\), and \(f'(x)=2x\), so \(f'(X)=[2,4]\). Take \(A=1/f'(3/2)=1/3\). Then

\[
\begin{aligned}
x-Af(x) &= \tfrac{3}{2}-\tfrac{1}{12}=\tfrac{17}{12},\\
I-A\,f'(X) &= \bigl[-\tfrac13,\tfrac13\bigr],\\
X-x &= \bigl[-\tfrac12,\tfrac12\bigr].
\end{aligned}
\]

The product of those last two intervals is \([-1/6,1/6]\), so

\[
K(X)=\bigl[\tfrac{17}{12}-\tfrac16,\ \tfrac{17}{12}+\tfrac16\bigr]=\bigl[\tfrac54,\tfrac{19}{12}\bigr]\approx[1.25,1.583].
\]

This sits strictly inside \([1,2]\), and the factor \(I-A\,f'(X)\) has magnitude at most \(1/3<1\), so the box contains exactly one root. The function below recomputes that interval in the centered form that the multidimensional test uses. For a square system \(H\), take the box center \(c\), the box of offsets \(\Delta=X-c\), and an approximate inverse \(R\) of \(DH(c)\). The centered image \(-RH(c)+(I-R\,DH(X))\Delta\) must lie strictly inside \(\Delta\), and \(I-R\,DH(X)\) must have norm below 1; in one dimension the norm is the absolute value. The test does not evaluate the map: the caller supplies the enclosed value \(H(c)\) and derivative \(DH(X)\).

```python
from fractions import Fraction


def _mul(left, right):
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def centered_krawczyk(center, radius, h_at_center, derivative_box, preconditioner):
    """Scalar centered Krawczyk test.

    image = -R H(c) + (1 - R DH(X)) * [-r, r]
    Passes only when that image lies strictly inside [-r, r]
    and |1 - R DH(X)| < 1.
    """
    c = Fraction(center)
    r = Fraction(radius)
    scale = Fraction(preconditioner)
    value = Fraction(h_at_center)
    derivative = (Fraction(derivative_box[0]), Fraction(derivative_box[1]))
    defect = (1 - scale * derivative[1], 1 - scale * derivative[0])
    contraction = max(abs(defect[0]), abs(defect[1]))
    correction = _mul(defect, (-r, r))
    shift = (-scale * value + correction[0], -scale * value + correction[1])
    inside = max(abs(shift[0]), abs(shift[1])) < r
    return {
        "passed": contraction < 1 and inside,
        "image": (c + shift[0], c + shift[1]),
        "contraction": contraction,
    }


def quadratic_krawczyk():
    """f(x) = x^2 - 2 on [1, 2]. H and DH are enclosed, then tested."""
    low, high = Fraction(1), Fraction(2)
    center = (low + high) / 2
    radius = (high - low) / 2
    h_at_center = center * center - 2
    derivative_box = (2 * low, 2 * high)  # f'(x) = 2x is increasing
    preconditioner = 1 / (2 * center)
    return centered_krawczyk(
        center, radius, h_at_center, derivative_box, preconditioner
    )
```

`quadratic_krawczyk()` returns `passed=True` and image \((5/4, 19/12)\).

In the [note on chaos in a language model's feedback loop]({{< relref "posts/2026/10/chaos-bifurcations-language-model-feedback/index.md" >}}), the unknown is not a single number. It is a period-8 orbit of the soft-token map \(F\), whose states have 896 coordinates. The orbit is found as a root of a multiple-shooting system \(H(x_0,\ldots,x_7)\) whose eight blocks require \(F(x_i)=x_{i+1}\), with \(x_8=x_0\). A root is a cycle of eight states, each mapped to the next, so its period divides 8; separate checks rule out periods 1, 2 and 4. The derivative \(DH\) in the test is assembled from the eight one-step derivatives \(DF(x_i)\). Their product around the cycle is the [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}), which decides whether the orbit attracts, not whether it exists. Ball arithmetic encloses \(H\) and \(DH\) over boxes of radius \(2^{-640}\) around refined centers, and the centered test passes, with the norm taken as the largest row sum of absolute values weighted by the box radii. That proves exactly one root in that product of boxes, for the real-number map of the stored weights. It does not prove that every orbit in the note exists, and it does not prove that any cell is chaotic.

The test is from Krawczyk (1969), [*Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken*](https://doi.org/10.1007/BF02234767), *Computing* 4(3), 187–201.

See also: [monodromy]({{< relref "knowledge-base/glossary/monodromy/index.md" >}}), [Brouwer's fixed-point theorem]({{< relref "knowledge-base/glossary/brouwer-fixed-point/index.md" >}}).
