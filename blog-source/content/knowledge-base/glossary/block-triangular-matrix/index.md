---
title: "Block-triangular matrix"
slug: "block-triangular-matrix"
summary: "A matrix whose blocks on one side of the diagonal are all zero; its eigenvalues are exactly those of its diagonal blocks."
draft: true
pro_reviewed: true
og_image: "block-triangular-matrix-card.png"
og_image_alt: "A four-by-four grid of token blocks with zeros above the diagonal and the third diagonal block marked unstable, beside a staircase plot in which the running minimum of the token margins drops below zero at token 3."
---

A **block lower-triangular matrix** is a matrix divided into square blocks along its diagonal, with every block above the diagonal equal to zero:

\[
J=\begin{bmatrix}
D_1 & 0 & 0\\
B_{21} & D_2 & 0\\
B_{31} & B_{32} & D_3
\end{bmatrix},
\]

where each \(D_t\) is a square block on the diagonal and each \(B_{tu}\) is an arbitrary block below it. A block upper-triangular matrix is the mirror image. The determinant of such a matrix is the product of the determinants of its diagonal blocks, so

\[
\det(J-\mu I)=\prod_{t}\det(D_t-\mu I),
\]

where \(I\) is the identity matrix of matching size, and the eigenvalues of \(J\) are exactly the eigenvalues of \(D_1,D_2,\ldots\), taken together. The blocks below the diagonal do not change the eigenvalues, although they can change the eigenvectors and how much the matrix can stretch a vector in one step.

This structure appears whenever influence runs only one way. With [causal attention]({{< relref "knowledge-base/glossary/causal-attention/index.md" >}}), every position reads only itself and earlier positions. A [looped transformer]({{< relref "knowledge-base/glossary/looped-transformer/index.md" >}}) applies the same pass again and again, and a fixed point is a state that one pass leaves unchanged. If the model's attention is causal, the [Jacobian]({{< relref "knowledge-base/glossary/jacobian/index.md" >}}) of one pass at a fixed point, the matrix of its first derivatives, is block lower-triangular when its rows and columns are grouped by token in order: \(D_t\) is token \(t\)'s response to its own state with every earlier token held fixed, and \(B_{tu}\), for \(u\lt t\), is the influence of an earlier token \(u\) on a later token \(t\). Three consequences follow.

- The [multipliers]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}) of the fixed point, the eigenvalues of the whole Jacobian, are those of the token blocks \(D_t\) taken together, and the fixed point attracts nearby states when all of them lie inside the unit circle, that is, have magnitude below 1. Each token therefore has its own [stability margin]({{< relref "knowledge-base/glossary/stability-edge/index.md" >}}) \(m_t\), computed from the multipliers of \(D_t\) alone and positive exactly when they all lie inside the unit circle, and a passage of \(n\) tokens has margin \(M(n)=\min_{t\le n}m_t\), a running minimum.
- Appending tokens adds blocks without changing the existing ones, provided the earlier tokens keep the same fixed point, so \(M(n)\) can only fall as the text grows.
- Suppose each token \(t\) moves its state \(s_t\) only a fraction \(\lambda_t\), its **step**, of the way to its new value, \(s_t\mapsto(1-\lambda_t)\,s_t+\lambda_t\,G_t(s)\), where \(G_t(s)\) is token \(t\)'s part of one pass and every token is updated at once from the same old state \(s\). Changing \(\lambda_t\) alone then changes only block row \(t\): \(D_t\) becomes \((1-\lambda_t)I+\lambda_tD_t\), the blocks to its left are scaled by \(\lambda_t\), and every other token's blocks are untouched.

{{< reference-figure
  src="block-triangular-matrix.svg"
  alt="A four-by-four grid of token blocks: diagonal blocks D1 to D4, starred blocks below the diagonal, zeros above, and D3 outlined and labeled unstable. Beside it, a plot of toy token margins 0.40, 0.25, -0.10 and 0.30 as dots and their running minimum as a line that drops below zero at token 3 and stays there."
  caption="Toy values, not a measured passage. Left: the Jacobian of four tokens. Diagonal blocks D1 to D4 hold each token's response to itself; starred blocks below the diagonal are effects of earlier tokens on later ones; blocks above are zero. D3 has a multiplier outside the unit circle. Right: the toy token margins 0.40, 0.25, -0.10 and 0.30 (dots) and their running minimum (line), which drops below zero at token 3 and stays there even though token 4 is stable."
>}}

## Worked example

Toy values, no model. With two tokens of one number each,

\[
J=\begin{bmatrix}0.5 & 0\\ 3 & -1.2\end{bmatrix}
\]

has eigenvalues 0.5 and \(-1.2\), whatever the 3 below the diagonal. For a real multiplier \(\mu<1\), the stability edge is \(\lambda_c=2/(1-\mu)\), the largest step at which the fixed point still attracts, and the margin is \(m=\lambda_c-1\), how far the plain pass, step 1, sits inside the edge. Token 1's margin is \(2/0.5-1=3\) and token 2's is \(2/2.2-1\approx-0.091\), so \(M(2)\approx-0.091\): the pair fails at token 2. Give token 2 alone the step \(\lambda_2=0.8\). Its row becomes 0.2 times the identity row plus 0.8 times the old row:

\[
J'=\begin{bmatrix}0.5 & 0\\ 2.4 & -0.76\end{bmatrix},
\]

with eigenvalues 0.5 and \(-0.76\), both inside the unit circle, while token 1 is untouched.

```python
def lower_triangular_eigenvalues(rows):
    """Eigenvalues of a lower-triangular matrix with 1-by-1 blocks: its diagonal entries."""
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise ValueError("need a nonempty square matrix")
    if any(rows[i][j] != 0 for i in range(n) for j in range(i + 1, n)):
        raise ValueError("entries above the diagonal must be zero")
    return [rows[i][i] for i in range(n)]


def running_minimum(values):
    """M(n) = min(values[:n]) for n = 1, 2, ..."""
    if not values:
        raise ValueError("need at least one value")
    out, low = [], float("inf")
    for v in values:
        low = min(low, v)
        out.append(low)
    return out
```

`lower_triangular_eigenvalues([[0.5, 0], [3, -1.2]])` returns `[0.5, -1.2]`, and `running_minimum([3.0, -0.5, 1.0])` returns `[3.0, -0.5, -0.5]`.

## What the triangular structure does not tell you

The structure fixes the eigenvalues, not the size of transients: large blocks below the diagonal can make the matrix strongly [non-normal]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}). The statement that the margin can only fall assumes that appending text leaves the earlier tokens' fixed point unchanged. And if an eigenvalue belongs to one diagonal block \(D_t\) only, its eigenvector is zero before token \(t\) but can be largest at a later token, so locate an unstable token by its block, not by where the eigenvector is largest; if several blocks share the eigenvalue, assigning it to one token can be ambiguous.

The [field guide to the dynamics of looped transformers]({{< relref "posts/2026/11/looped-transformer-dynamics/index.md" >}}) uses this structure to name the token at which a long passage stops settling.

See also: [causal attention]({{< relref "knowledge-base/glossary/causal-attention/index.md" >}}), [multiplier]({{< relref "knowledge-base/glossary/multiplier/index.md" >}}), [stability edge and margin]({{< relref "knowledge-base/glossary/stability-edge/index.md" >}}), [non-normal matrix]({{< relref "knowledge-base/glossary/non-normal-matrix/index.md" >}}).
