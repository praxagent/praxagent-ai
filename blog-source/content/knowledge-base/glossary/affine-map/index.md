---
title: "Affine map"
slug: "affine-map"
summary: "A transformation that applies a linear map and then adds a fixed offset; it preserves affine combinations, whose coefficients sum to 1, but need not preserve the origin, or zero vector."
draft: false
pro_reviewed: true
---

An **affine map** applies a linear transformation and then adds a fixed offset. For a real-valued input vector with \(n\) coordinates and an output vector with \(m\) coordinates, an affine map has the form

\[
f(x)=Ax+b,
\]

where \(\mathbb{R}\) denotes the real numbers, \(A\in\mathbb{R}^{m\times n}\) is a matrix that linearly combines the input coordinates, for example to rotate, rescale, reduce the number of coordinates, or shear by slanting one coordinate in proportion to another, and \(b\in\mathbb{R}^{m}\) is a fixed **bias** or offset. The [matrix multiplication]({{< relref "knowledge-base/glossary/matmul/index.md" >}}) \(Ax\) determines how differences between inputs change; adding \(b\) determines where the transformed points sit relative to the origin, meaning the zero vector.

## A checkable example

Let

\[
x=
\begin{bmatrix}
2 \\
3
\end{bmatrix},
\qquad
A=
\begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
1 \\
4
\end{bmatrix}.
\]

Then

\[
f(x)=Ax+b=
\begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}\;\times\;
\begin{bmatrix}
2 \\
3
\end{bmatrix}\;+\;
\begin{bmatrix}
1 \\
4
\end{bmatrix}\;=\;
\begin{bmatrix}
5 \\
1
\end{bmatrix}.
\]

The matrix doubles the first coordinate and reverses the sign of the second; the bias then adds 1 and 4. In particular, \(f(0)=b\). This map is linear only when \(b=0\), because a linear map must send the zero vector to the zero vector.

## What affine maps preserve

For any two inputs \(x_1,x_2\) and any real number \(t\),

\[
f\bigl((1-t)x_1+t x_2\bigr)
=(1-t)f(x_1)+t f(x_2).
\]

Thus an affine map sends an entire straight line either to an entire straight line or to a single point; it sends a line segment either to a line segment or to a point. When the direction is not collapsed, meaning \(A(x_2-x_1)\ne0\), the same parameter \(t\) identifies corresponding points along the input and output lines. If the direction is collapsed, distinct points on the input line map to the same point. An affine map does not necessarily preserve lengths, angles, areas, or the origin. The equation allows values of \(t\) outside the interval from 0 to 1; restricting \(t\) to that interval describes points on the line segment between \(x_1\) and \(x_2\).

Differences cancel the bias:

\[
f(x_1)-f(x_2)=A(x_1-x_2).
\]

This is why \(A\) controls how small changes, or **perturbations**, and directions are transported even when the full map includes an offset.

## Affine maps in neural-network analysis

Neural-network libraries often call \(x\mapsto Ax+b\) a **linear layer**, although the operation is mathematically affine when \(b\ne0\). Such layers are building blocks of a [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}). An **activation function** is a function applied to a layer's intermediate values; inserting a nonlinear activation function between affine maps generally makes the resulting MLP nonlinear.

For a function \(h:\mathbb{R}^{n}\to\mathbb{R}^{m}\) that is differentiable at a chosen base input \(x_0\), there is a linear map for which the approximation error divided by the distance between \(x\) and \(x_0\) tends to zero as \(x\) approaches \(x_0\). Its first-order approximation is affine:

\[
g(x)=h(x_0)+J_h(x_0)(x-x_0)
=J_h(x_0)x+\bigl(h(x_0)-J_h(x_0)x_0\bigr),
\]

where \(J_h(x_0)\) is the Jacobian matrix, the matrix of first derivatives of \(h\) evaluated at \(x_0\). The approximation describes local behavior near \(x_0\); it does not make \(h\) globally affine.

In the construction called a [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}) in this glossary, Jacobian matrices of a specified later-computed, or **downstream**, quantity with respect to a specified internal state are averaged over a specified **corpus**, meaning a collection of examples. Each Jacobian is a local sensitivity: it maps a small change in the internal state to a first-order change in the downstream quantity. Reusing the average as a fixed **readout**, an analyst-applied map used to estimate a chosen target from internal activations, is an additional approximation; it does not show that the model uses that map in its own computation. Adding or fitting an offset makes the readout affine. Its predictive accuracy must be evaluated for the states and change sizes where it will be used.

## What an affine map does not establish

- Fitting an affine relation to observations does not show that the system is globally affine, nor does it establish a causal relation: by itself, the fit does not show that intervening to change the input would produce the mapped change in the output.
- An affine map need not be **invertible**: there may be no inverse map that recovers \(x\) from \(f(x)\). For \(f:\mathbb{R}^{n}\to\mathbb{R}^{n}\), the map is invertible if and only if \(Av=0\) has no solution other than the zero vector \(v=0\). Equivalently, every vector \(y\in\mathbb{R}^{n}\) can be written as \(y=Av\) for some \(v\in\mathbb{R}^{n}\). This condition is called **full rank**.
- The bias \(b\) is a geometric offset. It is unrelated to statistical or social bias unless a particular application gives it that meaning.
- A score produced by an affine map is not automatically a probability. For a finite set of mutually exclusive, exhaustive outcomes, a probability vector must have nonnegative entries that sum to 1. A transformation such as **softmax**, exponentiating every score and dividing each exponential by the sum of all the exponentials, enforces these algebraic constraints. It does not by itself guarantee **calibration**, a distribution-dependent empirical property under which, among cases assigned a given predicted probability or a narrow range around it, the relevant event occurs at approximately that frequency.

See also: [matmul]({{< relref "knowledge-base/glossary/matmul/index.md" >}}), [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}), [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}).
