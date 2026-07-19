---
title: "Matmul"
slug: "matmul"
summary: "Matrix multiplication: the core linear-algebra operation behind attention, multilayer perceptron layers, final hidden-state-to-vocabulary projections often called unembeddings, and local linear maps represented by Jacobian matrices. Basic Linear Algebra Subprograms (BLAS) libraries commonly implement matrix-matrix cases with general matrix multiplication (GEMM) routines."
draft: false
pro_reviewed: false
aliases:
  - /references/matmul/
  - /references/matrix-multiplication/
  - /references/gemm/
---

A **matmul** (matrix multiplication) combines two arrays of numbers by
multiplying rows of the first with columns of the second and summing. For
\(A\in\mathbb{R}^{m\times k}\) and \(B\in\mathbb{R}^{k\times n}\),

\[
(AB)_{ij}=\sum_{t=1}^{k} A_{it}\,B_{tj}.
\]

Here \(\mathbb{R}\) denotes real-valued entries; \(i\) indexes an output row,
\(j\) indexes an output column, and \(t\) runs over the \(k\) entries in the
shared dimension. The shared size \(k\) must match. The product has shape
\(m\times n\). A **matrix-vector** product such as \(Wh\) is the same operation
with the vector treated as a single-column matrix.

Libraries use several related interfaces:

| Name | Where you see it | Meaning |
|---|---|---|
| **matmul** | NumPy `np.matmul`, PyTorch `torch.matmul` / `@` | General matrix multiply, including vector and batched cases |
| **general matrix multiplication (GEMM)** | Basic Linear Algebra Subprograms (**BLAS**), including NVIDIA's Compute Unified Device Architecture (**CUDA**) BLAS library (**cuBLAS**) | A matrix-matrix routine that may scale the product and add it into an existing output matrix; highly tuned central processing unit (**CPU**) and graphics processing unit (**GPU**) kernels implement this operation |
| **linear layer** | `nn.Linear` | Matmul plus optional bias: \(y = x W^{\mathsf T} + b\), where \(x\) is an input row or batch, \(W\) is the learned weight matrix, \(W^{\mathsf T}\) is its transpose, \(b\) is the bias, and \(y\) is the output |


## Worked example

\[
A=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix},
\qquad
AB=
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}.
\]

The top-left entry of \(AB\) is the first row of \(A\) dotted with the first
column of \(B\):

\[
1\cdot 5 + 2\cdot 7 = 19.
\]

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
C = A @ B  # same as np.matmul(A, B)
# [[19., 22.], [43., 50.]]
```

Cost grows quickly: for the standard algorithm, multiplying two \(d\times d\)
matrices, where \(d\) is their width and height, uses on the order of \(d^{3}\)
floating-point operations (**FLOPs**, naive count). In a standard dense
Transformer, most arithmetic FLOPs in a forward pass come from matmuls.
That is why GPU **Tensor Cores** (specialized matrix-multiply units) and
reduced-precision formats such as [bfloat16 (BF16) and 16-bit floating point
(FP16)]({{< relref "bf16.md" >}}) matter for large models. Typical dense
sources include attention projections, attention scores,
[multilayer perceptron (MLP)]({{< relref "mlp.md" >}}) up/down projections, and
the final hidden-state-to-vocabulary projection (the
[unembedding]({{< relref "unembedding.md" >}})).

## Where these notes use matmuls

- **Model internals:** a standard dense Transformer (an architecture built from
  attention modules that form data-dependent weighted combinations of token
  representations and feed-forward layers) often spends most of its arithmetic
  on matmuls. Its blocks also contain normalization, residual additions,
  masking, softmax, and pointwise nonlinearities.
- **Jacobian and identity transport:** for a function \(f\) and base state
  \(h_0\), the Jacobian \(J=\partial f/\partial h\) evaluated at \(h_0\) maps a
  small residual-stream perturbation \(\Delta h\) to the first-order output
  change \(J\Delta h\). With row-vector storage, this is `delta_h @ J.T`, where
  `.T` denotes matrix transpose. Predicting the output itself also requires the
  base value: \(f(h_0+\Delta h)\approx f(h_0)+J\Delta h\). Identity transport
  uses \(I\Delta h=\Delta h\). If `residual @ J.T` is instead applied to an
  uncentered residual state as a lens, treat it as a linear readout rather than
  as the Jacobian approximation to \(f\).
- **Precision and determinism:** the operand or storage dtype, such as 32-bit
  floating point (**FP32**) or bfloat16 (**BF16**); the multiplication mode,
  such as TensorFloat-32 (**TF32**) for FP32 inputs; the accumulation
  precision; and the selected cuBLAS algorithm can all change result bits. See
  [BF16, FP16, and FP32]({{< relref "bf16.md" >}}) and the Deep Dive
  [Precision, Range, Swamping, and Determinism]({{< relref "knowledge-base/deep-dives/bf16-fp16-fp32/index.md" >}}).

## What "matmul" does not establish

- It does not by itself include a following nonlinearity such as the Gaussian
  error linear unit (**GELU**), sigmoid linear unit (**SiLU**), or softmax, an
  exponentiate-and-normalize operation that produces nonnegative entries
  summing to one.
- It does not specify the dtype, the memory layout or arrangement of elements
  in memory, or whether a bias is added.
- Saying "we ran a matmul" does not imply a particular GPU kernel or bit-exact
  match across libraries.

See also: [residual stream]({{< relref "residual-stream.md" >}}),
[unembedding]({{< relref "unembedding.md" >}}),
[BF16, FP16, and FP32]({{< relref "bf16.md" >}}),
[BF16 Deep Dive]({{< relref "knowledge-base/deep-dives/bf16-fp16-fp32/index.md" >}}).
