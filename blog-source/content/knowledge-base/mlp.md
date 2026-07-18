---
title: "Multilayer perceptron (MLP)"
slug: "mlp"
summary: "The position-wise nonlinear feed-forward sublayer that transforms each token state and writes an update to the residual stream."
pro_reviewed: false
---

A **multilayer perceptron (MLP)** is a learned function composed of affine maps, matrix multiplications followed by optional additions of bias vectors, and typically nonlinear activation functions applied to intermediate coordinates. Neural-network software often calls the affine maps *linear layers*; an affine map with zero bias is mathematically linear. In [Transformer]({{< relref "transformer.md" >}}) discussions, **MLP** usually names the block's position-wise [feed-forward network (FFN)]({{< relref "feed-forward-network.md" >}}): the same learned function transforms every token position and returns an update with the model's hidden width.

Unlike attention, a standard Transformer MLP does not directly mix token positions. It acts on one position at a time. Its input can still contain information about other tokens because attention or another sequence-mixing operation may already have written that context into the token's [residual-stream]({{< relref "residual-stream.md" >}}) vector, the running hidden-state vector carried at that position.

{{< reference-figure
  src="knowledge-base/glossary/mlp-standard-and-gated.svg"
  alt="A standard multilayer perceptron sends one token vector through an input projection, a nonlinear activation, and an output projection; a gated variant uses gate and value branches that are multiplied before the output projection."
  caption="Two common Transformer MLP structures. In the standard path, one token-position state is projected to a model-specific intermediate width, transformed by an elementwise nonlinear activation, and projected back to a model-width update. In the gated path, separate gate and value projections are combined by elementwise multiplication before the down projection. Both paths return an update with the residual stream's model width. Biases are omitted, and the illustration does not imply a universal intermediate width, activation, gating rule, or parameter name."
>}}

## A common form

Let \(\mathbb{R}\) denote the set of real numbers. For a column vector \(h_t\in\mathbb{R}^{d_{\text{model}}}\) at token position \(t\), a common MLP form is

\[
\operatorname{MLP}(h_t)
= W_{\text{out}}\,\phi(W_{\text{in}}h_t+b_{\text{in}})+b_{\text{out}}.
\]

Here \(d_{\text{model}}\) is the model's hidden width and \(d_{\text{mlp}}\) is the MLP's intermediate width. The matrix \(W_{\text{in}}\in\mathbb{R}^{d_{\text{mlp}}\times d_{\text{model}}}\) maps the token vector into that intermediate space, \(\phi\) is an elementwise activation function, and \(W_{\text{out}}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{mlp}}}\) maps the result back to the residual width. The vectors \(b_{\text{in}}\) and \(b_{\text{out}}\) are optional biases.

Within one layer, the same \(W_{\text{in}}\), \(W_{\text{out}}\), and biases are used at every token position. Different layers normally have different parameters. The original Transformer described this component as a position-wise feed-forward network with two learned affine transformations, called linear transformations in the paper, and a rectified linear unit (ReLU), the elementwise function \(x\mapsto\max(0,x)\) ([Vaswani et al., 2017, Section 3.3](https://arxiv.org/abs/1706.03762v7)).

## Shape check

This example uses PyTorch, a Python library for tensors (multidimensional arrays of values) and neural networks. The imported name `nn` refers to PyTorch's neural-network module. `nn.Sequential` runs the listed modules in order. The Gaussian error linear unit (GELU), \(x\mapsto x\Phi(x)\), supplies the activation; \(\Phi(x)\) is the probability that a standard-normal random variable with mean zero and variance one is at most \(x\). The function `torch.randn` creates a random tensor with the requested shape. In the code, `residual` has shape `(2, 4, 768)`: a batch of two input examples, with four token positions and 768 coordinates per position. By default, `nn.Linear` applies an affine map, a matrix multiplication plus a learned bias, to the final tensor dimension, so the token-position dimension remains four throughout:

```python
import torch
from torch import nn

mlp = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
)

residual = torch.randn(2, 4, 768)  # batch, token positions, model width
delta = mlp(residual)
assert delta.shape == (2, 4, 768)
```

For each example in the batch, the output vector at token position \(t\) depends only on the input vector at that same position. The same learned function is applied separately at every token position and to every batch example. The resulting `delta` is normally added back to the residual stream; it is not yet a vector of vocabulary logits, the unnormalized next-token scores.

## Gated variants

A Transformer block can instead use a gated MLP. One bias-free schematic form is

\[
\operatorname{GatedMLP}(h_t)
= W_{\text{down}}
\left[
\phi(W_{\text{gate}}h_t)
\odot
(W_{\text{up}}h_t)
\right],
\]

where \(\odot\) is elementwise multiplication. The matrices \(W_{\text{gate}}\) and \(W_{\text{up}}\) project into a shared intermediate width, and \(W_{\text{down}}\) maps the gated product back to the residual width. These are gated linear unit (GLU) variants. A Swish-based gated linear unit (SwiGLU) uses a Swish-family activation on the gate branch. That family is \(x\mapsto x/(1+\exp(-\beta x))\), where \(\exp\) is the exponential function and \(\beta\) controls the slope; many implementations use the sigmoid linear unit (SiLU), the \(\beta=1\) form. A GELU-based gated linear unit (GEGLU) uses the Gaussian error linear unit (GELU) on that branch. These variants have two input projections and one output projection, not merely a different activation ([Shazeer, 2020](https://arxiv.org/abs/2002.05202v1)).

Activation, gating, intermediate width, bias use, and parameter names such as `up_proj`, `gate_proj`, and `down_proj` are model choices. Do not assume every Transformer MLP uses ReLU, expands to exactly four times the model width, or has biases.

## What the term does not imply

- **Position-wise** does not mean **context-free**. The input vector may already contain context gathered by attention.
- An MLP hidden unit is not automatically a clean, human-interpretable feature.
- “MLP activation” is underspecified in an interpretability analysis. State whether it means the input projection, the value before or after the activation, a gated product, or the output update.
- The MLP output is an internal residual-stream update. It has not yet been mapped by the final [unembedding]({{< relref "unembedding.md" >}}), the learned output projection from model-width states to vocabulary logits, or converted into next-token probabilities.

See also: [feed-forward network (FFN)]({{< relref "feed-forward-network.md" >}}), [Transformer]({{< relref "transformer.md" >}}), [residual stream]({{< relref "residual-stream.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
