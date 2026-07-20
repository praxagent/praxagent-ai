---
title: "Feed-forward network (FFN)"
slug: "feed-forward-network"
summary: "A computation with no feedback cycle; in a Transformer, usually the shared position-wise sublayer that transforms each token state separately."
pro_reviewed: false
---

A **feed-forward network (FFN)** is a neural-network computation in which values move from inputs through a finite sequence, or another arrangement with no feedback cycle, to outputs. During one forward computation, a later value is not fed back into an earlier operation. A residual or skip connection can still be feed-forward when it only carries an earlier value to a later operation.

In [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) writing, **FFN** usually means something narrower: the **position-wise feed-forward sublayer** inside each block. The same learned function transforms every token-position state separately. A [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}) is a common implementation of that function.

{{< reference-figure
  src="feed-forward-network-positionwise.svg"
  alt="Three token-state vectors pass independently through feed-forward network copies with the same parameters, producing three corresponding updates without direct cross-position connections."
  caption="Within one Transformer layer, the same feed-forward network parameters map each token-position state to an update at that position. No position directly reads another position inside this sublayer, although each input state may already contain context written by attention. The illustration omits normalization and residual addition and does not imply that parameters are shared between different layers."
>}}

## Position-wise means one row at a time

Let \(\mathbb{R}\) denote the real numbers. Suppose \(H\in\mathbb{R}^{n\times d_{\text{model}}}\) contains one row for each of \(n\) token positions, with \(d_{\text{model}}\) coordinates per row. A position-wise FFN computes

\[
Y_{t,:}=f_\theta(H_{t,:}),
\qquad t\in\{1,\ldots,n\}.
\]

Here \(t\) selects a token position, the colon means all coordinates in that row, \(f_\theta\) is the learned feed-forward function, and \(\theta\) denotes its learned numeric parameters. The same \(\theta\) is reused for every position within this layer. Different layers normally have different parameters.

The output row \(Y_{t,:}\) does not directly read a different input row \(H_{s,:}\) when \(s\ne t\). That does **not** make it context-free: \(H_{t,:}\) may already contain information gathered from other positions by attention. The original Transformer described this sublayer as two learned affine transformations with a rectified linear unit between them, applied to each position separately and identically ([Vaswani et al., 2017, Section 3.3](https://arxiv.org/abs/1706.03762v7)). Modern blocks may use gated variants instead ([Shazeer, 2020](https://arxiv.org/abs/2002.05202v1)).

## FFN, position-wise FFN, and MLP

| Term | What it identifies |
| --- | --- |
| Feed-forward network | A forward computation with no feedback cycle |
| Position-wise FFN | The Transformer sublayer's role: apply one learned function separately at each token position |
| Multilayer perceptron | A common layered implementation using learned affine maps and nonlinear activation or gating |

These labels often refer to the same module in Transformer code, but they emphasize different facts. **Feed-forward** describes connectivity, **position-wise** describes how token positions are handled, and **MLP** describes a common internal construction.

## What feed-forward does not mean

- It does not mean that training only moves forward. Training can still work backward through the operations to determine how learned parameters should change.
- **Fully connected** means a learned map can combine coordinates within one token vector. It does not mean the FFN directly mixes token positions.
- Shared parameters across positions do not imply shared parameters across layers.
- The FFN's output is an internal update to the [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), not a final token probability.

See also: [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}), [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}).
