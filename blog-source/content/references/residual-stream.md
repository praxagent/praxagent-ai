---
title: "Residual stream"
slug: "residual-stream"
summary: "The model's running hidden-state channel: each layer reads it, adds an update, and passes it on."
pro_reviewed: true
---

The **residual stream** is the transformer's running hidden state. More precisely, there is **one residual vector per token position**, not one vector for the whole prompt. At layer \(\ell\), a sequence of \(n\) tokens has a residual-state matrix

\[
H_\ell =
\begin{bmatrix}
h_{\ell,1}^{\top} \\
\vdots \\
h_{\ell,n}^{\top}
\end{bmatrix}
\in \mathbb{R}^{n\times d_{\text{model}}}.
\]

Each transformer block reads those vectors and adds updates back into them. In a common pre-normalization block, the schematic update is

\[
U_\ell = H_\ell + \operatorname{Attention}_\ell(N(H_\ell)),
\qquad
H_{\ell+1} = U_\ell + \operatorname{MLP}_\ell(N(U_\ell)).
\]

Attention means the update at one position can depend on other permitted positions in the sequence. The residual stream is therefore better pictured as a sequence of evolving row vectors than as a single global scratchpad.

{{< reference-figure src="references/glossary/residual-stream.svg" alt="Four token positions each retain their own residual vector across a transformer block; attention can mix permitted positions, the MLP updates each position, and a readout selects one layer-position vector." caption="Figure structure: the toy tokens `The`, `capital`, `is`, and `Paris` each have a separate vector before the block. Attention can use permitted earlier or current positions to update a token, and the MLP then updates each position. Four separate vectors continue after the block; the highlighted last-position vector is one selected state, not a summary of the full prompt. This is a schematic common pre-normalization block, not a universal architecture or tokenizer specification." >}}

## Worked example

Suppose a tokenizer turns “The capital is Paris” into four tokens. A saved activation tensor might have shape `(layers, 4, d_model)`. Reading the state after layer 20 at the last token means selecting one vector:

```python
layer = 20
position = -1
h = residual[layer, position]  # shape: (d_model,)
```

Changing `position` changes the state being inspected. A result at the final prompt position does not automatically describe every earlier token position.

## What a readout adds

Mid-network \(h_{\ell,t}\) is a high-dimensional vector, not a sentence in English. Readouts such as the [logit lens]({{< relref "logit-lens.md" >}}) and [Jacobian lens]({{< relref "jacobian-lens.md" >}}) normalize and map a selected vector into vocabulary scores. Those scores are a measurement imposed by the analyst. Seeing a word score highly does not by itself show that the model represents a clean, discrete English concept there or that the scored direction causally controls behavior. Causal claims require an intervention and an observed downstream effect.

See also: [workspace]({{< relref "workspace.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
