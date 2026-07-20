---
title: "Unembedding"
slug: "unembedding"
summary: "The final linear map from residual-stream coordinates to a score for every vocabulary token."
pro_reviewed: true
---

**Unembedding** (the output head) maps the model's final hidden state to one score, or **logit**, for every vocabulary token. In a common decoder-only Transformer:

\[
\tilde h = \operatorname{FinalNorm}(h_L), \qquad
z = W_U\tilde h + b_U.
\]

Here \(h_L\in\mathbb{R}^d\) is the last [residual-stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}) vector, \(W_U\in\mathbb{R}^{|V|\times d}\) contains one readout direction per vocabulary item, and \(z\in\mathbb{R}^{|V|}\) is the logit vector. Softmax converts those logits into next-token probabilities.

The “final linear map” shorthand hides two architecture details:

- The model's **final normalization** (often RMSNorm or LayerNorm) is normally applied before the output head. Skipping it can change logit magnitudes and, depending on the normalization, token rankings.
- The output bias \(b_U\) is **optional**. Many modern language models omit it. The unembedding weight may also be tied to the input embedding matrix, but weight tying is an architecture choice, not part of the definition.

## Worked example

Suppose the already normalized hidden state is \(\tilde h=[0.6,0.8]\). A toy output head has the following token rows and biases:

| Token | Weight row | Bias | Logit |
| --- | --- | ---: | ---: |
| `red` | \([1,0]\) | 0.0 | 0.6 |
| `blue` | \([0,1]\) | 0.0 | 0.8 |
| `green` | \([0.5,0.5]\) | 0.2 | 0.9 |

The `green` logit is \(0.5(0.6)+0.5(0.8)+0.2=0.9\), so it is the greedy choice even though neither coordinate alone favors it.

```python
# Shape: hidden [d], weight [vocabulary, d], logits [vocabulary]
normalized = model.final_norm(hidden)
logits = model.lm_head.weight @ normalized
if model.lm_head.bias is not None:
    logits = logits + model.lm_head.bias
```

Mid-layer readouts such as the [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}) reuse the output head on an earlier residual state, commonly with the model's final norm. A [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}) instead uses a transport derived from downstream local Jacobians, typically averaged over a fitting corpus. In either case, document which normalization, bias, and weight matrix were used: “apply the unembedding” is otherwise underspecified.

See also: [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), [rank]({{< relref "knowledge-base/glossary/rank/index.md" >}}), [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}).
