---
title: "Sampling"
slug: "sampling"
summary: "Choosing the next token from a distribution over the vocabulary, rather than always taking the single highest-scoring token."
aliases:
  - /references/token-sampling/
---

**Sampling** means drawing the next token from a probability distribution over the vocabulary, instead of always emitting the argmax. The raw next-token scores from the [unembedding]({{< relref "unembedding.md" >}}) are turned into probabilities (usually via softmax), then a token is chosen according to a rule such as multinomial draw, often after reshaping the distribution with [temperature]({{< relref "temperature.md" >}}) or truncating it with [top-p]({{< relref "top-p.md" >}}).

A [greedy continuation]({{< relref "greedy-continuation.md" >}}) is the opposite: no sampling, just the highest-scoring token at every step. That path is deterministic given the prompt and model; sampled paths are not.

See also: [temperature]({{< relref "temperature.md" >}}), [top-p]({{< relref "top-p.md" >}}), [greedy continuation]({{< relref "greedy-continuation.md" >}}).
