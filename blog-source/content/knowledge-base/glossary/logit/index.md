---
title: "Logit"
slug: "logit"
summary: "An unnormalized next-token score. Softmax turns a vector of logits into probabilities."
og_image: "logit-card.png"
og_image_alt: "Toy next-token logits of 3.2, 2.7, and -0.4 become softmax probabilities 0.612, 0.371, and 0.017. Adding 5 to every logit leaves the probabilities unchanged."
draft: false
pro_reviewed: true
---

A **logit**, in these notes, is one unnormalized score for a vocabulary token, before that score is turned into a probability. The [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}), the final layer that maps the model's hidden state to one score per vocabulary token, produces a vector \(z\), with one logit \(z_v\) for each token \(v\) in the vocabulary \(V\). Softmax converts the vector into the next-token distribution, where \(p_v\) is the probability assigned to token \(v\) and \(\exp\) is the exponential function:

\[
p_v=\frac{\exp(z_v)}{\sum_{u\in V}\exp(z_u)}.
\]

A logit is not a probability. It can be negative, and it has no fixed range. Adding the same number to every logit leaves every probability unchanged, because that number cancels between the numerator and the denominator. The differences are log probability ratios:

\[
z_i-z_j=\log\frac{p_i}{p_j}.
\]

Here \(\log\) is the natural logarithm, the inverse of \(\exp\).

A [greedy continuation]({{< relref "knowledge-base/glossary/greedy-continuation/index.md" >}}) emits a token whose logit is the largest; if several tokens tie, the implementation's tie-breaking rule decides among them. Dividing every logit by a positive [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}) keeps the order of the logits, ties included. It changes the probabilities unless the temperature is 1 or all the logits are equal.

This is not the binary log-odds function \(\operatorname{logit}(p)=\log\frac{p}{1-p}\), which sends one probability to one number. A language-model logit vector is the input to softmax over the whole vocabulary. When the choice is between exactly two tokens, the difference of their logits equals that log-odds.

{{< reference-figure
  src="logit-to-probability.svg"
  alt="Toy next-token logits of 3.2, 2.7, and -0.4 become softmax probabilities 0.612, 0.371, and 0.017. The negative logit still receives positive probability, and red, the token with the largest logit, is selected greedily."
  caption="Softmax converts the worked-example logits for red, blue, and green into probabilities that are positive and sum to 1. Bar lengths match those values. Adding 5 to every logit leaves the probabilities unchanged, so only differences between logits matter. The illustration uses three toy tokens, not a measured model distribution."
>}}

## Worked example

The same scores as on the greedy-continuation page:

| Token | Logit \(z\) | Probability, rounded to 3 decimals |
| --- | ---: | ---: |
| `red` | 3.2 | 0.612 |
| `blue` | 2.7 | 0.371 |
| `green` | -0.4 | 0.017 |

The unrounded weights are \(\exp(z-\max z)\), then divided by their sum. `green` is negative and still has positive probability. `red` is the greedy token because \(3.2\) is the largest logit. The value \(0.612\) is the probability the model assigns to `red`; on its own it does not show that the model is calibrated, meaning that, across many predictions, tokens given probability near 0.612 turn out to be the actual next token about 61.2% of the time. Shifting every logit by \(+5\), to \(8.2\), \(7.7\), and \(4.6\), returns the same three probabilities. The log ratio of `red` to `blue` is \(3.2-2.7=0.5\).

```python
from math import exp

def softmax(logits: list[float]) -> list[float]:
    """Turn unnormalized scores into probabilities that sum to one."""
    if len(logits) == 0:
        raise ValueError("logits must be nonempty")
    maximum = max(logits)  # Cancels in the ratio; keeps exp in range.
    weights = [exp(logit - maximum) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]
```

A [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}) is a separate procedure: it applies a specified readout, a mapping from a hidden state to one score per token, to the hidden state of an intermediate layer rather than the last one. Those scores are logits only under the readout the analyst specified. They are not automatically the model's next-token logits.

See also: [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}), [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}), [sampling]({{< relref "knowledge-base/glossary/sampling/index.md" >}}), [greedy continuation]({{< relref "knowledge-base/glossary/greedy-continuation/index.md" >}}), [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}).
