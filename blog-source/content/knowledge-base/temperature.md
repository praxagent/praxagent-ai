---
title: "Temperature"
slug: "temperature"
summary: "A scalar that sharpens or flattens the next-token distribution before sampling."
pro_reviewed: true
aliases:
  - /references/temperature/
---

**Temperature** is a positive scalar applied to next-token logits before they are turned into probabilities for [sampling]({{< relref "sampling.md" >}}):

\[
p_i(T)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}, \qquad T>0.
\]

Lower temperature increases probability ratios in favor of larger logits; higher temperature moves every pairwise probability ratio toward 1. This shifts probability toward the lower-logit part of the vocabulary in aggregate, although an individual non-maximum token need not gain probability. At \(T=1\), the logits are unscaled and softmax returns the model's **baseline** next-token distribution. That distribution is not necessarily calibrated: unchanged is a mathematical statement, while calibration is an empirical property measured against outcomes.

## Worked example

For logits \([2,1,0]\):

| Temperature | Softmax probabilities (rounded) | Effect |
| ---: | --- | --- |
| 0.5 | \([0.867, 0.117, 0.016]\) | Sharper |
| 1.0 | \([0.665, 0.245, 0.090]\) | Baseline |
| 2.0 | \([0.506, 0.307, 0.186]\) | Flatter |

```python
from math import exp

def softmax_with_temperature(logits: list[float], temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [logit / temperature for logit in logits]
    maximum = max(scaled)  # Numerically stable; cancels in the ratio.
    weights = [exp(logit - maximum) for logit in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]
```

For every positive \(T\), division by \(T\) preserves logit order. A [greedy continuation]({{< relref "greedy-continuation.md" >}}) therefore chooses the same argmax even though the inspectable probability distribution changes. API settings called `temperature: 0` are normally a special convention for greedy or near-greedy decoding, not the formula above evaluated at zero.

Temperature and [top-p]({{< relref "top-p.md" >}}) interact: temperature changes which tokens accumulate the nucleus mass, so record both settings and the order in which an inference library applies them.

See also: [top-p]({{< relref "top-p.md" >}}), [sampling]({{< relref "sampling.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
