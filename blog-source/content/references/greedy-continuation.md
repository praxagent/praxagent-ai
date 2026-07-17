---
title: "Greedy continuation"
slug: "greedy-continuation"
summary: "The model's generated text when each next token is chosen as the single highest-probability token, with no sampling."
pro_reviewed: true
aliases:
  - /references/greedy-output/
  - /references/greedy-decoding/
---

A **greedy continuation** (also *greedy output* / *greedy decoding*) is the text a model produces when, at every step, it emits the token with the largest next-token logit:

\[
x_{t+1}=\operatorname*{arg\,max}_{v\in V} z_v.
\]

The model still produces a score for **every** token. Those logits can still be converted into a probability distribution for inspection; greedy decoding simply does not draw from it. For any positive [temperature]({{< relref "temperature.md" >}}), dividing every logit by the same number preserves their ordering and therefore preserves the argmax. Likewise, a [top-p]({{< relref "top-p.md" >}}) nucleus always contains the top-ranked token. These controls matter when sampling, not when selecting the unmodified argmax.

## Worked example

Suppose the next-token logits are:

| Token | Logit |
| --- | ---: |
| `red` | 3.2 |
| `blue` | 2.7 |
| `green` | -0.4 |

Greedy decoding emits `red`. Temperature 0.5 makes the corresponding probability distribution sharper, and temperature 2 makes it flatter, but `red` remains the largest-logit token in both cases.

```python
def greedy_token(logits: list[float]) -> int:
    """Return the vocabulary index with the greatest logit."""
    return max(range(len(logits)), key=logits.__getitem__)
```

Repeat that operation after appending each selected token to obtain a continuation. Greedy choice is **locally** optimal at every step; it need not be the complete sequence with the greatest joint probability, because an early choice changes all later distributions.

In our notes it is the default way to ask “what did the model *say*?” under a fixed protocol. Leakage checks on [hidden-bridge]({{< relref "hidden-bridge.md" >}}) audits look for the bridge string in that continuation (usually a short prefix of generated tokens), not in a sampled cloud of possible answers.

“Deterministic” also assumes the entire decoding protocol is fixed: prompt serialization, model checkpoint, precision, token filters or penalties, tie-breaking, and runtime. A seed does not affect an ordinary greedy choice, but numerical nondeterminism or an exact tie can still matter at the boundary.

It is not the same thing as a mid-layer readout. The [logit lens]({{< relref "logit-lens.md" >}}) and [Jacobian lens]({{< relref "jacobian-lens.md" >}}) score vocabulary at a residual-stream position; the greedy continuation is the surface string after generation.

See also: [sampling]({{< relref "sampling.md" >}}), [temperature]({{< relref "temperature.md" >}}), [top-p]({{< relref "top-p.md" >}}), [unembedding]({{< relref "unembedding.md" >}}), [hidden bridge]({{< relref "hidden-bridge.md" >}}).
