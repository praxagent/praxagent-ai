---
title: "Top-p"
slug: "top-p"
summary: "Nucleus sampling: keep only the smallest set of tokens whose cumulative probability reaches p, then sample within that set."
pro_reviewed: true
aliases:
  - /references/nucleus-sampling/
  - /references/top_p/
---

**Top-p** (also *nucleus sampling*) truncates the next-token distribution before [sampling]({{< relref "sampling.md" >}}). Tokens are ordered by probability; you keep the smallest prefix whose cumulative mass is at least \(p\) (for example 0.9), zero out the rest, renormalize, and sample from what remains.

The token that **crosses the boundary is included**. For probabilities \([0.55, 0.25, 0.12, 0.08]\) and \(p=0.90\), the first two tokens total only 0.80, so the third is also retained: the nucleus has mass 0.92. After renormalization its probabilities are approximately \([0.598, 0.272, 0.130]\). Top-p does not mean “keep every token whose individual probability is at least \(p\),” and it does not discard the token that takes the cumulative sum past \(p\).

```python
def nucleus(tokens, probabilities, p):
    if not 0 < p <= 1:
        raise ValueError("top-p must be in (0, 1]")

    ranked = sorted(zip(tokens, probabilities),
                    key=lambda pair: pair[1], reverse=True)
    kept, mass = [], 0.0
    for token, probability in ranked:
        kept.append((token, probability))  # Include the boundary token.
        mass += probability
        if mass >= p:
            break
    return [(token, probability / mass) for token, probability in kept]
```

At very small positive \(p\), the nucleus contains at least the highest-probability token; at \(p=1\), it normally contains the full nonzero distribution. Exact ties, floating-point rounding, and library options such as a minimum number of retained tokens can affect the boundary, so record the implementation when it matters.

It is a different knob from [temperature]({{< relref "temperature.md" >}}): temperature reshapes the whole distribution; top-p hard-cuts the long tail. They are often used together.

Like temperature, top-p does not change an ordinary [greedy continuation]({{< relref "greedy-continuation.md" >}}): the retained prefix contains the top-ranked token, and greedy decoding does not draw from the renormalized nucleus.

See also: [temperature]({{< relref "temperature.md" >}}), [sampling]({{< relref "sampling.md" >}}), [greedy continuation]({{< relref "greedy-continuation.md" >}}).
