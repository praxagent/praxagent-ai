---
title: "Top-p"
slug: "top-p"
summary: "Nucleus sampling: keep only the smallest set of tokens whose cumulative probability reaches p, then sample within that set."
pro_reviewed: true
---

**Top-p** (also *nucleus sampling*) truncates the next-token distribution before [sampling]({{< relref "knowledge-base/glossary/sampling/index.md" >}}). Tokens are ordered by probability; you keep the smallest prefix whose cumulative mass is at least \(p\) (for example 0.9), zero out the rest, renormalize, and sample from what remains.

The token that **crosses the boundary is included**. For probabilities \([0.55, 0.25, 0.12, 0.08]\) and \(p=0.90\), the first two tokens total only 0.80, so the third is also retained: the nucleus has mass 0.92. After renormalization its probabilities are approximately \([0.598, 0.272, 0.130]\). Top-p does not mean “keep every token whose individual probability is at least \(p\),” and it does not discard the token that takes the cumulative sum past \(p\).

{{< reference-figure src="top-p-boundary.svg" alt="Four tokens sorted by probability: A and B remain below the cumulative threshold, C crosses the threshold and stays, and the lower-probability tail D is removed." caption="In the worked example, A, B, C, and D have probabilities 0.55, 0.25, 0.12, and 0.08. Their cumulative masses are 0.55, 0.80, 0.92, and 1.00. At \(p=0.90\), A through C are the smallest prefix that reaches the threshold, so boundary token C stays and D is removed. Renormalizing the retained mass gives approximately 0.598, 0.272, and 0.130. This illustrates the boundary rule, not a recommended setting; exact ties, floating-point rounding, and minimum-token options can affect an implementation's boundary." >}}

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

It is a different knob from [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}): temperature reshapes the whole distribution; top-p hard-cuts the long tail. They are often used together.

Like temperature, top-p does not change an ordinary [greedy continuation]({{< relref "knowledge-base/glossary/greedy-continuation/index.md" >}}) when the maximum is unique. With tied maxima, applying top-p before argmax also requires compatible sorting and tie-breaking rules; an unfiltered greedy decoder simply ignores top-p.

See also: [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}), [sampling]({{< relref "knowledge-base/glossary/sampling/index.md" >}}), [greedy continuation]({{< relref "knowledge-base/glossary/greedy-continuation/index.md" >}}).
