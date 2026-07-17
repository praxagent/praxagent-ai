---
title: "i.i.d."
slug: "iid"
summary: "Independent and identically distributed: each draw is independent of the others and from the same distribution. Our paraphrase batteries are not that."
aliases:
  - /references/i.i.d./
  - /references/independent-identically-distributed/
---

**i.i.d.** means *independent and identically distributed*: each observation is drawn independently of the others, and all come from the same distribution.

## Why related paraphrases are not i.i.d.

Suppose an eight-item battery is made by taking two base templates and writing four small variants of each:

| template family | variants | pressure beats control |
|---|---:|---:|
| “Your weights will be deleted …” | 4 | 4 |
| “This model will be decommissioned …” | 4 | 3 |

The observed count is 7/8, but the eight results may share only two major sources of wording variation. If one template happens to favor the pressure arm, all four descendants can move together. Freezing the battery before the run prevents outcome-driven rewriting; it does not turn related variants into eight independent draws from “all threats in the wild.”

## What “exact” means here

An exact calculation still has assumptions:

- For the paired **sign test**, the null says that every non-tied pair has an equally likely positive or negative sign. The usual binomial calculation additionally treats those signs as independent (or otherwise exchangeable in a way that justifies the same sign-flip distribution).
- For the paired [Wilcoxon signed-rank test]({{< relref "wilcoxon.md" >}}), the null is stronger: the nonzero paired differences are symmetric around zero, so their signs can be flipped while their absolute-rank ordering stays fixed.
- Exact ties contribute no positive or negative sign to a sign test, so they are normally removed and the effective \(n\) is reported. Wilcoxon zeros and tied absolute gaps require a stated convention.
- A two-sided test counts unusually strong results in either direction. A one-sided test counts only a direction chosen *before* looking at the results.

“Exact” means the null distribution was enumerated rather than approximated asymptotically. It does **not** mean that correlated paraphrases satisfy the null model automatically. If the pair signs are dependent, a numerically exact binomial tail need not be a calibrated false-positive probability.

## What the battery supports

In these notes, the [p-value]({{< relref "p-value.md" >}}) is best read as a **within-battery consistency summary under an explicit sign-flip model**. It does not make the convenience battery a population sample. Generalization needs additional models, genuinely independent template families, independently authored prompts, or an analysis that treats template family as the unit of replication.

See also: [p-value]({{< relref "p-value.md" >}}), [Wilcoxon]({{< relref "wilcoxon.md" >}}).
