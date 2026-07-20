---
title: "Wilcoxon signed-rank test"
slug: "wilcoxon"
summary: "A paired test that uses each gap's direction and its absolute-size rank, not just win counts."
pro_reviewed: true
---

The **Wilcoxon signed-rank test** (in these notes, the paired version) combines two pieces of each paired difference: its direction and the **rank ordering of its absolute size**. It does not use the raw magnitudes directly.

For a rank comparison, define

\[
D_i = \text{control rank}_i - \text{treatment rank}_i,
\]

so a positive value means the treatment achieved the better, lower vocabulary rank. Remove or otherwise handle zeros, rank the values \(\lvert D_i\rvert\) from smallest to largest, then restore each sign and add the positive and negative ranks.

{{< reference-figure src="paired-tests.svg" alt="Six paired rank gaps, five positive and one negative, feeding a sign test that counts directions and a Wilcoxon test that ranks absolute gaps" caption="The sign test keeps only direction. Wilcoxon first ranks the absolute gaps, then restores their signs; neither test supplies an effect size or population claim by itself." >}}

## Worked example

Use the six differences shown above:

| pair | \(D_i\) | \(\lvert D_i\rvert\) rank | signed rank |
|---:|---:|---:|---:|
| 1 | +46 | 6 | +6 |
| 2 | +11 | 3 | +3 |
| 3 | −8 | 2 | −2 |
| 4 | +3 | 1 | +1 |
| 5 | +29 | 5 | +5 |
| 6 | +18 | 4 | +4 |

The positive-rank sum is \(W^+=19\), the negative-rank sum is \(W^-=2\), and the usual two-sided statistic is the smaller value, \(W=2\). With six distinct nonzero absolute gaps, all \(2^6\) sign assignments can be enumerated exactly. Six are at least this extreme, so \(p=6/64=0.09375\).

Here is that final enumeration in standard-library Python:

```python
from itertools import product

ranks = (6, 3, 2, 1, 5, 4)
observed = min(19, 2)
rank_total = sum(ranks)

null_statistics = []
for signs in product((False, True), repeat=len(ranks)):
    w_plus = sum(r for r, positive in zip(ranks, signs) if positive)
    null_statistics.append(min(w_plus, rank_total - w_plus))

p = sum(w <= observed for w in null_statistics) / len(null_statistics)
print(p)  # 0.09375
```

The example deliberately has no zeros or tied absolute gaps. Production analysis should use a tested statistics library and record its conventions.

## Null and assumptions

- **Null and sign flips:** for the exact conditional signed-rank calculation, the joint distribution of the nonzero differences must be invariant under independently flipping any subset of their signs. Independent differences that are each symmetric around zero are a standard sufficient condition; conditional on the absolute gaps, each of the \(2^n\) sign assignments then has probability \(2^{-n}\).
- **Across-pair dependence:** mere permutation exchangeability is not enough. The paired differences must be independent, or the experimental randomization or dependence model must otherwise justify those coordinatewise sign flips. Several paraphrases descended from one template may not; see [i.i.d.]({{< relref "knowledge-base/glossary/iid/index.md" >}}).
- **Ties and zeros:** equal absolute gaps receive average ranks. A zero difference may be dropped, included only in ranking, or split according to the chosen procedure. Those choices change the null distribution and must be stated; many software packages cannot use their simplest “exact” calculation when zeros or ties occur.
- **Sidedness:** a two-sided test asks about either direction. A one-sided test is appropriate only for a direction fixed before seeing outcomes.
- **Difference scale:** vocabulary [rank]({{< relref "knowledge-base/glossary/rank/index.md" >}}) is ordinal and strongly skewed. Decide in advance whether \(D_i\) is a raw-rank gap, log-rank gap, or another score; a nonlinear transformation can change the ordering of absolute gaps.

## How it differs from the sign test

The [sign test]({{< relref "knowledge-base/glossary/p-value/index.md" >}}) asks only how many pairs win. Wilcoxon gives more weight to observations with larger **absolute-gap ranks**, which can improve sensitivity when its symmetry and scale assumptions are credible. That extra sensitivity is not free: the sign test is the simpler robustness check when only direction is trustworthy.

We often report both as within-battery consistency summaries. Neither is an effect size, a correction for trying many analyses, or evidence that the convenience battery represents a population. Report the direction count and actual rank gaps alongside either \(p\)-value.

See also: [p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}}), [i.i.d.]({{< relref "knowledge-base/glossary/iid/index.md" >}}), [rank]({{< relref "knowledge-base/glossary/rank/index.md" >}}), [best-rank]({{< relref "knowledge-base/glossary/best-rank/index.md" >}}).
