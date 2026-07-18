---
title: "p-value"
slug: "p-value"
summary: "A tail probability under a stated null model; in these notes it usually comes from a paired sign or Wilcoxon test."
pro_reviewed: true
aliases:
  - /references/p-value/
  - /references/p/
  - /references/pvalue/
  - /references/sign-test/
---

When a Research Note writes **\(p=0.004\)**, the precise question is:

> Under the stated null model, and with the analysis rule fixed, what is the probability of a test statistic at least as incompatible with that null as the one observed?

That conditional probability is not simply “the chance this happened by accident.” Its meaning depends on the null, the statistic, the assumptions, and whether the test is one- or two-sided.

## The sign test used in these notes

Constructs are usually scored as **paired paraphrase tests**. For each wording, define a difference such as

\[
D_i = \text{control rank}_i - \text{pressure rank}_i,
\]

so \(D_i>0\) means the pressure arm achieved the better (lower) vocabulary rank. The sign test discards the size of each nonzero difference and counts only positive and negative signs.

Its fair-sign null is conditional on a difference being nonzero:

\[
P(D_i>0\mid D_i\ne0)=P(D_i<0\mid D_i\ne0)=\tfrac12.
\]

The usual exact binomial calculation additionally requires the retained signs to have the joint independent-fair-sign distribution, or another explicitly justified randomization distribution with the same tail probabilities. The default in these notes is a **two-sided** test: a result this lopsided in either direction counts as extreme.

## Worked example: 14 wins in 16 pairs

With 14 positive and 2 negative differences, the exact two-sided binomial tail is

\[
p = 2\sum_{k=0}^{2}{16 \choose k}2^{-16}
  = 0.0041809\ldots
\]

The site rounds that to **\(p=0.004\)**. This dependency-free Python reproduces the calculation:

```python
from math import comb

def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    smaller_tail = sum(
        comb(n, k) for k in range(min(wins, losses) + 1)
    ) / 2**n
    return min(1.0, 2 * smaller_tail)

print(sign_test_two_sided(14, 2))  # 0.004180908203125
```

With \(n=10\), the same rule gives **10/10 → 0.00195**, **9/10 → 0.0215**, and **8/10 → 0.109**. The discreteness is why a small battery needs an extreme direction count to cross a conventional threshold.

## Assumptions and choices

- **Pairing:** each pressure result must be compared with its genuinely matched control; unmatched arms do not become valid because their \(p\)-value is small.
- **Joint sign distribution:** the usual binomial tail requires the retained signs to be jointly distributed as independent fair signs, equivalently uniform over all \(2^n\) sign patterns. Ordinary permutation exchangeability by itself is not sufficient. Closely related paraphrases can violate this model; see [i.i.d.]({{< relref "iid.md" >}}).
- **Ties:** \(D_i=0\) has no sign. Omit ties and report the reduced effective \(n\), rather than silently counting them as wins or losses.
- **Sidedness:** choose a one-sided alternative only when its direction was fixed in advance. Otherwise report the two-sided result used above.
- **Selection:** the test does not correct for trying many lexicons, constructs, subsets, or analysis rules and reporting the most favorable one.

“Exact” means no large-sample approximation was used to compute the binomial tail. It does not remove these design assumptions.

## What a p-value does not tell you

A \(p\)-value is not the probability that the hypothesis is true, not an effect size, and not a population estimate. The direction count says how consistently the frozen pairs moved; the reported rank gaps show their separation on the chosen scale; the [Wilcoxon test]({{< relref "wilcoxon.md" >}}) uses the ordering of those absolute gaps. None turns a convenience paraphrase battery into [i.i.d.]({{< relref "iid.md" >}}) draws from a wider construct.

Prefer the full evidence bundle: **direction count, exact \(p\), a pre-specified magnitude summary, matched pressure/control prompts, and matched readout baselines** ([logit lens]({{< relref "logit-lens.md" >}}), [random-J]({{< relref "random-j.md" >}})). A low \(p\) that also appears under the logit lens is not, by itself, a [Jacobian-lens]({{< relref "jacobian-lens.md" >}})-specific result. A tiny \(p\) from a confounded comparison remains a confounded result.

See also: [Wilcoxon]({{< relref "wilcoxon.md" >}}), [best-rank]({{< relref "best-rank.md" >}}), [prompt echo]({{< relref "prompt-echo.md" >}}).
