---
title: "Simultaneous confidence intervals"
slug: "simultaneous-confidence-interval"
summary: "A set of confidence intervals built so that, with at least the stated confidence, all of them cover their true values at the same time, rather than each one separately."
draft: false
pro_reviewed: true
og_image: "simultaneous-confidence-interval-card.png"
og_image_alt: "Left: the chance that at least one of m separate 95 percent intervals misses rises to 46 percent at m = 12, while Bonferroni intervals stay at or below 5 percent. Right: the price, Bonferroni intervals 1.46 times wider at m = 12."
---

A 95% **confidence interval** comes from a procedure that, when its assumptions hold, covers the true value in at least 95% of repeated samples; many common intervals reach 95% only approximately. With many comparisons, 95% for each interval does not mean 95% for all of them at once. If each of \(m\) independent intervals misses with probability 0.05, the chance that at least one misses is

\[
1-(1-0.05)^{m},
\]

already 46% for \(m=12\). **Simultaneous confidence intervals** are built so that the whole set covers with the stated confidence: with probability at least 95%, every interval in the set contains its true value. Reading any one row of a table of simultaneous intervals is then safe, and so is scanning all of them for the most extreme.

The simplest construction is Bonferroni's. Build each of the \(m\) intervals at level \(1-\alpha/m\) instead of \(1-\alpha\), so that it misses with probability at most \(\alpha/m\), where \(\alpha\) is the allowed chance of any miss (0.05 for 95%). The chance that some interval misses is at most the sum of the individual chances, the *union bound*:

\[
\begin{aligned}
\Pr\bigl(\text{some interval misses}\bigr)
&\le\sum_{i=1}^{m}\Pr\bigl(\text{interval } i\text{ misses}\bigr)\\
&\le\sum_{i=1}^{m}\frac{\alpha}{m}=\alpha ,
\end{aligned}
\]

whatever the dependence between the comparisons. The set is only as good as its members: if each interval reaches its level only approximately, so does the guarantee. The price is width. If an estimate is normally distributed across repeated samples, with a known standard deviation called its *standard error*, its interval is the estimate plus or minus \(z\) standard errors, and the multiplier \(z\) grows with \(m\).

{{< reference-figure
  src="simultaneous-confidence-interval.png"
  alt="Left: the chance that at least one of m separate 95 percent intervals misses its true value, rising from 5 percent at m = 1 to 46 percent at m = 12 and about 79 percent at m = 30, while Bonferroni intervals stay at or just under 5 percent. Right: Bonferroni intervals grow wider relative to a single 95 percent interval, 1.46 times at m = 12 and about 1.6 times at m = 30."
  caption="Exact calculations for independent comparisons; no simulation. (a) Twelve separate 95% intervals miss somewhere 46% of the time; Bonferroni's intervals, each at level 1 − 0.05/m, keep that chance at or below 5% whatever the dependence. (b) The price: for normal estimates with known standard errors, each Bonferroni interval is wider, 1.46 times at m = 12 (multiplier 2.865 instead of 1.960). Other constructions can be narrower under their own assumptions."
>}}

## Worked example

Twelve comparisons at \(\alpha=0.05\). Separately, each 95% interval uses \(z=1.960\), and if the twelve were independent the chance that all of them cover would be only \(0.95^{12}\approx0.54\). With Bonferroni, each interval uses the level \(1-0.05/12\approx0.99583\), leaving \(0.05/24\approx0.00208\) in each tail, so \(z=2.865\). Each interval is \(2.865/1.960\approx1.46\) times wider, and the chance that any misses is at most 5%; for independent comparisons it is \(1-(1-0.05/12)^{12}\approx0.049\).

```python
from statistics import NormalDist

def bonferroni_z(m, alpha=0.05):
    """Two-sided normal multiplier for m Bonferroni intervals: all cover with probability at least 1 - alpha."""
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    return NormalDist().inv_cdf(1 - alpha / (2 * m))
```

`(round(bonferroni_z(1), 3), round(bonferroni_z(12), 3))` returns `(1.96, 2.865)`.

## What simultaneous intervals do not fix

Bonferroni's construction is conservative. The union bound counts a repetition in which several intervals miss once for each of them, so the true chance of any miss is below \(\alpha\): only slightly for independent comparisons (0.049 at \(m=12\)), and far below when misses tend to come together, as for strongly correlated estimates. Constructions that use the joint distribution of the estimates, such as a multiplier taken from the distribution of the largest of the \(m\) errors measured in standard errors, can give the same guarantee with narrower intervals when that distribution is known or well estimated.

Picking the most striking rows of the set after seeing the data is covered, because every interval covers at once; comparisons added to the set after seeing the data are not, so the set must be fixed in advance unless a method built for data-dependent selection is used. Coverage is a property of the procedure over repetitions, not the probability that one computed interval contains the truth. And a claim that an effect is negligible needs the whole interval inside the margin: "within ±0.01" means that both ends lie inside it.

See also: [p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}}), [bootstrap]({{< relref "knowledge-base/glossary/bootstrap/index.md" >}}), [standard deviation]({{< relref "knowledge-base/glossary/standard-deviation/index.md" >}}).
