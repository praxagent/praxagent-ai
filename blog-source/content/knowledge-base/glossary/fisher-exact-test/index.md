---
title: "Fisher's exact test"
slug: "fisher-exact-test"
summary: "An exact conditional test for association in a count table, commonly used for a 2 by 2 comparison of two groups and two outcomes."
draft: false
pro_reviewed: true
---

**Fisher's exact test** calculates a
[p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}}) for
association in a table of counts. In the common \(2\times2\) case, the rows are
two groups and the columns are two outcomes. The calculation conditions on the
observed row and column totals, enumerates tables compatible with those totals,
and sums a stated null tail.

Informal phrases such as **Fisher \(p\)** mean the \(p\)-value from Fisher's
exact test. They do not mean **Fisher's method**, which combines several
\(p\)-values into one test statistic.

## The conditional null distribution

Write the observed table as:

|  | outcome 1 | outcome 2 | row total |
|---|---:|---:|---:|
| group 1 | \(a\) | \(b\) | \(r_1=a+b\) |
| group 2 | \(c\) | \(d\) | \(r_2=c+d\) |
| column total | \(c_1=a+c\) | \(c_2=b+d\) | \(N=a+b+c+d\) |

Here \(a,b,c,d\) are nonnegative integer counts. The **margins** are the
outside totals written beside and below the table: the row totals \(r_1\) and
\(r_2\) (how many units sit in each group), the column totals \(c_1\) and
\(c_2\) (how many units have each outcome), and the grand total \(N\). They
are called margins because they sit on the margin of the table, outside the
four interior cells.

**Held fixed** means: in the probability calculation, those outside totals are
treated as known constants equal to the values you observed. Only the interior
can move. So every table the null compares must still have the same group sizes
and the same overall outcome counts; what varies is how the outcomes are split
between the two groups. That restriction is called **conditioning on the
margins**.

The observed count in the **group 1, outcome 1** cell is \(a\). Under the null
model below, that same cell is treated as a random variable \(A\): how many
outcome-1 units fall in group 1 when those margins stay at their observed
values. The data are one realized value \(A=a\).

If \(p_1\) and \(p_2\) are the probabilities of outcome 1 in groups 1 and 2,
respectively, the **population odds ratio** is

\[
\theta = \frac{p_1/(1-p_1)}{p_2/(1-p_2)}.
\]

Nondegenerate means \(0\lt p_1\lt 1\) and \(0\lt p_2\lt 1\), so both odds are
defined. The odds for a group are \(p/(1-p)\): how many times more likely
outcome 1 is than outcome 2 in that group. The ratio \(\theta\) compares those
two odds.

Why the null sets \(\theta=1\): that is the “no association” claim. If
\(\theta=1\), then \(p_1=p_2\): knowing which group a unit belongs to does not
change its chance of outcome 1. The groups have the same hit rate in the
population. Values \(\theta\gt 1\) or \(\theta\lt 1\) would mean one group is
more (or less) prone to outcome 1 than the other. Fisher’s test asks whether
the observed table is extreme under that no-difference null; it does not claim
the sample odds ratio must equal 1.

Under the conditional null \(\theta=1\), \(A\) given the fixed margins follows
a central hypergeometric distribution (defined next).

{{< panel "info" >}}
**Why the null is hypergeometric.** Conditioning on the observed margins freezes
the outside totals \(N\), \(r_1\), \(r_2\), \(c_1\), and \(c_2\) at the values
in your table. Under the null that group membership does not change the chance
of outcome 1, those \(c_1\) outcome-1 labels are exchangeable across the \(N\)
units. Fixing group 1's size at \(r_1\) is then exactly like dealing \(r_1\)
seats out of \(N\) without putting anyone back: a finite population, drawn
**without replacement**.

That is the definition of a hypergeometric count, not a binomial one. A
binomial model would need independent trials with a fixed success probability
and would not automatically keep the column total \(c_1\) exact. Here the
column total is locked because we conditioned on it, so each draw changes what
remains in the pool.

**Urn picture.** An urn holds \(N\) balls, \(c_1\) marked “outcome 1” and
\(c_2=N-c_1\) marked “outcome 2.” Draw \(r_1\) balls without replacement for
group 1; the rest are group 2. Let \(A\) be how many of those \(r_1\) balls are
marked outcome 1. Then

\[
P(A=x) = \frac{\binom{c_1}{x}\binom{c_2}{r_1-x}}{\binom{N}{r_1}}.
\]

The symbols \(\binom{\cdot}{\cdot}\) here are **binomial coefficients** (counts
of combinations), not binomial probabilities. Read \(\binom{n}{k}\) as “number
of ways to choose \(k\) items from \(n\).” Under the null every way to choose
group 1's \(r_1\) seats from the \(N\) units is equally likely, so the
probability is a ratio of counts:

- **Denominator** \(\binom{N}{r_1}\): ways to choose which \(r_1\) of the \(N\)
  units go to group 1 (the rest go to group 2).
- **Numerator first factor** \(\binom{c_1}{x}\): ways to fill \(x\) of those
  group-1 seats from the \(c_1\) outcome-1 units.
- **Numerator second factor** \(\binom{c_2}{r_1-x}\): ways to fill the remaining
  \(r_1-x\) group-1 seats from the \(c_2\) outcome-2 units.

Multiply the two numerator factors because you choose the outcome-1 members and
the outcome-2 members of group 1 separately, then divide by the total number of
group-1 assignments. The result is already a probability; there is no extra
\(p^{x}(1-p)^{r_1-x}\) factor.

**Contrast with a binomial probability.** If each of \(r_1\) independent trials
had the same success chance \(p\), you would write

\[
P(A=x) = \binom{r_1}{x}\,p^{x}(1-p)^{r_1-x}.
\]

That binomial coefficient only chooses which of the \(r_1\) trials succeed; the
powers of \(p\) and \(1-p\) supply the probability weight, and the total number
of successes is not forced to equal a fixed \(c_1\). Fisher's hypergeometric
formula instead uses three counting coefficients and a locked column total, with
equal weight on every seat assignment that respects the margins.

The formula applies for every integer \(x\) with
\(\max(0,r_1-c_2)\le x\le \min(r_1,c_1)\). Each such \(x\) fills the whole
\(2\times2\) table once the margins are fixed. Fisher's calculation uses this
urn story with the observed \(N,r_1,c_1,c_2\).
{{< /panel >}}

So the conditional null probability for a candidate count \(x\) is

\[
\begin{aligned}
P(A=x\mid r_1,r_2,c_1,c_2)
&= \frac{\binom{c_1}{x}\binom{c_2}{r_1-x}}{\binom{N}{r_1}}.
\end{aligned}
\]

## Worked toy example

Suppose the observed table is:

|  | hit (outcome 1) | miss (outcome 2) | row total |
|---|---:|---:|---:|
| method A (group 1) | \(a=6\) | \(b=2\) | \(r_1=8\) |
| method B (group 2) | \(c=1\) | \(d=4\) | \(r_2=5\) |
| column total | \(c_1=7\) | \(c_2=6\) | \(N=13\) |

The margins are written on the edge of that table: \(r_1=8\), \(r_2=5\),
\(c_1=7\), \(c_2=6\), and \(N=13\). Not every integer is a legal value of
\(A\). You cannot put more hits in method A than exist overall, and you cannot
put more hits there than method A has seats:

\[
0 \le x \le \min(r_1,c_1) = \min(8,7) = 7.
\]

You also cannot leave method A so short of hits that the six misses cannot fill
its remaining seats. Method A needs \(r_1=8\) seats; if it takes only \(x\)
hits, it needs \(8-x\) misses, and there are only \(c_2=6\) misses available, so
\(8-x\le 6\), which is \(x\ge 2\). Combining both sides:

\[
\max(0,r_1-c_2) \le x \le \min(r_1,c_1),
\]

which here is \(2\le x\le 7\). So the possible values are
\(x=2,3,4,5,6,7\). Each one corresponds to one full table with these same
margins. Two examples (rows still method A / method B; columns still hit /
miss):

\[
x=6:\quad
\begin{bmatrix}
6 & 2 \\
1 & 4
\end{bmatrix}
\qquad
x=7:\quad
\begin{bmatrix}
7 & 1 \\
0 & 5
\end{bmatrix}.
\]

Both keep \(r_1=8\), \(r_2=5\), \(c_1=7\), \(c_2=6\), and \(N=13\). The
\(x=6\) matrix is the observed table; \(x=7\) moves one more hit into method
A and therefore one more miss into method B.

Now choose the scientific question **before** looking at the table: is method
A better at producing hits than method B? That is a **one-sided** alternative
in the “A has more hits” direction. Under the null the groups are equally
prone to hits, so large values of \(A\) (many hits piled into method A) are the
surprising ones for that claim.

A **tail** is the set of outcomes counted as “at least this extreme” for the
chosen direction. A **one-sided tail** looks only in that one direction. We
observed \(a=6\), so the one-sided upper tail is every legal table with at
least as many method-A hits as we saw: \(x=6\) and \(x=7\). The
[p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}}) is the
probability of landing in that tail **if the null is true** (here: if the two
methods are equally prone to hits). It is not the probability that the null is
true, and not the probability that the methods are equal. Tables with
\(x\le 5\) go the other way and are left out of this one-sided sum (a
two-sided test would also count an equally extreme lower direction).

\[
\begin{aligned}
p_{\mathrm{greater}}
&= P(A\ge6) \\
&= \frac{\binom{7}{6}\binom{6}{2}+\binom{7}{7}\binom{6}{1}}{\binom{13}{8}} \\
&= \frac{111}{1287}
\approx 0.08625.
\end{aligned}
\]

A **two-sided** test also counts tables that are extreme in the other
direction (method A unusually *short* of hits). SciPy and R use a common rule
for what “extreme” means here: compare tables by how **rare** they are under
the null, not only by how large \(A\) is.

Under the null, every way to choose which \(r_1\) of the \(N\) units sit in
group 1 is equally likely. There are \(\binom{N}{r_1}\) such assignments in
total (here \(\binom{13}{8}=1287\)). For a particular value \(x\), the number
of those assignments that produce \(A=x\) is

\[
w(x) = \binom{c_1}{x}\binom{c_2}{r_1-x}:
\]

choose which \(x\) of the \(c_1\) hits go to group 1, and which \(r_1-x\) of the
\(c_2\) misses fill the rest of group 1. Call that count the table’s
**weight**. The null probability is the weight divided by the total number of
assignments,

\[
P(A=x) = \frac{w(x)}{\binom{N}{r_1}} = \frac{\binom{c_1}{x}\binom{c_2}{r_1-x}}{\binom{N}{r_1}}.
\]

A smaller weight means a rarer table under the null, so comparing null
probabilities is the same as comparing weights. The two-sided rule keeps every
table whose weight is **less than or equal to** the observed table’s weight,
then sums those probabilities.

For this toy table the counts and null probabilities are:

| \(x\) | weight \(\binom{7}{x}\binom{6}{8-x}\) | null \(P(A=x)\) |
|---:|---:|---:|
| 2 | 21 | \(21/1287\approx0.0163\) |
| 3 | 210 | \(210/1287\approx0.1632\) |
| 4 | 525 | \(525/1287\approx0.4079\) |
| 5 | 420 | \(420/1287\approx0.3263\) |
| 6 (observed) | 105 | \(105/1287\approx0.0816\) |
| 7 | 6 | \(6/1287\approx0.0047\) |

The observed table has weight 105. Tables with weight \(\le 105\) are
\(x=2\) (weight 21), \(x=6\) (105), and \(x=7\) (6). The middle values
\(x=3,4,5\) are *more* common under the null than what we saw, so they are not
treated as extreme under this convention. Adding the three kept probabilities
gives

\[
p = \frac{21+105+6}{1287} = \frac{132}{1287} \approx 0.10256.
\]

Notice \(x=2\) enters even though it is on the low side of \(A\): it is rarer
under the null than the observed table. Other two-sided definitions exist, so
a report should name the software and convention rather than write only
“two-sided Fisher \(p\).”

This standard-library Python reproduces both values without an asymptotic
approximation:

```python
from math import comb

table = ((6, 2), (1, 4))
row_1 = sum(table[0])
column_1 = table[0][0] + table[1][0]
total = sum(map(sum, table))
column_2 = total - column_1

support = range(
    max(0, row_1 - column_2),
    min(row_1, column_1) + 1,
)
denominator = comb(total, row_1)
weights = {
    x: comb(column_1, x) * comb(column_2, row_1 - x)
    for x in support
}

observed = table[0][0]
observed_weight = weights[observed]
p_greater = sum(
    weight for x, weight in weights.items() if x >= observed
) / denominator
p_two_sided = sum(
    weight for weight in weights.values()
    if weight <= observed_weight
) / denominator

print(p_greater)    # 0.08624708624708624
print(p_two_sided)  # 0.10256410256410256
```

## What the design must justify

- **Counts and units:** cells contain counts of observational or randomized
  units, not percentages, fitted scores, or repeated rows treated as new
  independent units.
- **Conditional reference set:** the calculation conditions on the observed
  margins. A design with fixed group sizes can still justify the conditional
  null by conditioning on the total number of outcomes. Because the attainable
  conditional \(p\)-values are discrete, the rejection probability at a nominal
  level \(\alpha\) can be strictly below \(\alpha\) for particular fixed
  margins. In that sense the test can be conservative. When only the group
  sizes are fixed by design, unconditional exact procedures such as Barnard's
  or Boschloo's test may be more powerful, although they use a different
  reference construction. State the sampling or randomization design that makes
  the chosen reference set relevant.
- **Dependence and reference distribution:** the sampling or randomization
  design must justify the hypergeometric conditional distribution used by the
  test. In the usual independent-groups sampling model, observational units
  must be independent; clustered, longitudinal, repeated, or otherwise
  dependent observations require a method that represents that dependence. A
  fixed-allocation randomized design may justify Fisher's test through its
  assignment mechanism rather than through independent Bernoulli sampling.
- **Pairing:** if both methods are evaluated on the same items, the observations
  are paired. A method-by-outcome Fisher table discards that matching. For
  paired binary outcomes, an exact McNemar test instead uses the two discordant
  pair counts. For paired continuous or ranked differences, use a paired method
  such as the
  [Wilcoxon signed-rank test]({{< relref "knowledge-base/glossary/wilcoxon/index.md" >}}).
- **Sidedness:** choose a one-sided direction before seeing the table. A
  two-sided test needs an explicit extremeness convention.
- **Selection:** the test does not correct for trying many outcomes, thresholds,
  subsets, or analysis rules and reporting the smallest \(p\)-value.

The word **exact** means the conditional null probabilities are calculated
without a large-sample approximation. It does not mean the study design,
dependence assumptions, or scientific interpretation are automatically
correct.

## What to report with the p-value

Report the full count table, the alternative and sidedness, the two-sided
convention when applicable, the software and version, and an effect estimate
with an interval. State the odds-ratio estimator and interval construction:
for example, the sample cross-product odds ratio \(\widehat{\theta}=ad/bc\),
or the conditional maximum-likelihood estimate from the noncentral
hypergeometric model. These are not always numerically identical; SciPy's
`fisher_exact` reports the sample estimate, while R's `fisher.test` reports
the conditional MLE. A Fisher \(p\)-value is not the probability that the null
hypothesis is true, not an effect size, and not a causal conclusion. Zero cells
can make the simple sample odds ratio zero or infinite even though the exact
\(p\)-value remains finite, another reason to show the counts.

When both methods are evaluated on the same items, their separate total hit and
miss counts may be shown descriptively, but an ordinary Fisher \(p\)-value does
not account for within-item pairing. Use a paired analysis, such as an exact
McNemar test, for inference about the matched comparison.

See also: [p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}})
and
[Wilcoxon signed-rank test]({{< relref "knowledge-base/glossary/wilcoxon/index.md" >}}).

## Sources

- [SciPy 1.17: `scipy.stats.fisher_exact`](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.fisher_exact.html)
- [SciPy tutorial: Fisher's exact test](https://docs.scipy.org/doc/scipy/tutorial/stats/hypothesis_fisher_exact.html)
- [R `stats::fisher.test` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html)
