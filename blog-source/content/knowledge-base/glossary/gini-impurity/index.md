---
title: "Gini impurity"
slug: "gini-impurity"
summary: "A measure of how mixed the class labels are inside a group of rows, used by decision trees to choose which feature cutoff to ask next."
og_image: "gini-split.png"
og_image_alt: "Binary Gini impurity peaks at 0.5 for an even mix, and a useful cutoff lowers the weighted child impurity from 0.42 to 0.34."
draft: false
pro_reviewed: true
---

**Gini impurity** scores how mixed the labels are inside a group of training
rows. A group that is all one class scores 0. For a binary label, an even split
scores the maximum, 0.5. Classification trees configured to use the Gini
criterion choose each cutoff by that score: among the allowed questions, keep
the one that lowers impurity the most.

## The binary definition

Take a group of \(n\) rows, of which a fraction \(q\) are labeled positive.
Binary Gini impurity is

\[
G=1-q^{2}-(1-q)^{2}=2q(1-q).
\]

It is 0 at \(q=0\) and at \(q=1\), and it peaks at \(0.5\) when \(q=0.5\). The
middle form makes the idea plain. Draw two rows independently from the group,
with replacement: \(q^{2}\) is the probability that both are positive and
\((1-q)^{2}\) is the probability that both are negative, so \(G\) is the
probability that their labels disagree. The drawing must be with replacement
for those squares to be exact; sampling two distinct rows without replacement
gives a slightly different disagreement probability. The same
\(q(1-q)\) shape appears as the uncertainty term inside
[Brier loss]({{< relref "knowledge-base/glossary/brier-loss/index.md" >}}), for
the same reason: both measure how unsettled a binary outcome is before anything
else is known.

With more than two classes the definition is
\(G=1-\sum_{c}q_{c}^{2}\), where \(q_{c}\) is the fraction of the group in
class \(c\). The binary formula above is that sum written out for two classes.
The maximum depends on how many classes there are: with \(K\) classes present
in equal proportion, \(G=1-1/K\), so the ceiling of 0.5 quoted above is the
binary case.

{{< reference-figure
  src="gini-split.svg"
  alt="Left: binary Gini impurity rises from 0 at a pure negative group to a peak of 0.5 at an even mix and back to 0 at a pure positive group, with a marked parent group at positive rate 0.30 and impurity 0.42. Right: that parent is split two ways. Cutoff A lowers the weighted child impurity to 0.34. Cutoff B leaves both children looking like the parent, so the improvement is zero."
  caption="Toy arithmetic, not a fitted tree. Left: the impurity curve against the positive fraction in a group. Right: the parent group of 100 rows with 30 positives, impurity 0.42, under two candidate cutoffs. Cutoff A pulls most positives to one side and earns an improvement of 0.08. Cutoff B splits the rows in half without sorting the labels, so both children keep impurity 0.42 and the question is discarded."
>}}

## Choosing a cutoff

A tree does not report the impurity of a leaf as a quality score for the
finished model. It uses impurity only while growing. At a node, every allowed
cutoff on every offered feature is scored by the **impurity decrease**

\[
\Delta G=G_{\mathrm{parent}}
-\left(\frac{n_{L}}{n}G_{L}+\frac{n_{R}}{n}G_{R}\right),
\]

where \(n_{L}\) and \(n_{R}\) are the row counts in the left and right
children. The tree keeps the cutoff with the largest \(\Delta G\). An
improvement of zero means the question sorted nothing on these training rows.

The figure's parent has \(q=0.30\) and \(G=0.42\). Cutoff A makes children with
impurities 0.18 and 0.50; the weighted average is 0.34, so \(\Delta G=0.08\).
Cutoff B makes two children identical to the parent, so \(\Delta G=0\).

```python
def gini(positive_fraction: float) -> float:
    q = positive_fraction
    return 2.0 * q * (1.0 - q)

assert abs(gini(0.30) - 0.42) < 1e-12
assert abs(gini(0.10) - 0.18) < 1e-12
assert abs(0.5 * gini(0.10) + 0.5 * gini(0.50) - 0.34) < 1e-12
```

scikit-learn's `DecisionTreeClassifier` and `RandomForestClassifier` use Gini
impurity when `criterion="gini"`, which is the default. Both accept multiclass
targets, where the criterion is the general \(1-\sum_{c}q_{c}^{2}\); a binary
target is the case that reduces to the \(2q(1-q)\) form above
([documentation](https://scikit-learn.org/stable/modules/tree.html)).

## Entropy is the common alternative

The other standard split rule replaces Gini with **binary entropy**,
\(H(q)=-q\log_{2}q-(1-q)\log_{2}(1-q)\), using the usual convention
\(0\log_{2}0=0\) so that a pure group scores 0 rather than being undefined. The
tree then keeps the cutoff that most reduces that quantity (**information
gain**). Entropy also bottoms at a pure group and peaks at an even mix, but it
rises more sharply near the middle. The two criteria can select different
splits. Treat the criterion as a fitting choice and judge the finished model on
held-out data, rather than reading the choice itself as evidence that one
forest is better.

## What Gini impurity does not establish

- It is a training-time split rule, not a held-out performance metric. A tree
  can drive every leaf impurity to zero and still fail on new rows.
- A leaf's impurity is not the same object as the probability the leaf reports.
  The probability is the positive fraction \(q\) itself; impurity is \(2q(1-q)\).
- A feature that never wins a split on one path can still win elsewhere, so a
  zero improvement at one node is not a global irrelevance claim.
- Gini is not a proper scoring rule for the finished model's probabilities.
  Grade those with
  [Brier loss]({{< relref "knowledge-base/glossary/brier-loss/index.md" >}}) or
  [log loss]({{< relref "knowledge-base/glossary/log-loss/index.md" >}}).

See also: [Brier loss]({{< relref "knowledge-base/glossary/brier-loss/index.md" >}}),
[bootstrap]({{< relref "knowledge-base/glossary/bootstrap/index.md" >}}), and the
Deep Dive
[Logistic Regression and Random Forest]({{< relref "knowledge-base/deep-dives/logistic-regression-random-forest/index.md" >}}),
where Gini chooses every split inside the fitted forest.
