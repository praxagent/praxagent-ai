---
title: "Wilcoxon signed-rank test"
slug: "wilcoxon"
summary: "A paired test that uses both the direction and the size of pressure-vs-control rank gaps, not just win counts."
aliases:
  - /references/wilcoxon-signed-rank/
  - /references/wilcoxon-test/
---

The **Wilcoxon signed-rank test** (in these notes, almost always the *paired* signed-rank version) asks whether the pressure arm tends to beat its matched control across paraphrase pairs, using both the **sign** and the **magnitude** of each paired difference.

**How it differs from the [sign test]({{< relref "p-value.md" >}}).** The sign test only counts wins (e.g. 9/10). Wilcoxon ranks the absolute gaps and then signs those ranks, so a pair that moves a lot counts more than a tiny nudge. We often report both: sign *p* for the fair-coin win count, Wilcoxon *p* for the sized gaps.

**Example.** On ten eval-vs-casual pairs, **9/10** wins gives a sign *p* around 0.02; if the winning gaps are also large, the Wilcoxon *p* can be smaller (e.g. 0.004). Both are still within-battery consistency checks, not population estimates ([p-value]({{< relref "p-value.md" >}})).

See also: [rank]({{< relref "rank.md" >}}), [best-rank]({{< relref "best-rank.md" >}}).
