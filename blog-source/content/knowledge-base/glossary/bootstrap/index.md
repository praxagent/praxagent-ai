---
title: "Bootstrap"
slug: "bootstrap"
summary: "A resampling method that repeatedly draws observed units with replacement to study how a statistic varies."
og_image: "bootstrap-resample-units.png"
og_image_alt: "A bootstrap resample selects one observed plant twice and omits another."
draft: false
pro_reviewed: true
---

The **bootstrap** is a resampling method that repeatedly draws from the observed data **with replacement**, meaning that a selected unit is returned to the pool and can be selected again. A resample normally contains the same number of units as the original dataset, but some original units can appear several times and others not at all.

For each resample, calculate the statistic of interest, such as a mean, median, accuracy difference, or retrieval-score difference. The resulting distribution shows how that statistic changes across bootstrap resamples.

{{< reference-figure
  src="bootstrap-resample-units.svg"
  alt="Four observed plants are resampled into four slots, with plant P3 selected twice and plant P2 omitted."
  caption="One illustrative bootstrap resample draws four times from the four observed plants with replacement, producing P3, P1, P3, and P4. P3 can repeat because each selected plant is returned before the next draw; P2 happens not to appear. A real analysis repeats this process many times and recalculates the chosen statistic. The diagram does not imply that plants are always the correct resampling unit."
>}}

## A small example

For observed values ([2,4,5,9]), one possible resample is ([5,2,5,9]). Its mean is 5.25, while the original mean is 5. Repeating the draw many times gives many bootstrap means.

```python
import random

values = [2, 4, 5, 9]
rng = random.Random(7)

bootstrap_means = []
for _ in range(10_000):
    resample = rng.choices(values, k=len(values))
    bootstrap_means.append(sum(resample) / len(resample))

bootstrap_means.sort()
lower = bootstrap_means[int(0.025 * len(bootstrap_means))]
upper = bootstrap_means[int(0.975 * len(bootstrap_means))]
```

The last two lines form a simple percentile interval. Other bootstrap confidence intervals use different corrections and assumptions. Report the interval method, number of resamples, random seed when reproducibility matters, and the exact statistic.

## Choose the resampling unit carefully

The bootstrap does not make dependent observations independent. If ten measurements come from each plant, resampling all 100 measurements as though they were unrelated can overstate the effective information. A cluster bootstrap might resample whole plants and keep each plant's measurements together. For retrieval evaluation, resampling queries is common when the question is how mean performance varies across the observed query set.

The unit should match the source of independent sampling or the population claim. Related paraphrases, repeated measurements, sites, batches, and families may require grouped or hierarchical resampling rather than a row-wise bootstrap. See [i.i.d.]({{< relref "knowledge-base/glossary/iid/index.md" >}}) for why observed rows are not automatically independent and identically distributed.

## What a bootstrap interval does not establish

- It does not repair selection bias, data leakage, a poor statistic, or an unrepresentative dataset.
- A 95% interval is not a 95% probability that a fixed true value lies inside this already computed interval.
- Too few independent units can make the resampling distribution unstable or misleading.
- Overlapping intervals do not by themselves provide a calibrated test of a difference.

See also: [i.i.d.]({{< relref "knowledge-base/glossary/iid/index.md" >}}), [p-value]({{< relref "knowledge-base/glossary/p-value/index.md" >}}), [retrieval ranking metrics]({{< relref "knowledge-base/glossary/retrieval-ranking-metrics/index.md" >}}).
