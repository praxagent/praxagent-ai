---
title: "Base-rate bias"
slug: "base-rate-bias"
summary: "Common, high-prior tokens look artificially strong under mid-layer unembedding and best-rank statistics; a low rank can be the prior, not the experiment."
aliases:
  - /references/base-rate/
  - /references/base-rate-biased/
---

**Base-rate bias** (in these notes) means a probe word looks active in a lens readout mainly because it is a *common, high-prior* token, not because the prompt condition did distinctive work.

Two places it shows up:

1. **[Logit lens]({{< relref "logit-lens.md" >}}) / unembedding.** Mid-layer readouts reuse the model's output map. Frequent words and near-certain completions (famous capitals, function words, survival verbs like *self* / *shut*) sit near the top of that map by default. The Tuned Lens literature documents that the plain logit lens is especially prone to this kind of output-adjacent, base-rate-heavy readout ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112)).
2. **[Best-rank]({{< relref "best-rank.md" >}}).** Taking the *minimum* rank over many layers and positions pulls any high-prior word downward further. A random transport can look “good” on the same statistic.

**How we treat it.** Prefer pressure-vs-control contrasts, run [random-J]({{< relref "random-j.md" >}}) and logit-lens controls on the same probe, and do not read a low absolute rank on a famous capital or common survival word as proof of resistance or self-preservation.

See also: [rank]({{< relref "rank.md" >}}), [prompt echo]({{< relref "prompt-echo.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
