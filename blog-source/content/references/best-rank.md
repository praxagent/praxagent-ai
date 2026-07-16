---
title: "Best-rank"
slug: "best-rank"
summary: "The minimum vocabulary rank a probe word hits across layers and positions; a max-statistic that needs matched controls."
---

**Best-rank** is the *minimum* (best) vocabulary rank a target word or small lexicon achieves across many layers and prompt positions. Lower is stronger. Because it is a max-statistic, it is easy to over-read unless every control transport is scored the same way.

See also: [rank]({{< relref "rank.md" >}}), [logit lens]({{< relref "logit-lens.md" >}}), [prompt echo]({{< relref "prompt-echo.md" >}}).
