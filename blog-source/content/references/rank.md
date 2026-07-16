---
title: "Rank"
slug: "rank"
summary: "Position of a target word in a scored vocabulary list (1 = most active). Lower is stronger."
aliases:
  - /references/vocabulary-rank/
---

**Rank** is the position of a target word (or token) in a scored vocabulary list after unembedding: **1** means most active. Lower is stronger. On [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) the vocabulary has 248,320 entries, so a rank of a few hundred is still near the top of a huge list.

Notes often report a single number that is actually [best-rank]({{< relref "best-rank.md" >}}) (the minimum over layers and positions). Absolute ranks are not findings by themselves; compare a pressure arm to its matched control, and run the same statistic through [logit lens]({{< relref "logit-lens.md" >}}) and [random-J]({{< relref "random-j.md" >}}).

See also: [unembedding]({{< relref "unembedding.md" >}}), [workspace]({{< relref "workspace.md" >}}).
