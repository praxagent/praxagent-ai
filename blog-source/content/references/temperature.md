---
title: "Temperature"
slug: "temperature"
summary: "A scalar that sharpens or flattens the next-token distribution before sampling."
---

**Temperature** is a positive scalar applied to next-token logits before they are turned into probabilities for [sampling]({{< relref "sampling.md" >}}). Lower temperature sharpens the distribution toward the top tokens; higher temperature flattens it and makes lower-ranked tokens more likely. Temperature 1 leaves the model's calibrated distribution unchanged (under the usual softmax formulation).

Temperature does not apply to a [greedy continuation]({{< relref "greedy-continuation.md" >}}): greedy decoding never samples, so there is no distribution to reshape.

See also: [top-p]({{< relref "top-p.md" >}}), [sampling]({{< relref "sampling.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
