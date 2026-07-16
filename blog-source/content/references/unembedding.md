---
title: "Unembedding"
slug: "unembedding"
summary: "The final linear map from residual-stream coordinates to a score for every vocabulary token."
aliases:
  - /references/output-head/
---

**Unembedding** (the output head) is the final linear map from the residual stream to a score for every vocabulary token. Softmax turns those scores into next-token probabilities. Mid-layer readouts reuse that map after (optionally) transporting the residual into near-final coordinates.

See also: [residual stream]({{< relref "residual-stream.md" >}}), [rank]({{< relref "rank.md" >}}), [logit lens]({{< relref "logit-lens.md" >}}).
