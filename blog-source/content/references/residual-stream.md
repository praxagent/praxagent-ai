---
title: "Residual stream"
slug: "residual-stream"
summary: "The model's running hidden-state scratchpad: each layer reads it, adds an update, and passes it on."
---

The **residual stream** is the transformer's shared hidden-state vector that every layer reads and writes. Mid-network it is a high-dimensional vector, not English. Readouts such as the [logit lens]({{< relref "logit-lens.md" >}}) and [Jacobian lens]({{< relref "jacobian-lens.md" >}}) score that vector against vocabulary directions.

See also: [workspace]({{< relref "workspace.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
