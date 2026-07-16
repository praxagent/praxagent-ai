---
title: "Logit lens"
slug: "logit-lens"
summary: "The free mid-layer readout that skips any fitted transport: unembed the residual as if it were already final-layer coordinates."
aliases:
  - /references/identity-lens/
---

The **logit lens** (also called the *identity* transport in our notes) takes a mid-layer residual-stream vector and unembeds it *as if* it were already in final-layer coordinates. Same ranking code as a [Jacobian lens]({{< relref "jacobian-lens.md" >}}), no published matrix required.

It is the mandatory control. If the logit lens already surfaces your probe words, the fitted Jacobian file was not doing the distinctive work on that task.

See also: [base-rate bias]({{< relref "base-rate-bias.md" >}}), [Jacobian lens]({{< relref "jacobian-lens.md" >}}), [random-J]({{< relref "random-j.md" >}}), [best-rank]({{< relref "best-rank.md" >}}).
