---
title: "Jacobian lens"
slug: "jacobian-lens"
summary: "A fitted linear map that turns a mid-layer residual-stream state into vocabulary-ranked readout scores."
aliases:
  - /references/j-lens/
  - /references/jlens/
---

A **Jacobian lens** (J-lens) is a *fitted* linear transport \(J_\ell\) from a mid-layer residual-stream state into near-final coordinates, then unembedded into vocabulary ranks. The readout answers: what the network looks like it is “about to say,” without waiting for the next token.

It is not free. You download (or fit) a published matrix file, hash it, and apply it per layer. The cheap baseline that skips the fitted map is the [logit lens]({{< relref "logit-lens.md" >}}).

**Use it when** you care about mid-network content that may be decoupled from the next-token distribution. **Do not claim it was necessary** if the logit lens already sees the same signal.

See also: [logit lens]({{< relref "logit-lens.md" >}}), [residual stream]({{< relref "residual-stream.md" >}}), [hidden bridge]({{< relref "hidden-bridge.md" >}}), [workspace]({{< relref "workspace.md" >}}).
