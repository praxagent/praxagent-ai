---
title: "Top-p"
slug: "top-p"
summary: "Nucleus sampling: keep only the smallest set of tokens whose cumulative probability reaches p, then sample within that set."
aliases:
  - /references/nucleus-sampling/
  - /references/top_p/
---

**Top-p** (also *nucleus sampling*) truncates the next-token distribution before [sampling]({{< relref "sampling.md" >}}). Tokens are ordered by probability; you keep the smallest prefix whose cumulative mass is at least \(p\) (for example 0.9), zero out the rest, renormalize, and sample from what remains.

It is a different knob from [temperature]({{< relref "temperature.md" >}}): temperature reshapes the whole distribution; top-p hard-cuts the long tail. They are often used together.

Like temperature, top-p is irrelevant to a [greedy continuation]({{< relref "greedy-continuation.md" >}}), which never samples.

See also: [temperature]({{< relref "temperature.md" >}}), [sampling]({{< relref "sampling.md" >}}), [greedy continuation]({{< relref "greedy-continuation.md" >}}).
