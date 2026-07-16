---
title: "Motor layer / motor-late"
slug: "motor-layer"
summary: "Anthropic's name for late, near-output layers where J-lens readouts align with the imminent next token; motor-late features (e.g. digits) only become legible there."
aliases:
  - /references/motor-late/
  - /references/motor-local/
  - /references/motor-layers/
---

**Motor layers** come from Anthropic's workspace paper ([Gurnee, Lindsey et al. 2026](https://transformer-circuits.pub/2026/workspace/)): they divide the stack into early **sensory**, middle **workspace**, and late **motor** bands. In the motor band, [Jacobian lens]({{< relref "jacobian-lens.md" >}}) readouts stop looking like intermediate computation and line up with the model's imminent next token (the “about to say it now” regime, near the [unembedding]({{< relref "unembedding.md" >}})). Anthropic also uses **motor features** for SAE directions that fire when the model is about to utter a specific token.

**Motor-late** / **motor-local** (our shorthand in the pressure and release notes) applies that framing to a concrete finding: some probes, especially **digit tokens**, stay illegible through the mid-band [workspace]({{< relref "workspace.md" >}}) and only spike in the final fitted layers. The establishment is our free/local digit-geometry run on the n=24 397B lens ([`digit_geometry_397b.json`](https://github.com/praxagent/jacobian-lens-research-202607a/blob/d7ef84518135ee4c2d350a4b434a760e043114e9/projects/jacobian-lens-and-identifiability/experiments/lens_demo/digit_geometry_397b.json), script [`digit_geometry.py`](https://github.com/praxagent/jacobian-lens-research-202607a/blob/d7ef84518135ee4c2d350a4b434a760e043114e9/projects/jacobian-lens-and-identifiability/experiments/lens_demo/digit_geometry.py)): digit directions stay near-Gaussian-flat (κ≈3.5) through the workspace and spike (κ≈14.6) only at the last fitted layers. Writeup: [`lens_demo/results.md`](https://github.com/praxagent/jacobian-lens-research-202607a/blob/d7ef84518135ee4c2d350a4b434a760e043114e9/projects/jacobian-lens-and-identifiability/experiments/lens_demo/results.md) ("Where they live: deep, and *only* deep"). That is why arithmetic answers are a poor mid-band probe; prefer single-token city names and similar content words. The [Jacobian lens release note](/blog/posts/2026/07/praxagent-jacobian-lens-qwen3-5-397b-a17b/) alludes to the finding; the digit-geometry receipt is the primary evidence.

**Motor-layer convergence** is an eval name in our fitting receipts (not Anthropic's coinage): a healthy lens should agree more with next-token argmax as depth increases (near zero mid-network, higher at the last fitted layer).

See also: [logit lens]({{< relref "logit-lens.md" >}}), [best-rank]({{< relref "best-rank.md" >}}), [residual stream]({{< relref "residual-stream.md" >}}).
