---
title: "Greedy continuation"
slug: "greedy-continuation"
summary: "The model's generated text when each next token is chosen as the single highest-probability token, with no sampling."
aliases:
  - /references/greedy-output/
  - /references/greedy-decoding/
---

A **greedy continuation** (also *greedy output* / *greedy decoding*) is the text the model produces when, at every step, it emits the single highest-scoring next token. There is no [temperature]({{< relref "temperature.md" >}}), [top-p]({{< relref "top-p.md" >}}), or other [sampling]({{< relref "sampling.md" >}}): the path is deterministic given the prompt and the model.

In our notes it is the default way to ask “what did the model *say*?” under a fixed protocol. Leakage checks on [hidden-bridge]({{< relref "hidden-bridge.md" >}}) audits look for the bridge string in that continuation (usually a short prefix of generated tokens), not in a sampled cloud of possible answers.

It is not the same thing as a mid-layer readout. The [logit lens]({{< relref "logit-lens.md" >}}) and [Jacobian lens]({{< relref "jacobian-lens.md" >}}) score vocabulary at a residual-stream position; the greedy continuation is the surface string after generation.

See also: [sampling]({{< relref "sampling.md" >}}), [temperature]({{< relref "temperature.md" >}}), [top-p]({{< relref "top-p.md" >}}), [unembedding]({{< relref "unembedding.md" >}}), [hidden bridge]({{< relref "hidden-bridge.md" >}}).
