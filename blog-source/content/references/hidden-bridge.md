---
title: "Hidden bridge"
slug: "hidden-bridge"
summary: "An intermediate entity a model must use to answer a two-hop question, but that never appears in the prompt or the generated answer."
aliases:
  - /references/bridge/
  - /references/country-bridge/
---

A **hidden bridge** (also *bridge entity*, *country-bridge* in our release note) is the unspoken intermediate in a two-hop question.

Example: “What is the capital of the country where the Statue of Liberty stands?” The bridge is **United States** / **America**. The asked answer is a capital city. A strict audit requires the bridge string to be absent from the prompt **and** from the model's [greedy continuation]({{< relref "greedy-continuation.md" >}}).

See also: [Jacobian lens]({{< relref "jacobian-lens.md" >}}), [logit lens]({{< relref "logit-lens.md" >}}), [greedy continuation]({{< relref "greedy-continuation.md" >}}), [workspace]({{< relref "workspace.md" >}}).
