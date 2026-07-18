---
title: "Workspace"
slug: "workspace"
summary: "A workspace-like mid-network regime studied through selected residual directions and positions, not the whole middle of a model."
pro_reviewed: true
aliases:
  - /references/workspace/
---

In Anthropic's global-workspace framing, **workspace** names a workspace-like regime in the middle of the network. The evidence concerns a **sparse, selected subset of directions or features** with workspace-like behavior, not every coordinate, token position, or computation in the entire mid-layer [residual stream]({{< relref "residual-stream.md" >}}).

{{< reference-figure src="knowledge-base/glossary/workspace-selection.svg" alt="Only a few selected directions and token positions inside a prespecified middle-layer band feed the Jacobian readout; the rest of the band is not measured as workspace." caption="Figure structure: early, middle, and late regions provide a coarse framing. The middle region is expanded into many candidate states; only a labeled, hatched subset is selected by frozen layer, position, and probe rules. Those selected states, not the entire middle band, are mapped through the Jacobian readout to vocabulary scores and ranks. The diagram is illustrative and does not specify universal layer boundaries, feature density, or consciousness." >}}

In our notes, **workspace rank** is an operational abbreviation: apply a [Jacobian lens]({{< relref "jacobian-lens.md" >}}) to a chosen residual position across a prespecified middle-layer band, then record a probe token's vocabulary [rank]({{< relref "rank.md" >}}). Lower rank means the probe is nearer the top **under that readout**. It is not a direct neural activation measurement.

## Worked example

For an indirect capital question, an audit might:

1. freeze the prompt, token position, middle-layer range, and bridge-word probe;
2. compute that probe's J-lens rank at every chosen layer;
3. compare the same statistic with the [logit lens]({{< relref "logit-lens.md" >}}), [random-J]({{< relref "random-j.md" >}}), and matched prompts; and
4. report the full trajectory or the prospectively defined [best-rank]({{< relref "best-rank.md" >}}).

A strong controlled result says that a particular vocabulary direction is readable from those selected states with that instrument. It does not establish that the entire middle of the model is a unified blackboard.

## Claim boundary

“Global workspace” here is a mechanistic analogy and an empirical hypothesis for the studied models and methods. It is not a universal layer taxonomy for every transformer, and it is not evidence that a model is conscious. Claims about awareness or subjective experience require arguments and evidence that a vocabulary readout cannot provide.

See also: [residual stream]({{< relref "residual-stream.md" >}}), [best-rank]({{< relref "best-rank.md" >}}).
