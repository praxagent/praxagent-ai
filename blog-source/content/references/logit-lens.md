---
title: "Logit lens"
slug: "logit-lens"
summary: "The free mid-layer readout that skips any fitted transport: unembed the residual as if it were already final-layer coordinates."
aliases:
  - /references/identity-lens/
---

The **logit lens** (also called the *identity* transport in our notes) takes one mid-layer [residual-stream]({{< relref "residual-stream.md" >}}) vector and applies the model's final readout *as if that vector were already in final-layer coordinates*. A typical implementation is

\[
z_{\ell,t} = W_U\,N_f(h_{\ell,t}) + b_U,
\]

where \(N_f\) is the model's final normalization and \(W_U\) and \(b_U\) are its [unembedding]({{< relref "unembedding.md" >}}) parameters. Architectural details differ: the correct normalization, bias, and tied-weight convention must match the model being inspected. Skipping the final normalization can materially change the ranking.

```python
# Schematic only: use the target model's actual final norm and output head.
h = residual[layer, position]
scores = output_head(final_norm(h))
ranking = scores.argsort(descending=True)
```

The softmax of `scores` is a convenient display distribution, but at an intermediate layer it is not necessarily calibrated as that layer's true next-token probability distribution. The rest of the network has not run yet.

## How to read the result

If *Paris* ranks first at a middle layer, the careful claim is: “under the model's final vocabulary readout, this selected normalized state gives *Paris* a very high score.” It is not yet a claim that the model will emit *Paris*, that a dedicated “Paris feature” exists, or that this score causally drives the answer. Later blocks can strengthen, erase, or redirect it.

It is the mandatory control. If the logit lens already surfaces your probe words, the fitted Jacobian file was not doing the distinctive work on that task.

Conversely, a difference between the identity readout and a [Jacobian lens]({{< relref "jacobian-lens.md" >}}) shows sensitivity to the transport. It does not, without matched prompts and interventions, establish that either lens has recovered the model's mechanism.

See also: [base-rate bias]({{< relref "base-rate-bias.md" >}}), [Jacobian lens]({{< relref "jacobian-lens.md" >}}), [random-J]({{< relref "random-j.md" >}}), [best-rank]({{< relref "best-rank.md" >}}).
