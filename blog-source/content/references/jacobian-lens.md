---
title: "Jacobian lens"
slug: "jacobian-lens"
summary: "A corpus-averaged linear map that turns a mid-layer residual-stream state into vocabulary-ranked readout scores."
pro_reviewed: true
aliases:
  - /references/j-lens/
  - /references/jlens/
---

A **Jacobian lens** (J-lens) is a fixed linear transport built from derivatives of the network downstream of layer \(\ell\). If \(F_{\ell\rightarrow L}\) denotes the remaining computation, a prompt-specific local Jacobian describes how a small change propagates:

\[
\delta h_L \approx
\frac{\partial F_{\ell\rightarrow L}}{\partial h_\ell}\,
\delta h_\ell.
\]

A released J-lens averages such local Jacobians over a fitting corpus and over prescribed source/target positions, schematically

\[
J_\ell = \mathbb{E}_{(x,t,t')\sim\mathcal{C}}
\left[
\frac{\partial h_{L,t'}}{\partial h_{\ell,t}}
\right], \qquad t'\ge t.
\]

The average makes \(J_\ell\) reusable, but it also smooths over context dependence. It is not the exact local Jacobian for every new prompt. A strict local Taylor reconstruction would also include a context-dependent intercept and apply the Jacobian to a perturbation; the lens deliberately reuses only the averaged linear part as a context-general directional transport.

To inspect a residual or a direction \(v\), the transported vector is normalized and [unembedded]({{< relref "unembedding.md" >}}) into vocabulary scores:

\[
z_\ell(v) = W_U\,N_f(J_\ell v) + b_U.
\]

The output bias \(b_U\) is optional. Omit it when the architecture has no bias or when the analysis explicitly defines a bias-free directional score; document that choice because it can affect absolute vocabulary ranks.

```python
# Matrix orientation varies by implementation; shapes should be checked.
transported = J[layer] @ vector
scores = output_head(final_norm(transported))
probe_rank = rank_of(scores, probe_token_id)
```

This is best read as a **verbalizability or vocabulary-alignment readout** under the corpus-derived transport. It is not simply “what the model is about to say.” A highly ranked token can be absent from the eventual continuation, and an important internal computation need not align cleanly with any single token.

{{< reference-figure src="references/glossary/lens-transports.svg" alt="Comparison of identity, Jacobian, and randomized transports from a middle-layer residual vector to vocabulary scores." caption="The output head is held fixed while the transport changes. Comparing identity, corpus-derived Jacobian, and randomized transports tests whether the derived transport contributes beyond the stated baselines; what the comparison isolates depends on which scale or geometry the random-J construction preserves." >}}

## Not the tuned lens

A tuned lens learns a per-layer affine translator directly to predict the model's final output distribution. A Jacobian lens instead estimates its transport by averaging downstream derivatives, giving it a local perturbation interpretation before that averaging. Both are linear probes, but one is optimized for output prediction while the other estimates average local propagation; they should not be treated as interchangeable.

It is not free. You download (or estimate) a published matrix file, hash it, and apply it per layer. The cheap baseline that skips the corpus-derived map is the [logit lens]({{< relref "logit-lens.md" >}}).

## Worked interpretation

For a [hidden-bridge]({{< relref "hidden-bridge.md" >}}) prompt, score the same absent bridge token at the same position and layers with the J-lens, the logit lens, and [random-J]({{< relref "random-j.md" >}}). If only the corpus-derived transport gives the bridge a strong, repeatable rank, that supports a transport-specific readout of bridge-related vocabulary on those examples. It still does not prove that the model would say the bridge, that a single bridge variable exists, or that the measured state caused the final answer.

**Use it when** you care about intermediate directions whose downstream vocabulary alignment may be poorly expressed by the identity readout. **Do not claim it was necessary** if the logit lens already sees the same signal, and do not turn a probe result into a causal claim without an intervention.

See also: [logit lens]({{< relref "logit-lens.md" >}}), [residual stream]({{< relref "residual-stream.md" >}}), [hidden bridge]({{< relref "hidden-bridge.md" >}}), [workspace]({{< relref "workspace.md" >}}).
