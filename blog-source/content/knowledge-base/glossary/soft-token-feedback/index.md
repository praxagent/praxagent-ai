---
title: "Soft-token feedback"
slug: "soft-token-feedback"
summary: "A feedback rule that feeds back a probability-weighted average of token embeddings instead of selecting a next token."
draft: true
pro_reviewed: true
og_image: "soft-token-feedback-card.png"
og_image_alt: "A model's probabilities weight two toy embeddings; their average returns to the same input slot after a fixed prompt."
---

**Soft-token feedback** feeds a language model a weighted average of token [embeddings]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), the vectors that ordinarily represent individual tokens. The weights come from a next-token probability distribution. The result is one vector, which need not equal any token's embedding. It is not a sampled word or a collection of separate model runs.

The probability-weighted construction appears in [Soft Thinking](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7396d1c54d51416958d63e285377103-Abstract-Conference.html). The feedback protocol matters: appending successive soft tokens creates a growing context, whereas repeatedly replacing one slot after a fixed prompt creates the fixed-window map used in this site's [dynamics note]({{< relref "posts/2026/10/chaos-bifurcations-language-model-feedback/index.md" >}}). These are different experiments.

## The fixed-window rule

Let \(V\) be the vocabulary, \(e_i\) the embedding vector for token \(i\), and \(p_i(x)\) the model's next-token probability for that token when the current feedback vector is \(x\). Hold the prompt, input position, temperature, and other control settings fixed. One step is

\[
x_{n+1}=F(x_n)=\sum_{i\in V}p_i(x_n)e_i,
\qquad p_i(x_n)\ge0,
\qquad\sum_{i\in V}p_i(x_n)=1.
\]

Here \(n\) counts feedback steps and \(F\) names the whole update rule. Feed \(x_{n+1}\) into the same input slot for the next step. The vocabulary probability vector and the resulting embedding vector are different objects, usually of very different lengths.

This weighted average lies in the [convex hull]({{< relref "knowledge-base/glossary/convex-hull/index.md" >}}) of the embeddings, meaning the set of all their nonnegative weighted averages with weights summing to one. If an implementation first drops low-probability tokens, it must specify the selection and renormalization rule. That modified update need not have the same dynamics as the full-vocabulary rule above.

{{< reference-figure
  src="soft-token-feedback.svg"
  alt="Toy probabilities of three quarters and one quarter produce the vector (1.5, 1), which replaces the old vector at the same input slot."
  caption="One illustrative update, not measured model data. A fixed prompt and current vector enter the model. Its toy probabilities are three quarters for embedding (2, 0) and one quarter for embedding (0, 4). The weighted contributions are (1.5, 0) and (0, 1), so their sum is (1.5, 1). That vector replaces the previous vector at the same slot after the unchanged prompt. The diagram specifies one update only; probabilities at the following step must be recomputed. No token is sampled, and the context does not grow."
>}}

## Worked example

For a toy vocabulary with embeddings \(e_A=(2,0)\) and \(e_B=(0,4)\), suppose the current probabilities are \(p_A=3/4\) and \(p_B=1/4\). Then

\[
x_{n+1}=\tfrac34(2,0)+\tfrac14(0,4)=(1.5,1).
\]

Choosing the most probable token would instead feed back \((2,0)\). Sampling would choose one of the two embeddings at random with the stated probabilities. Feeding their average into a nonlinear model does not generally equal averaging the results of feeding each embedding separately. The average therefore does not, by itself, mean that the model explored both reasoning paths.

## What the rule does not guarantee

A fixed deterministic update defines a **dynamical system**, a rule repeatedly applied to a changing state. It does not guarantee convergence, repeating behavior, chaos, memory, or better reasoning. Those are separate properties to test. A word displayed by an auxiliary readout is also not necessarily the state being fed back.

The numerical implementation is part of the specification. [Temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}), normalization, [dtype]({{< relref "knowledge-base/glossary/dtype/index.md" >}}), accumulation precision, and input position can change the update. Fixing the mathematical rule does not guarantee bit-identical trajectories across hardware or software implementations.

The schematic and its exact toy arithmetic are generated and checked by [make_figure.py](make_figure.py). Run it with `--verify` to check the saved SVG and PNG.
