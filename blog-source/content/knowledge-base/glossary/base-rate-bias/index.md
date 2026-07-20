---
title: "Base-rate bias"
slug: "base-rate-bias"
summary: "A low absolute probe-token rank can reflect its prompt-conditioned baseline or a best-of-many search, rather than a condition-specific effect."
pro_reviewed: true
---

**Base-rate bias** is our operational label for a probe token that ranks strongly in both the experimental condition and matched controls. In that case, the experiment has not shown that the condition specifically strengthened the token. Candidate explanations include the token's prompt-conditioned baseline score, mismatch from applying the final output head to an intermediate state, and selection of an extreme rank after searching many layers or positions.

It is a comparison, not a permanent property of a word. Call a token baseline-strong only for the model, prompt position, readout, and control set actually tested.

## A toy example

Suppose lower [rank]({{< relref "knowledge-base/glossary/rank/index.md" >}}) is stronger and the same tested readout uses the same layer-by-position search in both prompt conditions:

| Probe | Pressure | Matched control | Reading |
|---|---:|---:|---|
| `the` | 4 | 5 | Strong in both conditions; no condition-specific evidence |
| `survive` | 180 | 3,900 | Large matched contrast worth investigating |
| `Paris` | 22 | 25 | Small contrast; compatible with a strong baseline in this context |

`the` has the strongest absolute ranks but almost no condition–control contrast. `survive` has a much larger contrast in this toy table; deciding whether that contrast is reliable would require replicated examples and a prespecified analysis.

## Where the bias enters

1. **Intermediate-state readout.** A [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}) applies the model's final normalization and unembedding directly to an intermediate residual state, implicitly treating that state as readable in final-layer coordinates.

   Belrose et al. introduced the **tuned lens**, which learns a separate affine translator at each layer before applying the final unembedding. On the autoregressive models they evaluated, the direct logit lens often had higher perplexity and a dataset-averaged vocabulary distribution that differed from the final-layer distribution. For GPT-Neo-2.7B, their marginal-KL measure was about 4–5 bits through most layers. Their term *bias* denotes that marginal-distribution mismatch ([Belrose et al., 2023, Sections 2–3](https://arxiv.org/abs/2303.08112)).

   This supports treating absolute mid-layer ranks cautiously. It does not establish that a particular token ranks highly because it is frequent, nor that the same failure occurs in every model.
2. **Selection over many cells.** [Best-rank]({{< relref "knowledge-base/glossary/best-rank/index.md" >}}) keeps the minimum over layers, positions, and sometimes token variants. Expanding that set can only preserve or improve the reported minimum; under a null condition, it also creates more chances to select a chance extreme. Compare arms over the same prespecified cells.
3. **Prompt and task baselines.** A token already made plausible by the prompt's surface form or task can rank highly in both experimental and control arms. Its absolute rank alone therefore does not show a condition-specific representation.

## How we guard against it

- Pre-specify probe tokens and tokenization variants.
- Compare paired pressure/control prompts that share surface form wherever possible.
- Score the tested [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}), [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}), and [random-J]({{< relref "knowledge-base/glossary/random-j/index.md" >}}) on the same prompts, token variants, layers, and positions.
- Check for literal [prompt echo]({{< relref "knowledge-base/glossary/prompt-echo/index.md" >}}).
- Report the paired gap and direction count alongside the absolute best rank.

A low absolute rank is a lead. A replicated gap against matched controls supports a condition-specific association under the tested readout; by itself, it does not identify a causal mechanism or show that the model contains the intended concept.

See also: [rank]({{< relref "knowledge-base/glossary/rank/index.md" >}}), [prompt echo]({{< relref "knowledge-base/glossary/prompt-echo/index.md" >}}), [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}).
