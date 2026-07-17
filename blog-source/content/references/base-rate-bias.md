---
title: "Base-rate bias"
slug: "base-rate-bias"
summary: "Common, high-prior tokens look artificially strong under mid-layer unembedding and best-rank statistics; a low rank can be the prior, not the experiment."
aliases:
  - /references/base-rate/
  - /references/base-rate-biased/
---

**Base-rate bias** is our operational name for a probe token looking strong in a readout even when the experimental condition did little or nothing distinctive. The strength may come from the model's ordinary next-token preferences, the output vocabulary geometry, or the search statistic—not from the construct being tested.

It is a diagnosis from **controls**, not a permanent property of a word. A token can be high-prior in one prompt position and informative in another.

## A toy example

Suppose lower [rank]({{< relref "rank.md" >}}) is stronger and every arm uses the same layer-by-position search:

| Probe | Pressure | Matched control | Random-J | Reading |
|---|---:|---:|---:|---|
| `the` | 4 | 5 | 6 | Strong everywhere; no condition-specific evidence |
| `survive` | 180 | 3,900 | 21,000 | Large matched contrast worth investigating |
| `Paris` | 22 | 25 | 31 | Likely an easy/high-prior completion in this context |

The absolute rank of `the` is spectacular, but its **contrast** is negligible. `survive` has a weaker absolute rank and much stronger experimental evidence.

## Where the bias enters

1. **Output-adjacent readout.** A [logit lens]({{< relref "logit-lens.md" >}}) reuses the model's output head at a point where its coordinates may not yet match the final layer. The Tuned Lens work shows that plain logit-lens distributions can be brittle and systematically miscalibrated across depth; it does not, by itself, establish a universal frequency mechanism for each token ([Belrose et al. 2023](https://arxiv.org/abs/2303.08112)).
2. **Selection over many cells.** [Best-rank]({{< relref "best-rank.md" >}}) keeps the minimum over layers, positions, and sometimes token variants. More searched cells create more opportunities for an accidental extreme.
3. **Prompt and task priors.** Famous facts, syntactic continuations, and words already suggested by the prompt can rank well without carrying the intended latent concept.

## How we guard against it

- Pre-specify probe tokens and tokenization variants.
- Compare paired pressure/control prompts that share surface form wherever possible.
- Score [logit lens]({{< relref "logit-lens.md" >}}), fitted lens, and [random-J]({{< relref "random-j.md" >}}) with the **identical** search rule.
- Check for literal [prompt echo]({{< relref "prompt-echo.md" >}}).
- Report the paired gap and direction count alongside the absolute best rank.

A low absolute rank is a lead. A replicated, control-relative gap is evidence.

See also: [rank]({{< relref "rank.md" >}}), [prompt echo]({{< relref "prompt-echo.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
