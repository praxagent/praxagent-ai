---
title: "Causal attention"
slug: "causal-attention"
summary: "A rule that lets each token use itself and earlier tokens, and not later ones."
draft: false
pro_reviewed: true
og_image: "causal-attention-card.png"
og_image_alt: "Each prefix token reads itself and earlier prefix tokens. The last position x reads itself and the prefix, and the prefix cannot read x."
---

**Causal attention** lets each position in a sequence of tokens use information from itself and from earlier positions, and not from later ones. A token is one unit of model input, often a word or part of a word. Decoder-only language models, [transformers]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) that predict each next token from the tokens before it, use this rule so that every next-token prediction depends only on tokens already available.

At each attention layer the model forms three vectors at every position: a **query**, which scores the positions it may read, a **key**, which those scores are computed against, and a **value**, which is what a read returns. The causal rule limits which positions a query may read. Every operation in a standard decoder that mixes positions follows the same ordering, so the representation at a position, and the keys and values computed from it, can depend only on that position and earlier ones. The vectors at a prefix position, in the early part of the sequence, therefore never depend on a token that comes after it.

{{< reference-figure
  src="causal-attention.svg"
  alt="Token A reads itself. Token B reads A and itself. The last position x reads A, B, and itself. Arrows show allowed information flow from A to B and from A and B to x. A box marks flow from x into the keys or values of A or B as blocked."
  caption="For two prefix tokens and one later position, arrows run from each source of information to the later position that may read it. No arrow runs from x back into the prefix. Changing x can therefore change the vectors at x but cannot change the keys or values at A or B."
>}}

## Worked example

Label the positions \(A\), \(B\), and \(x\), with \(x\) last. The allowed reads are:

| Position | May read | Its keys and values may depend on, at most |
| --- | --- | --- |
| \(A\) | \(A\) | \(A\) only |
| \(B\) | \(A, B\) | \(A\) and \(B\) |
| \(x\) | \(A, B, x\) | \(A\), \(B\), and \(x\) |

The last column is an upper bound: in the first layer, for example, keys and values are computed from the input at their own position only. Replacing the input vector at the last position can change the query, key, and value at \(x\), but it cannot change any vector at \(A\) or \(B\). That one-way dependency lets an implementation compute the fixed prefix's keys and values once and reuse them while \(x\) varies, for example when \(x\) is a **soft token**, a probability-weighted average of token embeddings fed in place of a single token.

This rule is not the same thing as **padding**, placeholder positions added so that sequences of different lengths can share one array. An [attention mask]({{< relref "knowledge-base/glossary/attention-mask/index.md" >}}) is the implementation's grid of allowed query and source pairs; it can encode the causal rule, hide padding positions, or do both. Which job a mask does is a property of the implementation.

See also: [attention mask]({{< relref "knowledge-base/glossary/attention-mask/index.md" >}}), [transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}).
