---
title: "Transformer"
slug: "transformer"
summary: "A family of sequence-processing neural networks that combine attention, position-wise transformations, residual connections, normalization, and position information."
pro_reviewed: false
---

A **Transformer** is a family of neural-network architectures, arrangements of learned computations, for processing sequences. In the canonical design, repeated blocks alternate an **attention** sublayer, which lets each position gather information from positions permitted by an attention rule, with a position-wise [feed-forward network (FFN)]({{< relref "feed-forward-network.md" >}}), which transforms each position separately. Residual connections carry a running state through the stack, normalization rescales that state according to a specified rule, and positional information lets the model represent sequence order.

The original Transformer was an encoder-decoder model for translation. The name now covers a family rather than one fixed diagram: encoder-only, decoder-only, encoder-decoder, and other variants can change the attention rule, block order, normalization placement, positional mechanism, and feed-forward implementation ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762v7)).

{{< reference-figure
  src="knowledge-base/glossary/transformer-overview.svg"
  alt="Token vectors with position information travel along a residual path through repeated blocks; attention mixes permitted positions, a feed-forward network transforms each position separately, and a task-specific head reads the final states."
  caption="A canonical Transformer path. Tokens are mapped to vectors with position information before entering repeated blocks. Within a block, attention can combine information from positions allowed by the model's attention rule, while the feed-forward network applies the same learned transformation separately at each position. Both contribute updates to the residual stream. The final head depends on the task. Encoder and decoder arrangements, attention masks, normalization placement, and exact sublayer order vary, so this is a family-level teaching diagram rather than a specification for every Transformer."
>}}

## From tokens to states

A **tokenizer** maps an input such as text into discrete token identifiers. An **embedding** maps each identifier to a vector of learned numbers. A positional mechanism supplies information about order or relative position; it may add position vectors, modify attention calculations, or use another model-specific method.

Those position-aware vectors enter a stack of Transformer blocks. Within a block:

1. **Self-attention**, attention among states in the same sequence, computes input-dependent mixtures of positions allowed by the attention rule.
2. A position-wise FFN, often implemented as a [multilayer perceptron (MLP)]({{< relref "mlp.md" >}}), transforms each position separately.
3. Residual connections add the sublayers' updates to the running [residual stream]({{< relref "residual-stream.md" >}}).
4. Normalization modules rescale, and in some variants recenter, states around those updates. Their placement and exact formula are architecture choices.

A task-specific **output head** reads the final states. For a next-token language model, an [unembedding]({{< relref "unembedding.md" >}}) maps a final state to one score for each vocabulary token. Other Transformers can produce sequence representations, classifications, reconstructed inputs, actions, images, or other outputs.

## Common arrangements

| Arrangement | Information flow | Common use |
| --- | --- | --- |
| Encoder-only | Positions can usually use context on both sides, subject to the model's attention rule | Sequence representations and classification |
| Decoder-only | A **causal mask**, a rule that blocks access to later positions, supports prediction from earlier positions | Autoregressive generation, which produces one new token after another |
| Encoder-decoder | An encoder represents a source sequence; a decoder reads earlier target positions and attends to encoder states | Sequence-to-sequence tasks such as translation |

These are common patterns, not exclusive definitions. A model may use different masks, sparse or local attention, multiple data types, or task-specific block components.

## What Transformer does not specify

- It does not by itself mean a language model, a chatbot, or a model of any particular size.
- It does not determine the tokenizer, context length, parameter count, positional mechanism, normalization placement, FFN variant, attention pattern, or training objective.
- An attention weight is an internal mixing coefficient. It is not automatically a probability that a statement is true or a causal explanation of a model output.
- Parallel processing of known sequence positions during training does not make one-token-at-a-time autoregressive generation fully parallel.

See also: [feed-forward network (FFN)]({{< relref "feed-forward-network.md" >}}), [multilayer perceptron (MLP)]({{< relref "mlp.md" >}}), [residual stream]({{< relref "residual-stream.md" >}}), [unembedding]({{< relref "unembedding.md" >}}).
