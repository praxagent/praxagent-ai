---
title: "Transformer"
slug: "transformer"
summary: "A family of sequence-processing neural networks that combine attention, position-wise transformations, residual connections, normalization, and position information."
og_image: "transformer-overview.png"
og_image_alt: "A Transformer block carries a residual stream through attention and a position-wise feed-forward network before a task-specific head reads the final states."
draft: false
pro_reviewed: true
---

A **Transformer** is a family of neural-network architectures, arrangements of learned computations, for processing sequences. In common designs, repeated blocks contain one or more **attention** sublayers, which let each position gather information from positions permitted by an attention rule, and a position-wise [feed-forward network (FFN)]({{< relref "knowledge-base/glossary/feed-forward-network/index.md" >}}), which transforms each position separately. Residual connections carry a running state through the stack, normalization rescales that state according to a specified rule, and positional information lets the model represent sequence order.

The original Transformer used an **encoder**, a stack that represented a source (input) sequence, and a **decoder**, a stack that generated a target (output) sequence using earlier target positions and encoder states. The name now covers a family rather than one fixed diagram: encoder-only models typically let positions gather context from both earlier and later positions; decoder-only models typically block attention to later positions; and encoder-decoder models use both connected stacks. Other variants can change the attention rule, block order, normalization placement, positional mechanism, and feed-forward implementation ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762v7)).

{{< reference-figure
  src="transformer-overview.svg"
  alt="A residual path receives an attention update followed by a position-wise feed-forward update before a task-specific head reads the final states."
  caption="A common Transformer block pattern: attention combines information from allowed positions, then a position-wise feed-forward network transforms each state; both add updates to the residual stream. Position handling, normalization, masks, sublayer order, and encoder-decoder structure vary between architectures."
>}}

## From tokens to states

For a typical text Transformer, a **tokenizer** maps text to a sequence of discrete units called **tokens** and represents each token with an identifier from a **token vocabulary**, the set of token types available to the model. An **embedding** maps each identifier to a learned vector. Transformers for other input types may instead map their input elements directly to vectors. A positional mechanism supplies information about order or relative position; it may add position vectors, modify attention calculations, or use another model-specific method.

Those position-aware vectors enter a stack of Transformer blocks. Within a block:

1. **Self-attention**, attention among states in the same sequence, combines information from positions allowed by the attention rule in an input-dependent way.
2. Decoder blocks in encoder-decoder models commonly also use **cross-attention**, in which decoder positions gather information from encoder states.
3. A position-wise FFN, often implemented as a [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}), transforms each position separately.
4. Residual connections add the sublayers' updates to the running [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}).
5. Normalization modules rescale, and in some variants recenter, states around those updates. Their placement and exact formula are architecture choices.

A task-specific **output head** reads the final states. For a next-token language model, an [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}) maps a final state to one score for each vocabulary token. Other Transformers can produce sequence representations, classifications, reconstructed inputs, actions, images, or other outputs.

## Common arrangements

| Arrangement | Information flow | Common use |
| --- | --- | --- |
| Encoder-only | Positions can usually use context on both sides, subject to the model's attention rule | Sequence representations and classification |
| Decoder-only | A **causal mask**, a rule that blocks access to later positions, supports prediction from earlier positions | Autoregressive generation, which produces one new token after another |
| Encoder-decoder | An encoder represents a source sequence; a decoder reads earlier target positions and attends to encoder states | Tasks that map an input sequence to an output sequence, such as translation |

These are common patterns, not exclusive definitions. A model may use attention masks that permit or block specified pairs of positions, including sparse patterns that permit only selected pairs or local patterns centered on nearby positions. It may also combine multiple data types or use task-specific block components.

## What Transformer does not specify

- It does not by itself mean a language model, a chatbot, or a model of any particular size.
- It does not determine the tokenizer; context length, meaning the number of sequence positions the model is configured to process together; parameter count, meaning the number of learned scalar parameters; positional mechanism; normalization placement; FFN variant; attention pattern; or training objective, meaning the criterion optimized during training.
- An attention weight is an internal mixing coefficient. It is not automatically a probability that a statement is true or a causal explanation of a model output.
- Parallel processing of known sequence positions during training does not make one-token-at-a-time autoregressive generation fully parallel.

See also: [feed-forward network (FFN)]({{< relref "knowledge-base/glossary/feed-forward-network/index.md" >}}), [multilayer perceptron (MLP)]({{< relref "knowledge-base/glossary/mlp/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}).
