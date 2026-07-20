---
title: "Attention mask"
slug: "attention-mask"
summary: "A model or library rule that marks usable positions or specifies which sequence positions may exchange information."
og_image: "attention-mask-two-meanings.png"
og_image_alt: "A valid-token mask and a pairwise attention mask perform different jobs."
draft: false
pro_reviewed: true
---

An **attention mask** is a model or library rule that controls which sequence positions are usable or which positions may exchange information. The phrase is overloaded: it is used for at least two related but different objects.

1. A one-dimensional **valid-token mask** marks real input positions and padding positions. For example, \([1,1,1,0]\) often means “use the first three positions and ignore the padded fourth position.”
2. A two-dimensional **pairwise attention mask** says whether a position may attend to another position. In a causal language model, a position usually cannot attend to later positions.

The meaning of 0 and 1, the shape, and whether a library expects allowed positions or blocked positions are implementation conventions. Always check the model and library documentation.

{{< reference-figure
  src="attention-mask-two-meanings.svg"
  alt="A one-dimensional mask excludes a padding position, while a separate pairwise grid permits only selected token-to-token attention links."
  caption="The left panel shows a common valid-token convention: three content positions are included and one padding position is excluded. The right panel shows a causal pairwise rule in which each row may use itself and earlier positions. These are different shapes with different jobs. An attention mask is not automatically block-diagonal, and software may encode allowed and blocked cells with opposite numeric conventions."
>}}

## A concrete four-position example

For tokens `roots`, `need`, `water`, and `[PAD]`, a common valid-token mask is

\[
m=(1,1,1,0).
\]

Here \(m_i=1\) includes position \(i\), and \(m_i=0\) excludes it. The mask can keep the padding state out of [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}).

A causal pairwise mask for the three content positions can instead be written as

\[
A=
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}.
\]

In this displayed convention, \(A_{ij}=1\) means position \(i\) may attend to position \(j\). The third position may use all three content positions, while the first may use only itself. Some software uses additive values such as 0 for allowed pairs and a large negative number for blocked pairs, so the matrix above is a concept, not a universal application programming interface.

## Block boundaries are a separate choice

A mask is **block-diagonal** only when positions are divided into groups and cross-group attention is explicitly blocked. A normal padding mask does not create that pattern. When a long document is encoded with ordinary full attention, contextual token states may exchange information across later chunk boundaries. That cross-chunk context is central to late chunking.

## What an attention mask does not establish

- A permitted attention link does not prove that the model used that link strongly.
- An attention weight is not automatically a causal explanation or a probability that text is true.
- Masking padding during attention does not automatically exclude padding from a later pooling operation.
- The name alone does not specify directionality, local windows, sparse patterns, or block boundaries.

See also: [tokenization]({{< relref "knowledge-base/glossary/tokenization/index.md" >}}), [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}), [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}).
