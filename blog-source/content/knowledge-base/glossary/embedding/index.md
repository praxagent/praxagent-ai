---
title: "Embedding"
slug: "embedding"
summary: "A vector of numbers used to represent an item so a model or retrieval system can compare and transform it mathematically."
og_image: "embedding-items-to-vectors.png"
og_image_alt: "Text items become coordinate vectors and points in a toy embedding space."
draft: false
pro_reviewed: true
---

An **embedding** is a vector of numbers used to represent an item. A **vector** is an ordered list of numbers, such as \((0.2,-0.7,1.1)\). Each number is a **coordinate**, and the number of coordinates is the vector's **dimension**. Embeddings let software compare or transform text, images, proteins, and other objects with mathematical operations.

The coordinates usually do not have simple names such as “color” or “topic.” Their meaning comes from how the model was trained and how the vectors are used.

{{< reference-figure
  src="embedding-items-to-vectors.svg"
  alt="Three text items are assigned different two-coordinate vectors and placed as points in a toy embedding space."
  caption="Toy two-dimensional embeddings make the mapping visible on a page. Each item becomes an ordered list of coordinates and therefore a point. Nearby points are similar only according to this particular representation and comparison rule. Real embeddings commonly have hundreds or thousands of coordinates, and the diagram does not claim that either visible axis has a biological or linguistic meaning."
>}}

## Token embeddings and contextual states

For a text model, a tokenizer first maps text to [tokens]({{< relref "knowledge-base/glossary/tokenization/index.md" >}}) and token identifiers. A learned lookup table can map token identifier \(k\) to a starting vector \(e_k\). If the vocabulary contains \(V\) token types and each vector has \(d\) coordinates, the lookup table is a matrix \(E\in\mathbb{R}^{V\times d}\). Here \(\mathbb{R}\) is the set of real numbers, \(V\) is vocabulary size, \(d\) is embedding dimension, and row \(k\) of \(E\) is \(e_k\).

A [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) then updates those starting vectors using surrounding context. The final vector for `bank` in “river bank” can therefore differ from the vector for `bank` in “bank account.” These updated vectors are often called **contextual token states**. A pooled passage embedding can summarize several contextual states into one vector.

## A small practical example

Suppose an encoder returns these illustrative two-coordinate embeddings:

| item | embedding |
| --- | --- |
| `root pressure` | \((0.8,0.3)\) |
| `soil compaction` | \((0.7,0.4)\) |
| `protein folding` | \((-0.5,0.6)\) |

A system could compare their directions with [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}). The resulting similarity belongs to this encoder and preprocessing pipeline. It is not a universal distance between the underlying concepts.

## What an embedding does not establish

- Nearby vectors are not automatically synonyms, biologically equivalent samples, or factually consistent statements.
- A coordinate is not automatically an interpretable feature.
- A two-dimensional plot may be a projection of much larger vectors and can hide structure.
- Comparisons are meaningful only when the vectors come from compatible models, layers, pooling rules, and normalization conventions.

See also: [tokenization]({{< relref "knowledge-base/glossary/tokenization/index.md" >}}), [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}), [L2 normalization]({{< relref "knowledge-base/glossary/l2-normalization/index.md" >}}), [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}), [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}).
