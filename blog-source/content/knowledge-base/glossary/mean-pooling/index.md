---
title: "Mean pooling"
slug: "mean-pooling"
summary: "A rule that averages selected vectors to produce one summary vector."
og_image: "mean-pooling-selected-states.png"
og_image_alt: "Two selected token-state vectors are averaged while a padding state is excluded."
draft: false
pro_reviewed: true
---

**Mean pooling** averages selected vectors to produce one summary vector. In a text encoder, it commonly turns several token states into one [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}) for a sentence, passage, or chunk.

If \(h_i\in\mathbb{R}^{d}\) is the \(d\)-coordinate state at token position \(i\), and \(m_i\) is 1 when that position should be included and 0 otherwise, masked mean pooling computes

\[
p=
\frac{\sum_{i=1}^{T}m_i h_i}
{\sum_{i=1}^{T}m_i}.
\]

Here \(\mathbb{R}\) is the set of real numbers, \(T\) is the number of available positions, \(i\) identifies one position, and \(p\) is the pooled vector. The denominator counts included positions. At least one position must be included.

{{< reference-figure
  src="mean-pooling-selected-states.svg"
  alt="Two included token-state vectors are averaged while a much larger padding vector is excluded by its mask value."
  caption="Toy states for `root` and `water` are \((2,0)\) and \((0,2)\), so their coordinate-wise mean is \((1,1)\). A padding state \((100,100)\) is crossed out and has mask value 0, so it contributes to neither the numerator nor denominator. The numbers teach the operation and are not model measurements."
>}}

## Worked example

Suppose the token states are \(h_1=(2,0)\), \(h_2=(0,2)\), and a padding state \(h_3=(100,100)\). With mask \(m=(1,1,0)\),

\[
p=
\frac{1(2,0)+1(0,2)+0(100,100)}{1+1+0}
=(1,1).
\]

```python
states = [(2.0, 0.0), (0.0, 2.0), (100.0, 100.0)]
mask = [1, 1, 0]

count = sum(mask)
if count == 0:
    raise ValueError("mean pooling needs at least one included state")

pooled = tuple(
    sum(use * state[j] for state, use in zip(states, mask)) / count
    for j in range(len(states[0]))
)
assert pooled == (1.0, 1.0)
```

Many retrieval pipelines apply [L2 normalization]({{< relref "knowledge-base/glossary/l2-normalization/index.md" >}}) after pooling. Cosine similarity already ignores vector length; normalization makes a simple dot product between two normalized vectors equal their cosine similarity.

## What happens to word order?

The averaging operation itself is order-invariant: rearranging the same input vectors produces the same mean. Contextual token states can nevertheless differ with word order because a [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) has already mixed position and surrounding context into each state. Mean pooling preserves order and context only insofar as the states being averaged already encode them. It does not add order information on its own.

This distinction matters for late chunking. The model first creates contextual states while reading a longer span, then the application pools the states belonging to each smaller chunk. The pooling rule is still an average; the extra context lives in the input states.

## What mean pooling does not establish

- Every token does not necessarily deserve equal scientific or semantic weight.
- Excluding padding during model attention does not guarantee that later code excludes it during pooling.
- A pooled vector does not reveal which token caused a retrieval match.
- Different layer choices, special-token rules, and masks can produce different vectors from the same text.

See also: [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [attention mask]({{< relref "knowledge-base/glossary/attention-mask/index.md" >}}), [L2 normalization]({{< relref "knowledge-base/glossary/l2-normalization/index.md" >}}), [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}).
