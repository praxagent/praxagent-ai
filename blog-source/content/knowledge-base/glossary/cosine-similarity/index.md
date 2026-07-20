---
title: "Cosine similarity"
slug: "cosine-similarity"
summary: "A comparison of two nonzero vectors based on the angle between them rather than their lengths."
og_image: "cosine-similarity-angles.png"
og_image_alt: "Vector pairs at zero, ninety, and one hundred eighty degrees show cosine similarities of one, zero, and negative one."
draft: false
pro_reviewed: true
---

**Cosine similarity** compares the directions of two nonzero vectors. It is the cosine of the angle between them: vectors pointing in the same direction score 1, perpendicular vectors score 0, and vectors pointing in opposite directions score (-1).

For nonzero vectors \(a,b\in\mathbb{R}^{d}\),

\[
\operatorname{cosine}(a,b)
=
\frac{a\cdot b}{\lVert a\rVert_2\lVert b\rVert_2}
=
\frac{\sum_{j=1}^{d}a_jb_j}
{\sqrt{\sum_{j=1}^{d}a_j^2}\sqrt{\sum_{j=1}^{d}b_j^2}}.
\]

Here \(\mathbb{R}\) is the set of real numbers, \(d\) is the number of coordinates, \(j\) identifies one coordinate, \(a\cdot b\) is the dot product obtained by multiplying matching coordinates and summing them, and \(\lVert\cdot\rVert_2\) is Euclidean length.

{{< reference-figure
  src="cosine-similarity-angles.svg"
  alt="Vector pairs with zero, ninety, and one hundred eighty degree angles have cosine similarities one, zero, and negative one."
  caption="Cosine similarity depends on angle: the same direction gives 1, a right angle gives 0, and the opposite direction gives -1. Doubling a nonzero vector's length does not change its cosine similarity to another vector. These two-dimensional arrows teach the geometry and do not claim that real embedding spaces contain only two coordinates."
>}}

## Worked example

Let \(a=(1,0)\) and \(b=(1,1)\). Their dot product is 1, their lengths are 1 and \(\sqrt{2}\), and therefore

\[
\operatorname{cosine}(a,b)
=
\frac{1}{\sqrt{2}}
\approx 0.707.
\]

If both inputs have already received [L2 normalization]({{< relref "knowledge-base/glossary/l2-normalization/index.md" >}}), each has length 1 and cosine similarity reduces to their dot product.

```python
import math

def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(x * y for x, y in zip(a, b)) / (length_a * length_b)

assert math.isclose(cosine_similarity((1, 0), (1, 1)), 1 / math.sqrt(2))
```

## Similarity, distance, and ranking

Some systems define **cosine distance** as \(1-\operatorname{cosine}(a,b)\). Libraries can use other names or conventions, so record the exact function. A retrieval system can rank document [embeddings]({{< relref "knowledge-base/glossary/embedding/index.md" >}}) by decreasing cosine similarity to a query embedding. Ranking quality then needs relevance judgments and [retrieval metrics]({{< relref "knowledge-base/glossary/retrieval-ranking-metrics/index.md" >}}); a large similarity score alone is not an evaluation.

## What cosine similarity does not establish

- It is not a probability, confidence level, or percentage match.
- A score near 1 does not prove factual agreement, biological equivalence, or causation.
- It is undefined when either vector is zero.
- Its meaning depends on the encoder, layer, pooling rule, and data distribution that produced the vectors.

See also: [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}), [L2 normalization]({{< relref "knowledge-base/glossary/l2-normalization/index.md" >}}), [retrieval ranking metrics]({{< relref "knowledge-base/glossary/retrieval-ranking-metrics/index.md" >}}).
