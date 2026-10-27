---
title: "Convex hull"
slug: "convex-hull"
summary: "Every weighted average of a set of points whose weights are nonnegative and sum to one."
draft: false
pro_reviewed: true
og_image: "convex-hull-card.png"
og_image_alt: "Three embedding points form a triangle. The mixture with weights one half, one quarter, and one quarter lands inside it."
---

A **convex combination** of vectors \(v_1,\ldots,v_n\) is a weighted average

\[
\sum_{i=1}^{n} p_i v_i,
\qquad p_i\ge 0,
\qquad \sum_{i=1}^{n} p_i=1.
\]

The **convex hull** of those vectors is the set of all such combinations. For a finite set of points in a plane it is the filled convex polygon enclosed by a rubber band stretched around them, boundary included. If the points all lie on one line, the hull is a line segment, and a single point is its own hull. In higher dimension the hull is the corresponding convex solid, which can still be flat: three points in a space of any dimension span only a triangle.

For each token \(t\), let \(e_t\) be its input embedding, the vector the model looks up when it reads that token, and let \(p_t\) be the model's next-token probability for it. A soft token is \(\sum_t p_t e_t\). Those weights are nonnegative and sum to 1, so the soft token lies in the convex hull \(K\) of the embeddings. That hull is closed and bounded, because it is the convex hull of finitely many points. The soft-token map feeds its input vector in the place where a token's embedding would go, so any vector of the embedding space, and in particular any point of \(K\), is a valid input. It therefore sends every input into \(K\), and in particular it sends \(K\) into itself.

{{< reference-figure
  src="convex-hull.svg"
  alt="Three points form a triangle. Weights one half, one quarter, and one quarter land at (1, 1) inside the triangle. The point (3, 3) lies outside."
  caption="The vertices are toy embeddings (0, 0), (4, 0), and (0, 4). The interior point is the convex combination with weights 1/2, 1/4, and 1/4. The point (3, 3) is outside the triangle, so it is not a convex combination of these three embeddings. The embeddings of a language model have hundreds or thousands of coordinates; the triangle is the same rule drawn for three points in two."
>}}

## Worked example

Let \(e_1=(0,0)\), \(e_2=(4,0)\), \(e_3=(0,4)\), and let the weights be \(1/2\), \(1/4\), \(1/4\). Then

\[
\frac12 e_1+\frac14 e_2+\frac14 e_3=(1,1),
\]

which lies inside the triangle. The point \((3,3)\) does not. Any weights that reproduce it would have to sum to 1 and stay nonnegative, and no such weights exist: the first coordinate would require weight \(3/4\) on \(e_2\), the second weight \(3/4\) on \(e_3\), and those two weights already sum to more than 1.

```python
def convex_combination(points, weights):
    if len(points) == 0 or len(points) != len(weights):
        raise ValueError("points and weights must be nonempty and the same length")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be nonnegative")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    width = len(points[0])
    if any(len(point) != width for point in points):
        raise ValueError("all points must have the same dimension")
    return [
        sum(weight * point[i] for weight, point in zip(weights, points))
        for i in range(width)
    ]
```

Membership in the convex hull is not a probability that a token is correct. The weights \(p_t\) are the model's next-token probabilities; the combination they form is a vector, not a token.

See also: [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [Brouwer's fixed-point theorem]({{< relref "knowledge-base/glossary/brouwer-fixed-point/index.md" >}}).
