---
title: "Brouwer's fixed-point theorem"
slug: "brouwer-fixed-point"
summary: "A continuous map from a nonempty compact convex set into itself has at least one fixed point."
draft: false
pro_reviewed: true
og_image: "brouwer-fixed-point-card.png"
og_image_alt: "On the unit interval, the map 0.3 plus 0.4 x stays inside the interval and meets the diagonal at one half."
---

**Brouwer's fixed-point theorem** says: if \(K\) is a nonempty, compact, convex subset of \(\mathbb{R}^n\), the space of vectors with \(n\) real coordinates, and \(F\) is a continuous function from \(K\) into \(K\), then some \(x \in K\) satisfies \(F(x)=x\). Such an \(x\) is a **fixed point**: one application of the rule leaves it unchanged. Continuous means that small enough changes of the input give arbitrarily small changes of the output. Compact means closed and bounded. Convex means that the segment between any two points of the set stays in the set.

For the soft-token map, let \(K\) be the [convex hull]({{< relref "knowledge-base/glossary/convex-hull/index.md" >}}) of the token embeddings, the vectors that represent the vocabulary's tokens. The hull of finitely many points is compact and convex. The map feeds a vector \(x \in K\) to a model at one fixed input position after a fixed prompt, and returns the probability-weighted average of the embeddings under the model's next-token probabilities, which again lies in \(K\). If that map is continuous, the theorem guarantees an \(x \in K\) with \(F(x)=x\): a soft token that the next step reproduces exactly. This is an equality of vectors; \(x\) need not be the embedding of any one token, and the theorem says nothing about sampled text.

{{< reference-figure
  src="brouwer-fixed-point.svg"
  alt="The map 0.3 plus 0.4 x sends the unit interval into a shorter interval inside it, and crosses the diagonal at one half."
  caption="The horizontal axis is the input on the closed unit interval. The curve is F(x) = 0.3 + 0.4 x, which sends that interval into [0.3, 0.7]. The dashed diagonal is y = x; where the graph y = F(x) meets it, F(x) = x, and here they meet at x = 1/2. This is a one-dimensional illustration of the hypothesis the theorem uses, not a soft-token orbit."
>}}

## Worked example

On the interval \([0,1]\), let \(F(x)=0.3+0.4x\). Then \(F(0)=0.3\) and \(F(1)=0.7\), so the image lies inside \([0,1]\). Solving \(x=0.3+0.4x\) gives \(0.6x=0.3\), hence \(x=1/2\). And \(F(1/2)=1/2\).

The theorem does not say that the fixed point is unique, or that applying \(F\) again and again from other starting points leads to it. For example, \(F(x)=1-x\) on \([0,1]\) has the fixed point \(1/2\), but every other starting point alternates between \(x\) and \(1-x\) forever; in higher dimensions, repeated application can also wander on a chaotic set. A fixed point of a soft-token map is also not the same thing as a text that repeats when tokens are sampled.

See also: [convex hull]({{< relref "knowledge-base/glossary/convex-hull/index.md" >}}), [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}).
