---
title: "RMS normalization"
slug: "rms-normalization"
summary: "Rescaling a vector by a stabilized root mean square of its coordinates, then by a learned gain on each coordinate."
draft: false
pro_reviewed: true
og_image: "rms-normalization-card.png"
og_image_alt: "The vector 3, 4 is divided by its root mean square, about 3.536, and then multiplied by learned gains."
---

**RMS normalization** rescales a hidden vector, the list of \(d\) numbers a model carries for one token between its layers, by a stabilized root mean square of its coordinates, then multiplies each coordinate by its own learned gain. For \(h\in\mathbb{R}^d\) and a gain vector \(g\in\mathbb{R}^d\),

\[
r(h)=\sqrt{\frac{1}{d}\sum_{i=1}^{d} h_i^2},
\qquad
y_i=g_i\,\frac{h_i}{\sqrt{r(h)^2+\varepsilon}},
\qquad i=1,\ldots,d.
\]

Here \(r(h)\) is the ordinary root mean square of the coordinates, and the small constant \(\varepsilon>0\) keeps the denominator at least \(\sqrt{\varepsilon}\), including at \(h=0\). The map is therefore smooth at the origin. Unlike a layer normalization, which also subtracts the mean of the coordinates, RMS normalization does not remove the mean, so shifting every coordinate by the same amount changes the result. The Qwen2.5 models studied in the [note on chaos in a language model's feedback loop]({{< relref "posts/2026/10/chaos-bifurcations-language-model-feedback/index.md" >}}) use RMS normalization.

{{< reference-figure
  src="rms-normalization.svg"
  alt="The coordinates 3 and 4 are divided by the root mean square 3.536, giving about 0.849 and 1.131 before any learned gain."
  caption="For h = (3, 4) and epsilon = 0 in this toy arithmetic, the mean of the squares is 12.5 and the root mean square is the square root of 12.5. Dividing the coordinates by that factor shortens this vector; both panels are drawn at the same scale. For epsilon = 10^-6 or 10^-5, the normalized coordinates are unchanged at three decimal places. Learned gains then rescale the coordinates independently, and a negative gain also reverses a coordinate's sign."
>}}

## Worked example

Take \(h=(3,4)\), \(d=2\), \(\varepsilon=0\), and \(g=(1,1)\). The mean of the squares is \((9+16)/2=12.5\), and

\[
r(h)=\sqrt{12.5}\approx 3.5355.
\]

The normalized coordinates are \(3/3.5355\approx 0.8485\) and \(4/3.5355\approx 1.1314\). For a small positive \(\varepsilon\) such as \(10^{-6}\) or \(10^{-5}\), they are unchanged to four decimal places. At \(h=(0,0)\), the denominator is \(\sqrt{\varepsilon}\) and \(y=0\).

```python
def rms_normalize(values, gain, epsilon):
    if len(values) == 0 or len(values) != len(gain):
        raise ValueError("values and gain must be nonempty and the same length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean_square = sum(value * value for value in values) / len(values)
    scale = (mean_square + epsilon) ** 0.5
    return [g * value / scale for g, value in zip(gain, values)]
```

The output is a hidden vector, not a vector of vocabulary scores. Those scores appear only after the [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}).

See also: [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), [unembedding]({{< relref "knowledge-base/glossary/unembedding/index.md" >}}).
