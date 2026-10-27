---
title: "Rotary embedding"
slug: "rotary-embedding"
summary: "A transformer position method that rotates pairs of attention-query and attention-key coordinates by position-dependent angles, so that their dot product depends on relative position."
draft: false
pro_reviewed: true
og_image: "rotary-embedding-card.png"
og_image_alt: "At position 0 a coordinate pair is unchanged. At the next position, with a right angle of rotation, the pair (1, 0) becomes (0, 1)."
---

A **rotary position embedding**, or RoPE, encodes position in a [transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) by rotating pairs of coordinates in its attention query and key vectors, the vectors that attention compares to decide how much one position reads from another. At position \(m\), counted from zero, one pair \((x_0,x_1)\) is rotated through the angle \(m\theta\), where \(\theta\) is that pair's angle step in radians per position:

\[
\begin{bmatrix} y_0 \\ y_1 \end{bmatrix} =
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix} x_0 \\ x_1 \end{bmatrix}.
\]

Different pairs use different angle steps \(\theta\). Write \(R(m\theta)\) for the rotation matrix above. Because queries and keys are rotated with the same angle steps, \(R(m\theta)^{\top}R(n\theta)=R((n-m)\theta)\): in the dot product of a query at position \(m\) with a key at position \(n\), the rotations combine into one that depends only on the offset \(n-m\). That is how RoPE makes attention scores sensitive to relative position.

For a fixed position \(m\), each pair's rotation matrix is constant, so changing the input pair changes the output linearly; moving to another position changes the angle. In the soft-token loop of the [note on chaos in a language model's feedback loop]({{< relref "posts/2026/10/chaos-bifurcations-language-model-feedback/index.md" >}}), the soft token always enters at the same position, so its rotations never change from step to step; only the vectors being rotated do.

{{< reference-figure
  src="rotary-embedding.svg"
  alt="Position 0 leaves the pair (1, 0) on the horizontal axis. One position later, a right-angle rotation puts that pair, at the same length, on the vertical axis."
  caption="The drawing uses one pair and one angle step, theta = pi/2 radians per position, so each position advances the angle by a right angle. Position 0 is the identity. Position 1 sends (1, 0) to (0, 1), and the length is unchanged. Implementations use many pairs and a range of angle steps, most of them far smaller than pi/2. The figure illustrates the rotation rule, not a vector taken from a model."
>}}

## Worked example

Take \(\theta=\pi/2\) and the pair \((1,0)\).

| Position \(m\) | Angle | Result |
| ---: | --- | --- |
| 0 | 0 | \((1, 0)\) |
| 1 | \(\pi/2\) | \((0, 1)\) |
| 2 | \(\pi\) | \((-1, 0)\) |

```python
from math import cos, sin

def rotate_pair(x0, x1, position, theta):
    angle = position * theta
    return (
        x0 * cos(angle) - x1 * sin(angle),
        x0 * sin(angle) + x1 * cos(angle),
    )
```

The rotation does not produce vocabulary scores, and it does not by itself decide which token comes next. It modifies attention queries and keys so that position can affect the attention scores before attention mixes the value vectors.

See also: [causal attention]({{< relref "knowledge-base/glossary/causal-attention/index.md" >}}), [transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}).
