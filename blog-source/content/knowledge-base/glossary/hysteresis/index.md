---
title: "Hysteresis"
slug: "hysteresis"
summary: "A system's response can depend on how its controls arrived at their present values."
draft: true
pro_reviewed: false
og_image: "hysteresis-card.png"
og_image_alt: "Two toy write histories return to the same control value, zero, but retain different output bits."
---

**Hysteresis** means that a system's response depends on the history of its controls, so the present control value alone does not determine the response. Turning a control up and then down can follow different response paths. The retained difference is a form of memory.

The ideal, **rate-independent** case depends on the path taken, rather than how quickly it is traversed. Experiments also use “dynamic hysteresis” for loops affected by the rate of driving. A short-lived lag can therefore resemble persistent memory; state the convention and the observation time.

## A two-threshold example

Let \(u_n\) be a dimensionless control at step \(n\), and let \(s_n\) be a stored bit, either 0 or 1, after that step. This toy relay switches to 1 at an upper threshold, switches to 0 at a lower threshold, and otherwise retains its previous bit:

\[
s_n=
\begin{cases}
0, & u_n\le -1,\\
1, & u_n\ge 1,\\
s_{n-1}, & -1<u_n<1.
\end{cases}
\]

The inequalities specify what happens at the thresholds themselves. The starting bit must be supplied if the first control lies between them. Physical circuits can implement the same two-threshold principle; see Texas Instruments' [comparator with hysteresis reference design](https://www.ti.com/lit/ug/tidu020a/tidu020a.pdf).

{{< reference-figure
  src="hysteresis.svg"
  alt="A write at minus two leaves bit zero at a shared control of zero; a write at plus two leaves bit one at that same control."
  caption="Toy relay, not measured model data. In the upper row, control minus two writes bit zero. Returning the control to zero retains zero. In the lower row, control plus two writes bit one; returning to the same control of zero retains one. Repeating zero retains each bit indefinitely under this exact toy rule. The two readouts differ while the current control is identical. This illustration does not establish retention time or noise tolerance in a physical or numerical system."
>}}

Start each trial with either bit. The first control forces the state, so the answer does not depend on that choice:

| Control sequence | Bit after writing | Bit after returning to zero |
|---|---:|---:|
| \(-2,\ 0\) | 0 | 0 |
| \(+2,\ 0\) | 1 | 1 |

Applying zero again changes neither result. Reading at \(-2\) in one trial and \(+2\) in the other would miss the key test: different present controls can produce different outputs without storing history.

## Hysteresis, bistability, and finite evidence

**Bistability** means that two attracting states coexist under the same fixed controls. An attracting state draws nearby states toward it; its [basin of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}) is the set of starting states that approach it. Attracting states can be fixed points or repeating cycles, rather than two constant scalar values. Changing a control can move a trajectory between their basins, allowing the system to retain a history-dependent choice after the control returns.

Coexisting attractors alone do not establish that a proposed write operation reaches both, that a chosen readout distinguishes them, or that the write operation survives noise. Conversely, two distinguishable traces over a finite interval establish only finite history dependence unless longer-lived stability is also supported. A **transient**, the part of a trajectory before its eventual behavior becomes clear, can be long.

For a numerical memory claim, record the write histories, the common controls used during retention and reading, the initial states, the observation duration, the readout rule, and the [arithmetic precision]({{< relref "knowledge-base/glossary/dtype/index.md" >}}). Test whether the distinction survives longer holds and modest perturbations before generalizing beyond the measured interval. None of these tests alone establishes secrecy, indefinite retention, or useful storage capacity.

The schematic and its exact toy arithmetic are generated and checked by [make_figure.py](make_figure.py). Run it with `--verify` to check the saved SVG and PNG.
