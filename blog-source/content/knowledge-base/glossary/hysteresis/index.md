---
title: "Hysteresis"
slug: "hysteresis"
summary: "A system's response can depend on how its controls arrived at their present values."
draft: false
pro_reviewed: true
og_image: "hysteresis-card.png"
og_image_alt: "Two toy write histories return to the same control value, zero, but retain different output bits."
---

**Hysteresis** means that a system's response depends on the history of its controls, so the present control value alone does not determine the response. Turning a control up and then down can follow different response paths. The retained difference is a form of memory.

A household thermostat is a familiar case. Suppose it turns the heat on at or below 19 °C and off at or above 21 °C. At 20 °C, the heat is on if the room has been warming from 19 °C or below, and off if it has been cooling from 21 °C or above. The current temperature alone does not say which.

The ideal, **rate-independent** case depends on the path taken, not on how quickly it is traversed. The term **dynamic hysteresis** is also used for rate-dependent responses, as in [Bermúdez, Gómez and Venegas’s analysis of hysteresis models](https://arxiv.org/abs/1902.08175). A short-lived lag can therefore resemble persistent memory, so state the convention and the observation time.

## A two-threshold example

Let \(u_n\) be a dimensionless control at step \(n\), and let \(s_n\) be a stored bit, either 0 or 1, after that step. This toy relay switches to 1 at an upper threshold, switches to 0 at a lower threshold, and otherwise retains its previous bit:

\[
s_n=
\begin{cases}
0, & u_n\le -1,\\
1, & u_n\ge 1,\\
s_{n-1}, & -1\lt u_n\lt 1.
\end{cases}
\]

The inequalities specify what happens at the thresholds themselves. The starting bit must be supplied if the first control lies between them. Physical circuits implement the same two-threshold principle; see Texas Instruments' [comparator with hysteresis reference design](https://www.ti.com/lit/ug/tidu020a/tidu020a.pdf).

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

Applying zero again changes neither result. The key test is the read at the shared control of zero. Reading at \(-2\) in one trial and \(+2\) in the other would show only that different controls give different outputs, which needs no memory at all.

## Hysteresis, bistability, and finite evidence

**Bistability** means that two stable attractors coexist under the same fixed controls. An attractor draws nearby trajectories toward it; its [basin of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}) is the set of starting states whose trajectories approach it. The two attractors can be stable fixed points or stable repeating cycles (periodic orbits), not only two constant scalar values. Bistability can support hysteresis when changing a control moves a trajectory from one basin into the other, so the system retains a history-dependent choice after the control returns. Coexistence alone, however, does not establish hysteresis: the proposed write histories must actually reach different attractors and produce distinguishable readouts at identical controls.

Conversely, two distinguishable traces over a finite interval establish only finite history dependence unless longer-lived stability is also supported. A **transient**, the part of a trajectory before its eventual behavior becomes clear, can be long.

For a numerical memory claim, record the write histories, the common controls used during retention and reading, the initial states, the observation duration, the readout rule, and the [arithmetic precision]({{< relref "knowledge-base/glossary/dtype/index.md" >}}). Test whether the distinction survives longer holds and modest perturbations before generalizing beyond the measured interval. None of these tests alone establishes secrecy, indefinite retention, or useful storage capacity.

## A reported example in a language-model loop

The [November note on hiding bits and logic in a language-model feedback loop]({{< relref "posts/2026/11/bits-in-language-model-dynamics-short-read/index.md#memory" >}}) reports an experiment that runs Qwen2.5-0.5B as a [soft-token feedback]({{< relref "knowledge-base/glossary/soft-token-feedback/index.md" >}}) loop, where each step feeds back a probability-weighted average of token representations as the next continuous input vector. Its two controls are the base-10 logarithm of [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}) (\(\log_{10} T\)) and a scalar **steering strength** (\(\alpha\)) that scales a fixed internal activation-steering vector. At one fixed temperature (\(\log_{10} T = -0.1539515\)), **numerical continuation** (tracking a family of repeating solutions as a control parameter varies) of a **period-six orbit** (a sequence of states that repeats every six steps) traces an S-shaped curve with two attracting branches overlapping across a shared interval of steering strength:

{{< reference-figure
  src="qwen-kite-branch.png"
  alt="Numerical continuation of a period-six orbit at a fixed base-10 logarithm of temperature. Two solid attracting segments overlap in a green interval, with dashed nonattracting segments and open circles marking fold turning points."
  caption="**Two possible states at identical controls (reported in the November note).** At base-10 logarithm of temperature −0.1539515, the two attracting branches of the period-six orbit family overlap over steering strengths 0.072907–0.072992 (green). Dashed segments do not attract; open circles mark folds, turning points in steering strength where an attracting branch meets a nonattracting branch. Vertical position is only a display coordinate obtained by projecting the 896-component state vector onto a principal-component direction; it is not a control or a scalar equation of motion."
>}}

At one shared hold-and-read control setting \(H\) inside that green overlap, the loop has two coexisting six-step cycles, labeled \(H_0\) (for stored bit \(0\)) and \(H_1\) (for stored bit \(1\)). To write an intended bit \(b\in\{0,1\}\), the reported protocol moves the controls from \(H\) out to a write endpoint \(E_b\) and back to \(H\). On the way out and on the way back, the controls pass the same intermediate setting \(P_f = H + f\cdot(E_b - H)\) twice. Each group of runs tests all 4 combinations of starting cycle (\(H_0\) or \(H_1\)) and write endpoint (\(E_0\) or \(E_1\)), and the note reports 4 such groups (16 outward-and-return comparisons in total): the midpoint \(f = 0.50\) with 240 updates per stage (4 rows), the midpoint \(f = 0.50\) with 480 updates per stage (4 rows), \(f = 0.25\) with 480 updates per stage (4 rows), and \(f = 0.75\) with 480 updates per stage (4 rows).

Of those 16 reported comparisons, 13 qualified, meaning both visits settled into a primitive period-six cycle (closing within \(10^{-8}\) in Euclidean, or L2, distance after six steps and separated from shorter divisor periods 1, 2, and 3 by more than \(10^{-4}\)). To compare two six-step cycles \(Q\) and \(R\) at \(P_f\) without sensitivity to starting phase, **phase alignment** cyclically shifts \(R\) through all six possible starting phases \(s\in\{0,\ldots,5\}\) and takes the minimum over \(s\) of the maximum over the six steps \(j\in\{0,\ldots,5\}\) of the Euclidean distance \(\|Q_j - R_{(j+s)\bmod 6}\|_2\). In the seven qualified comparisons where the write flipped the stored bit (\(E_0\) starting from \(H_1\), or \(E_1\) starting from \(H_0\)), the return cycle at \(P_f\) differed from the outward cycle by roughly \(10^{-2}\) in that phase-aligned distance. In the six where the write repeated the bit already held, the two visits matched to floating-point rounding. The remaining three (all on the \(E_1\) direction: both \(H_0\) and \(H_1\) at \(f = 0.50\) with 240 updates, and \(H_1\) at \(f = 0.75\) with 480 updates) had at least one visit with no qualified period from 1 to 64 (marked `U`) and stay unresolved:

{{< reference-figure
  src="qwen-hysteresis-loops.png"
  alt="Of sixteen reported comparisons at an identical intermediate control, seven show a Euclidean distance gap of about 0.01 between the outward and return period-six cycles, six show no gap beyond rounding, and three are unresolved."
  caption="Reported Qwen outward-and-return loops from the November note (the underlying raw arrays are not bundled in this glossary entry). Each row follows one trajectory from the shared hold setting H, through the intermediate control \\(P_f\\), to the write endpoint \\(E_b\\), back through \\(P_f\\), and finally to H across the four tested (fraction, dwell) groups. The plotted marker is the phase-aligned Euclidean (L2) cycle distance between the outward and return visits to \\(P_f\\), on a base-10 logarithmic scale. Seven qualified rows sit near \\(10^{-2}\\); these are the bit-flipping writes (\\(E_0\\) from \\(H_1\\) and \\(E_1\\) from \\(H_0\\)). Six sit between about \\(10^{-14}\\) and \\(10^{-11}\\); those writes repeated the bit already held. U marks a visit with no qualified period from 1 to 64 under the closure threshold. Every final readout at H returned the written bit. Each stage lasted a fixed 240 or 480 updates, so this illustrates finite-dwell history dependence rather than a slow-sweep limit, long-term retention, or a proven fold mechanism."
>}}
