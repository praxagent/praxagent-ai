---
title: "Bits in a language model’s dynamics"
slug: "bits-in-language-model-dynamics"
date: 2026-11-01
author: Timothy Jones
author_id: "timothy-jones"
draft: true
pro_reviewed: false
toc: true
tags: ["AI", "LLM", "dynamical-systems", "memory", "computation"]
summary: "A small language model’s continuous feedback loop carried complete requests and program-computed replies. A second model retained writable bits at common controls and supported finite stateful AND and OR chains."
description: "Two public numerical demonstrations: useful data carried through Pythia feedback responses, and history-dependent memory and finite logic chains in Qwen."
og_image: "og-card.png"
og_image_alt: "Bits in a language model’s dynamics: Pythia carries a complete measurement request and calculated reply; Qwen retains a state at common controls. Public numerical demonstrations, with no concealment claim."
lead: |
  A language model can carry a message through something other than the words it
  generates. Run it as a continuous feedback loop, and its numerical behavior
  becomes an alphabet: choose the controls, let the model evolve, and measure a
  zero or a one. We used that route to send a complete measurement request and
  return an answer calculated from the recovered data. A second model adds
  memory: after the controls return to the same setting, its state remembers
  which value was written.
key_result: |
  Pythia-160m delivered **470 application bytes across two complete request/reply
  exchanges**, with the answer changing correctly when one reading changed.
  The test reused eight known numerical addresses and took about **150.5 minutes**.
  Separately, Qwen2.5-0.5B retained writable state at common controls and passed
  finite stateful AND and OR panels. These are public, recorded demonstrations
  of a numerical side channel and memory primitive.
---

{{< panel "info" >}}
**AI-use disclosure.** Generative-AI tools helped implement, audit, execute,
interpret, visualize, review, and draft this study. The author selected the
research question and authorized the compute. Human inspection of the final
artifacts and editorial approval remain pending in this draft. This is an independent,
non-peer-reviewed Research Note. Verify numbers against the released receipts
before relying on them.
{{< /panel >}}

*Editorial draft. Human review of the artifacts, disclosure attestation, and final text is pending.*

{{< panel "info" >}}
**Abstract.** Continuous feedback turns a fixed language model into a numerical
system whose behavior depends on its controls and current state. We tested
whether that behavior can carry data and support writable memory.

**(1)** An adapted Pythia test delivered 470 bytes through four numerical
receivers. Changing one measurement changed the programmed reply correctly;
offline substitutions of saved readouts were rejected.

**(2)** A separate spatial sample qualified 48 addresses. Each of two fresh
receivers classified 56 of 64 candidates correctly, erased five, and resolved
three incorrectly; only complete passing pairs entered the bank.

**(3)** Exploratory Qwen tests wrote and read carried state at common controls.
All eight tested input triples passed for each of two-gate AND and OR chains.

**Takeaway.** These finite, audited numerical observations establish working
instances of transport and stateful operations. They establish neither secrecy
nor general reliability.
{{< /panel >}}

## From predicting words to carrying bits

An ordinary chatbot appends a chosen token to a growing conversation. Here the
prompt stays fixed. One input slot receives a weighted blend of the model’s
next-token representations, and the model repeatedly predicts from that blend.
No token is randomly sampled. This [soft-token feedback]({{< relref "knowledge-base/glossary/soft-token-feedback/index.md" >}})
makes the model a deterministic dynamical system.

Two controls shape its behavior: [temperature]({{< relref "knowledge-base/glossary/temperature/index.md" >}}) changes the concentration of the
token probabilities, and steering changes an internal activation. Some settings
produce recurring states; others produce irregular motion. The earlier
[feedback study](https://arxiv.org/abs/2610.31692) mapped that structure.
Here the question is operational: can we use it to carry information, keep a
bit, and act on the bit already present? A **numerical address** is a pair of
steering and temperature settings. We call this a **side channel** because it
uses a numerical response interface alongside the model’s ordinary text output.

![Two numerical routes from a message to recovered bytes. Pythia resets the state for each chosen coordinate and classifies its response. Qwen carries the state through a write excursion, restores common hold controls, and reads which cycle remains.](two-routes.svg)

<p class="figure-note"><strong>Two routes, different jobs.</strong> In Pythia the sender chooses a parameter address for each symbol. In Qwen the sender writes the evolving state and returns to a shared hold setting. In both, explicit English instructions cross between endpoints; local model trajectories supply the measured bits. This is a conceptual diagram, not a measured trace. <a href="fig-two-routes.receipt.json">Figure receipt</a>; <a href="build_figures.py">generator</a>.</p>

**Prior work.** Attractor-based memory has a classical precedent in
[Hopfield’s networks](#ref-hopfield), and using nonlinear dynamics for logical
operations predates this study, including [Sinha and Ditto’s work](#ref-sinha).
[Jones’s feedback preprint](#ref-jones) supplies the immediate model and
parameter-map context. This note does not invent those ideas.

**Contribution.** The additions are an executed request/reply path through a
small Transformer’s responses, a separate address-qualification experiment with
all outcomes counted, and bounded stateful memory and logic tests in another
Transformer feedback loop.

**Not claimed.** The instructions openly expose their controls. There is no
concealment demonstration or autonomous conversation, no proof of a reliable
general-purpose memory, and no demonstrated advantage over sending digital
bytes directly.

## A complete request crosses the numerical channel

The sender turns a message into UTF-8, a standard encoding of text as bytes, then into bits. For each bit it chooses
one of the previously tested Pythia coordinates associated with that response.
The receiver gets control instructions, without the sender’s labeled address
bank, and runs the model to recover the bit.

These are the first two symbol instructions in the measured request, quoted
verbatim and selected because they begin the message:

```text
For byte 1, sample 1, use steering -0.0019569471624266144 and log-ten temperature 0.20264187866927602.
For byte 1, sample 2, use steering 0.06800391389432486 and log-ten temperature -0.05675146771037179.
```

Each instruction resets the state, allows 32 warm-up updates, then observes
128 updates. The receiver measures recurrence and whether a small perturbation
grows or shrinks. A recurrent, contracting response means zero; a nonrecurring,
expanding response means one. An ambiguous response is an **erasure**: the decoder withholds the bit. The growth
measurement is related to a [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}),
but decoding uses a finite observation window and does not require proof of
perpetual chaos.

The opening byte was recovered as `01010011`, the letter `S`. Continuing recovered
this complete request. The latency readings are synthetic test data, not
measurements from a production service:

```text
Summarize these latency samples in ms: 15,19,27,23,83,25,18,57,27,11,28,20,14,74,25,28.
Report count, min, max, sum, mean, and indices above 40 ms.
```

A programmed application read those recovered bytes and calculated its reply.
A fresh numerical receiver then reconstructed the reply through the same
channel:

```text
Count 16; min 11 ms; max 83 ms; sum 494 ms; mean 30.88 ms. Indices above 40 ms: 5,8,14.
```

A fixed reply could imitate success. The paired test therefore changed the
fifth reading from `83` to `23` and repeated the entire exchange:

```text
Count 16; min 11 ms; max 74 ms; sum 434 ms; mean 27.12 ms. Indices above 40 ms: 8,14.
```

The maximum, sum, mean, and threshold list changed correctly. An independent
Structured Query Language (SQL) calculation checked the arithmetic; the audit checked that the application
used the decoded request. The model carried the data; program code performed
the arithmetic. The displayed means use the application’s rounding to two
decimal places, with exact halfway cases rounded to the even final digit.

Together, the two requests and replies delivered **470 bytes, or 3,760 bits**,
using **38,541 model forward calls** and **9,028.3 seconds**, about 150.5 minutes.
The 9,028.3 seconds cover the complete paired test. That is about 0.052
delivered application bytes per second. Sixteen separately evolved
symbols shared a forward pass, with separate numerical states; every symbol
was evaluated, including repeated addresses.

![The four delivered Pythia messages: a 148-byte request and 88-byte reply, followed by a changed 148-byte request and 86-byte reply. The chart shows message sizes and measured receiver times separately.](transfer-results.png)

<p class="figure-note"><strong>Every message, both ways.</strong> These four numerical receivers delivered 470 application bytes in total. Per-message times include model loading, readiness, and decoding, not just steady-state model work. The paired exchanges reuse eight known addresses; their payload bits are not independent reliability trials. See the <a href="receipts/evidence-summary.json">evidence receipt</a>, <a href="provenance.json">provenance</a>, and <a href="build_figures.py">generator</a> (`--verify`).</p>

These are complete transfers from one paired family, reusing only eight known
coordinates. The thousands of carried bits are not thousands of independent
reliability trials. The wire used 387,433 English instruction bytes,
about 824 times the application payload.

The failure policy also mattered. Bytes were assembled from observed bits
before checking length and checksum. Offline tests altered the saved readout
at the first symbol: they substituted an opposite response, inserted an erasure,
or flipped a bit. Each failed delivery was
rejected. Earlier, shorter observation windows also contained erasures; the
full declared window was needed for these completed messages.

## Can both symbols live in the same neighborhood?

A handful of successful addresses leaves a narrow address bank. A separate
experiment asked whether small patches of the existing Pythia map could each
supply both a zero and a one.

![Pythia parameter maps comparing the eight known addresses used for message transfer with a separately tested bank of locally paired addresses. Each retained patch supplies both labels; excluded pairs remain visible.](pythia-spatial-map.png)

<p class="figure-note"><strong>Address diversity is a separate result from message delivery.</strong> The background is the earlier scan, not a fresh reliability census. The left panel locates the eight addresses used for transport. The right shows the new paired sample: both labels had to reproduce in both fresh receivers for a pair to qualify. Excluded pairs are shown even when only one member failed. See the <a href="receipts/evidence-summary.json">evidence receipt</a>, <a href="provenance.json">provenance</a>, and <a href="build_figures.py">generator</a> (`--verify`).</p>

We drew 32 eligible patches once, then one candidate of each class per patch,
before running the fresh receivers. Each receiver tested all 64 coordinates:

| Fresh receiver | Correct label | Erasure | Wrong resolved label |
|---|---:|---:|---:|
| A | 56 | 5 | 3 |
| B | 56 | 5 | 3 |

Only pairs whose labels reproduced in both receivers entered the bank:
**24 pairs, or 48 addresses**. No replacement was drawn for a failure. The
receivers checked the same candidates, not independent population samples.
This bank was **not** used in the long exchange above, and a complete message
through it has not yet been tested. Spatial mixing supplies no secrecy: the
public response map still determines the labels.

## Qwen adds a bit that survives returning the controls

Pythia resets for every symbol. Qwen2.5-0.5B-Instruct supports a more demanding
operation: keep the evolving state, change the controls briefly, then return
them to exactly the same hold setting.

At that hold setting, two distinguishable repeating states provide the labels
zero and one. A write excursion selects the desired state. The controls at
readout are identical, so the bit resides in the state’s history. This is the
operational memory idea behind [hysteresis]({{< relref "knowledge-base/glossary/hysteresis/index.md" >}}).
The candidate attractors have different [basins of attraction]({{< relref "knowledge-base/glossary/basin-of-attraction/index.md" >}}),
the sets of states that approach them.

![The previously mapped Qwen period-six shrimp window, with a large gray region and smaller branching periodic structures in an irregular background.](qwen-shrimp.png)

<p class="figure-note"><strong>The landscape that motivated the memory test.</strong> This image is reproduced unchanged from the <a href="/blog/posts/2026/10/chaos-bifurcations-language-model-feedback/#period-6-shrimp">earlier feedback study</a>. It locates a shrimp-shaped periodic window; it is not a new storage measurement (<a href="fig-qwen-shrimp.receipt.json">image provenance</a>). The write tests target a candidate region where two repeating states coexist; each repeats on a six-step cycle.</p>

That geometry guided the write settings. The evidence establishes bounded
history-dependent writes; it does not yet show that a finite pulse switches
the state because an attractor disappears.

In a complete Qwen exchange, the generated text `Bob` and generated reply `A`
survived numerical encoding and decoding. Each byte began from two reference starts, one chosen position on each
reference cycle. Both trajectories recovered all 32 payload bits. Each bit used
120 write updates followed by 240 common-hold updates. The state carried across the writes within each
byte; each new byte restarted the reference pair. The short outputs missed the
request for exactly two letters and were retained unedited. The decoded request
entered the reply-generation prompt. Transporting these 4 application bytes
used 11,716 model calls and took 116.4 minutes. This establishes a working sequence,
not arbitrary retention or disturbance tolerance.

Memory also makes a simple logical operation possible. Store input \(A\) in the
state. To apply AND with a new input \(B\), reset to zero if \(B=0\); otherwise
hold the current state. To apply OR, reset to one if \(B=1\); otherwise hold.
The controller needs only the new input. It never reads \(A\) and computes the
desired answer on the host.

The tests then applied a second input \(C\) to the literal carried terminal
state. All eight input triples returned both correct intermediate and final
values for AND, and all eight did so for OR.

![All eight input triples for each Qwen AND and OR chain, showing correct independently decoded intermediate and final outputs with literal state carried between the two gates.](stateful-logic.png)

<p class="figure-note"><strong>Logic that uses the stored state.</strong> Each row begins from a chosen starting position on the reference cycle for input A. Controls depend only on the incoming B and C; the second gate receives the exact final state of the first. These finite panels do not establish autonomous combinational gates. See the <a href="receipts/evidence-summary.json">evidence receipt</a>, <a href="provenance.json">provenance</a>, and <a href="build_figures.py">generator</a> (`--verify`).</p>

Hold and reset alone do not supply a **toggle**, which would switch either
stored bit to its opposite. These results therefore do not establish exclusive
OR (XOR) or universal computation.

## Discussion

The useful distinction is between **a response that represents a bit** and
**a state that retains one**. Pythia participates in a complete transaction.
Qwen adds history, letting a later input act on a value already present.
Neither requires the model to understand the message.

For transport alone, direct bytes avoid the model work entirely. The demonstrated
utility is an inspectable auxiliary representation with functioning reply
dataflow and failure handling. Its speed is poor. A competitive application
would need a reason to use this representation despite its cost.

A possible use is studying execution audits. A visible trace can expose every
operation while leaving an observer to reconstruct what the states mean.
Testing that interpretability question would require specified observer access,
competing decoders, and measured recovery. Here the grammar and labels are public,
and a digital lookup can bypass the dynamical calculation.

For memory, the next question is whether writes remain dependable under changed
phases, disturbances, and longer compositions. Shrimp geometry may help locate
useful regions; comparable controls away from it are needed before claiming an
advantage. These existence demonstrations supply a starting point.

## Methods and reproducibility

The reference uses effective float64 (double-precision arithmetic), checked in the full map and saved
trajectories, including normalization and positional operations. A tensor’s
[dtype label]({{< relref "knowledge-base/glossary/dtype/index.md" >}}) alone is insufficient. The schematic feedback rule is

\[
x_{n+1}=E^{\mathsf T}\operatorname{softmax}\!\left(z_\alpha([P;x_n])/T\right).
\]

Here \(n\) counts updates; \(x_n\) is the continuous state, and \(P\) is the fixed
prefix. The brackets \([P;x_n]\) append that state to the prefix. The rows of
\(E\) are token [embeddings]({{< relref "knowledge-base/glossary/embedding/index.md" >}});
\(E^{\mathsf T}\) transposes the matrix. The model produces scores, or
[logits]({{< relref "knowledge-base/glossary/logit/index.md" >}}), denoted
\(z_\alpha\), under steering strength \(\alpha\). Softmax converts these scores
to probabilities after division by temperature \(T\). Multiplication by
\(E^{\mathsf T}\) takes their probability-weighted embedding average. Model weights, prefix, steering location,
initial state, precision, and readout rules define the instrument.

| Artifact | What it supports |
|---|---|
| [Compact evidence receipt](receipts/evidence-summary.json) | Message text, costs, spatial outcomes, and Qwen stateful tests |
| [Numerical provenance](provenance.json) | Sources and derivations for reported numbers |
| [Figure generator](build_figures.py) | Regenerates new diagrams and plots, checks reused image hashes, and verifies quoted numbers and messages; run with `--verify` |

### Appendix: scope of the record

The substantive Pythia transfer is an adapted integration test using known
addresses. Its measurement record was drawn once after the implementation was
fixed, with the changed-reading comparison declared in advance. The spatial
sample and Qwen operations are exploratory numerical observations. No interval
proof or population reliability guarantee is claimed for these operations.

The test harness, control instructions, numerical readouts, and failed or
unresolved outcomes are part of the evidence. Successful recovery here means
the recorded receiver produced the transmitted bytes under the specified
conditions; it does not imply that the model understood their content.

## References

1. <span id="ref-jones"></span> Timothy D. Jones (2026). “Chaos and bifurcations in continuous language-model feedback.” Preprint, [arXiv:2610.31692](https://arxiv.org/abs/2610.31692).
2. <span id="ref-hopfield"></span> J. J. Hopfield (1982). “Neural networks and physical systems with emergent collective computational abilities.” *Proceedings of the National Academy of Sciences*, 79(8), 2554–2558. [doi:10.1073/pnas.79.8.2554](https://doi.org/10.1073/pnas.79.8.2554).
3. <span id="ref-sinha"></span> S. Sinha and W. L. Ditto (1998). “Dynamics Based Computation.” *Physical Review Letters*, 81, 2156–2159. [doi:10.1103/PhysRevLett.81.2156](https://doi.org/10.1103/PhysRevLett.81.2156).
