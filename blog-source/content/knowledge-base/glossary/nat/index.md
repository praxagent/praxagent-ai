---
title: "Nat"
slug: "nat"
summary: "The natural-logarithm unit of growth and uncertainty: one nat is about 1.44 bits, and an entropy of H nats is as uncertain as an even choice among about e^H options."
draft: false
pro_reviewed: false
og_image: "nat-card.png"
og_image_alt: "A factor of e is 1 nat. A factor of 2 is ln 2, about 0.693 nats, which is 1 bit."
---

A **nat** is the unit of a natural logarithm, the logarithm with base \(e\approx 2.718\). A quantity that grows by a factor \(r\) changes by \(\ln r\) nats. One nat is a factor of \(e\). The same change in bits, which use log base 2, is larger by a factor of \(\log_2 e \approx 1.443\):

\[
1\text{ nat} = \log_2 e \text{ bits} \approx 1.443\text{ bits},
\qquad
\ln r = \frac{\log_2 r}{\log_2 e}.
\]

A [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}) of \(\lambda\) nats per step is a long-run average rate of logarithmic growth. A tiny displacement grows by a factor of roughly \(e^{\lambda n}\) over \(n\) steps, even though the factor of a single step can be larger or smaller than \(e^{\lambda}\). An exponent of \(\ln 2 \approx 0.693\) nats per step therefore means doubling per step on average, not necessarily exact doubling at every step.

When natural logarithms are used, nats also measure the entropy of a next-token distribution, the probabilities a model assigns to the possible next tokens. Entropy is the probability-weighted average of \(-\ln p\) over the outcomes. They also measure a Kullback-Leibler divergence, the probability-weighted average of the log ratio of one distribution's probabilities to another's. A fair coin has entropy \(\ln 2\) nats, which is exactly 1 bit. A uniform distribution over \(V\) tokens has entropy \(\ln V\) nats.

{{< reference-figure
  src="nat.svg"
  alt="A factor of e is labeled 1 nat. A factor of 2 is labeled ln 2, about 0.693 nats, and also 1 bit."
  caption="Two changes drawn with length equal to their size in nats. Multiplying by e, about 2.718, is 1 nat. Multiplying by 2 is ln 2, about 0.693 nats, and that same change is 1 bit. The bars are the unit conversion, not a measured trajectory."
>}}

## Reading an entropy in nats

A nat is also a unit of uncertainty. It works like a bit, but with the natural logarithm, so one nat is about 1.44 bits. At each step a language model gives a probability for every possible next token, and the entropy of those probabilities measures how spread out its prediction is:

\[
H=-\sum_{t} p_t \ln p_t ,
\]

where \(p_t\) is the probability of token \(t\) and the sum runs over the vocabulary. A simple way to read it: an even choice among \(N\) tokens has entropy \(\ln N\), so an entropy of \(H\) nats is as uncertain as choosing evenly among about \(e^{H}\) tokens. That number is the distribution's effective number of choices, also called its perplexity.

| Entropy | Even choice among |
|---|---|
| 0 nats | 1 token (certain) |
| 2 nats | about 7 tokens |
| 5 nats | about 150 tokens |
| 6.3 nats | about 545 tokens |
| 11.93 nats | all 151,936 tokens |

The exact values are \(e^{2}\approx 7.4\), \(e^{5}\approx 148\) and \(e^{6.3}\approx 545\). The last row, \(\ln 151{,}936\approx 11.93\) nats, is the largest entropy a prediction over the 151,936-token Qwen2.5 vocabulary can have.

Because the scale is logarithmic, regimes a few nats apart differ greatly. A loop whose predictions average 5 nats spreads each step over the equivalent of about 150 tokens; one near 2 nats spreads it over about 7. Comparing the two compares a competition among a few tokens with one about twenty times broader.

The equivalence holds for uncertainty only. A real next-token distribution with an entropy of 5 nats is rarely even over 150 tokens: it may put most of its probability on a handful of tokens and spread the rest thinly over thousands. So \(e^{H}\) summarizes how spread out a prediction is. It is not a count of the tokens with nonzero probability, and two distributions with the same entropy can favor entirely different tokens. The perplexity reported when a model is evaluated on text is a different quantity: it exponentiates the average cross-entropy against the tokens that actually came next, so it measures how well the predictions fit the text, not how spread out they are.

## Worked example

Suppose a tiny distance doubles exactly at every step. After \(n\) steps it has grown by a factor \(2^n\). In nats per step that rate is

\[
\frac{1}{n}\ln(2^n)=\ln 2\approx 0.693.
\]

In bits per step it is \(1\). The two reports are the same growth. A next-token distribution that puts probability \(1/2\) on each of two tokens has entropy \(-\ln(1/2)=\ln 2\) nats, again 1 bit. One that puts probability \(1/7\) on each of seven tokens has entropy \(\ln 7\approx 1.95\) nats, about 2.81 bits, and reading that entropy back gives \(e^{1.95}\approx 7\) tokens.

A nat is not a token, a probability, or a count of events. A positive exponent in nats per step gives the long-run average rate at which a tiny displacement grows. It does not count tokens, and on its own it does not predict at which step a perturbation will change the generated text.

See also: [Lyapunov exponent]({{< relref "knowledge-base/glossary/lyapunov-exponent/index.md" >}}), [logit]({{< relref "knowledge-base/glossary/logit/index.md" >}}).
