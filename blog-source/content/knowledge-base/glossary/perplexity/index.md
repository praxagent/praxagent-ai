---
title: "Perplexity"
slug: "perplexity"
summary: "The exponential of a model's average next-token loss on a text, read as an effective number of choices: perplexity k means the same loss as giving every token that came next probability 1/k."
draft: false
pro_reviewed: true
og_image: "perplexity-card.png"
og_image_alt: "Perplexity rising exponentially with the average next-token loss, with marks at ln 2, ln 4 and ln 10 nats, beside two toy two-token texts, with probabilities one quarter and one quarter or one half and one eighth, that both have perplexity 4."
---

**Perplexity** turns a language model's average next-token loss into a count of choices. If the model's [log loss]({{< relref "knowledge-base/glossary/log-loss/index.md" >}}) on a text averages \(L\) [nats]({{< relref "knowledge-base/glossary/nat/index.md" >}}) per token, nats being the units of the natural logarithm, its perplexity is

\[
\mathrm{PPL}=e^{L},
\qquad
L=-\frac{1}{N}\sum_{t=1}^{N}\ln p_t ,
\]

where \(\mathrm{PPL}\) is the perplexity, \(\ln\) is the natural logarithm, whose base is \(e\approx2.718\), \(p_t\) is the probability the model gave to the token that actually came next at position \(t\), and \(N\) is the number of positions scored. A model that gave every true token probability \(1/k\) would have \(L=\ln k\) and perplexity \(k\): the loss of an even guess among \(k\) tokens that always includes the right one. In that sense perplexity is an effective number of choices. Lower is better, and the floor is 1, for a model that is always certain and always right.

Because the exponential of an average logarithm is a geometric mean, perplexity is the reciprocal of the geometric mean of the probabilities given to the true tokens:

\[
\mathrm{PPL}=\Bigl(\prod_{t=1}^{N}p_t\Bigr)^{-1/N}.
\]

A change in loss is a fixed ratio of perplexities: a loss higher by \(\Delta\) nats multiplies the perplexity by \(e^{\Delta}\), whatever its starting value. A change of 0.01 nats is about 1% (\(e^{0.01}\approx1.01005\)), and about 0.69 nats doubles the perplexity (\(e^{0.69}\approx1.994\)).

{{< reference-figure
  src="perplexity.png"
  alt="Left: perplexity, e to the loss, rising from 1 at zero loss to about 20 at 3 nats, with marked points at ln 2 nats for 2, ln 4 nats for 4 and ln 10 nats for 10. Right: bars for the probabilities given to the true tokens of two toy texts, 0.25 and 0.25 in blue and 0.5 and 0.125 in brown, with a dashed line at their shared geometric mean, 0.25."
  caption="Toy values, no language model. (a) Perplexity is e raised to the average loss in nats, so each additional ln 2 ≈ 0.69 nats doubles it. (b) Two texts of two tokens each: the model gives the true tokens 1/4 and 1/4, or 1/2 and 1/8. Both have geometric mean 1/4, and so perplexity 4: the same average loss as giving every true token probability 1/4. A perplexity summarizes a whole text; it does not say which tokens were hard."
>}}

## Worked example

In the first text the model gives each true token probability 1/4, so \(L=\ln4\approx1.386\) nats and the perplexity is 4. In the second it gives 1/2 and then 1/8: \(L=(\ln2+\ln8)/2\approx(0.693+2.079)/2\approx1.386\) nats again, and the perplexity is again 4, because \(\sqrt{(1/2)(1/8)}=1/4\). One well-predicted token and one poorly predicted one give the same loss as two middling ones.

```python
import math

def perplexity(probs):
    """e raised to the average negative log probability given to the true tokens."""
    values = [float(p) for p in probs]
    if not values or any(not 0 < p <= 1 for p in values):
        raise ValueError("need one or more probabilities in (0, 1]")
    return math.exp(-sum(math.log(p) for p in values) / len(values))
```

`round(perplexity([0.5, 0.125]), 6)` returns `4.0`.

## What perplexity does not tell you

Perplexity scores only the probabilities given to the tokens that came next, so it does not measure how uncertain the model was. A model that gives a wrong token 0.75 and the true one 0.25 has perplexity 4 on that token, like one that gives each of four tokens 0.25, yet its prediction is far more concentrated: its [next-token entropy]({{< relref "knowledge-base/glossary/next-token-entropy/index.md" >}}) is 0.56 nats against \(\ln4\approx1.39\). Perplexity is per token, so it depends on the [tokenizer]({{< relref "knowledge-base/glossary/tokenization/index.md" >}}): models that split the same text into different numbers of tokens are not comparable until both are converted to a common unit, such as nats per byte or per word. It depends on the text too, and a value means something only for the text it was measured on. The base of the logarithm does not matter if it is used consistently: 2 raised to the loss in bits gives the same number. An average hides individual tokens, so a small change in perplexity can come from many tiny changes or from a few large ones in opposite directions. And a single probability of zero would make the loss, and the perplexity, infinite.

See also: [log loss]({{< relref "knowledge-base/glossary/log-loss/index.md" >}}), [nat]({{< relref "knowledge-base/glossary/nat/index.md" >}}), [tokenization]({{< relref "knowledge-base/glossary/tokenization/index.md" >}}), [next-token entropy]({{< relref "knowledge-base/glossary/next-token-entropy/index.md" >}}).
