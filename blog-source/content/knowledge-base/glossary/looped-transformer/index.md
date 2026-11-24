---
title: "Looped transformer"
slug: "looped-transformer"
summary: "A transformer that applies one shared group of layers to its own output several times, so the number of passes, and with it the depth, can be chosen when it runs."
draft: false
pro_reviewed: true
og_image: "looped-transformer-card.png"
og_image_alt: "An ordinary transformer drawn as eight distinct layers beside a looped transformer drawn as a prelude, one shared core that may hold several blocks and feeds its output back into itself for a chosen number of passes, with the input re-entering, and a coda."
---

A **looped transformer** is a [transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}) that applies one shared group of layers to its own output several times before it reads out a prediction. An ordinary transformer is a stack of distinct transformer blocks, or layers, each used once for every token. A looped transformer usually has three parts: a short *prelude* that reads the input once, a *core* of one or more blocks that is applied again and again, and a short *coda* that turns the final state into the output (next-token scores, in a language model). One application of the whole core is a **pass**. Because every pass reuses the same weights, the number of passes can be chosen when the model runs, so depth becomes a setting rather than a fixed property of the architecture. Models built this way are also called recurrent-depth models, and language models built this way are also called looped language models.

With the input held fixed, one common way to write the computation is

\[
e=P(x),\qquad s_{k+1}=G(s_k;\,e)\quad(0\le k\lt K),\qquad \hat{y}=C(s_K),
\]

where \(x\) is the input text, \(P\) is the prelude and \(e\) its output, \(s_k\) is the state after \(k\) passes (one vector per token, the [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}})), \(s_0\) is the start state, set by one of the rules below, \(G\) is one pass of the core together with whatever the design does between passes, \(K\) is the number of passes run, and \(C\) is the coda with the output head, which gives the scores \(\hat{y}\). Because \(G\) is the same function on every pass, the passes iterate a map, the object described on the [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}) page.

Designs differ in five choices that shape how the loop behaves:

- **Trained pass count.** A model can be trained to be read after a fixed number of passes, after a number drawn at random for each training step, or with a learned rule for when to stop.
- **How the input re-enters.** The prelude output \(e\) can be added to the state after every pass, combined with it before every pass by a small learned layer called an *adapter*, or not used again after the first pass.
- **Start state.** The start state \(s_0\) can be \(e\), the token embeddings, or random noise.
- **Normalization.** Rescaling each token's state to a fixed size at the end of every pass keeps the state bounded. If normalization acts only on the inputs to the layers inside the core, the state itself is never rescaled and can grow from pass to pass.
- **Exit rule.** A model can always run a fixed number of passes, or stop early, either when its outputs change by less than a set threshold or when a learned *gate*, a small module that scores whether to stop, says so.

{{< reference-figure
  src="looped-transformer.svg"
  alt="Left: an ordinary transformer drawn as a stack of eight distinct layers, each used once. Right: a looped transformer drawn as a prelude, one shared core that may hold several blocks, with an arrow feeding its output back into itself for the next pass, and a coda; a dashed path shows the prelude output e re-entering the core on every pass."
  caption="Conceptual drawing; the eight-layer stack is a toy count, and no particular model is shown. An ordinary transformer visits each of its distinct layers once. A looped transformer runs a prelude once, applies one shared core, which may hold several blocks, for a chosen number of passes K, and runs a coda once; in many designs the prelude output e re-enters the core on every pass. The drawing says nothing about whether extra passes help."
>}}

## Worked example

Count the blocks applied to each token. A model with \(n_p\) prelude blocks, \(n_c\) core blocks and \(n_d\) coda blocks, run for \(K\) passes, applies

\[
n_p+K\,n_c+n_d
\]

blocks to each token, while storing the weights of only \(n_p+n_c+n_d\) blocks. (Here \(n_p\), \(n_c\) and \(n_d\) are counts of blocks; \(P\) and \(C\) above are the prelude and the coda themselves.) Huginn-0125, the recurrent-depth model of [Geiping et al. (2025)](https://arxiv.org/abs/2502.05171), has 2 prelude, 4 core and 2 coda blocks, and its [Hugging Face release](https://huggingface.co/tomg-group-umd/huginn-0125) runs 32 passes by default: \(2+32\times4+2=132\) block applications per token, from the weights of 8 blocks.

```python
def block_applications(prelude, core, coda, passes):
    """Blocks applied to each token by a prelude, core and coda model run for `passes` passes."""
    counts = (prelude, core, coda, passes)
    if any(isinstance(n, bool) or not isinstance(n, int) for n in counts):
        raise ValueError("all counts must be integers")
    if prelude < 0 or coda < 0 or core < 1 or passes < 1:
        raise ValueError("need prelude >= 0, coda >= 0, core >= 1 and passes >= 1")
    return prelude + passes * core + coda
```

`block_applications(2, 4, 2, 32)` returns `132`.

## What looping does not guarantee

Running more passes does not imply that the state settles, and a settled state need not be a better prediction. A model trained to be read after exactly \(K\) passes was never asked what its later states mean, so reading it after more passes is a new use of the model, not a longer version of the old one. Whether extra passes help is a measured property of a model, a task and an input length, not a consequence of the architecture.

See also: [transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), [iterated map]({{< relref "knowledge-base/glossary/iterated-map/index.md" >}}), [causal attention]({{< relref "knowledge-base/glossary/causal-attention/index.md" >}}).
