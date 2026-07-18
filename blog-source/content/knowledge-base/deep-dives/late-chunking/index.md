---
title: "Late Chunking: Context Before Pooling"
slug: "late-chunking"
date: 2025-08-02
author: Timothy Jones
summary: "Late chunking lets an embedding model contextualize a longer input before pooling smaller retrieval chunks, preserving one vector per chunk without promising that more context always helps."
weight: 10
pro_reviewed: true
aliases:
  - /posts/2025/08/late-chunking/
  - /posts/late-chunking/
ai_disclosure: |
  **AI-use disclosure.** Generative-AI tools helped implement, audit, execute,
  interpret, visualize, review, and draft this study. The author selected the
  research question, authorized the compute, has inspected the artifacts, and is
  responsible for the final text and claims. This is an independent,
  non-peer-reviewed Deep Dive. Verify numbers against the released receipts
  before relying on them.
---

**Late chunking** creates an embedding for each small retrieval chunk *after* a transformer has contextualized a longer input containing that chunk. The chunk boundaries can stay exactly where they were. What moves later is their use in pooling.

That distinction matters:

- Late chunking does **not** ask a model to choose the chunk boundaries.
- It does **not** store one vector for every token.
- It does **not** guarantee better retrieval whenever more context is available.
- It does require access to token-level states from a compatible embedding model.

The method was introduced by Günther et al. in [*Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*](https://arxiv.org/abs/2409.04701v3). This Deep Dive explains the operation, the published evidence, the implementation boundaries, and the cases where the standard approach can still win.

## The short version

Suppose a document contains three chunks:

1. “Berlin is the capital and largest city of Germany.”
2. “Its population exceeds 3.85 million.”
3. “The city is also one of Germany's states.”

With **naive chunking**, the embedding model encodes each numbered span independently. Tokens such as *Its* and *The city* cannot attend to the earlier mention of *Berlin* during that encoding pass.

With **late chunking**, the model encodes all three spans in one in-window input. It then mean-pools the token states inside each of the same three spans. Earlier context can influence the vectors for spans 2 and 3, but only tokens assigned to each span are pooled into that span's final vector.

{{< reference-figure src="knowledge-base/deep-dives/late-chunking-order.svg" label="Naive and late chunking pipelines" alt="Two pipelines use the same document and chunk boundaries. Naive chunking splits before encoding, so the phrase 'the city' cannot attend to the earlier mention of 'Berlin'. Late chunking encodes the in-window document first and then pools the same spans, so that earlier context can influence the chunk vector." caption="Late chunking changes when chunk boundaries isolate context, not necessarily where the boundaries are drawn. Naive chunking encodes each span independently. Late chunking first contextualizes all tokens in one encoder input, then pools token states inside the same spans. Context is available, but the figure does not claim that every reference is resolved or that retrieval must improve." >}}

## The operation, precisely

Let \(x\) be the actual in-window encoder sequence, including any chosen instruction and special tokens, and let \(a_i\) be its attention mask. Here \(x\) also stands for the other fixed encoder inputs, such as position or token-type IDs where the model uses them. The transformer produces a contextual state for each encoded position:

\[
H = E(x,a) = \left[h_1(x,a), h_2(x,a), \ldots, h_m(x,a)\right].
\]

Let \(S_j\) be the encoded positions assigned to retrieval chunk \(j\), with \(\sum_{i \in S_j} a_i > 0\). For attention-mask mean pooling followed by L2 normalization, and assuming the pooled vector has nonzero finite norm:

\[
\begin{aligned}
\mu_j^{\mathrm{late}}
&= \frac{\sum_{i \in S_j} a_i h_i(x,a)}{\sum_{i \in S_j} a_i}, \\
z_j^{\mathrm{late}}
&= \frac{\mu_j^{\mathrm{late}}}{\lVert\mu_j^{\mathrm{late}}\rVert_2}.
\end{aligned}
\]

For naive chunking, let \(x^{(j)}\), \(a^{(j)}\), and \(S_j^{(j)}\) denote chunk \(j\)'s separately encoded input, attention mask, and local pooled positions:

\[
\begin{aligned}
\mu_j^{\mathrm{naive}}
&= \frac{\sum_{r \in S_j^{(j)}} a_r^{(j)} h_r(x^{(j)},a^{(j)})}{\sum_{r \in S_j^{(j)}} a_r^{(j)}}, \\
z_j^{\mathrm{naive}}
&= \frac{\mu_j^{\mathrm{naive}}}{\lVert\mu_j^{\mathrm{naive}}\rVert_2}.
\end{aligned}
\]

The different indices matter. A late position \(i\) is contextualized inside the longer input \(x\); a naive position \(r\) is local to the separately encoded chunk input \(x^{(j)}\). These equations describe the reproduction's pooling recipe. Another embedding model may specify different pooling or normalization.

### What changes and what stays fixed

| Property | Naive chunking | Late chunking |
|---|---|---|
| Chunk text and boundaries | Chosen before encoding | Chosen before encoding |
| Transformer input | One chunk at a time | A longer input containing several chunks |
| Pooling | Over states conditioned on that chunk | Over the same span's states, conditioned on the longer input |
| Stored representation | One vector per chunk | One vector per chunk |
| Query encoding | Model's documented query path | The same documented query path |
| Context reach | The isolated chunk | The encoder's usable input window |

Late chunking changes the **input-conditioned representation**, not the model's learned vector space. There is no per-document retraining.

## A worked example, and its limit

The paper's introductory Berlin example compares the query *Berlin* with three sentence embeddings produced by `jina-embeddings-v2-small-en`. Its reported cosine similarities are:

| Sentence | Naive | Late |
|---|---:|---:|
| Contains “Berlin” directly | 0.8486 | 0.8495 |
| Begins “Its more than 3.85 million inhabitants...” | 0.7084 | 0.8249 |
| Begins “The city is also one of the states...” | 0.7535 | 0.8498 |

This is a useful mechanism-level illustration: the two later sentence vectors change when their token states are conditioned on the preceding sentence. It is **not** a retrieval benchmark. A cosine increase for one query does not establish a better ranking, fewer false matches, or resolution of every anaphor. Those claims require a corpus, queries, relevance judgments, and a retrieval metric.

Source: [Günther et al., Table 1](https://arxiv.org/html/2409.04701v3#S1.T1).

## What the published retrieval evaluation found

The paper compares naive and late chunking on four selected BEIR datasets, NFCorpus, SciFact, FiQA, and TREC-COVID, using three embedding models. Each dataset supplies queries, a document corpus, and qrels that identify relevant documents. Chunk rankings are converted to document rankings under the paper's protocol and scored with nDCG@10.

The paper reports nDCG@10 on a 0-100 scale. The changes below are absolute points on that reporting scale:

| Boundary method | Naive nDCG@10 (×100) | Late nDCG@10 (×100) | Paper-reported change (points) |
|---|---:|---:|---:|
| Five sentences per chunk | 52.4 | 54.3 | +1.9 |
| 256 tokens per chunk | 52.2 | 54.0 | +1.8 |
| Semantic sentence boundaries | 52.4 | 53.8 | +1.5 from unrounded scores |

These are averages across the paper's three models and four selected BEIR tasks, not expected gains for a new corpus. The evaluation used non-overlapping retrieval chunks and a specific protocol for assigning instruction and special-token states to the first or last chunk. See [Section 4.1](https://arxiv.org/html/2409.04701v3#S4.SS1) for the table and [Section 4.2](https://arxiv.org/html/2409.04701v3#S4.SS2) for the chunk-size ablation.

The aggregate hides heterogeneous results. In Section 4.2's nDCG retrieval ablation, the authors use `jina-embeddings-v2-small-en`, fixed-size chunks at several tested sizes, NFCorpus and LongEmbed tasks, and truncate inputs at 8,192 tokens. They report stronger late-chunking results particularly at smaller chunk sizes, while some reading-comprehension configurations with larger chunks favor naive chunking. They also report no late-chunking benefit on the tested Needle-8192 and Passkey-8192 configurations, where short relevant text is placed inside unrelated context. Those are results and interpretations for that experiment, not a rule that a task family universally favors either method.

The defensible conclusion is that late chunking can improve retrieval when surrounding text helps represent the target span. “More context is always better” is not supported.

## An auditable SciFact matched-content-token re-evaluation

This page also ships a runnable re-evaluation on the complete SciFact test retrieval benchmark: 5,183 candidate documents, 300 test queries, and 339 positive query-document judgments. It uses [`jina-embeddings-v2-small-en` at commit `44e7d1d`](https://huggingface.co/jinaai/jina-embeddings-v2-small-en/tree/44e7d1d6caec8c883c2d4b207588504d519788d0), its remote implementation at commit `f3ec4cf`, and 256-content-token spans without overlap. The frozen corpus produced 9,356 chunks. Its longest document had 1,937 content tokens, so no document was truncated against the model's 8,192-token limit.

The naive and late arms match the corpus, query path, tokenizer and model revisions, content-token boundaries, mean-pooling operation, normalization, and cosine scoring. They match frozen content-token IDs and slices, but this is not a context-only causal ablation. Each naive chunk is encoded with its own `[CLS]` and `[SEP]`, whereas the late arm uses one document-level pair assigned to the first and last spans. Those special tokens can affect content-token hidden states even when the special-token states themselves are excluded from pooling. More generally, independently encoded chunks may differ in position IDs or other position-dependent model inputs. A context-only ablation must hold fixed every non-contextual encoder input and every pooled position, then vary only cross-span context or attention. Where a compatible model supports it, one possible design is to encode the same full token sequence with carefully matched full-attention and block-diagonal-attention masks.

The three evaluated document representations are:

1. **Naive chunks:** each frozen content-token slice receives its own special tokens and is encoded independently.
2. **Late chunks:** the complete document is encoded once, then the same content-token spans are pooled. `[CLS]` belongs to the first span and `[SEP]` to the last.
3. **Whole document:** the same full-document states used by the late arm are pooled into one vector. This is a control for “use all context” without retaining one retrieval vector per chunk.

Queries receive no instruction prefix and use the same attention-mask mean pooling and L2 normalization in every arm. The script scores every corpus chunk, takes each document's maximum chunk score for the two chunked arms, ranks the complete document corpus, and calculates document-level nDCG@10, Recall@10, and MRR@10 from the frozen binary qrels. Stable corpus order breaks exact score ties.

This is an **auditable matched-content-token re-evaluation**, not an exact replication of one paper-table cell or a context-only causal ablation. The pinned paper helper decodes token slices before its evaluator retokenizes them; that can alter boundary tokenization. This version passes the original token IDs directly so the naive and late arms share exact content spans. It also scores the full corpus before max-chunk document aggregation instead of collapsing a previously retrieved candidate-chunk list. Those choices make the content-token comparison cleaner for this tutorial, but its values should not be substituted for the paper's protocol.

{{< reference-figure src="knowledge-base/deep-dives/late-chunking/fig-scifact-retrieval.svg" label="SciFact retrieval quality for three document protocols" alt="SciFact test document nDCG at 10 on a zero-to-one axis, with query-bootstrap intervals: naive 256-token chunks 0.6414 [0.5942, 0.6876]; late 256-token chunks 0.6610 [0.6145, 0.7071]; whole-document encoding 0.6389 [0.5918, 0.6852]. Higher is better." caption="**Finding:** on this frozen SciFact test benchmark, the late-chunk protocol reached 0.6610 document nDCG@10, compared with 0.6414 for the naive-chunk protocol and 0.6389 for one whole-document vector. The naive and late arms match content-token slices but follow different stated special-token pooling policies, so this is an end-to-end protocol comparison. Horizontal lines show descriptive query-bootstrap intervals from 20,000 resamples of the 300 fixed queries. They describe benchmark stability, not performance on another corpus or a population confidence interval. The table below is the full visible text equivalent. [Figure receipt](fig-scifact-retrieval.receipt.json)." >}}

| Document representation | nDCG@10 | 95% query-bootstrap interval | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Naive 256-token chunks | 0.6414 | [0.5942, 0.6876] | 0.7626 | 0.6123 |
| Late 256-token chunks | 0.6610 | [0.6145, 0.7071] | 0.7776 | 0.6337 |
| Whole document | 0.6389 | [0.5918, 0.6852] | 0.7592 | 0.6105 |

Late chunking had the highest aggregate score of these three arms, but the aggregate hides how sparse the changes were.

{{< reference-figure src="knowledge-base/deep-dives/late-chunking/fig-query-deltas.svg" label="Sorted query-level late-minus-naive changes" alt="Sorted query-level SciFact nDCG at 10 differences for late minus naive chunking: 49 queries improved, 229 tied, and 22 worsened. The mean difference is +0.0196, with a paired query-bootstrap interval from +0.0033 to +0.0358. Points above zero favor late chunking; points below zero favor naive chunking." caption="**Finding:** the mean late-minus-naive nDCG@10 difference was +0.0196, with a descriptive paired query-bootstrap interval of [+0.0033, +0.0358]. At query level, 49 improved, 229 tied, and 22 worsened on nDCG@10. Circles above zero favor late chunks, squares below zero favor naive chunks, and diamonds on zero are ties, so color is not required to interpret the plot. The interval resamples the fixed query-level paired differences; it is not a p-value or a universal retrieval claim. [Figure receipt](fig-query-deltas.receipt.json)." >}}

| Paired comparison | Mean nDCG@10 difference | 95% paired query-bootstrap interval | Improved | Tied | Worsened |
|---|---:|---:|---:|---:|---:|
| Late minus naive | +0.0196 | [+0.0033, +0.0358] | 49 | 229 | 22 |

The correct reading is narrow: late chunking improved the mean ranking metric for this pinned model and benchmark, while some queries regressed and most had the same nDCG@10. The whole-document control also did not match the late-chunk score, so contextualizing a document and compressing it into one vector is not equivalent to retaining context-conditioned retrieval chunks. This comparison does not identify why an individual query changed, benchmark indexing throughput, or establish an expected gain on another corpus.

### Reproduce and audit the result

The page bundle contains the [reproduction script](reproduce.py), [locked dependency graph](reproduce.py.lock), [byte-identical test qrels](receipts/scifact-test-qrels.tsv), [aggregate metrics](receipts/aggregate.json), [per-query metrics](receipts/per-query.csv), [top-ten rankings](receipts/top-10-rankings.jsonl), [run receipt](receipts/run.receipt.json), [provenance manifest](provenance.json), and [SciFact attribution](ATTRIBUTION.md). Model weights, corpus text, and caches remain uncommitted.

Run the canonical model evaluation with:

```bash
uv run --frozen reproduce.py --run --device cpu --batch-size 32 --threads 8
```

Before inference, the script validates the SciFact archive; model, tokenizer, and remote Python snapshots; the expected lockfile hash; and the material installed package versions. The documented `uv run --frozen` command enforces the complete locked dependency graph. The receipt distinguishes that canonical command from the observed Python argument vector and records platform and package versions, CPU settings, input hashes, protocol choices, phase timings, and output hashes. Deterministic-kernel mode is enabled, but the receipt does not promise bitwise identity across different hardware or math libraries.

CI uses the standard-library-only offline check:

```bash
python3 reproduce.py --verify
```

That check binds the run to the script and lockfile, validates the frozen qrels hash, requires one complete ten-document ranking with unique document IDs for every query-arm pair, re-derives relevance labels and query metrics, re-runs the query bootstraps, and regenerates both SVGs and their receipts byte for byte.

## The implementation boundary

Late chunking needs more than an API that returns one already-pooled vector for an input. You need:

1. token-level hidden states from the embedding model;
2. one frozen tokenization of the longer input;
3. one half-open token span for each retrieval chunk; if boundaries originate in character or sentence space, request offset mappings and align those boundaries to the frozen tokenization;
4. the model's compatible pooling, instruction, and normalization rules; and
5. an explicit policy for special tokens, padding, truncation, and over-length inputs.

The pooling core can be small. The hard part is producing correct spans and model inputs:

```python
from collections.abc import Sequence

import torch
from torch import Tensor


@torch.inference_mode()
def mean_pool_spans(
    token_states: Tensor,          # [tokens, hidden]
    attention_mask: Tensor,        # [tokens], 1 for usable positions
    spans: Sequence[tuple[int, int]],  # half-open [start, stop)
) -> Tensor:
    """Pool already-contextualized token states into normalized chunk vectors."""
    if token_states.ndim != 2:
        raise ValueError("token_states must have shape [tokens, hidden]")
    if attention_mask.shape != token_states.shape[:1]:
        raise ValueError("attention_mask must have one value per token")

    usable = attention_mask.to(dtype=torch.bool)
    vectors: list[Tensor] = []
    for start, stop in spans:
        if not 0 <= start < stop <= token_states.shape[0]:
            raise ValueError(f"invalid token span: {(start, stop)}")
        keep = usable[start:stop]
        if not torch.any(keep):
            raise ValueError(f"token span contains no usable states: {(start, stop)}")

        pooled = token_states[start:stop][keep].mean(dim=0)
        norm = torch.linalg.vector_norm(pooled)
        if not torch.isfinite(pooled).all() or not torch.isfinite(norm) or norm <= 0:
            raise ValueError(
                f"token span produced a non-finite or zero-norm vector: {(start, stop)}"
            )
        vectors.append(pooled / norm)

    if not vectors:
        raise ValueError("at least one token span is required")
    return torch.stack(vectors)
```

This function assumes the transformer has already processed the complete in-window input. It intentionally does not pretend that sentence splitting, offset alignment, model loading, or instruction handling are universal.

### Boundary alignment is not optional

A reliable pipeline should tokenize the document once and derive every chunk span against that frozen token sequence. If the chunker emits character or sentence boundaries, request offset mappings and map those boundaries to token positions. If the chunker operates directly on token counts, define half-open spans over the frozen content-token IDs, as the reproduction does. In either case, do not decode slices and re-tokenize them to recover boundaries.

Decide and test all of the following:

- whether chunk spans overlap;
- where prepended instructions, `[CLS]`, and appended separator states are pooled;
- whether padding is excluded by the attention mask, how instruction and special tokens consume the input budget, and how silent truncation is prevented; if document content exceeds the remaining budget, reject it or route it through the documented macro-window path rather than pooling an incomplete chunk;
- whether normalization occurs before or after pooling;
- whether query and document instructions differ;
- which model, tokenizer, and remote-code revisions are pinned; and
- what happens when a chunk maps to no usable tokens.

The paper's evaluation assigns prepended special and instruction-token states to the first chunk and appended states to the last. Other model families may require a different documented policy.

For a full reference implementation, inspect the authors' [official repository at commit `1d3bb02`](https://github.com/jina-ai/late-chunking/tree/1d3bb02bf091becd0771455e4e7959463935e26c). Treat that implementation and the chosen model revision as versioned dependencies.

## Documents longer than the encoder window

Full-document contextualization is possible only when the document fits within the embedding model's usable input. If it does not, the paper's **long late chunking** method divides the token sequence into overlapping macro-windows and then stitches one contextual state back onto each document token position before span pooling.

The authors' [pinned implementation](https://github.com/jina-ai/late-chunking/blob/1d3bb02bf091becd0771455e4e7959463935e26c/chunked_pooling/mteb_chunked_eval.py#L128-L159) uses a concrete ownership rule. It keeps every state from the first macro-window. For each later window, it discards the states for that window's repeated leading overlap and appends only the states for new document positions. The overlap therefore supplies preceding context to the new positions, while the stitched sequence still has exactly one state per document token. Chunk boundaries are then applied to that stitched sequence, producing one vector and one ID per retrieval chunk.

{{< reference-figure src="knowledge-base/deep-dives/late-chunking-context-window.svg" label="Long-document overlap stitching" alt="A document that fits one encoder window is contextualized in one pass. For an over-length document, the pinned long-late-chunking implementation keeps all states from the first macro-window, discards the repeated leading-overlap states from the next window, and appends only its new positions, yielding one stitched state per document token." caption="The scope of late chunking is the embedding model's usable input window. If the document fits, all retrieval spans can be contextualized in one encoder pass. In the pinned long-document implementation shown here, macro-window B repeats positions 4 and 5 as context, but those repeated B states are discarded; the retained B states begin at position 6. Concatenating the kept A and B states restores positions 1 through 8 exactly once before retrieval-span pooling. This one-sided ownership rule is implementation-specific, and overlap does not recreate one full-document encoder pass." >}}

This is not permission to improvise a duplicate rule. A different implementation that averages overlap states, chooses centered states, or emits window-local chunk vectors is a different algorithm and should be documented and evaluated as such. Reserve capacity for instruction and special tokens, prevent tokenizer truncation, and ensure every pooled span is fully covered. Overlap reduces one boundary problem; it does not create global context. Long inputs also cost more to encode because transformer attention and intermediate activations grow with sequence length. Measure indexing throughput and peak memory with the actual model and window size.

## Related techniques are not synonyms

| Technique | Context mechanism | Stored representation |
|---|---|---|
| Naive overlapping chunks | Repeats neighboring text in independently encoded chunks | One vector per overlapping chunk |
| Late chunking | Contextualizes a longer encoder input, then pools smaller spans | One vector per pooled span |
| Contextual text augmentation | Adds generated or extracted context to chunk text before embedding | Usually one vector per augmented chunk |
| ColBERT-style late interaction | Keeps multiple token vectors and scores them with query-time MaxSim | Multiple vectors per passage |

Late chunking and ColBERT's “late interaction” share a word, not an operation. ColBERT retains token-level vectors for query-time comparison; late chunking pools contextual token states into a single vector per retrieval chunk. See the [ColBERT paper](https://arxiv.org/abs/2004.12832) for that separate design.

## How to evaluate it on your corpus

Do not evaluate late chunking by hand-picking positive and negative words and asking whether absolute cosine scores moved in a preferred direction. Evaluate the retrieval system you intend to deploy.

1. Freeze a document corpus, queries, and relevance judgments.
2. Freeze the embedding model and tokenizer revisions, instructions, chunker, boundaries, and query encoding.
3. Decide whether the target is a context-only ablation or an end-to-end protocol comparison. For a context-only ablation, hold fixed input IDs, including instructions and special tokens, position and token-type inputs, pooled positions, pooling, and normalization, and vary only cross-span context or attention. Merely excluding special-token states from the mean is insufficient because those tokens can still influence content-token states. Otherwise, disclose every additional protocol difference.
4. Compare ranking metrics such as nDCG@10, Recall@k, and MRR at the query level.
5. Report uncertainty over defensible units such as queries or datasets, preserving known grouping and dependence.
6. Report indexing time, peak memory, index size, and query latency.
7. Stratify by document length, chunk size, cross-chunk dependency, and irrelevant surrounding context.
8. Inspect failures. A mean gain can hide queries that regress badly.

For a production decision, the relevant quantity is the control-relative retrieval change on representative data, not whether one illustrative cosine similarity increased.

## Decision checklist

Late chunking is a plausible candidate when:

- useful context often lives outside the target retrieval chunk;
- the embedding model exposes token states and a compatible pooling recipe;
- several retrieval chunks fit inside one practical encoder window; and
- additional indexing cost is acceptable while query-time vector search stays unchanged.

Prefer a simpler baseline, or at least expect a smaller gain, when:

- chunks are already self-contained;
- surrounding text is mostly irrelevant;
- documents routinely exceed the usable window and local macro-context is insufficient;
- the provider exposes only a pooled embedding endpoint; or
- the evaluation shows no robust ranking benefit for the added indexing cost.

## Primary references

- Michael Günther et al., [*Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*, v3](https://arxiv.org/abs/2409.04701v3), 2024.
- Jina AI, [official Late Chunking implementation, pinned commit](https://github.com/jina-ai/late-chunking/tree/1d3bb02bf091becd0771455e4e7959463935e26c).
- Michael Günther et al., [*Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents*](https://arxiv.org/abs/2310.19923), 2023.
- Jina AI, [`jina-embeddings-v2-small-en` model card, pinned revision](https://huggingface.co/jinaai/jina-embeddings-v2-small-en/blob/44e7d1d6caec8c883c2d4b207588504d519788d0/README.md).
- Nandan Thakur et al., [*BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*](https://arxiv.org/abs/2104.08663), 2021.
- David Wadden et al., [*Fact or Fiction: Verifying Scientific Claims*](https://arxiv.org/abs/2004.14974), 2020; see also the [SciFact data license notice](https://github.com/allenai/scifact/blob/master/LICENSE.md).
- Omar Khattab and Matei Zaharia, [*ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*](https://arxiv.org/abs/2004.12832), 2020.
