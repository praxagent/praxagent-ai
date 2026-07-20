---
title: "Retrieval ranking metrics"
slug: "retrieval-ranking-metrics"
summary: "Measures that compare a ranked retrieval list with relevance judgments, including position-discounted gain, recall, and reciprocal rank."
og_image: "retrieval-ranking-metrics-top-ten.png"
og_image_alt: "A toy top-ten ranking shows how recall, reciprocal rank, and discounted gain read the same results differently."
draft: false
pro_reviewed: true
---

An information-retrieval system returns documents in an ordered list for a query. **Retrieval ranking metrics** summarize how that list agrees with [relevance judgments]({{< relref "knowledge-base/glossary/relevance-judgments/index.md" >}}). The notation `@10` means that only the first 10 returned positions are evaluated. Different metrics reward different things, so a report should name each metric rather than saying only that retrieval “improved.”

Three common metrics are:

- **normalized discounted cumulative gain at 10 (nDCG@10)** rewards relevant documents near the top and can use graded relevance;
- **Recall@10** measures what fraction of the known relevant documents appear in the first 10 results;
- **mean reciprocal rank at 10 (MRR@10)** averages the reciprocal position of the first relevant result, giving 1 for rank 1, \(1/2\) for rank 2, and 0 when no relevant result appears by rank 10.

The reference labels are often stored as **qrels**, short for query relevance judgments.

{{< reference-figure
  src="retrieval-ranking-metrics-top-ten.svg"
  alt="A toy top-ten ranking marks relevant documents at ranks two, five, and nine and shows how recall, reciprocal rank, and discounted gain read the same list differently."
  caption="In this toy query, relevant documents appear at ranks 2, 5, and 9, while the qrels contain four known relevant documents in total. Recall@10 is therefore 3/4, reciprocal rank is 1/2 because the first relevant result is at rank 2, and nDCG discounts the later relevant results more strongly. The example uses binary labels; graded labels can give different gains."
>}}

## One ranked-list example

Suppose one query has four known relevant documents. The system returns three of them at ranks 2, 5, and 9 within its first 10 results.

### Recall@10

\[
\operatorname{Recall@10}
=
\frac{\text{known relevant documents retrieved in the top 10}}
{\text{known relevant documents}}
=
\frac{3}{4}
=0.75.
\]

Recall ignores the order among those first 10 positions. Moving a relevant document from rank 9 to rank 2 does not change Recall@10.

### Reciprocal rank and MRR@10

For one query, reciprocal rank at 10 is

\[
\operatorname{RR@10}
=
\begin{cases}
1/r, & \text{if the first relevant result is at rank }r\le 10,\\
0, & \text{if no relevant result appears by rank 10}.
\end{cases}
\]

The first relevant result in the example is at rank \(r=2\), so \(\operatorname{RR@10}=1/2=0.5\). **Mean reciprocal rank at 10 (MRR@10)** is the arithmetic mean of RR@10 across evaluation queries. It mostly answers “how soon did the first relevant result appear?” and ignores additional relevant documents after the first.

### nDCG@10

Let \(g_i\ge 0\) be the relevance grade of the document returned at rank \(i\). One common definition of discounted cumulative gain at 10 is

\[
\operatorname{DCG@10}
=
\sum_{i=1}^{10}\frac{2^{g_i}-1}{\log_2(i+1)}.
\]

The symbol \(\log_2\) means a base-two logarithm. The denominator discounts relevant documents at later ranks. **Ideal DCG at 10 (IDCG@10)** is the DCG of the best possible ordering of the judged documents for that query. Then

\[
\operatorname{nDCG@10}
=
\frac{\operatorname{DCG@10}}{\operatorname{IDCG@10}}.
\]

For binary grades in the toy example, the relevant ranks 2, 5, and 9 give DCG@10 of approximately 1.319. Placing all four known relevant documents at ranks 1 through 4 gives IDCG@10 of approximately 2.562, so nDCG@10 is approximately 0.515. If a query has no known relevant documents, nDCG and recall need an explicit dataset or library convention rather than division by zero.

## From per-query scores to a reported result

Evaluation normally computes one score per query and then takes an unweighted mean across queries. This is **macro-averaging**: each query contributes equally even if queries have different numbers of relevant documents. A [bootstrap]({{< relref "knowledge-base/glossary/bootstrap/index.md" >}}) over queries can summarize uncertainty or paired differences between systems, provided the resampling unit and dependence assumptions fit the evaluation design.

## What these metrics do not establish

- A higher score does not prove that the retrieved text is factually correct or useful in a downstream biological decision.
- Scores are conditional on the qrels, cutoff, query set, tie handling, and treatment of unjudged documents.
- MRR@10 does not reward a second relevant result, while Recall@10 does not reward earlier ordering within the top 10.
- Differences between systems need paired per-query analysis; two rounded averages alone do not show how consistently one system wins.

See also: [relevance judgments]({{< relref "knowledge-base/glossary/relevance-judgments/index.md" >}}), [bootstrap]({{< relref "knowledge-base/glossary/bootstrap/index.md" >}}), [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [cosine similarity]({{< relref "knowledge-base/glossary/cosine-similarity/index.md" >}}).
