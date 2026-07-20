---
title: "Relevance judgments"
slug: "relevance-judgments"
summary: "Reference labels that say which documents are considered relevant to each evaluation query."
og_image: "relevance-judgments-query-document-grid.png"
og_image_alt: "A query-document grid distinguishes relevant, not relevant, and unjudged pairs."
draft: false
pro_reviewed: true
---

**Relevance judgments** are reference labels that say which documents are considered relevant to each evaluation query. In information retrieval, a collection of these labels is often called **qrels**, short for query relevance judgments.

A qrels record usually identifies a query, a document, and a relevance label. A binary label can mean relevant or not relevant. A graded label can distinguish levels such as highly relevant, partly relevant, and not relevant.

{{< reference-figure
  src="relevance-judgments-query-document-grid.svg"
  alt="A query-by-document grid contains relevant, not relevant, and unjudged cells, showing that missing judgments are distinct from explicit negative labels."
  caption="Toy qrels for two queries label some query-document pairs relevant or not relevant while leaving others unjudged. Relevance belongs to a pair, not to a document alone: D2 can be relevant to Q1 and not relevant to Q2. An empty cell means no judgment was supplied; it does not logically prove nonrelevance."
>}}

## A practical record

One simple tabular format is:

| query | document | label |
| --- | --- | ---: |
| Q1 | D2 | 1 |
| Q1 | D5 | 1 |
| Q1 | D8 | 0 |
| Q2 | D2 | 0 |

The exact file format and label scale must be documented. Some benchmark qrels contain only known relevant documents. Others include explicit nonrelevant labels. Treating every unlisted pair as nonrelevant may be a benchmark convention, but it is not the same claim as a human explicitly judging every pair.

## Where judgments come from

Judgments can come from subject-matter experts, dataset creators, observed interactions, or a pooling process in which assessors label documents returned by several systems. Each source has limitations. Experts can disagree, interaction data can reflect position bias, and pooled judgments can miss relevant documents that no contributing system retrieved.

For scientific or medical retrieval, write down the relevance question. “Discusses the same topic,” “supports the claim,” and “contains enough evidence to answer the query” are different criteria. A label is useful only in relation to a stated task.

## How retrieval metrics use qrels

[Retrieval ranking metrics]({{< relref "knowledge-base/glossary/retrieval-ranking-metrics/index.md" >}}) compare a system's ranked document list with the qrels. Recall needs the number of known relevant documents, mean reciprocal rank uses the position of the first relevant result, and normalized discounted cumulative gain can use graded relevance. Changing the qrels can change every score even when the ranked list stays fixed.

## What relevance judgments do not establish

- They are not ground truth in the sense of being complete, objective, or error-free.
- A document judged relevant to one query is not automatically relevant to another.
- An unjudged document is not logically identical to a judged nonrelevant document.
- Agreement with qrels measures the stated evaluation task, not factuality, usefulness to every user, or downstream biological validity.

See also: [retrieval ranking metrics]({{< relref "knowledge-base/glossary/retrieval-ranking-metrics/index.md" >}}), [bootstrap]({{< relref "knowledge-base/glossary/bootstrap/index.md" >}}).
