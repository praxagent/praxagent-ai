---
title: "Tokenization"
slug: "tokenization"
summary: "The model-specific process that divides an input into tokens and maps those tokens to integer identifiers."
og_image: "tokenization-text-to-ids.png"
og_image_alt: "One text string is divided into token pieces with illustrative integer identifiers."
draft: false
pro_reviewed: true
---

**Tokenization** is the model-specific process that divides an input into units called **tokens** and maps each token to an integer **token identifier**. A token can be a word, part of a word, punctuation, whitespace attached to another piece, a byte, or a special control symbol. Tokens are not reliably the same as words.

{{< reference-figure
  src="tokenization-text-to-ids.svg"
  alt="The text Root-like growth is divided into four token pieces, each with its own illustrative integer identifier."
  caption="One illustrative tokenizer divides `Root-like growth` into `Root`, `-`, `like`, and ` growth`, then maps those pieces to integer identifiers. Another tokenizer could divide the same characters differently. The identifiers are arbitrary labels within one tokenizer vocabulary, not measurements or universal word codes."
>}}

## Why token boundaries matter

Suppose one tokenizer turns `unbelievable` into one token while another produces `un`, `believ`, and `able`. The second model uses three sequence positions for the same visible word. This changes token counts, where a fixed token limit cuts the text, and which token states are included by [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}).

For retrieval systems, a **chunk** is usually a span of the original text chosen by an application. A chunk is not automatically one token. The application may define a chunk in characters, words, sentences, or tokens, but it must eventually align the span with the tokenizer used by the encoder.

## What a tokenizer usually returns

A practical tokenizer interface may provide:

- token identifiers for model input;
- special tokens required by the model;
- an [attention mask]({{< relref "knowledge-base/glossary/attention-mask/index.md" >}}) that distinguishes usable positions from padding;
- offsets connecting token positions back to character spans in the original text.

Offsets are especially useful when a long document is encoded once and token states must later be assigned to smaller chunks. Special-token and whitespace conventions differ across libraries, so inspect the actual output rather than guessing from the displayed text.

## A boundary check worth running

Before processing a full dataset, print the tokens, token identifiers, offsets, and reconstructed text for a few difficult inputs. Include punctuation, accented or non-Latin characters, long scientific names, and leading spaces. Also check the exact model revision because a different tokenizer or vocabulary can change every downstream boundary.

## What tokenization does not establish

- A token is not necessarily a complete word, morpheme, or biological concept.
- Token identifiers have no useful numeric order. Identifier 500 is not “larger” in meaning than identifier 20.
- Decoding one token at a time may not reproduce the exact visible spacing of the full decoded sequence.
- Equal token counts across models do not imply equal information content.

See also: [embedding]({{< relref "knowledge-base/glossary/embedding/index.md" >}}), [attention mask]({{< relref "knowledge-base/glossary/attention-mask/index.md" >}}), [mean pooling]({{< relref "knowledge-base/glossary/mean-pooling/index.md" >}}), [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}).
