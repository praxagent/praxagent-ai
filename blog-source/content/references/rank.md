---
title: "Rank"
slug: "rank"
summary: "Position of a target token ID in a scored vocabulary list (1 = highest-scoring). Lower is stronger."
aliases:
  - /references/vocabulary-rank/
---

**Rank** is the position of a target **token ID** in a vocabulary ordered by readout score: rank 1 is highest-scoring, so lower numerical rank is stronger.

For a target score \(s_t\), a simple competition rank is

\[
r_t = 1 + \#\{j : s_j > s_t\}.
\]

```python
def token_rank(scores, token_id):
    target = scores[token_id]
    return 1 + sum(score > target for score in scores)
```

This definition gives tied tokens the same rank. Implementations should state their tie convention, although exact floating-point ties are uncommon.

## Words are not always tokens

A displayed word may correspond to several possible token IDs:

- `Paris` and ` Paris` can be different tokens.
- Case variants may split (`self`, `Self`, `SELF`).
- A word such as `decommissioned` may break into several subword tokens.
- Some model vocabularies include padded or reserved entries that are never generated normally.

The probe specification should say which token IDs count and how they are combined. Our common convention is to pre-register sensible single-token variants and take the best variant **in every arm**. A multi-token concept needs a different score (for example, a sequence log-probability); silently ranking only its first piece is misleading.

## Scale intuition

On [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), the padded output vocabulary has 248,320 entries. Rank 500 lies in roughly the top 0.20% of entries:

```python
100 * 500 / 248_320  # 0.201...
```

That percentile is descriptive, not a significance level. Vocabulary entries are not exchangeable, and absolute rank is heavily affected by [base-rate bias]({{< relref "base-rate-bias.md" >}}).

Notes often report [best-rank]({{< relref "best-rank.md" >}}), the minimum over a pre-specified search grid. Compare pressure to a matched control and apply the identical statistic to [logit lens]({{< relref "logit-lens.md" >}}) and [random-J]({{< relref "random-j.md" >}}).

See also: [unembedding]({{< relref "unembedding.md" >}}), [workspace]({{< relref "workspace.md" >}}).
