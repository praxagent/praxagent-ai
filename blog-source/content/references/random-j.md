---
title: "Random-J"
slug: "random-j"
summary: "A family of randomized Jacobian-lens controls; each construction states which scale or geometry it preserves."
pro_reviewed: true
---

**Random-J** is a family of randomized transports used as null controls for a fitted [Jacobian lens]({{< relref "jacobian-lens.md" >}}). It asks whether a result depends on the fitted input-to-output alignment, or whether many matrices with similar coarse geometry would look good under the same statistic.

There is no single canonical random-J. The name must be accompanied by what was preserved:

- **Dimension- and norm-matched:** draw an i.i.d. random matrix, then rescale it to the fitted matrix's Frobenius norm. This controls size, but not conditioning or singular spectrum.
- **Spectrum-matched:** if \(J=U\Sigma V^\top\), replace its singular-vector alignment with random orthogonal bases while retaining \(\Sigma\), schematically \(J_{\text{rand}}=Q_L\Sigma Q_R^\top\). This preserves all singular values while scrambling which input directions map to which output directions.
- **Signed axis permutation:** set \(J_{\text{rand}}=P_LJP_R\), where \(P_L\) and \(P_R\) are signed permutation matrices. This preserves the singular values and the multiset of absolute entry values while scrambling coordinate alignment.
- **Entrywise sign randomization:** set \(J_{\text{rand}}=J\odot S\) for a random \(\{-1,+1\}\) sign matrix. This preserves every entry magnitude and the Frobenius norm, but generally **not** the singular spectrum.

The stronger the matching, the narrower the claim available when the fitted lens wins. Beating a norm-matched draw shows more than matrix scale; beating a spectrum-matched draw shows more than the singular-value profile. Neither comparison proves that the fitted readout is the model's causal mechanism.

{{< reference-figure src="references/glossary/lens-transports.svg" alt="Random-J replaces the fitted Jacobian transport with a named randomized transport while keeping the selected state, output head, probe, search rule, and rank statistic matched." caption="Figure structure: one selected residual state branches through identity, a corpus-derived Jacobian, and a named Random-J construction. The Random-J lane must state which scale or geometry it preserves. Every lane uses the same final normalization, output head, probe tokens, search rule, and summary statistic. A fitted lens beating Random-J supports alignment beyond what that particular null preserves; it does not establish a causal mechanism or rule out an already-strong logit lens." >}}

## Run the whole analysis under the null

Randomization must happen *before* every choice that can make the reported statistic look favorable. If the result takes the [best-rank]({{< relref "best-rank.md" >}}) across many layers and positions, each random seed must also take its best rank across those same layers and positions.

```python
real = evaluate(J, prompts, layers, positions, probes)

null = []
for seed in frozen_seeds:
    J_random = spectrum_matched_scramble(J, seed=seed)
    null.append(evaluate(J_random, prompts, layers, positions, probes))
```

Use multiple seeds and report the null distribution, not only its most convenient member. Five random matrices can expose an obvious failure, but “the fitted lens beat all five” is not automatically a well-powered p-value, especially after searching many probes.

If random-J ranks your probe as well as the fitted lens, you do not have a lens-specific finding.

The reverse needs equal care: a win over random-J supports the fitted alignment under the tested design. It does not remove [base-rate bias]({{< relref "base-rate-bias.md" >}}), prompt leakage, or an already-strong [logit lens]({{< relref "logit-lens.md" >}}) baseline.

See also: [logit lens]({{< relref "logit-lens.md" >}}), [Jacobian lens]({{< relref "jacobian-lens.md" >}}).
