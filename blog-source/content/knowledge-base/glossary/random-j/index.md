---
title: "Random-J"
slug: "random-j"
summary: "A family of randomized Jacobian-lens controls; each construction states which scale or geometry it preserves."
pro_reviewed: true
---

**Random-J** is a family of randomized transport controls for a fitted [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}). It asks whether a result depends on the fitted input-to-output alignment, or whether matrices with similar coarse geometry also look good under the same statistic and prespecified sampling law.

There is no single canonical random-J. The name must be accompanied by what was preserved and by the complete sampling law: the entry distribution for an independent draw, the distribution over orthogonal bases, signed permutations, or signs, the seed policy, whether a separate map is drawn per layer, and how draws are coupled across matched prompt conditions.

- **Dimension- and norm-matched:** draw matrix entries independently from a stated distribution, then rescale the matrix to the fitted matrix's Frobenius norm. This controls size, but not conditioning or singular spectrum.
- **Spectrum-matched:** if \(J=U\Sigma V^\top\), replace its singular-vector alignment with orthogonal bases sampled from a stated distribution while retaining \(\Sigma\), schematically \(J_{\text{rand}}=Q_L\Sigma Q_R^\top\). This preserves all singular values while scrambling which input directions map to which output directions.
- **Signed axis permutation:** set \(J_{\text{rand}}=P_LJP_R\), where \(P_L\) and \(P_R\) are signed permutation matrices. This preserves the singular values and the multiset of absolute entry values while scrambling coordinate alignment.
- **Entrywise sign randomization:** set \(J_{\text{rand}}=J\odot S\) for a random \(\{-1,+1\}\) sign matrix. This preserves every entry magnitude and the Frobenius norm, but generally **not** the singular spectrum.

The stronger the matching, the narrower the comparison. Performance relative to a prespecified distribution of norm-matched draws tests whether the fitted result exceeds what matrix scale alone typically produces under that reference construction; spectrum-matched draws analogously control for the singular-value profile. A win over one draw is not sufficient, and neither comparison proves that the fitted readout is the model's causal mechanism.

{{< reference-figure src="knowledge-base/glossary/lens-transports.svg" alt="Random-J replaces the fitted Jacobian transport with a named randomized transport while keeping the selected state, output head, probe, search rule, and rank statistic matched." caption="Figure structure: one selected residual state branches through identity, a corpus-derived Jacobian, and a named Random-J construction. The Random-J lane must state which scale or geometry it preserves. Every lane uses the same final normalization, output head, probe tokens, search rule, and summary statistic. A fitted lens outperforming a prespecified Random-J reference distribution supports alignment beyond what that construction typically produces; it does not establish a causal mechanism, create a calibrated hypothesis test automatically, or rule out an already-strong logit lens." >}}

## Run the whole analysis under the randomized reference

Randomization must happen *before* every choice that can make the reported statistic look favorable. If the result takes the [best-rank]({{< relref "knowledge-base/glossary/best-rank/index.md" >}}) across many layers and positions, each random seed must also take its best rank across those same layers and positions.

```python
real = evaluate(J, prompts, layers, positions, probes)

reference = []
for seed in frozen_seeds:
    J_random = spectrum_matched_scramble(J, seed=seed)
    reference.append(evaluate(J_random, prompts, layers, positions, probes))
```

Use multiple seeds and report the randomized reference distribution, not only its most convenient member. Five random matrices can expose an obvious failure, but “the fitted lens beat all five” is not automatically a well-powered p-value, especially after searching many probes.

If the fitted result is not better than the prespecified Random-J reference distribution under the same statistic, the analysis does not show fitted-alignment specificity relative to that construction. A random draw that matches or beats the fitted lens should be reported, but the comparison should use the full reference distribution and frozen randomization rule rather than treating one draw as dispositive.

Call this a calibrated null distribution or derive a randomization p-value only when the stated null makes the fitted map exchangeable with the randomized maps, or another valid testing argument supplies the required reference law.

The reverse needs equal care: a fitted result that exceeds the prespecified Random-J reference distribution supports a relative alignment advantage under the tested design. It does not remove [base-rate bias]({{< relref "knowledge-base/glossary/base-rate-bias/index.md" >}}), prompt leakage, or an already-strong [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}) baseline.

See also: [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}), [Jacobian lens]({{< relref "knowledge-base/glossary/jacobian-lens/index.md" >}}).
