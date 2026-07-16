---
title: "p-value"
slug: "p-value"
summary: "In these notes, p is usually a sign-test or Wilcoxon p-value across paraphrase pairs: how often the predicted direction would appear by chance if there were no effect."
aliases:
  - /references/p/
  - /references/pvalue/
  - /references/sign-test/
---

When a Research Note writes **p=0.004**, it is answering: *if there were no real effect, how often would a result this extreme (or more) show up by chance?*

**How we usually compute it here.** Constructs are scored as **paired paraphrase tests**: each of *n* wordings gives a pressure/control (or self/other) pair. We count how many pairs move the predicted way, then report a **sign test** p-value (and often a [Wilcoxon]({{< relref "wilcoxon.md" >}}) signed-rank p-value on the paired differences). Example: **14/16** pairs in the predicted direction with **p=0.004** means that under a fair coin (no effect), seeing at least 14 wins in 16 trials is unlikely.

**With n=10 the test is conservative.** Only extreme direction counts clear usual thresholds (about **10/10 → p≈0.002**, **9/10 → p≈0.02**; **8/10** is already not significant). A soft majority (6–4, 7–3) correctly fails. That matches the job of the paraphrase battery: show the effect is not one lucky sentence.

**What it is not.** It is not the probability that the hypothesis is true, not a measure of effect size (that lives in the rank gap), and not a population estimate. The paraphrases are a frozen convenience set on *this* battery, not [i.i.d.]({{< relref "iid.md" >}}) draws from “all threats in the wild.” It is also not a license to ignore design flaws: a tiny *p* with a confounded arm is still a confounded arm; that is why the pressure note retracts mismatched human/log comparisons even when ranks look striking.

**Rule of thumb for reading these notes.** Prefer the triple: *direction count* (e.g. 14/16), *p*, and the **controls** ([logit lens]({{< relref "logit-lens.md" >}}), [random-J]({{< relref "random-j.md" >}})). A low *p* that also appears under the logit lens is usually not a [Jacobian-lens]({{< relref "jacobian-lens.md" >}})-specific claim. Generalization beyond the battery needs more models and more independently authored templates.

See also: [Wilcoxon]({{< relref "wilcoxon.md" >}}), [best-rank]({{< relref "best-rank.md" >}}), [prompt echo]({{< relref "prompt-echo.md" >}}).
