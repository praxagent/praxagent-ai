---
title: "Thinking on / off"
slug: "thinking-mode"
summary: "Qwen3.5's reasoning mode: long <think> deliberation before the answer (on), or a pre-closed empty think block so the first generated token is the answer (off)."
aliases:
  - /references/thinking-on-off/
  - /references/thinking-on/
  - /references/thinking-off/
---

**Thinking on / off** is how these notes talk about Qwen3.5's reasoning mode.

- **Thinking on:** the model writes a long `<think>…</think>` block before the visible answer. Useful when you care what it deliberates, and costly when you need a one-token commit.
- **Thinking off:** a pre-closed empty think block so the first generated token is the answer. Useful for forced one-word probes and for reading the output head without a long preamble.

Results can differ by mode (for example, refusal-to-lie and forced-choice self-sacrifice in the pressure note). When a claim depends on mode, the note states which one was used.

See also: [greedy continuation]({{< relref "greedy-continuation.md" >}}), [sampling]({{< relref "sampling.md" >}}).
