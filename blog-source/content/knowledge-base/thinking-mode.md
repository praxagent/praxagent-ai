---
title: "Thinking on / off"
slug: "thinking-mode"
summary: "The template-controlled reasoning protocols used in these notes: reasoning enabled, or disabled so generation begins after a pre-closed local think block."
pro_reviewed: true
aliases:
  - /references/thinking-mode/
  - /references/thinking-on-off/
  - /references/thinking-on/
  - /references/thinking-off/
---

**Thinking on / off** is how these notes talk about Qwen3.5's reasoning mode.

- **Thinking on:** the model is prompted through its chat template to use its reasoning path before the answer. The path need not be long. Some local runtimes render it inside `<think>…</think>`; hosted systems may expose it separately, summarize it, or hide it.
- **Thinking off:** the model's supported template or API disables that reasoning path. In the local template used by these notes, this is represented by a pre-closed empty think block so the first newly generated token can be the answer.

The literal tags are **model- and template-specific**, not a universal switch. Manually appending `<think></think>` to an arbitrary raw prompt may create ordinary text rather than reproduce the intended protocol. The exact tokenizer, chat-template revision, `enable_thinking` flag, system/user messages, and generation prefix are part of the experimental condition.

## Reproducible template check

```python
messages = [{"role": "user", "content": "Answer yes or no: ..."}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Only if this template documents the option.
)

print(repr(prompt[-200:]))  # Record the actual serialized suffix.
```

For the template convention used here, an abbreviated comparison looks like:

| Mode | Serialized generation prefix | Expected path in this local template |
| --- | --- | --- |
| On | assistant prefix that opens reasoning | reasoning-channel content, then an answer |
| Off | assistant prefix with an already closed empty reasoning block | generation begins in the answer channel and can answer directly |

Always inspect the rendered prompt rather than inferring it from the flag name. A library upgrade can change serialization while leaving experiment code apparently unchanged.

Results can differ by mode (for example, refusal-to-lie and forced-choice self-sacrifice in the pressure note). When a claim depends on mode, the note states which one was used.

Thinking on and off are different inference protocols, not merely two views of one fixed hidden computation. Turning reasoning off changes the input prefix and the computation that follows; an observed answer difference therefore supports a mode-dependent behavioral claim, not a claim that a private chain of thought has been faithfully recovered.

See also: [greedy continuation]({{< relref "greedy-continuation.md" >}}), [sampling]({{< relref "sampling.md" >}}).
