---
title: "Forward hook"
slug: "forward-hook"
summary: "A callback registered on a neural-network layer that runs during each forward pass to inspect or replace that layer's activation without changing the layer's saved weights."
draft: false
pro_reviewed: true
og_image: "forward-hook-card.png"
og_image_alt: "A layer computes output [2.0, -1.0] from input [1.0, 2.0]. Without a hook, the next layer receives [2.0, -1.0]. With a forward hook that adds 0.5 times [4.0, 2.0], the next layer receives [4.0, 0.0] while the layer's weights stay unchanged."
---

A **forward hook** is a callback function attached to a neural-network layer (called a **module** in deep-learning libraries such as PyTorch) that runs automatically every time a **forward pass** sends input data through that layer to compute an output. By default in PyTorch, a forward hook receives the module, a tuple of its positional inputs, and its computed output (when registered with `with_kwargs=True`, it also receives a dictionary of keyword arguments). That output is often an **activation**, a multidimensional array of numbers (**tensor**) produced at runtime, though a module may instead return a tuple or another documented structure. The hook can record the output for inspection or return a compatible replacement for downstream computation, without editing the model's source code or, by itself, changing its saved **parameters** (the learned weights and biases stored in the layer).

In a [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), a neural network built from repeated processing blocks, layers are stacked in sequence along a [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), the running vector representation passed from one block to the next. Let \(\ell\) denote a layer's index, let \(d\) be the number of numbers in each vector, let \(\mathbb{R}^{d}\) denote the space of \(d\)-dimensional real-valued vectors, and let \(f_{\ell}\) be the transformation computed by layer \(\ell\). Normally, layer \(\ell\) maps its input vector \(\vec{u}_{\ell}\in\mathbb{R}^{d}\) to an output activation vector

\[
\vec{h}_{\ell} = f_{\ell}(\vec{u}_{\ell}),
\]

and passes \(\vec{h}_{\ell}\) directly to layer \(\ell+1\). When a forward hook function \(g\) is registered on layer \(\ell\), the layer still computes \(\vec{h}_{\ell} = f_{\ell}(\vec{u}_{\ell})\) first, and then the hook runs immediately afterward to produce the post-hook output \(\tilde{\vec{h}}_{\ell}\) (where the tilde marks the hook-modified activation):

\[
\tilde{\vec{h}}_{\ell} = g(\vec{u}_{\ell},\ \vec{h}_{\ell}).
\]

If \(g\) neither mutates \(\vec{h}_{\ell}\) in place nor returns a replacement tensor and instead returns `None`, PyTorch passes the original \(\vec{h}_{\ell}\) onward to layer \(\ell+1\). Returning `None` alone does not enforce read-only behavior, because an in-place modification of \(\vec{h}_{\ell}\) inside the hook still affects downstream layers; an inspection-only hook must avoid mutating the tensor in place. If \(g\) returns a new tensor \(\tilde{\vec{h}}_{\ell}\) of the same shape, layer \(\ell+1\) and every downstream layer receive \(\tilde{\vec{h}}_{\ell}\) instead (an **activation intervention**).

**Activation steering** is a common intervention implemented with a forward hook. Given a fixed steering direction vector \(\vec{v}\in\mathbb{R}^{d}\) and a scalar steering strength \(\alpha\in\mathbb{R}\), the hook adds \(\alpha\vec{v}\) to the layer's output:

\[
\tilde{\vec{h}}_{\ell} = \vec{h}_{\ell} + \alpha\vec{v}.
\]

{{< reference-figure
  src="forward-hook.svg"
  alt="A layer computes output [2.0, -1.0] from input [1.0, 2.0]. Without a hook, the next layer receives [2.0, -1.0]. With a forward hook that adds 0.5 times [4.0, 2.0], the next layer receives [4.0, 0.0] while the layer's weights stay unchanged."
  caption="Toy two-number activation vector passing through a layer. In both rows, the layer applies the same fixed weights to layer input [1.0, 2.0] and produces layer output [2.0, -1.0]. In the upper row, no hook is registered, so the next layer receives [2.0, -1.0]. In the lower row, a forward hook adds steering strength 0.5 times steering vector [4.0, 2.0], adding [2.0, 1.0] so the next layer receives steered output [4.0, 0.0]. Removing the hook handle restores the upper path immediately because the layer's saved weights were never modified. This two-dimensional schematic illustrates execution order and arithmetic; it is not a measured Transformer layer."
>}}

## Worked example

Suppose a two-dimensional layer \(\ell\) takes input \(\vec{u}_{\ell} = [1.0,\ 2.0]\) and its weights produce the output activation

\[
\vec{h}_{\ell} = f_{\ell}(\vec{u}_{\ell}) = [2.0,\ -1.0].
\]

Register a steering hook on layer \(\ell\) with direction \(\vec{v} = [4.0,\ 2.0]\) and steering strength \(\alpha = 0.5\). During the forward pass:

1. Layer \(\ell\) computes its normal output \(\vec{h}_{\ell} = [2.0,\ -1.0]\).
2. The hook computes the offset \(\alpha\vec{v} = 0.5\times [4.0,\ 2.0] = [2.0,\ 1.0]\).
3. The hook returns the steered activation
   \[
   \tilde{\vec{h}}_{\ell} = [2.0 + 2.0,\ -1.0 + 1.0] = [4.0,\ 0.0].
   \]
4. Layer \(\ell+1\) receives \([4.0,\ 0.0]\) as its input. When the hook is removed, the same input \(\vec{u}_{\ell} = [1.0,\ 2.0]\) again sends \([2.0,\ -1.0]\) to layer \(\ell+1\).

In PyTorch, `register_forward_hook` attaches the callback and returns a handle whose `.remove()` method detaches it:

```python
import torch
from torch import nn

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[2.0, 0.0], [1.0, -1.0]]))

v = torch.tensor([4.0, 2.0])
alpha = 0.5

def steer_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
    return output + alpha * v

u = torch.tensor([1.0, 2.0])
unsteered = layer(u)            # tensor([ 2., -1.])

handle = layer.register_forward_hook(steer_hook)
try:
    steered = layer(u)          # tensor([ 4.,  0.])
finally:
    handle.remove()

restored = layer(u)             # tensor([ 2., -1.])
```

Wrapping evaluation in a `try ... finally` block ensures the hook is removed even if an exception occurs; leaving a handle attached across later calls silently applies the hook to every subsequent forward pass on that module instance.

## Variants and execution details

- **Forward pre-hook (`register_forward_pre_hook`):** Runs *before* the layer computes \(f_{\ell}(\vec{u}_{\ell})\) and can inspect or replace the input \(\vec{u}_{\ell}\), whereas a standard forward hook runs *after* \(f_{\ell}(\vec{u}_{\ell})\) and sees the output \(\vec{h}_{\ell}\).
- **Backward hook (`register_full_backward_hook`):** Runs during **backpropagation** (the reverse pass that computes **gradients**, or derivatives of a loss with respect to activations and weights) and receives tuples containing gradients with respect to the module's inputs and outputs. It may inspect both; to alter gradient propagation through this API, it can return a replacement tuple for the input gradients (it does not support returning replacement output gradients). Forward-only inference loops do not run backpropagation and therefore do not trigger backward hooks.
- **Multiple hooks on one module:** By default, module-specific forward hooks run in the order they were registered, and each hook receives any non-`None` output returned by the preceding hook. A hook registered with `prepend=True` is placed at the front of that module's queue, and global module forward hooks run before module-specific hooks.
- **Tuple outputs and key-value caches:** In many Transformer implementations (such as Hugging Face `transformers`), a decoder block returns a tuple whose first element is the hidden-state tensor `(batch, sequence_length, hidden_size)` and whose remaining elements hold attention cache states or weights; inspect the model's documented output structure rather than assuming fixed tuple positions, and reconstruct the tuple when replacing its hidden-state tensor. When key-value caching is enabled, the initial prompt pass exposes all prompt positions while each later cached step usually exposes only the newly computed position. A position-wise operation such as adding a fixed steering vector \(\alpha\vec{v}\) can be applied consistently in both modes, whereas a hook that depends on whole-sequence statistics needs explicit retained state or full recomputation if cached and uncached evaluations are to agree.

## What a forward hook does and does not do

Recording an activation without mutating it leaves the computed output unchanged. Returning a replacement changes the runtime value used by downstream layers while the hook handle remains attached, but the hook call does not by itself write the module's parameters; in the inference example above, which performs no optimizer step, `model.state_dict()` still holds the original unsteered weights. (During training, a replacement output can alter the loss and gradients and thereby change a later optimizer update, and because a hook executes arbitrary Python code it could also mutate parameters or buffers if explicitly written to do so.) Because registered hooks live on the module in host memory (for example, in `module._forward_hooks` in PyTorch), they are visible to host-level runtime inspection even when the text emitted by the model looks ordinary. In the [November note on hiding bits and logic in a language-model feedback loop]({{< relref "posts/2026/11/bits-in-language-model-dynamics-short-read/index.md" >}}), a forward hook on an internal Transformer block applies the steering push \(\alpha\vec{v}\) on each step of a [soft-token feedback]({{< relref "knowledge-base/glossary/soft-token-feedback/index.md" >}}) loop.

See also: [Transformer]({{< relref "knowledge-base/glossary/transformer/index.md" >}}), [residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}), [soft-token feedback]({{< relref "knowledge-base/glossary/soft-token-feedback/index.md" >}}), [logit lens]({{< relref "knowledge-base/glossary/logit-lens/index.md" >}}).
