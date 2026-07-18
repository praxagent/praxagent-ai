---
title: "Dtype"
slug: "dtype"
summary: "The data type (dtype) of a tensor or array: the storage format shared by every element, such as float32, bfloat16, or int8. A dtype label names storage, not the whole arithmetic path."
draft: true
pro_reviewed: false
aliases:
  - /references/dtype/
  - /references/data-type/
---

A **dtype** (data type) is the storage format shared by every element of a
tensor or array: how many bits each element occupies and how those bits are
interpreted as a number. In PyTorch and NumPy the dtype is a property of the
whole tensor, not of individual elements.

| Dtype | Bytes per element | Interpretation |
|---|---:|---|
| `float32` (FP32) | 4 | 32-bit floating point |
| `float16` (FP16) | 2 | 16-bit floating point (IEEE 754 binary16) |
| `bfloat16` (BF16) | 2 | 16-bit brain floating point |
| `int64` | 8 | 64-bit signed integer (default for indices) |
| `int8` | 1 | 8-bit signed integer (common in quantized inference) |
| `bool` | 1 | true/false mask element |

```python
import torch

t = torch.tensor([1.0, 2.0])       # default floating dtype: float32
print(t.dtype)                     # torch.float32
h = t.to(torch.bfloat16)           # cast: convert to another dtype
print(h.dtype, h.element_size())   # torch.bfloat16 2 (bytes per element)
```

**Casting** converts a tensor to another dtype, rounding each element onto the
target format's representable grid. Casting to a shorter floating dtype can
change values; see [BF16, FP16, and FP32]({{< relref "bf16.md" >}}) for how the
floating-point dtypes trade range against precision.

## What a dtype label does not establish

A dtype names **storage**, not the full arithmetic path. In particular:

- It does not name the multiplication or accumulation precision. GPU
  [matmul]({{< relref "matmul.md" >}}) kernels commonly multiply FP16 or BF16
  operands while accumulating in FP32.
- It does not name the kernel or hardware path (Tensor Cores versus
  general-purpose lanes, TF32 rounding of FP32 inputs).
- It does not make results bit-reproducible or comparable across devices or
  library versions.

The Deep Dive
[BF16, FP16, and FP32: Precision, Range, Swamping, and Determinism]({{< relref "knowledge-base/deep-dives/bf16-fp16-fp32/index.md" >}})
works through these distinctions with runnable demos and receipts.

See also: [BF16, FP16, and FP32]({{< relref "bf16.md" >}}),
[matmul]({{< relref "matmul.md" >}}).
