---
title: "BF16, FP16, and FP32"
slug: "bf16"
summary: "Floating-point formats for model arithmetic: 32-bit floating point (FP32), standard 16-bit floating point (FP16), and bfloat16 (BF16), and how their bit layouts trade range against precision."
draft: false
pro_reviewed: false
---

Neural-network code stores numbers in **floating-point formats**: a sign bit,
an exponent that largely controls range, and a significand that controls
precision. The stored low-order portion is the fraction field, informally also
called the mantissa. This page names three common
[data types (**dtypes**)]({{< relref "knowledge-base/glossary/dtype/index.md" >}}):

- **FP32**: 32-bit floating point (also called float32 or F32)
- **FP16**: 16-bit floating point (also called float16 or F16; IEEE 754 binary16)
- **BF16**: bfloat16 (brain floating point)

When these notes say **BF16 activations**, they mean intermediate tensors
(multidimensional arrays) produced during a model's forward computation that
have BF16 storage dtype at the observation or write point.

| Name | Bits | Exponent bits | Trailing significand bits | Typical role |
|---|---:|---:|---:|---|
| **FP32** | 32 | 8 | 23 | Reference / high-precision math |
| **FP16** | 16 | 5 | 10 | Mixed precision; narrower range |
| **BF16** | 16 | 8 | 7 | Activations / [matmuls]({{< relref "knowledge-base/glossary/matmul/index.md" >}}); FP32-like range, less precision |

{{< reference-figure
  src="knowledge-base/glossary/fp-bit-layouts.svg"
  alt="Bit layouts left-aligned from the sign bit. FP16 uses 1 sign, 5 exponent, and 10 mantissa bits. BF16 uses 1 sign, 8 exponent, and 7 mantissa bits. FP32 uses 1 sign, 8 exponent, and 23 mantissa bits."
  caption="BF16 matches FP32's 8-bit exponent and truncates the mantissa. FP16 uses the same 16-bit total width as BF16 but spends more bits on the mantissa and fewer on the exponent."
>}}

**BF16 is not "FP16 with a different name."** Same width as FP16; same exponent
width as FP32; much shorter mantissa. Tiny increments can be lost when added to
larger BF16 values (**swamping**).

For GPU Tensor Core support, PyTorch cast demos, the accumulation-swamping
chart, overflow examples, and cuBLAS determinism notes, see the Deep Dive
[BF16, FP16, and FP32: Precision, Range, Swamping, and Determinism]({{< relref "knowledge-base/deep-dives/bf16-fp16-fp32/index.md" >}}).

See also: [matmul]({{< relref "knowledge-base/glossary/matmul/index.md" >}}),
[root mean square (RMS)]({{< relref "knowledge-base/glossary/rms/index.md" >}}),
[residual stream]({{< relref "knowledge-base/glossary/residual-stream/index.md" >}}).
