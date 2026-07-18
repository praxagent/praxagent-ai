#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.2",
#   "matplotlib>=3.8",
# ]
# ///
"""Reproduce the BF16 / FP16 / FP32 dtype demos and refresh the swamping chart.

Run:

    uv run reproduce.py

This writes:

* demo.receipt.json
* fig-accumulation-swamping.png (page bundle)
* ../../../../static/knowledge-base/deep-dives/bf16-accumulation-swamping.png

The chart is a high-resolution PNG rendered by Matplotlib from the same
histories as the Deep Dive (not a hand-traced SVG). It is a teaching demo of
storage-format accumulation, not a GEMM benchmark.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
STATIC = HERE.parents[3] / "static" / "knowledge-base" / "deep-dives"
N_STEPS = 10_000
STEP = 1e-4
PNG_DPI = 200


def run_precision() -> list[dict[str, float]]:
    rows = []
    for v in (1.0, 1.0 + 2**-12, 1.0 + 2**-10):
        t_fp32 = torch.tensor([v], dtype=torch.float32)
        rows.append(
            {
                "input": float(v),
                "fp32": float(t_fp32.item()),
                "fp16": float(t_fp32.to(torch.float16).item()),
                "bf16": float(t_fp32.to(torch.bfloat16).item()),
            }
        )
    return rows


def run_accumulation() -> tuple[list[float], list[float], list[float]]:
    fp32_sum = torch.tensor([0.0], dtype=torch.float32)
    fp16_sum = torch.tensor([0.0], dtype=torch.float16)
    bf16_sum = torch.tensor([0.0], dtype=torch.bfloat16)
    fp32_history: list[float] = []
    fp16_history: list[float] = []
    bf16_history: list[float] = []
    for _ in range(N_STEPS):
        fp32_sum = (
            fp32_sum + torch.tensor([STEP], dtype=torch.float32)
        ).to(torch.float32)
        fp16_sum = (
            fp16_sum + torch.tensor([STEP], dtype=torch.float16)
        ).to(torch.float16)
        bf16_sum = (
            bf16_sum + torch.tensor([STEP], dtype=torch.bfloat16)
        ).to(torch.bfloat16)
        fp32_history.append(float(fp32_sum.item()))
        fp16_history.append(float(fp16_sum.item()))
        bf16_history.append(float(bf16_sum.item()))
    return fp32_history, fp16_history, bf16_history


def run_overflow() -> dict[str, float | str]:
    large = 1e5
    fp16 = float(torch.tensor([large], dtype=torch.float16).item())
    bf16 = float(torch.tensor([large], dtype=torch.bfloat16).item())
    return {
        "value": large,
        "fp16": "inf" if math.isinf(fp16) else fp16,
        "bf16": bf16,
    }


def write_png(
    fp32_history: list[float],
    fp16_history: list[float],
    bf16_history: list[float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fp32_history, label="FP32 (Float32)", linewidth=2)
    ax.plot(fp16_history, label="FP16 (Half)", linewidth=2, linestyle="--")
    ax.plot(bf16_history, label="BF16 (Bfloat16)", linewidth=2, linestyle=":")
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.5, label="Target (1.0)")
    ax.set_title("Accumulation Swamping: Adding 0.0001 ten thousand times")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Running Sum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    precision_rows = run_precision()
    fp32_h, fp16_h, bf16_h = run_accumulation()
    overflow = run_overflow()
    receipt = {
        "protocol": "cast(s + cast(1e-4)) for 10000 steps; torch CPU dtype casts",
        "n_steps": N_STEPS,
        "step": STEP,
        "final": {
            "fp32": fp32_h[-1],
            "fp16": fp16_h[-1],
            "bf16": bf16_h[-1],
        },
        "precision_rows": precision_rows,
        "overflow": overflow,
        "figure": {
            "file": "bf16-accumulation-swamping.png",
            "dpi": PNG_DPI,
            "generator": "reproduce.py (matplotlib)",
        },
        "hardware_example": {
            "gpu": "NVIDIA A100-SXM4-40GB",
            "native_bf16_tensor_cores": True,
            "note": "Hardware banner from one Colab A100 run; the numeric demos below are dtype casts and do not require that GPU.",
        },
        "torch_version": torch.__version__,
    }
    (HERE / "demo.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    bundle_png = HERE / "fig-accumulation-swamping.png"
    static_png = STATIC / "bf16-accumulation-swamping.png"
    write_png(fp32_h, fp16_h, bf16_h, bundle_png)
    STATIC.mkdir(parents=True, exist_ok=True)
    write_png(fp32_h, fp16_h, bf16_h, static_png)
    # Remove superseded hand-traced SVG chart if present.
    for stale in (
        HERE / "fig-accumulation-swamping.svg",
        STATIC / "bf16-accumulation-swamping.svg",
    ):
        if stale.is_file():
            stale.unlink()
    print("Wrote demo.receipt.json and swamping PNGs.")
    print(
        "Final sums:",
        f"FP32={fp32_h[-1]:.5f}",
        f"FP16={fp16_h[-1]:.5f}",
        f"BF16={bf16_h[-1]:.5f}",
    )


if __name__ == "__main__":
    main()
