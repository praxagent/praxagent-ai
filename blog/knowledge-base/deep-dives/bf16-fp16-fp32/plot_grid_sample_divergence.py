#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.8",
# ]
# ///
"""Render the cross-device grid_sample divergence figure from the receipt.

Run from this Deep Dive's content directory:

    uv run plot_grid_sample_divergence.py

Reads receipts/grid-sample-divergence.json and writes:

* fig-grid-sample-divergence.png (page bundle)
* ../../../../static/knowledge-base/deep-dives/bf16-grid-sample-divergence.png

Idle-arm values are plotted as solid lines; the background-matmul arm is
plotted as thin dashed lines in the same color to show the overlap without
implying a load effect. The CPU control is bit-stable (zero divergence), so
it appears only in the left panel at zero and is stated in the caption for
the log-scale right panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STATIC = HERE.parents[3] / "static" / "knowledge-base" / "deep-dives"
RECEIPT = HERE / "receipts" / "grid-sample-divergence.json"
PNG_DPI = 200

DEVICE_STYLE = {
    "t4": ("#4B6787", "Tesla T4 (Turing, 7.5)"),
    "a100": ("#A67C52", "A100 (Ampere, 8.0)"),
    "l4": ("#6F8D5E", "L4 (Ada, 8.9)"),
    "rtx_pro_6000_blackwell": ("#8E5A8A", "RTX PRO 6000 (Blackwell, 12.0)"),
    "cpu": ("#5A544C", "CPU control"),
}


def main() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    resolutions = receipt["protocol"]["output_resolutions"]
    total_coords = receipt["protocol"]["total_coordinates"]
    devices = receipt["devices"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for key, (color, label) in DEVICE_STYLE.items():
        device = devices[key]
        idle = device["idle"]
        load = device["background_matmul"]

        ax1.plot(
            resolutions,
            idle["max_drift_coords"],
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
        ax1.plot(
            resolutions,
            load["max_drift_coords"],
            color=color,
            linewidth=1,
            linestyle="--",
            alpha=0.6,
        )

        # Right panel: log scale, so zero-divergence CPU points are omitted.
        idle_pts = [
            (r, d) for r, d in zip(resolutions, idle["max_div"]) if d > 0
        ]
        load_pts = [
            (r, d) for r, d in zip(resolutions, load["max_div"]) if d > 0
        ]
        if idle_pts:
            ax2.plot(
                [p[0] for p in idle_pts],
                [p[1] for p in idle_pts],
                marker="o",
                color=color,
                linewidth=2,
                label=label,
            )
        if load_pts:
            ax2.plot(
                [p[0] for p in load_pts],
                [p[1] for p in load_pts],
                color=color,
                linewidth=1,
                linestyle="--",
                alpha=0.6,
            )

    # ULP reference lines: 2^-23 and 2^-16 for orientation.
    for exponent in (-23, -20, -17, -14):
        ax2.axhline(2.0**exponent, color="#C4B8A8", linewidth=0.8, zorder=0)
        ax2.annotate(
            f"$2^{{{exponent}}}$",
            xy=(resolutions[-1], 2.0**exponent),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=8,
            color="#7F786D",
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Output grid resolution (fixed 32x32 input)")
    ax1.set_ylabel(f"Max pairwise drifting coordinates (of {total_coords})")
    ax1.set_title("Coordinate instability (30 trials, all pairs)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Output grid resolution (fixed 32x32 input)")
    ax2.set_ylabel("Max pairwise FP32 divergence")
    ax2.set_title(
        "Divergence magnitude (log scale;\nCPU control is exactly zero, omitted)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "grid_sample backward: run-to-run divergence across five stacks\n"
        "(solid = idle arm, dashed = background-matmul arm; "
        "PyTorch 2.11.0, CUDA 12.8)",
        y=1.02,
    )
    fig.tight_layout()

    bundle_png = HERE / "fig-grid-sample-divergence.png"
    fig.savefig(bundle_png, dpi=PNG_DPI, bbox_inches="tight")
    STATIC.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        STATIC / "bf16-grid-sample-divergence.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Wrote {bundle_png} and static copy.")


if __name__ == "__main__":
    main()
