#!/usr/bin/env python3
"""Render publication figures from the audited signed-dose calibration summary.

The script consumes the compact, hash-bound ``CALIBRATION_SUMMARY.json`` and
writes SVG, PDF, and 300-DPI PNG versions of every figure.  It intentionally
does not read model tensors or reconstruct unpublished outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1784309640")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


EXPECTED_SUMMARY_SHA256 = (
    "b490b101c112f774ae7bffc9c54294a70b91c874c4cee78eddaa7890446c08f6"
)
FIXED_ARTIFACT_DATE = datetime(2026, 7, 17, tzinfo=timezone.utc)
RUN_LABEL = "signed-dose-a084caa-wl8obvtuq0ax8t-v2 / C9 audit recovery"

NAVY = "#17324D"
BLUE = "#3973AC"
TEAL = "#1B9E77"
ORANGE = "#D95F02"
PURPLE = "#7570B3"
RED = "#C43C39"
GOLD = "#C89118"
GRAY = "#66717E"
LIGHT = "#E8EDF2"

FIGURE_SPECS = {
    "fig-1-linearity-census": {
        "title": "The source edit stayed linear; the model response did not",
        "description": (
            "A paired census of the frozen 24 prompt-by-direction cells compares "
            "cross-dose linearity for the realized source edit, a fixed corpus-average "
            "Jacobian applied to that edit, and the model's actual final state."
        ),
        "alt_text": (
            "Two paired dot plots show all 24 frozen prompt-by-direction cells over the 2%, "
            "3%, and 4% dose panel, using 3% as the anchor. In the left plot, minimum "
            "cosine to the 3% anchor clusters "
            "between 0.993 and 0.996 for the realized source edit and J of the realized edit, "
            "above the 0.95 threshold, then drops to 0.778–0.848 for the actual final state, "
            "below threshold in all 24 cells. In the right plot, maximum dose-normalized RMS "
            "discrepancy from the 3% anchor is "
            "0.089–0.117 for the first two quantities, below the 0.15 ceiling, but 0.572–0.703 "
            "for the actual final state, above threshold in every cell. Faint gray lines connect "
            "the same cell across the three quantities. Exact proportionality is preserved by a "
            "fixed linear map; here the approximately proportional delivered edits also remain "
            "within both numerical bounds after projection, while the downstream model response "
            "does not."
        ),
        "data_sections": ["linearity_rows"],
        "transformation": (
            "Plot every frozen row without inferential aggregation; pair identical prompt and "
            "direction cells across the three prespecified quantities."
        ),
    },
    "fig-2-dose-depth-arc": {
        "title": "A nonlinear response unfolds after the layer-50 edit",
        "description": (
            "The full 43,200-row actual-state arc is summarized by dose and depth, with all "
            "24 final-state cell trajectories retained."
        ),
        "alt_text": (
            "A two-panel dose-by-depth view of the complete downstream response. The upper heatmap "
            "places requested dose from 0.5% to 30% on the horizontal axis and states 50 through 79 "
            "on the vertical axis. Every state-50 cell begins at gain 1 because gain is normalized "
            "to the realized edit. Median gain generally accumulates in later blocks, reaching its "
            "largest values at the smallest doses and generally tapering as dose rises. A dashed "
            "vertical line marks 2%, the first "
            "dose where every cell met the frozen delivery criteria. The lower panel shows all 24 final-state "
            "trajectories as pale lines and their median as a dark line. The median is 1.82 times at "
            "2%, 1.66 at 3%, 1.60 at 4%, 1.55 at 8%, and 1.48 at 30%. The 0.5–1.5% region is shaded "
            "because its delivery gate failed."
        ),
        "data_sections": ["primary_actual_state_arc_rows"],
        "transformation": (
            "For the heatmap, take the fixed-census median across 24 prompt-direction cells within "
            "each requested-dose and state cell. For the profile, plot each state-79 cell and the "
            "fixed-census median; do not use a confidence interval."
        ),
    },
    "fig-3-bf16-delivery-floor": {
        "title": "Observed low-dose delivery boundary under BF16 execution",
        "description": (
            "Requested-versus-realized direction and magnitude fidelity under this BF16 execution "
            "setup are plotted across the full 0.5%–30% scan, together with the exact number of "
            "failed delivery cells."
        ),
        "alt_text": (
            "Three aligned plots show the low-dose delivery transition under BF16 execution in this "
            "model and intervention setup, across requested doses from 0.5% to 30%. Directional "
            "cosine rises sharply from a median 0.961 at 0.5% toward 1.0 and clears "
            "its 0.995 threshold by the eligible panel. Relative RMSE falls from a median 0.287 at "
            "0.5% through its 0.10 ceiling near 1.5–2% and continues toward zero. Pale bands show the "
            "minimum-to-maximum range across the fixed 24-cell census, not uncertainty. The bottom "
            "strip reports the full paired-branch delivery rule: 24 failed cells at 0.5%, 24 at 1%, "
            "18 at 1.5%, and zero from 2% through 30%."
        ),
        "data_sections": ["by_dose"],
        "transformation": (
            "Plot the stored minimum, median, and maximum requested-realized metrics at every nonzero "
            "dose and the exact requested_delivery_failure_count."
        ),
    },
    "fig-4-j-versus-identity": {
        "title": "The released J clears absolute thresholds—but not identity",
        "description": (
            "Source-state profiles separate absolute released-J prediction quality from the stronger "
            "requirement that J add value over identity in both logit and residual space; every map "
            "targets the same post-block-79 state."
        ),
        "alt_text": (
            "Four profiles compare predictions from source-state layers 50 through 78 to the same "
            "post-block-79 target. The left column shows that absolute released-J fixed-token logit "
            "correlation and residual cosine both exceed their frozen thresholds at every source "
            "layer and rise as the source approaches the target. The right column shows released J "
            "minus identity. Logit advantage generally rises above 0.02 after early layers, but "
            "residual advantage remains below the 0.02 margin at 28 of 29 source layers. Its sole "
            "residual pass at layer 78 does not coincide with a logit pass. At source layer 50, "
            "selected in advance for the primary comparison, logit correlation is "
            "0.344 but J minus identity is only 0.011 with a lower bound below zero; residual J minus "
            "identity is slightly negative. No source layer satisfies the identity condition for both "
            "metrics. "
            "Shaded bands are 20,000-replicate prompt-resampling stability intervals over eight fixed "
            "prompts, not population confidence intervals."
        ),
        "data_sections": [
            "readout_transport.fixed_token_logit_delta_pearson",
            "readout_transport.residual_delta_cosine",
        ],
        "transformation": (
            "Plot stored per-layer estimates and 95% prompt-resampling stability bounds for absolute "
            "real J and real-J-minus-identity; retain the prospectively frozen thresholds."
        ),
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Avenir", "Arial", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "axes.edgecolor": "#7B8794",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DCE2E8",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "signed-dose-c9-20260717",
            "pdf.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path, allow_hash_mismatch: bool) -> dict:
    observed = sha256(path)
    if observed != EXPECTED_SUMMARY_SHA256 and not allow_hash_mismatch:
        raise SystemExit(
            "Refusing to plot an unbound summary: "
            f"expected {EXPECTED_SUMMARY_SHA256}, observed {observed}. "
            "Use --allow-hash-mismatch only for deliberate exploratory work."
        )
    with path.open() as handle:
        result = json.load(handle)
    if result.get("status") != "pass":
        raise SystemExit(f"Summary status is not pass: {result.get('status')!r}")
    expected_counts = {
        "linearity_rows": 24,
        "primary_actual_state_arc_rows": 43_200,
        "by_dose": 61,
    }
    for key, expected in expected_counts.items():
        observed_count = len(result[key])
        if observed_count != expected:
            raise SystemExit(f"Unexpected {key} count: expected {expected}, observed {observed_count}")
    if result.get("target_feature_vector_count") != 0 or result.get("target_prompt_render_count") != 0:
        raise SystemExit("Target-blind guard failed: target vectors or prompts are present")
    return result


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        va="bottom",
    )


def save_figure(fig: mpl.figure.Figure, stem: Path, formats: list[str]) -> list[Path]:
    outputs: list[Path] = []
    spec = FIGURE_SPECS[stem.name]
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        kwargs = {"format": fmt, "facecolor": "white"}
        if fmt == "png":
            kwargs["dpi"] = 300
            kwargs["metadata"] = {
                "Title": spec["title"],
                "Description": spec["alt_text"],
                "Software": f"Matplotlib {mpl.__version__}",
                "Source": EXPECTED_SUMMARY_SHA256,
            }
        elif fmt == "svg":
            kwargs["metadata"] = {
                "Title": spec["title"],
                "Description": spec["alt_text"],
                "Creator": f"Matplotlib {mpl.__version__}",
                "Source": EXPECTED_SUMMARY_SHA256,
                "Date": FIXED_ARTIFACT_DATE.isoformat(),
            }
        elif fmt == "pdf":
            kwargs["metadata"] = {
                "Title": spec["title"],
                "Subject": spec["alt_text"],
                "Creator": f"Matplotlib {mpl.__version__}",
                "CreationDate": FIXED_ARTIFACT_DATE,
                "ModDate": FIXED_ARTIFACT_DATE,
            }
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def figure_linearity(summary: dict, output_dir: Path, formats: list[str]) -> list[Path]:
    rows = summary["linearity_rows"]
    categories = ["Realized\nsource edit", "J(realized\nedit)", "Actual final\nstate"]
    cosine_fields = [
        "realized_source_linearity_cosine_min",
        "j_of_realized_linearity_cosine_min",
        "actual_final_linearity_cosine_min",
    ]
    slope_fields = [
        "realized_source_slope_discrepancy_max",
        "j_of_realized_slope_discrepancy_max",
        "actual_final_slope_discrepancy_max",
    ]
    x = np.arange(3)
    rng = np.random.default_rng(20260717)
    jitter = rng.uniform(-0.055, 0.055, len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.35), constrained_layout=True)
    direction_colors = {0: TEAL, 1: PURPLE, 2: ORANGE}
    for row_i, row in enumerate(rows):
        color = direction_colors[int(row["direction"])]
        cos_values = [float(row[field]) for field in cosine_fields]
        slope_values = [float(row[field]) for field in slope_fields]
        for ax, values in zip(axes, [cos_values, slope_values]):
            ax.plot(x + jitter[row_i], values, color="#9AA6B2", alpha=0.24, lw=0.65, zorder=1)
            ax.scatter(
                x + jitter[row_i],
                values,
                s=12,
                color=color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.25,
                zorder=2,
            )

    axes[0].axhline(0.95, color=RED, lw=1.1, ls=(0, (4, 3)), label="Frozen threshold")
    axes[0].set_ylabel("Minimum cosine\nto 3% anchor")
    axes[0].set_ylim(0.75, 1.012)
    axes[0].set_title("Direction stability")
    axes[0].text(1.96, 0.858, "24 / 24 fail", color=ORANGE, ha="center", fontweight="bold")
    axes[0].text(0.5, 0.968, "24 / 24 pass in each", color=TEAL, ha="center", fontweight="bold")

    axes[1].axhline(0.15, color=RED, lw=1.1, ls=(0, (4, 3)), label="Frozen threshold")
    axes[1].set_ylabel("Maximum RMS discrepancy\nto 3% anchor")
    axes[1].set_ylim(0.0, 0.75)
    axes[1].set_title("Magnitude stability")
    axes[1].text(1.96, 0.535, "24 / 24 fail", color=ORANGE, ha="center", fontweight="bold")
    axes[1].text(0.5, 0.125, "24 / 24 pass in each", color=TEAL, ha="center", fontweight="bold")
    legend_handles = [
        Line2D([0], [0], color=RED, lw=1.1, ls=(0, (4, 3)), label="Frozen threshold"),
        *[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=direction_colors[direction],
                markeredgecolor="white",
                markersize=5,
                label=f"Generic direction {direction + 1}",
            )
            for direction in sorted(direction_colors)
        ],
    ]
    axes[0].legend(handles=legend_handles, loc="lower left", ncol=2, columnspacing=0.8)
    axes[1].legend(handles=[legend_handles[0]], loc="upper left")

    for label, ax in zip(["A", "B"], axes):
        panel_label(ax, label)
        ax.set_xticks(x, categories)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "The source edit stayed linear; the model response did not",
        x=0.51,
        y=1.055,
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.51,
        -0.025,
        "All 24 frozen prompt × direction cells; prospective 2%, 3%, and 4% dose panel. Lines pair the same cell.",
        ha="center",
        color=GRAY,
        fontsize=7.8,
    )
    return save_figure(fig, output_dir / "fig-1-linearity-census", formats)


def figure_dose_depth(summary: dict, output_dir: Path, formats: list[str]) -> list[Path]:
    rows = [
        row
        for row in summary["primary_actual_state_arc_rows"]
        if int(row["requested_dose_basis_points"]) > 0
    ]
    doses_bp = sorted({int(row["requested_dose_basis_points"]) for row in rows})
    states = sorted({int(row["state_index"]) for row in rows})
    dose_index = {dose: i for i, dose in enumerate(doses_bp)}
    state_index = {state: i for i, state in enumerate(states)}
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    final_by_cell: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        dose = int(row["requested_dose_basis_points"])
        state = int(row["state_index"])
        value = float(row["central_gain_over_realized_source_rms"])
        grouped[(state, dose)].append(value)
        if state == states[-1]:
            final_by_cell[(str(row["prompt_id"]), int(row["direction"]))][dose] = value
    matrix = np.empty((len(states), len(doses_bp)))
    for state in states:
        for dose in doses_bp:
            matrix[state_index[state], dose_index[dose]] = np.median(grouped[(state, dose)])

    doses_pct = np.asarray(doses_bp, dtype=float) / 100.0
    fig = plt.figure(figsize=(7.4, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.22, 1.0], hspace=0.12)
    ax_heat = fig.add_subplot(grid[0])
    ax_profile = fig.add_subplot(grid[1])

    image = ax_heat.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
        extent=[doses_pct[0] - 0.25, doses_pct[-1] + 0.25, states[0] - 0.5, states[-1] + 0.5],
        vmin=float(np.nanpercentile(matrix, 1)),
        vmax=float(np.nanpercentile(matrix, 99)),
    )
    ax_heat.grid(False)
    ax_heat.set_ylabel("Block output")
    ax_heat.set_xlabel("Requested residual-RMS dose (%)")
    ax_heat.set_title("Median downstream gain over the realized source edit")
    ax_heat.axvline(2.0, color="white", lw=1.1, ls=(0, (4, 3)))
    cbar = fig.colorbar(image, ax=ax_heat, pad=0.015, shrink=0.9)
    cbar.set_label("Central-response RMS / realized-edit RMS")
    panel_label(ax_heat, "A")

    trajectories = []
    for values in final_by_cell.values():
        trajectory = np.asarray([values[dose] for dose in doses_bp], dtype=float)
        trajectories.append(trajectory)
        ax_profile.plot(doses_pct, trajectory, color=BLUE, alpha=0.16, lw=0.65)
    trajectory_matrix = np.asarray(trajectories)
    median = np.median(trajectory_matrix, axis=0)
    ax_profile.plot(doses_pct, median, color=NAVY, lw=2.25, label="Fixed-panel median")
    ax_profile.scatter(doses_pct, median, s=8, color=NAVY, zorder=3)
    ax_profile.axvspan(0.5, 1.5, color=RED, alpha=0.09, label="Delivery gate failed")
    ax_profile.axvline(2.0, color=RED, lw=1.05, ls=(0, (4, 3)))
    ax_profile.set_xlim(0.25, 30.25)
    ax_profile.set_ylim(bottom=0.95)
    ax_profile.set_xlabel("Requested residual-RMS dose (%)")
    ax_profile.set_ylabel("Final gain (× source edit)")
    ax_profile.set_title("Every final-state trajectory (24-cell census)")
    ax_profile.legend(loc="upper right")
    panel_label(ax_profile, "B")
    annotation_offsets = {
        2: (-14, 9),
        3: (2, 9),
        4: (18, 9),
        8: (0, 11),
        30: (0, -13),
    }
    for dose in [2, 3, 4, 8, 30]:
        idx = int(np.argmin(np.abs(doses_pct - dose)))
        ax_profile.annotate(
            f"{median[idx]:.2f}×",
            (doses_pct[idx], median[idx]),
            xytext=annotation_offsets[dose],
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=NAVY,
        )

    fig.suptitle(
        "A nonlinear response unfolds after the layer-50 edit",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )
    return save_figure(fig, output_dir / "fig-2-dose-depth-arc", formats)


def figure_delivery(summary: dict, output_dir: Path, formats: list[str]) -> list[Path]:
    entries = sorted(
        (
            float(dose) * 100.0,
            values,
        )
        for dose, values in summary["by_dose"].items()
        if float(dose) > 0
    )
    x = np.asarray([entry[0] for entry in entries])

    def series(field: str, statistic: str) -> np.ndarray:
        return np.asarray([float(entry[1][field][statistic]) for entry in entries])

    failures = np.asarray([int(entry[1]["requested_delivery_failure_count"]) for entry in entries])
    fig = plt.figure(figsize=(7.4, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.35])
    axes = [fig.add_subplot(grid[i]) for i in range(3)]

    cos_med, cos_min, cos_max = (
        series("requested_realized_cosine", statistic) for statistic in ["median", "min", "max"]
    )
    axes[0].fill_between(x, cos_min, cos_max, color=BLUE, alpha=0.18, label="Min–max over 24 cells")
    axes[0].plot(x, cos_med, color=BLUE, lw=2.0, label="Median")
    axes[0].axhline(0.995, color=RED, lw=1.0, ls=(0, (4, 3)), label="Frozen threshold")
    axes[0].set_ylabel("Requested–realized\ncosine")
    axes[0].set_title("Directional fidelity")
    axes[0].legend(loc="lower right", ncol=3)
    panel_label(axes[0], "A")

    rmse_med, rmse_min, rmse_max = (
        series("requested_realized_relative_rmse", statistic)
        for statistic in ["median", "min", "max"]
    )
    axes[1].fill_between(x, rmse_min, rmse_max, color=ORANGE, alpha=0.18, label="Min–max over 24 cells")
    axes[1].plot(x, rmse_med, color=ORANGE, lw=2.0, label="Median")
    axes[1].axhline(0.10, color=RED, lw=1.0, ls=(0, (4, 3)), label="Frozen threshold")
    axes[1].set_ylabel("Relative RMSE")
    axes[1].set_title("Magnitude fidelity")
    axes[1].legend(loc="upper right", ncol=3)
    panel_label(axes[1], "B")

    colors = np.where(failures > 0, RED, TEAL)
    axes[2].bar(x, failures, width=0.42, color=colors, alpha=0.82)
    axes[2].set_ylim(0, 25)
    axes[2].set_yticks([0, 24])
    axes[2].set_ylabel("Failed\ncells")
    axes[2].set_xlabel("Requested residual-RMS dose (%)")
    axes[2].grid(axis="x", visible=False)
    axes[2].text(0.5, 22.8, "24", ha="center", va="top", color="white", fontsize=7, fontweight="bold")
    axes[2].text(1.0, 22.8, "24", ha="center", va="top", color="white", fontsize=7, fontweight="bold")
    axes[2].text(1.5, 16.8, "18", ha="center", va="top", color="white", fontsize=7, fontweight="bold")
    panel_label(axes[2], "C")

    for ax in axes[:2]:
        ax.axvspan(0.25, 1.75, color=RED, alpha=0.055)
        ax.set_xlim(0.25, 30.25)
        ax.tick_params(labelbottom=False)
    axes[2].set_xlim(0.25, 30.25)

    fig.suptitle(
        "Observed low-dose delivery boundary under BF16 execution",
        fontsize=14,
        fontweight="bold",
        color=NAVY,
    )
    return save_figure(fig, output_dir / "fig-3-bf16-delivery-floor", formats)


def figure_readout(summary: dict, output_dir: Path, formats: list[str]) -> list[Path]:
    metrics = [
        ("fixed_token_logit_delta_pearson", "Fixed-token logit Δ Pearson", 0.25),
        ("residual_delta_cosine", "Residual Δ cosine", 0.10),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.45, 5.8), constrained_layout=True, sharex=True)
    for row_i, (metric_key, label, absolute_threshold) in enumerate(metrics):
        layer_data = summary["readout_transport"][metric_key]["by_readout_layer"]
        layers = np.asarray(sorted(int(layer) for layer in layer_data))
        ordered = [layer_data[str(layer)] for layer in layers]
        for col_i, (field, title, threshold) in enumerate(
            [
                ("absolute_real_j", "Released J against final target", absolute_threshold),
                ("real_j_minus_identity", "Added value over identity", 0.02),
            ]
        ):
            ax = axes[row_i, col_i]
            estimate = np.asarray([float(item[field]["estimate"]) for item in ordered])
            low = np.asarray([float(item[field]["lcb_95"]) for item in ordered])
            high = np.asarray([float(item[field]["ucb_95"]) for item in ordered])
            color = BLUE if col_i == 0 else ORANGE
            ax.axvspan(50.5, 78.5, color=GRAY, alpha=0.055)
            ax.fill_between(layers, low, high, color=color, alpha=0.20, lw=0)
            ax.plot(layers, estimate, color=color, lw=1.8)
            ax.scatter(layers, estimate, color=color, s=9, zorder=3)
            ax.axhline(threshold, color=RED, lw=1.0, ls=(0, (4, 3)))
            if col_i == 1:
                ax.axhline(0, color=GRAY, lw=0.8)
            ax.axvline(50, color=NAVY, lw=1.0, ls=(0, (2, 2)))
            ax.set_title(title if row_i == 0 else "")
            ax.set_ylabel(label)
            ax.set_xlim(49.5, 78.5)
            ax.text(
                50,
                ax.get_ylim()[1],
                " primary",
                color=NAVY,
                va="top",
                ha="left",
                fontsize=7.3,
            )
            panel_label(ax, chr(ord("A") + row_i * 2 + col_i))
    for ax in axes[-1]:
        ax.set_xlabel("J source layer ℓ (target: state 79)")
    axes[0, 1].text(
        64.5,
        axes[0, 1].get_ylim()[0] + 0.012,
        "No layer passes the composite\nidentity condition for both metrics",
        color=ORANGE,
        ha="center",
        fontsize=7.8,
        fontweight="bold",
    )
    fig.suptitle(
        "The released J clears absolute thresholds—but not identity",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.5,
        -0.018,
        "Bands: 20,000-replicate prompt-resampling stability intervals over 8 fixed prompts; not population CIs.",
        ha="center",
        color=GRAY,
        fontsize=7.8,
    )
    return save_figure(fig, output_dir / "fig-4-j-versus-identity", formats)


def write_figure_receipts(summary: dict, source: Path, output_dir: Path, files: list[Path]) -> None:
    rows = summary["linearity_rows"]
    source_sha = sha256(source)
    generator_path = Path(__file__).resolve()
    output_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        output_by_stem[path.stem].append(path)

    selected_doses = {}
    for dose in ["0.005", "0.01", "0.015", "0.02", "0.03", "0.04", "0.08", "0.3"]:
        entry = summary["by_dose"][dose]
        selected_doses[dose] = {
            "requested_delivery_failure_count": entry["requested_delivery_failure_count"],
            "requested_realized_cosine": entry["requested_realized_cosine"],
            "requested_realized_relative_rmse": entry["requested_realized_relative_rmse"],
            "final_downstream_gain_over_realized_source": entry["actual_final_state_curve"][
                "downstream_rms_gain_over_realized_source"
            ],
        }

    derived = {
        "fig-1-linearity-census": {
            "prompt_direction_cells": len(rows),
            "linearity_gate_doses_percent": [2, 3, 4],
            "actual_final_failure_count": sum(row["actual_final_status"] == "fail" for row in rows),
            "realized_source_failure_count": sum(row["realized_source_status"] == "fail" for row in rows),
            "j_of_realized_failure_count": sum(row["j_of_realized_status"] == "fail" for row in rows),
            "realized_source_cosine_range": [
                min(row["realized_source_linearity_cosine_min"] for row in rows),
                max(row["realized_source_linearity_cosine_min"] for row in rows),
            ],
            "realized_source_slope_discrepancy_range": [
                min(row["realized_source_slope_discrepancy_max"] for row in rows),
                max(row["realized_source_slope_discrepancy_max"] for row in rows),
            ],
            "j_of_realized_cosine_range": [
                min(row["j_of_realized_linearity_cosine_min"] for row in rows),
                max(row["j_of_realized_linearity_cosine_min"] for row in rows),
            ],
            "j_of_realized_slope_discrepancy_range": [
                min(row["j_of_realized_slope_discrepancy_max"] for row in rows),
                max(row["j_of_realized_slope_discrepancy_max"] for row in rows),
            ],
            "actual_final_cosine_range": [
                min(row["actual_final_linearity_cosine_min"] for row in rows),
                max(row["actual_final_linearity_cosine_min"] for row in rows),
            ],
            "actual_final_slope_discrepancy_range": [
                min(row["actual_final_slope_discrepancy_max"] for row in rows),
                max(row["actual_final_slope_discrepancy_max"] for row in rows),
            ],
        },
        "fig-2-dose-depth-arc": {
            "source_row_count": len(summary["primary_actual_state_arc_rows"]),
            "prompt_direction_cells": 24,
            "requested_nonzero_dose_count": 60,
            "state_count": 30,
            "selected_final_gain_medians": {
                dose: values["final_downstream_gain_over_realized_source"]["median"]
                for dose, values in selected_doses.items()
                if dose in {"0.02", "0.03", "0.04", "0.08", "0.3"}
            },
        },
        "fig-3-bf16-delivery-floor": {
            "requested_nonzero_dose_count": 60,
            "selected_doses": selected_doses,
            "first_universal_delivery_pass_dose": 0.02,
        },
        "fig-4-j-versus-identity": {
            "readout_layers": list(range(50, 79)),
            "primary_layer": 50,
            "primary_fixed_token_logit_delta_pearson": summary["readout_transport"][
                "fixed_token_logit_delta_pearson"
            ]["by_readout_layer"]["50"],
            "primary_residual_delta_cosine": summary["readout_transport"][
                "residual_delta_cosine"
            ]["by_readout_layer"]["50"],
            "descriptive_threshold_pass_layers": summary["readout_transport"][
                "diagnostic_descriptive_j_readout_threshold_pass_layers"
            ],
            "learned_j_added_value_threshold_pass_layers": summary["readout_transport"][
                "diagnostic_learned_j_added_value_threshold_pass_layers"
            ],
        },
    }

    receipt_files = []
    for stem, spec in FIGURE_SPECS.items():
        receipt = {
            "schema_version": 1,
            "figure_id": stem,
            "title": spec["title"],
            "description": spec["description"],
            "alt_text": spec["alt_text"],
            "data_source": {
                "artifact": "CALIBRATION_SUMMARY.json",
                "path_at_generation": str(source),
                "sha256": source_sha,
                "sections": spec["data_sections"],
                "selection_and_aggregation": spec["transformation"],
            },
            "provenance": {
                "run": RUN_LABEL,
                "study_id": summary["study_id"],
                "protocol_version": summary["protocol_version"],
                "audit_receipt_sha256": summary["audit_receipt_sha256"],
                "summary_receipt_sha256": summary["receipt_sha256"],
                "generator": {
                    "path": str(generator_path),
                    "sha256": sha256(generator_path),
                    "matplotlib_version": mpl.__version__,
                    "manual_data_geometry": False,
                },
                "outputs": {
                    path.name: sha256(path) for path in sorted(output_by_stem[stem])
                },
            },
            "derived_values": derived[stem],
            "accessibility": {
                "alt_text_embedded_in_svg_and_png_metadata": True,
                "color_is_not_the_only_channel": True,
                "text_alternative_required_at_embed": True,
            },
            "claim_scope": [
                "Target-blind generic-direction mechanics validation only.",
                "No SAE direction, paper prompt, semantic endpoint, behavior, or consciousness claim.",
                "Fixed-census ranges are descriptive; readout bands are stability intervals, not population confidence intervals.",
            ],
        }
        receipt_path = output_dir / f"{stem}.receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        receipt_files.append(receipt_path)

    index = {
        "schema_version": 1,
        "source_sha256": source_sha,
        "generator_sha256": sha256(generator_path),
        "receipts": {path.name: sha256(path) for path in sorted(receipt_files)},
    }
    (output_dir / "figure-receipts-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--formats",
        default="svg,png,pdf",
        help="Comma-separated output formats (default: svg,png,pdf)",
    )
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Regenerate in a temporary directory and require byte-identical outputs",
    )
    parser.add_argument(
        "--post",
        type=Path,
        help="Optional Markdown post whose embeds, links, and alt text must match receipts",
    )
    return parser.parse_args()


def render_all(summary: dict, source: Path, output_dir: Path, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(figure_linearity(summary, output_dir, formats))
    outputs.extend(figure_dose_depth(summary, output_dir, formats))
    outputs.extend(figure_delivery(summary, output_dir, formats))
    outputs.extend(figure_readout(summary, output_dir, formats))
    write_figure_receipts(summary, source, output_dir, outputs)
    return outputs


def verify_outputs(expected_dir: Path, regenerated_dir: Path) -> None:
    regenerated = sorted(path for path in regenerated_dir.iterdir() if path.is_file())
    missing = [path.name for path in regenerated if not (expected_dir / path.name).is_file()]
    if missing:
        raise SystemExit(f"Missing expected generated files: {missing}")
    mismatched = [
        path.name
        for path in regenerated
        if (expected_dir / path.name).read_bytes() != path.read_bytes()
    ]
    if mismatched:
        raise SystemExit(f"Generated output drift detected: {mismatched}")


def verify_post_embeds(post: Path, output_dir: Path) -> None:
    text = post.read_text(encoding="utf-8")
    failures = []
    for stem, spec in FIGURE_SPECS.items():
        if f"![{spec['alt_text']}]({stem}.svg)" not in text:
            failures.append(f"{stem}: SVG embed or alt text differs from receipt")
        for suffix in ("receipt.json", "pdf", "png"):
            target = f"{stem}.{suffix}"
            if target not in text:
                failures.append(f"{stem}: post does not link {target}")
        receipt_path = output_dir / f"{stem}.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("alt_text") != spec["alt_text"]:
            failures.append(f"{stem}: receipt alt text differs from generator spec")
    if failures:
        raise SystemExit("Post-embed verification failed:\n- " + "\n- ".join(failures))


def main() -> None:
    args = parse_args()
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = set(formats) - {"svg", "png", "pdf"}
    if unsupported:
        raise SystemExit(f"Unsupported formats: {sorted(unsupported)}")
    configure_style()
    summary = load_summary(args.summary, args.allow_hash_mismatch)
    if args.verify:
        if not args.output_dir.is_dir():
            raise SystemExit(f"Output directory does not exist: {args.output_dir}")
        with tempfile.TemporaryDirectory(prefix="signed-dose-figure-verify-") as temporary:
            temporary_dir = Path(temporary)
            outputs = render_all(summary, args.summary, temporary_dir, formats)
            verify_outputs(args.output_dir, temporary_dir)
        if args.post is not None:
            verify_post_embeds(args.post, args.output_dir)
        print(f"Verified {len(outputs)} byte-identical figure files and 4 receipts")
        return
    outputs = render_all(summary, args.summary, args.output_dir, formats)
    if args.post is not None:
        verify_post_embeds(args.post, args.output_dir)
    print(f"Rendered {len(outputs)} figure files and 4 receipts to {args.output_dir}")


if __name__ == "__main__":
    main()
