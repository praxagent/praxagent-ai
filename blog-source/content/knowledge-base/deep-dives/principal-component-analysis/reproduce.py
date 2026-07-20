#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib==3.10.3",
#   "numpy==2.3.2",
# ]
# ///
"""Generate and verify the real-data wheat-kernel PCA teaching artifacts.

The source data are the UCI Seeds dataset (DOI 10.24432/C5H30K), licensed
CC BY 4.0. The committed ``seeds_dataset.txt`` file is the unmodified UCI
download. This script parses that file, standardizes the seven measurements,
fits PCA with a singular value decomposition, and generates every empirical
figure and published numerical artifact.

Run:

    uv run --frozen reproduce.py --generate
    uv run --frozen reproduce.py --verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_TXT = "seeds_dataset.txt"
MEASUREMENTS_CSV = "wheat_kernel_measurements.csv"
SCORES_CSV = "pca_scores.csv"
LOADINGS_CSV = "pca_loadings.csv"
FEATURE_FIGURE = "fig-wheat-kernel-feature-space.png"
ALL_FEATURES_FIGURE = "fig-wheat-kernel-all-features.png"
CORRELATION_FIGURE = "fig-wheat-kernel-correlations.png"
PCA_FIGURE = "fig-wheat-kernel-pca.png"
LOADINGS_FIGURE = "fig-wheat-kernel-loadings.png"
OG_CARD = "og-card.png"
RECEIPT = "wheat-kernel-pca.receipt.json"
FIGURE_RECEIPT = "fig-wheat-kernel-pca.receipt.json"
PROVENANCE = "provenance.json"
LOCK_FILE = "reproduce.py.lock"

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/"
    "seeds_dataset.txt"
)
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/236/seeds"
SOURCE_DOI = "10.24432/C5H30K"
EXPECTED_SOURCE_SHA256 = (
    "1f3f83c0d8485ae9148061389d19628607e3f5660e3d6f40ec9102fb398bb12f"
)

FEATURES = (
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
)
FEATURE_LABELS = {
    "area": "Area",
    "perimeter": "Perimeter",
    "compactness": "Compactness",
    "kernel_length": "Kernel length",
    "kernel_width": "Kernel width",
    "asymmetry_coefficient": "Asymmetry coefficient",
    "kernel_groove_length": "Kernel-groove length",
}
VARIETIES = {1: "Kama", 2: "Rosa", 3: "Canadian"}
VARIETY_STYLES = {
    "Kama": ("#4B6787", "o"),
    "Rosa": ("#A67C52", "s"),
    "Canadian": ("#6F8D5E", "^"),
}

PAPER = "#F7F1E7"
INK = "#2D2A26"
MUTED = "#6C6258"
BLUE = "#4B6787"
OCHRE = "#A67C52"
GREEN = "#6F8D5E"
GRID = "#D8CEC1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"{path.name} is not a PNG")
    return struct.unpack(">II", header[16:24])


def load_source() -> tuple[np.ndarray, np.ndarray]:
    source = HERE / SOURCE_TXT
    if not source.is_file():
        raise RuntimeError(f"missing UCI source file: {source}")
    observed_hash = sha256_file(source)
    if observed_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "UCI source checksum changed: "
            f"expected {EXPECTED_SOURCE_SHA256}, observed {observed_hash}"
        )
    data = np.loadtxt(source, dtype=float)
    if data.shape != (210, 8):
        raise RuntimeError(f"expected a 210 by 8 source table, observed {data.shape}")
    matrix = data[:, :7]
    variety_codes = data[:, 7].astype(int)
    if set(variety_codes) != set(VARIETIES):
        raise RuntimeError("unexpected variety codes in UCI source")
    if np.isnan(matrix).any():
        raise RuntimeError("the UCI source should contain no missing measurements")
    counts = {code: int(np.sum(variety_codes == code)) for code in VARIETIES}
    if counts != {1: 70, 2: 70, 3: 70}:
        raise RuntimeError(f"unexpected variety counts: {counts}")
    return matrix, variety_codes


def compute_pca(matrix: np.ndarray) -> dict[str, Any]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    if np.any(scales <= 0):
        raise RuntimeError("a measurement column is constant")
    scaled = (matrix - means) / scales

    left, singular_values, right_t = np.linalg.svd(scaled, full_matrices=False)
    scores = left * singular_values
    loadings = right_t.T

    # A PCA axis and its mirror image are mathematically equivalent. Orient every
    # exported component by a declared feature so repeated runs use the same signs.
    # PC1 and PC2 keep the anchors used in the article. Later components use the
    # feature with the largest absolute coefficient as a stable, auditable anchor.
    anchor_indices = [0, 2]
    anchor_indices.extend(
        int(np.argmax(np.abs(loadings[:, component])))
        for component in range(2, loadings.shape[1])
    )

    sign_rules: list[str] = []
    for component, anchor_index in enumerate(anchor_indices):
        if loadings[anchor_index, component] < 0:
            scores[:, component] *= -1
            loadings[:, component] *= -1
        feature = FEATURES[anchor_index]
        sign_rules.append(
            f"PC{component + 1} is oriented so "
            f"{FEATURE_LABELS[feature].lower()} has a positive principal-axis coefficient"
        )

    component_variances = singular_values**2 / (matrix.shape[0] - 1)
    ratios = component_variances / component_variances.sum()
    return {
        "scaled": scaled,
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": ratios,
        "means": means,
        "scales": scales,
        "sign_rules": sign_rules,
    }


def write_measurements_csv(
    path: Path, matrix: np.ndarray, variety_codes: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["kernel_id", *FEATURES, "variety_code", "variety"])
        for index, (values, code) in enumerate(zip(matrix, variety_codes), start=1):
            writer.writerow(
                [
                    f"kernel_{index:03d}",
                    *[f"{value:.6f}" for value in values],
                    int(code),
                    VARIETIES[int(code)],
                ]
            )


def write_scores_csv(
    path: Path, scores: np.ndarray, variety_codes: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["kernel_id", "variety_code", "variety", *[f"PC{i}" for i in range(1, 8)]]
        )
        for index, (row, code) in enumerate(zip(scores, variety_codes), start=1):
            writer.writerow(
                [
                    f"kernel_{index:03d}",
                    int(code),
                    VARIETIES[int(code)],
                    *[f"{value:.6f}" for value in row],
                ]
            )


def write_loadings_csv(path: Path, loadings: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["feature", "label", *[f"PC{i}" for i in range(1, 8)]])
        for feature, row in zip(FEATURES, loadings):
            writer.writerow(
                [feature, FEATURE_LABELS[feature], *[f"{value:.6f}" for value in row]]
            )


def style_axis(axis: plt.Axes, *, zero_lines: bool = False) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(color=GRID, linewidth=0.7, alpha=0.7)
    if zero_lines:
        axis.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
        axis.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)


def scatter_by_variety(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    variety_codes: np.ndarray,
    *,
    size: float = 44,
) -> None:
    for code, variety in VARIETIES.items():
        color, marker = VARIETY_STYLES[variety]
        selected = variety_codes == code
        axis.scatter(
            x_values[selected],
            y_values[selected],
            s=size,
            c=color,
            marker=marker,
            edgecolors=PAPER,
            linewidths=0.65,
            alpha=0.88,
            label=f"{variety} (code {code})",
            zorder=3,
        )


def plot_feature_space(
    path: Path, matrix: np.ndarray, variety_codes: np.ndarray
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5), dpi=200)
    fig.patch.set_facecolor(PAPER)
    style_axis(axis)
    scatter_by_variety(axis, matrix[:, 3], matrix[:, 4], variety_codes, size=50)
    axis.set_xlabel("Kernel length", color=INK)
    axis.set_ylabel("Kernel width", color=INK)
    fig.suptitle(
        "Two original measurements show only one slice of the data",
        color=INK,
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.905,
        "Each symbol is one measured wheat kernel; five other measurements are not visible here.",
        color=MUTED,
        fontsize=9.5,
        ha="left",
    )
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.80)
    fig.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def plot_all_features(
    path: Path, matrix: np.ndarray, variety_codes: np.ndarray
) -> None:
    """Show every original measurement without forcing unlike units onto one axis."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), dpi=200)
    fig.patch.set_facecolor(PAPER)
    flat_axes = axes.ravel()
    jitter_rng = np.random.default_rng(236)

    for feature_index, (feature, axis) in enumerate(zip(FEATURES, flat_axes)):
        style_axis(axis)
        axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
        axis.grid(axis="y", visible=False)
        positions = np.arange(len(VARIETIES))
        grouped_values = [
            matrix[variety_codes == code, feature_index] for code in VARIETIES
        ]
        boxes = axis.boxplot(
            grouped_values,
            positions=positions,
            vert=False,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK, "linewidth": 1.3},
            whiskerprops={"color": MUTED, "linewidth": 1.0},
            capprops={"color": MUTED, "linewidth": 1.0},
            boxprops={"edgecolor": MUTED, "linewidth": 1.0},
        )
        for box, variety in zip(boxes["boxes"], VARIETIES.values()):
            box.set_facecolor(VARIETY_STYLES[variety][0])
            box.set_alpha(0.22)

        for position, (code, variety) in enumerate(VARIETIES.items()):
            values = matrix[variety_codes == code, feature_index]
            jitter = jitter_rng.uniform(-0.17, 0.17, size=values.size)
            color, marker = VARIETY_STYLES[variety]
            axis.scatter(
                values,
                position + jitter,
                s=15,
                c=color,
                marker=marker,
                edgecolors=PAPER,
                linewidths=0.35,
                alpha=0.72,
                zorder=3,
            )

        axis.set_yticks(positions, list(VARIETIES.values()))
        axis.set_xlabel(FEATURE_LABELS[feature], color=INK)
        axis.set_title(
            chr(ord("A") + feature_index) + ". " + FEATURE_LABELS[feature],
            color=INK,
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.invert_yaxis()

    flat_axes[-1].axis("off")
    fig.suptitle(
        "All seven original measurements, one scale at a time",
        color=INK,
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.955,
        "Each symbol is one kernel; the box marks the middle half of each variety's values.",
        color=MUTED,
        fontsize=9.5,
        ha="left",
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, bottom=0.07, top=0.91, hspace=0.65, wspace=0.34
    )
    fig.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def plot_correlations(path: Path, matrix: np.ndarray) -> None:
    """Show the linear correlations that motivate a joint summary."""
    correlations = np.corrcoef(matrix, rowvar=False)
    short_labels = [
        "Area",
        "Perimeter",
        "Compactness",
        "Length",
        "Width",
        "Asymmetry",
        "Groove length",
    ]
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "praxagent_correlation", [BLUE, PAPER, OCHRE]
    )
    fig = plt.figure(figsize=(9, 8), dpi=200)
    fig.patch.set_facecolor(PAPER)
    # A 0.56-wide axis on a 9:8 figure and a 0.63-high axis make the
    # heat-map grid physically square. Its left and right edges are equally
    # spaced around x=0.5, so the separate color bar cannot push it sideways.
    axis = fig.add_axes((0.22, 0.17, 0.56, 0.63))
    colorbar_axis = fig.add_axes((0.82, 0.17, 0.026, 0.63))
    axis.set_facecolor(PAPER)
    image = axis.imshow(correlations, vmin=-1, vmax=1, cmap=color_map)
    axis.set_xticks(np.arange(len(short_labels)), short_labels, rotation=38, ha="right")
    axis.set_yticks(np.arange(len(short_labels)), short_labels)
    axis.tick_params(colors=INK, labelsize=9)
    axis.set_xticks(np.arange(-0.5, len(short_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(short_labels), 1), minor=True)
    axis.grid(which="minor", color=PAPER, linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(correlations.shape[0]):
        for column in range(correlations.shape[1]):
            value = correlations[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=PAPER if abs(value) >= 0.68 else INK,
                fontsize=8.5,
                fontweight="bold" if row == column else "normal",
            )
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pearson correlation", color=INK)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(GRID)
    fig.suptitle(
        "Original measurements share information",
        color=INK,
        x=0.5,
        y=0.94,
        ha="center",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        "Positive values move together; negative values tend to move in opposite directions.",
        color=MUTED,
        fontsize=9.5,
        ha="center",
    )
    fig.canvas.draw()
    heatmap_box = axis.get_position()
    heatmap_center = (heatmap_box.x0 + heatmap_box.x1) / 2
    if not np.isclose(heatmap_center, 0.5, atol=1e-12):
        raise RuntimeError(f"heat-map grid is not centered: x={heatmap_center}")
    fig.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def plot_pca(
    path: Path, pca: dict[str, Any], variety_codes: np.ndarray
) -> None:
    scores = pca["scores"]
    ratios = pca["explained_variance_ratio"]
    fig, (score_axis, variance_axis) = plt.subplots(
        1, 2, figsize=(12, 5.5), dpi=200, gridspec_kw={"width_ratios": [1.55, 1]}
    )
    fig.patch.set_facecolor(PAPER)
    style_axis(score_axis, zero_lines=True)
    scatter_by_variety(score_axis, scores[:, 0], scores[:, 1], variety_codes, size=48)
    score_axis.set_xlabel(
        f"Principal component 1 (PC1; {ratios[0] * 100:.1f}% of variance)",
        color=INK,
    )
    score_axis.set_ylabel(
        f"Principal component 2 (PC2; {ratios[1] * 100:.1f}% of variance)",
        color=INK,
    )
    score_axis.set_title("A. Kernel scores", color=INK, loc="left", fontweight="bold")
    score_axis.legend(frameon=False, loc="upper right", fontsize=8.5)

    style_axis(variance_axis)
    components = np.arange(1, 8)
    percentages = ratios * 100
    colors = [BLUE if component <= 2 else "#C9BFB3" for component in components]
    bars = variance_axis.bar(components, percentages, color=colors, width=0.72, zorder=3)
    for component, (bar, value) in enumerate(zip(bars, percentages), start=1):
        label = f"{value:.1f}%" if component <= 3 else f"{value:.2f}%"
        variance_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            label,
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8,
        )
    variance_axis.set_xticks(components, [f"PC{i}" for i in components])
    variance_axis.set_ylim(0, 80)
    variance_axis.set_xlabel("Principal component", color=INK)
    variance_axis.set_ylabel("Variance represented (%)", color=INK)
    variance_axis.set_title(
        "B. Explained variance", color=INK, loc="left", fontweight="bold"
    )

    fig.suptitle(
        "Scaled PCA of 210 real wheat kernels",
        x=0.065,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "The first two components represent 89.0% of variation across seven measured features.",
        ha="left",
        color=MUTED,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.14, wspace=0.26)
    fig.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def plot_loadings(path: Path, loadings: np.ndarray) -> None:
    labels = [FEATURE_LABELS[feature] for feature in FEATURES]
    positions = np.arange(len(FEATURES))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), dpi=200, sharey=True)
    fig.patch.set_facecolor(PAPER)
    for component, axis in enumerate(axes):
        style_axis(axis, zero_lines=False)
        axis.axvline(0, color=MUTED, linewidth=1.0, zorder=1)
        values = loadings[:, component]
        colors = [BLUE if value >= 0 else OCHRE for value in values]
        axis.barh(positions, values, color=colors, height=0.62, zorder=3)
        for y, value in zip(positions, values):
            offset = 0.018 if value >= 0 else -0.018
            axis.text(
                value + offset,
                y,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                color=INK,
                fontsize=8,
            )
        axis.set_xlim(-0.82, 0.82)
        axis.set_xlabel("Principal-axis coefficient", color=INK)
        axis.set_title(
            f"Principal component {component + 1} (PC{component + 1})",
            color=INK,
            loc="left",
            fontweight="bold",
        )
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Principal-axis coefficients define the new axes",
        x=0.225,
        y=0.965,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.225,
        0.91,
        "Bar direction gives the sign; bar length gives the strength of each contribution.",
        ha="left",
        color=MUTED,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.82, bottom=0.13, wspace=0.16)
    fig.savefig(
        path,
        dpi=200,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def plot_og_card(
    path: Path, pca: dict[str, Any], variety_codes: np.ndarray
) -> None:
    scores = pca["scores"]
    ratios = pca["explained_variance_ratio"]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAPER)
    axis = fig.add_axes((0.055, 0.14, 0.46, 0.71))
    style_axis(axis, zero_lines=True)
    scatter_by_variety(axis, scores[:, 0], scores[:, 1], variety_codes, size=48)
    axis.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)", color=INK, fontsize=10)
    axis.set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)", color=INK, fontsize=10)
    axis.legend(frameon=False, loc="upper right", fontsize=8)

    fig.text(
        0.56,
        0.79,
        "Principal component\nanalysis",
        color=INK,
        fontsize=31,
        fontweight="bold",
        va="top",
        linespacing=1.0,
    )
    fig.text(
        0.563,
        0.52,
        "A map of variation,\nnot a verdict",
        color=OCHRE,
        fontsize=20,
        fontweight="bold",
        va="top",
        linespacing=1.15,
    )
    fig.text(
        0.563,
        0.28,
        "Real measurements from 210 wheat\nkernels make the calculation, plots,\nand limitations reproducible.",
        color=MUTED,
        fontsize=12,
        va="top",
        linespacing=1.45,
    )
    fig.savefig(
        path,
        dpi=100,
        facecolor=PAPER,
        metadata={"Software": "praxagent UCI Seeds PCA generator"},
    )
    plt.close(fig)


def loading_records(loadings: np.ndarray, component: int) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(loadings[:, component]))[::-1]
    return [
        {
            "feature": FEATURES[index],
            "label": FEATURE_LABELS[FEATURES[index]],
            "loading": rounded(loadings[index, component]),
        }
        for index in order
    ]


def build_outputs(output_dir: Path, generator_sha256: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, variety_codes = load_source()
    pca = compute_pca(matrix)

    write_measurements_csv(output_dir / MEASUREMENTS_CSV, matrix, variety_codes)
    write_scores_csv(output_dir / SCORES_CSV, pca["scores"], variety_codes)
    write_loadings_csv(output_dir / LOADINGS_CSV, pca["loadings"])
    plot_feature_space(output_dir / FEATURE_FIGURE, matrix, variety_codes)
    plot_all_features(output_dir / ALL_FEATURES_FIGURE, matrix, variety_codes)
    plot_correlations(output_dir / CORRELATION_FIGURE, matrix)
    plot_pca(output_dir / PCA_FIGURE, pca, variety_codes)
    plot_loadings(output_dir / LOADINGS_FIGURE, pca["loadings"])
    plot_og_card(output_dir / OG_CARD, pca, variety_codes)

    if png_dimensions(output_dir / OG_CARD) != (1200, 630):
        raise RuntimeError("og-card.png must be exactly 1200 by 630 pixels")

    ratios = pca["explained_variance_ratio"]
    correlations = np.corrcoef(matrix, rowvar=False)
    centroids: dict[str, dict[str, float]] = {}
    for code, variety in VARIETIES.items():
        selected = variety_codes == code
        centroids[variety] = {
            "PC1": rounded(pca["scores"][selected, 0].mean()),
            "PC2": rounded(pca["scores"][selected, 1].mean()),
        }

    generated_files = (
        MEASUREMENTS_CSV,
        SCORES_CSV,
        LOADINGS_CSV,
        FEATURE_FIGURE,
        ALL_FEATURES_FIGURE,
        CORRELATION_FIGURE,
        PCA_FIGURE,
        LOADINGS_FIGURE,
        OG_CARD,
    )
    figure_files = (
        FEATURE_FIGURE,
        ALL_FEATURES_FIGURE,
        CORRELATION_FIGURE,
        PCA_FIGURE,
        LOADINGS_FIGURE,
        OG_CARD,
    )
    receipt = {
        "analysis_id": "uci-seeds-scaled-pca",
        "analysis_scope": (
            "Descriptive PCA of the complete UCI Seeds dataset. No population "
            "inference, classifier evaluation, or causal claim is made."
        ),
        "source": {
            "dataset": "Seeds",
            "repository": "UCI Machine Learning Repository",
            "doi": SOURCE_DOI,
            "dataset_page": SOURCE_PAGE,
            "download_url": SOURCE_URL,
            "license": "CC BY 4.0",
            "committed_source": SOURCE_TXT,
            "source_sha256": sha256_file(HERE / SOURCE_TXT),
            "row_definition": "one measured wheat kernel",
            "variety_code_mapping": VARIETIES,
        },
        "counts": {
            "kernels": int(matrix.shape[0]),
            "features": int(matrix.shape[1]),
            "missing_measurement_cells": int(np.isnan(matrix).sum()),
            "kernels_by_variety": {
                variety: int(np.sum(variety_codes == code))
                for code, variety in VARIETIES.items()
            },
        },
        "preprocessing": {
            "aggregation": "none",
            "imputation": "none; the source table contains no missing values",
            "scaling": (
                "for each feature, subtract the full-dataset mean and divide by "
                "the full-dataset population standard deviation (ddof=0)"
            ),
            "feature_means": {
                feature: rounded(value)
                for feature, value in zip(FEATURES, pca["means"])
            },
            "feature_standard_deviations": {
                feature: rounded(value)
                for feature, value in zip(FEATURES, pca["scales"])
            },
        },
        "original_feature_correlations": {
            feature: {
                other_feature: rounded(correlations[row, column])
                for column, other_feature in enumerate(FEATURES)
            }
            for row, feature in enumerate(FEATURES)
        },
        "pca": {
            "method": "singular value decomposition of the centered, scaled matrix",
            "coefficient_terminology": (
                "Values called loadings in legacy artifact names are principal-axis "
                "coefficients, equivalently entries of the right singular vectors; "
                "they are not feature-component correlation loadings"
            ),
            "explained_variance_ratio": {
                f"PC{index + 1}": rounded(value) for index, value in enumerate(ratios)
            },
            "first_two_components_combined": rounded(ratios[:2].sum()),
            "loadings_ordered_by_absolute_value": {
                "PC1": loading_records(pca["loadings"], 0),
                "PC2": loading_records(pca["loadings"], 1),
            },
            "variety_score_centroids": centroids,
            "sign_orientation": pca["sign_rules"],
        },
        "figures": {
            FEATURE_FIGURE: {
                "question": "What is visible in two of the seven original measurements?",
                "dimensions": list(png_dimensions(output_dir / FEATURE_FIGURE)),
                "sha256": sha256_file(output_dir / FEATURE_FIGURE),
            },
            ALL_FEATURES_FIGURE: {
                "question": "How are all seven original measurements distributed within each recorded variety?",
                "dimensions": list(png_dimensions(output_dir / ALL_FEATURES_FIGURE)),
                "sha256": sha256_file(output_dir / ALL_FEATURES_FIGURE),
                "plotting_note": "Boxplots plus all observations with deterministic jitter from NumPy seed 236",
            },
            CORRELATION_FIGURE: {
                "question": "Which pairs of original measurements have strong linear correlations?",
                "dimensions": list(png_dimensions(output_dir / CORRELATION_FIGURE)),
                "sha256": sha256_file(output_dir / CORRELATION_FIGURE),
            },
            PCA_FIGURE: {
                "question": "Where do kernels land on PC1 and PC2, and how much variance do all components represent?",
                "dimensions": list(png_dimensions(output_dir / PCA_FIGURE)),
                "sha256": sha256_file(output_dir / PCA_FIGURE),
            },
            LOADINGS_FIGURE: {
                "question": "Which principal-axis coefficients define PC1 and PC2?",
                "dimensions": list(png_dimensions(output_dir / LOADINGS_FIGURE)),
                "sha256": sha256_file(output_dir / LOADINGS_FIGURE),
            },
            OG_CARD: {
                "question": "What is the page's central real-data example?",
                "dimensions": list(png_dimensions(output_dir / OG_CARD)),
                "sha256": sha256_file(output_dir / OG_CARD),
            },
        },
        "provenance": {
            "generator": "reproduce.py",
            "generator_sha256": generator_sha256,
            "lockfile": LOCK_FILE,
            "lockfile_sha256": sha256_file(HERE / LOCK_FILE),
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
            "generated_outputs": {
                name: sha256_file(output_dir / name) for name in generated_files
            },
        },
    }
    write_json(output_dir / RECEIPT, receipt)

    figure_receipt = {
        "schema_version": 1,
        "description": (
            "Hash binding for every empirical figure and the featured card "
            "generated from the UCI Seeds PCA analysis"
        ),
        "analysis_receipt": {
            "path": RECEIPT,
            "sha256": sha256_file(output_dir / RECEIPT),
        },
        "source": {
            "path": SOURCE_TXT,
            "sha256": sha256_file(HERE / SOURCE_TXT),
            "doi": SOURCE_DOI,
        },
        "provenance": {
            "generator": "reproduce.py",
            "generator_sha256": generator_sha256,
            "outputs": {
                name: sha256_file(output_dir / name) for name in figure_files
            },
        },
    }
    write_json(output_dir / FIGURE_RECEIPT, figure_receipt)

    manifest = {
        "schema_version": 1,
        "local_bundle": True,
        "source": {
            "path": SOURCE_TXT,
            "sha256": sha256_file(HERE / SOURCE_TXT),
            "doi": SOURCE_DOI,
            "license": "CC BY 4.0",
        },
        "generator": {
            "path": "reproduce.py",
            "sha256": generator_sha256,
            "verify": "uv run --frozen reproduce.py --verify",
        },
        "receipts": {
            FIGURE_RECEIPT: sha256_file(output_dir / FIGURE_RECEIPT),
            RECEIPT: sha256_file(output_dir / RECEIPT),
            LOCK_FILE: sha256_file(HERE / LOCK_FILE),
            SOURCE_TXT: sha256_file(HERE / SOURCE_TXT),
        },
        "figures": list(figure_files),
        "numbers": [
            {
                "id": "kernel-count",
                "value": 210,
                "appears_as": "210 kernels",
                "source": f"{RECEIPT}#counts.kernels",
            },
            {
                "id": "feature-count",
                "value": 7,
                "appears_as": "seven measurement features",
                "source": f"{RECEIPT}#counts.features",
            },
            {
                "id": "missing-measurement-count",
                "value": 0,
                "appears_as": "0 missing measurement cells",
                "source": f"{RECEIPT}#counts.missing_measurement_cells",
            },
            {
                "id": "pc1-explained-variance",
                "value": rounded(ratios[0]),
                "appears_as": "71.9%",
                "source": f"{RECEIPT}#pca.explained_variance_ratio.PC1",
            },
            {
                "id": "pc2-explained-variance",
                "value": rounded(ratios[1]),
                "appears_as": "17.1%",
                "source": f"{RECEIPT}#pca.explained_variance_ratio.PC2",
            },
            {
                "id": "pc1-pc2-combined-explained-variance",
                "value": rounded(ratios[:2].sum()),
                "appears_as": "89.0%",
                "source": f"{RECEIPT}#pca.first_two_components_combined",
            },
        ],
    }
    write_json(output_dir / PROVENANCE, manifest)


def generated_names() -> tuple[str, ...]:
    return (
        MEASUREMENTS_CSV,
        SCORES_CSV,
        LOADINGS_CSV,
        FEATURE_FIGURE,
        ALL_FEATURES_FIGURE,
        CORRELATION_FIGURE,
        PCA_FIGURE,
        LOADINGS_FIGURE,
        OG_CARD,
        RECEIPT,
        FIGURE_RECEIPT,
        PROVENANCE,
    )


def generate() -> None:
    build_outputs(HERE, sha256_file(HERE / "reproduce.py"))
    print("generated UCI Seeds PCA artifacts")


def verify() -> None:
    generator_hash = sha256_file(HERE / "reproduce.py")
    with tempfile.TemporaryDirectory(prefix="verify-uci-seeds-pca-") as temp:
        candidate = Path(temp)
        build_outputs(candidate, generator_hash)
        differences: list[str] = []
        for name in generated_names():
            committed = HERE / name
            rebuilt = candidate / name
            if not committed.is_file():
                differences.append(f"missing committed artifact: {name}")
            elif committed.read_bytes() != rebuilt.read_bytes():
                differences.append(f"byte mismatch: {name}")
        if differences:
            raise RuntimeError("verification failed:\n" + "\n".join(differences))
    print("verified all UCI Seeds PCA artifacts byte for byte")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate()
    else:
        verify()


if __name__ == "__main__":
    main()
