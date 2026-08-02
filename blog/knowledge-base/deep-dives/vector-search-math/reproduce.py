#!/usr/bin/env python3
"""Regenerate and verify the page-owned vector-search teaching figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, ft2font
import matplotlib.pyplot as plt
import numpy as np


BUNDLE = Path(__file__).resolve().parent
FIGURES = (
    "vector_viz_close.png",
    "vector_viz_far.png",
    "curse_of_dimensionality.png",
    "local_vs_global.png",
)
RECEIPT = "fig-vector-search-teaching.receipt.json"
PAPER = "#f7f0df"
INK = "#253338"
MUTED = "#596a6d"
TEAL = "#1d6f74"
RUST = "#b44f2a"
GOLD = "#c58b24"
GRID = "#c9c0ad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_environment() -> dict[str, str]:
    font_path = Path(font_manager.findfont("DejaVu Sans")).resolve()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "matplotlib_backend": matplotlib.get_backend(),
        "freetype": ft2font.__freetype_version__,
        "primary_font_file": font_path.name,
        "primary_font_sha256": sha256(font_path),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finish(fig: plt.Figure, filename: str) -> None:
    fig.savefig(
        BUNDLE / filename,
        dpi=150,
        facecolor=PAPER,
        metadata={"Software": "vector-search-math/reproduce.py"},
    )
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PAPER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def vector_figure(
    filename: str,
    other: np.ndarray,
    other_label: str,
    title: str,
    arc_color: str,
) -> None:
    query = np.array([0.80, 0.60])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    style_axis(ax)
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.10, 1.10)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, linewidth=0.9)
    ax.axvline(0, color=GRID, linewidth=0.9)
    ax.set_xlabel("Teaching coordinate 1", color=INK)
    ax.set_ylabel("Teaching coordinate 2", color=INK)
    ax.set_title(title, color=INK, fontsize=16, fontweight="bold", pad=14)

    for index, (vector, label, color, linestyle) in enumerate((
        (query, "query: weather made me happy", TEAL, "-"),
        (other, other_label, RUST, "--"),
    )):
        ax.annotate(
            "",
            xy=vector,
            xytext=(0, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 3,
                "linestyle": linestyle,
                "mutation_scale": 16,
            },
        )
        ax.scatter(
            vector[0],
            vector[1],
            s=90,
            color=color,
            edgecolor=PAPER,
            linewidth=1.5,
            zorder=3,
        )
        if vector[0] >= 0:
            label_x = vector[0] - 0.04
            label_y = vector[1] - 0.11 if index == 0 else vector[1] + 0.12
            align = "right"
        else:
            label_x = vector[0] + 0.04
            label_y = vector[1] + 0.08
            align = "left"
        ax.text(
            label_x,
            label_y,
            label,
            ha=align,
            va="center",
            color=INK,
            fontsize=9,
            bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": 0.92},
        )

    query_norm = query / np.linalg.norm(query)
    other_norm = other / np.linalg.norm(other)
    cosine = float(np.dot(query_norm, other_norm))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    euclidean = float(np.linalg.norm(query - other))
    ax.text(
        0.03,
        0.03,
        f"cosine similarity = {cosine:.3f}\n"
        f"angle = {angle:.1f} degrees\n"
        f"Euclidean distance = {euclidean:.3f}",
        transform=ax.transAxes,
        color=INK,
        fontsize=10,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#fffaf0",
            "edgecolor": arc_color,
            "linewidth": 1.5,
        },
    )
    finish(fig, filename)


def distance_contrast_figure() -> dict[str, object]:
    dimensions = [10, 25, 50, 100, 250, 500, 1000, 2000]
    point_count = 2000
    ratios: list[float] = []
    rng = np.random.default_rng(42)

    for dimension in dimensions:
        points = rng.standard_normal((point_count, dimension))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        distances = np.linalg.norm(points[1:] - points[0], axis=1)
        ratios.append(float(distances.min() / distances.max()))

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    style_axis(ax)
    ax.plot(
        dimensions,
        ratios,
        color=TEAL,
        linewidth=2.8,
        marker="o",
        markersize=6,
        markerfacecolor=PAPER,
        markeredgewidth=2,
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Vector dimension (log scale)", color=INK)
    ax.set_ylabel("Nearest / farthest distance", color=INK)
    ax.set_title(
        "Distance contrast narrows in this synthetic unit-sphere sample",
        color=INK,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        0.99,
        0.05,
        "seed 42 • 2,000 points per dimension\n"
        "one reference point • higher means less contrast",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=9,
    )
    finish(fig, "curse_of_dimensionality.png")
    return {
        "seed": 42,
        "point_count_per_dimension": point_count,
        "comparison_count_per_dimension": point_count - 1,
        "dimensions": dimensions,
        "nearest_to_farthest_ratios": ratios,
    }


def routing_landscape_figure() -> None:
    x = np.linspace(-5.0, 5.0, 450)
    y = np.linspace(-3.2, 3.2, 320)
    xx, yy = np.meshgrid(x, y)
    zz = (
        0.055 * (xx**2 + 0.8 * yy**2)
        - 2.6 * np.exp(-((xx + 2.2) ** 2 / 1.3 + (yy + 0.3) ** 2 / 0.9))
        - 4.2 * np.exp(-((xx - 2.35) ** 2 / 1.0 + (yy - 0.15) ** 2 / 1.2))
        + 0.5 * np.exp(-(xx**2 / 0.8 + yy**2 / 2.5))
    )

    fig, ax = plt.subplots(figsize=(8, 16 / 3), constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    contour = ax.contourf(
        xx,
        yy,
        zz,
        levels=24,
        cmap="YlGnBu_r",
        alpha=0.82,
    )
    ax.contour(xx, yy, zz, levels=18, colors=INK, linewidths=0.38, alpha=0.35)
    fig.colorbar(contour, ax=ax, shrink=0.78, label="conceptual distance")

    start = (-4.1, 2.35)
    local_path = np.array(
        [start, (-3.5, 1.5), (-3.0, 0.7), (-2.25, -0.25)]
    )
    wider_path = np.array(
        [start, (-3.0, 2.5), (-1.5, 2.3), (0.1, 1.8), (1.3, 1.0), (2.35, 0.15)]
    )
    ax.plot(
        local_path[:, 0],
        local_path[:, 1],
        color=RUST,
        linewidth=3,
        linestyle="--",
        marker="s",
        markersize=5,
        label="one-candidate greedy route",
    )
    ax.plot(
        wider_path[:, 0],
        wider_path[:, 1],
        color=TEAL,
        linewidth=3,
        marker="o",
        markersize=5,
        label="wider candidate route",
    )
    ax.scatter(*start, s=115, marker="^", color=GOLD, edgecolor=INK, zorder=5)
    ax.scatter(-2.25, -0.25, s=125, marker="s", color=RUST, edgecolor=PAPER, zorder=5)
    ax.scatter(2.35, 0.15, s=145, marker="*", color=TEAL, edgecolor=PAPER, zorder=5)
    ax.text(start[0] + 0.15, start[1] + 0.08, "start", color=INK, fontsize=10)
    ax.text(-2.25, -0.62, "nearby basin", ha="center", color=INK, fontsize=10)
    ax.text(2.35, -0.28, "deeper basin", ha="center", color=INK, fontsize=10)
    ax.set(
        xlim=(-5, 5),
        ylim=(-3.2, 3.2),
        xlabel="conceptual search coordinate 1",
        ylabel="conceptual search coordinate 2",
    )
    ax.set_title(
        "Keeping alternatives can escape a locally attractive route",
        color=INK,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors=MUTED)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#fffaf0",
        edgecolor=GRID,
        framealpha=0.95,
    )
    finish(fig, "local_vs_global.png")


def generate() -> None:
    vector_figure(
        "vector_viz_close.png",
        np.array([0.75, 0.66]),
        "related candidate",
        "Related sentences point in nearly the same direction",
        TEAL,
    )
    vector_figure(
        "vector_viz_far.png",
        np.array([-0.60, -0.80]),
        "unrelated candidate",
        "An unrelated sentence points in another direction",
        RUST,
    )
    simulation = distance_contrast_figure()
    routing_landscape_figure()

    outputs = {name: sha256(BUNDLE / name) for name in FIGURES}
    environment = generation_environment()
    receipt = {
        "figure_id": "vector-search-teaching-path",
        "description": (
            "Three conceptual vector-search teaching figures and one seeded "
            "synthetic distance-contrast simulation."
        ),
        "claim_scope_exclusions": [
            "The sentence vectors are hand-authored teaching coordinates, not model outputs.",
            "The routing landscape is conceptual, not an HNSW execution trace.",
            "The distance-ratio curve is not a benchmark of an embedding model or ANN index.",
        ],
        "distance_contrast_simulation": simulation,
        "generation_environment": environment,
        "provenance": {
            "generator": "reproduce.py",
            "outputs": outputs,
        },
    }
    write_json(BUNDLE / RECEIPT, receipt)

    provenance = {
        "local_bundle": True,
        "generator": {
            "path": "reproduce.py",
            "sha256": sha256(Path(__file__).resolve()),
            "regenerate": (
                "uv run --python 3.13.11 --with-requirements "
                "requirements.lock reproduce.py"
            ),
            "verify": (
                "uv run --python 3.13.11 --with-requirements "
                "requirements.lock reproduce.py --verify"
            ),
        },
        "environment": environment,
        "reproducibility_boundary": (
            "--verify checks the integrity of committed artifacts without "
            "recomputing the simulation. Byte-identical PNG regeneration "
            "additionally depends on the recorded Python, NumPy, Matplotlib, "
            "platform, FreeType, and font-file environment."
        ),
        "receipts": {RECEIPT: sha256(BUNDLE / RECEIPT)},
        "figures": list(FIGURES),
        "numbers": [
            {
                "id": "distance-simulation-point-count",
                "value": 2000,
                "appears_as": "2,000",
                "source": RECEIPT,
            },
            {
                "id": "distance-simulation-comparison-count",
                "value": 1999,
                "appears_as": "1,999",
                "source": RECEIPT,
            },
        ],
    }
    write_json(BUNDLE / "provenance.json", provenance)


def verify() -> int:
    manifest_path = BUNDLE / "provenance.json"
    if not manifest_path.is_file():
        print("FAIL provenance.json is missing")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        manifest["generator"]["path"]: manifest["generator"]["sha256"],
        **manifest["receipts"],
    }
    receipt = json.loads((BUNDLE / RECEIPT).read_text(encoding="utf-8"))
    expected.update(receipt["provenance"]["outputs"])

    failures = 0
    for relative, wanted in expected.items():
        path = BUNDLE / relative
        if not path.is_file():
            print(f"FAIL missing {relative}")
            failures += 1
            continue
        found = sha256(path)
        if found != wanted:
            print(f"FAIL {relative}: expected {wanted}, found {found}")
            failures += 1
        else:
            print(f"OK   {relative}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify committed hashes without regenerating figures",
    )
    args = parser.parse_args()
    if args.verify:
        return verify()
    generate()
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
