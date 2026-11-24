#!/usr/bin/env python3
"""Generate the simultaneous-confidence-interval teaching figure and its share card.

Exact calculations, no simulation and no language model. For m independent 95% intervals the chance
that at least one misses its true value is 1 - 0.95^m (46% at m = 12). Bonferroni's construction gives
each interval the level 1 - 0.05/m, which keeps that chance at or below 5% whatever the dependence, at
the cost of wider intervals: for normal estimates the multiplier rises from 1.960 to 2.865 at m = 12.
--verify rechecks the values and compares both PNGs pixel for pixel.

    uv run --no-project --with matplotlib==3.10.8 --with pillow python make_figure.py [--verify]
"""

import argparse
import io
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = "simultaneous-confidence-interval"
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})

ALPHA = 0.05
M_MAX = 30
MARK = 12


def any_miss_separate(m, alpha=ALPHA):
    """Chance that at least one of m independent intervals, each at level 1 - alpha, misses."""
    return 1 - (1 - alpha) ** m


def any_miss_bonferroni(m, alpha=ALPHA):
    """The same chance when each of m independent intervals has level 1 - alpha / m."""
    return 1 - (1 - alpha / m) ** m


def z(m, alpha=ALPHA):
    """Two-sided normal multiplier for one of m Bonferroni intervals."""
    return NormalDist().inv_cdf(1 - alpha / (2 * m))


def check_values():
    if round(any_miss_separate(MARK), 4) != 0.4596:
        raise SystemExit("1 - 0.95^12 is not 0.4596")
    if round(z(1), 3) != 1.96 or round(z(MARK), 3) != 2.865 or round(z(MARK) / z(1), 3) != 1.462:
        raise SystemExit("normal multipliers disagree with the worked example")
    if not all(any_miss_bonferroni(m) <= ALPHA + 1e-12 for m in range(1, M_MAX + 1)):
        raise SystemExit("Bonferroni intervals exceed the 5% family-wise miss chance")


def frame(size, title, subtitle, footer):
    s = size[0] / FIGURE[0]
    fig = plt.figure(figsize=size, dpi=200, facecolor=PAPER)
    fig.text(0.03, 0.935, title, fontsize=12.5 * s, fontweight="bold", color=INK, va="center")
    fig.text(0.03, 0.875, subtitle, fontsize=8.5 * s, color=MUTED, va="center")
    fig.patches.append(FancyBboxPatch((0.03, 0.025), 0.94, 0.07, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, facecolor="#EAF1E5", edgecolor=GREEN,
                                      linewidth=1.0 * s))
    fig.text(0.045, 0.06, footer, fontsize=9 * s, color=INK, va="center")
    return fig, s


def style_axes(ax, s):
    ax.tick_params(labelsize=7.5 * s, length=3 * s, width=0.6 * s)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7 * s)


def draw(size):
    fig, s = frame(size, "Many intervals at once: separate 95% is not joint 95%",
                   "m independent comparisons, each with a two-sided interval for a normal estimate. "
                   "Exact calculations; no simulation.",
                   "Simultaneous intervals keep the chance of any miss at or below 5%, at the cost of width.")
    ms = list(range(1, M_MAX + 1))
    ax = fig.add_axes([0.08, 0.2, 0.39, 0.58])
    style_axes(ax, s)
    ax.plot(ms, [100 * any_miss_separate(m) for m in ms], "-o", color=CLAY, linewidth=1.4 * s, ms=2.8 * s,
            label="separate 95% intervals")
    ax.plot(ms, [100 * any_miss_bonferroni(m) for m in ms], "-s", color=BLUE, linewidth=1.4 * s, ms=2.6 * s,
            label="Bonferroni: each at 1 \u2212 0.05/m")
    ax.axhline(5, color=GRAY, linewidth=0.7 * s, linestyle=(0, (4, 3)))
    ax.plot([MARK], [100 * any_miss_separate(MARK)], "o", color=CLAY, ms=6 * s, mfc=PANEL, mew=1.4 * s)
    ax.text(MARK + 0.8, 100 * any_miss_separate(MARK) - 2, "46% at m = 12", fontsize=7.6 * s, color=CLAY,
            va="top")
    ax.set_xlim(0, M_MAX + 1)
    ax.set_ylim(0, 85)
    ax.set_xlabel("number of comparisons m", fontsize=8 * s)
    ax.set_ylabel("chance at least one interval misses (%)", fontsize=8 * s)
    ax.legend(fontsize=7.2 * s, frameon=False, loc="upper left")
    ax.set_title("(a) The chance of a miss somewhere", fontsize=8.2 * s, loc="left", pad=4 * s)

    bx = fig.add_axes([0.58, 0.2, 0.38, 0.58])
    style_axes(bx, s)
    bx.plot(ms, [z(m) / z(1) for m in ms], "-s", color=BLUE, linewidth=1.4 * s, ms=2.6 * s)
    bx.axhline(1, color=GRAY, linewidth=0.7 * s, linestyle=(0, (4, 3)))
    bx.plot([MARK], [z(MARK) / z(1)], "s", color=BLUE, ms=6 * s, mfc=PANEL, mew=1.4 * s)
    bx.text(MARK + 0.8, z(MARK) / z(1) - 0.03, "1.46 times wider at m = 12\n(multiplier 2.865 for 1.960)",
            fontsize=7.4 * s, color=BLUE, va="top")
    bx.set_xlim(0, M_MAX + 1)
    bx.set_ylim(0.9, 1.75)
    bx.set_xlabel("number of comparisons m", fontsize=8 * s)
    bx.set_ylabel("width relative to one 95% interval", fontsize=8 * s)
    bx.set_title("(b) The price: wider intervals", fontsize=8.2 * s, loc="left", pad=4 * s)
    return fig


def render(size):
    fig = draw(size)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor=PAPER, metadata={"Software": None})
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    expected = (round(size[0] * 200), round(size[1] * 200))
    if image.size != expected:
        raise SystemExit(f"rendered {image.size}, expected {expected}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    check_values()
    outputs = {f"{NAME}.png": render(FIGURE), f"{NAME}-card.png": render(CARD)}
    for name, image in outputs.items():
        path = HERE / name
        if args.verify:
            old = Image.open(path).convert("RGB")
            if old.size != image.size or ImageChops.difference(old, image).getbbox():
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            image.save(path, format="PNG", optimize=True)
    print("Verified values and both PNGs." if args.verify else "Generated both PNGs.")


if __name__ == "__main__":
    main()
