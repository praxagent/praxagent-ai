#!/usr/bin/env python3
"""Generate the period-doubling teaching figure (logistic bifurcation diagram) and its card.

Toy example, no language model. Doubling onsets are computed, not typed. The first two
come from exact formulas, r = 3 and 1 + sqrt(6); each later one is solved by Newton's
method in 60-digit arithmetic (mpmath) from two conditions: the state returns after
p = 4, 8, 16, ... steps, and the multiplier of that p-cycle equals -1. The known value of
Feigenbaum's delta only seeds that search and extrapolates the accumulation marker from
the last two onsets. --verify recomputes the values and compares both PNGs pixel for
pixel; identical pixels need the pinned Matplotlib and the same fonts.

    uv run --no-project --with matplotlib==3.10.8 --with pillow --with numpy --with mpmath python make_figure.py [--verify]
"""

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

HERE = Path(__file__).resolve().parent
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY, RULE = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D", "#A89B8C"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})


def frame(size, title, subtitle, footer):
    """Figure with the house title, subtitle and footer takeaway bar; returns (fig, scale)."""
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

import mpmath as mp  # noqa: E402

NAME = "period-doubling"
DELTA = mp.mpf("4.669201609102990671853203820466")


def cycle(x, r, p):
    d = mp.mpf(1)
    for _ in range(p):
        d *= r * (1 - 2 * x)
        x = r * x * (1 - x)
    return x, d


def onsets(count):
    """Parameters at which the period-2**k cycle's multiplier reaches -1, k = 0, 1, ..."""
    mp.mp.dps = 60
    c = [mp.mpf(3), 1 + mp.sqrt(6)]
    while len(c) < count:
        p = 2 ** len(c)
        guess = c[-1] + (c[-1] - c[-2]) / DELTA
        rs = c[-1] + (guess - c[-1]) / 2
        x = mp.mpf("0.5")
        for _ in range(4000):
            x = rs * x * (1 - x)
        sol = mp.findroot(lambda x, r: [cycle(x, r, p)[0] - x, cycle(x, r, p)[1] + 1], (x, guess))
        c.append(sol[1])
    return [float(v) for v in c]


ONSETS = onsets(6)
ACCUMULATION = ONSETS[-1] + (ONSETS[-1] - ONSETS[-2]) / (float(DELTA) - 1)


def check_values():
    if abs(ONSETS[1] - 3.449490) > 1e-6 or abs(ONSETS[2] - 3.544090) > 1e-6:
        raise SystemExit("doubling onsets disagree with the worked example")
    if abs(ACCUMULATION - 3.569946) > 1e-5:
        raise SystemExit("accumulation estimate is off")


def draw(size):
    fig, s = frame(size, "Period doubling: the attracting cycle's period doubles, again and again",
                   "Logistic map x -> r x (1 - x): long-run values of x against r. Toy example; no language model.",
                   "Each doubling needs a smaller change in r than the last; past the accumulation, chaos with periodic windows.")
    ax = fig.add_axes([0.085, 0.2, 0.88, 0.6])
    style_axes(ax, s)
    rs, ys = [], []
    n = 1200
    for i in range(n):
        r = 2.8 + 1.2 * i / (n - 1)
        x = 0.5
        for _ in range(1500):
            x = r * x * (1 - x)
        for _ in range(160):
            x = r * x * (1 - x)
            rs.append(r)
            ys.append(x)
    ax.scatter(rs, ys, s=0.04 * s, color=INK, alpha=0.35, linewidths=0, rasterized=True)
    labels = ["period 2 born", "period 4", "period 8"]
    for r, label in zip(ONSETS[:3], labels):
        ax.axvline(r, color=CLAY, linewidth=0.9 * s, linestyle=(0, (5, 4)))
        ax.text(r - 0.008, 0.03, f"{label}\nr = {r:.4f}", fontsize=6.8 * s, color=CLAY, ha="right", va="bottom")
    ax.axvline(ACCUMULATION, color=BLUE, linewidth=0.9 * s)
    ax.text(ACCUMULATION + 0.008, 0.03, f"accumulation\nr = {ACCUMULATION:.4f}", fontsize=6.8 * s, color=BLUE,
            ha="left", va="bottom")
    ax.set_xlim(2.8, 4.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("parameter r", fontsize=8 * s)
    ax.set_ylabel("long-run x", fontsize=8 * s)
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
    parser = argparse.ArgumentParser(description=__doc__)
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
