#!/usr/bin/env python3
"""Generate the Feigenbaum-constant teaching figure (gap ratios of the logistic cascade) and its card.

Toy example, no language model. The doubling onsets r_1, r_2, ... of the logistic map
are the parameters at which a cycle's multiplier equals -1. The first two come from
exact formulas, 3 and 1 + sqrt(6); each later one is solved by Newton's method in
60-digit arithmetic (mpmath) from two conditions: the state returns after p = 4, 8,
16, ... steps, and the multiplier of that p-cycle equals -1. The known value of delta
only seeds that search. The ratios (r_n - r_(n-1)) / (r_(n+1) - r_n) for n = 2 to 7
are plotted against delta = 4.6692... --verify recomputes everything and compares both
PNGs pixel for pixel; identical pixels need the pinned Matplotlib and the same fonts.

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

NAME = "feigenbaum-constant"
DELTA = mp.mpf("4.669201609102990671853203820466")


def cycle(x, r, p):
    d = mp.mpf(1)
    for _ in range(p):
        d *= r * (1 - 2 * x)
        x = r * x * (1 - x)
    return x, d


def onsets(count):
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
    return c


ONSETS = onsets(8)
RATIOS = [float((ONSETS[i] - ONSETS[i - 1]) / (ONSETS[i + 1] - ONSETS[i])) for i in range(1, len(ONSETS) - 1)]


def check_values():
    quoted = [3.0, 3.449489743, 3.544090360, 3.564407266, 3.568759420, 3.569691610]
    if [round(float(r), 9) for r in ONSETS[:6]] != quoted:
        raise SystemExit("onsets disagree with the worked example")
    expected = [4.7514, 4.6563, 4.6682, 4.6687, 4.6691, 4.6692]
    if [round(q, 4) for q in RATIOS] != expected:
        raise SystemExit(f"ratios {RATIOS} disagree with the worked example")


def draw(size):
    fig, s = frame(size, "Feigenbaum's delta: how fast the doublings crowd together",
                   "Ratios of successive gaps between doubling onsets of the logistic map. Toy example; no language model.",
                   "The ratios approach 4.6692..., the limit shared by generic cascades of smooth one-hump maps with a quadratic top.")
    ax = fig.add_axes([0.09, 0.2, 0.86, 0.58])
    style_axes(ax, s)
    n = list(range(2, len(RATIOS) + 2))
    ax.axhline(float(DELTA), color=CLAY, linewidth=1.0 * s, linestyle=(0, (5, 4)))
    ax.text(n[-1] + 0.15, float(DELTA) + 0.004, "delta = 4.6692...", fontsize=7.5 * s, color=CLAY,
            ha="right", va="bottom")
    ax.plot(n, RATIOS, "-o", color=BLUE, linewidth=1.3 * s, ms=5 * s, mfc=PANEL, mew=1.3 * s)
    for k, q in zip(n, RATIOS):
        ax.text(k, q + (0.012 if k == n[0] else -0.016), f"{q:.4f}", fontsize=7 * s, color=INK, ha="center",
                va="bottom" if k == n[0] else "top")
    ax.set_xticks(n)
    ax.set_xlabel("doubling n, the birth of period 2^n: the gap before it over the gap after it", fontsize=7.5 * s)
    ax.set_ylabel("ratio of successive gaps", fontsize=8 * s)
    ax.set_xlim(n[0] - 0.4, n[-1] + 0.4)
    ax.set_ylim(4.62, 4.78)
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
