#!/usr/bin/env python3
"""Generate the iterated-map teaching figure (three logistic-map cobwebs) and its card.

Toy example, no language model: the logistic map x -> r x (1 - x) at r = 2.8,
3.2 and 3.9, started from x0 = 0.1, a start from which the r = 2.8 orbit visibly
alternates around its fixed point; the quoted Lyapunov estimate follows the same
start. --verify recomputes the quoted values and compares both PNGs pixel
for pixel.
"""

import argparse
import io
import math
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

NAME = "iterated-map"
CASES = ((2.8, "r = 2.8: approaches a fixed point"), (3.2, "r = 3.2: approaches a 2-point cycle"),
         (3.9, "r = 3.9: no repeat in 60 steps"))
X0 = 0.1


def logistic(r, x):
    return r * x * (1 - x)


def check_values():
    if abs((1 - 1 / 2.8) - 0.642857) > 1e-6 or abs((2 - 2.8) + 0.8) > 1e-12:
        raise SystemExit("fixed point or slope at r = 2.8 is wrong")
    p, q = 0.513045, 0.799455
    if abs(logistic(3.2, p) - q) > 1e-5 or abs(logistic(3.2, q) - p) > 1e-5:
        raise SystemExit("two-cycle at r = 3.2 is wrong")
    x, total, n = X0, 0.0, 200000
    for _ in range(1000):
        x = logistic(3.9, x)
    for _ in range(n):
        total += math.log(abs(3.9 * (1 - 2 * x)))
        x = logistic(3.9, x)
    if abs(total / n - 0.496) > 0.003:
        raise SystemExit("Lyapunov exponent at r = 3.9 is not about 0.496")


def draw(size):
    fig, s = frame(size, "An iterated map: one rule, applied again and again",
                   "Logistic map x -> r x (1 - x), started at x = 0.1. Toy example; no language model.",
                   "The same rule can settle, cycle or wander, depending on the parameter r.")
    grid = [0.03 + i * 0.322 for i in range(3)]
    xs = [i / 400 for i in range(401)]
    for left, (r, label) in zip(grid, CASES):
        ax = fig.add_axes([left + 0.035, 0.2, 0.25, 0.58])
        style_axes(ax, s)
        ax.plot(xs, [logistic(r, x) for x in xs], color=BLUE, linewidth=1.4 * s)
        ax.plot([0, 1], [0, 1], color=GRAY, linewidth=0.8 * s, linestyle=(0, (5, 4)))
        steps = 60 if r > 3.5 else 40
        px, py, x = [X0], [0.0], X0
        for _ in range(steps):
            y = logistic(r, x)
            px += [x, y]
            py += [y, y]
            x = y
        ax.plot(px, py, color=CLAY, linewidth=0.8 * s, alpha=0.9)
        if r < 3:
            fx = 1 - 1 / r
            ax.plot([fx], [fx], "o", ms=5 * s, mfc="none", mec=INK, mew=1.1 * s)
        elif r < 3.5:
            for p in (0.513045, 0.799455):
                ax.plot([p], [p], "o", ms=4.5 * s, mfc="none", mec=INK, mew=1.1 * s)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=8.5 * s, color=INK, pad=4 * s)
        ax.set_xlabel("x(n)", fontsize=7.5 * s, labelpad=1)
        ax.set_ylabel("x(n+1)", fontsize=7.5 * s, labelpad=1)
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
