#!/usr/bin/env python3
"""Generate the non-normal-matrix teaching figure (transient growth) and its card.

Toy example, no language model: A = [[0.5, 4], [0, 0.5]] and v = (0, 1). Both
eigenvalues are 0.5, yet one step stretches some vectors by 4.06. --verify
recomputes the lengths and compares both PNGs pixel for pixel.
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

import math  # noqa: E402

NAME = "non-normal-matrix"
A = ((0.5, 4.0), (0.0, 0.5))
STEPS = 14


def lengths():
    x, y = 0.0, 1.0
    out = [math.hypot(x, y)]
    for _ in range(STEPS):
        x, y = A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y
        out.append(math.hypot(x, y))
    return out


def sigma_max():
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    t = a * a + b * b + c * c + d * d
    det = a * d - b * c
    return math.sqrt((t + math.sqrt(t * t - 4 * det * det)) / 2)


def check_values():
    got = [round(v, 3) for v in lengths()[:8]]
    if got != [1.0, 4.031, 4.008, 3.003, 2.001, 1.25, 0.75, 0.438]:
        raise SystemExit(f"lengths {got} disagree with the worked example")
    if round(sigma_max(), 4) != 4.0616:
        raise SystemExit("largest singular value is not 4.0616")


def draw(size):
    fig, s = frame(size, "Non-normal: small eigenvalues, a large first step",
                   "A = [[0.5, 4], [0, 0.5]], v = (0, 1). Both eigenvalues are 0.5. Toy example; no language model.",
                   "Eigenvalues set the long run; the largest singular value sets what one step can do.")
    ax = fig.add_axes([0.085, 0.2, 0.87, 0.6])
    style_axes(ax, s)
    k = list(range(STEPS + 1))
    ax.plot(k, lengths(), "-o", color=CLAY, linewidth=1.4 * s, ms=4.5 * s, label="length of A^k v (actual)")
    ax.plot(k, [0.5 ** i for i in k], "--s", color=BLUE, linewidth=1.2 * s, ms=3.5 * s, mfc=PANEL,
            label="0.5^k (the eigenvalue rate)")
    ax.axhline(sigma_max(), color=GRAY, linewidth=0.8 * s, linestyle=(0, (5, 4)))
    ax.text(STEPS, sigma_max() + 0.08, "largest one-step stretch, 4.06", fontsize=7.5 * s, color=MUTED,
            ha="right", va="bottom")
    ax.axhline(1.0, color=RULE, linewidth=0.6 * s)
    ax.set_xlabel("step k", fontsize=8 * s)
    ax.set_ylabel("length", fontsize=8 * s)
    ax.set_xlim(-0.4, STEPS + 0.4)
    ax.set_ylim(0, 4.7)
    ax.set_xticks(k)
    ax.legend(fontsize=7.5 * s, frameon=False, loc="center right")
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
