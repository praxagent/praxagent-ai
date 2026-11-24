#!/usr/bin/env python3
"""Generate the Kaplan-Yorke-dimension teaching figure and its share card.

Toy example, no language model: the Henon map x' = 1 - a x^2 + y, y' = b x with a = 1.4 and b = 0.3.
Its two Lyapunov exponents are measured with Benettin's method: two small displacements are evolved by
the Jacobian, the matrix of first partial derivatives, and made perpendicular and of unit length again
after every step (Gram-Schmidt); each exponent is the average natural logarithm of one displacement's
stretch factor. Their sum must equal ln b exactly, because every step's Jacobian has determinant -b.
The Kaplan-Yorke estimate 1 + L1 / |L2| is about 1.26. --verify rechecks the values and compares both
PNGs pixel for pixel.

    uv run --no-project --with matplotlib==3.10.8 --with pillow python make_figure.py [--verify]
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
NAME = "kaplan-yorke-dimension"
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY, RED = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D", "#B5483A"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})

A, B = 1.4, 0.3
START = (0.1, 0.1)
TRANSIENT, STEPS, SHOWN = 1_000, 200_000, 20_000


def henon(x, y):
    return 1 - A * x * x + y, B * x


def exponents():
    """Both Lyapunov exponents of the Henon map by Benettin's method, and points of its attractor."""
    x, y = START
    for _ in range(TRANSIENT):
        x, y = henon(x, y)
    e1, e2 = (1.0, 0.0), (0.0, 1.0)
    log1 = log2 = 0.0
    points = []
    for step in range(STEPS):
        jx = -2 * A * x
        v1 = (jx * e1[0] + e1[1], B * e1[0])
        v2 = (jx * e2[0] + e2[1], B * e2[0])
        n1 = math.hypot(*v1)
        e1 = (v1[0] / n1, v1[1] / n1)
        proj = v2[0] * e1[0] + v2[1] * e1[1]
        w = (v2[0] - proj * e1[0], v2[1] - proj * e1[1])
        n2 = math.hypot(*w)
        e2 = (w[0] / n2, w[1] / n2)
        log1 += math.log(n1)
        log2 += math.log(n2)
        x, y = henon(x, y)
        if step < SHOWN:
            points.append((x, y))
    return log1 / STEPS, log2 / STEPS, points


def kaplan_yorke(values):
    """Kaplan-Yorke estimate of a complete spectrum, largest first; d when no running sum is negative."""
    total, j = 0.0, 0
    for lam in values:
        if total + lam < 0:
            return j + total / abs(lam)
        total += lam
        j += 1
    return float(j)


def check_values(l1, l2):
    if abs(l1 - 0.419) > 0.003:
        raise SystemExit(f"largest exponent {l1:.4f} is not about 0.419")
    if abs((l1 + l2) - math.log(B)) > 1e-9:
        raise SystemExit("the exponents do not sum to ln b")
    if round(kaplan_yorke((l1, l2)), 2) != 1.26:
        raise SystemExit("Kaplan-Yorke estimate is not about 1.26")
    if abs(kaplan_yorke((0.10, 0.04, -0.02, -0.15, -0.30)) - 3.8) > 1e-12:
        raise SystemExit("the five-exponent example does not give 3.8")
    if kaplan_yorke((0.5, -0.5)) != 2.0:
        raise SystemExit("a spectrum whose running sum never turns negative does not give the full dimension")


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


def draw(size, l1, l2, points):
    d = kaplan_yorke((l1, l2))
    fig, s = frame(size, "Kaplan\u2013Yorke: a dimension from the exponents",
                   "The H\u00e9non map, a = 1.4, b = 0.3: a textbook chaotic map of the plane. Toy example; "
                   "no language model.",
                   f"One direction expands at {l1:.2f} per step, one contracts at {abs(l2):.2f}: "
                   f"D = 1 + {l1:.2f} / {abs(l2):.2f} \u2248 {d:.2f}.")
    ax = fig.add_axes([0.08, 0.2, 0.42, 0.58])
    style_axes(ax, s)
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=0.25 * s, color=CLAY, linewidths=0)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("x", fontsize=8 * s)
    ax.set_ylabel("y", fontsize=8 * s)
    ax.set_title(f"(a) {len(points):,} points of the attractor", fontsize=8.2 * s, loc="left", pad=4 * s)

    bx = fig.add_axes([0.6, 0.2, 0.36, 0.58])
    style_axes(bx, s)
    sums = [0.0, l1, l1 + l2]
    bx.axhline(0, color=GRAY, linewidth=0.8 * s)
    bx.plot(range(3), sums, "-o", color=BLUE, linewidth=1.6 * s, ms=4.5 * s)
    bx.plot([d], [0], "D", color=GREEN, ms=6.5 * s, zorder=5)
    bx.text(1.08, l1 + 0.04, f"\u039b\u2081 \u2248 {l1:.2f}", fontsize=7.6 * s, color=RED, va="bottom")
    bx.text(1.85, -1.25, f"\u039b\u2081 + \u039b\u2082 = ln 0.3 \u2248 {l1 + l2:.3f}".replace("-", "\u2212"),
            fontsize=7.6 * s, color=BLUE, ha="right", va="center")
    bx.annotate(f"crosses zero at {d:.2f}", xy=(d, 0), xytext=(0.15, -0.55), fontsize=7.8 * s, color=INK,
                arrowprops={"arrowstyle": "->", "color": GREEN, "linewidth": 1.0 * s})
    bx.set_xticks([0, 1, 2])
    bx.set_xlim(-0.1, 2.3)
    bx.set_ylim(-1.35, 0.7)
    bx.set_xlabel("number of exponents summed, k", fontsize=8 * s)
    bx.set_ylabel("running sum\n(natural-log units per step)", fontsize=8 * s)
    bx.set_title("(b) Running sum of the exponents", fontsize=8.2 * s, loc="left", pad=4 * s)
    return fig


def render(size, l1, l2, points):
    fig = draw(size, l1, l2, points)
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
    l1, l2, points = exponents()
    check_values(l1, l2)
    outputs = {f"{NAME}.png": render(FIGURE, l1, l2, points), f"{NAME}-card.png": render(CARD, l1, l2, points)}
    for name, image in outputs.items():
        path = HERE / name
        if args.verify:
            old = Image.open(path).convert("RGB")
            if old.size != image.size or ImageChops.difference(old, image).getbbox():
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            image.save(path, format="PNG", optimize=True)
    print(f"exponents {l1:.5f} and {l2:.5f}; Kaplan-Yorke {kaplan_yorke((l1, l2)):.4f}")
    print("Verified values and both PNGs." if args.verify else "Generated both PNGs.")


if __name__ == "__main__":
    main()
