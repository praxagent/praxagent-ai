#!/usr/bin/env python3
"""Generate the Jacobian teaching figure and its share card.

Toy example, no language model: F(x, y) = (x^2 + y, x y) near p = (2, 1), where the Jacobian is
[[4, 1], [1, 2]]. The figure maps a small circle around p, compares its exact image with the
Jacobian's prediction, and shows that the error of the prediction is exactly (dx^2, dx dy), at most
r^2 on a circle of radius r. --verify rechecks the values and compares both PNGs pixel for pixel.

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
NAME = "jacobian"
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})

P = (2.0, 1.0)
RADIUS = 0.3
MARKS = 12


def F(x, y):
    return x * x + y, x * y


def jacobian_at(x, y):
    return ((2 * x, 1.0), (y, x))


def nudge(r, theta):
    return r * math.cos(theta), r * math.sin(theta)


def linear_error(r, theta):
    """Size of F(p + d) - F(p) - J d for the nudge d of length r at angle theta."""
    dx, dy = nudge(r, theta)
    (a, b), (c, d) = jacobian_at(*P)
    fx, fy = F(P[0] + dx, P[1] + dy)
    f0x, f0y = F(*P)
    return math.hypot(fx - f0x - (a * dx + b * dy), fy - f0y - (c * dx + d * dy))


def check_values():
    if jacobian_at(*P) != ((4.0, 1.0), (1.0, 2.0)):
        raise SystemExit("the Jacobian at (2, 1) is not [[4, 1], [1, 2]]")
    fx, fy = F(2.01, 1.0)
    if round(fx - 5.0, 10) != 0.0401 or round(fy - 2.0, 10) != 0.01:
        raise SystemExit("the worked nudge (0.01, 0) does not change F by (0.0401, 0.01)")
    for r in (0.3, 0.1, 0.03, 0.01):
        worst = max(linear_error(r, 2 * math.pi * i / 720) for i in range(720))
        if abs(worst - r * r) > 1e-12:
            raise SystemExit(f"the largest error on the circle of radius {r} is not r^2")


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
    fig, s = frame(size, "The Jacobian is the best linear stand-in for a smooth map",
                   "F(x, y) = (x\u00b2 + y, x y) near p = (2, 1), where J = [[4, 1], [1, 2]]. "
                   "Toy example; no language model.",
                   "The prediction F(p) + J d misses by exactly (dx\u00b2, dx dy): at most r\u00b2 for a nudge of length r.")
    thetas = [2 * math.pi * i / 360 for i in range(361)]
    marks = [2 * math.pi * i / MARKS for i in range(MARKS)]
    (a, b), (c, d) = jacobian_at(*P)
    f0 = F(*P)

    ax = fig.add_axes([0.07, 0.2, 0.22, 0.58])
    style_axes(ax, s)
    ax.plot([P[0] + nudge(RADIUS, t)[0] for t in thetas], [P[1] + nudge(RADIUS, t)[1] for t in thetas],
            color=BLUE, linewidth=1.3 * s)
    ax.plot([P[0] + nudge(RADIUS, t)[0] for t in marks], [P[1] + nudge(RADIUS, t)[1] for t in marks], "o",
            color=BLUE, ms=3 * s)
    ax.plot(*P, "o", color=INK, ms=4 * s)
    ax.text(P[0] + 0.03, P[1] + 0.03, "p", fontsize=8 * s)
    ax.set_aspect("equal")
    ax.set_xlim(1.55, 2.45)
    ax.set_ylim(0.55, 1.45)
    ax.set_xlabel("x", fontsize=8 * s)
    ax.set_ylabel("y", fontsize=8 * s)
    ax.set_title("(a) Inputs: a circle of radius 0.3", fontsize=8.2 * s, loc="left", pad=4 * s)

    bx = fig.add_axes([0.37, 0.2, 0.28, 0.58])
    style_axes(bx, s)
    exact = [F(P[0] + nudge(RADIUS, t)[0], P[1] + nudge(RADIUS, t)[1]) for t in thetas]
    linear = [(f0[0] + a * nudge(RADIUS, t)[0] + b * nudge(RADIUS, t)[1],
               f0[1] + c * nudge(RADIUS, t)[0] + d * nudge(RADIUS, t)[1]) for t in thetas]
    bx.plot([p[0] for p in exact], [p[1] for p in exact], color=CLAY, linewidth=1.5 * s, label="exact image")
    bx.plot([p[0] for p in linear], [p[1] for p in linear], color=BLUE, linewidth=1.2 * s,
            linestyle=(0, (4, 3)), label="F(p) + J d")
    for t in marks:
        e = F(P[0] + nudge(RADIUS, t)[0], P[1] + nudge(RADIUS, t)[1])
        l_ = (f0[0] + a * nudge(RADIUS, t)[0] + b * nudge(RADIUS, t)[1],
              f0[1] + c * nudge(RADIUS, t)[0] + d * nudge(RADIUS, t)[1])
        bx.plot([e[0], l_[0]], [e[1], l_[1]], color=GRAY, linewidth=0.6 * s)
        bx.plot(*e, "o", color=CLAY, ms=3 * s)
    bx.plot(*f0, "o", color=INK, ms=4 * s)
    bx.text(f0[0] + 0.04, f0[1] + 0.04, "F(p)", fontsize=8 * s)
    bx.set_aspect("equal")
    bx.set_xlim(3.6, 6.4)
    bx.set_ylim(1.2, 2.8)
    bx.set_xlabel("first output", fontsize=8 * s)
    bx.set_ylabel("second output", fontsize=8 * s)
    bx.legend(fontsize=7 * s, frameon=False, loc="upper left")
    bx.set_title("(b) Exact image and linear prediction", fontsize=8.2 * s, loc="left", pad=4 * s)

    cx = fig.add_axes([0.74, 0.2, 0.23, 0.58])
    style_axes(cx, s)
    radii = [10 ** (-2 + 2 * i / 40) for i in range(41)]
    worst = [max(linear_error(r, 2 * math.pi * i / 180) for i in range(180)) for r in radii]
    cx.loglog(radii, worst, color=CLAY, linewidth=1.5 * s)
    cx.loglog(radii, radii, color=GRAY, linewidth=0.9 * s, linestyle=(0, (4, 3)))
    cx.text(0.17, 0.3, "the nudge itself, r", fontsize=7 * s, color=MUTED, ha="right")
    cx.text(0.038, 0.0004, "error = r\u00b2\n(slope 2)", fontsize=7.3 * s, color=CLAY)
    cx.set_xlabel("nudge length r", fontsize=8 * s)
    cx.set_ylabel("largest error on the circle", fontsize=8 * s)
    cx.set_title("(c) Error shrinks like r\u00b2", fontsize=8.2 * s, loc="left", pad=4 * s)
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
