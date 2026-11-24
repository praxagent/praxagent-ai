#!/usr/bin/env python3
"""Generate the perplexity teaching figure and its share card.

Toy example, no language model. Perplexity is e raised to the average next-token loss in nats, which
is the reciprocal of the geometric mean of the probabilities given to the true tokens. Two two-token
examples, (1/4, 1/4) and (1/2, 1/8), both have perplexity exactly 4. --verify rechecks the values and
compares both PNGs pixel for pixel.

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
NAME = "perplexity"
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})

EXAMPLES = (("even: 1/4 and 1/4", (0.25, 0.25), BLUE), ("uneven: 1/2 and 1/8", (0.5, 0.125), CLAY))
MARKED = ((math.log(2), "2"), (math.log(4), "4"), (math.log(10), "10"))


def loss(probs):
    return -sum(math.log(p) for p in probs) / len(probs)


def perplexity(probs):
    return math.exp(loss(probs))


def check_values():
    for _, probs, _ in EXAMPLES:
        if abs(perplexity(probs) - 4.0) > 1e-12:
            raise SystemExit(f"perplexity of {probs} is not 4")
    if abs(loss((0.5, 0.125)) - math.log(4)) > 1e-12:
        raise SystemExit("the uneven example's loss is not ln 4")
    if round(math.exp(0.01), 5) != 1.01005 or round(math.exp(0.69), 3) != 1.994:
        raise SystemExit("e^0.01 or e^0.69 disagrees with the text")


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
    fig, s = frame(size, "Perplexity: average loss read as a number of equally likely choices",
                   "Perplexity = e^L for an average loss of L nats (natural log ln, base e \u2248 2.718). "
                   "Toy values; no language model.",
                   "Both examples average ln 4 nats per token: the same loss as probability 1/4 on every true token.")
    ax = fig.add_axes([0.08, 0.2, 0.4, 0.58])
    style_axes(ax, s)
    ls = [3 * i / 300 for i in range(301)]
    ax.plot(ls, [math.exp(v) for v in ls], color=CLAY, linewidth=1.6 * s)
    for value, label in MARKED:
        ax.plot([value], [math.exp(value)], "o", color=INK, ms=4 * s)
        ax.plot([value, value], [0, math.exp(value)], color=GRAY, linewidth=0.6 * s, linestyle=(0, (2, 2)))
        ax.text(value - 0.02, math.exp(value) + 0.6, f"ln {label} nats \u2192 {label}", fontsize=7.4 * s,
                ha="right", va="bottom")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 21)
    ax.set_xlabel("average next-token loss L (nats per token)", fontsize=8 * s)
    ax.set_ylabel("perplexity e^L", fontsize=8 * s)
    ax.set_title("(a) Perplexity grows exponentially with the loss", fontsize=8.2 * s, loc="left", pad=4 * s)

    bx = fig.add_axes([0.58, 0.2, 0.38, 0.58])
    style_axes(bx, s)
    width = 0.32
    for i, (label, probs, color) in enumerate(EXAMPLES):
        xs = [i * 1.1 + k * (width + 0.06) for k in range(len(probs))]
        bx.bar(xs, probs, width=width, color=color)
        for x, p in zip(xs, probs):
            bx.text(x, p + 0.012, f"{p:g}", fontsize=7.2 * s, ha="center", va="bottom")
        bx.text(sum(xs) / 2, -0.07, label, fontsize=7.4 * s, ha="center", va="top", color=color)
    bx.axhline(0.25, color=GREEN, linewidth=1.1 * s, linestyle=(0, (4, 3)))
    bx.text(2.3, 0.26, "geometric mean 1/4,\nperplexity 4", fontsize=7.2 * s, color=GREEN, ha="right",
            va="bottom")
    bx.set_xticks([])
    bx.set_xlim(-0.3, 2.35)
    bx.set_ylim(0, 0.6)
    bx.set_ylabel("probability given to the true token", fontsize=8 * s)
    bx.set_title("(b) Two texts of two tokens each", fontsize=8.2 * s, loc="left", pad=4 * s)
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
