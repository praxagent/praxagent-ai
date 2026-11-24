#!/usr/bin/env python3
"""Generate the Arnoldi-iteration teaching figure and its share card.

Toy example, no language model: a 200-by-200 nonsymmetric matrix built with known eigenvalues, a bulk
inside the disk of radius 0.55 plus four outliers (-1.1, 0.95 and 0.75 +- 0.55i). Arnoldi iteration from
a seeded random start uses only matrix-vector products; its Ritz values find the outliers in a few dozen
steps while the bulk stays unresolved. The matrix and start are real, so the Ritz values come in
conjugate pairs and one error curve serves the pair 0.75 +- 0.55i. Checks every published number: the
hand example diag(3, 2, 1), all four outliers near after 10 steps and within 1e-10 after 40, -1.1
fastest, and no bulk eigenvalue resolved. --verify also compares both PNGs pixel for pixel.

    uv run --no-project --with matplotlib==3.10.8 --with pillow --with numpy python make_figure.py [--verify]
"""

import argparse
import io
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = "arnoldi-iteration"
PAPER, PANEL, INK, MUTED = "#F7F4F0", "#FBF9F6", "#2C2924", "#5A544C"
BLUE, CLAY, GREEN, GRAY, RED = "#4B6787", "#A67C52", "#6F8D5E", "#7F786D", "#B5483A"
FIGURE = (7.6, 4.0)   # 1520 x 800 pixels at 200 dpi
CARD = (6.0, 3.15)    # 1200 x 630 pixels at 200 dpi
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.facecolor": PANEL})

N, SEED, K_MAX = 200, 7, 40
OUTLIERS = (-1.1, 0.95, complex(0.75, 0.55))
SHOWN = (10, 40)


def arnoldi(matvec, v0, k):
    """k steps of real Arnoldi: Q with orthonormal columns and the (k+1)-by-k Hessenberg H with
    A Q[:, :k] = Q H. Stops early, with a square H, if the new direction is negligible next to A q."""
    if k < 1:
        raise ValueError("k must be at least 1")
    q = np.asarray(v0, dtype=float)
    Q = np.zeros((q.size, k + 1))
    H = np.zeros((k + 1, k))
    Q[:, 0] = q / np.linalg.norm(q)
    for j in range(k):
        w = np.asarray(matvec(Q[:, j]), dtype=float)
        scale = np.linalg.norm(w)
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ w
            w = w - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] <= 1e-12 * scale:
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = w / H[j + 1, j]
    return Q, H


def ritz_values(H, k):
    return np.linalg.eigvals(H[:k, :k])


def test_matrix():
    """A = S D S^-1 with D block diagonal (2-by-2 rotation blocks for complex pairs) and S near the identity."""
    rng = np.random.default_rng(SEED)
    blocks, eigs = [], []
    for value in OUTLIERS:
        if isinstance(value, complex):
            blocks.append(np.array([[value.real, value.imag], [-value.imag, value.real]]))
            eigs += [value, value.conjugate()]
        else:
            blocks.append(np.array([[value]]))
            eigs.append(complex(value))
    while len(eigs) < N:
        radius, angle = 0.55 * math.sqrt(rng.random()), 2 * math.pi * rng.random()
        if N - len(eigs) >= 2 and rng.random() < 0.8:
            a, b = radius * math.cos(angle), radius * math.sin(angle)
            blocks.append(np.array([[a, b], [-b, a]]))
            eigs += [complex(a, b), complex(a, -b)]
        else:
            r = radius * (1 if rng.random() < 0.5 else -1)
            blocks.append(np.array([[r]]))
            eigs.append(complex(r))
    D = np.zeros((N, N))
    i = 0
    for block in blocks:
        m = block.shape[0]
        D[i:i + m, i:i + m] = block
        i += m
    S = np.eye(N) + 0.5 * rng.standard_normal((N, N)) / math.sqrt(N)
    A = S @ D @ np.linalg.inv(S)
    return A, np.array(eigs), rng.standard_normal(N)


def outlier_errors(A, v0):
    """Distance from each of the four outliers to its nearest Ritz value after k steps, for k = 1 ... K_MAX."""
    _, H = arnoldi(lambda v: A @ v, v0, K_MAX)
    targets = (-1.1, 0.95, complex(0.75, 0.55), complex(0.75, -0.55))
    return {t: [float(np.min(np.abs(ritz_values(H, k) - t))) for k in range(1, K_MAX + 1)] for t in targets}, H


def check_values(A, eigs, errors, H):
    _, H2 = arnoldi(lambda v: np.diag([3.0, 2.0, 1.0]) @ v, np.ones(3), 2)
    got = sorted(np.round(ritz_values(H2, 2).real, 4))
    want = sorted([round(2 - math.sqrt(2 / 3), 4), round(2 + math.sqrt(2 / 3), 4)])
    if got != want or want != [1.1835, 2.8165]:
        raise SystemExit(f"hand example Ritz values {got} are not 1.1835 and 2.8165")
    _, H3 = arnoldi(lambda v: np.diag([3.0, 2.0, 1.0]) @ v, np.ones(3), 3)
    if H3.shape != (3, 3) or np.max(np.abs(np.sort(np.linalg.eigvals(H3).real) - [1.0, 2.0, 3.0])) > 1e-14:
        raise SystemExit("three steps on diag(3, 2, 1) do not stop with a square H holding 1, 2 and 3")
    if np.max(np.abs(np.sort_complex(np.linalg.eigvals(A)) - np.sort_complex(eigs))) > 1e-8:
        raise SystemExit("the test matrix does not have the intended eigenvalues")
    if errors[complex(0.75, 0.55)] != errors[complex(0.75, -0.55)]:
        raise SystemExit("the conjugate outliers do not share one error curve")
    first = {t: e[0] for t, e in errors.items()}
    near = {t: e[SHOWN[0] - 1] for t, e in errors.items()}
    final = {t: e[SHOWN[1] - 1] for t, e in errors.items()}
    if not all(0.3 < v < 3 for v in first.values()):
        raise SystemExit(f"the errors after one step are not about 1: {first}")
    if not all(1e-6 < v < 0.05 for v in near.values()):
        raise SystemExit(f"after 10 steps the outliers are not near yet unresolved: {near}")
    if not all(1e-13 < v <= 1e-10 for v in final.values()):
        raise SystemExit(f"after 40 steps the outliers are not found to between 1e-10 and 1e-13: {final}")
    if final[-1.1] >= min(v for t, v in final.items() if t != -1.1):
        raise SystemExit("the outlier at -1.1 does not converge fastest")
    outliers = np.array([complex(t) for t in errors])
    bulk = np.array([e for e in eigs if np.min(np.abs(e - outliers)) > 1e-9])
    ritz = ritz_values(H, SHOWN[1])
    if bulk.size != N - 4 or min(float(np.min(np.abs(b - ritz))) for b in bulk) < 1e-6:
        raise SystemExit("a bulk eigenvalue is resolved to 1e-6 after 40 steps")


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


def draw(size, eigs, errors, H):
    fig, s = frame(size, "Arnoldi iteration finds the outlying eigenvalues first",
                   "A 200 \u00d7 200 nonsymmetric matrix with known eigenvalues, probed only through products A v. "
                   "Toy example; no language model.",
                   "After 40 products every outlier is found to about ten digits or better; the bulk is not.")
    ax = fig.add_axes([0.08, 0.2, 0.38, 0.58])
    style_axes(ax, s)
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=GRAY, linewidth=0.8 * s, linestyle=(0, (4, 3))))
    ax.scatter(eigs.real, eigs.imag, s=7 * s, color="#C9C2B8", linewidths=0, label="true eigenvalues")
    for k, color, marker, label in ((SHOWN[0], BLUE, "o", "Ritz values, 10 steps"),
                                    (SHOWN[1], RED, "x", "Ritz values, 40 steps")):
        r = ritz_values(H, k)
        ax.scatter(r.real, r.imag, s=16 * s, color=color, marker=marker, linewidths=1.0 * s,
                   facecolors="none" if marker == "o" else color, label=label)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("real part", fontsize=8 * s)
    ax.set_ylabel("imaginary part", fontsize=8 * s)
    ax.legend(fontsize=6.8 * s, frameon=True, facecolor=PANEL, edgecolor="none", framealpha=1.0,
              loc="upper left", borderaxespad=0.1)
    ax.set_title("(a) The complex plane; dashed: unit circle", fontsize=8.2 * s, loc="left", pad=4 * s)

    bx = fig.add_axes([0.58, 0.2, 0.38, 0.58])
    style_axes(bx, s)
    ks = list(range(1, K_MAX + 1))
    for target, color, style, marker, start, label in (
            (-1.1, RED, "-", "o", 1, "outlier \u22121.1"),
            (0.95, BLUE, "--", "s", 3, "outlier 0.95"),
            (complex(0.75, 0.55), CLAY, ":", "^", 5, "outlier pair 0.75 \u00b1 0.55i")):
        bx.semilogy(ks, [max(e, 1e-16) for e in errors[target]], color=color, linestyle=style,
                    linewidth=1.4 * s, marker=marker, markevery=slice(start, K_MAX - 1, 6), ms=3.6 * s,
                    label=label)
    bx.set_xlim(1, K_MAX)
    bx.set_ylim(1e-15, 3)
    bx.set_xlabel("Arnoldi steps k (matrix-vector products)", fontsize=8 * s)
    bx.set_ylabel("distance to the nearest Ritz value", fontsize=8 * s)
    bx.legend(fontsize=7 * s, frameon=False, loc="upper right")
    bx.set_title("(b) Each outlier converges geometrically", fontsize=8.2 * s, loc="left", pad=4 * s)
    return fig


def render(size, eigs, errors, H):
    fig = draw(size, eigs, errors, H)
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
    A, eigs, v0 = test_matrix()
    errors, H = outlier_errors(A, v0)
    check_values(A, eigs, errors, H)
    outputs = {f"{NAME}.png": render(FIGURE, eigs, errors, H), f"{NAME}-card.png": render(CARD, eigs, errors, H)}
    for name, image in outputs.items():
        path = HERE / name
        if args.verify:
            old = Image.open(path).convert("RGB")
            if old.size != image.size or ImageChops.difference(old, image).getbbox():
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            image.save(path, format="PNG", optimize=True)
    for t, e in errors.items():
        print(f"outlier {t}: nearest Ritz distance {e[SHOWN[0] - 1]:.2e} at 10 steps, {e[SHOWN[1] - 1]:.2e} at 40")
    print("Verified values and both PNGs." if args.verify else "Generated both PNGs.")


if __name__ == "__main__":
    main()
