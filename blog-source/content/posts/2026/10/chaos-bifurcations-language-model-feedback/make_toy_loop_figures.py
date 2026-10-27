#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "matplotlib>=3.8"]
# ///
"""Build the two toy-model figures of "Why the loop can be chaotic".

The toy keeps two tokens of the soft-token loop. The soft token is then fixed by
one number, u = p_1, and a function g(u) stands in for the network: it returns the
logit gap z_1 - z_2 for the current state. The loop is

    u_{t+1} = f(u_t) = 1 / (1 + exp(-g(u_t) / T)),   f'(u) = (p_1 p_2 / T) g'(u),

so its derivative is the two-token readout gain times the network factor. No
language model is involved; every value comes from the formulas below.

    uv run make_toy_loop_figures.py            # write both PNGs and the receipt
    uv run make_toy_loop_figures.py --verify   # rebuild in memory, compare bytes
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
COBWEBS = HERE / "fig-toy-cobwebs.png"
ROUTE = HERE / "fig-toy-route.png"
RECEIPT = HERE / "toy-loop.receipt.json"

FOLD_A, FOLD_B = 10.0, 1.0          # folding network g(u) = A u (1 - u) - B
CHAOS_INV_T = 3.6                   # inverse temperature of the chaotic cobweb
INV_T_MAX, INV_T_POINTS = 7.0, 3000
TRANSIENT, RECORDED = 4000, 300
DPI = 200

INK, MUTED, GRID = "#2C2924", "#7F786D", "#D9D0C4"
WEB, ATTRACTOR, STRETCH = "#B03A2E", "#2F5D8A", "#2C2924"


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


PANELS = [
    {
        "key": "rising",
        "title": "(a) no network: g rises",
        "outcome": "settles on a fixed point",
        "g": lambda u: 6.0 * u - 2.5,
        "dg": lambda u: np.full_like(u, 6.0),
        "g_text": "g(u) = 6u - 2.5",
        "g_math": r"$g(u) = 6u - 2.5$",
        "inv_t": 1.0,
        "u0": 0.04,
        "steps": 40,
    },
    {
        "key": "falling",
        "title": "(b) alternation: g falls",
        "outcome": "settles on a 2-cycle",
        "g": lambda u: 3.0 - 6.0 * u,
        "dg": lambda u: np.full_like(u, -6.0),
        "g_text": "g(u) = 3 - 6u",
        "g_math": r"$g(u) = 3 - 6u$",
        "inv_t": 1.0,
        "u0": 0.56,
        "steps": 40,
    },
    {
        "key": "folding",
        "title": "(c) a fold: g rises, then falls",
        "outcome": "never settles: chaos",
        "g": lambda u: FOLD_A * u * (1.0 - u) - FOLD_B,
        "dg": lambda u: FOLD_A * (1.0 - 2.0 * u),
        "g_text": "g(u) = 10u(1 - u) - 1",
        "g_math": r"$g(u) = 10u(1 - u) - 1$",
        "inv_t": CHAOS_INV_T,
        "u0": 0.3,
        "steps": 40,
    },
]


def f_of(panel, u):
    return sigmoid(panel["inv_t"] * panel["g"](u))


def df_of(panel, u):
    p = f_of(panel, u)
    return p * (1.0 - p) * panel["inv_t"] * panel["dg"](u)


def orbit(panel, u0, n):
    out = [u0]
    for _ in range(n):
        out.append(float(f_of(panel, np.array(out[-1]))))
    return np.array(out)


def attractor(panel, period_max=8, n_transient=5000):
    u = orbit(panel, panel["u0"], n_transient)[-1]
    tail = orbit(panel, u, 64)
    for q in range(1, period_max + 1):
        if abs(tail[-1] - tail[-1 - q]) < 1e-12:
            return sorted(set(np.round(tail[-q:], 6).tolist())), q
    return None, None


def lyapunov(panel, n_transient=5000, n=20000):
    u = orbit(panel, panel["u0"], n_transient)[-1]
    total = 0.0
    for _ in range(n):
        total += np.log(abs(float(df_of(panel, np.array(u)))))
        u = float(f_of(panel, np.array(u)))
    return total / n


def fold_scan():
    inv_t = np.linspace(0.05, INV_T_MAX, INV_T_POINTS)
    u = np.full(inv_t.shape, 0.5)
    g = lambda v: FOLD_A * v * (1.0 - v) - FOLD_B
    for _ in range(TRANSIENT):
        u = sigmoid(inv_t * g(u))
    rec = np.empty((RECORDED, inv_t.size))
    lam = np.zeros(inv_t.size)
    for k in range(RECORDED):
        p = sigmoid(inv_t * g(u))
        lam += np.log(np.maximum(np.abs(p * (1 - p) * inv_t * FOLD_A * (1 - 2 * u)), 1e-300))
        u = p
        rec[k] = u
    lam /= RECORDED
    distinct = np.array([np.unique(np.round(rec[-64:, i], 7)).size for i in range(inv_t.size)])
    return inv_t, rec, lam, distinct


def style(ax):
    ax.tick_params(labelsize=9, color=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def cobweb_figure():
    fig = plt.figure(figsize=(9.2, 8.0))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 2.5, 1.15], hspace=0.62, wspace=0.3,
                            left=0.085, right=0.985, top=0.86, bottom=0.065)
    u = np.linspace(0.0, 1.0, 801)
    for col, panel in enumerate(PANELS):
        g_ax = fig.add_subplot(grid[0, col])
        g_ax.axhline(0.0, color=GRID, lw=1.0)
        g_ax.plot(u, panel["g"](u), color=INK, lw=1.6)
        g_ax.set_xlim(0, 1)
        g_ax.set_title(panel["title"], fontsize=10, color=INK, loc="left", fontweight="bold")
        g_ax.set_xlabel(r"soft token $u$;  " + panel["g_math"], fontsize=9, color=INK)
        if col == 0:
            g_ax.set_ylabel("logit gap\n" + r"$g(u) = z_1 - z_2$", fontsize=9, color=INK)
        style(g_ax)

        c_ax = fig.add_subplot(grid[1, col])
        c_ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.0)
        fu, slope = f_of(panel, u), np.abs(df_of(panel, u))
        c_ax.plot(u, fu, color=INK, lw=1.0)
        steep = np.where(slope > 1.0, fu, np.nan)
        c_ax.plot(u, steep, color=STRETCH, lw=3.2, solid_capstyle="butt")
        path = orbit(panel, panel["u0"], panel["steps"])
        xs, ys = [path[0]], [0.0]
        for a, b in zip(path[:-1], path[1:]):
            xs += [a, b]
            ys += [b, b]
        c_ax.plot(xs, ys, color=WEB, lw=0.9, alpha=0.9)
        pts, q = attractor(panel)
        if pts is not None:
            c_ax.plot(pts, [float(f_of(panel, np.array(p))) for p in pts], "o", color=ATTRACTOR,
                      ms=6, zorder=5)
        c_ax.set_xlim(0, 1)
        c_ax.set_ylim(0, 1)
        c_ax.set_aspect("equal")
        c_ax.set_title(f"1/T = {panel['inv_t']:g}: {panel['outcome']}", fontsize=9.5, color=INK,
                       loc="left")
        c_ax.set_xlabel(r"$u_t$", fontsize=10, color=INK)
        if col == 0:
            c_ax.set_ylabel(r"$u_{t+1} = f(u_t)$", fontsize=10, color=INK)
        style(c_ax)

        t_ax = fig.add_subplot(grid[2, col])
        series = orbit(panel, panel["u0"], 40)
        t_ax.plot(np.arange(series.size), series, "-o", color=WEB, lw=0.9, ms=2.4)
        t_ax.set_ylim(-0.03, 1.03)
        t_ax.set_xlim(0, 40)
        t_ax.set_xlabel(r"step $t$", fontsize=9, color=INK)
        if col == 0:
            t_ax.set_ylabel(r"$u_t$", fontsize=10, color=INK)
        style(t_ax)
    fig.suptitle("A two-token toy loop: the shape of the network decides what the loop can do",
                 fontsize=11.5, color=INK, x=0.085, ha="left", y=0.985)
    fig.text(0.085, 0.935, r"thick curve: where the map stretches, $|f'(u)| > 1$     red: the cobweb from one "
             "start     blue: the attracting fixed point or cycle", fontsize=8.5, color=MUTED, ha="left")
    return fig


def route_figure(scan):
    inv_t, rec, lam, distinct = scan
    chaotic = (lam > 1e-3) & (distinct > 32)
    first_flip = float(inv_t[np.argmax(distinct >= 2)])
    first_chaos = float(inv_t[np.argmax(chaotic)])
    last_chaos = float(inv_t[len(chaotic) - 1 - np.argmax(chaotic[::-1])])
    bound = 4.0 / FOLD_A

    fig, (b_ax, l_ax) = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08,
                                                  "left": 0.08, "right": 0.985, "top": 0.88,
                                                  "bottom": 0.1})
    for ax in (b_ax, l_ax):
        ax.axvspan(0, bound, color="#EDE8E1", lw=0)
        ax.axvline(CHAOS_INV_T, color=ATTRACTOR, ls="--", lw=1.0)
        style(ax)
    b_ax.plot(np.repeat(inv_t, RECORDED), rec.T.ravel(), ",", color=INK, alpha=0.35, rasterized=True)
    b_ax.set_ylim(-0.02, 1.02)
    b_ax.set_ylabel("states visited\nafter the transient, u", fontsize=9, color=INK)
    labels = [
        (first_flip / 2, "one fixed\npoint"),
        ((first_flip + first_chaos) / 2, "doubling"),
        ((first_chaos + last_chaos) / 2, "chaos with periodic windows"),
        ((last_chaos + INV_T_MAX) / 2, "greedy decoding:\none token forever"),
    ]
    for x, text in labels:
        b_ax.text(x, 1.04, text, ha="center", va="bottom", fontsize=8.5, color=INK,
                  transform=b_ax.get_xaxis_transform())
    l_ax.axhline(0.0, color=MUTED, lw=0.8)
    clipped = np.maximum(lam, -3.0)
    l_ax.plot(inv_t, clipped, color=INK, lw=0.7)
    l_ax.fill_between(inv_t, 0, clipped, where=clipped > 0, color=WEB, alpha=0.75, lw=0)
    l_ax.set_ylim(-3.1, 1.0)
    l_ax.set_ylabel("Lyapunov\nexponent " + r"$\lambda$", fontsize=9, color=INK)
    l_ax.set_xlabel(r"inverse temperature $1/T$   (the temperature falls to the right)", fontsize=9,
                    color=INK)
    l_ax.set_xlim(0, INV_T_MAX)
    l_ax.text(CHAOS_INV_T + 0.06, 0.62, "cobweb (c)", fontsize=8.5, color=ATTRACTOR)
    b_ax.text(bound / 2, 0.22, "provably one fixed point", ha="center", va="center", fontsize=7.5,
              color=MUTED, rotation=90)
    fig.suptitle(r"The folding toy loop as the temperature falls: $g(u) = 10u(1 - u) - 1$",
                 fontsize=11.5, color=INK, x=0.08, ha="left", y=0.985)
    facts = {"first_flip_inv_t": round(first_flip, 3), "first_chaos_inv_t": round(first_chaos, 3),
             "last_chaos_inv_t": round(last_chaos, 3), "contraction_bound_inv_t": bound,
             "max_lyapunov": round(float(lam.max()), 3),
             "chaotic_share_of_scan": round(float(chaotic.mean()), 3),
             "greedy_state_at_max_inv_t": round(float(rec[-1, -1]), 6)}
    return fig, facts


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, metadata={"Software": None}, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def build():
    cob = png_bytes(cobweb_figure())
    fig, facts = route_figure(fold_scan())
    route = png_bytes(fig)
    panels = {}
    for panel in PANELS:
        pts, q = attractor(panel)
        panels[panel["key"]] = {"g": panel["g_text"], "inv_t": panel["inv_t"], "start": panel["u0"],
                                "attractor": pts, "period": q,
                                "lyapunov": round(lyapunov(panel), 3)}
    receipt = {
        "figures": {COBWEBS.name: hashlib.sha256(cob).hexdigest(),
                    ROUTE.name: hashlib.sha256(route).hexdigest()},
        "model": "u_{t+1} = 1 / (1 + exp(-g(u_t) / T)); f'(u) = p1 p2 g'(u) / T; toy, no language model",
        "panels": panels,
        "fold_scan": {"g": "10u(1 - u) - 1", "inv_t_range": [0.05, INV_T_MAX], "points": INV_T_POINTS,
                      "transient": TRANSIENT, "recorded": RECORDED, "start": 0.5, **facts},
        "generator": Path(__file__).name,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
    }
    return cob, route, (json.dumps(receipt, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true", help="rebuild in memory and compare bytes")
    args = parser.parse_args()
    outputs = dict(zip((COBWEBS, ROUTE, RECEIPT), build()))
    if args.verify:
        stale = [p.name for p, data in outputs.items() if not p.exists() or p.read_bytes() != data]
        if stale:
            print("stale or missing: " + ", ".join(stale))
            return 1
        print("verified: " + ", ".join(p.name for p in outputs))
        return 0
    for path, data in outputs.items():
        path.write_bytes(data)
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
