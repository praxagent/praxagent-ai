#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Build the diagram of how the spread of the next-token distribution sets the readout gain.

Two tokens whose embeddings lie a distance d apart get the probabilities p and 1 - p.
Along the line between them the covariance of the embeddings is p (1 - p) d^2, and
across it the covariance is 0. With tied embeddings, a small change of the network's
output along that line therefore moves the soft token

    gain = p (1 - p) d^2 / T

times as far. The four cases and every printed number come from that formula at
T = 1; no language model is involved. Pillow is needed only for --check-layout,
which measures the text with the Inter and Arimo fonts.

    uv run make_spread_gain_figure.py                  # write spread-readout-gain.svg
    uv run make_spread_gain_figure.py --verify         # rebuild in memory, compare bytes
    uv run make_spread_gain_figure.py --check-layout   # text inside its boxes, no collisions
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG = HERE / "spread-readout-gain.svg"

FONTS = {
    False: ("/usr/share/fonts/opentype/inter/Inter-Regular.otf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
    True: ("/usr/share/fonts/opentype/inter/Inter-Bold.otf",
           "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
}
CLS = {"title": (15, True), "h": (13, True), "s": (12, False), "v": (12, True), "m": (11, False),
       "k": (11, True)}

W, H = 760, 456
T = Fr(1)
CASES = [  # heading, probability of the left token, distance between the two embeddings
    ("(a) Sure", Fr(99, 100), Fr(3)),
    ("(b) Leaning", Fr(9, 10), Fr(3)),
    ("(c) Torn, far apart", Fr(1, 2), Fr(3)),
    ("(d) Torn, close together", Fr(1, 2), Fr(1, 2)),
]
CW, CY, CH = 173, 82, 190            # card width, top and height
CX = [16 + k * (CW + 12) for k in range(4)]
S = 40                               # pixels per embedding unit
DOT_Y = 164
BAR_Y, BAR_SCALE = 198, 60           # gain bar: top edge, pixels per unit of gain
AXIS_Y, AX0, AX1, VMIN, VMAX = 336, 70, 690, -3, 1
FOOT_Y = 384


def spread(p, d):
    """Variance of the drawn embedding along the line between the two tokens."""
    return p * (1 - p) * d * d


assert [spread(p, d) for _, p, d in CASES] == [Fr(891, 10000), Fr(81, 100), Fr(9, 4), Fr(1, 16)]


def exact(v):
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    assert Fr(s) == v, v
    return s


def signed(x, digits=2):
    return ("+" if x > 0 else "\u2212" if x < 0 else "") + f"{abs(x):.{digits}f}"


def percent(g):
    return f"{float(100 * g):.0f}%"


def fmt(v):
    s = f"{float(v):.1f}"
    return s[:-2] if s.endswith(".0") else s


def enc(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "".join(c if ord(c) < 128 else f"&#x{ord(c):x};" for c in s)


def axis_x(v):
    return AX0 + (v - VMIN) * (AX1 - AX0) / (VMAX - VMIN)


def cases():
    out = []
    for head, p, d in CASES:
        g = spread(p, d) / T
        out.append(dict(head=head, p=p, d=d, spread=spread(p, d), gain=g, log=math.log(g)))
    return out


class Canvas:
    def __init__(self):
        self.out, self.texts, self.segs = [], [], []

    def text(self, x, y, s, cls="m", anchor="start", extra="", box=None):
        self.texts.append((s, x, y, cls, anchor, box))
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.out.append(f'  <text x="{fmt(x)}" y="{fmt(y)}"{a} class="t {cls}{(" " + extra) if extra else ""}">'
                        f'{enc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls, name="line"):
        self.out.append(f'  <line class="{cls}" x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}"/>')
        self.segs.append((name, x1, y1, x2, y2))

    def svg(self):
        return "\n".join(self.out) + "\n"


STYLE = """  <style>
    .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}
    .title{font-size:15px;font-weight:700}
    .h{font-size:13px;font-weight:700}
    .s{font-size:12px}
    .v{font-size:12px;font-weight:700}
    .m{font-size:11px;fill:#5A544C}
    .k{font-size:11px;font-weight:700}
    .shrink{fill:#2F5D8A}
    .stretch{fill:#7A5634}
    .card{fill:#FBF9F6;stroke:#D9D0C4;stroke-width:1.2}
    .tick{stroke:#2C2924;stroke-width:1.2;stroke-dasharray:2 2}
    .seg{stroke:#8A8072;stroke-width:1.4}
    .axis{stroke:#2C2924;stroke-width:1.3}
    .foot{fill:#EAF1E5;stroke:#6F8D5E;stroke-width:1.2}
  </style>"""


def describe(rows):
    parts = []
    for r, letter in zip(rows, "abcd"):
        verb = "shrinks" if r["gain"] < 1 else "stretches"
        parts.append(f"({letter}) {r['head'][4:]}: probabilities {exact(r['p'])} and {exact(1 - r['p'])}, "
                     f"distance {exact(r['d'])}, spread {exact(r['spread'])}, log gain "
                     f"{'plus' if r['log'] > 0 else 'minus'} {abs(r['log']):.2f} nats, so a push "
                     f"{verb} to {percent(r['gain'])[:-1]} percent.")
    return (
        "Four cards at temperature 1. Each shows two tokens as dots sized by probability, the soft token as a "
        "diamond at their balance point, and a bar for the gain on a shared scale, with a tick at a gain of 1. "
        + " ".join(parts) + " Below, a number line of the log gain from minus 3 to plus 1 nats per step "
        "places d and a far below zero, b just below zero, and c above zero; zero means a gain of 1, no change. "
        "A footer states that a spread is an average of squares and never negative, that only the log gain can "
        "fall below zero, which means the push shrinks, and that across the line between the two tokens the "
        "spread is 0, so a push in that direction does not move the soft token."
    )


def figure():
    rows = cases()
    c = Canvas()
    title = "How the spread sets the readout gain: four two-token cases"
    c.out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 f'role="img" aria-labelledby="gainTitle gainDesc">')
    c.out.append(f'  <title id="gainTitle">{enc(title)}</title>')
    c.out.append(f'  <desc id="gainDesc">{enc(describe(rows))}</desc>')
    c.out.append(f'  <rect width="{W}" height="{H}" fill="#F7F4F0"/>')
    c.out.append(STYLE)
    c.text(24, 30, title, "title", box="top")
    c.text(24, 50, "Toy numbers, T = 1. How far the soft token moves when the network's output changes along "
                   "the line between two tokens.", "m", box="top")
    c.text(24, 67, "Dots: tokens, sized by probability. Diamond: the soft token, their balance point. "
                   "Bar: the gain, equal to the spread at T = 1; tick: gain 1.", "m", box="top")

    for k, r in enumerate(rows):
        x0, p, d = CX[k], r["p"], r["d"]
        cx = x0 + CW / 2
        c.out.append(f"  <!-- card {k + 1}: p = {exact(p)}, distance {exact(d)} -->")
        c.out.append(f'  <rect class="card" x="{x0}" y="{CY}" width="{CW}" height="{CH}" rx="10"/>')
        c.text(x0 + 12, CY + 24, r["head"], "h", box=k)
        c.text(x0 + 12, CY + 42, f"p = {exact(p)} and {exact(1 - p)}", "m", box=k)
        c.text(x0 + 12, CY + 57, f"distance {exact(d)}", "m", box=k)
        left, right = cx - float(d) * S / 2, cx + float(d) * S / 2
        soft = left + float(1 - p) * float(d) * S
        c.line(left, DOT_Y, right, DOT_Y, "seg", name=f"segment{k}")
        for x, q in ((left, p), (right, 1 - p)):
            radius = 2 + 5.5 * math.sqrt(float(q))
            c.out.append(f'  <circle cx="{fmt(x)}" cy="{DOT_Y}" r="{fmt(radius)}" fill="#2C2924"/>')
        if right - left >= 40:
            c.text(left, DOT_Y + 22, exact(p), "m", anchor="middle", box=k)
            c.text(right, DOT_Y + 22, exact(1 - p), "m", anchor="middle", box=k)
        else:
            c.text(cx, DOT_Y + 22, f"{exact(p)} and {exact(1 - p)}", "m", anchor="middle", box=k)
        c.out.append(f'  <path d="M{fmt(soft)} {DOT_Y - 5}L{fmt(soft + 5)} {DOT_Y}L{fmt(soft)} {DOT_Y + 5}'
                     f'L{fmt(soft - 5)} {DOT_Y}Z" fill="#A67C52" stroke="#FBF9F6" stroke-width="1.3"/>')
        bar_x = x0 + 12
        c.out.append(f'  <rect x="{bar_x}" y="{BAR_Y}" width="{fmt(max(float(r["gain"]) * BAR_SCALE, 1.5))}" '
                     f'height="9" rx="2" fill="{"#2F5D8A" if r["gain"] < 1 else "#7A5634"}"/>')
        c.line(bar_x + BAR_SCALE, BAR_Y - 3, bar_x + BAR_SCALE, BAR_Y + 12, "tick", name=f"one{k}")
        c.text(x0 + 12, CY + 142, f"spread {exact(r['spread'])}", "s", box=k)
        c.text(x0 + 12, CY + 160, f"log gain {signed(r['log'])} nats", "s", box=k)
        shrink = r["gain"] < 1
        verdict = f"{'shrinks' if shrink else 'stretches'} a push to {percent(r['gain'])}"
        c.text(x0 + 12, CY + 178, verdict, "v", extra="shrink" if shrink else "stretch", box=k)

    c.text(24, 298, "The same four cases on the log scale that this note uses for growth per step, in nats",
           "k", box="axis")
    zero = axis_x(0)
    c.out.append(f'  <rect x="{fmt(axis_x(VMIN))}" y="{AXIS_Y - 14}" width="{fmt(zero - axis_x(VMIN))}" '
                 f'height="14" fill="#E8F0F7"/>')
    c.out.append(f'  <rect x="{fmt(zero)}" y="{AXIS_Y - 14}" width="{fmt(axis_x(VMAX) - zero)}" height="14" '
                 f'fill="#F3E8E0"/>')
    c.line(AX0, AXIS_Y, AX1, AXIS_Y, "axis", name="axis")
    for v in range(VMIN, VMAX + 1):
        x = axis_x(v)
        c.line(x, AXIS_Y, x, AXIS_Y + 5, "axis", name=f"tick{v}")
        c.text(x, AXIS_Y + 18, "0 (gain 1)" if v == 0 else signed(v, 0), "m", anchor="middle", box="axis")
    for r, letter in zip(rows, "abcd"):
        x = axis_x(r["log"])
        colour = "#2F5D8A" if r["gain"] < 1 else "#7A5634"
        c.out.append(f'  <circle cx="{fmt(x)}" cy="{AXIS_Y}" r="5" fill="{colour}" stroke="#FBF9F6" '
                     f'stroke-width="1.2"/>')
        c.text(x, AXIS_Y - 20, letter, "k", anchor="middle", box="axis")
    c.text(AX0, AXIS_Y + 36, "below 0: the readout shrinks a push", "m", box="axis")
    c.text(AX1, AXIS_Y + 36, "above 0: it stretches a push", "m", anchor="end", box="axis")

    c.out.append(f'  <rect class="foot" x="24" y="{FOOT_Y}" width="712" height="56" rx="8"/>')
    c.text(380, FOOT_Y + 22, "A spread is an average of squares, so it is never negative. Only the log gain can "
                             "fall below zero, and then the push shrinks.", "s", anchor="middle", box="foot")
    c.text(380, FOOT_Y + 42, "Across the line between the two tokens the spread is 0: a push in that direction "
                             "does not move the soft token.", "s", anchor="middle", box="foot")
    c.out.append("</svg>")
    return c


def check_layout(c):
    from PIL import ImageFont

    cache = {}

    def width(s, size, bold):
        return max(cache.setdefault((p, size), ImageFont.truetype(p, size)).getlength(s) for p in FONTS[bold])

    boxes = {k: (CX[k] + 4, CY + 4, CX[k] + CW - 4, CY + CH - 4) for k in range(4)}
    boxes.update({"top": (16, 0, W - 16, 76), "axis": (16, 284, W - 16, FOOT_Y - 4),
                  "foot": (28, FOOT_Y + 2, 732, FOOT_Y + 54)})
    rects = []
    for s, x, y, cls, anchor, box in c.texts:
        size, bold = CLS[cls]
        w = width(s, size, bold)
        x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
        rects.append((s, x0, y - 0.8 * size, x0 + w, y + 0.22 * size, box))

    def hits(seg, r, pad=1.0):
        x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
        _, ax, ay, bx, by = seg
        t0, t1, dx, dy = 0.0, 1.0, bx - ax, by - ay
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0:
                    return False
            else:
                t0, t1 = (max(t0, q / p), t1) if p < 0 else (t0, min(t1, q / p))
                if t0 > t1:
                    return False
        return True

    problems = []
    for i, (s, x0, y0, x1, y1, box) in enumerate(rects):
        b = boxes[box]
        if not (x0 >= b[0] and y0 >= b[1] and x1 <= b[2] and y1 <= b[3]):
            problems.append(f"outside {box}: {s!r}")
        for t in rects[i + 1:]:
            if x0 < t[3] and t[1] < x1 and y0 < t[4] and t[2] < y1:
                problems.append(f"text overlaps text: {s!r} / {t[0]!r}")
        for seg in c.segs:
            if hits(seg, (x0, y0, x1, y1)):
                problems.append(f"text crosses a line: {s!r} / {seg[0]}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true", help="rebuild in memory and compare bytes")
    parser.add_argument("--check-layout", action="store_true", help="test text against the drawing")
    args = parser.parse_args()
    c = figure()
    if args.check_layout:
        problems = check_layout(c)
        print("\n".join(problems) if problems else "layout: no collisions")
        if problems:
            return 1
    if args.verify:
        if not SVG.exists() or SVG.read_text(encoding="ascii") != c.svg():
            print(f"stale or missing: {SVG.name}")
            return 1
        print(f"verified: {SVG.name}")
        return 0
    SVG.write_text(c.svg(), encoding="ascii")
    print(f"wrote {SVG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
