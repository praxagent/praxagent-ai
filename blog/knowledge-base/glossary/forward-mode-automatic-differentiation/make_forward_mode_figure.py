#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Build the evaluation-trace figure of the forward-mode automatic differentiation entry.

The figure runs f(x1, x2) = x1 * x1 + x1 * x2 at (3, 2) with the seed direction (1, 0)
and shows every intermediate value beside its tangent. All numbers come from the small
dual-number class below, so the drawing cannot drift from the arithmetic. Pillow is
needed only for --check-layout, which measures the text with the Inter and Arimo fonts.

    uv run make_forward_mode_figure.py                  # write the SVG
    uv run make_forward_mode_figure.py --verify         # rebuild in memory, compare bytes
    uv run make_forward_mode_figure.py --check-layout   # text inside its boxes, no collisions

The social card is the figure exported at 630 px high:

    inkscape forward-mode-automatic-differentiation.svg --export-type=png \\
        --export-filename=forward-mode-automatic-differentiation-card.png --export-height=630
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SVG = HERE / "forward-mode-automatic-differentiation.svg"

FONTS = {
    False: ("/usr/share/fonts/opentype/inter/Inter-Regular.otf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
    True: ("/usr/share/fonts/opentype/inter/Inter-Bold.otf",
           "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
}
CLS = {"title": (15, True), "h": (13, True), "s": (12, False), "m": (11, False)}

W, H = 760, 412
BW, BH = 196, 86
COLS = (24, 282, 540)
TOP, BOTTOM, MIDDLE = 92, 212, 152
DOT = "\u00b7"


@dataclass(frozen=True)
class Dual:
    value: float
    tangent: float

    def __add__(self, other):
        return Dual(self.value + other.value, self.tangent + other.tangent)

    def __mul__(self, other):
        return Dual(self.value * other.value, self.tangent * other.value + self.value * other.tangent)


def trace(seed):
    x1, x2 = Dual(3, seed[0]), Dual(2, seed[1])
    w3 = x1 * x1
    w4 = x1 * x2
    return x1, x2, w3, w4, w3 + w4


X1, X2, W3, W4, W5 = trace((1, 0))
SECOND = trace((0, 1))[-1]
assert (W3, W4, W5) == (Dual(9, 6), Dual(6, 2), Dual(15, 8)) and SECOND == Dual(15, 3)


def n(v):
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def fmt(v):
    s = f"{float(v):.1f}"
    return s[:-2] if s.endswith(".0") else s


def enc(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return "".join(c if ord(c) < 128 else f"&#x{ord(c):x};" for c in s)


class Canvas:
    def __init__(self):
        self.out, self.texts, self.segs = [], [], []

    def text(self, x, y, s, cls="m", anchor="start", extra="", box=None):
        self.texts.append((s, x, y, cls, anchor, box))
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.out.append(f'  <text x="{fmt(x)}" y="{fmt(y)}"{a} class="t {cls}{(" " + extra) if extra else ""}">'
                        f'{enc(s)}</text>')

    def arrow(self, x1, y1, x2, y2, name):
        self.out.append(f'  <line class="flow" x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
                        f'marker-end="url(#head)"/>')
        self.segs.append((name, x1, y1, x2, y2))

    def svg(self):
        return "\n".join(self.out) + "\n"


STYLE = """  <defs>
    <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9"
            markerUnits="userSpaceOnUse" orient="auto">
      <path d="M0,1 L9,5 L0,9 z" fill="#2C2924"/>
    </marker>
  </defs>
  <style>
    .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}
    .title{font-size:15px;font-weight:700}
    .h{font-size:13px;font-weight:700}
    .s{font-size:12px}
    .m{font-size:11px;fill:#5A544C}
    .tan{fill:#7A5634}
    .input{fill:#E8F0F7;stroke:#4B6787;stroke-width:1.3}
    .step{fill:#FBF9F6;stroke:#C4B8A8;stroke-width:1.3}
    .output{fill:#F3E8E0;stroke:#A67C52;stroke-width:2.2}
    .flow{stroke:#2C2924;stroke-width:1.6}
    .foot{fill:#EAF1E5;stroke:#6F8D5E;stroke-width:1.2}
  </style>"""

NODES = [  # key, column, top, css class, heading, value line, tangent line
    ("x1", 0, TOP, "input", "x1, an input", f"value {n(X1.value)}", f"tangent {n(X1.tangent)}, the seed"),
    ("x2", 0, BOTTOM, "input", "x2, an input", f"value {n(X2.value)}", f"tangent {n(X2.tangent)}, the seed"),
    ("w3", 1, TOP, "step", f"w3 = x1 {DOT} x1",
     f"value {n(X1.value)} {DOT} {n(X1.value)} = {n(W3.value)}",
     f"tangent {n(X1.tangent)} {DOT} {n(X1.value)} + {n(X1.value)} {DOT} {n(X1.tangent)} = {n(W3.tangent)}"),
    ("w4", 1, BOTTOM, "step", f"w4 = x1 {DOT} x2",
     f"value {n(X1.value)} {DOT} {n(X2.value)} = {n(W4.value)}",
     f"tangent {n(X1.tangent)} {DOT} {n(X2.value)} + {n(X1.value)} {DOT} {n(X2.tangent)} = {n(W4.tangent)}"),
    ("w5", 2, MIDDLE, "output", "w5 = w3 + w4, the output",
     f"value {n(W3.value)} + {n(W4.value)} = {n(W5.value)}",
     f"tangent {n(W3.tangent)} + {n(W4.tangent)} = {n(W5.tangent)}"),
]

DESC = (
    f"An evaluation trace of f(x1, x2) = x1 times x1 plus x1 times x2 at x1 = {n(X1.value)}, x2 = {n(X2.value)}, "
    f"run once with the seed direction ({n(X1.tangent)}, {n(X2.tangent)}). Two input boxes on the left hold the "
    f"values {n(X1.value)} and {n(X2.value)} with tangents {n(X1.tangent)} and {n(X2.tangent)}. Arrows lead to "
    f"two product boxes: w3 = x1 times x1 has value {n(W3.value)} and tangent {n(W3.tangent)}, and w4 = x1 times "
    f"x2 has value {n(W4.value)} and tangent {n(W4.tangent)}, each tangent from the product rule. Arrows from both "
    f"lead to the output box w5 = w3 + w4, with value {n(W5.value)} and tangent {n(W5.tangent)} from the sum "
    f"rule. A footer states that one pass gives f(3, 2) = {n(W5.value)} and the partial derivative with respect "
    f"to x1, {n(W5.tangent)}, and that seeding (0, 1) instead gives the partial derivative with respect to x2, "
    f"{n(SECOND.tangent)}, in a second pass."
)


def figure():
    c = Canvas()
    title = "Forward mode carries a tangent beside every value"
    c.out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 f'role="img" aria-labelledby="fwdTitle fwdDesc">')
    c.out.append(f'  <title id="fwdTitle">{enc(title)}</title>')
    c.out.append(f'  <desc id="fwdDesc">{enc(DESC)}</desc>')
    c.out.append(f'  <rect width="{W}" height="{H}" fill="#F7F4F0"/>')
    c.out.append(STYLE)
    c.text(24, 30, title, "title", box="top")
    c.text(24, 50, f"f(x1, x2) = x1 {DOT} x1 + x1 {DOT} x2 at (3, 2), run once with the seed direction (1, 0).",
           "m", box="top")
    c.text(24, 67, "Each box holds a value and its tangent: how fast that value changes along the seed "
                   "direction.", "m", box="top")
    for key, col, top, cls, head, value, tangent in NODES:
        x0 = COLS[col]
        c.out.append(f'  <rect class="{cls}" x="{x0}" y="{top}" width="{BW}" height="{BH}" rx="10"/>')
        c.text(x0 + 14, top + 25, head, "h", box=key)
        c.text(x0 + 14, top + 48, value, "s", box=key)
        c.text(x0 + 14, top + 70, tangent, "s", extra="tan", box=key)
    left, mid, right = COLS[0] + BW, COLS[1], COLS[1] + BW
    c.arrow(left, TOP + 43, mid - 2, TOP + 43, "x1 to w3")
    c.arrow(left, TOP + 66, mid - 2, BOTTOM + 22, "x1 to w4")
    c.arrow(left, BOTTOM + 43, mid - 2, BOTTOM + 43, "x2 to w4")
    c.arrow(right, TOP + 50, COLS[2] - 2, MIDDLE + 24, "w3 to w5")
    c.arrow(right, BOTTOM + 36, COLS[2] - 2, MIDDLE + 62, "w4 to w5")
    c.out.append('  <rect class="foot" x="24" y="322" width="712" height="66" rx="8"/>')
    c.text(380, 348, f"One pass gives f(3, 2) = {n(W5.value)} and the derivative along (1, 0): df/dx1 = "
                     f"{n(W5.tangent)}, which is x2 + 2 x1.", "s", anchor="middle", box="foot")
    c.text(380, 370, f"Seeding (0, 1) instead gives df/dx2 = {n(SECOND.tangent)} in a second pass: one pass per "
                     f"direction.", "s", anchor="middle", box="foot")
    c.out.append("</svg>")
    return c


def check_layout(c):
    from PIL import ImageFont

    cache = {}

    def width(s, size, bold):
        return max(cache.setdefault((p, size), ImageFont.truetype(p, size)).getlength(s) for p in FONTS[bold])

    boxes = {key: (COLS[col] + 4, top + 4, COLS[col] + BW - 4, top + BH - 4)
             for key, col, top, *_ in NODES}
    boxes.update({"top": (16, 0, W - 16, 78), "foot": (28, 324, 732, 386)})
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
