#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow>=10"]
# ///
"""Build the two figures of the covariance-matrix glossary entry.

Both use the toy embeddings (0, 0), (4, 0) and (0, 4) of the worked example under
three sets of weights. covariance-matrix.svg draws the one-standard-deviation
ellipse of each covariance matrix; covariance-response.svg draws how far the
weighted mean moves when the score of the token at (4, 0) rises. Every coordinate
and printed number comes from the formulas below. They are toy values, not model
embeddings.

    python3 make_covariance_figures.py                  # write both SVGs
    python3 make_covariance_figures.py --verify         # rebuild in memory, compare bytes
    uv run make_covariance_figures.py --check-layout    # also test text against the drawing

--check-layout measures text with Pillow and the Inter and Arimo fonts. The social
card is the first figure exported at 630 px high:

    inkscape covariance-matrix.svg --export-type=png \\
        --export-filename=covariance-matrix-card.png --export-height=630
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPREAD_SVG = HERE / "covariance-matrix.svg"
RESPONSE_SVG = HERE / "covariance-response.svg"

FONTS = {
    False: ("/usr/share/fonts/opentype/inter/Inter-Regular.otf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
    True: ("/usr/share/fonts/opentype/inter/Inter-Bold.otf",
           "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
}
CLS = {"title": (15, True), "h": (13, True), "s": (12, False), "m": (11, False)}

W, H = 760, 422
POINTS = [(0, 0), (4, 0), (0, 4)]
PANELS = [
    ("Weight shared", [Fr(1, 2), Fr(1, 4), Fr(1, 4)], ["1/2", "1/4", "1/4"]),
    ("One point dominant", [Fr(8, 10), Fr(1, 10), Fr(1, 10)], ["0.8", "0.1", "0.1"]),
    ("All weight on one point", [Fr(1), Fr(0), Fr(0)], ["1", "0", "0"]),
]
PX, PW, PY, PH = [24, 264, 504], 232, 68, 296
S = 34                      # pixels per embedding unit
PUSHED = 1                  # index of the token whose score rises, at (4, 0)
ARROW_SCALE = 2             # the response arrows are drawn at twice their length


def mean_cov(weights):
    mu = [sum(w * p[i] for w, p in zip(weights, POINTS)) for i in range(2)]
    cov = [[sum(w * (p[i] - mu[i]) * (p[j] - mu[j]) for w, p in zip(weights, POINTS)) for j in range(2)]
           for i in range(2)]
    return mu, cov


EXPECTED = [((1, 1), [[3, -1], [-1, 3]]),
            ((Fr(2, 5), Fr(2, 5)), [[Fr(36, 25), Fr(-4, 25)], [Fr(-4, 25), Fr(36, 25)]]),
            ((0, 0), [[0, 0], [0, 0]])]
for (_, _weights, _), (_mu, _cov) in zip(PANELS, EXPECTED):
    assert tuple(mean_cov(_weights)[0]) == tuple(_mu) and mean_cov(_weights)[1] == _cov


def fmt(v):
    s = f"{float(v):.1f}"
    return s[:-2] if s.endswith(".0") else s


def num(v):
    return fmt(v) if v == int(v) else f"{float(v):g}"


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

    def line(self, x1, y1, x2, y2, cls, name="line", marker=""):
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.out.append(f'  <line class="{cls}" x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}"{m}/>')
        self.segs.append((name, x1, y1, x2, y2))

    def svg(self):
        return "\n".join(self.out) + "\n"


STYLE = """  <style>
    .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}
    .title{font-size:15px;font-weight:700}
    .h{font-size:13px;font-weight:700}
    .s{font-size:12px}
    .m{font-size:11px;fill:#5A544C}
    .tan{fill:#7A5634}
    .card{fill:#FBF9F6;stroke:#D9D0C4;stroke-width:1.2}
    .spread{fill:#E8F0F7;fill-opacity:0.9;stroke:#4B6787;stroke-width:1.6}
    .lead{stroke:#A67C52;stroke-width:1}
    .foot{fill:#EAF1E5;stroke:#6F8D5E;stroke-width:1.2}%s
  </style>"""

SPREAD_DESC = (
    "Three panels show the same toy embeddings, (0, 0), (4, 0) and (0, 4), drawn to equal scales, under three "
    "sets of weights. A shaded ellipse around the weighted mean shows the spread: its half-width in any "
    "direction is one standard deviation of the points projected onto that direction. With weights one half, "
    "one quarter and one quarter, the mean is (1, 1), the ellipse is wide and tilted along the line from (0, 4) "
    "to (4, 0), and the covariance matrix has variances 3 and 3 and covariance minus 1. With weights 0.8, 0.1 and "
    "0.1, the mean is (0.4, 0.4), the ellipse is smaller, and the variances are 1.44 and 1.44 with covariance "
    "minus 0.16. With all the weight on (0, 0), the mean is that point, there is no ellipse, and every entry of "
    "the covariance matrix is 0. A footer states that moving weight from (4, 0) and (0, 4) onto (0, 0) shrinks "
    "the covariance here, down to zero when (0, 0) has all the weight."
)


def header(c, title_id, desc_id, title, desc, extra_style="", defs=""):
    c.out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 f'role="img" aria-labelledby="{title_id} {desc_id}">')
    c.out.append(f'  <title id="{title_id}">{enc(title)}</title>')
    c.out.append(f'  <desc id="{desc_id}">{enc(desc)}</desc>')
    c.out.append(f'  <rect width="{W}" height="{H}" fill="#F7F4F0"/>')
    if defs:
        c.out.append(defs)
    c.out.append(STYLE % extra_style)


def panel_frame(c, k, head, wlab, wprefix="weights "):
    px = PX[k]
    c.out.append(f"  <!-- panel {k + 1}: weights {', '.join(wlab)} -->")
    c.out.append(f'  <rect class="card" x="{px}" y="{PY}" width="{PW}" height="{PH}" rx="10"/>')
    c.text(px + 14, 92, head, "h", box=k)
    c.text(px + 14, 110, wprefix + ", ".join(wlab), "m", box=k)
    ox, oy = px + 64, 286
    return px, (lambda v: (ox + S * float(v[0]), oy - S * float(v[1])))


def points_and_labels(c, k, pts, weights, wlab, left_at):
    for (x, y), w, lab, slot in zip(pts, weights, wlab, ("l0", "b", "l")):
        if w > 0:
            c.out.append(f'  <circle cx="{fmt(x)}" cy="{fmt(y)}" r="5" fill="#2C2924"/>')
        else:
            c.out.append(f'  <circle cx="{fmt(x)}" cy="{fmt(y)}" r="4.5" fill="#FBF9F6" stroke="#2C2924" '
                         f'stroke-width="1.4"/>')
        if slot == "l0":
            c.text(left_at, y + 4, lab, "s", anchor="end", box=k)
        elif slot == "b":
            c.text(x, y + 20, lab, "s", anchor="middle", box=k)
        else:
            c.text(x - 11, y + 4, lab, "s", anchor="end", box=k)


def spread_figure():
    c = Canvas()
    header(c, "covTitle", "covDesc", "A covariance matrix measures how weighted points spread around their mean",
           SPREAD_DESC)
    c.text(24, 32, "A covariance matrix measures how weighted points spread around their mean", "title", box="top")
    c.text(24, 52, "Toy embeddings (0, 0), (4, 0) and (0, 4), equal scales. Ellipse: one standard deviation of "
                   "spread in every direction.", "m", box="top")
    for k, (head, weights, wlab) in enumerate(PANELS):
        px, scr = panel_frame(c, k, head, wlab)
        pts = [scr(p) for p in POINTS]
        mu, cov = mean_cov(weights)
        mx, my = scr(mu)
        a, b = float(cov[0][0]), float(cov[0][1])
        if a > 0:
            lam_big, lam_small = a - b, a + b          # [[a, b], [b, a]], b <= 0: (1, -1) carries a - b
            rx, ry = math.sqrt(lam_big) * S, math.sqrt(lam_small) * S
            c.out.append(f'  <ellipse class="spread" cx="{fmt(mx)}" cy="{fmt(my)}" rx="{fmt(rx)}" ry="{fmt(ry)}" '
                         f'transform="rotate(45 {fmt(mx)} {fmt(my)})"/>')
            half_chord = S / math.sqrt(a / (a * a - b * b))
            lx = mx + half_chord + 6
            c.line(mx + 7, my, lx - 3, my, "lead", name=f"lead{k}")
            c.text(lx, my + 4, f"mean ({num(mu[0])}, {num(mu[1])})", "m", extra="tan", box=k)
            c.out.append(f'  <circle cx="{fmt(mx)}" cy="{fmt(my)}" r="5.5" fill="#A67C52" stroke="#FBF9F6" '
                         f'stroke-width="1.5"/>')
        else:
            c.out.append(f'  <circle cx="{fmt(mx)}" cy="{fmt(my)}" r="9.5" fill="#A67C52"/>')
            c.text(mx + 16, my + 4, "mean (0, 0)", "m", extra="tan", box=k)
        left_at = pts[0][0] - 12
        if a > 0:
            # keep the (0, 0) weight label clear of the ellipse outline across the label's own height
            band = (pts[0][1] + 4 - 0.8 * 12 - 2, pts[0][1] + 4 + 0.22 * 12 + 2)
            c45 = math.cos(math.radians(45))
            outline = [(mx + rx * math.cos(u) * c45 - ry * math.sin(u) * c45,
                        my + rx * math.cos(u) * c45 + ry * math.sin(u) * c45)
                       for u in (2 * math.pi * i / 1440 for i in range(1440))]
            xs_in_band = [x for x, y in outline if band[0] <= y <= band[1]]
            if xs_in_band:
                left_at = min(left_at, min(xs_in_band) - 6)
        points_and_labels(c, k, pts, weights, wlab, left_at)
        c.text(px + 14, 334, f"variances {num(cov[0][0])} and {num(cov[0][0])}", "s", box=k)
        c.text(px + 14, 351, f"covariance {num(cov[0][1]).replace('-', chr(0x2212))}", "s", box=k)
    c.out.append('  <rect class="foot" x="24" y="376" width="712" height="34" rx="8"/>')
    c.text(380, 397, "Moving weight from (4, 0) and (0, 4) onto (0, 0) shrinks the covariance here, down to zero "
                     "when (0, 0) has all of it.", "s", anchor="middle", box="foot")
    c.out.append("</svg>")
    return c


def response(weights):
    """First-order shift of the mean per unit rise in the pushed token's score at T = 1."""
    mu, _ = mean_cov(weights)
    p = weights[PUSHED]
    offset = [POINTS[PUSHED][i] - mu[i] for i in range(2)]
    return mu, p, offset, [p * o for o in offset]


def response_figure():
    c = Canvas()
    shifts = [math.hypot(*map(float, response(w)[3])) for _, w, _ in PANELS]
    offsets = [math.hypot(*map(float, response(w)[2])) for _, w, _ in PANELS]
    desc = (
        "Three panels show the same toy embeddings and weights as the spread figure. In each, a dashed line runs "
        "from the weighted mean to the token at (4, 0), and an arrow along it shows how far the mean moves, to "
        "first order at temperature 1, per unit rise in that token's score; the arrows are drawn at twice their "
        "length. The shift is that token's probability times its offset from the mean. With weights one half, one "
        f"quarter and one quarter, the shift is {shifts[0]:.2f}: probability 0.25 times offset {offsets[0]:.2f}. "
        f"With weights 0.8, 0.1 and 0.1 it is {shifts[1]:.2f}: probability 0.1 times offset {offsets[1]:.2f}. The "
        "third panel shows the limit as all the weight moves onto (0, 0), which a softmax approaches but never "
        "reaches at finite scores: the probability of the token at (4, 0) tends to 0, and so does the mean's "
        "first-order shift, so there is no arrow. A footer states that shared weight makes the mean responsive, "
        "and that as the weights approach a point mass the first-order shift approaches zero."
    )
    defs = ('  <defs>\n    <marker id="headTan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">\n'
            '      <path d="M0,1 L9,5 L0,9 z" fill="#A67C52"/>\n    </marker>\n  </defs>')
    extra_style = ("\n    .offset{stroke:#8A8072;stroke-width:1.2;stroke-dasharray:4 3}"
                   "\n    .push{stroke:#A67C52;stroke-width:2.4}")
    title = "Raising one score moves the weighted mean, in proportion to the spread"
    header(c, "respTitle", "respDesc", title, desc, extra_style, defs)
    c.text(24, 32, title, "title", box="top")
    c.text(24, 52, "Raise the score of the token at (4, 0). Arrow: the mean's first-order shift per unit of score "
                   "at T = 1, drawn at twice its length.", "m", box="top")
    for k, (head, weights, wlab) in enumerate(PANELS):
        limit = weights[PUSHED] == 0
        px, scr = panel_frame(c, k, "Point-mass limit" if limit else head, wlab,
                              "weights tend to " if limit else "weights ")
        pts = [scr(p) for p in POINTS]
        mu, p, offset, shift = response(weights)
        mx, my = scr(mu)
        tx, ty = pts[PUSHED]
        c.line(mx, my, tx - 7, ty, "offset", name=f"offset{k}")
        if p > 0:
            ax = mx + ARROW_SCALE * S * float(shift[0])
            ay = my - ARROW_SCALE * S * float(shift[1])
            c.line(mx, my, ax, ay, "push", name=f"push{k}", marker="headTan")
            c.out.append(f'  <circle cx="{fmt(mx)}" cy="{fmt(my)}" r="5.5" fill="#A67C52" stroke="#FBF9F6" '
                         f'stroke-width="1.5"/>')
            c.text(mx, my - 12, f"mean ({num(mu[0])}, {num(mu[1])})", "m", anchor="middle", extra="tan", box=k)
        else:
            c.out.append(f'  <circle cx="{fmt(mx)}" cy="{fmt(my)}" r="9.5" fill="#A67C52"/>')
            c.text(mx + 16, my - 10, "mean (0, 0)", "m", extra="tan", box=k)
        points_and_labels(c, k, pts, weights, wlab, pts[0][0] - 12)
        size, off = math.hypot(*map(float, shift)), math.hypot(*map(float, offset))
        if limit:
            c.text(px + 14, 334, "shift tends to 0 per unit of score", "s", box=k)
            c.text(px + 14, 351, f"probability tends to 0; offset {off:.2f}", "s", box=k)
        else:
            c.text(px + 14, 334, f"shift {size:.2f} per unit of score", "s", box=k)
            c.text(px + 14, 351, f"= probability {float(p):g} \u00d7 offset {off:.2f}", "s", box=k)
    c.out.append('  <rect class="foot" x="24" y="376" width="712" height="34" rx="8"/>')
    c.text(380, 397, "Shared weight makes the mean responsive; as the weights approach a point mass, the "
                     "first-order shift approaches zero.", "s", anchor="middle", box="foot")
    c.out.append("</svg>")
    return c


def check_layout(c):
    from PIL import ImageFont

    cache = {}

    def width(s, size, bold):
        return max(cache.setdefault((p, size), ImageFont.truetype(p, size)).getlength(s) for p in FONTS[bold])

    boxes = {k: (PX[k] + 4, PY + 4, PX[k] + PW - 4, PY + PH - 4) for k in range(3)}
    boxes.update({"top": (16, 0, W - 16, 62), "foot": (28, 378, 732, 408)})
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
    canvases = {SPREAD_SVG: spread_figure(), RESPONSE_SVG: response_figure()}
    if args.check_layout:
        problems = [f"{p.name}: {msg}" for p, c in canvases.items() for msg in check_layout(c)]
        print("\n".join(problems) if problems else "layout: no collisions")
        if problems:
            return 1
    if args.verify:
        stale = [p.name for p, c in canvases.items()
                 if not p.exists() or p.read_text(encoding="ascii") != c.svg()]
        if stale:
            print("stale or missing: " + ", ".join(stale))
            return 1
        print("verified: " + ", ".join(p.name for p in canvases))
        return 0
    for path, c in canvases.items():
        path.write_text(c.svg(), encoding="ascii")
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
