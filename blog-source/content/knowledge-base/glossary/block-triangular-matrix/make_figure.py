#!/usr/bin/env python3
"""Generate the block-triangular-matrix teaching diagram and its PNG share card.

Conceptual only: four tokens with toy margins (0.40, 0.25, -0.10, 0.30) and
their running minimum. The script checks the running minimum, writes the SVG as
text and renders the 1200 x 630 card with the site's Sharp package. --verify
compares both assets byte for byte.
"""

import argparse
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
NAME = "block-triangular-matrix"
MARGINS = (0.40, 0.25, -0.10, 0.30)
RUNNING_MIN = (0.40, 0.25, -0.10, -0.10)


def check_toy_values():
    low, out = float("inf"), []
    for m in MARGINS:
        low = min(low, m)
        out.append(low)
    if tuple(out) != RUNNING_MIN:
        raise SystemExit("running minimum does not match the drawn values")


def make_svg():
    p = []
    p.append('<rect x="24" y="64" width="372" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    x0, y0, c = 112, 96, 56
    p.append(f'<text class="t m" x="{x0}" y="{y0 - 14}">state of token 1 to 4 (columns)</text>')
    p.append(f'<text class="t m" x="36" y="{y0 + 2 * c + 4}">token t</text>')
    p.append(f'<text class="t m" x="36" y="{y0 + 2 * c + 18}">(rows)</text>')
    for i in range(4):
        for j in range(4):
            x, y = x0 + j * c, y0 + i * c
            if i == j:
                unstable = i == 2
                fill, stroke = ("#F3E8E0", "#A67C52") if unstable else ("#E8F0F7", "#4B6787")
                width = "2.4" if unstable else "1.2"
                p.append(f'<rect x="{x}" y="{y}" width="{c - 4}" height="{c - 4}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
                label_y = y + c / 2 - 4 if unstable else y + c / 2 + 2
                p.append(f'<text class="t h" x="{x + c / 2 - 2:.0f}" y="{label_y:.0f}" text-anchor="middle">D{i + 1}</text>')
                if unstable:
                    p.append(f'<text class="t m" x="{x + c / 2 - 2:.0f}" y="{y + c / 2 + 12:.0f}" text-anchor="middle">unstable</text>')
            elif j < i:
                p.append(f'<rect x="{x}" y="{y}" width="{c - 4}" height="{c - 4}" rx="4" fill="#EDE8E1" stroke="#7F786D" stroke-width="0.8"/>')
                p.append(f'<text class="t s" x="{x + c / 2 - 2:.0f}" y="{y + c / 2 + 4:.0f}" text-anchor="middle">*</text>')
            else:
                p.append(f'<rect x="{x}" y="{y}" width="{c - 4}" height="{c - 4}" rx="4" fill="#FBF9F6" stroke="#C4B8A8" stroke-width="0.8" stroke-dasharray="5 4"/>')
                p.append(f'<text class="t m" x="{x + c / 2 - 2:.0f}" y="{y + c / 2 + 4:.0f}" text-anchor="middle">0</text>')
    p.append(f'<text class="t m" x="40" y="330">* earlier token affects later one; 0 never the reverse</text>')
    # Right panel: margins and running minimum.
    p.append('<rect x="412" y="64" width="324" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    p.append('<text class="t h" x="428" y="88">Token margins and the passage margin</text>')
    gx0, gx1, gy_top, gy_bot = 470, 688, 150, 284
    lo, hi = -0.2, 0.5

    def gy(v):
        return gy_bot - (v - lo) / (hi - lo) * (gy_bot - gy_top)

    def gx(t):
        return gx0 + (t - 1) * (gx1 - gx0) / 3

    p.append(f'<rect x="{gx0 - 20}" y="{gy(0):.1f}" width="{gx1 - gx0 + 40}" height="{gy(lo) - gy(0):.1f}" fill="#F3E8E0"/>')
    p.append(f'<text class="t m" x="{gx0 - 14}" y="{gy(lo) - 6:.1f}">past the edge (below 0)</text>')
    p.append(f'<line x1="{gx0 - 20}" y1="{gy(0):.1f}" x2="{gx1 + 20}" y2="{gy(0):.1f}" stroke="#A89B8C" stroke-dasharray="5 4"/>')
    p.append(f'<text class="t m" x="{gx0 - 34}" y="{gy(0) + 4:.1f}">0</text>')
    pts = []
    for t, v in enumerate(RUNNING_MIN, start=1):
        if pts:
            pts.append(f"L {gx(t):.1f} {gy(pts_last):.1f}")
        pts.append(("M" if not pts else "L") + f" {gx(t):.1f} {gy(v):.1f}")
        pts_last = v
    p.append(f'<path d="{" ".join(pts)}" fill="none" stroke="#2C2924" stroke-width="2"/>')
    for t, v in enumerate(MARGINS, start=1):
        p.append(f'<circle cx="{gx(t):.1f}" cy="{gy(v):.1f}" r="5" fill="#4B6787" stroke="#FBF9F6" stroke-width="1"/>')
        p.append(f'<text class="t m" x="{gx(t) - 14:.1f}" y="{gy_bot + 16}">token {t}</text>')
    p.append('<circle cx="438" cy="106" r="4" fill="#4B6787"/>')
    p.append('<text class="t m" x="450" y="110">dots: each token\'s own margin</text>')
    p.append('<line x1="428" y1="122" x2="446" y2="122" stroke="#2C2924" stroke-width="2"/>')
    p.append('<text class="t m" x="450" y="126">line: the passage margin, a running minimum</text>')
    body = "\n".join(p)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400" role="img" aria-labelledby="{NAME}-title {NAME}-desc">
<title id="{NAME}-title">A block-triangular Jacobian makes stability a running minimum over tokens</title>
<desc id="{NAME}-desc">Conceptual diagram with toy values, not measured data. Left: a four-by-four grid of blocks for tokens 1 to 4. Diagonal blocks D1 to D4 hold each token's response to its own state; D3 is outlined and labeled unstable. Blocks below the diagonal, marked with a star, are effects of earlier tokens on later ones; blocks above the diagonal are zero, since a later token never affects an earlier one. Right: toy token margins 0.40, 0.25, -0.10 and 0.30 drawn as dots, and the running minimum drawn as a line through 0.40, 0.25, -0.10 and -0.10, which drops below zero, into the shaded region past the edge, at token 3 and stays there. Footer: eigenvalues come from the diagonal blocks alone, so a passage is only as stable as its weakest token.</desc>
<style>.t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}.title{{font-size:15px;font-weight:700}}.h{{font-size:13px;font-weight:700}}.s{{font-size:12px}}.m{{font-size:11px;fill:#5A544C}}.arrow{{fill:none;stroke:#A67C52;stroke-width:1.8}}.axis{{stroke:#7F786D;stroke-width:1}}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">Block-triangular: only the diagonal blocks set the eigenvalues</text>
<text class="t m" x="24" y="48">Four tokens with toy margins; not a measured passage.</text>
{body}
<rect x="24" y="354" width="712" height="30" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="373">Eigenvalues come from the diagonal blocks alone, so a passage is only as stable as its weakest token.</text>
</svg>
'''


def png_bytes(svg, node_modules):
    if node_modules is None:
        roots = [Path.cwd(), *HERE.parents]
        node_modules = next((root / "node_modules" for root in roots
                             if (root / "node_modules" / "sharp").is_dir()), None)
    if node_modules is None:
        raise RuntimeError("Install the site's Node dependencies or supply --node-modules")
    return subprocess.run(
        ["node", "-e", "require('sharp')(require('fs').readFileSync(0)).resize(1200,630).png().toBuffer().then(b=>process.stdout.write(b)).catch(e=>{console.error(e);process.exit(1)})"],
        input=svg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        env={**os.environ, "NODE_PATH": str(Path(node_modules).resolve())},
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--node-modules", type=Path)
    args = parser.parse_args()
    check_toy_values()
    svg = make_svg().encode("utf-8")
    outputs = {f"{NAME}.svg": svg, f"{NAME}-card.png": png_bytes(svg, args.node_modules)}
    for name, data in outputs.items():
        path = HERE / name
        if args.verify:
            if path.read_bytes() != data:
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            path.write_bytes(data)
    print("Verified toy values, SVG and PNG." if args.verify else "Generated SVG and PNG.")


if __name__ == "__main__":
    main()
