#!/usr/bin/env python3
"""Generate the stability-edge teaching diagram and its PNG share card.

Conceptual only: the two multipliers drawn, -0.6 and 0.9, are toy values. For a
relaxed step lambda, the multipliers that stay strictly inside the unit circle
form the open disk (boundary excluded) with center 1 - 1/lambda and radius
1/lambda; the script draws those disks for steps 1, 1.25 and 2 and checks the
toy edges (1.25 and 20) exactly with fractions. --verify also compares the SVG
and the 1200 x 630 card byte for byte.
"""

import argparse
from fractions import Fraction
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
NAME = "stability-edge"
CX, CY, UNIT = 190, 196, 104
STEPS = (Fraction(1), Fraction(5, 4), Fraction(2))
TOY = (Fraction(-3, 5), Fraction(9, 10))


def edge(mu):
    return 2 / (1 - mu)


def check_toy_values():
    if edge(TOY[0]) != Fraction(5, 4) or edge(TOY[1]) != 20:
        raise SystemExit("toy edges are not 1.25 and 20")
    center, radius = 1 - 1 / STEPS[1], 1 / STEPS[1]
    if abs(TOY[0] - center) != radius:
        raise SystemExit("-0.6 must lie on the boundary of the step-1.25 disk")


def make_svg():
    p = []
    p.append('<rect x="24" y="64" width="400" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    p.append(f'<line x1="{CX - 150}" y1="{CY}" x2="{CX + 200}" y2="{CY}" class="axis"/>')
    p.append(f'<line x1="{CX}" y1="{CY - 116}" x2="{CX}" y2="{CY + 112}" class="axis"/>')
    styles = [("#EDE8E1", "#7F786D", ""), ("#E8F0F7", "#4B6787", ' stroke-dasharray="5 4"'), ("#EAF1E5", "#6F8D5E", "")]
    labels = ["step 1 (unrelaxed)", "step 1.25", "step 2"]
    for lam, (fill, stroke, dash), label in zip(STEPS, styles, labels):
        cx = CX + UNIT * float(1 - 1 / lam)
        r = UNIT * float(1 / lam)
        p.append(f'<circle cx="{cx:.1f}" cy="{CY}" r="{r:.1f}" fill="{fill}" fill-opacity="0.55" stroke="{stroke}" stroke-width="1.6"{dash}/>')
    for i, ((fill, stroke, dash), label) in enumerate(zip(styles, labels)):
        ly = 284 + 16 * i
        p.append(f'<line x1="288" y1="{ly - 4}" x2="310" y2="{ly - 4}" stroke="{stroke}" stroke-width="2"{dash}/>')
        p.append(f'<text class="t m" x="316" y="{ly}">{label}</text>')
    one = CX + UNIT
    p.append(f'<circle cx="{one}" cy="{CY}" r="3" fill="#2C2924"/>')
    p.append(f'<text class="t s" x="{one + 6}" y="{CY + 18}">1</text>')
    p.append(f'<text class="t m" x="{CX - UNIT - 18}" y="{CY + 18}">-1</text>')
    a = CX + UNIT * float(TOY[0])
    b = CX + UNIT * float(TOY[1])
    p.append(f'<circle cx="{a:.1f}" cy="{CY}" r="5" fill="#A67C52"/>')
    p.append(f'<circle cx="{b:.1f}" cy="{CY}" r="5" fill="#A67C52"/>')
    p.append(f'<text class="t s" x="{a - 30:.0f}" y="{CY + 30}">-0.6</text>')
    p.append(f'<text class="t s" x="{b - 12:.0f}" y="{CY - 10}">0.9</text>')
    p.append('<text class="t m" x="40" y="330">Every disk passes through the point 1; longer steps shrink it.</text>')
    cards = [
        ("The safe region shrinks", ["For a step lambda, a multiplier stays", "inside the unit circle only within a", "disk of radius 1/lambda touching 1."]),
        ("-0.6: edge at step 1.25", ["The step-1.25 disk passes through -0.6,", "so a step 25% longer than step 1 moves", "this multiplier onto the circle at -1."]),
        ("0.9: edge at step 20", ["A multiplier near +1 slows settling but", "stays inside every disk drawn, so here", "the slowest one does not set the edge."]),
    ]
    y = 64
    for head, lines in cards:
        p.append(f'<rect x="440" y="{y}" width="296" height="86" rx="10" fill="#F3E8E0" stroke="#A67C52"/>')
        p.append(f'<text class="t h" x="456" y="{y + 22}">{head}</text>')
        for i, line in enumerate(lines):
            p.append(f'<text class="t s" x="456" y="{y + 42 + 16 * i}">{line}</text>')
        y += 95
    body = "\n".join(p)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400" role="img" aria-labelledby="{NAME}-title {NAME}-desc">
<title id="{NAME}-title">Longer steps shrink the region where a fixed point stays stable</title>
<desc id="{NAME}-desc">Conceptual diagram with toy values, not measured data. Left: the complex plane with three nested disks that all touch the point 1: the unit disk for step 1, a dashed disk of radius 0.8 for step 1.25, and a disk of radius 0.5 for step 2. Two toy multipliers sit on the real axis: -0.6, which lies exactly on the boundary of the step-1.25 disk, and 0.9, which lies inside all three. Right: for a step lambda, a multiplier stays inside the unit circle only within a disk of radius 1/lambda touching 1; the multiplier -0.6 reaches the circle at -1 at step 1.25, 25 percent longer than the unrelaxed step 1; the multiplier 0.9, near +1, slows settling but stays inside every disk drawn, and its edge is at step 20, so here the slowest multiplier does not set the edge. Footer: the edge is set by the multiplier the shrinking disk reaches first, which need not be the largest one.</desc>
<style>.t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}.title{{font-size:15px;font-weight:700}}.h{{font-size:13px;font-weight:700}}.s{{font-size:12px}}.m{{font-size:11px;fill:#5A544C}}.arrow{{fill:none;stroke:#A67C52;stroke-width:1.8}}.axis{{stroke:#7F786D;stroke-width:1}}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">The stability edge: how far a step can grow</text>
<text class="t m" x="24" y="48">Toy multipliers on the complex plane; not a measured spectrum.</text>
{body}
<rect x="24" y="354" width="712" height="30" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="373">The edge is set by the multiplier the shrinking disk reaches first, which need not be the largest one.</text>
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
