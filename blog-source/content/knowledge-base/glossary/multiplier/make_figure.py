#!/usr/bin/env python3
"""Generate the multiplier teaching diagram and its PNG share card.

Conceptual only: the multipliers drawn are toy values (0.5, -0.3 and the pair
0.4 +/- 0.5i), not measurements. The SVG is written as text with coordinates
computed from those values; the 1200 x 630 card is rendered from it with the
site's Sharp package. --verify checks the toy values and compares both assets
byte for byte.
"""

import argparse
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
NAME = "multiplier"
CX, CY, UNIT = 224, 196, 100
TOY = (0.5, -0.3, complex(0.4, 0.5), complex(0.4, -0.5))


def X(z):
    return CX + UNIT * complex(z).real


def Y(z):
    return CY - UNIT * complex(z).imag


def check_toy_values():
    if any(abs(z) >= 1 for z in TOY):
        raise SystemExit("toy multipliers must lie inside the unit circle")


def arrow(z0, z1):
    x0, y0, x1, y1 = X(z0), Y(z0), X(z1), Y(z1)
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    bx, by = x1 - 9 * ux, y1 - 9 * uy
    px, py = -uy * 4.5, ux * 4.5
    return (f'<path d="M {x0:.1f} {y0:.1f} L {bx:.1f} {by:.1f}" class="arrow"/>'
            f'<path d="M {bx + px:.1f} {by + py:.1f} L {x1:.1f} {y1:.1f} L {bx - px:.1f} {by - py:.1f} z" fill="#A67C52"/>')


def make_svg():
    p = []
    p.append('<rect x="24" y="64" width="400" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    p.append(f'<line x1="{CX - 190}" y1="{CY}" x2="{CX + 190}" y2="{CY}" class="axis"/>')
    p.append(f'<line x1="{CX}" y1="{CY - 116}" x2="{CX}" y2="{CY + 112}" class="axis"/>')
    p.append(f'<circle cx="{CX}" cy="{CY}" r="{UNIT}" fill="#E8F0F7" fill-opacity="0.6" stroke="#4B6787" stroke-width="1.6"/>')
    p.append(f'<text class="t m" x="{CX + 190}" y="{CY + 16}" text-anchor="end">real axis</text>')
    p.append(f'<text class="t m" x="{CX - 6}" y="{CY - 110}" text-anchor="end">imaginary axis</text>')
    p.append(f'<text class="t m" x="{X(-1) - 22:.0f}" y="{CY + 16}">-1</text>')
    p.append(f'<text class="t m" x="{X(1) + 4:.0f}" y="{CY + 16}">+1</text>')
    p.append(f'<text class="t m" x="{CX - 70}" y="{CY - 56}">unit circle</text>')
    for z in TOY:
        p.append(f'<circle cx="{X(z):.1f}" cy="{Y(z):.1f}" r="4.5" fill="#4B6787"/>')
    p.append(f'<text class="t m" x="{X(0.5) - 8:.0f}" y="{CY - 9}">0.5</text>')
    p.append(f'<text class="t m" x="{X(-0.3) - 12:.0f}" y="{CY - 9}">-0.3</text>')
    p.append(f'<text class="t m" x="{X(complex(0.4, 0.5)):.0f}" y="{Y(complex(0.4, 0.5)) - 10:.0f}" text-anchor="middle">0.4 + 0.5i</text>')
    p.append(f'<text class="t m" x="{X(complex(0.4, -0.5)):.0f}" y="{Y(complex(0.4, -0.5)) + 18:.0f}" text-anchor="middle">0.4 - 0.5i</text>')
    p.append(arrow(-0.72, -1.36))
    p.append(arrow(0.72, 1.36))
    p.append(arrow(complex(0.50, 0.80), complex(0.65, 1.10)))
    p.append(arrow(complex(0.50, -0.80), complex(0.65, -1.10)))
    p.append(f'<text class="t s" x="{X(-1.36):.0f}" y="{CY - 12}">1</text>')
    p.append(f'<text class="t s" x="{X(1.30):.0f}" y="{CY - 12}">2</text>')
    p.append(f'<text class="t s" x="{X(complex(0.65, 1.10)) + 6:.0f}" y="{Y(complex(0.65, 1.10)) + 4:.0f}">3</text>')
    p.append('<text class="t m" x="40" y="330">Toy multipliers inside the circle (dots); arrows: ways out.</text>')
    cards = [
        ("1  Through -1: period doubling", ["A nudge flips sign every step as it", "grows; typically a period-two cycle is", "born or destroyed."]),
        ("2  Through +1: fold", ["The nudge grows without flipping;", "typically this fixed point merges with", "another and both vanish."]),
        ("3  Complex pair: Neimark-Sacker", ["The nudge turns as it grows, by an angle", "that need not be small; typically an", "invariant circle is born or destroyed."]),
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
<title id="{NAME}-title">Multipliers inside the unit circle make a fixed point attract; three ways out</title>
<desc id="{NAME}-desc">Conceptual diagram with toy values, not measured data. Left: the complex plane, with a horizontal real axis and a vertical imaginary axis, and the unit circle shaded. Four toy multipliers lie inside it: 0.5, -0.3, and the complex pair 0.4 plus and minus 0.5i, where i is the imaginary unit, the square root of -1. Arrows show three ways a multiplier can leave the circle as a parameter changes: 1, through -1 on the real axis; 2, through +1; 3, as a complex pair crossing elsewhere. Right: the usual bifurcation at each exit, not a guaranteed outcome. Through -1, a small nudge flips sign every step as it grows, and typically a cycle of period two is born or destroyed, a period doubling. Through +1, the nudge grows without flipping, and typically this fixed point merges with another and both vanish, a fold. As a complex pair, the nudge turns as it grows, by an angle that need not be small, and typically an invariant circle is born or destroyed, a Neimark-Sacker bifurcation. Footer: inside the unit circle small nudges eventually die away; each exit marks the usual bifurcation, not a certain one.</desc>
<style>.t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}.title{{font-size:15px;font-weight:700}}.h{{font-size:13px;font-weight:700}}.s{{font-size:12px}}.m{{font-size:11px;fill:#5A544C}}.arrow{{fill:none;stroke:#A67C52;stroke-width:1.8}}.axis{{stroke:#7F786D;stroke-width:1}}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">Multipliers decide whether a fixed point attracts</text>
<text class="t m" x="24" y="48">Toy values on the complex plane; not a measured spectrum.</text>
{body}
<rect x="24" y="354" width="712" height="30" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="373">Inside the unit circle small nudges eventually die away; each exit marks the usual bifurcation, not a certain one.</text>
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
