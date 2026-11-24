#!/usr/bin/env python3
"""Generate the looped-transformer teaching diagram and its PNG share card.

Conceptual only: no measurements or model runs. The eight layers on the left are
a toy count. The right side draws the prelude, the shared core and the coda as
one box each and does not say how many blocks the prelude or coda holds; the
core box notes only that it may hold one or more. The SVG is written as text;
the 1200 x 630 card is rendered from it with the site's Sharp package. --verify
rebuilds both in memory and compares them byte for byte.
"""

import argparse
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
NAME = "looped-transformer"


def box(x, y, w, h, fill, stroke, label, cls="s"):
    cx, cy = x + w / 2, y + h / 2 + 4
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>'
            f'<text class="t {cls}" x="{cx:g}" y="{cy:g}" text-anchor="middle">{label}</text>')


def make_svg():
    parts = []
    # Left: a tall stack of distinct layers, each used once.
    parts.append('<rect x="24" y="64" width="300" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    parts.append('<text class="t h" x="40" y="88">Ordinary transformer</text>')
    parts.append('<text class="t m" x="40" y="106">8 distinct layers, each used once</text>')
    for i in range(8):
        y = 300 - i * 24
        parts.append(box(110, y, 128, 20, "#E8F0F7", "#4B6787", f"layer {i + 1}", cls="m"))
    parts.append('<text class="t m" x="40" y="330">input at the bottom, prediction at the top</text>')
    # Right: prelude, one shared core run K times, coda.
    parts.append('<rect x="344" y="64" width="392" height="276" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>')
    parts.append('<text class="t h" x="360" y="88">Looped transformer</text>')
    parts.append('<text class="t m" x="360" y="106">one shared core, run K times; K is chosen at run time</text>')
    parts.append(box(362, 196, 74, 40, "#EDE8E1", "#7F786D", "prelude"))
    parts.append('<rect x="486" y="196" width="120" height="40" rx="6" fill="#F3E8E0" stroke="#A67C52"/>')
    parts.append('<text class="t h" x="546" y="213" text-anchor="middle">core</text>')
    parts.append('<text class="t m" x="546" y="228" text-anchor="middle">one or more blocks</text>')
    parts.append(box(650, 196, 70, 40, "#EDE8E1", "#7F786D", "coda"))
    parts.append('<path d="M 436 216 H 478" class="arrow"/><path d="M 478 211 L 486 216 L 478 221 z" fill="#A67C52"/>')
    parts.append('<path d="M 606 216 H 642" class="arrow"/><path d="M 642 211 L 650 216 L 642 221 z" fill="#A67C52"/>')
    parts.append('<text class="t m" x="444" y="208">e</text>')
    # The loop arc over the core.
    parts.append('<path d="M 570 196 C 570 140, 522 140, 522 190" class="arrow"/>')
    parts.append('<path d="M 517 184 L 522 193 L 527 184 z" fill="#A67C52"/>')
    parts.append('<text class="t s" x="546" y="138" text-anchor="middle">fed back in for the next pass</text>')
    # The input re-entering on every pass (dashed).
    parts.append('<path d="M 450 216 V 272 H 530 V 244" fill="none" stroke="#A89B8C" stroke-width="1.6" stroke-dasharray="5 4"/>')
    parts.append('<path d="M 525 248 L 530 240 L 535 248 z" fill="#A89B8C"/>')
    parts.append('<text class="t m" x="456" y="290">in many designs the input</text>')
    parts.append('<text class="t m" x="456" y="304">representation e re-enters each pass</text>')
    parts.append('<text class="t m" x="614" y="262">same weights</text>')
    parts.append('<text class="t m" x="614" y="276">on every pass</text>')
    body = "\n".join(parts)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400" role="img" aria-labelledby="{NAME}-title {NAME}-desc">
<title id="{NAME}-title">A tall stack of distinct layers beside one shared core run several times</title>
<desc id="{NAME}-desc">Conceptual diagram, no measured data. Left: an ordinary transformer drawn as eight distinct layers, each used once. Right: a looped transformer drawn as a prelude, one shared core, which may hold one or more transformer blocks, and a coda. An arrow labeled fed back in for the next pass returns the core's output to the core; the core runs K times in all, where K is the number of passes chosen at run time. A dashed path shows the prelude output e re-entering the core on every pass, as it does in many designs. Footer: depth becomes a number of passes chosen at run time, and more passes need not settle or help.</desc>
<style>.t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}.title{{font-size:15px;font-weight:700}}.h{{font-size:13px;font-weight:700}}.s{{font-size:12px}}.m{{font-size:11px;fill:#5A544C}}.arrow{{fill:none;stroke:#A67C52;stroke-width:1.8}}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">A looped transformer reuses one shared core</text>
<text class="t m" x="24" y="48">Conceptual drawing; the eight-layer stack is a toy count. No model or measurement is shown.</text>
{body}
<rect x="24" y="354" width="712" height="30" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="373">Depth becomes a number of passes chosen at run time. More passes need not settle, or help.</text>
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
    svg = make_svg().encode("utf-8")
    outputs = {f"{NAME}.svg": svg, f"{NAME}-card.png": png_bytes(svg, args.node_modules)}
    for name, data in outputs.items():
        path = HERE / name
        if args.verify:
            if path.read_bytes() != data:
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            path.write_bytes(data)
    print("Verified SVG and PNG." if args.verify else "Generated SVG and PNG.")


if __name__ == "__main__":
    main()
