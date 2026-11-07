#!/usr/bin/env python3
"""Generate a conceptual soft-token feedback diagram and PNG share card.

Requires Python 3, Node.js, and the site's installed Sharp package. Toy arithmetic uses
exact fractions; --verify compares the saved SVG and PNG byte for byte.
No measurements or language-model runs are performed.
"""

import argparse
from fractions import Fraction
import os
from pathlib import Path
import subprocess


def make_svg():
    embeddings = ((2, 0), (0, 4))
    weights = (Fraction(3, 4), Fraction(1, 4))
    contributions = tuple(tuple(w * x for x in e) for w, e in zip(weights, embeddings))
    result = tuple(sum(row[k] for row in contributions) for k in range(2))
    assert sum(weights) == 1
    assert result == (Fraction(3, 2), Fraction(1))
    first = f"({float(contributions[0][0]):g}, {float(contributions[0][1]):g})"
    second = f"({float(contributions[1][0]):g}, {float(contributions[1][1]):g})"
    average = f"({float(result[0]):g}, {float(result[1]):g})"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400" role="img" aria-labelledby="soft-token-title soft-token-desc">
<title id="soft-token-title">The average embedding returns to one fixed input slot</title>
<desc id="soft-token-desc">Toy update, not measured model data. The model takes a fixed prompt and one changing vector. Probabilities three quarters and one quarter weight embeddings (2, 0) and (0, 4). Contributions {first} and {second} add to {average}, which replaces the old vector at the same position. The next step recomputes probabilities.</desc>
<defs><marker id="arrowhead" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="#A67C52"/></marker></defs>
<style>.t{{font-family:Arial,Helvetica,sans-serif;fill:#2C2924}}.title{{font-size:15px;font-weight:700}}.h{{font-size:13px;font-weight:700}}.s{{font-size:12px}}.m{{font-size:11px;fill:#5A544C}}.arrow{{fill:none;stroke:#A67C52;stroke-width:1.8}}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">Soft-token feedback: average the embeddings, then replace one slot</text>
<text class="t m" x="24" y="48">One toy update with two tokens. No measured data and no claim about the next step.</text>
<path d="M 620 150 V 98 H 120 V 139" class="arrow"/>
<path d="M 115 139 L 120 147 L 125 139 z" fill="#A67C52"/>
<text class="t s" x="230" y="85">Return the new vector to the same slot</text>
<rect x="24" y="150" width="190" height="158" rx="10" fill="#E8F0F7" stroke="#4B6787"/>
<text class="t h" x="40" y="177">Model input</text>
<rect x="40" y="192" width="158" height="34" rx="6" fill="#FBF9F6" stroke="#C4B8A8"/>
<text class="t s" x="54" y="214">Fixed prompt</text>
<rect x="40" y="237" width="158" height="34" rx="6" fill="#FBF9F6" stroke="#4B6787"/>
<text class="t s" x="54" y="259">Current vector slot</text>
<text class="t m" x="40" y="292">Its position stays fixed.</text>
<path d="M 220 229 H 244" class="arrow"/>
<path d="M 244 224 L 252 229 L 244 234 z" fill="#A67C52"/>
<rect x="254" y="150" width="202" height="158" rx="10" fill="#FBF9F6" stroke="#C4B8A8"/>
<text class="t h" x="270" y="177">Model probabilities</text>
<text class="t s" x="270" y="211">Token A: three quarters</text>
<text class="t s" x="270" y="236">Token B: one quarter</text>
<text class="t m" x="270" y="269">Recompute at every step.</text>
<text class="t m" x="270" y="289">Do not select one token.</text>
<path d="M 462 229 H 486" class="arrow"/>
<path d="M 486 224 L 494 229 L 486 234 z" fill="#A67C52"/>
<rect x="496" y="150" width="240" height="158" rx="10" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t h" x="512" y="177">Weighted embedding average</text>
<text class="t s" x="512" y="207">Three quarters of (2, 0): {first}</text>
<text class="t s" x="512" y="232">One quarter of (0, 4): {second}</text>
<path d="M 512 245 H 720" stroke="#6F8D5E"/>
<text class="t h" x="512" y="269">New vector: {average}</text>
<text class="t m" x="512" y="290">Neither individual embedding.</text>
<rect x="24" y="352" width="712" height="28" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="370">One changing vector returns; the prompt is unchanged and the context does not grow.</text>
</svg>
'''


def png_bytes(svg, node_modules):
    if node_modules is None:
        roots = [Path.cwd(), *Path(__file__).resolve().parents]
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
    outputs = {"soft-token-feedback.svg": svg, "soft-token-feedback-card.png": png_bytes(svg, args.node_modules)}
    for name, data in outputs.items():
        path = Path(__file__).resolve().parent / name
        if args.verify:
            if path.read_bytes() != data:
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            path.write_bytes(data)
    print("Verified exact toy average, SVG and PNG." if args.verify else "Generated toy feedback SVG and PNG.")


if __name__ == "__main__":
    main()
