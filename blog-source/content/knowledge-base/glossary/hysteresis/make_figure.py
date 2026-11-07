#!/usr/bin/env python3
"""Generate a conceptual relay diagram and PNG share card; no measured data.

Requires Python 3, Node.js, and the site's installed Sharp package. All toy outputs are
derived from relay(), and --verify compares both saved assets byte for byte.
"""

import argparse
import os
from pathlib import Path
import subprocess


def relay(control, previous):
    if previous not in (0, 1):
        raise ValueError("previous state must be a bit")
    if control <= -1:
        return 0
    if control >= 1:
        return 1
    return previous


def sequence(controls, previous):
    states = []
    for control in controls:
        previous = relay(control, previous)
        states.append(previous)
    return states


def make_svg():
    for initial in (0, 1):
        assert sequence((-2, 0, 0), initial) == [0, 0, 0]
        assert sequence((2, 0, 0), initial) == [1, 1, 1]
    assert relay(-1, 1) == 0
    assert relay(1, 0) == 1
    assert relay(0, 0) == 0 and relay(0, 1) == 1
    rows = []
    for control, y, label in ((-2, 134, "First history"), (2, 244, "Second history")):
        written, held = sequence((control, 0), 0)
        display_control = f"{control:+d}"
        rows.append(f'''<text class="t m" x="24" y="{y - 10}">{label}</text>
<rect x="24" y="{y}" width="204" height="68" rx="10" fill="#E8F0F7" stroke="#4B6787"/>
<text class="t h" x="40" y="{y + 25}">Control {display_control}: write {written}</text>
<text class="t s" x="40" y="{y + 47}">Either starting bit becomes {written}.</text>
<path d="M 238 {y + 34} H 295" class="arrow"/>
<path d="M 295 {y + 29} L 303 {y + 34} L 295 {y + 39} z" fill="#A67C52"/>
<rect x="307" y="{y}" width="188" height="68" rx="10" fill="#FBF9F6" stroke="#7F786D"/>
<text class="t h" x="325" y="{y + 25}">Control 0: hold</text>
<text class="t s" x="325" y="{y + 47}">Retain the previous bit.</text>
<path d="M 505 {y + 34} H 562" class="arrow"/>
<path d="M 562 {y + 29} L 570 {y + 34} L 562 {y + 39} z" fill="#A67C52"/>
<rect x="574" y="{y}" width="162" height="68" rx="10" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t h" x="592" y="{y + 25}">Read bit {held}</text>
<text class="t s" x="592" y="{y + 47}">At control 0.</text>''')
    return '''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="400" viewBox="0 0 760 400" role="img" aria-labelledby="hysteresis-title hysteresis-desc">
<title id="hysteresis-title">Different histories retain different bits at one common control</title>
<desc id="hysteresis-desc">Toy relay, not measured data. Control minus two writes zero, and plus two writes one. Returning either history to control zero retains the written bit. Repeating the hold retains it under the exact toy rule, with thresholds minus one and plus one.</desc>
<defs><marker id="arrowhead" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="#A67C52"/></marker></defs>
<style>.t{font-family:Arial,Helvetica,sans-serif;fill:#2C2924}.title{font-size:15px;font-weight:700}.h{font-size:13px;font-weight:700}.s{font-size:12px}.m{font-size:11px;fill:#5A544C}.arrow{fill:none;stroke:#A67C52;stroke-width:1.8}</style>
<rect width="760" height="400" fill="#F7F4F0"/>
<text class="t title" x="24" y="28">Hysteresis: the present control does not determine the bit</text>
<text class="t m" x="24" y="48">Toy relay with switching thresholds at minus one and plus one. No measured data.</text>
<text class="t h" x="24" y="90">1. Write using different histories</text>
<text class="t h" x="307" y="90">2. Hold the same control</text>
<text class="t h" x="574" y="90">3. Read the bit</text>
<rect x="297" y="112" width="208" height="212" rx="12" fill="none" stroke="#7F786D" stroke-dasharray="5 4"/>
''' + "\n".join(rows) + '''
<rect x="24" y="352" width="712" height="28" rx="8" fill="#EAF1E5" stroke="#6F8D5E"/>
<text class="t s" x="40" y="370">At control zero, the current bit records which write happened earlier.</text>
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
    outputs = {"hysteresis.svg": svg, "hysteresis-card.png": png_bytes(svg, args.node_modules)}
    for name, data in outputs.items():
        path = Path(__file__).resolve().parent / name
        if args.verify:
            if path.read_bytes() != data:
                raise SystemExit(f"asset differs from generator: {name}")
        else:
            path.write_bytes(data)
    print("Verified toy relay, SVG and PNG." if args.verify else "Generated toy relay SVG and PNG.")


if __name__ == "__main__":
    main()
