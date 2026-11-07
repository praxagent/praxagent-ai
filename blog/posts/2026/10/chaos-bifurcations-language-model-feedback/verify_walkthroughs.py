#!/usr/bin/env python3
"""Verify the numbers of the post's two step-by-step readings, and of later receipt-backed additions, against their pinned receipts.

Run: python verify_walkthroughs.py --verify [--extension ID] [--research-repo /path/to/neural-shrimp]
The default extension is walkthroughs-20261101; huginn-item18-20261103 covers the frozen test without item 18.
Each added number is rendered again from its receipt field and must appear in index.md. With a research repository, the
receipts and figures must also equal the files at the pinned commit. This does not rerun the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

FIGURES = {"walkthroughs-20261101": ("fig-branch-three-readings.png", "fig-passage-steps.png", "fig-free-orbit-words.png")}


def field(obj, path):
    for part in path:
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def render(value, style):
    if style == "d":
        if value != int(value):
            raise ValueError("noninteger count")
        return str(int(value))
    if style.startswith("ceil"):
        places = int(style[4:])
        return format(math.ceil(value * 10 ** places) / 10 ** places, f".{places}f")
    if style in ("sci0", "sci1"):
        mantissa, exponent = format(value, f".{style[3]}e").split("e")
        return mantissa + r"\times10^{" + str(int(exponent)) + "}"
    if style == "pct1":
        return format(100 * value, ".1f")
    return format(value, style)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--extension", default="walkthroughs-20261101")
    parser.add_argument("--research-repo", type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    manifest = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    ext = next(e for e in manifest["extensions"] if e["id"] == args.extension)
    prose = (bundle / "index.md").read_text(encoding="utf-8").replace("−", "-")
    receipts = {}
    for name, meta in ext["source_receipts"].items():
        data = (bundle / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError(f"{name}: hash differs from the manifest")
        receipts[name] = json.loads(data)
        if args.research_repo:
            pinned = subprocess.check_output(["git", "show", f"{meta['commit']}:{meta['source_path']}"], cwd=args.research_repo)
            if pinned != data:
                raise ValueError(f"{name}: differs from the pinned commit")
    for entry in ext["new_numbers"]:
        shown = render(field(receipts[entry["receipt"]], entry["field"]), entry["style"])
        if shown != entry["appears_as"].replace("−", "-"):
            raise ValueError(f"{entry['id']}: {shown} against {entry['appears_as']}")
        if shown not in prose:
            raise ValueError(f"{entry['id']}: {shown} not in index.md")
    if args.research_repo:
        for fig in FIGURES.get(ext["id"], ()):
            pinned = subprocess.check_output(["git", "show", f"{ext['source_commit']}:docs/figure-walkthroughs/{fig}"], cwd=args.research_repo)
            if pinned != (bundle / fig).read_bytes():
                raise ValueError(f"{fig}: differs from the pinned commit")
    print(f"{ext['id']} verified: {len(ext['new_numbers'])} numbers, {len(ext['source_receipts'])} receipts")


if __name__ == "__main__":
    main()
