#!/usr/bin/env python3
"""Generate a conceptual four-quadrant guide to PC1 and PC2 kernel scores."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


OUTPUT_DEFAULT = (
    Path(__file__).resolve().parents[4]
    / "static/knowledge-base/deep-dives/pca-toy-score-extremes.svg"
)


@dataclass(frozen=True)
class ToyKernel:
    pc1: str
    pc2: str
    center_x: float
    center_y: float
    length: float
    width: float
    asymmetry: float
    groove_fraction: float


KERNELS = (
    ToyKernel("low", "high", 300, 205, 126, 82, 2, 0.48),
    ToyKernel("high", "high", 730, 205, 184, 120, 3, 0.48),
    ToyKernel("low", "low", 300, 438, 142, 60, 16, 0.72),
    ToyKernel("high", "low", 730, 438, 205, 86, 22, 0.72),
)


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def kernel_path(kernel: ToyKernel) -> str:
    """Return a smooth seed-like outline whose parameters encode the toy traits."""
    cx = kernel.center_x
    cy = kernel.center_y
    half_length = kernel.length / 2
    half_width = kernel.width / 2
    skew = kernel.asymmetry
    return (
        f"M {fmt(cx - half_length)} {fmt(cy)} "
        f"C {fmt(cx - half_length * 0.72)} {fmt(cy - half_width - skew * 0.18)}, "
        f"{fmt(cx + half_length * 0.35)} {fmt(cy - half_width + skew * 0.08)}, "
        f"{fmt(cx + half_length)} {fmt(cy - skew * 0.12)} "
        f"C {fmt(cx + half_length * 0.55)} {fmt(cy + half_width + skew * 0.22)}, "
        f"{fmt(cx - half_length * 0.35)} {fmt(cy + half_width - skew * 0.05)}, "
        f"{fmt(cx - half_length)} {fmt(cy)} Z"
    )


def groove_path(kernel: ToyKernel) -> str:
    half_groove = kernel.length * kernel.groove_fraction / 2
    shift = kernel.asymmetry * 0.18
    return (
        f"M {fmt(kernel.center_x - half_groove)} {fmt(kernel.center_y + shift)} "
        f"Q {fmt(kernel.center_x)} {fmt(kernel.center_y - shift * 0.7)}, "
        f"{fmt(kernel.center_x + half_groove)} {fmt(kernel.center_y - shift)}"
    )


def kernel_group(kernel: ToyKernel) -> str:
    label = f"{kernel.pc1} PC1, {kernel.pc2} PC2"
    return f'''  <g data-pc1="{kernel.pc1}" data-pc2="{kernel.pc2}" aria-label="{label}">
    <path class="kernel" d="{kernel_path(kernel)}"/>
    <path class="groove" d="{groove_path(kernel)}"/>
    <circle class="center-mark" cx="{fmt(kernel.center_x)}" cy="{fmt(kernel.center_y)}" r="3"/>
  </g>'''


def render_svg() -> str:
    kernels = "\n".join(kernel_group(kernel) for kernel in KERNELS)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650" role="img" aria-labelledby="toyPcaTitle toyPcaDesc">
  <title id="toyPcaTitle">Toy kernels at high and low PC1 and PC2 scores</title>
  <desc id="toyPcaDesc">A two-by-two conceptual guide shows four toy wheat kernels. The left column has low PC1 scores and smaller overall kernels; the right column has high PC1 scores and larger overall kernels. The top row has high PC2 scores and rounder, more symmetric kernels with shorter centered grooves. The bottom row has low PC2 scores and more elongated, asymmetric kernels with longer off-center grooves. The drawings are teaching cartoons, not reconstructed observations.</desc>
  <defs>
    <marker id="toyArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 Z" fill="#8A7355"/>
    </marker>
    <style>
      .t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}
      .title{{font-size:20px;font-weight:700}}
      .heading{{font-size:15px;font-weight:700}}
      .label{{font-size:13px;font-weight:700}}
      .body{{font-size:12px;fill:#4F4942}}
      .muted{{font-size:12px;fill:#5A544C}}
      .grid{{stroke:#C4B8A8;stroke-width:1.4}}
      .axis{{stroke:#8A7355;stroke-width:2;marker-start:url(#toyArrow);marker-end:url(#toyArrow)}}
      .kernel{{fill:#E8D6A9;stroke:#6F573D;stroke-width:2.2}}
      .groove{{fill:none;stroke:#4B6787;stroke-width:3;stroke-linecap:round}}
      .center-mark{{fill:#4B6787}}
      .tag{{fill:#F7F3EC;stroke:#B9A88F;stroke-width:1}}
    </style>
  </defs>
  <rect width="1000" height="650" fill="#F7F4F0"/>
  <text x="30" y="36" class="t title">PC1 and PC2 create four score combinations</text>
  <text x="30" y="59" class="t muted">Conceptual extremes for reading the map, not four observed or reconstructed kernels</text>

  <text x="300" y="96" text-anchor="middle" class="t heading">LOW PC1</text>
  <text x="300" y="116" text-anchor="middle" class="t body">smaller across the shared size measurements</text>
  <text x="730" y="96" text-anchor="middle" class="t heading">HIGH PC1</text>
  <text x="730" y="116" text-anchor="middle" class="t body">larger across the shared size measurements</text>

  <text x="22" y="178" class="t heading">HIGH PC2</text>
  <text x="22" y="199" class="t body">more compact</text>
  <text x="22" y="217" class="t body">less asymmetric</text>
  <text x="22" y="235" class="t body">shorter groove</text>
  <text x="22" y="410" class="t heading">LOW PC2</text>
  <text x="22" y="431" class="t body">less compact</text>
  <text x="22" y="449" class="t body">more asymmetric</text>
  <text x="22" y="467" class="t body">longer groove</text>

  <line x1="515" y1="132" x2="515" y2="520" class="grid"/>
  <line x1="145" y1="324" x2="930" y2="324" class="grid"/>
{kernels}

  <rect x="221" y="270" width="158" height="30" rx="15" class="tag"/>
  <text x="300" y="290" text-anchor="middle" class="t label">low PC1, high PC2</text>
  <rect x="651" y="270" width="158" height="30" rx="15" class="tag"/>
  <text x="730" y="290" text-anchor="middle" class="t label">high PC1, high PC2</text>
  <rect x="221" y="503" width="158" height="30" rx="15" class="tag"/>
  <text x="300" y="523" text-anchor="middle" class="t label">low PC1, low PC2</text>
  <rect x="651" y="503" width="158" height="30" rx="15" class="tag"/>
  <text x="730" y="523" text-anchor="middle" class="t label">high PC1, low PC2</text>

  <line x1="210" y1="570" x2="820" y2="570" class="axis"/>
  <text x="515" y="560" text-anchor="middle" class="t label">PC1: broad size-related direction</text>
  <text x="515" y="599" text-anchor="middle" class="t body">Move horizontally to compare PC1 while holding the illustrated PC2 shape type fixed.</text>
  <text x="515" y="621" text-anchor="middle" class="t body">Move vertically to compare PC2 while holding the illustrated PC1 size level fixed.</text>
</svg>
'''


def verify_concept() -> None:
    combinations = {(kernel.pc1, kernel.pc2) for kernel in KERNELS}
    assert combinations == {
        ("low", "low"),
        ("low", "high"),
        ("high", "low"),
        ("high", "high"),
    }
    by_scores = {(kernel.pc1, kernel.pc2): kernel for kernel in KERNELS}
    for pc2 in ("low", "high"):
        low_pc1 = by_scores[("low", pc2)]
        high_pc1 = by_scores[("high", pc2)]
        assert high_pc1.length > low_pc1.length
        assert high_pc1.width > low_pc1.width
    for pc1 in ("low", "high"):
        low_pc2 = by_scores[(pc1, "low")]
        high_pc2 = by_scores[(pc1, "high")]
        assert high_pc2.width / high_pc2.length > low_pc2.width / low_pc2.length
        assert abs(high_pc2.asymmetry) < abs(low_pc2.asymmetry)
        assert high_pc2.groove_fraction < low_pc2.groove_fraction


def verify_svg(svg_text: str) -> None:
    root = ElementTree.fromstring(svg_text)
    assert root.tag.endswith("svg")
    assert root.get("role") == "img"
    assert root.get("aria-labelledby") == "toyPcaTitle toyPcaDesc"
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace) is not None
    assert root.find("svg:desc", namespace) is not None
    groups = root.findall(".//svg:g[@data-pc1]", namespace)
    assert len(groups) == 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    verify_concept()
    svg_text = render_svg()
    verify_svg(svg_text)

    if args.verify:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != svg_text:
            raise SystemExit(f"verification failed: {args.output}")
        print(f"verified toy PCA score guide in {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
