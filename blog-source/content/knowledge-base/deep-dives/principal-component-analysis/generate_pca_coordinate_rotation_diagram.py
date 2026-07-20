#!/usr/bin/env python3
"""Generate the conceptual PCA coordinate-rotation diagram deterministically."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree


# Conceptual paired measurements. They are standardized below before PCA.
# Both displayed panels are rendered from the resulting POINTS list.
RAW_POINTS = (
    (-1.65, -1.25),
    (-1.35, -1.05),
    (-1.05, -1.12),
    (-0.75, -0.45),
    (-0.45, 0.55),
    (-0.10, 0.08),
    (0.22, -0.35),
    (0.50, 0.55),
    (0.78, 0.20),
    (1.05, -0.02),
    (1.35, 0.55),
    (1.63, 1.20),
)

SCALE_PX = 54.0
LEFT_ORIGIN = (224.0, 238.0)
RIGHT_ORIGIN = (716.0, 238.0)
OUTPUT_DEFAULT = Path(__file__).with_name("pca-center-scale-rotate.svg")


def standardize(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    """Center and population-scale each of the two conceptual features."""
    means = tuple(sum(point[j] for point in points) / len(points) for j in range(2))
    sds = tuple(
        math.sqrt(sum((point[j] - means[j]) ** 2 for point in points) / len(points))
        for j in range(2)
    )
    return tuple(
        tuple((point[j] - means[j]) / sds[j] for j in range(2))
        for point in points
    )


POINTS = standardize(RAW_POINTS)


def principal_axes(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the first and second unit eigenvectors of the 2D covariance matrix."""
    n = len(points)
    cov_xx = sum(x * x for x, _ in points) / n
    cov_xy = sum(x * y for x, y in points) / n
    cov_yy = sum(y * y for _, y in points) / n

    angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    pc1 = (math.cos(angle), math.sin(angle))
    pc2 = (-pc1[1], pc1[0])

    # Give PC1 a stable direction for a byte-identical artifact.
    if pc1[0] < 0:
        pc1 = (-pc1[0], -pc1[1])
        pc2 = (-pc2[0], -pc2[1])
    return pc1, pc2


PC1, PC2 = principal_axes(POINTS)


def fmt(value: float) -> str:
    rounded = 0.0 if abs(value) < 0.0005 else value
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def screen_point(point: tuple[float, float]) -> tuple[float, float]:
    """Convert mathematical coordinates to local SVG coordinates."""
    x, y = point
    return (x * SCALE_PX, -y * SCALE_PX)


def point_marks(panel: str) -> str:
    circles = []
    for index, point in enumerate(POINTS, start=1):
        x, y = screen_point(point)
        circles.append(
            f'    <circle class="point" data-panel="{panel}" data-profile="{index:02d}" '
            f'data-local-x="{fmt(x)}" data-local-y="{fmt(y)}" '
            f'cx="{fmt(x)}" cy="{fmt(y)}" r="5"/>'
        )
    return "\n".join(circles)


def focus_profiles() -> tuple[tuple[int, tuple[float, float]], tuple[int, tuple[float, float]]]:
    scored = [
        (x * PC2[0] + y * PC2[1], index, (x, y))
        for index, (x, y) in enumerate(POINTS, start=1)
    ]
    low = min(scored)
    high = max(scored)
    return ((low[1], low[2]), (high[1], high[2]))


FOCUS = focus_profiles()


def focus_marks(panel: str, include_projections: bool) -> str:
    lines = []
    circles = []
    labels = []
    for label, (index, point) in zip(("A", "B"), FOCUS, strict=True):
        x, y = point
        sx, sy = screen_point(point)
        if include_projections:
            score_pc1 = x * PC1[0] + y * PC1[1]
            projection = (score_pc1 * PC1[0], score_pc1 * PC1[1])
            px, py = screen_point(projection)
            lines.append(
                f'    <line class="projection" x1="{fmt(px)}" y1="{fmt(py)}" '
                f'x2="{fmt(sx)}" y2="{fmt(sy)}"/>'
            )
        circles.append(
            f'    <circle class="focus" data-panel="{panel}" data-profile="{index:02d}" '
            f'cx="{fmt(sx)}" cy="{fmt(sy)}" r="7"/>'
        )
        label_x = sx + (11 if sx <= 0 else -11)
        anchor = "start" if sx <= 0 else "end"
        labels.append(
            f'    <text class="t profile-label" x="{fmt(label_x)}" y="{fmt(sy - 9)}" '
            f'text-anchor="{anchor}">{label}</text>'
        )
    return "\n".join(lines + circles + labels)


def axis_line(vector: tuple[float, float], length: float, css_class: str) -> str:
    vx, vy = screen_point(vector)
    norm = math.hypot(vx, vy)
    dx = vx / norm * length
    dy = vy / norm * length
    return (
        f'<line class="{css_class}" x1="{fmt(-dx)}" y1="{fmt(-dy)}" '
        f'x2="{fmt(dx)}" y2="{fmt(dy)}"/>'
    )


def render_svg() -> str:
    left_points = point_marks("feature")
    right_points = point_marks("pca")
    left_focus = focus_marks("feature", include_projections=False)
    right_focus = focus_marks("pca", include_projections=True)
    pc1_line = axis_line(PC1, 142.0, "pc1")
    pc2_line = axis_line(PC2, 142.0, "pc2")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="940" height="430" viewBox="0 0 940 430" role="img" aria-labelledby="pcaTransformTitle pcaTransformDesc">
  <title id="pcaTransformTitle">The same profiles shown with feature axes and principal-component axes</title>
  <desc id="pcaTransformDesc">Two equal panels contain the same twelve profile points at identical local coordinates. The left panel uses horizontal and vertical standardized-feature axes. The right panel keeps every point fixed and replaces those axes with a solid principal component 1 axis and a dashed perpendicular principal component 2 axis. Profiles A and B are highlighted in both panels. In the right panel, blue segments show their principal component 2 distances from the principal component 1 axis.</desc>
  <defs>
    <marker id="pcaTransformArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0 0 L10 5 L0 10 Z" fill="#A67C52"/>
    </marker>
    <style>
      .t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}
      .title{{font-size:18px;font-weight:700}}
      .h{{font-size:13px;font-weight:700;letter-spacing:.3px}}
      .s{{font-size:11px}}
      .m{{font-size:10px;fill:#5A544C}}
      .panel{{fill:#FBF9F6;stroke:#C4B8A8;stroke-width:1.3}}
      .axis{{stroke:#7F786D;stroke-width:1.4}}
      .pc1{{stroke:#A67C52;stroke-width:2.8}}
      .pc2{{stroke:#4B6787;stroke-width:2.4;stroke-dasharray:7 4}}
      .projection{{stroke:#4B6787;stroke-width:2}}
      .point{{fill:#4B6787;stroke:#F7F4F0;stroke-width:1}}
      .focus{{fill:#EAF1E5;stroke:#5A7346;stroke-width:2.3}}
      .profile-label{{font-size:11px;font-weight:700}}
      .origin{{fill:#2C2924}}
      .flow{{fill:none;stroke:#A67C52;stroke-width:1.8;marker-end:url(#pcaTransformArrow)}}
    </style>
  </defs>
  <rect width="940" height="430" fill="#F7F4F0"/>
  <text x="24" y="31" class="t title">The profiles stay; the coordinate system changes</text>
  <text x="24" y="52" class="t m">Both panels are drawn from one shared coordinate table. PCA computes the new axes from those same coordinates.</text>

  <rect x="24" y="78" width="400" height="288" rx="12" class="panel"/>
  <text x="44" y="103" class="t h">STANDARDIZED FEATURE AXES</text>
  <text x="44" y="121" class="t m">Each profile has a coordinate on feature 1 and feature 2</text>
  <g transform="translate({fmt(LEFT_ORIGIN[0])} {fmt(LEFT_ORIGIN[1])})">
    <line x1="-146" y1="0" x2="146" y2="0" class="axis"/>
    <line x1="0" y1="-104" x2="0" y2="104" class="axis"/>
    <circle cx="0" cy="0" r="3" class="origin"/>
{left_points}
{left_focus}
    <text x="132" y="20" text-anchor="end" class="t s">feature 1</text>
    <text x="-14" y="-90" text-anchor="end" class="t s">feature 2</text>
  </g>
  <text x="224" y="348" text-anchor="middle" class="t m">Profiles A and B are ringed so you can track them.</text>

  <path class="flow" d="M439 218 H493"/>
  <text x="466" y="192" text-anchor="middle" class="t s">same x,y</text>
  <text x="466" y="207" text-anchor="middle" class="t s">positions</text>

  <rect x="516" y="78" width="400" height="288" rx="12" class="panel"/>
  <text x="536" y="103" class="t h">PRINCIPAL-COMPONENT AXES</text>
  <text x="536" y="121" class="t m">Every profile stays fixed while the perpendicular axes change</text>
  <g transform="translate({fmt(RIGHT_ORIGIN[0])} {fmt(RIGHT_ORIGIN[1])})">
    {pc1_line}
    {pc2_line}
    <circle cx="0" cy="0" r="3" class="origin"/>
{right_points}
{right_focus}
    <text x="112" y="-87" class="t h">PC1 (solid)</text>
    <text x="-112" y="-87" text-anchor="end" class="t h">PC2 (dashed)</text>
  </g>
  <text x="716" y="348" text-anchor="middle" class="t m">Blue segments are the profiles' PC2 coordinates.</text>

  <text x="470" y="397" text-anchor="middle" class="t s">Coordinate check: every profile has the same local x and y in both panels.</text>
  <text x="470" y="416" text-anchor="middle" class="t m">PCA changes the directions used to describe the profiles. It does not rearrange the profiles.</text>
</svg>
'''


def verify_geometry(svg_text: str) -> None:
    root = ElementTree.fromstring(svg_text)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    coordinates: dict[str, dict[str, tuple[str, str]]] = {"feature": {}, "pca": {}}
    for circle in root.findall(".//svg:circle", namespace):
        panel = circle.get("data-panel")
        profile = circle.get("data-profile")
        if panel in coordinates and profile and circle.get("data-local-x") is not None:
            coordinates[panel][profile] = (
                circle.get("data-local-x", ""),
                circle.get("data-local-y", ""),
            )

    assert len(coordinates["feature"]) == len(POINTS)
    assert len(coordinates["pca"]) == len(POINTS)
    assert coordinates["feature"] == coordinates["pca"], (
        "The feature-axis and PCA-axis panels do not reuse identical profile coordinates."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    svg_text = render_svg()
    verify_geometry(svg_text)

    if args.verify:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != svg_text:
            raise SystemExit(f"verification failed: {args.output} is not the generated artifact")
        print(f"verified identical profile coordinates in {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
