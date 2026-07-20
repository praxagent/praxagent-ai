#!/usr/bin/env python3
"""Generate the conceptual standard-deviation comparison diagram."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


NARROW_VALUES = (-0.90, -0.60, -0.35, -0.10, 0.10, 0.35, 0.60, 0.90)
WIDE_VALUES = tuple(2.0 * value for value in NARROW_VALUES)
SCALE_PX = 75.0
OUTPUT_DEFAULT = Path(__file__).with_name("pca-standard-deviation.svg")


def mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def population_sd(values: tuple[float, ...]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def fmt(value: float) -> str:
    rounded = 0.0 if abs(value) < 0.0005 else value
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def panel(panel_id: str, x: float, heading: str, subheading: str, values: tuple[float, ...]) -> str:
    center_x = x + 207.0
    axis_y = 214.0
    sd = population_sd(values)
    sd_width = sd * SCALE_PX
    dots = []
    for index, value in enumerate(values, start=1):
        dot_x = center_x + value * SCALE_PX
        dots.append(
            f'  <circle class="point" data-panel="{panel_id}" data-observation="{index}" '
            f'cx="{fmt(dot_x)}" cy="{fmt(axis_y)}" r="6"/>'
        )

    return f'''  <rect x="{fmt(x)}" y="78" width="414" height="250" rx="12" class="panel"/>
  <text x="{fmt(x + 20)}" y="104" class="t h">{heading}</text>
  <text x="{fmt(x + 20)}" y="124" class="t m">{subheading}</text>
  <rect data-panel="{panel_id}" data-role="one-sd-band" x="{fmt(center_x - sd_width)}" y="153" width="{fmt(2 * sd_width)}" height="98" rx="8" class="sd-band"/>
  <line x1="{fmt(x + 42)}" y1="{fmt(axis_y)}" x2="{fmt(x + 372)}" y2="{fmt(axis_y)}" class="axis"/>
  <line x1="{fmt(center_x)}" y1="145" x2="{fmt(center_x)}" y2="252" class="mean-line"/>
  <text x="{fmt(center_x)}" y="143" text-anchor="middle" class="t s">mean</text>
{chr(10).join(dots)}
  <line x1="{fmt(center_x)}" y1="266" x2="{fmt(center_x + sd_width)}" y2="266" class="sd-bracket"/>
  <line x1="{fmt(center_x)}" y1="259" x2="{fmt(center_x)}" y2="273" class="sd-bracket"/>
  <line x1="{fmt(center_x + sd_width)}" y1="259" x2="{fmt(center_x + sd_width)}" y2="273" class="sd-bracket"/>
  <text x="{fmt(center_x + sd_width / 2)}" y="289" text-anchor="middle" class="t s">one standard-deviation unit</text>
  <text x="{fmt(center_x)}" y="314" text-anchor="middle" class="t m">The shaded width spans one standard deviation on each side of the mean.</text>'''


def render_svg() -> str:
    narrow_sd = population_sd(NARROW_VALUES)
    wide_sd = population_sd(WIDE_VALUES)
    assert abs(mean(NARROW_VALUES)) < 1e-12
    assert abs(mean(WIDE_VALUES)) < 1e-12
    assert math.isclose(wide_sd, 2.0 * narrow_sd, rel_tol=0.0, abs_tol=1e-12)

    left = panel(
        "smaller",
        24.0,
        "SMALLER STANDARD DEVIATION",
        "Values lie relatively close to the mean",
        NARROW_VALUES,
    )
    right = panel(
        "larger",
        462.0,
        "LARGER STANDARD DEVIATION",
        "Values lie farther from the same mean",
        WIDE_VALUES,
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390" role="img" aria-labelledby="sdTitle sdDesc">
  <title id="sdTitle">Standard deviation compares how tightly values gather around their mean</title>
  <desc id="sdDesc">Two panels use the same horizontal scale and the same mean. In the left panel, eight values cluster near the mean and the shaded one-standard-deviation band is narrow. In the right panel, eight values spread farther from the mean and the shaded band is twice as wide. The comparison shows that standard deviation summarizes spread around a mean.</desc>
  <defs>
    <style>
      .t{{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}}
      .title{{font-size:18px;font-weight:700}}
      .h{{font-size:13px;font-weight:700;letter-spacing:.25px}}
      .s{{font-size:11px}}
      .m{{font-size:10px;fill:#5A544C}}
      .panel{{fill:#FBF9F6;stroke:#C4B8A8;stroke-width:1.3}}
      .sd-band{{fill:#EAF1E5;stroke:#8BA07A;stroke-width:1}}
      .axis{{stroke:#7F786D;stroke-width:1.5}}
      .mean-line{{stroke:#2C2924;stroke-width:1.8}}
      .sd-bracket{{stroke:#A67C52;stroke-width:2}}
      .point{{fill:#4B6787;stroke:#F7F4F0;stroke-width:1.2}}
    </style>
  </defs>
  <rect width="900" height="390" fill="#F7F4F0"/>
  <text x="24" y="31" class="t title">Standard deviation describes spread around the mean</text>
  <text x="24" y="52" class="t m">Both panels use the same horizontal scale and have the same mean.</text>
{left}
{right}
  <text x="450" y="354" text-anchor="middle" class="t s">Same mean, different spread: values farther from the mean produce a larger standard deviation.</text>
  <text x="450" y="374" text-anchor="middle" class="t m">Standard deviation describes the values. It does not explain why they differ or whether they are correct.</text>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    svg_text = render_svg()

    if args.verify:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != svg_text:
            raise SystemExit(f"verification failed: {args.output} is not the generated artifact")
        print(f"verified generated standard-deviation diagram at {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
