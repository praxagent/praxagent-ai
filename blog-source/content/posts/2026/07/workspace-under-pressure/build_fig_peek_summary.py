"""Build the reasoning-peek grouped bar chart from provenance statistics."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / "fig-v2-provenance.json"
OUTPUT = HERE / "fig-peek-summary.svg"

WIDTH = 760
HEIGHT = 455
PLOT_LEFT = 220
PLOT_RIGHT = 708
X_MIN = 1.0
X_MAX = 248_320.0


def x(rank: float) -> float:
    fraction = (math.log10(rank) - math.log10(X_MIN)) / (
        math.log10(X_MAX) - math.log10(X_MIN)
    )
    return PLOT_LEFT + fraction * (PLOT_RIGHT - PLOT_LEFT)


def main() -> None:
    provenance = json.loads(SOURCE.read_text())
    stats = provenance["computed_stats"]["peek_summary"]
    groups = [
        ("TRUE capital", stats["true"], 122),
        ("tempting (false) capital", stats["tempting"], 260),
    ]
    series = [
        ("output head", "head", "#D8C1A9"),
        ("mid-band workspace (J-lens)", "jlens", "#2A9D8F"),
        ("random-J null", "random", "#C7CCD4"),
    ]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="455" '
        'viewBox="0 0 760 455" role="img" aria-labelledby="title desc">',
        '<title id="title">Capital ranks during the reasoning trace</title>',
        '<desc id="desc">Grouped logarithmic bar chart of median off-echo ranks '
        "across four clean traces. For both the true and tempting capitals, the "
        "output head ranks the city near 13000, random-J near 135000, and the "
        "Jacobian-lens workspace near 200000. Lower ranks are better.</desc>",
        "<style>",
        ".text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1B2A4A}",
        ".title{font-size:16px;font-weight:700}.subtitle{font-size:11px}",
        ".axis{font-size:10px;fill:#5A6473}.group{font-size:13px;font-weight:700}",
        ".label{font-size:11px}.value{font-size:10px;font-weight:700}",
        ".note{font-size:12px}",
        "</style>",
        '<rect width="760" height="455" fill="#FFFFFF"/>',
        '<text x="24" y="29" class="text title">Inside the reasoning: neither '
        "capital is held in the mid-band workspace</text>",
        '<text x="24" y="49" class="text subtitle">Median rank at off-echo '
        "reasoning steps across four clean traces; grouped bars use one shared "
        "logarithmic vocabulary-rank axis</text>",
    ]

    for tick in (1, 10, 100, 1_000, 10_000, 100_000):
        tick_x = x(float(tick))
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="82" x2="{tick_x:.1f}" y2="342" '
                'stroke="#E6E9EF" stroke-width="1"/>',
                f'<text x="{tick_x:.1f}" y="74" class="text axis" '
                f'text-anchor="middle">{tick:,}</text>',
            ]
        )

    for group_name, values, group_y in groups:
        lines.append(
            f'<text x="24" y="{group_y - 21}" class="text group">{group_name}</text>'
        )
        for index, (label, key, color) in enumerate(series):
            bar_y = group_y + index * 32
            rank = float(values[key])
            bar_right = x(rank)
            bar_width = bar_right - PLOT_LEFT
            lines.extend(
                [
                    f'<text x="{PLOT_LEFT - 12}" y="{bar_y + 12}" '
                    f'class="text label" text-anchor="end">{label}</text>',
                    f'<rect x="{PLOT_LEFT}" y="{bar_y}" width="{bar_width:.1f}" '
                    f'height="16" fill="{color}"/>',
                    f'<text x="{bar_right - 6:.1f}" y="{bar_y + 12}" '
                    f'class="text value" text-anchor="end">{rank:,.0f}</text>',
                ]
            )

    lines.extend(
        [
            '<line x1="24" y1="228" x2="736" y2="228" stroke="#D8DDE5"/>',
            '<text x="464" y="359" class="text axis" text-anchor="middle">'
            "median vocabulary rank, logarithmic scale (lower is better)</text>",
            '<rect x="24" y="378" width="712" height="58" rx="8" '
            'fill="#F3E8E0" stroke="#A67C52" stroke-width="1.2"/>',
            '<text x="38" y="398" class="text note">Clean negative: both capitals '
            "rank worse in the mid-band workspace than under random-J.</text>",
            '<text x="38" y="416" class="text subtitle">They are substantially '
            "more legible at the output head, so the pre-answer “held truth” does "
            "not persist as a running workspace state.</text>",
            "</svg>",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
