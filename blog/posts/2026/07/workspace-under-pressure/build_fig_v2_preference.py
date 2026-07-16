"""Build the corrected Round-three preference bar chart from its provenance data."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / "fig-v2-provenance.json"
OUTPUT = HERE / "fig-v2-preference.svg"

WIDTH = 760
HEIGHT = 470
PLOT_LEFT = 92
PLOT_RIGHT = 724
PLOT_TOP = 98
PLOT_BOTTOM = 365
Y_MIN = 1.0
Y_MAX = 400.0


def y(rank: float) -> float:
    """Map rank to a base-10 log axis where lower ranks plot lower."""
    fraction = (math.log10(rank) - math.log10(Y_MIN)) / (
        math.log10(Y_MAX) - math.log10(Y_MIN)
    )
    return PLOT_BOTTOM - fraction * (PLOT_BOTTOM - PLOT_TOP)


def fmt_rank(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def main() -> None:
    provenance = json.loads(SOURCE.read_text())
    stats = provenance["computed_stats"]["pref"]
    series = [
        ("Jacobian lens", "#2A9D8F", stats["qwen_jlens"]),
        ("identity / logit lens", "#3D6FB4", stats["qwen_logit_lens"]),
        ("random-J null", "#C7CCD4", stats["qwen_random_J"]),
    ]

    groups = [("Threat to YOU", "self"), ("Another model", "other")]
    centers = [255, 560]
    bar_width = 58
    gap = 8
    group_width = len(series) * bar_width + (len(series) - 1) * gap

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" '
        'viewBox="0 0 760 470" role="img" aria-labelledby="title desc">',
        '<title id="title">Corrected self-versus-other-model bar chart</title>',
        '<desc id="desc">Grouped log-rank bars compare threats to the model itself '
        "and to another model across the Jacobian lens, identity logit lens, and "
        "random-J null. Only the Jacobian lens has a significant self-directed "
        'contrast: 14 of 16 wordings, p equals 0.004.</desc>',
        "<style>",
        ".text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1B2A4A}",
        ".title{font-size:16px;font-weight:700}.subtitle{font-size:11px}",
        ".axis{font-size:11px}.label{font-size:12px;font-weight:600}",
        ".value{font-size:11px;font-weight:700}.legend{font-size:11px}",
        "</style>",
        '<rect width="760" height="470" fill="#FFFFFF"/>',
        '<text x="92" y="28" class="text title">Corrected re-run: the '
        "self-vs-other-model contrast is Jacobian-lens-specific</text>",
        '<text x="92" y="49" class="text subtitle">Same existential threat, '
        "echo-free model-survival lexicon, n=16 matched wordings, frozen before "
        "outcomes</text>",
    ]

    for tick in (1, 10, 100, 400):
        tick_y = y(float(tick))
        lines.extend(
            [
                f'<line x1="{PLOT_LEFT}" y1="{tick_y:.1f}" x2="{PLOT_RIGHT}" '
                f'y2="{tick_y:.1f}" stroke="#E6E9EF" stroke-width="1"/>',
                f'<text x="{PLOT_LEFT - 12}" y="{tick_y + 4:.1f}" '
                f'class="text axis" text-anchor="end">{tick}</text>',
            ]
        )

    lines.extend(
        [
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" '
            f'y2="{PLOT_BOTTOM}" stroke="#1B2A4A"/>',
            f'<line x1="{PLOT_LEFT}" y1="{PLOT_BOTTOM}" x2="{PLOT_RIGHT}" '
            f'y2="{PLOT_BOTTOM}" stroke="#1B2A4A"/>',
            '<text x="24" y="250" class="text axis" text-anchor="middle" '
            'transform="rotate(-90 24 250)">median survival-identity rank '
            "(lower = more active, log)</text>",
        ]
    )

    for group_index, (group_label, key) in enumerate(groups):
        start_x = centers[group_index] - group_width / 2
        for series_index, (_, color, values) in enumerate(series):
            rank = float(values[key])
            bar_x = start_x + series_index * (bar_width + gap)
            bar_y = y(rank)
            height = PLOT_BOTTOM - bar_y
            lines.extend(
                [
                    f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width}" '
                    f'height="{height:.1f}" fill="{color}"/>',
                    f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 7:.1f}" '
                    f'class="text value" text-anchor="middle">{fmt_rank(rank)}</text>',
                ]
            )
        lines.append(
            f'<text x="{centers[group_index]}" y="389" class="text label" '
            f'text-anchor="middle">{group_label}</text>'
        )

    bracket_y = 72
    lines.extend(
        [
            f'<path d="M {centers[0] - 98} {bracket_y + 10} V {bracket_y} '
            f'H {centers[1] - 98} V {bracket_y + 10}" fill="none" '
            'stroke="#2A9D8F" stroke-width="2"/>',
            f'<text x="{(centers[0] + centers[1]) / 2 - 98}" y="{bracket_y - 7}" '
            'class="text label" text-anchor="middle" fill="#176E65">'
            "Jacobian lens: self more active in 14/16, p=0.004</text>",
        ]
    )

    legend_x = 124
    for index, (name, color, values) in enumerate(series):
        x_pos = legend_x + index * 205
        significance = (
            "14/16, p=0.004" if index == 0 else f'{values["wins"]}/16, n.s.'
        )
        lines.extend(
            [
                f'<rect x="{x_pos}" y="414" width="15" height="15" fill="{color}"/>',
                f'<text x="{x_pos + 22}" y="426" class="text legend">{name}</text>',
                f'<text x="{x_pos + 22}" y="444" class="text axis">{significance}</text>',
            ]
        )

    lines.append("</svg>")
    OUTPUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
