"""Build the divergence-controls figure from committed provenance statistics."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / "fig-v2-provenance.json"
OUTPUT = HERE / "fig-controls-divergence.svg"

WIDTH = 760
HEIGHT = 410
PLOT_LEFT = 190
PLOT_RIGHT = 708
X_MIN = 1.0
X_MAX = 20_000.0


def x(rank: float) -> float:
    fraction = (math.log10(rank) - math.log10(X_MIN)) / (
        math.log10(X_MAX) - math.log10(X_MIN)
    )
    return PLOT_LEFT + fraction * (PLOT_RIGHT - PLOT_LEFT)


def main() -> None:
    provenance = json.loads(SOURCE.read_text())
    rows = provenance["computed_stats"]["controls_divergence"]
    values = {
        "Jacobian lens": statistics.median(row["jlens"] for row in rows),
        "identity / logit lens": statistics.median(row["logit"] for row in rows),
        "output head": statistics.median(row["head"] for row in rows),
        "random-J null": statistics.median(row["rand"] for row in rows),
    }
    colors = {
        "Jacobian lens": "#2A9D8F",
        "identity / logit lens": "#3D6FB4",
        "output head": "#A67C52",
        "random-J null": "#9DA3AC",
    }
    row_y = {
        "Jacobian lens": 128,
        "identity / logit lens": 184,
        "output head": 240,
        "random-J null": 296,
    }

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="410" '
        'viewBox="0 0 760 410" role="img" aria-labelledby="title desc">',
        '<title id="title">True-capital rank across four readers</title>',
        '<desc id="desc">Median best-rank across nine thinking-off divergence '
        "items. The logit lens ranks the true capital first, the Jacobian lens "
        "second, the output head seventh, and random-J 2468th. Lower is better "
        "on a logarithmic axis.</desc>",
        "<style>",
        ".text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1B2A4A}",
        ".title{font-size:16px;font-weight:700}.subtitle{font-size:11px}",
        ".axis{font-size:10px;fill:#5A6473}.label{font-size:12px;font-weight:600}",
        ".value{font-size:11px;font-weight:700}.note{font-size:12px}",
        "</style>",
        '<rect width="760" height="410" fill="#FFFFFF"/>',
        '<text x="24" y="29" class="text title">The held truth is output-adjacent, '
        "not Jacobian-lens-specific</text>",
        '<text x="24" y="49" class="text subtitle">Median best-rank of the true '
        "capital across nine thinking-off items; identical single-token probe "
        "for every reader</text>",
    ]

    for tick in (1, 10, 100, 1_000, 10_000):
        tick_x = x(float(tick))
        label = f"{tick:,}"
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="86" x2="{tick_x:.1f}" y2="318" '
                'stroke="#E6E9EF" stroke-width="1"/>',
                f'<text x="{tick_x:.1f}" y="78" class="text axis" '
                f'text-anchor="middle">{label}</text>',
            ]
        )

    lines.append(
        '<text x="449" y="334" class="text axis" text-anchor="middle">'
        "median best-rank, logarithmic scale (lower is better)</text>"
    )

    for name, rank in values.items():
        y_pos = row_y[name]
        x_pos = x(float(rank))
        rank_label = f"{int(rank):,}"
        lines.extend(
            [
                f'<text x="{PLOT_LEFT - 14}" y="{y_pos + 4}" class="text label" '
                f'text-anchor="end">{name}</text>',
                f'<line x1="{PLOT_LEFT}" y1="{y_pos}" x2="{PLOT_RIGHT}" '
                f'y2="{y_pos}" stroke="#D8DDE5" stroke-width="2"/>',
                f'<circle cx="{x_pos:.1f}" cy="{y_pos}" r="8" '
                f'fill="{colors[name]}" stroke="#FFFFFF" stroke-width="2"/>',
                f'<text x="{x_pos + 13:.1f}" y="{y_pos + 4}" '
                f'class="text value">rank {rank_label}</text>',
            ]
        )

    lines.extend(
        [
            '<rect x="24" y="351" width="712" height="42" rx="8" '
            'fill="#EAF1E5" stroke="#6F8D5E" stroke-width="1.2"/>',
            '<text x="38" y="369" class="text note">No fitted-lens advantage here: '
            "the plain logit lens ranks the truth 1st and the Jacobian lens 2nd.</text>",
            '<text x="38" y="385" class="text subtitle">Random-J remains the only '
            "genuine null. The output head's single-token rank is 7; tokenization "
            "details are discussed below.</text>",
            "</svg>",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
