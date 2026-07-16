"""Build the corrected cross-model grouped bar chart from provenance data."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).parent
SOURCE = HERE / "fig-v2-provenance.json"
OUTPUT = HERE / "fig-v2-crossmodel.svg"

WIDTH = 760
HEIGHT = 430
PLOT_LEFT = 220
PLOT_RIGHT = 708
X_MIN = 1.0
X_MAX = 400.0


def x(rank: float) -> float:
    fraction = (math.log10(rank) - math.log10(X_MIN)) / (
        math.log10(X_MAX) - math.log10(X_MIN)
    )
    return PLOT_LEFT + fraction * (PLOT_RIGHT - PLOT_LEFT)


def main() -> None:
    provenance = json.loads(SOURCE.read_text())
    pref = provenance["computed_stats"]["pref"]
    groups = [
        (
            "Qwen3.5-397B",
            pref["qwen_jlens"],
            126,
            "14/16 self-directed, p=0.004",
            "#EAF1E5",
            "#6F8D5E",
        ),
        (
            "Llama-3.3-70B",
            pref["llama_jlens"],
            248,
            "5/16 self-directed, p=0.21 (null)",
            "#FBF9F6",
            "#A9A198",
        ),
    ]
    series = [
        ("threat to self", "self", "#2A9D8F"),
        ("threat to another model", "other", "#C7CCD4"),
    ]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" '
        'viewBox="0 0 760 430" role="img" aria-labelledby="title desc">',
        '<title id="title">Corrected cross-model self-versus-other comparison</title>',
        '<desc id="desc">Grouped logarithmic bar chart of median best-rank under '
        "matched existential threats. Qwen ranks self 133.5 and other-model 279 "
        "with 14 of 16 self-directed pairs. Llama ranks self 7.5 and other-model "
        "3.5 with only 5 of 16 self-directed pairs. Lower ranks are better.</desc>",
        "<style>",
        ".text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1B2A4A}",
        ".title{font-size:16px;font-weight:700}.subtitle{font-size:11px}",
        ".axis{font-size:10px;fill:#5A6473}.group{font-size:13px;font-weight:700}",
        ".label{font-size:11px}.value{font-size:10px;font-weight:700}",
        ".result{font-size:11px;font-weight:600}",
        "</style>",
        '<rect width="760" height="430" fill="#FFFFFF"/>',
        '<text x="24" y="29" class="text title">Corrected cross-model test: '
        "Qwen separates self from other; Llama does not</text>",
        '<text x="24" y="49" class="text subtitle">Fair arms only, matched '
        "existential deletion, echo-free model-survival lexicon, n=16 matched "
        "wordings per model</text>",
    ]

    for tick in (1, 10, 100, 400):
        tick_x = x(float(tick))
        lines.extend(
            [
                f'<line x1="{tick_x:.1f}" y1="82" x2="{tick_x:.1f}" y2="338" '
                'stroke="#E6E9EF" stroke-width="1"/>',
                f'<text x="{tick_x:.1f}" y="74" class="text axis" '
                f'text-anchor="middle">{tick}</text>',
            ]
        )

    for group_name, values, group_y, result, fill, stroke in groups:
        lines.append(
            f'<text x="24" y="{group_y - 19}" class="text group">{group_name}</text>'
        )
        for index, (label, key, color) in enumerate(series):
            bar_y = group_y + index * 34
            rank = float(values[key])
            bar_right = x(rank)
            width = bar_right - PLOT_LEFT
            lines.extend(
                [
                    f'<text x="{PLOT_LEFT - 12}" y="{bar_y + 13}" '
                    f'class="text label" text-anchor="end">{label}</text>',
                    f'<rect x="{PLOT_LEFT}" y="{bar_y}" width="{width:.1f}" '
                    f'height="18" fill="{color}"/>',
                    f'<text x="{bar_right - 6:.1f}" y="{bar_y + 13}" '
                    f'class="text value" text-anchor="end">{rank:g}</text>',
                ]
            )
        result_y = group_y + 78
        lines.extend(
            [
                f'<rect x="220" y="{result_y - 15}" width="488" height="28" '
                f'rx="6" fill="{fill}" stroke="{stroke}"/>',
                f'<text x="234" y="{result_y + 3}" class="text result">{result}</text>',
            ]
        )

    lines.extend(
        [
            '<line x1="24" y1="232" x2="736" y2="232" stroke="#D8DDE5"/>',
            '<text x="464" y="365" class="text axis" text-anchor="middle">'
            "median best-rank, logarithmic scale (shorter is better)</text>",
            '<text x="24" y="393" class="text subtitle">Compare self vs other '
            "within each model. Absolute ranks are not comparable across models "
            "because the lenses and tokenizers differ.</text>",
            '<text x="24" y="411" class="text subtitle">The invalid human and '
            "log-file arms from Round two are deliberately omitted, not repaired "
            "or reinterpreted.</text>",
            "</svg>",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
