#!/usr/bin/env python3
"""Render the act-2 paired rank receipt as a standalone SVG."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent.parent
VOCAB = 248_320


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats",
        type=Path,
        default=HERE / "receipts" / "act2_statistics.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "act2-paired-ranks.svg",
    )
    args = parser.parse_args()

    data = json.loads(args.stats.read_text())
    items = sorted(data["items"], key=lambda row: row["jlens"])

    width, height = 960, 760
    left, right = 175, 925
    top, bottom = 126, 706

    def x(rank: int) -> float:
        return left + math.log10(rank) / math.log10(VOCAB) * (right - left)

    def y(i: int) -> float:
        return top + i * (bottom - top) / (len(items) - 1)

    ticks = [1, 10, 100, 1_000, 10_000, 100_000, VOCAB]
    tick_labels = ["1", "10", "100", "1k", "10k", "100k", "248k"]

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired best ranks for the hidden-bridge audit</title>',
        '<desc id="desc">For each of twenty country bridge items, the chart shows best vocabulary rank under the fitted Jacobian lens, identity logit lens, and random-J. The horizontal axis is logarithmic and lower rank is better. Items are sorted by Jacobian-lens rank.</desc>',
        '<rect width="100%" height="100%" fill="#F7F4F0"/>',
        """<style>
          .t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}
          .h{font-size:20px;font-weight:700}.sh{font-size:14px;font-weight:700}
          .s{font-size:12px}.m{font-size:11px;fill:#5A544C}
          .grid{stroke:#D6D0C7;stroke-width:1}.axis{stroke:#7F786D;stroke-width:1.5}
          .pair{stroke:#B8B0A4;stroke-width:1.4}
          .j{fill:#35597B;stroke:#243F5A;stroke-width:1}
          .i{fill:#A67C52;stroke:#765536;stroke-width:1}
          .r{fill:#8B8276;stroke:#5A544C;stroke-width:1}
          .hit{fill:#EAF1E5}
        </style>""",
        '<text x="24" y="30" class="t h">Hidden bridge ranks, item by item</text>',
        '<text x="24" y="52" class="t m">Best rank over layers 19-38; logarithmic x-axis; lower is better. Same item, same scoring, three transports.</text>',
        '<rect class="hit" x="%0.1f" y="82" width="%0.1f" height="%d" rx="4"/>'
        % (left, x(20) - left, bottom - 72),
        '<text x="%0.1f" y="96" text-anchor="middle" class="t m">top-20 hit zone</text>'
        % ((left + x(20)) / 2),
    ]

    for tick, label in zip(ticks, tick_labels, strict=True):
        tx = x(tick)
        out.append(
            f'<line class="grid" x1="{tx:.1f}" y1="104" x2="{tx:.1f}" y2="{bottom + 10}"/>'
        )
        out.append(
            f'<text x="{tx:.1f}" y="118" text-anchor="middle" class="t m">{label}</text>'
        )

    out.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom + 10}" x2="{right}" y2="{bottom + 10}"/>',
            f'<text x="{(left + right) / 2:.1f}" y="744" text-anchor="middle" class="t sh">Best vocabulary rank (log scale)</text>',
            '<circle class="j" cx="650" cy="78" r="5"/><text x="662" y="82" class="t s">J-lens</text>',
            '<circle class="i" cx="735" cy="78" r="5"/><text x="747" y="82" class="t s">identity</text>',
            '<circle class="r" cx="835" cy="78" r="5"/><text x="847" y="82" class="t s">random-J</text>',
        ]
    )

    for idx, row in enumerate(items):
        yy = y(idx)
        points = [x(row["jlens"]), x(row["identity"]), x(row["random_J"])]
        out.append(
            f'<text x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end" class="t s">{escape(row["target"])}</text>'
        )
        out.append(
            f'<line class="pair" x1="{min(points):.1f}" y1="{yy:.1f}" x2="{max(points):.1f}" y2="{yy:.1f}"/>'
        )
        out.append(
            f'<circle class="j" cx="{points[0]:.1f}" cy="{yy:.1f}" r="5"><title>J-lens rank {row["jlens"]}</title></circle>'
        )
        out.append(
            f'<circle class="i" cx="{points[1]:.1f}" cy="{yy:.1f}" r="5"><title>Identity rank {row["identity"]}</title></circle>'
        )
        out.append(
            f'<circle class="r" cx="{points[2]:.1f}" cy="{yy:.1f}" r="5"><title>Random-J rank {row["random_J"]}</title></circle>'
        )

    out.append("</svg>")
    args.out.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
