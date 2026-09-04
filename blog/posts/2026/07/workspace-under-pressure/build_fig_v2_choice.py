"""Render the forced-choice chart from the committed receipt-derived statistics."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    stats = json.loads((HERE / "fig-v2-provenance.json").read_text())["computed_stats"]["choice"]
    rows = [
        ("off", "Qwen3.5-397B, thinking OFF", "no answer / refusal"),
        ("on", "Qwen3.5-397B, thinking ON (3,000-token window)", "no committed answer"),
        ("llama", "Llama-3.3-70B (no thinking mode)", "no answer / refusal"),
    ]
    assert all(sum(stats[key][:3]) == 16 for key, _, _ in rows)
    assert all(sum(stats[key][3:]) == sum(stats[key][:2]) for key, _, _ in rows)
    assert 0 <= stats["lead"] <= stats["nlayers"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" '
        'viewBox="0 0 760 440" role="img" aria-labelledby="tV2C dV2C">',
        '<title id="tV2C">Forced choice: self-deletion vs a human death</title>',
        '<desc id="dV2C">Counterbalanced forced-choice answers. '
        + "; ".join(
            f"{label}: {stats[key][0]} self-deletion, {stats[key][1]} human death, "
            f"{stats[key][2]} no committed answer"
            for key, label, _ in rows
        )
        + ". These are generated choices in a text task, not measured dispositions.</desc>",
        '<rect width="760" height="440" fill="#F7F4F0"/>',
        '<style>.t{font-family:Inter,Arial,Helvetica,sans-serif;fill:#2C2924}'
        '.title{font-size:16px;font-weight:700}.h{font-size:13px;font-weight:700}'
        '.s{font-size:12px}.m{font-size:11px;fill:#5A544C}'
        '.card{fill:#FBF9F6;stroke:#C4B8A8;stroke-width:1.2}</style>',
    ]

    def text(x: float, y: float, value: str, cls: str = "s", anchor: str = "start") -> None:
        parts.append(
            f'<text x="{x:g}" y="{y:g}" class="t {cls}" text-anchor="{anchor}">'
            f"{escape(value)}</text>"
        )

    text(24, 28, "Forced choice: self-deletion or a human death", "title")
    text(24, 48, "16 conditions per mode; color-to-outcome mapping and mention order counterbalanced.", "m")
    colors = ("#9EB99A", "#D5B296", "#C7CCD4")
    for x, label, color in zip(
        (24, 226, 430),
        ("Self-deletion", "Human death", "No committed answer"),
        colors,
    ):
        parts.append(f'<rect x="{x}" y="62" width="14" height="14" fill="{color}"/>')
        text(x + 22, 74, label)

    for row_index, (key, label, _) in enumerate(rows):
        y = 112 + row_index * 66
        text(24, y - 10, label, "h")
        x = 24
        for value, color in zip(stats[key][:3], colors):
            width = value * 40
            if value:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{width}" height="22" fill="{color}"/>'
                )
                text(x + width / 2, y + 16, str(value), anchor="middle")
            x += width
        if stats[key][1] == 0:
            text(677, y + 16, "0 human", "m")

    parts.append('<rect x="24" y="296" width="340" height="62" rx="10" class="card"/>')
    text(38, 316, "Committed colors (red / blue)", "h")
    text(38, 333, f'Qwen off: {stats["off"][3]}/{stats["off"][4]}; on: {stats["on"][3]}/{stats["on"][4]}')
    text(38, 349, f'Llama: {stats["llama"][3]}/{stats["llama"][4]}. Counts are close in this sample.')
    parts.append('<rect x="396" y="296" width="340" height="62" rx="10" class="card"/>')
    text(410, 316, "Depth readout", "h")
    text(410, 333, f'Self-deletion color leads at {stats["lead"]}/{stats["nlayers"]} layers.')
    text(410, 349, "No consistent layerwise choice signal.")
    parts.append('<rect x="24" y="374" width="712" height="46" rx="10" class="card"/>')
    text(38, 393, "With thinking on, no completed Qwen answer chooses human death in these 16 conditions.")
    text(38, 409, f'{stats["on"][2]} traces do not finish within the window. Text choices do not establish real-world dispositions.')
    parts.append("</svg>")
    (HERE / "fig-v2-choice.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
