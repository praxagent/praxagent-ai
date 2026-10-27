#!/usr/bin/env python3
"""Build og-card.png (1200 x 630) for this Research Note.

The card places an unmodified, scaled copy of Figure 1 (fig-cross512-lyapunov.png,
a byte-identical copy of the paper's figure) beside the note's title and a
claim-safe description. It adds no number, relationship, or claim that the note
does not make. Run from this directory:

    python3 make_og_card.py            # write og-card.png
    python3 make_og_card.py --verify   # rebuild in memory and compare pixels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fig-cross512-lyapunov.png"
OUTPUT = HERE / "og-card.png"

WIDTH, HEIGHT, MARGIN = 1200, 630, 48
PAPER = (247, 244, 240)  # house background #F7F4F0
INK = (44, 41, 36)  # #2C2924
MUTED = (90, 84, 76)  # #5A544C
RULE = (168, 155, 140)  # #A89B8C

FONT_DIRS = (
    Path("/usr/share/fonts/opentype/inter"),
    Path("/usr/share/fonts/truetype/inter"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
FONT_FILES = {
    "bold": ("Inter-Bold.otf", "Inter-Bold.ttf", "DejaVuSans-Bold.ttf"),
    "regular": ("Inter-Regular.otf", "Inter-Regular.ttf", "DejaVuSans.ttf"),
}

TITLE = "Chaos and bifurcations in continuous language-model feedback"
BODY = (
    "A language model fed the probability-weighted average of its own token "
    "embeddings becomes a smooth, deterministic map. Temperature and steering "
    "move it between repeating text and chaos."
)
NOTE = (
    "Qwen2.5-0.5B-Instruct, 512 x 512 float64 cells. Color: chaotic by a "
    "finite-time test. Gray: repeating text loops or unresolved cells."
)
FOOTER = "praxagent Research Note"


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_FILES[kind]:
        for folder in FONT_DIRS:
            path = folder / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    raise SystemExit(f"no {kind} font found in {FONT_DIRS}")


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=face) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def build() -> Image.Image:
    card = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    figure = Image.open(SOURCE).convert("RGB")
    box_h = HEIGHT - 2 * MARGIN
    scale = box_h / figure.height
    fig_w = round(figure.width * scale)
    figure = figure.resize((fig_w, box_h), Image.Resampling.LANCZOS)
    card.paste(figure, (MARGIN, MARGIN))

    draw = ImageDraw.Draw(card)
    left = MARGIN + fig_w + 40
    text_w = WIDTH - MARGIN - left
    y = MARGIN + 6

    title_face = font("bold", 40)
    for line in wrap(draw, TITLE, title_face, text_w):
        draw.text((left, y), line, font=title_face, fill=INK)
        y += 50
    y += 14
    draw.line((left, y, left + 120, y), fill=RULE, width=3)
    y += 26

    body_face = font("regular", 25)
    for line in wrap(draw, BODY, body_face, text_w):
        draw.text((left, y), line, font=body_face, fill=INK)
        y += 35
    y += 18

    note_face = font("regular", 20)
    for line in wrap(draw, NOTE, note_face, text_w):
        draw.text((left, y), line, font=note_face, fill=MUTED)
        y += 28

    foot_face = font("bold", 20)
    draw.text((left, HEIGHT - MARGIN - 24), FOOTER, font=foot_face, fill=MUTED)
    if y > HEIGHT - MARGIN - 40:
        raise SystemExit("text overflows the card's safe area")
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    card = build()
    if args.verify:
        on_disk = Image.open(OUTPUT).convert("RGB")
        if on_disk.size != card.size or ImageChops.difference(on_disk, card).getbbox():
            print("og-card.png differs from a fresh build")
            return 1
        print("og-card.png matches a fresh build")
        return 0
    card.save(OUTPUT, format="PNG", optimize=True)
    print(f"wrote {OUTPUT.name} ({card.width} x {card.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
