#!/usr/bin/env python3
"""Draw the 1200 by 630 social card for the forward-hook glossary entry.

The values match the worked example in index.md and forward-hook.svg.
Run from this directory:

    uv run --no-project --with pillow python make_card.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "forward-hook-card.png"
WIDTH, HEIGHT = 1200, 630
PAPER, INK, MUTED = (247, 244, 240), (44, 41, 36), (90, 84, 76)
CARD, RULE = (251, 249, 246), (196, 184, 168)
BOX, BOX_RULE = (255, 255, 255), (140, 130, 117)
HOOK_FILL, HOOK_RULE = (235, 241, 246), (75, 103, 135)
FONT_DIRS = (
    Path("/usr/share/fonts/opentype/inter"),
    Path("/usr/share/fonts/truetype/inter"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
FONT_FILES = {
    "bold": ("Inter-Bold.otf", "Inter-Bold.ttf", "DejaVuSans-Bold.ttf"),
    "regular": ("Inter-Regular.otf", "Inter-Regular.ttf", "DejaVuSans.ttf"),
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_FILES[kind]:
        for folder in FONT_DIRS:
            path = folder / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    raise SystemExit(f"no {kind} font found")


def arrow(draw: ImageDraw.ImageDraw, x0: int, y: int, x1: int, color: tuple[int, int, int]) -> None:
    draw.line((x0, y, x1 - 10, y), fill=color, width=4)
    draw.polygon([(x1, y), (x1 - 14, y - 8), (x1 - 14, y + 8)], fill=color)


def card() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    title, head, body = font("bold", 38), font("bold", 24), font("regular", 21)

    draw.text((48, 34), "Forward hook: intercept or steer a layer's output", font=title, fill=INK)
    draw.text((48, 88), "Same layer input [1.0, 2.0] and fixed layer weights in both rows.", font=body, fill=MUTED)

    # Row 1: Standard pass
    draw.rounded_rectangle((48, 140, 1152, 345), radius=16, fill=CARD, outline=RULE, width=2)
    draw.text((74, 160), "Without hook", font=head, fill=INK)

    draw.rounded_rectangle((74, 206, 304, 316), radius=10, fill=BOX, outline=BOX_RULE, width=2)
    draw.text((100, 224), "Layer input", font=head, fill=INK)
    draw.text((100, 264), "[1.0, 2.0]", font=body, fill=MUTED)
    arrow(draw, 314, 261, 410, INK)

    draw.rounded_rectangle((420, 206, 720, 316), radius=10, fill=BOX, outline=BOX_RULE, width=2)
    draw.text((446, 224), "Layer output", font=head, fill=INK)
    draw.text((446, 264), "[2.0, -1.0]", font=body, fill=MUTED)
    arrow(draw, 730, 261, 826, INK)

    draw.rounded_rectangle((836, 206, 1124, 316), radius=10, fill=BOX, outline=BOX_RULE, width=2)
    draw.text((862, 224), "Next layer gets", font=head, fill=INK)
    draw.text((862, 264), "[2.0, -1.0]", font=body, fill=INK)

    # Row 2: Hooked pass
    draw.rounded_rectangle((48, 372, 1152, 582), radius=16, fill=CARD, outline=RULE, width=2)
    draw.text((74, 392), "With forward hook  (+ 0.5 * [4.0, 2.0])", font=head, fill=INK)

    draw.rounded_rectangle((74, 440, 274, 552), radius=10, fill=BOX, outline=BOX_RULE, width=2)
    draw.text((96, 458), "Layer input", font=head, fill=INK)
    draw.text((96, 498), "[1.0, 2.0]", font=body, fill=MUTED)
    arrow(draw, 282, 496, 336, INK)

    draw.rounded_rectangle((344, 440, 574, 552), radius=10, fill=BOX, outline=BOX_RULE, width=2)
    draw.text((366, 458), "Layer output", font=head, fill=INK)
    draw.text((366, 498), "[2.0, -1.0]", font=body, fill=MUTED)
    arrow(draw, 582, 496, 636, HOOK_RULE)

    draw.rounded_rectangle((644, 440, 868, 552), radius=10, fill=HOOK_FILL, outline=HOOK_RULE, width=2)
    draw.text((664, 458), "Forward hook", font=head, fill=INK)
    draw.text((664, 498), "+ [2.0, 1.0]", font=body, fill=HOOK_RULE)
    arrow(draw, 876, 496, 926, HOOK_RULE)

    draw.rounded_rectangle((934, 440, 1126, 552), radius=10, fill=HOOK_FILL, outline=HOOK_RULE, width=2)
    draw.text((954, 458), "Next layer gets", font=head, fill=INK)
    draw.text((954, 498), "[4.0, 0.0]", font=body, fill=INK)
    return image


def main() -> None:
    image = card()
    image.save(OUT, format="PNG", optimize=True)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.name} {image.size[0]}x{image.size[1]} sha256={digest}")


if __name__ == "__main__":
    main()
