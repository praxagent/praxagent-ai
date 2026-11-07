#!/usr/bin/env python3
"""Draw the 1200 by 630 social card for the next-token entropy glossary entry.

The bars use the same toy probabilities as index.md and next-token-entropy.svg.
Run from this directory:

    uv run --no-project --with pillow python make_card.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "next-token-entropy-card.png"
WIDTH, HEIGHT = 1200, 630
PAPER, INK, MUTED = (247, 244, 240), (44, 41, 36), (90, 84, 76)
CARD, RULE = (251, 249, 246), (196, 184, 168)
PEAK, SMALL, EVEN = (166, 124, 82), (75, 103, 135), (111, 141, 94)
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


def bar(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((x, y, x + max(width, 4), y + 22), radius=4, fill=fill)


def card() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)
    title, head, body = font("bold", 40), font("bold", 25), font("regular", 21)
    draw.text((48, 36), "Next-token entropy is how spread out one prediction is", font=title, fill=INK)
    draw.text((48, 92), "Same three tokens. Bar length is probability; ln is the natural logarithm.", font=body, fill=MUTED)

    draw.rounded_rectangle((48, 150, 576, 520), radius=16, fill=CARD, outline=RULE, width=2)
    draw.rounded_rectangle((624, 150, 1152, 520), radius=16, fill=CARD, outline=RULE, width=2)
    draw.text((72, 174), "Peaked", font=head, fill=INK)
    draw.text((648, 174), "Even", font=head, fill=INK)

    left = (("A  0.90", 324, PEAK), ("B  0.05", 18, SMALL), ("C  0.05", 18, SMALL))
    right = (("A  1/3", 120, EVEN), ("B  1/3", 120, EVEN), ("C  1/3", 120, EVEN))
    for i, (label, width, fill) in enumerate(left):
        y = 240 + i * 52
        draw.text((72, y), label, font=body, fill=INK)
        bar(draw, 190, y + 2, width, fill)
    for i, (label, width, fill) in enumerate(right):
        y = 240 + i * 52
        draw.text((648, y), label, font=body, fill=INK)
        bar(draw, 766, y + 2, width, fill)

    draw.text((72, 420), "Entropy H ≈ 0.394 nats", font=head, fill=INK)
    draw.text((72, 460), "Effective number of equal choices ≈ 1.48", font=body, fill=MUTED)
    draw.text((648, 420), "Entropy H = ln 3 ≈ 1.099 nats", font=head, fill=INK)
    draw.text((648, 460), "Effective number of equal choices = 3", font=body, fill=MUTED)
    return image


def main() -> None:
    image = card()
    image.save(OUT, format="PNG", optimize=True)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT.name} {image.size[0]}x{image.size[1]} sha256={digest}")


if __name__ == "__main__":
    main()
