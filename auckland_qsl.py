#!/usr/bin/env python3
"""ZL2TOY photographic QSL card generator - Auckland Skyline Edition.

Positions the station identity and coordinates cleanly in the lower half of the card,
keeping the full upper Auckland skyline and Sky Tower completely unobstructed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Canvas dimensions: 5.5 x 3.5 in at 300 DPI
W, H = 1650, 1050
S = 2  # Supersample multiplier for crisp antialiasing

# Station Metadata
CALL = "ZL2TOY"
NAME = "Dusty"
CITY = "Auckland"
COUNTRY = "New Zealand"
QTH = f"{CITY}, {COUNTRY}"
REGION_TE_REO = "Tāmaki Makaurau"
GRID = "RF73"
ITU_ZONE = "60"
CQ_ZONE = "32"
QRZ_URL = f"qrz.com/db/{CALL}"
PHOTO_CREDIT = "Photo by Sam Ferrara on Unsplash"

# Palette
PAPER = (255, 255, 255)
SUBTEXT = (210, 230, 238)
TEAL_ACCENT = (80, 225, 190)
SHADOW_COL = (0, 0, 0, 230)
CREDIT_COL = (150, 175, 182, 190)

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR / "fonts"


def resolve_font(
    filename: str, size: int, fallback_candidates: List[str]
) -> ImageFont.FreeTypeFont:
    target = FONT_DIR / filename
    if target.exists():
        return ImageFont.truetype(str(target), int(size * S))
    for candidate in fallback_candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, int(size * S))
    return ImageFont.load_default()


def font(style: str, size: int) -> ImageFont.FreeTypeFont:
    filename = f"Inter-{style}.otf"
    fallbacks = [
        f"/usr/share/fonts/opentype/inter/Inter-{style}.otf",
        f"/usr/share/fonts/truetype/inter/Inter-{style}.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    return resolve_font(filename, size, fallbacks)


def text_with_shadow(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    s: str,
    f: ImageFont.FreeTypeFont,
    fill: tuple[int, ...],
    spacing: float = 0,
    shadow_offset: tuple[float, float] = (2.0, 2.0),
    shadow_col: tuple[int, int, int, int] = SHADOW_COL,
) -> None:
    x, y = xy[0] * S, xy[1] * S
    so_x, so_y = shadow_offset[0] * S, shadow_offset[1] * S

    if not spacing:
        d.text((x + so_x, y + so_y), s, font=f, fill=shadow_col)
        d.text((x, y), s, font=f, fill=fill)
        return

    cur_x = x
    for ch in s:
        d.text((cur_x + so_x, y + so_y), ch, font=f, fill=shadow_col)
        d.text((cur_x, y), ch, font=f, fill=fill)
        cur_x += f.getlength(ch) + spacing * S


def twidth(s: str, f: ImageFont.FreeTypeFont, spacing: float = 0) -> float:
    return (f.getlength(s) + spacing * S * len(s)) / S


def prepare_background(photo_path: Path, target_w: int, target_h: int) -> Image.Image:
    """Crops photo centered to standard 3:2 postcard proportions."""
    if not photo_path.exists():
        raise FileNotFoundError(f"Skyline image '{photo_path}' not found.")

    with Image.open(photo_path) as src:
        img = src.convert("RGBA")

    tgt_ratio = target_w / target_h
    src_ratio = img.width / img.height

    if src_ratio > tgt_ratio:
        new_w = int(img.height * tgt_ratio)
        offset_x = (img.width - new_w) // 2
        img = img.crop((offset_x, 0, offset_x + new_w, img.height))
    else:
        new_h = int(img.width / tgt_ratio)
        offset_y = (img.height - new_h) // 2
        img = img.crop((0, offset_y, img.width, offset_y + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def make_front(
    photo_path: Path | str = "auckland_skyline.jpg",
    out_path: Path | str = "front.png",
) -> Path:
    w, h = W * S, H * S

    base = prepare_background(Path(photo_path), w, h)

    # 1. Dark Frosted Scrim Bar in the Lower Half (y: 530 -> 965)
    scrim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d_scrim = ImageDraw.Draw(scrim)
    scrim_box = [60 * S, 530 * S, (W - 60) * S, 965 * S]

    d_scrim.rounded_rectangle(
        scrim_box,
        radius=16 * S,
        fill=(6, 15, 22, 230),
        outline=(80, 225, 190, 90),
        width=int(1.8 * S),
    )
    scrim = scrim.filter(ImageFilter.GaussianBlur(1.2 * S))
    base.alpha_composite(scrim)

    d = ImageDraw.Draw(base)

    # 2. Main Callsign (Left-aligned, lower half)
    f_call = font("Bold", 175)
    text_with_shadow(
        d,
        (96, 560),
        CALL,
        f_call,
        PAPER,
        spacing=-4,
        shadow_offset=(3.0, 3.0),
    )

    # Vertical divider separating Callsign from Details
    divider_x = 940
    d.line(
        [divider_x * S, 560 * S, divider_x * S, 935 * S],
        fill=(80, 225, 190, 110),
        width=int(1.8 * S),
    )

    # 3. Name & Location (Right side of divider)
    text_with_shadow(
        d,
        (divider_x + 50, 565),
        NAME,
        font("SemiBold", 44),
        PAPER,
        shadow_offset=(2.0, 2.0),
    )
    text_with_shadow(
        d,
        (divider_x + 50, 622),
        REGION_TE_REO,
        font("Medium", 30),
        SUBTEXT,
        shadow_offset=(1.5, 1.5),
    )
    text_with_shadow(
        d,
        (divider_x + 50, 662),
        QTH,
        font("Medium", 30),
        SUBTEXT,
        shadow_offset=(1.5, 1.5),
    )

    # Separator rule
    d.line(
        [(divider_x + 50) * S, 715 * S, (W - 96) * S, 715 * S],
        fill=(80, 225, 190, 80),
        width=int(1.2 * S),
    )

    # 4. Station Metadata Row (Grid, CQ Zone, ITU Zone)
    fl, fv = font("Regular", 20), font("Bold", 42)
    meta_x = divider_x + 50
    for label, val in (
        ("Grid", GRID),
        ("CQ Zone", CQ_ZONE),
        ("ITU Zone", ITU_ZONE),
    ):
        text_with_shadow(d, (meta_x, 735), label, fl, SUBTEXT, shadow_offset=(1.5, 1.5))
        text_with_shadow(d, (meta_x, 765), val, fv, TEAL_ACCENT, shadow_offset=(2.0, 2.0))
        meta_x += max(twidth(val, fv), twidth(label, fl)) + 48

    # 5. Bottom Identity & Attribution Strip
    footer = Image.new("RGBA", (w, 48 * S), (4, 10, 16, 235))
    base.alpha_composite(footer, (0, h - 48 * S))
    d = ImageDraw.Draw(base)

    text_with_shadow(d, (84, H - 34), QRZ_URL, font("Medium", 22), PAPER, shadow_offset=(1.0, 1.0))

    # Halved credit micro-print (10pt font)
    f_cred = font("Light", 10)
    cred_x = W - 84 - twidth(PHOTO_CREDIT, f_cred)
    text(d, (cred_x, H - 28), PHOTO_CREDIT, f_cred, CREDIT_COL)

    out = Path(out_path)
    base.convert("RGB").resize((W, H), Image.LANCZOS).save(out, dpi=(300, 300))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Lower-Half Photo QSL Card.")
    parser.add_argument("--photo", default="auckland_skyline.jpg", help="Path to skyline image")
    parser.add_argument("--out", default="front.png", help="Output file path")
    args = parser.parse_args()

    saved = make_front(args.photo, args.out)
    print(f"Generated front card: {saved}")