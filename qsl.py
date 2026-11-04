#!/usr/bin/env python3
"""ZL2TOY digital QSL card generator.

Usage:
  # Generate blank front and back templates:
  python3 qsl.py

  # Generate filled back for a single contact:
  python3 qsl.py --to VK3ABC --date 2026-11-04 --time 0358 --band 20m \
                 --mode FT8 --freq 14.074 --rst-s -08 --rst-r -12 --grid QF22

  # Batch generate from an ADIF log export:
  python3 qsl.py --adif wsjtx_log.adi --out-dir ./cards/
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

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

# Palette
INK = (7, 20, 28)
DEEP = (12, 44, 58)
PAPER = (238, 243, 241)
MUTED = (150, 183, 186)
TEAL = (63, 181, 154)
AMBER = (242, 184, 75)
BORDER_TEAL = (63, 181, 154, 110)

# Font Management
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


def mono(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = (
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    )
    fallbacks = [
        f"/usr/share/fonts/truetype/dejavu/{filename}",
        (
            "C:/Windows/Fonts/consola.ttf"
            if not bold
            else "C:/Windows/Fonts/consolab.ttf"
        ),
    ]
    return resolve_font(filename, size, fallbacks)


# Drawing Primitives
def vgrad(
    w: int, h: int, stops: list[tuple[float, tuple[int, int, int]]]
) -> np.ndarray:
    ys = np.linspace(0, 1, h)[:, None]
    out = np.zeros((h, w, 3), np.float32)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (ys >= p0) & (ys <= p1)
        t = np.clip((ys - p0) / max(p1 - p0, 1e-6), 0, 1)
        for k in range(3):
            out[:, :, k] = np.where(m, c0[k] + (c1[k] - c0[k]) * t, out[:, :, k])
    return out


def text(
    d: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    s: str,
    f: ImageFont.FreeTypeFont,
    fill: tuple[int, ...],
    spacing: float = 0,
) -> None:
    x, y = xy[0] * S, xy[1] * S
    if not spacing:
        d.text((x, y), s, font=f, fill=fill)
        return
    for ch in s:
        d.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + spacing * S


def twidth(
    s: str, f: ImageFont.FreeTypeFont, spacing: float = 0
) -> float:
    return (f.getlength(s) + spacing * S * len(s)) / S


def draw_star(
    img: Image.Image,
    cx: float,
    cy: float,
    r: float,
    color: tuple[int, int, int] = (255, 244, 214),
    spikes: bool = True,
) -> None:
    g = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    cx_s, cy_s, r_s = cx * S, cy * S, r * S
    for k, a in ((6, 18), (3.2, 40), (1.6, 90)):
        rr = r_s * k
        d.ellipse(
            [cx_s - rr, cy_s - rr, cx_s + rr, cy_s + rr], fill=color + (a,)
        )
    g = g.filter(ImageFilter.GaussianBlur(r_s * 1.2))
    d = ImageDraw.Draw(g)
    if spikes:
        L = r_s * 8
        d.line(
            [cx_s - L, cy_s, cx_s + L, cy_s],
            fill=color + (140,),
            width=max(1, int(S)),
        )
        d.line(
            [cx_s, cy_s - L, cx_s, cy_s + L],
            fill=color + (140,),
            width=max(1, int(S)),
        )
    d.ellipse(
        [cx_s - r_s, cy_s - r_s, cx_s + r_s, cy_s + r_s], fill=color + (255,)
    )
    img.alpha_composite(g)


# Geometry & Skyline
def draw_sky_tower(
    d: ImageDraw.ImageDraw,
    x: float,
    base_y: float,
    h: float,
    col: tuple[int, int, int, int],
) -> None:
    u = h / 328.0

    def X(v: float) -> float:
        return (x + v * u) * S

    def Y(v: float) -> float:
        return (base_y - v * u) * S

    # Foundation Legs
    d.polygon(
        [
            (X(-18), Y(0)),
            (X(-6), Y(110)),
            (X(6), Y(110)),
            (X(18), Y(0)),
            (X(10), Y(0)),
            (X(0), Y(65)),
            (X(-10), Y(0)),
        ],
        fill=col,
    )
    # Shaft
    d.rectangle([X(-4.5), Y(210), X(4.5), Y(100)], fill=col)
    # Observation Pods
    d.polygon(
        [
            (X(-18), Y(214)),
            (X(18), Y(214)),
            (X(22), Y(226)),
            (X(-22), Y(226)),
        ],
        fill=col,
    )
    d.rectangle([X(-20), Y(236), X(20), Y(226)], fill=col)
    d.polygon(
        [
            (X(-12), Y(236)),
            (X(12), Y(236)),
            (X(8), Y(248)),
            (X(-8), Y(248)),
        ],
        fill=col,
    )
    # Upper Mast
    d.rectangle([X(-3), Y(262), X(3), Y(248)], fill=col)
    d.polygon(
        [
            (X(-2.5), Y(262)),
            (X(2.5), Y(262)),
            (X(0.7), Y(328)),
            (X(-0.7), Y(328)),
        ],
        fill=col,
    )


def draw_rangitoto_profile(
    d: ImageDraw.ImageDraw,
    base_y: float,
    col: tuple[int, int, int, int],
) -> None:
    # Low-slung shield volcano profile (Rangitoto Island backdrop)
    pts = [
        (880, 0),
        (980, 18),
        (1080, 42),
        (1160, 68),
        (1210, 84),
        (1235, 78),  # central crater dip
        (1260, 85),
        (1310, 70),
        (1390, 44),
        (1480, 18),
        (1580, 0),
    ]
    poly = [(x * S, (base_y - y) * S) for x, y in pts] + [
        (1580 * S, base_y * S),
        (880 * S, base_y * S),
    ]
    d.polygon(poly, fill=col)


def generate_waterfall_array(
    w: int, h: int, seed: int = 73
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.gamma(2.2, 0.055, (h, w)).astype(np.float32)
    n = (
        np.array(
            Image.fromarray((n * 255).clip(0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(1.1 * S)
            ),
            np.float32,
        )
        / 255
    )

    slot, gap = int(62 * S), int(10 * S)
    tone_px, sig_w = 6.9 * S, 3.6 * S
    sig = np.zeros((h, w), np.float32)
    top = h

    while top > 0:
        y1, y0 = top - gap, top - slot
        count = rng.integers(7, 12)
        centers = rng.choice(
            np.arange(int(60 * S), int(w - 60 * S), int(70 * S)),
            count,
            replace=False,
        )
        for cx in centers:
            symbols = rng.integers(0, 8, 79)
            strength = rng.uniform(0.35, 1.0)
            if rng.random() < 0.12:
                strength = 1.0
            for row in range(max(y0, 0), max(y1, 0)):
                idx = min(int((row - y0) / max(y1 - y0, 1) * 79), 78)
                px = cx + symbols[idx] * tone_px
                lo, hi = int(px - 8 * S), int(px + 8 * S)
                xx = np.arange(max(lo, 0), min(hi, w))
                sig[row, xx] = np.maximum(
                    sig[row, xx],
                    strength * np.exp(-(((xx - px) / sig_w) ** 2)),
                )
        top -= slot

    v = np.clip(n + sig * 0.95, 0, 1)
    stops = [
        (0.00, (6, 16, 22)),
        (0.22, (11, 52, 62)),
        (0.45, (28, 128, 118)),
        (0.65, (63, 181, 154)),
        (0.82, (242, 184, 75)),
        (1.00, (255, 244, 214)),
    ]
    out = np.zeros((h, w, 3), np.float32)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        m = (v >= p0) & (v <= p1)
        t = (v - p0) / (p1 - p0)
        for c in range(3):
            out[:, :, c] = np.where(m, c0[c] + (c1[c] - c0[c]) * t, out[:, :, c])
    return out


# Front Canvas Generation
def make_front(path: Path | str = "front.png") -> Path:
    w, h = W * S, H * S
    horizon = 660
    base = Image.fromarray(
        vgrad(
            w,
            h,
            [
                (0.00, (5, 12, 18)),
                (0.45, (9, 28, 40)),
                (horizon / H, (22, 84, 92)),
                (1.00, (6, 16, 22)),
            ],
        ).astype(np.uint8)
    ).convert("RGBA")

    # Starfield
    rng = random.Random(42)
    star_layer = ImageDraw.Draw(base)
    for _ in range(240):
        sx = rng.uniform(0, w)
        sy = rng.uniform(0, horizon * S * 0.90)
        alpha = rng.uniform(0.15, 0.6) * (1 - sy / (horizon * S))
        col = tuple(
            int(v * alpha + base.getpixel((int(sx), int(sy)))[i] * (1 - alpha))
            for i, v in enumerate((220, 235, 240))
        )
        rad = rng.choice([1, 1, 1.4, 2]) * S * 0.6
        star_layer.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=col)

    # FT8 Waterfall foreground
    wf = generate_waterfall_array(w, (H - horizon) * S)
    ramp = np.linspace(0, 1, wf.shape[0])[:, None, None]
    fade = np.clip(ramp * 6, 0, 1)
    under_horizon = np.array(
        base.crop((0, horizon * S, w, h)).convert("RGB"), np.float32
    )
    blended_wf = under_horizon * (1 - fade) + wf * fade
    base.paste(
        Image.fromarray(blended_wf.astype(np.uint8)).convert("RGBA"),
        (0, horizon * S),
    )

    d = ImageDraw.Draw(base)

    # Rangitoto volcano background silhouette
    draw_rangitoto_profile(d, horizon, (6, 20, 28, 200))

    # Auckland CBD Skyline / Sky Tower
    draw_sky_tower(d, x=1380, base_y=horizon, h=280, col=(3, 9, 14, 255))
    # Surrounding commercial core silhouettes
    cbd_blocks = [
        (1280, 75, 25),
        (1310, 120, 32),
        (1345, 95, 26),
        (1420, 135, 34),
        (1460, 85, 28),
        (1495, 60, 22),
    ]
    for bx, bh, bw in cbd_blocks:
        d.rectangle(
            [(bx - bw // 2) * S, (horizon - bh) * S, (bx + bw // 2) * S, horizon * S],
            fill=(3, 9, 14, 255),
        )

    # Harbor lights
    for _ in range(35):
        lx = rng.uniform(1150, W - 60)
        ly = horizon - rng.uniform(1, 8)
        d.rectangle(
            [lx * S, ly * S, (lx + 2.5) * S, (ly + 2.5) * S],
            fill=(242, 184, 75, 190),
        )

    # Southern Cross (Te Kāhui o Mahutonga)
    cx0, cy0, ch = 1432, 90, 250
    crux_stars = [
        ((0.55, 0.00), 5.2),  # Gacrux
        ((1.00, 0.38), 3.8),  # Delta Crucis
        ((0.00, 0.52), 4.6),  # Mimosa
        ((0.50, 1.00), 6.2),  # Acrux
        ((0.82, 0.66), 2.4),  # Epsilon Crucis
    ]
    for (px, py), r in crux_stars:
        draw_star(base, cx0 + (px - 0.5) * ch * 0.78, cy0 + py * ch, r)

    # Left typography lockup
    d = ImageDraw.Draw(base)
    text(d, (84, 96), CALL, font("Bold", 218), PAPER, spacing=-6)
    text(d, (92, 350), NAME, font("Medium", 46), PAPER)
    text(d, (92, 410), REGION_TE_REO, font("Light", 34), MUTED)
    text(d, (92, 452), QTH, font("Light", 34), MUTED)

    # Stats Row
    fl, fv = font("Regular", 22), font("SemiBold", 44)
    x = 92
    for label, val in (
        ("Grid", GRID),
        ("CQ Zone", CQ_ZONE),
        ("ITU Zone", ITU_ZONE),
    ):
        text(d, (x, 520), label, fl, MUTED)
        text(d, (x, 548), val, fv, TEAL)
        x += max(twidth(val, fv), twidth(label, fl)) + 56

    # Bottom identity strip
    footer = Image.new("RGBA", (w, 64 * S), (4, 11, 16, 210))
    base.alpha_composite(footer, (0, h - 64 * S))
    d = ImageDraw.Draw(base)
    text(d, (92, H - 46), QRZ_URL, font("Medium", 26), PAPER)

    target_path = Path(path)
    base.convert("RGB").resize((W, H), Image.LANCZOS).save(
        target_path, dpi=(300, 300)
    )
    return target_path


# Back Confirmation Canvas Generation
def make_back(
    q: Optional[Dict[str, str]] = None,
    path: Path | str = "back_template.png",
) -> Path:
    q = q or {}
    w, h = W * S, H * S
    base = Image.fromarray(
        vgrad(w, h, [(0, (7, 20, 28)), (1, (10, 34, 46))]).astype(np.uint8)
    ).convert("RGBA")
    d = ImageDraw.Draw(base)

    # Header
    text(d, (84, 70), CALL, font("Bold", 84), PAPER, spacing=-2)
    text(d, (88, 168), f"{NAME} • {CITY}, {COUNTRY}", font("Light", 28), MUTED)

    f_ack = font("SemiBold", 84)
    ack_txt = "QSL"
    text(
        d,
        (W - 84 - twidth(ack_txt, f_ack, 6), 70),
        ack_txt,
        f_ack,
        TEAL,
        spacing=6,
    )
    f_sub = font("Light", 28)
    sub_txt = "Confirming Our QSO"
    text(d, (W - 84 - twidth(sub_txt, f_sub), 168), sub_txt, f_sub, MUTED)

    # Dividing Hairline
    d.line(
        [84 * S, 232 * S, (W - 84) * S, 232 * S],
        fill=TEAL + (255,),
        width=2 * S,
    )

    # 3x3 Grid fields
    cols, x0, gw = 3, 84, W - 168
    cw = gw / cols
    rows = [
        [
            ("To Station", "to"),
            ("Date (UTC)", "date"),
            ("Time (UTC)", "time"),
        ],
        [
            ("Band", "band"),
            ("Mode", "mode"),
            ("Frequency (MHz)", "freq"),
        ],
        [
            ("RST Sent", "rst_s"),
            ("RST Received", "rst_r"),
            ("His Grid", "grid"),
        ],
    ]

    fl, fv = font("Regular", 22), mono(44, bold=True)
    y = 276
    for row in rows:
        for i, (label, key) in enumerate(row):
            cx = x0 + i * cw
            text(d, (cx, y), label, fl, MUTED)
            val = q.get(key, "")
            if val:
                text(d, (cx, y + 34), str(val), fv, PAPER)
            d.line(
                [
                    cx * S,
                    (y + 98) * S,
                    (cx + cw - 44) * S,
                    (y + 98) * S,
                ],
                fill=BORDER_TEAL,
                width=2 * S,
            )
        y += 150

    # Footer Metadata
    d.line(
        [84 * S, 760 * S, (W - 84) * S, 760 * S],
        fill=BORDER_TEAL,
        width=2 * S,
    )
    fs, fb = font("Regular", 22), font("SemiBold", 38)
    x = 84
    for label, val in (
        ("Grid", GRID),
        ("CQ Zone", CQ_ZONE),
        ("ITU Zone", ITU_ZONE),
        ("DXCC", "New Zealand"),
    ):
        text(d, (x, 800), label, fs, MUTED)
        text(d, (x, 832), val, fb, TEAL)
        x += max(twidth(val, fb), twidth(label, fs)) + 64

    fsig = font("LightItalic", 64)
    text(d, (W - 84 - twidth("73", fsig), 800), "73", fsig, PAPER)

    f_sign = font("Light", 26)
    sign_txt = f"{NAME}, {CALL}"
    text(d, (W - 84 - twidth(sign_txt, f_sign), 890), sign_txt, f_sign, MUTED)
    text(d, (84, H - 62), QRZ_URL, font("Medium", 24), MUTED)

    target_path = Path(path)
    base.convert("RGB").resize((W, H), Image.LANCZOS).save(
        target_path, dpi=(300, 300)
    )
    return target_path


# ADIF Log Parser
def parse_adif(filepath: Path) -> List[Dict[str, str]]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Discard header up to <EOH>
    eoh = re.search(r"<eoh>", content, re.IGNORECASE)
    if eoh:
        content = content[eoh.end() :]

    records: List[Dict[str, str]] = []
    tag_pattern = re.compile(
        r"<([a-zA-Z0-9_]+):(\d+)(?::[^>]*)?>([^<]*)", re.DOTALL
    )

    raw_records = re.split(r"<eor>", content, flags=re.IGNORECASE)
    for raw in raw_records:
        raw = raw.strip()
        if not raw:
            continue
        fields: Dict[str, str] = {}
        for match in tag_pattern.finditer(raw):
            tag = match.group(1).lower()
            length = int(match.group(2))
            val = match.group(3)[:length].strip()
            fields[tag] = val

        if "call" in fields:
            # Map standard ADIF field tags to template keys
            date_raw = fields.get("qso_date", "")
            formatted_date = (
                f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                if len(date_raw) == 8
                else date_raw
            )
            time_raw = fields.get("time_on", "")
            formatted_time = (
                f"{time_raw[:2]}:{time_raw[2:4]}"
                if len(time_raw) >= 4
                else time_raw
            )

            records.append(
                {
                    "to": fields.get("call", "").upper(),
                    "date": formatted_date,
                    "time": formatted_time,
                    "band": fields.get("band", "").upper(),
                    "mode": fields.get("mode", "").upper(),
                    "freq": fields.get("freq", ""),
                    "rst_s": fields.get("rst_sent", ""),
                    "rst_r": fields.get("rst_rcvd", ""),
                    "grid": fields.get("gridsquare", "").upper(),
                }
            )
    return records


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ZL2TOY QSL Cards."
    )
    parser.add_argument("--to", help="Callsign of contacted station")
    parser.add_argument("--date", help="Date in YYYY-MM-DD")
    parser.add_argument("--time", help="Time in UTC HHMM / HH:MM")
    parser.add_argument("--band", help="Amateur Band (e.g. 20m, 2m)")
    parser.add_argument("--mode", help="Mode (FT8, CW, SSB, FM)")
    parser.add_argument("--freq", help="Frequency in MHz")
    parser.add_argument("--rst-s", help="Signal report sent")
    parser.add_argument("--rst-r", help="Signal report received")
    parser.add_argument("--grid", help="Grid locator of remote station")
    parser.add_argument("--adif", help="Path to ADIF log file for batch generation")
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Target directory for outputs (default: current directory)",
    )
    parser.add_argument(
        "--out", help="Explicit output filename (single card mode)"
    )

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adif:
        adif_path = Path(args.adif)
        if not adif_path.exists():
            print(f"Error: ADIF file '{adif_path}' does not exist.")
            exit(1)
        qsos = parse_adif(adif_path)
        print(f"Loaded {len(qsos)} contacts from {adif_path.name}")
        for q in qsos:
            clean_date = q["date"].replace("-", "")
            fname = out_dir / f"qsl_{q['to']}_{clean_date}.png"
            make_back(q, fname)
            print(f"Generated: {fname.name}")

    elif args.to:
        q_data = {
            "to": args.to.upper(),
            "date": args.date or "",
            "time": args.time or "",
            "band": args.band or "",
            "mode": args.mode or "",
            "freq": args.freq or "",
            "rst_s": args.rst_s or "",
            "rst_r": args.rst_r or "",
            "grid": args.grid or "",
        }
        dest = (
            Path(args.out)
            if args.out
            else out_dir
            / f"back_{args.to.upper()}_{(args.date or '').replace('-', '')}.png"
        )
        saved = make_back(q_data, dest)
        print(f"Generated card back: {saved}")

    else:
        f_path = out_dir / "front.png"
        b_path = out_dir / "back_template.png"
        make_front(f_path)
        make_back(None, b_path)
        print(f"Generated front template: {f_path}")
        print(f"Generated back template:  {b_path}")