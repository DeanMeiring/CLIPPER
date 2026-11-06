"""Anime arenas for the anime-cut comment fights (6 Nov).

Dean saw six mockups behind a real fight and said "I like all the anime
arenas": rooftop at dusk, dojo, cherry blossom night, bamboo forest,
volcano, manga panel. Every anime-cut 1v1 gets one (``pick``: never the one
the last fight had), drawn once per video (``background``) with the fight's
ring and floor on it -- the duel's own ring physics, nothing else changes --
and a little life on top each frame (``animate``: falling petals, rising
embers, drifting light, dust in the dojo's light). All original drawing:
no symbols or writing on the banners, nobody's village or shrine.

The fight's names and hook are white, so every backdrop is dark behind the
top of the screen (the dojo and manga panel got a deeper shade than the
mockups, where the names were hard to read).
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from clipper import ball_evolution as be

W, H = be.W, be.H
CX, CY, R = 540, 1050, int(470 * 1.08)        # the clean duel's ring (ball_duel D_CX, C_CY, D_R * C_K)

ARENAS = ("rooftop", "dojo", "sakura", "bamboo", "volcano", "manga")
LABELS = {"rooftop": "Rooftop at dusk", "dojo": "Dojo", "sakura": "Cherry blossom night",
          "bamboo": "Bamboo forest", "volcano": "Volcano", "manga": "Manga panel"}


def pick(last, seed) -> str:
    """An arena for this fight, never the last fight's."""
    rng = random.Random(f"anime-arena-{seed}")
    return rng.choice([a for a in ARENAS if a != last] or list(ARENAS))


def _grad(stops):
    a = np.linspace(0, 1, H)
    cols = np.zeros((H, 3))
    for (y0, c0), (y1, c1) in zip(stops, stops[1:]):
        m = (a >= y0) & (a <= y1)
        k = ((a[m] - y0) / max(1e-6, y1 - y0))[:, None]
        cols[m] = np.array(c0) * (1 - k) + np.array(c1) * k
    arr = np.repeat(cols[:, None, :], W, axis=1)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")


def _glow(im, shape_fn, col, blur=30, times=1):
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    shape_fn(ImageDraw.Draw(layer), col)
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    for _ in range(times):
        im.alpha_composite(g)
    im.alpha_composite(layer)


def _floor(im, fill, ring, ring_w=8, glow_col=None):
    floor = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(floor).ellipse((CX - R, CY - R, CX + R, CY + R), fill=fill)
    im.alpha_composite(floor)
    if glow_col:
        _glow(im, lambda d, c: d.ellipse((CX - R, CY - R, CX + R, CY + R), outline=c + (255,), width=18),
              glow_col, 16, 2)
    ImageDraw.Draw(im).ellipse((CX - R, CY - R, CX + R, CY + R), outline=ring + (255,), width=ring_w)
    return im


def _hud_shade(im, h=330, a=150):
    """Dark behind the names, health bars and hook at the top."""
    shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for y in range(h):
        sd.line([(0, y), (W, y)], fill=(0, 0, 0, int(a * (1 - y / h) ** 0.7)))
    im.alpha_composite(shade)


def _rooftop():
    im = _grad([(0, (40, 20, 80)), (0.35, (200, 70, 120)), (0.55, (255, 150, 90)), (0.72, (255, 200, 120)),
                (1, (60, 30, 70))])
    sx, sy, sr = 540, 1180, 330
    _glow(im, lambda d, c: d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=c + (255,)), (255, 230, 150), 60)
    d = ImageDraw.Draw(im)
    for k in range(7):                                  # the sun, cut by stripes
        y = sy + 60 + k * 38
        d.rectangle((sx - sr - 10, y, sx + sr + 10, y + 10 + k * 3), fill=(255, 160, 100, 255))
    rng = random.Random(4)
    x = -20
    while x < W:                                         # the skyline, lit windows
        w_, h_ = rng.randint(70, 170), rng.randint(260, 640)
        top = 1520 - h_
        d.rectangle((x, top, x + w_, H), fill=(36, 18, 52, 255))
        for wy in range(top + 20, 1700, 34):
            for wx in range(x + 12, x + w_ - 12, 26):
                if rng.random() < 0.28:
                    d.rectangle((wx, wy, wx + 12, wy + 16), fill=(255, 210, 120, 220))
        x += w_ + rng.randint(4, 20)
    for k in range(3):                                   # power lines and a pole
        d.line([(0, 380 + k * 40), (W, 260 + k * 60)], fill=(30, 15, 40, 255), width=4)
    d.rectangle((900, 160, 930, 1600), fill=(30, 15, 40, 255))
    d.rectangle((840, 260, 990, 280), fill=(30, 15, 40, 255))
    d.polygon([(0, 1560), (W, 1520), (W, H), (0, H)], fill=(26, 14, 34, 255))
    _hud_shade(im, 420, 170)
    return _floor(im, (20, 8, 30, 150), (255, 240, 220), 8, (255, 140, 120))


def _dojo():
    im = _grad([(0, (70, 40, 24)), (1, (30, 18, 12))])
    d = ImageDraw.Draw(im)
    for x0 in range(0, W, 270):                          # paper walls
        d.rectangle((x0 + 8, 120, x0 + 262, 1000), fill=(240, 226, 196, 255), outline=(60, 34, 20, 255), width=10)
        for gx in range(1, 3):
            d.line([(x0 + 8 + gx * 85, 120), (x0 + 8 + gx * 85, 1000)], fill=(80, 46, 26, 255), width=6)
        for gy in range(1, 5):
            d.line([(x0 + 8, 120 + gy * 176), (x0 + 262, 120 + gy * 176)], fill=(80, 46, 26, 255), width=6)
    _glow(im, lambda dd, c: dd.ellipse((200, 300, 880, 900), fill=c + (90,)), (255, 200, 120), 120)
    d = ImageDraw.Draw(im)
    # a banner with a brush circle (an ensō: no writing)
    d.rectangle((420, 150, 660, 640), fill=(190, 30, 36, 255), outline=(40, 20, 10, 255), width=8)
    d.arc((450, 260, 630, 440), 30, 330, fill=(15, 10, 10, 255), width=26)
    d.rectangle((400, 130, 680, 160), fill=(60, 34, 20, 255))
    d.rectangle((0, 1000, W, 1060), fill=(50, 28, 16, 255))
    d.polygon([(0, 1060), (W, 1060), (W, H), (0, H)], fill=(150, 96, 56, 255))
    for k in range(-8, 9):                               # floorboards in perspective
        d.line([(540 + k * 70, 1060), (540 + k * 260, H)], fill=(110, 66, 36, 255), width=5)
    for y in (1180, 1360, 1600):
        d.line([(0, y), (W, y)], fill=(120, 74, 40, 255), width=3)
    # the paper is bright: a deep shade behind the white names and hook
    _hud_shade(im, 520, 225)
    return _floor(im, (200, 160, 100, 120), (40, 22, 12), 14)


def _sakura():
    im = _grad([(0, (8, 10, 40)), (0.6, (30, 30, 90)), (1, (14, 10, 40))])
    d = ImageDraw.Draw(im)
    rng = random.Random(9)
    for _ in range(120):
        x, y = rng.uniform(0, W), rng.uniform(0, 900)
        d.ellipse((x, y, x + 3, y + 3), fill=(255, 255, 255, rng.randint(80, 200)))
    mx, my, mr = 760, 420, 170
    _glow(im, lambda dd, c: dd.ellipse((mx - mr, my - mr, mx + mr, my + mr), fill=c + (255,)), (255, 248, 225), 70, 2)
    d = ImageDraw.Draw(im)

    def branch(x, y, a, ln, w, depth):
        if depth == 0 or ln < 30:
            return
        x2, y2 = x + ln * math.cos(a), y + ln * math.sin(a)
        d.line([(x, y), (x2, y2)], fill=(30, 14, 30, 255), width=int(w))
        for _ in range(int(ln / 30)):
            t = rng.random()
            bx, by = x + (x2 - x) * t + rng.uniform(-30, 30), y + (y2 - y) * t + rng.uniform(-30, 30)
            rr = rng.uniform(14, 30)
            d.ellipse((bx - rr, by - rr, bx + rr, by + rr),
                      fill=(255, rng.randint(150, 190), rng.randint(200, 220), rng.randint(180, 240)))
        branch(x2, y2, a + rng.uniform(-0.6, -0.1), ln * 0.72, w * 0.7, depth - 1)
        branch(x2, y2, a + rng.uniform(0.1, 0.6), ln * 0.65, w * 0.7, depth - 1)
    branch(-40, 160, 0.25, 330, 34, 5)
    branch(W + 40, 1500, math.pi + 0.35, 300, 30, 4)
    d.polygon([(0, 1650), (W, 1600), (W, H), (0, H)], fill=(10, 8, 28, 255))
    _hud_shade(im, 420, 150)
    return _floor(im, (10, 8, 40, 140), (255, 210, 230), 8, (255, 120, 190))


def _bamboo():
    im = _grad([(0, (20, 60, 40)), (0.5, (60, 140, 90)), (1, (14, 40, 26))])
    d = ImageDraw.Draw(im)
    rng = random.Random(5)
    for layer, (col, wmin, wmax) in enumerate((((40, 100, 60), 30, 50), ((24, 70, 40), 45, 70),
                                               ((12, 40, 22), 60, 90))):
        x = rng.uniform(-40, 40)
        while x < W:
            wd = rng.uniform(wmin, wmax)
            d.rectangle((x, 0, x + wd, H), fill=col + (255,))
            y = rng.uniform(0, 200)
            while y < H:                                 # the stalks' joints
                d.rectangle((x - 3, y, x + wd + 3, y + 10), fill=tuple(max(0, v - 18) for v in col) + (255,))
                y += rng.uniform(220, 340)
            x += wd + rng.uniform(80, 260) * (1.5 - layer * 0.3)
    shafts = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shafts)
    for k in range(4):                                   # light through the leaves
        x = 150 + k * 260
        sd.polygon([(x, 0), (x + 90, 0), (x + 380, H), (x + 200, H)], fill=(255, 255, 200, 34))
    im.alpha_composite(shafts.filter(ImageFilter.GaussianBlur(20)))
    _hud_shade(im, 420, 170)
    return _floor(im, (10, 30, 16, 150), (230, 255, 220), 8, (140, 255, 160))


def _volcano():
    im = _grad([(0, (30, 6, 10)), (0.4, (120, 20, 20)), (0.7, (230, 80, 30)), (1, (40, 8, 6))])
    d = ImageDraw.Draw(im)
    d.polygon([(-100, 1500), (300, 700), (420, 760), (520, 640), (640, 780), (1180, 1500)], fill=(30, 10, 12, 255))
    _glow(im, lambda dd, c: dd.polygon([(500, 650), (540, 600), (580, 650)], fill=c + (255,)), (255, 140, 40), 40, 3)
    rng = random.Random(2)
    for _ in range(8):                                    # lava cracks
        x, y = rng.uniform(250, 800), rng.uniform(800, 1400)
        pts = [(x, y)]
        for _ in range(6):
            x += rng.uniform(-50, 50)
            y += rng.uniform(20, 70)
            pts.append((x, y))
        _glow(im, lambda dd, c, p=pts: dd.line(p, fill=c + (255,), width=6), (255, 120, 30), 10)
    d = ImageDraw.Draw(im)
    d.polygon([(0, 1560), (W, 1500), (W, H), (0, H)], fill=(24, 8, 8, 255))
    _hud_shade(im, 420, 150)
    return _floor(im, (20, 4, 4, 160), (255, 200, 140), 8, (255, 90, 30))


def _manga():
    im = Image.new("RGBA", (W, H), (250, 250, 246, 255))
    d = ImageDraw.Draw(im)
    for y in range(0, H, 22):                             # screentone, denser to the edges
        for x in range(0, W, 22):
            r = max(0.0, (math.hypot((x - CX) / W, (y - CY) / H) - 0.18) * 14)
            if r > 0.4:
                d.ellipse((x - r, y - r, x + r, y + r), fill=(30, 30, 36, 255))
    from clipper import ball_anime
    im.alpha_composite(ball_anime._lines_at(ball_anime._speed_lines(3, (20, 20, 26), 110, R + 40), CX, CY))
    d = ImageDraw.Draw(im)
    d.rectangle((24, 24, W - 24, H - 24), outline=(10, 10, 14, 255), width=14)
    # a black top panel under the white names, bars and hook
    d.polygon([(38, 38), (W - 38, 38), (W - 38, 450), (38, 490)], fill=(18, 18, 24, 255))
    d.line([(24, 498), (W - 24, 458)], fill=(10, 10, 14, 255), width=14)
    return _floor(im, (255, 255, 255, 255), (10, 10, 14), 14)


_DRAW = {"rooftop": _rooftop, "dojo": _dojo, "sakura": _sakura, "bamboo": _bamboo, "volcano": _volcano,
         "manga": _manga}


@lru_cache(maxsize=2)
def background(name: str) -> Image.Image:
    """The arena with the fight's floor and ring, RGB (the painter's ``bg``)."""
    return _DRAW[name]().convert("RGB")


# ---- a little life on top (deterministic per video seed + time) ----------------

_PARTS = {"sakura": 46, "volcano": 60, "bamboo": 30, "dojo": 34}


@lru_cache(maxsize=8)
def _particles(name: str, seed: int):
    rng = random.Random(f"{name}-{seed}")
    return [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.6, 1.4), rng.uniform(0, 6.28), rng.uniform(0.4, 1))
            for _ in range(_PARTS.get(name, 0))]


def animate(name: str, bg: Image.Image, t: float, seed: int = 0) -> Image.Image:
    """The background for the frame at fight time t."""
    parts = _particles(name, seed)
    if not parts:
        return bg
    im = bg.copy()
    d = ImageDraw.Draw(im, "RGBA")
    for x0, y0, sp, ph, a in parts:
        if name == "sakura":                      # petals drifting down and sideways, swaying
            y = (y0 + t * 110 * sp) % (H + 40) - 20
            x = (x0 + t * 45 * sp + math.sin(t * 2 * sp + ph) * 30) % (W + 40) - 20
            rx, ry = 11 * sp, 5 + 3 * abs(math.sin(t * 3 + ph))
            d.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(255, 182, 212, int(220 * a)))
        elif name == "volcano":                   # embers rising and flickering
            y = (y0 - t * 140 * sp) % (H + 40) - 20
            x = x0 + math.sin(t * 1.5 * sp + ph) * 26
            r = 2.5 + 2.5 * sp
            fl = 0.55 + 0.45 * math.sin(t * 9 * sp + ph)
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, int(150 + 80 * fl), 60, int(255 * a * fl)))
        else:                                     # bamboo / dojo: slow motes in the light
            y = (y0 - t * 18 * sp) % H
            x = x0 + math.sin(t * 0.6 * sp + ph) * 40
            r = 2 + 2.5 * sp
            col = (255, 255, 210) if name == "bamboo" else (255, 230, 180)
            d.ellipse((x - r, y - r, x + r, y + r), fill=col + (int(150 * a),))
    return im
