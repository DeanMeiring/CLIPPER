"""Anime cut for the comment fights (6 Nov).

Dean wanted the 1v1 comment fights more cinematic: "anime style, and if the
final kill shot comes in, it zooms in, a small cutscene". A sandbox sample
(he: "I love the slow-mo and the final kill cam") became this module.

Two parts:

* ``face`` -- each commenter's own anime fighter (Dean picked these over
  emoji characters): an original face drawn from their channel id, so the
  same person always gets the same fighter -- hair, eyes, mouth, headband
  and marks vary (thousands of combinations). On a ball the picture code is
  "fighter:<channel id>" (``ball_duel._pic``).
* ``render`` -- directs a comment-fight duel (``recipe["cut"] == "anime"``):
  a VS intro, hit-stop on big hits, a red "LAST HEART" pulse, a slowed
  kill-cam with a fighting-game cut-in of the winner, impact frames
  (inverted black and white) on the K.O., a held freeze with a K.O. stamp,
  then the shatter in slow motion. The fight runs once first, so the cut
  knows when the final blow lands. The fight itself (sim, painter, result)
  is the normal comment fight. All sounds are made here (numpy taiko,
  booms, risers): nothing from any anime.

Techniques from research (animation blogs, Sakurai's hit-stop column,
fighting-game cut-ins): impact frames, hit-stop, speed lines, the cut-in,
a held still on the finishing blow -- and restraint: the effects sit on the
climax, the fight in between stays clean.
"""

from __future__ import annotations

import bisect
import math
import random
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from clipper import ball_evolution as be

W, H, FPS, SR = be.W, be.H, be.FPS, be.SR

INTRO_S = 0.9
SLOW_LEAD = 0.75          # the kill-cam starts this long (fight time) before the final blow
SLOW_EVERY = 3            # each fight frame held 3 frames in the kill-cam ("on threes")
ZOOM = 1.85
KO_HOLD = 0.55            # the freeze on the K.O.
AFTER_SLOW = 0.7          # fight seconds of the shatter at half speed
HITSTOP = 4               # extra frames held on a hit of 2+ hearts
EXTRA_S = INTRO_S + 0.05 + SLOW_LEAD * (SLOW_EVERY - 1) + KO_HOLD + 0.1 + AFTER_SLOW   # ~3.6 s added

# ---- the fighters ---------------------------------------------------------------

S = 512
SKIN = [(255, 224, 189), (241, 194, 150), (224, 172, 125), (198, 134, 90), (141, 85, 54), (255, 236, 214)]
HAIR = [(24, 22, 30), (250, 214, 90), (226, 70, 50), (60, 110, 240), (240, 240, 250), (240, 120, 190),
        (70, 200, 120), (120, 70, 40), (160, 90, 230)]
BAND = [(220, 40, 50), (30, 30, 40), (40, 90, 220), (245, 245, 250), (250, 190, 30)]
IRIS = [(60, 140, 255), (230, 60, 60), (60, 190, 110), (150, 80, 230), (250, 180, 40), (40, 40, 50)]
INK = (20, 18, 28, 255)


def face(key: str, size: int) -> Image.Image:
    """The anime fighter for ``key`` (a channel id), ``size`` px, transparent."""
    return _face(key).resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=64)
def _face(key: str) -> Image.Image:
    g = random.Random(f"fighter-{key}")
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r = S / 2, S * 0.345
    cy = c + S * 0.06                       # the head sits low: room for the hair
    skin = g.choice(SKIN)
    hair_c = g.choice(HAIR)
    hair = g.choice(["spiky", "spiky", "fringe", "long", "bun", "mohawk", "none"])
    eyes = g.choice(["big", "sharp", "sharp", "angry", "smug", "visor"])
    mouth = g.choice(["grin", "smirk", "shout", "line", "fang"])
    band = g.random() < 0.45
    mark = g.choice([None, None, "scar", "plaster", "blush", "paint"])
    iris = g.choice(IRIS)
    band_c = g.choice(BAND)
    visor_c = g.choice([(255, 60, 90), (60, 220, 255), (255, 210, 60)])
    paint_c = g.choice([(220, 40, 50), (40, 90, 220), (250, 190, 30)])
    spikes = [g.uniform(1.3, 1.45) for _ in range(8)]
    k_ = r / (S * 0.40)                     # sizes below were drawn for r = 0.40 S
    u = lambda v: v * k_                     # noqa: E731

    if hair == "long":
        d.rounded_rectangle((c - r * 1.08, cy - r * 0.6, c + r * 1.08, cy + r * 1.05), int(u(60)),
                            fill=hair_c + (255,), outline=INK, width=int(u(10)))
    d.ellipse((c - r, cy - r, c + r, cy + r), fill=skin + (255,), outline=INK, width=int(u(12)))
    # cel shading: a hard crescent of shadow bottom-right
    sh = Image.new("L", (S, S), 0)
    sd = ImageDraw.Draw(sh)
    sd.ellipse((c - r + 8, cy - r + 8, c + r - 8, cy + r - 8), fill=255)
    sd.ellipse((c - r - u(40), cy - r - u(50), c + r - u(50), cy + r - u(60)), fill=0)
    im.paste(Image.new("RGBA", (S, S), tuple(int(v * 0.8) for v in skin) + (255,)), (0, 0), sh)
    d = ImageDraw.Draw(im)
    d.ellipse((c - r, cy - r, c + r, cy + r), outline=INK, width=int(u(12)))
    if hair in ("spiky", "mohawk"):
        n = 7 if hair == "spiky" else 4
        a0, a1 = (math.pi * 1.05, math.pi * 1.95) if hair == "spiky" else (math.pi * 1.35, math.pi * 1.65)
        pts = []
        for k in range(n + 1):
            a = a0 + (a1 - a0) * k / n
            pts.append((c + r * 0.98 * math.cos(a), cy + r * 0.98 * math.sin(a) + u(30)))
            if k < n:
                am = a + (a1 - a0) / n / 2
                ln = r * spikes[k]
                pts.append((c + ln * math.cos(am), cy + ln * math.sin(am) + u(10)))
        pts.append((c, cy - r * 0.25))
        d.polygon(pts, fill=hair_c + (255,), outline=INK, width=int(u(10)))
    elif hair in ("fringe", "long"):
        d.chord((c - r * 1.02, cy - r * 1.05, c + r * 1.02, cy + r * 0.55), 180, 360, fill=hair_c + (255,),
                outline=INK, width=int(u(10)))
        for k in range(5):
            x = c - r * 0.8 + k * r * 0.4
            d.polygon([(x - r * 0.22, cy - r * 0.26), (x + r * 0.22, cy - r * 0.26), (x, cy - r * 0.02)],
                      fill=hair_c + (255,))
    elif hair == "bun":
        d.ellipse((c - r * 0.35, cy - r * 1.42, c + r * 0.35, cy - r * 0.8), fill=hair_c + (255,), outline=INK,
                  width=int(u(10)))
        d.chord((c - r * 1.02, cy - r * 1.02, c + r * 1.02, cy + r * 0.4), 180, 360, fill=hair_c + (255,),
                outline=INK, width=int(u(10)))
    if band:                                # a plain headband: no symbol, nobody's village
        y = cy - r * 0.42
        bw = u(26)
        d.rectangle((c - r * 0.99, y - bw, c + r * 0.99, y + bw), fill=band_c + (255,), outline=INK, width=int(u(8)))
        d.polygon([(c + r * 0.9, y - u(10)), (c + r * 1.4, y - u(60)), (c + r * 1.35, y - u(10))],
                  fill=band_c + (255,), outline=INK, width=int(u(7)))
        d.polygon([(c + r * 0.9, y + u(6)), (c + r * 1.42, y + u(30)), (c + r * 1.28, y + u(50))],
                  fill=band_c + (255,), outline=INK, width=int(u(7)))
    ey = cy - r * 0.05
    for side in (-1, 1):
        ex = c + side * r * 0.36
        if eyes == "big":
            d.ellipse((ex - u(46), ey - u(56), ex + u(46), ey + u(56)), fill=(255, 255, 255, 255), outline=INK,
                      width=int(u(8)))
            d.ellipse((ex - u(32), ey - u(38), ex + u(32), ey + u(50)), fill=iris + (255,))
            d.ellipse((ex - u(16), ey - u(8), ex + u(16), ey + u(30)), fill=INK)
            d.ellipse((ex - u(22), ey - u(34), ex - u(2), ey - u(14)), fill=(255, 255, 255, 255))
        elif eyes in ("sharp", "angry"):
            tilt = 22 if eyes == "angry" else 10
            # determined: the lid and the brow drop toward the nose
            d.polygon([(ex - side * u(46), ey - u(4)), (ex + side * u(50), ey - u(28 + tilt * 0.4)),
                       (ex + side * u(42), ey + u(22)), (ex - side * u(44), ey + u(22))],
                      fill=(255, 255, 255, 255), outline=INK, width=int(u(8)))
            d.ellipse((ex - u(20), ey - u(14), ex + u(20), ey + u(22)), fill=iris + (255,))
            d.ellipse((ex - u(9), ey - u(2), ex + u(9), ey + u(14)), fill=INK)
            d.line([(ex - side * u(54), ey - u(26 - tilt * 0.3)), (ex + side * u(58), ey - u(60 + tilt))], fill=INK,
                   width=int(u(16)))
        elif eyes == "smug":
            d.arc((ex - u(44), ey - u(20), ex + u(44), ey + u(40)), 200, 340, fill=INK, width=int(u(14)))
            d.line([(ex - side * u(50), ey - u(34)), (ex + side * u(54), ey - u(58))], fill=INK, width=int(u(12)))
    if eyes == "visor":
        d.rounded_rectangle((c - r * 0.85, ey - u(40), c + r * 0.85, ey + u(34)), int(u(30)), fill=(20, 20, 30, 255),
                            outline=INK, width=int(u(8)))
        d.rounded_rectangle((c - r * 0.75, ey - u(22), c + r * 0.75, ey + u(14)), int(u(16)), fill=visor_c + (255,))
    my = cy + r * 0.45
    if mouth == "grin":
        d.chord((c - u(70), my - u(40), c + u(70), my + u(50)), 0, 180, fill=(255, 255, 255, 255), outline=INK,
                width=int(u(9)))
        d.line([(c - u(64), my + u(5)), (c + u(64), my + u(5))], fill=INK, width=int(u(5)))
    elif mouth == "smirk":
        d.arc((c - u(40), my - u(40), c + u(80), my + u(20)), 20, 150, fill=INK, width=int(u(11)))
    elif mouth == "shout":
        d.ellipse((c - u(40), my - u(30), c + u(40), my + u(50)), fill=(120, 20, 30, 255), outline=INK,
                  width=int(u(9)))
        d.chord((c - u(30), my - u(26), c + u(30), my + u(4)), 180, 360, fill=(255, 255, 255, 255))
    elif mouth == "fang":
        d.arc((c - u(60), my - u(50), c + u(60), my + u(20)), 20, 160, fill=INK, width=int(u(11)))
        d.polygon([(c + u(20), my + u(12)), (c + u(40), my + u(10)), (c + u(30), my + u(36))],
                  fill=(255, 255, 255, 255), outline=INK, width=int(u(4)))
    else:
        d.line([(c - u(40), my), (c + u(40), my)], fill=INK, width=int(u(11)))
    if mark == "scar":
        x = c - r * 0.5
        d.line([(x - u(30), ey + u(50)), (x + u(26), ey + u(110))], fill=(170, 40, 50, 255), width=int(u(10)))
        for k in range(3):
            xx, yy = x - u(18) + k * u(20), ey + u(60) + k * u(20)
            d.line([(xx - u(14), yy + u(12)), (xx + u(14), yy - u(12))], fill=(170, 40, 50, 255), width=int(u(6)))
    elif mark == "plaster":
        x, y = c + r * 0.5, ey + u(80)
        d.rounded_rectangle((x - u(46), y - u(18), x + u(46), y + u(18)), int(u(12)), fill=(250, 220, 180, 255),
                            outline=INK, width=int(u(6)))
        for dx in (-10, 10):
            d.line([(x + u(dx), y - u(18)), (x + u(dx), y + u(18))], fill=INK, width=int(u(4)))
    elif mark == "blush":
        for side in (-1, 1):
            x = c + side * r * 0.58
            for k in range(3):
                d.line([(x - u(22) + k * u(16), ey + u(82)), (x - u(30) + k * u(16), ey + u(100))],
                       fill=(240, 90, 110, 255), width=int(u(6)))
    elif mark == "paint":
        for side in (-1, 1):
            x = c + side * r * 0.62
            d.polygon([(x - u(34), ey + u(60)), (x + u(34), ey + u(60)), (x, ey + u(100))], fill=paint_c + (255,))
    return im


# ---- the cut ----------------------------------------------------------------------

def _font(size, name="Anton.ttf"):
    return be._font(size, name)


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _speed_lines(seed, col=(255, 255, 255), n=90, r0=330, alpha=1.0):
    """Radial speed lines on a 2W x 2H sheet, centred."""
    rng = random.Random(seed)
    im = Image.new("RGBA", (2 * W, 2 * H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r1 = r0 + rng.uniform(0, 260)
        wdt = rng.uniform(0.006, 0.02)
        d.polygon([(W + r1 * math.cos(a), H + r1 * math.sin(a)),
                   (W + 2400 * math.cos(a - wdt), H + 2400 * math.sin(a - wdt)),
                   (W + 2400 * math.cos(a + wdt), H + 2400 * math.sin(a + wdt))],
                  fill=col + (int(rng.randint(110, 230) * alpha),))
    return im


def _lines_at(sheet, fx, fy):
    return sheet.crop((int(W - fx), int(H - fy), int(W - fx) + W, int(H - fy) + H))


def _shake(im, amp, seed):
    if amp <= 0:
        return im
    rng = random.Random(seed)
    return Image.fromarray(np.roll(np.asarray(im), (rng.randint(-amp, amp), rng.randint(-amp, amp)), (0, 1)))


def _zoomed(im, z, fx, fy):
    """Crop around (fx, fy) at zoom z; returns the image and where the focus lands."""
    if z <= 1.001:
        return im, (fx, fy)
    cw, ch = W / z, H / z
    x0 = min(max(fx - cw / 2, 0), W - cw)
    y0 = min(max(fy - ch / 2, 0), H - ch)
    out = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)
    return out, ((fx - x0) * z, (fy - y0) * z)


@lru_cache(maxsize=1)
def _vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dist = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip((dist - 0.55) / 0.75, 0, 1)[..., None]


def _dim_edges(im, k, tint=(0, 0, 0)):
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    m = _vignette() * k
    return Image.fromarray((a * (1 - m) + np.array(tint, np.float32) * m).astype(np.uint8))


def _impact(im, invert=True):
    """The anime impact frame: the shapes in stark black and white."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    thr = max(70.0, float(np.percentile(g, 62)) + 30)
    bw = np.where(g > thr, 0 if invert else 255, 255 if invert else 0).astype(np.uint8)
    return Image.fromarray(bw).convert("RGB")


def _text(text, f, fill, stroke, sw):
    l, t, rr, b = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=f, stroke_width=sw)
    im = Image.new("RGBA", (rr - l + 20, b - t + 20), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke)
    return im


def _fit(text, size, maxw, name="Anton.ttf", **kw):
    f = _font(size, name)
    im = _text(text, f, **kw)
    while im.width > maxw and size > 30:
        size -= 4
        im = _text(text, _font(size, name), **kw)
    return im


def _paste(base, sprite, cx, cy, scale=1.0, rot=0.0, alpha=1.0):
    s = sprite
    if scale != 1.0:
        s = s.resize((max(2, int(s.width * scale)), max(2, int(s.height * scale))), Image.BILINEAR)
    if rot:
        s = s.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        s = s.copy()
        s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(s, (int(cx - s.width / 2), int(cy - s.height / 2)))


def _disc(pic, col, size, rim=12):
    from clipper import ball_duel as bd
    disc = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(disc).ellipse((rim, rim, size - rim, size - rim), fill=tuple(col) + (255,),
                                 outline=(255, 255, 255, 255), width=rim)
    if pic:
        g = bd._pic(pic, int(size * 0.78))
        disc.alpha_composite(g, ((size - g.width) // 2, (size - g.height) // 2))
    return disc


class Cut:
    """Draws the cut's own pieces for one fight."""

    def __init__(self, recipe, win):
        self.recipe = recipe
        self.cast = recipe["crowd"]["cast"]
        self.players = recipe["players"]
        self.cols = [tuple(p[2]) for p in self.players]
        self.pics = [p[4] if len(p) > 4 else None for p in self.players]
        self.win = win
        self.lines = [_speed_lines(k) for k in range(3)]
        self.lines_soft = [_speed_lines(10 + k, n=60, r0=470, alpha=0.45) for k in range(3)]
        self.ko = _text("K.O.", _font(300), (255, 214, 40, 255), (0, 0, 0, 255), 18)
        self.lh = _text("LAST HEART!", _font(120), (255, 60, 70, 255), (255, 255, 255, 255), 9)
        self.vs = _text("VS", _font(330), (255, 214, 40, 255), (0, 0, 0, 255), 16)
        self.discs = [_disc(self.pics[i], self.cols[i], 420) for i in range(2)]
        self.names = [_fit(str(self.players[i][0]), 92, 600, fill=(255, 255, 255, 255), stroke=(0, 0, 0, 255), sw=8)
                      for i in range(2)]
        self.quotes = []
        for i in range(2):
            q = self.cast[i].get("quote")
            self.quotes.append(_fit(f"“{q}”", 50, 640, "PermanentMarker-Regular.woff", fill=(255, 236, 120, 255),
                                    stroke=(0, 0, 0, 255), sw=6) if q else None)
        self.band = None

    def intro(self, t):
        """The VS intro: the two slide in on diagonal panels, VS slams down."""
        top = [(0, 0), (W, 0), (W, H * 0.40), (0, H * 0.58)]
        bot = [(0, H * 0.62), (W, H * 0.44), (W, H), (0, H)]
        im = Image.new("RGBA", (W, H), (8, 8, 14, 255))
        d = ImageDraw.Draw(im)
        for poly, c in ((top, self.cols[0]), (bot, self.cols[1])):
            d.polygon(poly, fill=tuple(int(v * 0.55) for v in c) + (255,))
        lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lines)
        rng = random.Random(int(t * FPS) // 2)
        for _ in range(26):
            y, x, ln = rng.uniform(0, H), rng.uniform(-200, W), rng.uniform(160, 520)
            ld.line([(x, y), (x + ln, y - ln * 0.17)], fill=(255, 255, 255, rng.randint(40, 120)),
                    width=rng.randint(3, 9))
        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.polygon(top, fill=255)
        md.polygon(bot, fill=255)
        im.paste(lines, (0, 0), Image.fromarray(np.minimum(np.asarray(mask), np.asarray(lines.getchannel("A")))))
        d = ImageDraw.Draw(im)
        d.line([(0, H * 0.60), (W, H * 0.42)], fill=(255, 255, 255, 255), width=10)
        k = _ease(t / 0.32)
        for i, (cx, cy, side) in enumerate(((W * 0.30, H * 0.24, -1), (W * 0.70, H * 0.79, 1))):
            _paste(im, self.discs[i], cx + side * (1 - k) * 900, cy)
            nx = (W * 0.66 if i == 0 else W * 0.34) - side * (1 - k) * 700
            _paste(im, self.names[i], nx, cy - 70)
            if self.quotes[i] is not None and t > 0.18:
                _paste(im, self.quotes[i], nx, cy + 40, alpha=min(1.0, (t - 0.18) / 0.12))
        if t >= 0.35:
            _paste(im, self.vs, W / 2, H * 0.51, scale=1.0 + 2.0 * (1 - _ease((t - 0.35) / 0.12)), rot=-8)
        out = im.convert("RGB")
        if 0.35 <= t < 0.39:
            return Image.new("RGB", (W, H), (255, 255, 255))
        if 0.39 <= t < 0.6:
            out = _shake(out, int(26 * (1 - (t - 0.39) / 0.21)), int(t * 1000))
        return out

    def cut_band(self):
        """The fighting-game cut-in: a band across the screen with the winner."""
        if self.band is None:
            band = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
            d = ImageDraw.Draw(band)
            c = self.cols[self.win]
            d.polygon([(0, 30), (W, 0), (W, 270), (0, 300)], fill=(10, 10, 16, 240))
            rng = random.Random(7)
            for _ in range(40):
                y, x = rng.uniform(20, 280), rng.uniform(0, W)
                d.line([(x, y), (x + rng.uniform(120, 400), y)], fill=c + (rng.randint(60, 160),),
                       width=rng.randint(3, 8))
            d.line([(0, 30), (W, 0)], fill=c + (255,), width=10)
            d.line([(0, 300), (W, 270)], fill=c + (255,), width=10)
            band.alpha_composite(_disc(self.pics[self.win], c, 250, 8), (W - 330, 25))
            nm = _fit(str(self.players[self.win][0]), 110, W - 420, fill=(255, 255, 255, 255), stroke=(0, 0, 0, 255),
                      sw=8)
            band.alpha_composite(nm, (60, 150 - nm.height // 2))
            self.band = band
        return self.band


def _unflash(base, sim):
    """Take the duel's own white K.O. flash back out: the impact frames do that job."""
    if sim.done_at is not None and sim.t - sim.done_at < 0.25:
        a = 110 * (1 - (sim.t - sim.done_at) / 0.25) / 255
        arr = (np.asarray(base, dtype=np.float32) - 255 * a) / (1 - a)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return base


def frames(recipe, sim_cls, painter_cls):
    """Yields (image, fight time or None) for every output frame; the sim
    and the marks (output seconds of vs / fight / slow / ko) are on the
    returned generator's ``state`` dict once it's done."""
    probe = sim_cls(recipe)
    last_heart = None
    while not probe.finished:
        probe.step()
        if last_heart is None and probe.done_at is None and any(b["hp"] == 1 for b in probe.balls):
            last_heart = probe.t
    t_ko, win = probe.done_at, probe.winner["i"] if probe.winner else 0
    sim, painter = sim_cls(recipe), painter_cls(recipe)
    cut = Cut(recipe, win)
    state = {"sim": sim, "marks": {"vs": 0.35}, "t_ko": t_ko, "last_heart": last_heart}
    n = [0]

    def now():
        return n[0] / FPS

    def out(im, t):
        n[0] += 1
        return im, t

    yield state
    for f in range(int(INTRO_S * FPS)):
        yield out(cut.intro(f / FPS), None)
    for _ in range(3):
        yield out(Image.new("RGB", (W, H), (255, 255, 255)), 0.0)
    state["marks"]["fight"] = now()
    seen = 0
    focus = None
    ko_done = False
    slow_start = None
    while not sim.finished:
        sim.step()
        base = _unflash(painter.frame(sim), sim)
        new = sim.sfx[seen:]
        seen = len(sim.sfx)
        ts = sim.t
        in_slow = t_ko is not None and sim.done_at is None and ts >= t_ko - SLOW_LEAD
        after = sim.done_at is not None and ts - sim.done_at <= AFTER_SLOW
        pts = [painter.P(b["x"], b["y"]) for b in sim.balls]
        target = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2) if sim.done_at is None else pts[win]
        focus = target if focus is None else (focus[0] + (target[0] - focus[0]) * 0.25,
                                             focus[1] + (target[1] - focus[1]) * 0.25)
        if sim.done_at is not None and not ko_done:
            # the K.O.: impact frames, then a freeze with the stamp
            ko_done = True
            state["marks"]["ko"] = now()
            zf, fpos = _zoomed(base, ZOOM, *focus)
            for kind in ("inv", "inv", "white", "inv", "pos"):
                if kind == "white":
                    yield out(Image.new("RGB", (W, H), (255, 255, 255)), ts)
                else:
                    yield out(_impact(zf, kind == "inv"), ts)
            hold = int(KO_HOLD * FPS)
            for k in range(hold):
                fr = zf.convert("RGBA")
                fr.alpha_composite(_lines_at(cut.lines[k % 3], *fpos))
                fr = _dim_edges(fr, 0.5).convert("RGBA")
                _paste(fr, cut.ko, W / 2, H * 0.30, scale=1.0 + 1.3 * (1 - _ease(k / 6)), rot=-7)
                yield out(_shake(fr.convert("RGB"), int(30 * (1 - k / hold)), k), ts)
            continue
        if in_slow:
            if slow_start is None:
                slow_start = now()
                state["marks"]["slow"] = now()
            reps = SLOW_EVERY
        elif after:
            reps = 2
        else:
            reps = 1 + (HITSTOP if any(kind == "hit" and v >= 2 for (_t, kind, v) in new) else 0)
        for k in range(reps):
            fr = base
            if in_slow or after:
                z = (1 + (ZOOM - 1) * _ease((now() - slow_start) / 0.45)) if in_slow else \
                    ZOOM - (ZOOM - 1) * _ease((ts - sim.done_at) / AFTER_SLOW)
                fr, fpos = _zoomed(fr, z, *focus)
                if in_slow:
                    fr = fr.convert("RGBA")
                    fr.alpha_composite(_lines_at(cut.lines_soft[n[0] % 3], *fpos))
                    fr = _dim_edges(fr, 0.65)
                    age = now() - slow_start
                    if age < 0.7:
                        slide = (1 - _ease(age / 0.14)) * W if age < 0.5 else -_ease((age - 0.5) / 0.2) * W
                        fr = fr.convert("RGBA")
                        fr.alpha_composite(cut.cut_band(), (int(slide), int(H * 0.16)))
                else:
                    fr = _dim_edges(fr, 0.65 * (1 - (ts - sim.done_at) / AFTER_SLOW)).convert("RGBA")
                    _paste(fr, cut.ko, W / 2, H * 0.30, rot=-7, alpha=1 - _ease((ts - sim.done_at) / AFTER_SLOW))
            else:
                if k > 0:                                  # hit-stop: hold, and shake
                    fr = _shake(fr, 22 - 4 * k, n[0])
                if last_heart is not None and ts >= last_heart and sim.done_at is None:
                    # the last heart: red edges pulsing with a heartbeat
                    ph = ((ts - last_heart) * 1.25) % 1.0
                    beat = math.exp(-ph * 9) + (0.7 * math.exp(-(ph - 0.18) * 9) if ph > 0.18 else 0)
                    fr = _dim_edges(fr, 0.25 + 0.35 * min(1, beat), (170, 0, 12))
                    age = ts - last_heart
                    if age < 1.1:
                        fr = fr.convert("RGBA")
                        _paste(fr, cut.lh, W / 2, H * 0.82, scale=1 + 0.6 * (1 - _ease(age / 0.15)),
                               alpha=min(1, (1.1 - age) / 0.2))
            yield out(fr.convert("RGB"), ts)
    state["marks"]["end"] = now()


# ---- the sound ------------------------------------------------------------------

def _rng(seed):
    return np.random.default_rng(seed)


def _taiko(rng, f0=70, dur=0.5, gain=1.0):
    t = np.arange(int(SR * dur)) / SR
    f = f0 * (1 + 1.4 * np.exp(-t * 30))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    skin = rng.normal(0, 1, len(t)) * np.exp(-t * 60) * 0.35
    return ((body + skin) * gain).astype(np.float32)


def _rim(rng, gain=0.4):
    t = np.arange(int(SR * 0.06)) / SR
    return (rng.normal(0, 1, len(t)) * np.exp(-t * 90) * gain).astype(np.float32)


def _sweep(rng, dur, f_from, f_to, gain=0.5, rise=True):
    """Noise through a one-pole low-pass sweeping from f_from to f_to."""
    nn = max(1, int(SR * dur))
    x = rng.normal(0, 1, nn)
    a = 1 - np.exp(-2 * np.pi * np.geomspace(f_from, f_to, nn) / SR)
    out = np.empty(nn)
    y = 0.0
    for i in range(nn):
        y += a[i] * (x[i] - y)
        out[i] = y
    shape = np.linspace(0.05, 1, nn) ** 2 if rise else np.linspace(1, 0, nn) ** 2
    return (out / (np.max(np.abs(out)) or 1) * shape * gain).astype(np.float32)


def _boom(rng, dur=1.6, gain=1.0):
    t = np.arange(int(SR * dur)) / SR
    f = 30 + 70 * np.exp(-t * 6)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    crack = rng.normal(0, 1, len(t)) * np.exp(-t * 18) * 0.6
    return ((sub + crack) * gain).astype(np.float32)


def _shing(dur=1.4, gain=0.35):
    t = np.arange(int(SR * dur)) / SR
    out = np.zeros(len(t))
    for fr, dec in ((2093, 2.5), (3136, 3.0), (4699, 4.0), (6272, 5.5)):
        out += np.sin(2 * np.pi * fr * t * (1 + 0.002 * np.sin(2 * np.pi * 6 * t))) * np.exp(-t * dec)
    return (out / 4 * gain * np.minimum(1, t / 0.004)).astype(np.float32)


def mix(sim, tmap, marks, t_ko, last_heart, seed=0) -> np.ndarray:
    """The fight's own sounds moved to where they now land (deeper in the
    slow-motion), plus the cut's: intro whoosh + slam, our taiko beat
    (busier on the last heart), a heartbeat, a riser into a breath of
    silence, the K.O. boom, three drum hits for the win."""
    from clipper import ball_duel as bd
    rng = _rng(seed)
    n = int(SR * (len(tmap) / FPS + 0.5))
    out = np.zeros(n, np.float32)
    fight_ts = [t if t is not None else -1.0 for t in tmap]

    def at(t):
        return bisect.bisect_left(fight_ts, t) / FPS

    def put(w, when, gain=1.0):
        i = int(when * SR)
        if 0 <= i < n:
            seg = out[i:i + len(w)]
            seg += w[:len(seg)] * gain

    cache: dict = {}
    last: dict = {}
    for k, (t, kind, _val) in enumerate(sim.sfx):
        if kind == "wall" and t - last.get(kind, -9) < 0.05:
            continue
        last[kind] = t
        w = cache.get((kind, k % 3))
        if w is None:
            w = cache[(kind, k % 3)] = bd._sound(kind, k % 3).astype(np.float32)
        if t_ko is not None and t_ko - SLOW_LEAD <= t < t_ko:       # slowed: deeper
            w = np.interp(np.arange(0, len(w), 1 / 1.7), np.arange(len(w)), w).astype(np.float32)
        put(w, at(t), 0.8)
    put(_sweep(rng, 0.32, 300, 6000, 0.45), 0.0)
    put(_boom(rng, 1.0, 0.8), marks["vs"])
    put(_shing(0.9, 0.25), marks["vs"])
    put(_sweep(rng, 0.2, 5000, 400, 0.3, rise=False), INTRO_S - 0.05)
    ko = marks.get("ko", len(tmap) / FPS)
    slow = marks.get("slow", ko)
    lh = at(last_heart) if last_heart is not None else ko
    pattern = [1, 0, 2, 0, 1, 1, 2, 0]                       # 1 = don (drum), 2 = ka (rim)
    t, step = marks["fight"], 0
    while t < slow - 0.05:
        busy = t > lh
        p = pattern[step % 8]
        if p == 1:
            put(_taiko(rng, 68 if step % 4 else 62, 0.45, 0.45 if busy else 0.32), t)
        elif p == 2 or busy:
            put(_rim(rng, 0.26 if busy else 0.18), t)
        t += 0.2
        step += 1
    if last_heart is not None:
        beat = np.concatenate([_taiko(rng, 48, 0.16, 0.5), np.zeros(int(SR * 0.05), np.float32),
                               _taiko(rng, 44, 0.25, 0.35)])
        t = lh
        while t < slow - 0.1:
            put(beat, t)
            t += 0.8
    if "ko" in marks:
        put(_sweep(rng, max(0.3, ko - 0.12 - (slow - 1.2)), 200, 9000, 0.45), slow - 1.2)
        put(_sweep(rng, 0.35, 4000, 300, 0.35, rise=False), slow)
        put(_shing(0.8, 0.2), slow + 0.05)
        i0, i1 = int((ko - 0.12) * SR), int(ko * SR)
        if 0 <= i0 < i1 <= n:
            out[i0:i1] *= np.linspace(1, 0.05, i1 - i0)        # a breath of silence before the hit
        put(_boom(rng, 2.2, 1.3), ko)
        put(_shing(1.6, 0.4), ko)
        put(_taiko(rng, 55, 0.8, 0.9), ko + 0.02)
        won = at(t_ko + 0.6)
        for k, dt in enumerate((0.0, 0.18, 0.36)):
            put(_taiko(rng, 64 - 4 * k, 0.6, 0.55 + 0.2 * k), won + dt)
    out = np.tanh(out * 1.2)
    out = out / (np.max(np.abs(out)) or 1) * 0.9
    end = int(SR * len(tmap) / FPS)
    out = out[:end]
    k = int(SR * be.END_FADE)
    if 0 < k < len(out):
        out[-k:] *= np.linspace(1, 0, k)
    return out.astype(np.float32)


def render(recipe: dict, out_path, log=print, progress=None) -> dict:
    """``ball_evolution.render`` for an anime-cut comment fight."""
    out_path = Path(out_path)
    video = out_path.with_suffix(".video.mp4")
    wav = out_path.with_suffix(".wav")
    sim_cls, painter_cls = be._parts(recipe)
    gen = frames(recipe, sim_cls, painter_cls)
    state = next(gen)
    proc = subprocess.Popen(
        [be._ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
         str(video)], stdin=subprocess.PIPE)
    total = ((recipe.get("expected_end") or 40) + be.END_HOLD + EXTRA_S) * FPS
    tmap = []
    try:
        for im, t in gen:
            proc.stdin.write(im.tobytes())
            tmap.append(t)
            if progress and len(tmap) % 30 == 0:
                progress(min(0.97, len(tmap) / total))
    finally:
        proc.stdin.close()
        proc.wait()
    sim = state["sim"]
    pcm = (mix(sim, tmap, state["marks"], state["t_ko"], state["last_heart"], int(recipe.get("seed") or 0))
           * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    subprocess.run([be._ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)], check=True)
    video.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    seconds = len(tmap) / FPS
    chain = getattr(sim, "chain", None) or be.recipe_chain(recipe)
    info = dict(recipe, seconds=round(seconds, 1), dropped=sim.spawned,
                done_at=sim.done_at and round(sim.done_at, 1),
                unlocks={chain[k][1]: round(t, 1) for k, t in sorted(sim.unlocked_at.items())},
                cut_marks={k: round(v, 2) for k, v in state["marks"].items()})
    if recipe.get("crowd") and getattr(sim, "winner", None):
        info["crowd"] = dict(recipe["crowd"], winner=sim.winner["i"])
    log(f"rendered {out_path} (anime cut): {info}")
    return info
