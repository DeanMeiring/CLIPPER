"""Cursed picture packs, made by the app itself (Dean: "find a source you are
allowed to use or generate your own cursed ones").

Every picture is cut from the bundled Fluent Emoji 3D set (MIT, already
in ``assets/emoji``) and edited in Pillow/numpy, so the packs are ours to
post. A pack is one face going through the edits cursed / sigma meme
pages use, each a bit more absurd than the last:

    Cat > Chubby or Long Cat > Cool Cat (sunglasses) > Buff Cat (flexing
    arms) > Sigma Cat (black-and-white edit, shades, tie) > Cursed Cat
    (human lips, extra eyes, a swirl) > Deep Fried Cat (over-saturated,
    crunchy, red laser eyes) > Cat God (crown, wings, golden glow)

21 faces x a few variations, so the autopilot always has a fresh pack.
"""
from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .ball_packs import PackStore

EMOJI = Path(__file__).resolve().parent / "assets" / "emoji"
S = 512

# face emoji: (code, name, eye height, eye x offset from the middle) as
# fractions of the picture, measured on the Fluent art
BASES = [
    ("1f431", "Cat", 0.53, 0.13), ("1f436", "Dog", 0.42, 0.12), ("1f438", "Frog", 0.22, 0.25),
    ("1f435", "Monkey", 0.46, 0.10), ("1f437", "Pig", 0.42, 0.12), ("1f439", "Hamster", 0.53, 0.10),
    ("1f43b", "Bear", 0.46, 0.11), ("1f43c", "Panda", 0.46, 0.14), ("1f98a", "Fox", 0.58, 0.11),
    ("1f42f", "Tiger", 0.48, 0.12), ("1f42e", "Cow", 0.38, 0.17), ("1f42d", "Mouse", 0.60, 0.08),
    ("1f430", "Rabbit", 0.60, 0.10), ("1f981", "Lion", 0.55, 0.10), ("1f428", "Koala", 0.45, 0.15),
    ("1f99d", "Raccoon", 0.50, 0.13), ("1f43a", "Wolf", 0.55, 0.11), ("1f47d", "Alien", 0.47, 0.17),
    ("1f916", "Robot", 0.47, 0.12), ("1f383", "Pumpkin", 0.45, 0.14), ("1f4a9", "Poop", 0.50, 0.17),
]
HOOKS = ["What does the {n} become?", "What will the {n} turn into?", "Guess the final {n}",
         "How cursed does the {n} get?", "Wait for the final {n}..."]


def _emoji(code: str) -> Image.Image:
    for name in (f"{code}.webp", f"{code}-fe0f.webp"):
        if (EMOJI / name).is_file():
            return Image.open(EMOJI / name).convert("RGBA")
    raise FileNotFoundError(code)


def _fit(im: Image.Image, w: int, h: Optional[int] = None) -> Image.Image:
    return im.resize((int(w), int(h if h is not None else w)), Image.LANCZOS)


def _canvas() -> Image.Image:
    return Image.new("RGBA", (S, S), (0, 0, 0, 0))


def _put(canvas: Image.Image, im: Image.Image, cx: float, cy: float) -> None:
    canvas.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


class Face:
    def __init__(self, code: str, name: str, eye_y: float, eye_dx: float, im: Optional[Image.Image] = None):
        """An emoji face (code), or any square RGBA picture (im: a photo's
        face cut out by ball_photos)."""
        self.code, self.name, self.eye_y, self.eye_dx = code, name, eye_y, eye_dx
        self.im = _fit(im if im is not None else _emoji(code), S * 0.86)

    def eyes(self, size: float, cx: float, cy: float) -> list:
        """Where the eyes are when the face is drawn size wide at cx, cy."""
        top = cy - size / 2
        return [(cx + sx * self.eye_dx * size, top + self.eye_y * size) for sx in (-1, 1)]


# ---- the edits -----------------------------------------------------------

def plain(f: Face, rng) -> Image.Image:
    c = _canvas()
    _put(c, f.im, S / 2, S / 2)
    return c


def chubby(f: Face, rng) -> Image.Image:
    c = _canvas()
    _put(c, _fit(f.im, S * 0.98, S * 0.70), S / 2, S * 0.56)
    return c


def long(f: Face, rng) -> Image.Image:
    c = _canvas()
    _put(c, _fit(f.im, S * 0.50, S * 0.98), S / 2, S / 2)
    return c


def cool(f: Face, rng) -> Image.Image:
    c = plain(f, rng)
    size = f.im.width
    (lx, ly), (rx, ry) = f.eyes(size, S / 2, S / 2)
    glasses = _fit(_emoji("1f576"), min(size * 0.85, max(rx - lx, size * 0.2) * 2.1))
    _put(c, glasses, (lx + rx) / 2, ly + glasses.height * 0.05)
    return c


def buff(f: Face, rng) -> Image.Image:
    c = _canvas()
    # the 💪 points its fist up and left: flipped for the right side, so
    # both fists rise beside the face
    arm = _fit(_emoji("1f4aa"), S * 0.44)
    _put(c, arm, S * 0.20, S * 0.50)
    _put(c, ImageOps.mirror(arm), S * 0.80, S * 0.50)
    _put(c, _fit(f.im, S * 0.56), S / 2, S * 0.56)
    return c


def sigma(f: Face, rng) -> Image.Image:
    c = _canvas()
    size = S * 0.74
    face = _fit(f.im, size)
    cx, cy = S / 2, S * 0.40
    tie = _fit(_emoji("1f454"), S * 0.40)
    _put(c, tie, cx, cy + size * 0.52)
    _put(c, face, cx, cy)
    (lx, ly), (rx, ry) = f.eyes(size, cx, cy)
    glasses = _fit(_emoji("1f576"), min(size * 0.85, max(rx - lx, size * 0.2) * 2.1))
    _put(c, glasses, (lx + rx) / 2, ly + glasses.height * 0.05)
    # the black-and-white "sigma edit": grey, hard contrast
    a = c.getchannel("A")
    g = ImageOps.grayscale(c.convert("RGB")).point(lambda v: int(255 * (v / 255) ** 1.8))
    g = ImageEnhance.Contrast(g).enhance(1.5)
    out = Image.merge("RGBA", (g, g, g, a))
    return out


def _swirl(im: Image.Image, strength: float) -> Image.Image:
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, R = w / 2, h / 2, min(w, h) / 2
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    t = np.clip(1 - r / R, 0, 1) ** 2 * strength
    ang = np.arctan2(dy, dx) + t
    sx = np.clip(cx + r * np.cos(ang), 0, w - 1).astype(np.int32)
    sy = np.clip(cy + r * np.sin(ang), 0, h - 1).astype(np.int32)
    return Image.fromarray(a[sy, sx].astype(np.uint8), "RGBA")


def cursed(f: Face, rng) -> Image.Image:
    c = plain(f, rng)
    size = f.im.width
    (lx, ly), (rx, ry) = f.eyes(size, S / 2, S / 2)
    lips = _fit(_emoji("1f444"), size * 0.42)
    _put(c, lips, S / 2, min(S * 0.86, ly + size * 0.26))
    eye = _fit(_emoji("1f441"), size * 0.22)
    for x, y in ((S / 2, ly - size * 0.17), (lx - size * 0.05, ly - size * 0.02), (rx + size * 0.05, ly - size * 0.02)):
        _put(c, eye, x, y)
    return _swirl(c, rng.uniform(0.7, 1.1) * rng.choice((-1, 1)))


def deep_fried(f: Face, rng) -> Image.Image:
    c = plain(f, rng)
    a = c.getchannel("A")
    rgb = c.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(3.2)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.9)
    rgb = ImageEnhance.Sharpness(rgb).enhance(4.0)
    rgb = ImageChops.multiply(rgb, Image.new("RGB", rgb.size, (255, 200, 150)))
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=7)                  # the crunch
    rgb = Image.open(buf).convert("RGB")
    out = Image.merge("RGBA", (*rgb.split(), a))
    # red laser eyes: a glow on each eye and a beam out to the side
    size = f.im.width
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    for k, (x, y) in enumerate(f.eyes(size, S / 2, S / 2)):
        end = (0, y - 40) if k == 0 else (S, y - 40)
        d.line([(x, y), end], fill=(255, 30, 20, 210), width=26)
        d.ellipse((x - 46, y - 46, x + 46, y + 46), fill=(255, 40, 30, 230))
    glow = glow.filter(ImageFilter.GaussianBlur(10))
    d = ImageDraw.Draw(glow)
    for k, (x, y) in enumerate(f.eyes(size, S / 2, S / 2)):
        end = (0, y - 40) if k == 0 else (S, y - 40)
        d.line([(x, y), end], fill=(255, 240, 220, 255), width=6)
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=(255, 255, 240, 255))
    out.alpha_composite(glow)
    return out


def chad(f: Face, rng) -> Image.Image:
    """The chad-cat look: from the eyes down the face widens and stretches
    smoothly into a huge jaw, with harder contrast."""
    size = int(S * 0.70)
    src = np.asarray(_fit(f.im, size)).astype(np.float32)
    e = size * min(0.8, f.eye_y + 0.05)                   # the stretch starts just under the eyes
    stretch, widen = 1.5, 0.38
    total = e + (size - e) * stretch
    y0 = (S - total) / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ys = yy - y0
    ys = np.where(ys > e, e + (ys - e) / stretch, ys)
    t = np.clip((ys - e) / (size - e), 0, 1)
    k = 1 + widen * t * t * (3 - 2 * t)                    # smoothstep: no seam
    xs = (xx - S / 2) / k + size / 2
    ok = (ys >= 0) & (ys <= size - 1) & (xs >= 0) & (xs <= size - 1)
    out = src[np.clip(ys, 0, size - 1).astype(np.int32), np.clip(xs, 0, size - 1).astype(np.int32)]
    out[~ok] = 0
    c = Image.fromarray(out.astype(np.uint8), "RGBA")
    a = c.getchannel("A")
    rgb = ImageEnhance.Contrast(ImageEnhance.Sharpness(c.convert("RGB")).enhance(2.0)).enhance(1.3)
    return Image.merge("RGBA", (*rgb.split(), a))


def god(f: Face, rng) -> Image.Image:
    c = _canvas()
    # golden glow behind
    halo = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse((40, 60, S - 40, S - 20), fill=(255, 200, 60, 200))
    c.alpha_composite(halo.filter(ImageFilter.GaussianBlur(40)))
    wing = _fit(_emoji("1fabd"), S * 0.46)
    _put(c, ImageOps.mirror(wing), S * 0.22, S * 0.54)
    _put(c, wing, S * 0.78, S * 0.54)
    face = _fit(f.im, S * 0.62)
    gold = ImageChops.multiply(face.convert("RGB"), Image.new("RGB", face.size, (255, 225, 140)))
    face = Image.merge("RGBA", (*gold.split(), face.getchannel("A")))
    _put(c, face, S / 2, S * 0.58)
    _put(c, _fit(_emoji("1f451"), S * 0.40), S / 2, S * 0.20)
    spark = _fit(_emoji("2728"), S * 0.16)
    for x, y in ((S * 0.12, S * 0.18), (S * 0.88, S * 0.22), (S * 0.86, S * 0.86)):
        _put(c, spark, x, y)
    return c


# step name pattern, edit
STEPS = [
    [("{n}", plain)],
    [("Chubby {n}", chubby), ("Long {n}", long)],
    [("Cool {n}", cool)],
    [("Buff {n}", buff)],
    [("Sigma {n}", sigma)],
    [("Cursed {n}", cursed)],
    [("Deep Fried {n}", deep_fried)],
    [("{n} God", god)],
]


# a photo's steps: the chad jaw stretch can stand in for the sigma edit
PHOTO_STEPS = STEPS[:4] + [[("Sigma {n}", sigma), ("Chad {n}", chad)]] + STEPS[5:]


def make(base: str, seed: int, face: Optional[Face] = None, steps: Optional[list] = None) -> dict:
    """{title, hook, items: [(name, RGBA)]} for one face (an emoji from
    BASES, or a ready Face)."""
    if face is None:
        code, name, ey, ex = next(b for b in BASES if b[0] == base)
        face = Face(code, name, ey, ex)
    f, name = face, face.name
    rng = random.Random(f"cursed-{base}-{seed}")
    items = []
    for options in steps or STEPS:
        label, fn = rng.choice(options)
        items.append((label.format(n=name), fn(f, rng)))
    return {"title": f"{name} to {name} God", "hook": rng.choice(HOOKS).format(n=name.lower()),
            "items": items}


def next_base(store: PackStore, rng: Optional[random.Random] = None) -> str:
    """A face that hasn't had a cursed pack lately."""
    used = [p.get("cursed_base") for p in store.packs() if p.get("cursed_base")]
    recent = set(used[-len(BASES) + 3:]) if used else set()
    pool = [b[0] for b in BASES if b[0] not in recent] or [b[0] for b in BASES]
    return (rng or random.Random()).choice(pool)


def new_pack(store: PackStore, seed: Optional[int] = None, base: Optional[str] = None) -> dict:
    """Make a cursed pack and save it in the store (source "cursed")."""
    seed = seed if seed is not None else random.randrange(10 ** 6)
    base = base or next_base(store, random.Random(seed))
    p = make(base, seed)
    return store.save(p["title"], p["items"], "cursed",
                      {"hook": p["hook"], "cursed_base": base, "credit": None})
