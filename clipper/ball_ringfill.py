"""Will it fill the whole circle? (Dean, 17 Nov).

From a reel Dean sent (230K likes, 10K shares): a hollow circle bounces
inside a big one and grows on every bounce. When it fills the space it
locks in as a coloured ring, and a new small circle starts in the middle of
the smaller space that's left, round after round, until the last one fills
the centre. Dean's notes on the first sample: the growing circle, not a new
band per bounce; bigger and smoother growth; no music, sound effects only;
and it has to really finish (the first sample's last round filled itself).

Sim + painter for ``format: "ringfill"``, a ``NO_PACK`` kind
(``ball_evolution._parts`` dispatches here). The growth is eased: a bounce
raises the circle's target size and it grows into it over ~0.15 s, so it
swells smoothly instead of jumping. Each locked ring is as thick as a share
of the space it filled (`F_BAND`), so the rounds get quicker toward the end.
The sound (``music_track``) is ours, numpy only: a tap on each bounce that
gets deeper as the circle grows, a thump + whoosh when one locks in, a
longer one on the last.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from clipper.ball_evolution import FPS, H, SR, W, _font

END_HOLD = 1.8
SECONDS_CAP = 70.0
F_CX, F_CY, F_R0 = 540, 1010, 505      # the big circle (bigger than the sample's 470)
F_START = 14.0                         # a new circle's radius
F_STROKE = 12                          # its outline
F_GROW = 0.15                          # each bounce: +15% of the space left ...
F_GROW_ADD = 2.5                       # ... plus a little, so it always gets there
F_EASE = 14.0                          # how fast it swells into its new size (1/s)
F_SPEED = 900.0
F_SPEEDUP = 0.04                       # each round a little faster
F_BAND = 0.16                          # a locked ring is this share of the radius it filled
F_BAND_MIN = 16
F_GAP = 4
F_LAST = 52                            # below this, the next circle is the last: it fills the centre
PALETTES = [
    [(247, 214, 110), (243, 176, 86), (233, 136, 84), (220, 104, 100), (196, 80, 122), (150, 66, 140), (110, 62, 150)],
    [(140, 230, 255), (110, 190, 255), (120, 150, 250), (150, 120, 240), (190, 100, 220), (230, 90, 180)],
    [(180, 250, 170), (120, 225, 170), (80, 200, 190), (70, 160, 210), (80, 120, 220), (100, 90, 210)],
    [(255, 190, 200), (255, 150, 170), (245, 110, 140), (220, 80, 120), (180, 60, 110), (130, 50, 100)],
]
HOOKS = ["Will it fill the whole circle?", "Every bounce makes it bigger", "How many circles fit inside?",
         "It grows until it fits", "Fill it, then start again", "Circle inside a circle inside a circle",
         "One more circle every time", "Will the last one fill the middle?"]


def ringfill_recipe(recipe: dict) -> dict:
    rng = random.Random(f"ringfill-recipe-{recipe['seed']}")
    recipe.update(hook=rng.choice(HOOKS), palette=rng.randrange(len(PALETTES)), bed="off", ending="FULL!")
    return recipe


def rounds_for(r0: float = F_R0) -> list:
    """The radius of the space at the start of every round."""
    out, R = [], r0
    while True:
        out.append(R)
        if R < F_LAST:
            return out
        R -= max(F_BAND_MIN, F_BAND * R) + F_GAP


def colour(recipe: dict, k: int, n: int) -> tuple:
    pal = PALETTES[int(recipe.get("palette") or 0) % len(PALETTES)]
    t = k / max(1, n - 1) * (len(pal) - 1)
    a = int(t)
    f, b = t - a, min(a + 1, len(pal) - 1)
    return tuple(int(pal[a][i] + (pal[b][i] - pal[a][i]) * f) for i in range(3))


class RingFillSim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"ringfill-{recipe['seed']}")
        self.t = 0.0
        self.plan = rounds_for()
        self.n = len(self.plan)
        self.chain = [(None, f"circle {i + 1}") for i in range(self.n + 1)]   # (render's info: the rounds)
        self.R = F_R0
        self.round = 0
        self.locked: list = []          # (outer, inner, round, t); inner 0 = the solid centre
        self.sfx: list = []             # (t, kind, value)
        self.events: list = []          # the shared mixer's ball sounds: none
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.spawned = 1
        self.done_at = None
        self.bounces = 0
        self._spawn()

    def _spawn(self):
        self.x, self.y = float(F_CX), float(F_CY)
        self.r = self.target = F_START
        a = self.rng.uniform(0, 2 * math.pi)
        sp = F_SPEED * (1 + F_SPEEDUP * self.round)
        self.vx, self.vy = sp * math.cos(a), sp * math.sin(a)

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    @property
    def last(self) -> bool:
        return self.round == self.n - 1

    def step(self):
        dt = 1 / FPS
        self.t += dt
        if self.done_at is not None:
            return
        for _ in range(8):
            if self._move(dt / 8):
                break
        if self.t > SECONDS_CAP - END_HOLD - 0.1 and self.done_at is None:
            self.done_at = self.t

    def _move(self, dt) -> bool:
        self.r += (self.target - self.r) * min(1.0, dt * F_EASE)
        self.x += self.vx * dt
        self.y += self.vy * dt
        dx, dy = self.x - F_CX, self.y - F_CY
        d = math.hypot(dx, dy) or 1e-6
        if self.R - self.r < 2.5 and self.R - self.target < 2.5:
            return self._lock()
        if d + self.r >= self.R:
            nx, ny = dx / d, dy / d
            dot = self.vx * nx + self.vy * ny
            if dot > 0:
                self.vx -= 2 * dot * nx
                self.vy -= 2 * dot * ny
                a = math.atan2(self.vy, self.vx) + self.rng.uniform(-0.25, 0.25)
                sp = F_SPEED * (1 + F_SPEEDUP * self.round)
                self.vx, self.vy = sp * math.cos(a), sp * math.sin(a)
                self.target = min(self.R, self.target + F_GROW * (self.R - self.target) + F_GROW_ADD)
                self.bounces += 1
                self.sfx.append((self.t, "tap", self.r / F_R0))
            back = max(0.0, self.R - self.r - 0.5)
            self.x, self.y = F_CX + nx * min(d, back), F_CY + ny * min(d, back)
        return False

    def _lock(self) -> bool:
        if self.last:
            self.locked.append((self.R, 0.0, self.round, self.t))
            self.sfx.append((self.t, "end", 0))
            self.done_at = self.t
            self.unlocked_at[self.round] = self.t
            return True
        inner = self.R - max(F_BAND_MIN, F_BAND * self.R)
        self.locked.append((self.R, inner, self.round, self.t))
        self.sfx.append((self.t, "lock", self.round))
        self.round += 1
        self.unlocked_at[self.round] = self.t
        self.best = self.round
        self.R = self.plan[self.round]
        self.spawned += 1
        self._spawn()
        return True

    def music_track(self, n: int) -> np.ndarray:
        """Sound effects only (Dean: no music)."""
        mix = np.zeros(n, np.float32)
        g = np.random.default_rng(int(self.recipe.get("seed") or 0))
        last = -1.0
        for et, kind, v in self.sfx:
            a0 = int(et * SR)
            if a0 >= n:
                continue
            if kind == "tap":
                if et - last < 0.035:
                    continue
                last = et
                f = 1400 * (1 - 0.75 * v) + 180
                ts = np.arange(int(SR * 0.09)) / SR
                w = (np.sin(2 * np.pi * f * ts) * 0.6 + g.normal(0, 1, len(ts)) * 0.25) * np.exp(-ts * 55) * 0.35
            else:
                d = 0.9 if kind == "lock" else 1.6
                ts = np.arange(int(SR * d)) / SR
                f = 110 * np.exp(-ts * 4) + 42
                thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ts * (5 if kind == "lock" else 2.5))
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(40) / 40, "same")
                whoosh = noise * np.sin(np.pi * np.minimum(1, ts / d)) * 0.5
                w = (thump * 0.8 + whoosh * 0.6) * 0.6
            seg = mix[a0:a0 + len(w)]
            seg += w[:len(seg)].astype(np.float32)
        peak = float(np.abs(mix).max()) or 1.0
        return mix / max(1.0, peak / 0.9)


class RingFillPainter:
    SS = 2                              # rings are drawn at 2x and reduced (smooth edges)

    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.bg = Image.new("RGB", (W, H), (8, 8, 10))
        d = ImageDraw.Draw(self.bg)
        self.f_hook = _font(64)
        self.f_count = _font(46)
        self.f_end = _font(104)
        self.layer = None
        self.layer_n = -1
        self._base = None

    def _rings(self, sim: RingFillSim) -> Image.Image:
        """The big circle and the locked rings, redrawn only when one locks in."""
        if self.layer is not None and self.layer_n == len(sim.locked):
            return self.layer
        S = self.SS
        im = Image.new("RGB", (W * S, H * S), (8, 8, 10))
        d = ImageDraw.Draw(im)
        R0 = F_R0 + 8
        d.ellipse([S * (F_CX - R0), S * (F_CY - R0), S * (F_CX + R0), S * (F_CY + R0)], outline=(95, 85, 70), width=3 * S)
        for ro, ri, k, _t in sim.locked:
            d.ellipse([S * (F_CX - ro), S * (F_CY - ro), S * (F_CX + ro), S * (F_CY + ro)], fill=colour(self.recipe, k, sim.n))
            if ri > 0:
                d.ellipse([S * (F_CX - ri), S * (F_CY - ri), S * (F_CX + ri), S * (F_CY + ri)], fill=(8, 8, 10))
        self.layer = im.resize((W, H), Image.LANCZOS)
        self.layer_n = len(sim.locked)
        return self.layer

    def _circle(self, im: Image.Image, x: float, y: float, r: float, c: tuple, fill: bool = False, width: int = F_STROKE):
        """A smooth circle: drawn at 2x on a patch and reduced."""
        S = self.SS
        pad = width + 4
        x0, y0 = int(x - r - pad), int(y - r - pad)
        size = int(2 * (r + pad)) + 2
        patch = Image.new("L", (size * S, size * S), 0)
        cx, cy = (x - x0) * S, (y - y0) * S
        dd = ImageDraw.Draw(patch)
        box = [cx - r * S, cy - r * S, cx + r * S, cy + r * S]
        if fill:
            dd.ellipse(box, fill=255)
        else:
            dd.ellipse(box, outline=255, width=max(1, width * S))
        mask = patch.resize((size, size), Image.LANCZOS)
        im.paste(Image.new("RGB", (size, size), c), (x0, y0), mask)

    def frame(self, sim: RingFillSim) -> Image.Image:
        im = self._rings(sim).copy()
        if sim.locked:
            ro, ri, k, tl = sim.locked[-1]
            age = sim.t - tl
            if age < 0.2:                       # a flash as it locks in
                w = (1 - age / 0.2)
                self._circle(im, F_CX, F_CY, ro + 6 + 22 * w, (255, 245, 225), width=max(2, int(6 * w)))
        if sim.done_at is None:
            c = colour(self.recipe, sim.round, sim.n)
            self._circle(im, sim.x, sim.y, sim.r, c)
        d = ImageDraw.Draw(im)
        d.text((W // 2, 300), self.recipe.get("hook") or HOOKS[0], font=self.f_hook, fill=(240, 228, 205), anchor="mm")
        d.text((W // 2, 1600), f"Circle {min(sim.round + 1, sim.n)}", font=self.f_count, fill=(150, 140, 120),
               anchor="mm")
        if sim.done_at is not None and sim.t - sim.done_at > 0.25:
            k = min(1.0, (sim.t - sim.done_at - 0.25) / 0.2)
            f = self.f_end if k >= 1 else _font(int(60 + 44 * k))
            d.text((W // 2, 1710), self.recipe.get("ending") or "FULL!", font=f, fill=(247, 214, 110), anchor="mm",
                   stroke_width=6, stroke_fill=(0, 0, 0))
        return im
