"""N countries, 1 ring, 1 survivor (Dean, 29 Oct).

From the research on what breaks out in this niche: "198 countries. 1 ring.
1 survivor 🌍 Comment your country before you watch" had 1.4M views on a
1.2K-follower account, while our own 16-flag battle royale with guns and
hearts was the channel's weakest video. So this one is simple: flag balls
bounce in a ring with a gap that turns, a ball that finds the gap falls
out and is out, the last one in the ring wins. Nothing to explain.

The gap widens as fewer are left (`S_GAP0` x (N / left) ^ `S_GAP_GROW`), so
the first ones go quickly and the last few still don't drag on. Every ball
keeps a steady speed with a tiny random turn (two balls can't loop forever
missing the gap). ``format: "survivor"``, a ``NO_PACK`` kind (the balls are
the countries); ``ball_evolution._parts`` dispatches here.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from clipper import ball_duel as bd
from clipper.ball_evolution import FPS, H, W

END_HOLD = bd.END_HOLD
SECONDS_CAP = 70.0
S_N = 24                          # countries in the ring
S_CX, S_CY, S_R = 540, 1170, 440
S_BALL = 40
S_SPEED = 600.0
S_SPIN = 1.15                     # rad/s the gap turns
S_GAP0 = 0.19                     # the gap's half-angle with everyone in (rad)
S_GAP_GROW = 0.62                 # how much it widens as they go out
S_GAP_MAX = 0.95
S_FALL = 2200.0                   # px/s^2 once a ball is out
S_FIRST_OUT = 2.5                 # a good seed has its first one out by then
S_MAX_WAIT = 4.5                  # and never waits longer between two
SURVIVOR_HOOKS = ["{n} countries. 1 ring. 1 survivor", "{n} countries enter the ring. Only 1 stays",
                  "Last country in the ring wins", "Pick your country: {n} in, 1 survives"]


def survivor_recipe(recipe: dict) -> dict:
    rng = random.Random(f"survivor-recipe-{recipe['seed']}")
    codes = rng.sample(sorted(bd.COUNTRIES), S_N)
    recipe.update(look="flags", players=[bd._country(c) for c in codes], bed="off",
                  hook=rng.choice(SURVIVOR_HOOKS).format(n=S_N), ending="SURVIVES!")
    return recipe


def _diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class SurvivorSim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"survivor-{recipe['seed']}")
        self.t = 0.0
        self.gap = self.rng.uniform(0, 2 * math.pi)
        self.balls = []
        players = recipe.get("players") or [bd._country(c) for c in sorted(bd.COUNTRIES)[:S_N]]
        for i, p in enumerate(players):
            for _ in range(200):          # spread out, not overlapping
                a, r = self.rng.uniform(0, 2 * math.pi), (S_R - S_BALL - 10) * math.sqrt(self.rng.random())
                x, y = S_CX + r * math.cos(a), S_CY + r * math.sin(a)
                if all(math.hypot(x - b["x"], y - b["y"]) > 2 * S_BALL + 6 for b in self.balls):
                    break
            v = self.rng.uniform(0, 2 * math.pi)
            self.balls.append({"i": i, "name": p[0], "col": tuple(p[2]), "flag": p[3] if len(p) > 3 else None,
                               "x": x, "y": y, "vx": S_SPEED * math.cos(v), "vy": S_SPEED * math.sin(v),
                               "state": "in", "out_at": None, "px": x, "py": y})
        self.n = len(self.balls)
        self.sfx: list = []
        self.events: list = []        # the shared mixer's ball sounds: none (sfx instead)
        self.hits: list = []
        self.outs: list = []          # (t, ball index)
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.spawned = len(self.balls)   # (render's info: how many balls)
        self.winner = None
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def left(self) -> int:
        return sum(1 for b in self.balls if b["state"] == "in")

    def half_gap(self) -> float:
        return min(S_GAP_MAX, S_GAP0 * (self.n / max(1, self.left())) ** S_GAP_GROW)

    def step(self):
        dt = 1 / FPS
        self.t += dt
        for b in self.balls:
            b["px"], b["py"] = b["x"], b["y"]
        if self.winner is not None:
            w = self.winner
            w["x"] += (S_CX - w["x"]) * 0.08
            w["y"] += (S_CY - w["y"]) * 0.08
            self._fall(dt)
            return
        self.gap = (self.gap + S_SPIN * dt) % (2 * math.pi)
        for _ in range(4):
            self._move(dt / 4)
        self._fall(dt)
        if self.t > SECONDS_CAP - END_HOLD - 0.1 and self.done_at is None:
            self.done_at = self.t
            self.ok = False

    def _move(self, dt):
        live = [b for b in self.balls if b["state"] == "in"]
        hw = self.half_gap()
        for b in live:
            a = self.rng.uniform(-0.025, 0.025)            # a tiny turn: no endless loops
            c, s = math.cos(a), math.sin(a)
            b["vx"], b["vy"] = b["vx"] * c - b["vy"] * s, b["vx"] * s + b["vy"] * c
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            dx, dy = b["x"] - S_CX, b["y"] - S_CY
            dd = math.hypot(dx, dy)
            if dd > S_R - S_BALL:
                ang = math.atan2(dy, dx)
                if abs(_diff(ang, self.gap)) < hw - 0.6 * S_BALL / S_R:
                    b["state"] = "falling"                   # through the gap: out
                    continue
                nx, ny = dx / dd, dy / dd
                vn = b["vx"] * nx + b["vy"] * ny
                if vn > 0:
                    vx, vy = b["vx"] - 2 * vn * nx, b["vy"] - 2 * vn * ny
                    va = math.atan2(vy, vx) + self.rng.uniform(-0.15, 0.15)
                    b["vx"], b["vy"] = S_SPEED * math.cos(va), S_SPEED * math.sin(va)
                    self.hits.append((self.t, 0.5))
                    self.sfx.append((self.t, "wall", 0))
                b["x"], b["y"] = S_CX + nx * (S_R - S_BALL), S_CY + ny * (S_R - S_BALL)
        for i, a in enumerate(live):
            for c in live[i + 1:]:
                dx, dy = c["x"] - a["x"], c["y"] - a["y"]
                dd = math.hypot(dx, dy)
                if 0 < dd < 2 * S_BALL:
                    nx, ny = dx / dd, dy / dd
                    rel = (c["vx"] - a["vx"]) * nx + (c["vy"] - a["vy"]) * ny
                    if rel < 0:
                        a["vx"] += rel * nx
                        a["vy"] += rel * ny
                        c["vx"] -= rel * nx
                        c["vy"] -= rel * ny
                        self.sfx.append((self.t, "bump", 0))
                    over = 2 * S_BALL - dd
                    a["x"] -= nx * over / 2
                    a["y"] -= ny * over / 2
                    c["x"] += nx * over / 2
                    c["y"] += ny * over / 2
                    for b in (a, c):
                        s = math.hypot(b["vx"], b["vy"]) or 1
                        b["vx"], b["vy"] = b["vx"] / s * S_SPEED, b["vy"] / s * S_SPEED

    def _fall(self, dt):
        for b in self.balls:
            if b["state"] != "falling":
                continue
            b["vy"] += S_FALL * dt
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            if b["out_at"] is None and math.hypot(b["x"] - S_CX, b["y"] - S_CY) > S_R + S_BALL:
                b["out_at"] = self.t
                self.outs.append((self.t, b["i"]))
                self.sfx.append((self.t, "out", b["i"] % 3))
                left = [o for o in self.balls if o["state"] == "in"]
                if len(left) == 1 and self.winner is None:
                    self._end(left[0])
            if b["y"] > H + 200:
                b["state"] = "gone"

    def _end(self, winner):
        self.winner = winner
        self.done_at = self.t
        self.sfx.append((self.t + 0.2, "win", 0))
        times = [0.0] + [t for t, _ in self.outs]
        self.max_gap = max(b - a for a, b in zip(times, times[1:]))
        self.ok = len(times) > 1 and times[1] <= S_FIRST_OUT and self.max_gap <= S_MAX_WAIT

    def music_track(self, n: int) -> np.ndarray:
        sfx = [e for e in self.sfx if e[1] != "wall" or (int(e[0] * 40) % 3 == 0)]   # 24 balls: fewer ticks
        return bd._sfx_mix(self.recipe, sfx, self.hits, n, self)


def _ring_layer() -> Image.Image:
    """The glowing ring, drawn once (the gap is cut out each frame)."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((S_CX - S_R, S_CY - S_R, S_CX + S_R, S_CY + S_R), outline=(150, 170, 255, 200),
                                 width=18)
    lay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(14)))
    ImageDraw.Draw(lay).ellipse((S_CX - S_R, S_CY - S_R, S_CX + S_R, S_CY + S_R), outline=(245, 245, 255, 255),
                                width=7)
    return lay


class SurvivorPainter:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.players = recipe.get("players") or []
        self.bg = Image.new("RGBA", (W, H), (7, 9, 22, 255))
        bd_ = ImageDraw.Draw(self.bg)
        bd_.ellipse((S_CX - S_R, S_CY - S_R, S_CX + S_R, S_CY + S_R), fill=(12, 14, 34, 255))
        self.ring = _ring_layer()
        self.trails = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.f_hook = bd._font(56)
        self.f_left = bd._font(72)
        self.f_out = bd._font(40)
        self.f_win = bd._font(92)
        self.f_small = bd._font(34, "Inter-Bold.ttf")
        self.ball_img: dict = {}
        self.cell = 92                     # the grid of who's still in
        self.cols = 8

    def sprite(self, b, size):
        key = (b["i"], size)
        im = self.ball_img.get(key)
        if im is None:
            s3 = size * 3
            im = Image.new("RGBA", (s3 + 12, s3 + 12), (0, 0, 0, 0))
            if b.get("flag"):
                im.alpha_composite(bd._flag(b["flag"], s3), (6, 6))
            else:
                ImageDraw.Draw(im).ellipse((6, 6, 6 + s3, 6 + s3), fill=b["col"] + (255,))
            ImageDraw.Draw(im).ellipse((6, 6, 6 + s3, 6 + s3), outline=(255, 255, 255, 255), width=12)
            im = self.ball_img[key] = im.resize((size + 4, size + 4), Image.LANCZOS)
        return im

    def frame(self, sim: SurvivorSim) -> Image.Image:
        im = self.bg.copy()
        # trails: each ball's colour, fading
        a = np.asarray(self.trails).copy()
        a[..., 3] = (a[..., 3] * 0.84).astype(np.uint8)
        self.trails = Image.fromarray(a)
        td = ImageDraw.Draw(self.trails)
        for b in sim.balls:
            if b["state"] == "in":
                td.line([(b["px"], b["py"]), (b["x"], b["y"])], fill=b["col"] + (150,), width=14)
        im.alpha_composite(self.trails)
        ring = self.ring.copy()
        if sim.winner is None or sim.t - sim.done_at < 0.4:
            hw = sim.half_gap()
            rd = ImageDraw.Draw(ring)
            a0, a1 = math.degrees(sim.gap - hw), math.degrees(sim.gap + hw)
            rd.arc((S_CX - S_R - 30, S_CY - S_R - 30, S_CX + S_R + 30, S_CY + S_R + 30), a0, a1,
                   fill=(0, 0, 0, 0), width=60)
            for e in (sim.gap - hw, sim.gap + hw):           # the gap's two ends, so it's easy to see
                x, y = S_CX + S_R * math.cos(e), S_CY + S_R * math.sin(e)
                rd.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(255, 80, 90, 255), outline=(255, 255, 255, 255),
                           width=4)
        im.alpha_composite(ring)
        d = ImageDraw.Draw(im)
        for b in sim.balls:
            if b["state"] == "gone":
                continue
            size = 2 * S_BALL
            if sim.winner is b:
                size = int(2 * S_BALL * (1 + min(1.4, (sim.t - sim.done_at) * 2.5)))
            sp = self.sprite(b, size)
            im.alpha_composite(sp, (int(b["x"] - sp.width / 2), int(b["y"] - sp.height / 2)))
        self.hud(im, d, sim)
        if sim.winner is not None and sim.t - sim.done_at > 0.35:
            self.ending(im, d, sim)
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((W - 60 - d.textlength(mark, font=self.f_small), H - 70), mark, font=self.f_small,
                   fill=(255, 255, 255, 150))
        return im.convert("RGB")

    def hud(self, im, d, sim):
        hook = self.recipe.get("hook") or ""
        f = self.f_hook
        while d.textlength(hook, font=f) > W - 80 and f.size > 30:
            f = bd._font(f.size - 4)
        d.text((W / 2, 150), hook, font=f, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=4, stroke_fill=(7, 9, 22, 255))
        rows = math.ceil(len(sim.balls) / self.cols)
        x0 = (W - self.cols * self.cell) / 2 + self.cell / 2
        out_at = {i: t for t, i in sim.outs}
        for b in sim.balls:
            r, c = divmod(b["i"], self.cols)
            x, y = x0 + c * self.cell, 255 + r * self.cell
            sp = self.sprite(b, 72)
            t = out_at.get(b["i"])
            if t is not None:
                g = sp.copy()
                g.putalpha(g.getchannel("A").point(lambda v: int(v * 0.35)))
                im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
                k = min(1.0, (sim.t - t) / 0.15)
                q = 26 * k
                d.line([(x - q, y - q), (x + q, y + q)], fill=(255, 70, 80, 255), width=8)
                d.line([(x - q, y + q), (x + q, y - q)], fill=(255, 70, 80, 255), width=8)
            else:
                im.alpha_composite(sp, (int(x - sp.width / 2), int(y - sp.height / 2)))
                if sim.winner is b:
                    d.ellipse((x - 42, y - 42, x + 42, y + 42), outline=(255, 214, 80, 255), width=6)
        y_left = 255 + rows * self.cell + 10
        left = sim.left()
        d.text((W / 2, y_left), f"{left} LEFT", font=self.f_left, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=6, stroke_fill=(220, 40, 60, 255))
        if sim.outs and sim.winner is None:
            t, i = sim.outs[-1]
            age = sim.t - t
            if age < 1.0:
                b = sim.balls[i]
                a = int(255 * min(1.0, (1.0 - age) / 0.3))
                d.text((W / 2, y_left + 64), f"{b['name'].upper()} IS OUT", font=self.f_out,
                       fill=(255, 255, 255, a), anchor="mm", stroke_width=4, stroke_fill=bd._stroke(b["col"]) + (a,))

    def ending(self, im, d, sim):
        w = sim.winner
        k = min(1.0, (sim.t - sim.done_at - 0.35) / 0.25)
        text = f"{w['name'].upper()} {self.recipe.get('ending') or 'SURVIVES!'}"
        f = self.f_win
        while d.textlength(text, font=f) > W - 120:
            f = bd._font(f.size - 4)
        s = 0.6 + 0.4 * k
        txt = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((W / 2, 100), text, font=f, fill=(255, 255, 255, 255), anchor="mm",
                                 stroke_width=10, stroke_fill=bd._stroke(w["col"]) + (255,))
        txt = txt.resize((int(W * s), int(200 * s)), Image.LANCZOS)
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(S_CY + 250 - txt.height / 2)))
