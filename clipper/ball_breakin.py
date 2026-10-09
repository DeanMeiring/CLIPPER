"""Ball Evolution's "Break in" format (Dean's idea, picked from two
prototypes: the first had the tiles in the corridor and broke through
too quickly).

* One ball bounces in a rainbow ring at the bottom: no gravity, a steady
  speed and a tiny random turn each step, so it can't loop forever.
* The ring has a small gap at the top into a plain corridor that leads up
  to a room packed with tiles. Finding the gap is the wait.
* Every tile a ball breaks makes one more ball where it was, so once the
  first ball is in, it multiplies fast until the room is cleared. Tiles
  near the entrance break in one hit; deeper ones show how many hits they
  still need (up to 6), which keeps the chain reaction watchable (with
  one hit each the whole room went in about a second).

The theme's chain drives the ladder (ball count on a log scale up to the
number of tiles; the last item appears when the room is cleared), and the
look is the ring formats' neon one: trails, a glowing rainbow ring,
glowing balls, a dark background. ball_evolution dispatches here on
``recipe["format"] == "breakin"``, so picking, rendering and the sounds
are shared.
"""
from __future__ import annotations

import colorsys
import math
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from clipper import ball_circles as bc
from clipper import ball_evolution as be

W, H, FPS = be.W, be.H, be.FPS
RX, RY, RR = 540, 1440, 290          # the ring at the bottom
GAP_HW = 46                          # half the gap's width = the corridor's
COR_BOT = RY - math.sqrt(RR ** 2 - GAP_HW ** 2)
ROOM = (60, 350, 1020, 880)          # the tile room
COR_TOP = ROOM[3]
BALL_R = 12
SPEED = 520
TILE_W, TILE_H, TILE_GAP = 80, 40, 4
HP_NEAR, HP_DEEP = 1, 6              # hits a tile needs: the entrance row .. the far row
MAX_BALLS = 400
END_HOLD = 4.0
SECONDS_CAP = 75.0


def _walls(space, pymunk, pts) -> None:
    for p, q in pts:
        s = pymunk.Segment(space.static_body, p, q, 5)
        s.elasticity, s.friction = 1.0, 0.0
        s.collision_type = 3
        space.add(s)


def wall_lines() -> tuple:
    """The ring's points and the straight walls (corridor + room)."""
    a = math.asin(GAP_HW / RR)
    a0, span, n = -math.pi / 2 + a, 2 * math.pi - 2 * a, 80
    ring = [(RX + RR * math.cos(a0 + span * i / n), RY + RR * math.sin(a0 + span * i / n)) for i in range(n + 1)]
    x0, y0, x1, y1 = ROOM
    straight = [((RX - GAP_HW, COR_BOT + 4), (RX - GAP_HW, COR_TOP)), ((RX + GAP_HW, COR_BOT + 4), (RX + GAP_HW, COR_TOP)),
                ((x0, y0), (x1, y0)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1)),
                ((x0, y1), (RX - GAP_HW, y1)), ((RX + GAP_HW, y1), (x1, y1))]
    return ring, straight


class BreakInSim:
    """Same interface as ball_evolution.Sim (picking, rendering, audio)."""

    def __init__(self, recipe: dict):
        import pymunk

        self.pymunk = pymunk
        self.recipe = recipe
        self.chain = be.THEMES[recipe["theme"]]["chain"]
        self.last = len(self.chain) - 1
        self.rng = random.Random(f"breakin-{recipe['seed']}")
        sp = self.space = pymunk.Space()
        sp.gravity = (0, 0)
        sp.iterations = 10
        ring, straight = wall_lines()
        _walls(sp, pymunk, list(zip(ring, ring[1:])) + straight)
        # the room, packed with tiles
        self.tiles = []
        x0, y0, x1, y1 = ROOM
        cols = int((x1 - x0 - TILE_GAP) // (TILE_W + TILE_GAP))
        rows = int((y1 - y0 - TILE_GAP) // (TILE_H + TILE_GAP))
        ox = x0 + (x1 - x0 - cols * (TILE_W + TILE_GAP) + TILE_GAP) / 2
        oy = y0 + (y1 - y0 - rows * (TILE_H + TILE_GAP) + TILE_GAP) / 2
        for r in range(rows):
            depth = 1 - r / max(1, rows - 1)            # the top row is the deepest
            for c in range(cols):
                x, y = ox + c * (TILE_W + TILE_GAP), oy + r * (TILE_H + TILE_GAP)
                s = pymunk.Poly(sp.static_body, [(x, y), (x + TILE_W, y), (x + TILE_W, y + TILE_H), (x, y + TILE_H)])
                s.elasticity, s.friction = 1.0, 0.0
                s.collision_type = 5
                s.tile = {"rect": (x, y, x + TILE_W, y + TILE_H), "row": r, "col": c, "rows": rows, "cols": cols,
                          "hp": round(HP_NEAR + (HP_DEEP - HP_NEAR) * depth), "flash": 0.0}
                sp.add(s)
                self.tiles.append(s)
        self.tile_total = len(self.tiles)
        # the ladder: ball count on a log scale; the last item = room cleared
        self.thresholds = [round((self.tile_total + 1) ** (k / self.last)) for k in range(self.last + 1)]
        self.thresholds[-1] = 10 ** 9

        self.balls: list[dict] = []
        self.events: list[tuple] = []
        self.effects: list[list] = []
        self.reveal_flash = [0.0] * len(self.chain)
        self.unlocked_at = {0: 0.0}
        self.banner = None
        self.best = 0
        self.spawned = 1
        self.entered_at = None
        self.t = 0.0
        self.done_at = None
        self._dead: list = []
        self._spawns: list = []
        sp.on_collision(1, 5, begin=self._on_tile)
        sp.on_collision(1, 3, begin=self._on_wall)
        self._add((RX, RY + 60), self.rng.uniform(0, 2 * math.pi))

    def _add(self, pos, ang):
        pm = self.pymunk
        body = pm.Body()
        body.position = pos
        body.velocity = (SPEED * math.cos(ang), SPEED * math.sin(ang))
        shape = pm.Circle(body, BALL_R)
        shape.density, shape.elasticity, shape.friction = 1.0, 1.0, 0.0
        shape.collision_type = 1
        shape.filter = pm.ShapeFilter(group=1)      # balls pass through each other
        self.space.add(body, shape)
        b = {"body": body, "shape": shape, "tier": self.best, "r": BALL_R, "sounded": -1.0}
        shape.ball = b
        self.balls.append(b)
        return b

    def _on_tile(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        tile = arbiter.shapes[1]
        if b is None or tile in self._dead:
            return
        tile.tile["hp"] -= 1
        tile.tile["flash"] = 1.0
        if tile.tile["hp"] > 0:
            self.events.append((self.t, "tick", min(9, tile.tile["col"]), 0.6))
            return
        self._dead.append(tile)
        x0, y0, x1, y1 = tile.tile["rect"]
        self._spawns.append(((x0 + x1) / 2, (y0 + y1) / 2))
        if len(self.effects) < 14:
            self.effects.append([(x0 + x1) / 2, (y0 + y1) / 2, 24, 1.0, b["tier"]])
        self.events.append((self.t, "knock", "bumper", 0.8))

    def _on_wall(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        if b is None or self.t - b["sounded"] < 0.1:
            return
        b["sounded"] = self.t
        x = b["body"].position.x
        self.events.append((self.t, "tick", int(max(0, min(9, (x - 200) / 68))), 0.7))

    def _tier(self) -> int:
        if self.done_at is not None:
            return self.last
        return max(k for k, th in enumerate(self.thresholds) if self.spawned >= th or k == 0)

    def _reveal(self):
        tier = self._tier()
        if tier > self.best:
            for k in range(self.best + 1, tier + 1):
                self.unlocked_at[k] = self.t
                self.reveal_flash[k] = 1.0
            self.best = tier
            self.banner = (f"NEW: {self.chain[tier][1]}!", self.t)
            self.events.append((self.t, "unlock", tier, 1.0))

    def step(self):
        dt = 1 / FPS
        for _ in range(3):
            self.space.step(dt / 3)
        for tile in self._dead:
            if tile in self.tiles:
                self.space.remove(tile)
                self.tiles.remove(tile)
        self._dead = []
        for pos in self._spawns:
            if len(self.balls) < MAX_BALLS:
                self._add(pos, self.rng.uniform(0, 2 * math.pi))
                self.events.append((self.t, "pop", min(self.best, 6), 0.5))
        self._spawns = []
        # a steady speed, and a tiny random turn so no ball loops forever
        for b in self.balls:
            v = b["body"].velocity.rotated(self.rng.gauss(0, 0.02))
            b["body"].velocity = v * (SPEED / (v.length or 1))
        for tile in self.tiles:
            tile.tile["flash"] *= 0.8
        if self.entered_at is None and any(b["body"].position.y < COR_TOP for b in self.balls):
            self.entered_at = self.t
            self.banner = ("IT'S IN!", self.t)
            self.events.append((self.t, "unlock", 0, 1.0))
        self.spawned = len(self.balls)
        if self.done_at is None and not self.tiles:
            self.done_at = self.t
            self.events.append((self.t, "fanfare", 0, 1.0))
        self._reveal()
        self.t += dt

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP


class BreakInPainter(be.Painter):
    TRAIL_FADE = 0.8

    def __init__(self, recipe: dict):
        super().__init__(recipe)
        kind, seed = recipe.get("backdrop"), recipe.get("seed") or 0
        if kind == "space":
            self.bg = bc._space(seed)
        elif kind == "blurred":
            self.bg = bc._blurred(self.emoji, seed)
        else:
            self.bg = Image.new("RGB", (W, H), (0, 0, 0))
        self.trail = Image.new("RGB", (W, H))
        self.fade = [int(v * self.TRAIL_FADE) for v in range(256)] * 3
        self.trail_cols = [tuple(int(v * 0.75) for v in c) for c in self.colours]
        self.glow: dict = {}
        self.f_tile = be._font(28)
        self.f_count = be._font(46)
        self.static = self._walls()

    def _walls(self) -> Image.Image:
        """Rainbow ring, corridor and room outline with a soft glow, drawn once."""
        ring, straight = wall_lines()
        n = len(ring) - 1
        lines = []
        for i, (p, q) in enumerate(zip(ring, ring[1:])):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 1)
            lines.append(((p, q), (int(r * 255), int(g * 255), int(b * 255))))
        lines += [(seg, (235, 235, 255)) for seg in straight]
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for seg, c in lines:
            gd.line(seg, fill=c + (170,), width=26)
        im = glow.filter(ImageFilter.GaussianBlur(12))
        d = ImageDraw.Draw(im)
        for seg, c in lines:
            d.line(seg, fill=c + (255,), width=10)
        return im

    def sprite(self, tier: int, r: float) -> Image.Image:
        key = (tier, int(round(r)))
        if key not in self.glow:
            sp = super().sprite(tier, r)
            pad = 9
            im = Image.new("RGBA", (sp.width + 2 * pad, sp.height + 2 * pad), (0, 0, 0, 0))
            c, rr = im.width / 2, sp.width / 2 + 2
            ImageDraw.Draw(im).ellipse((c - rr, c - rr, c + rr, c + rr), fill=self.colours[tier] + (160,))
            im = im.filter(ImageFilter.GaussianBlur(5))
            im.alpha_composite(sp, (pad, pad))
            self.glow[key] = im
        return self.glow[key]

    def frame(self, sim: BreakInSim) -> Image.Image:
        self.trail = self.trail.point(self.fade)
        td = ImageDraw.Draw(self.trail)
        for b in sim.balls:
            x, y = b["body"].position
            r = b["r"] * 0.7
            col = self.trail_cols[b["tier"]]
            px, py = b.get("trail_at") or (x, y)
            if (px - x) ** 2 + (py - y) ** 2 > 4:
                td.line([(px, py), (x, y)], fill=col, width=int(2 * r))
            td.ellipse((x - r, y - r, x + r, y + r), fill=col)
            b["trail_at"] = (x, y)
        im = ImageChops.add(self.bg, self.trail)
        im.paste(self.static, (0, 0), self.static)
        d = ImageDraw.Draw(im, "RGBA")
        for t in sim.tiles:
            x0, y0, x1, y1 = t.tile["rect"]
            hue = (t.tile["col"] / t.tile["cols"] * 0.85 + t.tile["row"] / t.tile["rows"] * 0.15) % 1
            f = t.tile["flash"]
            col = tuple(int(v * 255 + (255 - v * 255) * f) for v in colorsys.hsv_to_rgb(hue, 0.7, 1))
            d.rounded_rectangle((x0, y0, x1, y1), 7, fill=col + (230,))
            if t.tile["hp"] > 1:
                txt = str(t.tile["hp"])
                tw = d.textlength(txt, font=self.f_tile)
                d.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - 17), txt, font=self.f_tile, fill=(20, 16, 30, 230))
        for b in sim.balls:
            x, y = b["body"].position
            sp = self.sprite(b["tier"], b["r"])
            im.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)
        for e in sim.effects:
            x, y, r, life, tier = e
            rr = r * (1.0 + (1 - life) * 1.3)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=(255, 255, 255, int(220 * life)), width=max(2, int(5 * life)))
            e[3] -= 0.1
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]
        self.header(im, d, sim)
        n = len(sim.balls)
        self.ctext(d, RY + RR + 22, f"{n:,} ball{'s' if n != 1 else ''} · {len(sim.tiles)} tiles left",
                   self.f_count, (255, 230, 140, 255))
        self.watermark(d, W - 40, H - 60)
        mid = (ROOM[1] + ROOM[3]) / 2
        self.overlays(im, d, sim, item_y=mid - 40, text_y=mid + 120)
        return im
