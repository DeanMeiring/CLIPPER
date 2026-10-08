"""Two more Ball Evolution formats, both in a big rotating ring.

* ``escape`` -- the ring has a small gap and spins. Balls bounce inside
  under gravity; every ball that gets out through the gap makes two new
  ones appear in the middle. Out goes one, in come two, until the ring is
  full.
* ``touch`` -- a closed ring, no gravity. Balls fly around; when two touch,
  a new one pops out between them (each ball rests a moment before it can
  multiply again), until the ring is full.

The theme's evolution chain still drives the ladder at the top: as the
count passes each milestone (spaced evenly on a log scale up to "full"),
new balls come out as the next item and it's revealed. Everything else --
themes, ball styles, backgrounds, bounce sounds, the noise bed, the
ending -- is shared with clipper/ball_evolution.py, which dispatches here
on ``recipe["format"]``.
"""
from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from clipper import ball_evolution as be

W, H, FPS = be.W, be.H, be.FPS
CX, CY, RING_R = 540, 1010, 410
BALL_R = 22
SECONDS_CAP = 75.0
END_HOLD = 4.0
FORMATS = ("escape", "touch")


def capacity(fmt: str = "touch") -> int:
    """How many balls make the ring "full": about half its area, a bit less
    for escape (with gravity they pile up, so it looks full sooner)."""
    return int((0.42 if fmt == "escape" else 0.5) * (RING_R / BALL_R) ** 2)


class CircleSim:
    """Same interface as ball_evolution.Sim, so picking, rendering and the
    audio mix work unchanged."""

    def __init__(self, recipe: dict):
        import pymunk

        self.pymunk = pymunk
        self.recipe = recipe
        self.format = recipe.get("format") or "escape"
        self.chain = be.THEMES[recipe["theme"]]["chain"]
        self.last = len(self.chain) - 1
        self.rng = random.Random(f"{self.format}-{recipe['seed']}")
        self.cap = capacity(self.format)
        # count thresholds for each tier: 1 .. cap on a log scale
        start = 2 if self.format == "touch" else 1
        self.thresholds = [round(start * (self.cap / start) ** (k / self.last)) for k in range(self.last + 1)]
        self.thresholds[-1] = self.cap

        self.space = pymunk.Space()
        self.space.gravity = (0, 700 if self.format == "escape" else 0)
        self.space.iterations = 15
        self.ring = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.ring.position = (CX, CY)
        self.spin = (self.rng.uniform(2.5, 3.2) if self.format == "escape" else self.rng.uniform(0.4, 0.8)) \
            * self.rng.choice((-1, 1))
        self.ring.angular_velocity = self.spin
        self.gap = self.rng.uniform(0.75, 0.9) if self.format == "escape" else 0.0
        segs = 80
        start_a = self.gap / 2
        span = 2 * math.pi - self.gap
        pts = [(RING_R * math.cos(start_a + span * i / segs), RING_R * math.sin(start_a + span * i / segs))
               for i in range(segs + 1)]
        self.space.add(self.ring)
        for a, b in zip(pts, pts[1:]):
            s = pymunk.Segment(self.ring, a, b, 6)
            s.elasticity, s.friction = (1.0, 0.2) if self.format == "escape" else (1.0, 0.0)
            s.collision_type = 3
            s.part = "ring"
            self.space.add(s)

        self.balls: list[dict] = []
        self.events: list[tuple] = []
        self.effects: list[list] = []
        self.reveal_flash = [0.0] * len(self.chain)
        self.unlocked_at = {0: 0.0}
        self.banner = None
        self.best = 0
        self.spawned = 0          # balls inside the ring (the count on screen)
        self.escaped = 0
        self.t = 0.0
        self.done_at = None
        self._spawns: list[tuple] = []
        self.space.on_collision(1, 3, begin=self._on_ring)
        self.space.on_collision(1, 1, begin=self._on_ball)
        if self.format == "touch":
            for k in range(2):
                self._add((CX + (-120 if k == 0 else 120), CY + self.rng.uniform(-40, 40)),
                          aim=(CX, CY))
        else:
            self._add((CX, CY))

    # ---- balls
    def _tier(self) -> int:
        n = len([b for b in self.balls if b["inside"]])
        return max(k for k, th in enumerate(self.thresholds) if n >= th or k == 0)

    def _add(self, pos, aim=None):
        pm = self.pymunk
        body = pm.Body()
        body.position = pos
        if aim:
            dx, dy = aim[0] - pos[0], aim[1] - pos[1]
            d = math.hypot(dx, dy) or 1
            speed = 420
            body.velocity = (dx / d * speed, dy / d * speed + self.rng.uniform(-60, 60))
        else:
            a = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(250, 420)
            body.velocity = (speed * math.cos(a), speed * math.sin(a) - (150 if self.format == "escape" else 0))
        shape = pm.Circle(body, BALL_R)
        shape.density = 1.0
        shape.elasticity = 1.0
        shape.friction = 0.2 if self.format == "escape" else 0.0
        shape.collision_type = 1
        self.space.add(body, shape)
        b = {"body": body, "shape": shape, "tier": self._tier() if self.balls else 0, "r": BALL_R,
             "inside": True, "rest_until": self.t + 0.8, "sounded": -1.0}
        shape.ball = b
        self.balls.append(b)
        self.spawned = sum(1 for x in self.balls if x["inside"])
        return b

    def _on_ring(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        if b is None or self.t - b["sounded"] < 0.1:
            return
        n = arbiter.contact_point_set.normal
        speed = abs((arbiter.shapes[0].body.velocity - arbiter.shapes[1].body.velocity).dot(n))
        if speed < 60:
            return
        b["sounded"] = self.t
        ang = math.atan2(b["body"].position.y - CY, b["body"].position.x - CX)
        note = int((math.cos(ang) + 1) / 2 * 9.99)          # left low, right high
        self.events.append((self.t, "tick", note, min(1.0, speed / 700)))

    def _on_ball(self, arbiter, space, data):
        if self.format != "touch" or self.done_at is not None:
            return
        a, c = (getattr(s, "ball", None) for s in arbiter.shapes)
        if not (a and c) or self.t < a["rest_until"] or self.t < c["rest_until"]:
            return
        a["rest_until"] = c["rest_until"] = self.t + 1.2
        pa, pc = a["body"].position, c["body"].position
        self._spawns.append(((pa.x + pc.x) / 2, (pa.y + pc.y) / 2))

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
        # touch: keep everyone moving at a lively, steady speed
        if self.format == "touch":
            for b in self.balls:
                # a tiny random turn, so two balls can't fly a loop forever
                # that never meets
                b["body"].velocity = b["body"].velocity.rotated(self.rng.gauss(0, 0.03))
                v = b["body"].velocity
                sp = v.length
                if sp < 280 and sp > 0:
                    b["body"].velocity = v * (280 / sp)
                elif sp > 650:
                    b["body"].velocity = v * (650 / sp)
        # escapes: out through the gap -> two new balls in the middle
        if self.format == "escape":
            for b in self.balls:
                if b["inside"] and (b["body"].position - (CX, CY)).length > RING_R + BALL_R + 10:
                    b["inside"] = False
                    self.escaped += 1
                    if self.done_at is None:
                        for _ in range(2):
                            self._spawns.append((CX + self.rng.uniform(-30, 30), CY + self.rng.uniform(-30, 30)))
                        self.events.append((self.t, "note", self.spawned + 2, 1.0))
            for b in [b for b in self.balls if not b["inside"] and b["body"].position.y > H + 80]:
                self.space.remove(b["body"], b["shape"])
                self.balls.remove(b)
        for pos in self._spawns:
            if self.done_at is None and self.spawned < self.cap:
                nb = self._add(pos)
                if len(self.effects) < 10:          # a flood of rings is just clutter
                    self.effects.append([pos[0], pos[1], BALL_R * 1.6, 1.0, nb["tier"]])
                self.events.append((self.t, "pop", min(nb["tier"], 6), 0.6))
        self._spawns.clear()
        self.spawned = sum(1 for x in self.balls if x["inside"])
        self._reveal()
        if self.done_at is None and self.spawned >= self.cap:
            self.done_at = self.t
            self.best = self.last
            self.events.append((self.t, "fanfare", 0, 1.0))
            # full: the ring stops and turns its gap to the top, so nothing
            # more falls out and the count holds
            self.ring.angular_velocity = 0
            self.ring.angle = -math.pi / 2
            self.space.reindex_shapes_for_body(self.ring)
        self.t += dt

    @property
    def finished(self) -> bool:
        return ((self.done_at is not None and self.t - self.done_at > END_HOLD)
                or self.t > SECONDS_CAP)


class CirclePainter(be.Painter):
    """The ring, the balls, a big count, plus the shared header/ending."""

    def frame(self, sim: CircleSim) -> Image.Image:
        im = self.bg.copy()
        d = ImageDraw.Draw(im, "RGBA")
        col = self.colours[-1]
        # the ring (a glow, then the line), with its gap where the body has turned it
        gap_deg = math.degrees(sim.gap)
        rot = math.degrees(sim.ring.angle)
        box = (CX - RING_R, CY - RING_R, CX + RING_R, CY + RING_R)
        for w, a in ((34, 40), (22, 70)):
            if sim.gap:
                d.arc(box, rot + gap_deg / 2, rot + 360 - gap_deg / 2, fill=col + (a,), width=w)
            else:
                d.ellipse(box, outline=col + (a,), width=w)
        if sim.gap:
            d.arc(box, rot + gap_deg / 2, rot + 360 - gap_deg / 2, fill=(255, 255, 255, 235), width=10)
        else:
            d.ellipse(box, outline=(255, 255, 255, 235), width=10)
        for b in sim.balls:
            x, y = b["body"].position
            sp = self.sprite(b["tier"], b["r"])
            im.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)
        for e in sim.effects:
            x, y, r, life, tier = e
            rr = r * (1.0 + (1 - life) * 0.9)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=self.colours[tier] + (int(230 * life),),
                      width=max(2, int(6 * life)))
            e[3] -= 0.08
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]
        self.header(im, d, sim)
        count = f"{sim.cap if sim.done_at is not None else sim.spawned:,} / {sim.cap}"
        self.ctext(d, CY + RING_R + 40, count, be._font(80), (255, 230, 140, 255))
        self.watermark(d, W - 60, H - 70)
        self.overlays(im, d, sim, item_y=CY - 120, text_y=CY + 60)
        return im
