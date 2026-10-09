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

import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

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


def _space(seed: int) -> Image.Image:
    """Stars over soft purple/blue/red clouds, kept dim."""
    rng = random.Random(f"space-{seed}")
    small = Image.new("RGB", (W // 4, H // 4))
    d = ImageDraw.Draw(small)
    for c in [(120, 30, 160), (30, 60, 170), (170, 30, 90)]:
        for _ in range(3):
            x, y, r = rng.uniform(25, 245), rng.uniform(75, 425), rng.uniform(62, 112)
            d.ellipse((x - r, y - r, x + r, y + r), fill=c)
    clouds = small.filter(ImageFilter.GaussianBlur(40)).resize((W, H), Image.BILINEAR)
    im = ImageChops.multiply(clouds, Image.new("RGB", (W, H), (90, 90, 90)))
    d = ImageDraw.Draw(im)
    for _ in range(300):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.choice([1, 1, 1.5, 2, 2.5])
        v = int(rng.randint(120, 255) * 0.6)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, min(255, v + 12)))
    return im


SYNTH_HORIZON = 1640       # under the count, so the grid never runs behind it


def _synth_sky() -> Image.Image:
    """The synthwave sky and the grid's lines to the vanishing point; the
    horizontal lines move, so frame() draws them."""
    g = np.linspace(0, 1, H)[:, None, None]
    rows = (np.array((4, 0, 14)) * (1 - g) + np.array((30, 0, 45)) * g).astype(np.uint8)
    im = Image.fromarray(np.repeat(rows, W, axis=1), "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    for k in range(-8, 9):
        d.line([(540, SYNTH_HORIZON), (540 + k * 220, H)], fill=(255, 60, 200, 80), width=2)
    d.rectangle((0, SYNTH_HORIZON - 3, W, SYNTH_HORIZON), fill=(255, 120, 230, 140))
    return im


def _blurred(emoji: list, seed: int) -> Image.Image:
    """The theme's own items, big, blurred and dim."""
    rng = random.Random(f"bokeh-{seed}")
    layer = Image.new("RGBA", (W // 2, H // 2), (0, 0, 0, 0))
    for _ in range(14):
        e, sz = rng.choice(emoji), rng.randint(110, 210)
        layer.alpha_composite(e.resize((sz, sz)), (rng.randint(-50, 450), rng.randint(-50, 900)))
    layer = layer.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR)
    layer.putalpha(layer.getchannel("A").point(lambda v: int(v * 0.25)))
    im = Image.new("RGBA", (W, H), (6, 4, 10, 255))
    im.alpha_composite(layer)
    return im.convert("RGB")


class CirclePainter(be.Painter):
    """The ring, the balls, a big count, plus the shared header/ending.

    With one of be.RING_BACKDROPS (every ring recipe since Dean picked
    them) the balls leave light trails, the ring is a glowing rainbow and
    the balls glow; older recipes draw exactly as before."""

    TRAIL_FADE = 0.82          # per frame

    def __init__(self, recipe: dict):
        super().__init__(recipe)
        self.neon = recipe.get("backdrop") in be.RING_BACKDROPS
        if not self.neon:
            return
        kind, seed = recipe["backdrop"], recipe.get("seed") or 0
        if kind == "space":
            self.bg = _space(seed)
        elif kind == "synthwave":
            self.bg = _synth_sky()
        elif kind == "blurred":
            self.bg = _blurred(self.emoji, seed)
        else:
            self.bg = Image.new("RGB", (W, H), (0, 0, 0))
        # trails only live from just above the ring down (balls that get
        # out fall down the screen), which keeps the per-frame work small
        self.trail_top = CY - RING_R - 40
        self.trail = Image.new("RGB", (W, H - self.trail_top))
        self.fade = [int(v * self.TRAIL_FADE) for v in range(256)] * 3
        self.trail_cols = [tuple(int(v * 0.75) for v in c) for c in self.colours]
        self.ring_img = self._rainbow()
        self.ring_alpha = self.ring_img.getchannel("A")
        self.glow_sprites: dict = {}

    # ---- the neon look
    RING_PAD = 50

    def _rainbow(self) -> Image.Image:
        """The whole ring as a rainbow line with a soft glow, drawn once;
        frame() cuts the gap out where the ring has turned it."""
        R, pad = RING_R, self.RING_PAD
        size = 2 * (R + pad)
        box = (pad, pad, pad + 2 * R, pad + 2 * R)
        n = 180

        def arcs(width, alpha):
            im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            for k in range(n):
                r, g, b = colorsys.hsv_to_rgb(k / n, 0.85, 1)
                d.arc(box, 360 * k / n, 360 * (k + 1) / n + 0.6,
                      fill=(int(r * 255), int(g * 255), int(b * 255), alpha), width=width)
            return im

        out = arcs(36, 170).filter(ImageFilter.GaussianBlur(14))
        out.alpha_composite(arcs(14, 255))
        return out

    def sprite(self, tier: int, r: float) -> Image.Image:
        if not self.neon:
            return super().sprite(tier, r)
        key = (tier, max(4, int(round(r))))
        if key not in self.glow_sprites:
            sp = super().sprite(tier, r)
            pad = 12
            im = Image.new("RGBA", (sp.width + 2 * pad, sp.height + 2 * pad), (0, 0, 0, 0))
            c, rr = im.width / 2, sp.width / 2 + 2
            ImageDraw.Draw(im).ellipse((c - rr, c - rr, c + rr, c + rr), fill=self.colours[tier] + (150,))
            im = im.filter(ImageFilter.GaussianBlur(7))
            im.alpha_composite(sp, (pad, pad))
            self.glow_sprites[key] = im
        return self.glow_sprites[key]

    def _neon_base(self, sim: CircleSim) -> Image.Image:
        """Background (+ the moving synthwave lines) with the trails added."""
        im = self.bg.copy()
        if self.recipe["backdrop"] == "synthwave":
            d = ImageDraw.Draw(im, "RGBA")
            phase = (sim.t * 0.6) % 1
            for k in range(11):
                y = SYNTH_HORIZON + (H - SYNTH_HORIZON) * ((k + phase) / 11) ** 2
                d.line([(0, y), (W, y)], fill=(255, 60, 200, 90), width=2)
        self.trail = self.trail.point(self.fade)
        d = ImageDraw.Draw(self.trail)
        top = self.trail_top
        for b in sim.balls:
            x, y = b["body"].position
            y -= top
            r = b["r"] * 0.7
            col = self.trail_cols[b["tier"]]
            # a line from where it was last frame, so fast balls leave a
            # streak instead of a row of dots
            px, py = b.get("trail_at") or (x, y)
            if (px - x) ** 2 + (py - y) ** 2 > 4:
                d.line([(px, py), (x, y)], fill=col, width=int(2 * r))
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)
            b["trail_at"] = (x, y)
        box = (0, top, W, H)
        im.paste(ImageChops.add(im.crop(box), self.trail), box)
        return im

    def _neon_ring(self, im: Image.Image, sim: CircleSim):
        """The rainbow ring, with the gap cut out where the ring has turned."""
        ring = self.ring_img
        pos = (CX - ring.width // 2, CY - ring.height // 2)
        if not sim.gap:
            im.paste(ring, pos, self.ring_alpha)
            return
        rot, half = math.degrees(sim.ring.angle), math.degrees(sim.gap) / 2
        cut = Image.new("L", ring.size, 255)
        ImageDraw.Draw(cut).pieslice((-200, -200, ring.width + 200, ring.height + 200),
                                     rot - half, rot + half, fill=0)
        im.paste(ring, pos, ImageChops.multiply(self.ring_alpha, cut))

    def frame(self, sim: CircleSim) -> Image.Image:
        if self.neon:
            im = self._neon_base(sim)
            self._neon_ring(im, sim)
            d = ImageDraw.Draw(im, "RGBA")
        else:
            im = self.bg.copy()
            d = ImageDraw.Draw(im, "RGBA")
            self._plain_ring(d, sim)
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

    def _plain_ring(self, d, sim: CircleSim):
        """The ring before the neon look (a glow, then the line), with its
        gap where the body has turned it."""
        col = self.colours[-1]
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
