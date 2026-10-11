"""Four more Ball Evolution kinds, after the music.bouncingball formats Dean
sent (Oct 2026). All use the video's picture pack and play a tune from
the bounces, like Merge (``music_track``), on the neon ring look.

* ``center`` -- Hold the center: every picture in the pack bounces in a
  ring around a black hole that slowly grows. A picture that falls in is
  out; the last one left wins. The seed is picked so the pack's final
  picture (the crowned country, the god form) is the one that wins.
* ``spikes`` -- Spikes will grow: one ball in an octagon. Every wall it
  hits grows a spike there; it changes into the next picture every few
  bounces; touching a spike pops it. Seeds where it reaches the final form.
* ``shrink`` -- The ball grows, the circle shrinks: every bounce makes the
  ball bigger and the rainbow ring smaller, until it's stuck.
* ``jaws`` -- 4 seconds to escape: a turning ring broken into arcs. Each
  ball has 4 seconds to get out through a gap or it pops, and every ball
  that escapes makes two more.

ball_evolution dispatches here (``_parts``), so picking seeds, rendering,
audio and publishing are shared.
"""
from __future__ import annotations

import colorsys
import math
import random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from clipper import ball_circles as bc
from clipper import ball_evolution as be
from clipper import ball_merge as bm

W, H, FPS = be.W, be.H, be.FPS
CX, CY = 540, 1060
END_HOLD = 4.0
SECONDS_CAP = 70.0
KINDS = ("center", "spikes", "shrink", "jaws")


# ---- shared ---------------------------------------------------------------

class ArenaSim:
    """What every Ball Evolution sim offers the shared code: events (sounds),
    hits (the tune), the ladder state, done_at, finished, plus ``ok``
    (False = this seed's run doesn't make the video it should)."""

    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.chain = be.recipe_chain(recipe)
        self.last = len(self.chain) - 1
        self.rng = random.Random(f"{recipe.get('format')}-{recipe['seed']}")
        self.balls: list = []
        self.events: list = []
        self.effects: list = []
        self.hits: list = []
        self.reveal_flash = [0.0] * len(self.chain)
        self.unlocked_at = {0: 0.0}
        self.banner = None
        self.best = 0
        self.spawned = 1
        self.t = 0.0
        self.done_at = None
        self.ok = True

    def reveal(self, tier: int) -> None:
        tier = min(self.last, tier)
        if tier > self.best:
            for k in range(self.best + 1, tier + 1):
                self.unlocked_at[k] = self.t
                self.reveal_flash[k] = 1.0
            self.best = tier
            self.banner = (f"NEW: {self.chain[tier][1]}!", self.t)
            self.events.append((self.t, "unlock", tier, 1.0))

    def finish(self, ok: bool = True) -> None:
        if self.done_at is None:
            self.done_at = self.t
            self.ok = ok
            if ok:
                self.reveal(self.last)
            self.events.append((self.t, "fanfare", 0, 1.0))

    def bounce(self, vol: float) -> None:
        self.hits.append((self.t, max(0.2, min(1.0, vol))))

    def music_track(self, n: int):
        return bm.MergeSim.music_track(self, n)

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP


def _pos(b: dict):
    body = b.get("body")
    return (body.position.x, body.position.y) if body is not None else (b["x"], b["y"])


def _rainbow_arcs(size: int, box: tuple, width: int, alpha: int, n: int = 180) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k in range(n):
        r, g, b = colorsys.hsv_to_rgb(k / n, 0.85, 1)
        d.arc(box, 360 * k / n, 360 * (k + 1) / n + 0.6, fill=(int(r * 255), int(g * 255), int(b * 255), alpha),
              width=width)
    return im


def rainbow_ring(R: float, pad: int = 50) -> Image.Image:
    """A glowing rainbow circle of radius R (an image 2(R+pad) square)."""
    size = int(2 * (R + pad))
    box = (pad, pad, pad + 2 * R, pad + 2 * R)
    out = _rainbow_arcs(size, box, 36, 170).filter(ImageFilter.GaussianBlur(14))
    out.alpha_composite(_rainbow_arcs(size, box, 14, 255))
    return out


def _glow_lines(lines: list, width: int = 11) -> Image.Image:
    """[(points, colour)] drawn as glowing lines on a full-frame layer."""
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for pts, c in lines:
        gd.line(pts, fill=c + (170,), width=width * 2 + 4, joint="curve")
    im = glow.filter(ImageFilter.GaussianBlur(12))
    d = ImageDraw.Draw(im)
    for pts, c in lines:
        d.line(pts, fill=c + (255,), width=width, joint="curve")
    return im


def _hue(i: int, n: int) -> tuple:
    r, g, b = colorsys.hsv_to_rgb(i / max(1, n), 0.85, 1)
    return (int(r * 255), int(g * 255), int(b * 255))


class ArenaPainter(be.Painter):
    """Dark backdrop (black / space / blurred), light trails, glowing
    picture balls, the shared header and ending."""

    TRAIL_FADE = 0.82

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
        self.trail_cols = [tuple(int(v * 0.7) for v in c) for c in self.colours]
        self.glow: dict = {}
        self.f_count = be._font(64)

    def sprite(self, tier: int, r: float) -> Image.Image:
        key = (tier, max(4, int(round(r))))
        if key not in self.glow:
            sp = super().sprite(tier, r)
            pad = 12
            im = Image.new("RGBA", (sp.width + 2 * pad, sp.height + 2 * pad), (0, 0, 0, 0))
            c, rr = im.width / 2, sp.width / 2 + 2
            ImageDraw.Draw(im).ellipse((c - rr, c - rr, c + rr, c + rr), fill=self.colours[tier] + (150,))
            im = im.filter(ImageFilter.GaussianBlur(7))
            im.alpha_composite(sp, (pad, pad))
            if len(self.glow) > 400:
                self.glow.clear()
            self.glow[key] = im
        return self.glow[key]

    def base(self, sim) -> Image.Image:
        self.trail = self.trail.point(self.fade)
        d = ImageDraw.Draw(self.trail)
        for b in sim.balls:
            x, y = _pos(b)
            r = b["r"] * 0.6
            col = self.trail_cols[b["tier"]]
            px, py = b.get("trail_at") or (x, y)
            if (px - x) ** 2 + (py - y) ** 2 > 4:
                d.line([(px, py), (x, y)], fill=col, width=max(2, int(2 * r)))
            b["trail_at"] = (x, y)
        return ImageChops.add(self.bg, self.trail)

    def balls(self, im: Image.Image, sim) -> None:
        for b in sim.balls:
            x, y = _pos(b)
            sp = self.sprite(b["tier"], b["r"])
            im.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)

    def effects(self, d, sim) -> None:
        for e in sim.effects:
            x, y, r, life, tier = e
            rr = r * (1.0 + (1 - life) * 1.1)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=self.colours[tier] + (int(230 * life),),
                      width=max(2, int(6 * life)))
            e[3] -= 0.07
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]

    def finish(self, im, d, sim, count: str, item_y: float = CY - 120, text_y: float = CY + 60) -> Image.Image:
        self.header(im, d, sim)
        self.ctext(d, 1580, count, self.f_count, (255, 230, 140, 255))
        self.watermark(d, W - 60, H - 70)
        self.overlays(im, d, sim, item_y=item_y, text_y=text_y)
        return im


# ---- Hold the center --------------------------------------------------------

C_R = 430              # the ring
C_BALL = 46            # the pictures are big here: they're the contestants
C_SPEED = 470
C_HOLE0, C_HOLE_GROW, C_HOLE_MAX = 20.0, 2.6, 230.0
C_IN = 0.8             # how far in (in ball radii) a picture must get to fall


class CenterSim(ArenaSim):
    def __init__(self, recipe: dict):
        super().__init__(recipe)
        import pymunk

        self.pm = pymunk
        sp = self.space = pymunk.Space()
        sp.iterations = 15
        n = 96
        pts = [(CX + C_R * math.cos(2 * math.pi * i / n), CY + C_R * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
        for p, q in zip(pts, pts[1:]):
            s = pymunk.Segment(sp.static_body, p, q, 6)
            s.elasticity, s.friction, s.collision_type = 1.0, 0.0, 3
            sp.add(s)
        sp.on_collision(1, 3, begin=self._on_wall)
        sp.on_collision(1, 1, begin=self._on_ball)
        k = len(self.chain)
        start = self.rng.uniform(0, 2 * math.pi)
        order = list(range(k))
        self.rng.shuffle(order)
        for j, tier in enumerate(order):
            a = start + 2 * math.pi * j / k
            body = pymunk.Body()
            body.position = (CX + 290 * math.cos(a), CY + 290 * math.sin(a))
            d = self.rng.uniform(0, 2 * math.pi)
            body.velocity = (C_SPEED * math.cos(d), C_SPEED * math.sin(d))
            shape = pymunk.Circle(body, C_BALL)
            shape.density, shape.elasticity, shape.friction, shape.collision_type = 1.0, 1.0, 0.0, 1
            sp.add(body, shape)
            b = {"body": body, "shape": shape, "tier": tier, "r": C_BALL}
            shape.ball = b
            self.balls.append(b)
        self.out: list = []            # tiers, in the order they fell in
        self.best = self.last          # every contestant is shown from the start
        self.unlocked_at = {0: 0.0}
        self.spawned = k
        self.hole = C_HOLE0

    def _on_wall(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        if b is not None:
            self.bounce(0.8)
            self.events.append((self.t, "tick", int((_pos(b)[0] - CX + C_R) / (2 * C_R) * 9.99), 0.6))

    def _on_ball(self, arbiter, space, data):
        self.bounce(0.6)
        self.events.append((self.t, "knock", "bumper", 0.4))

    def step(self):
        dt = 1 / FPS
        for _ in range(3):
            self.space.step(dt / 3)
        if self.done_at is None:
            self.hole = min(C_HOLE_MAX, C_HOLE0 + C_HOLE_GROW * self.t)
        for b in list(self.balls):
            v = b["body"].velocity.rotated(self.rng.gauss(0, 0.025))
            b["body"].velocity = v * (C_SPEED / (v.length or 1))
            x, y = _pos(b)
            if self.done_at is None and math.hypot(x - CX, y - CY) < self.hole - C_BALL * C_IN:
                self.space.remove(b["body"], b["shape"])
                self.balls.remove(b)
                self.out.append(b["tier"])
                self.unlocked_at[len(self.out)] = self.t
                self.effects.append([CX, CY, self.hole, 1.0, b["tier"]])
                self.events.append((self.t, "pop", 6, 1.0))
                self.banner = (f"OUT: {self.chain[b['tier']][1]}!", self.t)
        self.spawned = len(self.balls)
        if self.done_at is None and len(self.balls) == 1:
            self.finish(ok=self.balls[0]["tier"] == self.last)
        self.t += dt


class CenterPainter(ArenaPainter):
    def __init__(self, recipe: dict):
        super().__init__(recipe)
        self.ring = rainbow_ring(C_R)
        self.ring_pos = (CX - self.ring.width // 2, CY - self.ring.height // 2)

    def frame(self, sim: CenterSim) -> Image.Image:
        im = self.base(sim)
        im.paste(self.ring, self.ring_pos, self.ring)
        d = ImageDraw.Draw(im, "RGBA")
        h = sim.hole
        for k in range(6, 0, -1):                         # the hole: black, with a faint glow rim
            rr = h + k * 7
            d.ellipse((CX - rr, CY - rr, CX + rr, CY + rr), fill=(120, 60, 200, 12))
        d.ellipse((CX - h, CY - h, CX + h, CY + h), fill=(0, 0, 0, 255), outline=(170, 120, 255, 200), width=4)
        self.balls(im, sim)
        self.effects(d, sim)
        self.header(im, d, sim)
        self._crossed(im, d, sim)
        left = len(sim.balls)
        self.ctext(d, 1580, "WINNER" if sim.done_at is not None else f"{left} left", self.f_count,
                   (255, 230, 140, 255))
        self.watermark(d, W - 60, H - 70)
        self.overlays(im, d, sim, item_y=CY - 120, text_y=CY + 60)
        return im

    def _crossed(self, im, d, sim) -> None:
        """A red cross on each contestant in the ladder that fell in."""
        n, cell, ic_s, top = len(self.chain), self.cell, self.icon, 128
        x0 = (W - cell * n) / 2
        for tier in sim.out:
            cx = x0 + tier * cell + cell / 2
            half = cell / 2 - 6
            d.rounded_rectangle((cx - half, top - 6, cx + half, top + ic_s + 8), 16, fill=(0, 0, 0, 150))
            m = half * 0.7
            cy = top + ic_s / 2
            d.line([(cx - m, cy - m), (cx + m, cy + m)], fill=(255, 60, 60, 255), width=6)
            d.line([(cx - m, cy + m), (cx + m, cy - m)], fill=(255, 60, 60, 255), width=6)


# ---- one ball, own physics (spikes, shrink) -------------------------------------

class OneBallSim(ArenaSim):
    def __init__(self, recipe: dict, r: float, gravity: float, min_speed: float):
        super().__init__(recipe)
        self.gravity, self.min_speed = gravity, min_speed
        self.ball = {"x": CX + self.rng.uniform(-60, 60), "y": CY - 160, "vx": self.rng.choice((-1, 1)) * self.rng.uniform(250, 420),
                     "vy": 0.0, "tier": 0, "r": r}
        self.balls = [self.ball]
        self.bounces = 0

    def _kick(self, nx: float, ny: float) -> None:
        """Bounce off a wall with inward normal (nx, ny): reflect, keep it lively."""
        b = self.ball
        dot = b["vx"] * nx + b["vy"] * ny
        if dot >= 0:
            return
        b["vx"] -= 2 * dot * nx
        b["vy"] -= 2 * dot * ny
        a = self.rng.gauss(0, self.wobble)
        vx, vy = b["vx"], b["vy"]
        b["vx"], b["vy"] = vx * math.cos(a) - vy * math.sin(a), vx * math.sin(a) + vy * math.cos(a)
        sp = math.hypot(b["vx"], b["vy"])
        if sp < self.min_speed:
            b["vx"], b["vy"] = b["vx"] * self.min_speed / sp, b["vy"] * self.min_speed / sp
        self.bounces += 1
        self.bounce(min(1.0, sp / 1500))
        self.events.append((self.t, "tick", min(9, self.bounces % 10), 0.8))

    def move(self, dt: float) -> None:
        b = self.ball
        b["vy"] += self.gravity * dt
        b["x"] += b["vx"] * dt
        b["y"] += b["vy"] * dt


# ---- Spikes will grow -------------------------------------------------------------

S_R = 430              # the octagon's corner radius
S_BALL = 36
S_SPEED = 470
S_SPIKE_W, S_SPIKE_H = 50, 46
S_TIP = 10             # only a hit this close to the tip pops it...
S_HEAD_ON = 0.97       # ...and nearly head-on (cos of the angle to the spike's line)
S_BOUNCES = 28         # bounces to the final form, shared out over the pictures


def _octagon() -> list:
    a0 = math.pi / 8
    return [(CX + S_R * math.cos(a0 + k * math.pi / 4), CY + S_R * math.sin(a0 + k * math.pi / 4)) for k in range(8)]


class SpikesSim(ArenaSim):
    """One ball at a steady speed (no gravity, so its hits spread over every
    wall) in an octagon. Each wall it hits grows a solid spike there: the
    ball bounces off a spike's sides, and only a hit on the tip pops it.
    (With gravity it kept hitting the same spots, and with a spike's whole
    body deadly it popped after 2-5 spikes.)"""

    def __init__(self, recipe: dict):
        super().__init__(recipe)
        import pymunk

        self.pm = pymunk
        sp = self.space = pymunk.Space()
        sp.iterations = 20
        pts = _octagon()
        for p, q in zip(pts, pts[1:] + pts[:1]):
            s = pymunk.Segment(sp.static_body, p, q, 5)
            s.elasticity, s.friction, s.collision_type = 1.0, 0.0, 3
            sp.add(s)
        body = pymunk.Body()
        body.position = (CX + self.rng.uniform(-80, 80), CY + self.rng.uniform(-80, 80))
        a = self.rng.uniform(0, 2 * math.pi)
        body.velocity = (S_SPEED * math.cos(a), S_SPEED * math.sin(a))
        shape = pymunk.Circle(body, S_BALL)
        shape.density, shape.elasticity, shape.friction, shape.collision_type = 1.0, 1.0, 0.0, 1
        sp.add(body, shape)
        self.ball = {"body": body, "shape": shape, "tier": 0, "r": S_BALL}
        self.balls = [self.ball]
        self.spikes: list = []         # [base x, base y, nx, ny, born]
        self.bounces = 0
        self.per_tier = max(3, round(S_BOUNCES / max(1, self.last)))
        self._grow_at: list = []
        self._popped = False
        sp.on_collision(1, 3, post_solve=self._on_wall)
        sp.on_collision(1, 6, post_solve=self._on_spike)

    def _hit(self) -> None:
        self.bounces += 1
        self.bounce(0.8)
        self.events.append((self.t, "tick", self.bounces % 10, 0.8))

    def _on_wall(self, arbiter, space, data):
        if not arbiter.is_first_contact:
            return
        p = arbiter.contact_point_set.points[0].point_a
        n = arbiter.contact_point_set.normal            # from the ball to the wall
        self._grow_at.append((p.x, p.y, -n.x, -n.y))
        self._hit()

    def _on_spike(self, arbiter, space, data):
        if not arbiter.is_first_contact:
            return
        cps = arbiter.contact_point_set
        p = cps.points[0].point_a
        spike = arbiter.shapes[1]
        tip, (ax, ay) = spike.tip, spike.axis
        # the tip is the first thing a ball meets from most directions, so
        # only a near head-on hit pops it; the rest glance off
        head_on = -(cps.normal.x * ax + cps.normal.y * ay)
        if (p.x - tip[0]) ** 2 + (p.y - tip[1]) ** 2 < S_TIP ** 2 and head_on > S_HEAD_ON:
            self._popped = True
        else:
            self._hit()

    def step(self):
        dt = 1 / FPS
        if self.done_at is None:
            for _ in range(4):
                self.space.step(dt / 4)
                b = self.ball["body"]
                v = b.velocity.rotated(self.rng.gauss(0, 0.01))
                b.velocity = v * (S_SPEED / (v.length or 1))
                # a spike grows once the ball has moved off: grown at once it
                # pierced the ball that made it
                bx, by = b.position
                keep = []
                for x, y, nx, ny in self._grow_at:
                    if (bx - x) ** 2 + (by - y) ** 2 > (S_BALL + S_SPIKE_H + 6) ** 2:
                        self._grow(x, y, nx, ny)
                    else:
                        keep.append((x, y, nx, ny))
                self._grow_at[:] = keep
                if self._popped:
                    x, y = _pos(self.ball)
                    self.effects.append([x, y, S_BALL * 1.6, 1.0, self.ball["tier"]])
                    self.events.append((self.t, "pop", 6, 1.0))
                    self.banner = ("POPPED!", self.t)
                    self.space.remove(b, self.ball["shape"])
                    self.balls = []
                    self.finish(ok=False)
                    break
            if self.done_at is None:
                tier = min(self.last, self.bounces // self.per_tier)
                self.ball["tier"] = tier
                self.reveal(tier)
                if tier >= self.last:
                    self.finish(ok=True)
        self.spawned = len(self.spikes)
        self.t += dt

    def _grow(self, x: float, y: float, nx: float, ny: float) -> None:
        if any((x - s[0]) ** 2 + (y - s[1]) ** 2 < (S_SPIKE_W * 1.05) ** 2 for s in self.spikes):
            return
        tx, ty = -ny, nx
        hw = S_SPIKE_W / 2
        tip = (x + nx * S_SPIKE_H, y + ny * S_SPIKE_H)
        tri = [(x + tx * hw, y + ty * hw), (x - tx * hw, y - ty * hw), tip]
        s = self.pm.Poly(self.space.static_body, tri)
        s.elasticity, s.friction, s.collision_type = 1.0, 0.0, 6
        s.tip, s.axis = tip, (nx, ny)
        self.space.add(s)
        self.spikes.append([x, y, nx, ny, self.t])


class SpikesPainter(ArenaPainter):
    def __init__(self, recipe: dict):
        super().__init__(recipe)
        pts = _octagon()
        n = len(pts)
        self.static = _glow_lines([([p, q], _hue(i, n)) for i, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1]))])

    def frame(self, sim: SpikesSim) -> Image.Image:
        im = self.base(sim)
        im.paste(self.static, (0, 0), self.static)
        d = ImageDraw.Draw(im, "RGBA")
        for x, y, nx, ny, born in sim.spikes:
            grow = min(1.0, (sim.t - born) / 0.25)
            tx, ty = -ny, nx                             # along the wall
            hw, hh = S_SPIKE_W / 2, S_SPIKE_H * grow
            tri = [(x + tx * hw, y + ty * hw), (x - tx * hw, y - ty * hw), (x + nx * hh, y + ny * hh)]
            d.polygon(tri, fill=(255, 70, 90, 255), outline=(255, 210, 220, 255))
        self.balls(im, sim)
        self.effects(d, sim)
        n = len(sim.spikes)
        return self.finish(im, d, sim, f"{n} spike{'s' if n != 1 else ''}")


# ---- The ball grows, the circle shrinks ----------------------------------------------

G_R0, G_RING_STEP = 470.0, 4.0
G_BALL0, G_BALL_STEP = 34.0, 2.3
G_GRAVITY, G_MIN_SPEED, G_WOBBLE = 800.0, 760.0, 0.06


class ShrinkSim(OneBallSim):
    def __init__(self, recipe: dict):
        super().__init__(recipe, G_BALL0, G_GRAVITY, G_MIN_SPEED)
        self.wobble = G_WOBBLE
        self.ring = G_R0
        # where ball and ring meet: every tier gets an equal share of the way
        self.meet = (G_R0 - G_BALL0) / (G_RING_STEP + G_BALL_STEP)

    def step(self):
        dt = 1 / FPS
        b = self.ball
        if self.done_at is None:
            for _ in range(4):
                self.move(dt / 4)
                dx, dy = b["x"] - CX, b["y"] - CY
                dist = math.hypot(dx, dy) or 1e-6
                if dist + b["r"] > self.ring:
                    nx, ny = -dx / dist, -dy / dist
                    before = self.bounces
                    self._kick(nx, ny)
                    if self.bounces > before:
                        self.ring -= G_RING_STEP
                        b["r"] += G_BALL_STEP
                        self.effects.append([b["x"] - nx * b["r"], b["y"] - ny * b["r"], 26, 1.0, b["tier"]])
                    room = max(0.0, self.ring - b["r"])
                    # keep it inside the (smaller) ring
                    k = room / dist if dist > room else 1.0
                    b["x"], b["y"] = CX + dx * k, CY + dy * k
                if b["r"] >= self.ring - 8:
                    b["x"], b["y"], b["vx"], b["vy"] = CX, CY, 0.0, 0.0
                    b["r"] = self.ring - 8
                    self.finish(ok=True)
                    break
            b["tier"] = min(self.last, int(self.bounces / self.meet * (self.last + 0.999)))
            self.reveal(b["tier"])
        self.spawned = self.bounces
        self.t += dt


class ShrinkPainter(ArenaPainter):
    def __init__(self, recipe: dict):
        super().__init__(recipe)
        self.ring_full = rainbow_ring(G_R0)

    def frame(self, sim: ShrinkSim) -> Image.Image:
        im = self.base(sim)
        s = sim.ring / G_R0
        ring = self.ring_full.resize((max(8, int(self.ring_full.width * s)),) * 2, Image.BILINEAR)
        im.paste(ring, (int(CX - ring.width / 2), int(CY - ring.height / 2)), ring)
        d = ImageDraw.Draw(im, "RGBA")
        self.balls(im, sim)
        self.effects(d, sim)
        n = sim.bounces
        return self.finish(im, d, sim, f"{n} bounce{'s' if n != 1 else ''}")


# ---- 4 seconds to escape --------------------------------------------------------------

J_R = 430
J_GAPS = 3             # the ring is broken into arcs with this many gaps
J_GAP = 0.62           # radians, each
J_SPIN = 1.6
J_BALL = 24
J_LIFE = 4.0
J_GRAVITY = 760
J_ELASTIC = 0.8
J_MAX = 40
J_TARGET = 48          # escapes for the final form


def _gap_centres() -> list:
    return [2 * math.pi * k / J_GAPS for k in range(J_GAPS)]


class JawsSim(ArenaSim):
    """A turning ring broken into arcs; each ball has J_LIFE seconds to get
    out through a gap before it pops, and each that gets out makes two.
    (Rotating teeth were tried first: they swept through every ball resting
    on the bottom, so almost none got out.)"""

    def __init__(self, recipe: dict):
        super().__init__(recipe)
        import pymunk

        self.pm = pymunk
        sp = self.space = pymunk.Space()
        sp.gravity = (0, J_GRAVITY)
        sp.iterations = 15
        self.ringb = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.ringb.position = (CX, CY)
        self.spin = J_SPIN * self.rng.choice((-1, 1))
        self.ringb.angular_velocity = self.spin
        sp.add(self.ringb)
        arc = 2 * math.pi / J_GAPS - J_GAP
        for g in _gap_centres():
            a0 = g + J_GAP / 2
            n = 30
            pts = [(J_R * math.cos(a0 + arc * i / n), J_R * math.sin(a0 + arc * i / n)) for i in range(n + 1)]
            for p, q in zip(pts, pts[1:]):
                s = pymunk.Segment(self.ringb, p, q, 6)
                s.elasticity, s.friction, s.collision_type = 1.0, 0.1, 3
                sp.add(s)
        sp.on_collision(1, 3, begin=self._on_wall)
        self.escaped = 0
        self.popped = 0
        self.thresholds = [round(J_TARGET ** (k / self.last)) for k in range(self.last + 1)]
        self._spawns: list = []
        self._add((CX, CY))

    def _add(self, pos):
        pm = self.pm
        body = pm.Body()
        body.position = pos
        a = self.rng.uniform(0, 2 * math.pi)
        sp = self.rng.uniform(250, 420)
        body.velocity = (sp * math.cos(a), sp * math.sin(a) - 200)
        shape = pm.Circle(body, J_BALL)
        shape.density, shape.elasticity, shape.friction, shape.collision_type = 1.0, J_ELASTIC, 0.1, 1
        shape.filter = pm.ShapeFilter(group=1)               # balls pass through each other
        self.space.add(body, shape)
        b = {"body": body, "shape": shape, "tier": self.best, "r": J_BALL, "born": self.t, "inside": True}
        shape.ball = b
        self.balls.append(b)
        return b

    def _on_wall(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        if b is not None:
            self.bounce(0.7)
            self.events.append((self.t, "tick", int(self.rng.random() * 9.99), 0.5))

    def step(self):
        dt = 1 / FPS
        for _ in range(3):
            self.space.step(dt / 3)
        for b in list(self.balls):
            x, y = _pos(b)
            if b["inside"]:
                if math.hypot(x - CX, y - CY) > J_R + J_BALL + 8:
                    b["inside"] = False
                    if self.done_at is None:
                        self.escaped += 1
                        for _ in range(2):
                            self._spawns.append((CX + self.rng.uniform(-30, 30), CY + self.rng.uniform(-40, 10)))
                        self.events.append((self.t, "note", self.escaped, 1.0))
                elif self.done_at is None and self.t - b["born"] > J_LIFE:
                    self.space.remove(b["body"], b["shape"])
                    self.balls.remove(b)
                    self.popped += 1
                    self.effects.append([x, y, J_BALL * 1.6, 1.0, b["tier"]])
                    self.events.append((self.t, "pop", 4, 0.8))
            elif y > H + 80:
                self.space.remove(b["body"], b["shape"])
                self.balls.remove(b)
        inside = sum(1 for b in self.balls if b["inside"])
        for pos in self._spawns:
            if inside < J_MAX:
                nb = self._add(pos)
                inside += 1
                if len(self.effects) < 10:
                    self.effects.append([pos[0], pos[1], J_BALL * 1.5, 1.0, nb["tier"]])
        self._spawns.clear()
        if self.done_at is None and inside == 0:
            self._add((CX, CY))                              # never run dry: one more try
        tier = max(k for k, th in enumerate(self.thresholds) if self.escaped >= th or k == 0)
        self.reveal(tier)
        self.spawned = self.escaped
        if self.done_at is None and self.escaped >= J_TARGET:
            self.finish(ok=True)
            self.ringb.angular_velocity = 0
        self.t += dt


class JawsPainter(ArenaPainter):
    def __init__(self, recipe: dict):
        super().__init__(recipe)
        self.ring = rainbow_ring(J_R)
        self.ring_alpha = self.ring.getchannel("A")

    def frame(self, sim: JawsSim) -> Image.Image:
        im = self.base(sim)
        rot, half = math.degrees(sim.ringb.angle), math.degrees(J_GAP) / 2
        cut = Image.new("L", self.ring.size, 255)
        cd = ImageDraw.Draw(cut)
        for g in _gap_centres():
            c = rot + math.degrees(g)
            cd.pieslice((-200, -200, self.ring.width + 200, self.ring.height + 200), c - half, c + half, fill=0)
        im.paste(self.ring, (CX - self.ring.width // 2, CY - self.ring.height // 2),
                 ImageChops.multiply(self.ring_alpha, cut))
        d = ImageDraw.Draw(im, "RGBA")
        self.balls(im, sim)
        # each ball's clock: an arc that runs down over its 4 seconds
        for b in sim.balls:
            if not b["inside"]:
                continue
            x, y = _pos(b)
            left = max(0.0, 1 - (sim.t - b["born"]) / J_LIFE)
            r = J_BALL + 9
            col = (120, 255, 140, 230) if left > 0.5 else (255, 210, 80, 230) if left > 0.25 else (255, 70, 70, 240)
            d.arc((x - r, y - r, x + r, y + r), -90, -90 + 360 * left, fill=col, width=5)
        self.effects(d, sim)
        return self.finish(im, d, sim, f"{sim.escaped} escaped · {sim.popped} popped")


PARTS = {"center": (CenterSim, CenterPainter), "spikes": (SpikesSim, SpikesPainter),
         "shrink": (ShrinkSim, ShrinkPainter), "jaws": (JawsSim, JawsPainter)}
