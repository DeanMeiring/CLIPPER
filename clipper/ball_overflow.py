"""Will it overflow? Every hit splits a ball (Nov 2026).

The idea of a reel with 1.5M likes ("every hit splits a ball"), drawn our
own way: a big round bowl with a small opening at the top, a V-shaped floor
and one long paddle pinned at the bottom that swings like a windscreen
wiper (smoothly, or in the ``flick`` variant: wait at each side, sweep fast). It starts
with ONE ball. Every time a ball hits the paddle it splits in two (the new
one a little off the old one's path), so the count grows fast, the bowl
fills, and when the pile reaches the rim the balls pour out over the top:
OVERFLOW.

Sim + painter for ``format: "overflow"``, a ``NO_PACK`` kind
(``ball_evolution._parts`` dispatches here). Physics is pymunk with a
spatial hash and 9-10 px balls. Every new contact with the paddle (or the
paddle sweeping a ball along) is a hit; early on every hit splits, later a
throttle (``GROWTH * live^GROWTH_P`` splits/s) paces it so the bowl fills
steadily over ~22-32 s instead of ~8 s. A speed limit (``VMAX``) keeps a
squeezed pile from squirting out as a wave, so it only spills once it's
full; the live count is capped at ~1.3x what the bowl holds (~1.4-2k balls).
Balls are a numpy splat of pre-made sprites; the paddle is drawn over them.

Sound is ours, numpy only (``music_track``): a micro-grain crunch per split
(the design Dean approved for the satisfying sounds), rate-limited and
getting denser and softer as the count grows, over a soft "sand pour" bed
that grows with the pile, and a low swell + pour when it overflows.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from clipper.ball_evolution import FPS, H, SR, W, _font

END_HOLD = 1.8                  # the pour goes on this long after OVERFLOW (CLAUDE.md: 1.8 s everywhere)
SECONDS_CAP = 58.0
CX, CY, R = 540, 1010, 440      # the bowl
RIM_T = 8                       # the rim's physics half-thickness (segment radius)
GAP_HALF = math.radians(19)     # the opening at the top (each side of 12 o'clock)
RIM_A = math.radians(30)        # the floor plates meet the rim at 4 and 8 o'clock (30 deg under the middle)
HUB = (CX, CY + R - 44)         # where the plates meet: the paddle's pivot
HUB_R = 40                      # the half-disc over the pivot
# the plates' slope (from the hub up to the rim)
PLATE_A = math.atan2(HUB[1] - (CY + R * math.sin(RIM_A)), R * math.cos(RIM_A))
GRAVITY = 1300.0
SUBSTEPS = 2
PUSH_V = 100.0                  # a ball touching the paddle splits on first contact, or again while the paddle sweeps it along at this speed
GROWTH, GROWTH_P = 1.9, 0.6       # at most GROWTH * live^GROWTH_P splits a second (the pace)
COOLDOWN = 0.12                 # a ball splits at most once per this many s
BALL_E = 0.3
VMAX = 720.0                    # px/s: no ball moves faster (see _sync)
SPILL_N = 14                    # this many out over the top (with the bowl full) = OVERFLOW
FULL_SHARE = 0.86               # ... and at least this share of what the bowl holds inside
EARLY_SHARE, EARLY_MAX = 0.45, 4  # ok: at most 4 balls out before the bowl is 45% full
PACKING = 0.80                  # 2D random packing of equal discs
MOTIONS = ("swing", "swing", "flick")
HOOKS = ["Will It Overflow?", "Will the bowl overflow?", "Can the bowl hold them all?",
         "How fast does it overflow?", "One ball. Will it overflow?", "Will it spill over the top?",
         "Every hit splits a ball", "How many until it overflows?"]
SUBS = ["every hit splits a ball", "every hit makes two"]
ENDINGS = ["OVERFLOW!", "IT OVERFLOWED!", "SPILLING OVER!"]
PALETTES = [   # pastel balls: coral, lilac, mint, gold, sky -- and three more sets of our own
    [(255, 138, 128), (190, 160, 255), (120, 230, 190), (255, 214, 102), (120, 196, 255)],
    [(255, 160, 190), (255, 196, 140), (170, 230, 140), (140, 210, 240), (215, 170, 250)],
    [(250, 128, 114), (255, 190, 92), (130, 220, 210), (164, 180, 255), (240, 150, 220)],
    [(255, 230, 140), (150, 236, 200), (130, 190, 255), (255, 150, 150), (200, 170, 255)],
]
BACKS = [((14, 16, 26), (24, 28, 44)), ((20, 14, 28), (32, 24, 46)), ((10, 20, 24), (20, 34, 40))]


def overflow_recipe(recipe: dict) -> dict:
    rng = random.Random(f"overflow-recipe-{recipe['seed']}")
    hook = rng.choice(HOOKS)
    sub = SUBS[0] if hook != "Every hit splits a ball" else "will it overflow?"
    recipe.update(hook=hook, sub=sub, palette=rng.randrange(len(PALETTES)), back=rng.randrange(len(BACKS)),
                  motion=rng.choice(MOTIONS), speed=round(rng.uniform(0.9, 1.12), 3),
                  ball_r=rng.choice((9.0, 9.5, 10.0)), spin=rng.choice((1, -1)),
                  ending=rng.choice(ENDINGS), bed="off")
    return recipe


def floor_poly() -> list:
    """The floor under the V: hub, 4 o'clock, along the rim, 8 o'clock."""
    a4, a8 = RIM_A, math.pi - RIM_A
    arc = [(CX + R * math.cos(a), CY + R * math.sin(a)) for a in np.linspace(a4, a8, 60)]
    return [HUB] + arc


def capacity(r: float) -> int:
    """About how many balls of radius r fill the bowl (inside the rim, above the floor)."""
    Ri = R - RIM_T
    pts = floor_poly()
    floor = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2
    return int((math.pi * Ri * Ri - floor) * PACKING / (math.pi * r * r))


PADDLE_L, PADDLE_H0, PADDLE_H1 = 405.0, 10.0, 46.0
PERIOD = 3.0                    # s for the paddle's full there-and-back
PADDLE_E = 0.45                 # paddle bounciness (higher flings balls out of the top early)
# swing as far as the plates: the tip's edge just reaches them (no corner where balls could rest unswept)
SWING_A = math.pi / 2 - PLATE_A - math.atan2(PADDLE_H1, PADDLE_L) + math.radians(1.5)


def paddle_geom(recipe: dict):
    """(pivot, length, half-width at the pivot, half-width at the tip)."""
    return HUB, PADDLE_L, PADDLE_H0, PADDLE_H1


def _sweep(x: float, k: float = 0.35):
    """Position and speed along one sweep, x in 0..1: mostly even speed, eased
    a little at the ends (it still reaches the plates at speed, so balls lying
    there get hit)."""
    return (1 - k) * x + k * x * x * (3 - 2 * x), (1 - k) + k * 6 * x * (1 - x)


def paddle_angle(recipe: dict, t: float):
    """The paddle's angle (radians, 0 = pointing straight up, + = clockwise) and
    its angular speed. ``swing``: a windscreen wiper; ``flick``: it waits a
    moment at each side, then sweeps across faster."""
    sp = float(recipe.get("speed") or 1.0)
    A = SWING_A * int(recipe.get("spin") or 1)
    T = PERIOD / sp                       # a full there-and-back
    hold = 0.16 if recipe.get("motion") == "flick" else 0.0
    half = 0.5 - hold
    u = (t / T) % 1.0
    if u < hold:
        return -A, 0.0
    if u < 0.5:
        s, ds = _sweep((u - hold) / half)
        return -A + 2 * A * s, 2 * A * ds / (half * T)
    if u < 0.5 + hold:
        return A, 0.0
    s, ds = _sweep((u - 0.5 - hold) / half)
    return A - 2 * A * s, -2 * A * ds / (half * T)


def paddle_poly(recipe: dict, ang: float) -> list:
    (px, py), L, h0, h1 = paddle_geom(recipe)
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for lx, ly in ((-h0, 0), (h0, 0), (h1, -L), (-h1, -L)):
        # body frame: y up the paddle is -y on screen; rotate clockwise by ang
        out.append((px + lx * c - ly * s, py + lx * s + ly * c))
    return out


class OverflowSim:
    def __init__(self, recipe: dict):
        import pymunk

        self.pm = pymunk
        self.recipe = recipe
        self.rng = random.Random(f"overflow-{recipe['seed']}")
        self.r = float(recipe.get("ball_r") or 9.5)
        self.capacity = capacity(self.r)
        self.cap = int(self.capacity * 1.3)
        self.ncol = len(PALETTES[int(recipe.get("palette") or 0) % len(PALETTES)])
        self.t = 0.0
        self.chain = [(None, "1 ball"), (None, "10 balls"), (None, "100 balls"), (None, "1,000 balls"),
                      (None, "overflow")]
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.spawned = 1
        self.done_at = None
        self.max_gap = 0.0
        self.ok = False
        self.events: list = []          # the shared mixer's ball sounds: none (ours are in music_track)
        self.hits: list = []            # (t, vol): one per split
        self.splits: list = []          # (t, live count)
        self.spilled = 0
        self.early_spills = 0
        self.inside = 1
        self._queue: list = []
        self._tokens = 3.0

        sp = self.space = pymunk.Space()
        sp.gravity = (0, GRAVITY)
        sp.iterations = 10
        sp.use_spatial_hash(2 * self.r + 2, 12000)
        n = 120
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            mid = (a0 + a1) / 2
            if abs(_wrap(mid + math.pi / 2)) < GAP_HALF:      # the opening at the top
                continue
            s = pymunk.Segment(sp.static_body, (CX + R * math.cos(a0), CY + R * math.sin(a0)),
                               (CX + R * math.cos(a1), CY + R * math.sin(a1)), RIM_T)
            s.elasticity, s.friction, s.collision_type = 0.3, 0.5, 3
            sp.add(s)
        for a in (math.pi - RIM_A, RIM_A):                   # the V floor: 8 and 4 o'clock to the hub
            s = pymunk.Segment(sp.static_body, HUB, (CX + R * math.cos(a), CY + R * math.sin(a)), RIM_T)
            s.elasticity, s.friction, s.collision_type = 0.3, 0.5, 3
            sp.add(s)
        hub = pymunk.Circle(sp.static_body, HUB_R, HUB)
        hub.elasticity, hub.friction = 0.3, 0.5
        sp.add(hub)

        self.paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        (px, py), L, h0, h1 = paddle_geom(recipe)
        self.paddle.position = (px, py)
        # pymunk's angle turns +x toward +y (clockwise on screen), like paddle_poly
        shape = pymunk.Poly(self.paddle, [(-h0, 0), (h0, 0), (h1, -L), (-h1, -L)], radius=2)
        shape.elasticity, shape.friction, shape.collision_type = PADDLE_E, 0.3, 2
        sp.add(self.paddle, shape)
        self._set_paddle(0.0)
        self._filter = pymunk.ShapeFilter()
        sp.on_collision(1, 2, pre_solve=self._on_paddle)

        self.bodies: list = []
        self.cols: list = []
        self.out: list = []               # True once a ball has left the bowl
        self.last_split: list = []
        self.xy = np.zeros((0, 2), np.float32)
        self.col = np.zeros(0, np.int16)
        self._add(CX + self.rng.uniform(-60, 60), CY - 250, (self.rng.uniform(-40, 40), 0), self.rng.randrange(self.ncol))
        self._sync()

    # ---- physics ---------------------------------------------------------
    def _set_paddle(self, t: float):
        ang, w = paddle_angle(self.recipe, t)
        self.paddle.angle = ang
        self.paddle.angular_velocity = w
        self.space.reindex_shapes_for_body(self.paddle)

    def _add(self, x, y, v, c):
        pm = self.pm
        b = pm.Body(1.0, pm.moment_for_circle(1.0, 0, self.r))
        b.position = (x, y)
        b.velocity = v
        s = pm.Circle(b, self.r)
        s.elasticity, s.friction, s.collision_type = BALL_E, 0.3, 1
        s.idx = len(self.bodies)
        self.space.add(b, s)
        self.bodies.append(b)
        self.cols.append(c)
        self.out.append(False)
        self.last_split.append(-9.0)
        return s

    def _on_paddle(self, arbiter, space, data):
        if self.done_at is not None and self.t - self.done_at > END_HOLD - 0.4:
            return
        s = arbiter.shapes[0]
        i = getattr(s, "idx", None)
        if i is None or self.t - self.last_split[i] < COOLDOWN:
            return
        b = s.body
        p = arbiter.contact_point_set.points[0].point_a if arbiter.contact_point_set.points else b.position
        vp = self.paddle.velocity_at_world_point(p)
        if not arbiter.is_first_contact and math.hypot(vp.x, vp.y) < PUSH_V:
            # a hit: a new contact, or the paddle sweeping it along
            return
        self.last_split[i] = self.t
        self._queue.append(i)

    def _splits(self, dt: float):
        live = sum(1 for o in self.out if not o) + sum(1 for o in self.out if o) // 3
        # the pace: at most GROWTH * live^GROWTH_P splits a second (unthrottled
        # the bowl is full in ~8 s); sublinear, so the bowl fills steadily
        # instead of sitting empty for half the video; early on every hit splits
        rate = max(3.0, GROWTH * live ** GROWTH_P)
        self._tokens = min(self._tokens + dt * rate, max(3.0, 0.15 * rate))
        # after OVERFLOW only enough to keep the pour going (not a fountain)
        for i in self._queue:
            if live >= self.cap or self._tokens < 1 or (self.done_at is not None and self.inside >= self.capacity):
                break
            self._tokens -= 1
            b = self.bodies[i]
            vx, vy = b.velocity
            sp = math.hypot(vx, vy)
            a = math.atan2(vy, vx) if sp > 1 else -math.pi / 2
            da = self.rng.choice((-1, 1)) * self.rng.uniform(0.25, 0.6)
            k = self.rng.uniform(0.85, 1.08)
            nv = (max(sp, 220) * k * math.cos(a + da), max(sp, 220) * k * math.sin(a + da))
            spot = self._room(b.position, a + da)
            if spot is None:
                # packed in: it squeezes in next to it anyway and the pile swells
                # (pushed apart gently: the speed limit keeps it from bursting)
                o = self.rng.uniform(0, 2 * math.pi)
                spot = (b.position.x + 0.7 * self.r * math.cos(o), b.position.y + 0.7 * self.r * math.sin(o))
            x, y = spot
            c = self.rng.randrange(self.ncol) if self.rng.random() < 0.55 else self.cols[i]
            s = self._add(x, y, nv, c)
            self.last_split[s.idx] = self.t
            live += 1
            self.spawned += 1
            vol = 1.0 / (1.0 + live / 400.0)
            self.hits.append((self.t, vol))
            self.splits.append((self.t, live))
        self._queue.clear()

    def _room(self, p, a):
        """A free spot right next to the ball (its new half goes there), or None.
        Spawning into other balls would blast the pile apart."""
        for dist, dirs in ((2.05, (0, 1, -1, 2, -2, 3, -3, 4)), (3.0, (0, 1, -1, 2, -2, 3, -3, 4)),
                           (4.2, (0, 1, -1, 2, -2, 3, -3, 4))):
            for k in dirs:
                o = a + k * math.pi / 4 + (dist > 2.1) * math.pi / 8
                x, y = p.x + dist * self.r * math.cos(o), p.y + dist * self.r * math.sin(o)
                if math.hypot(x - CX, y - CY) > R - RIM_T - self.r - 1:
                    continue
                if y > HUB[1] - abs(x - CX) * math.tan(PLATE_A) - (RIM_T + self.r) / math.cos(PLATE_A):
                    continue
                if not self.space.point_query((x, y), self.r * 0.98, self._filter):
                    return x, y
        return None

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def step(self):
        dt = 1 / FPS
        h = dt / SUBSTEPS
        for k in range(SUBSTEPS):
            self._set_paddle(self.t + (k + 1) * h)
            self.space.step(h)
            self._splits(h)
        self.t += dt
        self._sync()
        self._milestones()

    def _sync(self):
        """Positions into numpy, spills counted, leaks put back, fallen balls dropped."""
        if not self.bodies:
            self.xy = np.zeros((0, 2), np.float32)
            return
        xy = np.array([tuple(b.position) for b in self.bodies], np.float32)
        # a speed limit: a pile squeezed between the paddle and the rim would
        # otherwise squirt balls out at any speed and empty the bowl as a wave
        vmax = VMAX
        for b in self.bodies:
            vx, vy = b.velocity
            if vx * vx + vy * vy > vmax * vmax:
                k = vmax / math.hypot(vx, vy)
                b.velocity = (vx * k, vy * k)
        d = np.hypot(xy[:, 0] - CX, xy[:, 1] - CY)
        out = np.array(self.out)
        leaving = (~out) & (d > R + 1)
        for i in np.nonzero(leaving)[0]:
            ang = math.atan2(xy[i, 1] - CY, xy[i, 0] - CX)
            if abs(_wrap(ang + math.pi / 2)) < GAP_HALF + 0.05:
                self.out[i] = True               # out over the top: a spill
                self.spilled += 1
                if self.inside < EARLY_SHARE * self.capacity:
                    self.early_spills += 1
            else:                                # squeezed through the rim: put it back
                b = self.bodies[i]
                rr = R - RIM_T - self.r - 1
                b.position = (CX + rr * math.cos(ang), CY + rr * math.sin(ang))
                b.velocity = (0, 0)
                xy[i] = b.position
        # squeezed under a floor plate (the paddle pinches balls against them): back on top
        tan = math.tan(PLATE_A)
        floor = HUB[1] - np.abs(xy[:, 0] - CX) * tan
        under = (~out) & (xy[:, 1] > floor - RIM_T * 0.5) & (d <= R + 1)
        for i in np.nonzero(under)[0]:
            b = self.bodies[i]
            lift = (RIM_T + self.r + 1) / math.cos(PLATE_A)
            b.position = (float(xy[i, 0]), float(floor[i] - lift))
            b.velocity = (b.velocity.x * 0.3, min(0.0, b.velocity.y))
            xy[i] = b.position
        gone = (xy[:, 1] > H + 40) | (xy[:, 0] < -60) | (xy[:, 0] > W + 60)
        if gone.any():
            keep = []
            for i, g in enumerate(gone):
                if g:
                    b = self.bodies[i]
                    self.space.remove(b, *b.shapes)
                else:
                    keep.append(i)
            self.bodies = [self.bodies[i] for i in keep]
            self.cols = [self.cols[i] for i in keep]
            self.out = [self.out[i] for i in keep]
            self.last_split = [self.last_split[i] for i in keep]
            for j, b in enumerate(self.bodies):
                for s in b.shapes:
                    s.idx = j
            xy = xy[keep]
        self.xy = xy
        self.col = np.array(self.cols, np.int16)
        self.inside = len(self.out) - sum(self.out)

    def _milestones(self):
        n = self.spawned
        for k, need in ((1, 10), (2, 100), (3, 1000)):
            if k not in self.unlocked_at and n >= need:
                self.unlocked_at[k] = self.t
                self.best = k
        if (self.done_at is None and self.spilled >= SPILL_N and self.inside >= FULL_SHARE * self.capacity):
            self.done_at = self.t
            self.unlocked_at[4] = self.t
            self.best = 4
            times = sorted(self.unlocked_at.values())
            self.max_gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
            self.ok = self.early_spills <= EARLY_MAX
        if self.done_at is None:
            times = sorted(self.unlocked_at.values())
            self.max_gap = max(max((b - a for a, b in zip(times, times[1:])), default=0.0), self.t - times[-1])
            if self.t > SECONDS_CAP - END_HOLD - 0.1:
                self.done_at = self.t            # never overflowed: end anyway (not ok)
                self.ok = False

    # ---- sound -----------------------------------------------------------
    def music_track(self, n: int) -> np.ndarray:
        """Crunch per split (denser and softer as it grows), a sand-pour bed, the overflow swell."""
        g = np.random.default_rng(int(self.recipe.get("seed") or 0) + 77)
        mix = np.zeros(n + SR, np.float64)
        bank = [_crunch(g) for _ in range(28)]
        ts = np.array([t for t, _ in self.splits]) if self.splits else np.zeros(0)
        last = -1.0
        for t, live in self.splits:
            i = int(t * SR)
            if i >= n:
                continue
            # hits per second around now: the gate opens up and each hit gets softer
            rate = (np.searchsorted(ts, t + 0.25) - np.searchsorted(ts, t - 0.25)) * 2.0
            gap = 1.0 / (7.0 + 45.0 * min(1.0, rate / 120.0))
            if t - last < gap:
                continue
            last = t
            h = bank[g.integers(len(bank))]
            gain = g.uniform(0.5, 1.0) / math.sqrt(1.0 + min(rate, 400) / 25.0)
            seg = mix[i:i + len(h)]
            seg += h[:len(seg)] * gain
        # the sand pour: bandpassed noise growing with how full the bowl is
        fill = np.zeros(len(mix))
        if self.splits:
            st = np.array([t for t, _ in self.splits])
            sl = np.array([c for _, c in self.splits], float)
            tt = np.arange(len(mix)) / SR
            fill = np.interp(tt, st, np.minimum(1.0, sl / self.capacity), left=0.0)
        bed = _band(g.standard_normal(len(mix)), 1500, 7000)
        bed /= max(1e-9, float(np.abs(bed).max()))
        amp = 0.05 * fill ** 1.5
        if self.done_at is not None:                    # the pour over the rim: louder, then it settles
            tt = np.arange(len(mix)) / SR - self.done_at
            amp = amp + 0.07 * np.clip(tt / 0.4, 0, 1) * np.exp(-np.clip(tt, 0, None) / 3.0)
            i = int(self.done_at * SR)
            k = int(1.6 * SR)
            ts2 = np.arange(k) / SR
            f = 55 + 40 * np.exp(-ts2 * 3)
            swell = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, ts2 / 0.08) * np.exp(-ts2 * 2.2) * 0.25
            seg = mix[i:i + k]
            seg += swell[:len(seg)]
        mix += bed * amp
        # a little room (FFT convolution)
        m = int(0.08 * SR)
        ir = g.standard_normal(m) * np.exp(-np.arange(m) / (0.015 * SR)) * 0.04
        ir[0] = 1.0
        L = 1 << int(math.ceil(math.log2(len(mix) + m)))
        mix = np.fft.irfft(np.fft.rfft(mix, L) * np.fft.rfft(ir, L), L)[:len(mix)]
        peak = float(np.abs(mix).max()) or 1.0
        mix = np.tanh(mix / peak * 1.6) * 0.89
        return mix[:n].astype(np.float32)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def _crunch(r) -> np.ndarray:
    """Micro-grains: a chip / gravel crunch with a tiny low knock under it."""
    out = np.zeros(int(0.06 * SR))
    for _ in range(int(r.integers(4, 10))):
        at = int(r.uniform(0, 0.03) * SR)
        m = int(r.uniform(0.002, 0.007) * SR)
        grain = _band(r.standard_normal(m * 4), 1800, 9000)[:m] * np.exp(-np.arange(m) / (0.0015 * SR))
        out[at:at + m] += grain * r.uniform(0.3, 1)
    m = int(0.02 * SR)
    t = np.arange(m) / SR
    out[:m] += 0.25 * np.sin(2 * np.pi * r.uniform(110, 160) * t) * np.exp(-np.arange(m) / (0.006 * SR))
    return out


# ---- drawing --------------------------------------------------------------

def _sprite(r: float, col: tuple, S: int = 4):
    """A flat ball: its colour, a thin darker edge and a small highlight, as (rgb, alpha) arrays."""
    size = 2 * int(math.ceil(r)) + 3
    c = size / 2
    big = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    dark = tuple(int(v * 0.72) for v in col)
    d.ellipse([S * (c - r), S * (c - r), S * (c + r), S * (c + r)], fill=dark + (255,))
    ri = r - 1.4
    d.ellipse([S * (c - ri), S * (c - ri), S * (c + ri), S * (c + ri)], fill=col + (255,))
    hx, hy, hr = c - r * 0.35, c - r * 0.38, r * 0.28
    lite = tuple(min(255, int(v + (255 - v) * 0.6)) for v in col)
    d.ellipse([S * (hx - hr), S * (hy - hr), S * (hx + hr), S * (hy + hr)], fill=lite + (255,))
    im = big.resize((size, size), Image.LANCZOS)
    a = np.asarray(im, np.float32)
    rgb = a[..., :3] / np.maximum(a[..., 3:] / 255.0, 1e-6)        # un-premultiply the edge
    return np.clip(rgb, 0, 255), a[..., 3] / 255.0, size


class OverflowPainter:
    SS = 2

    def __init__(self, recipe: dict):
        import cv2

        self.cv2 = cv2
        self.recipe = recipe
        self.pal = PALETTES[int(recipe.get("palette") or 0) % len(PALETTES)]
        back, inner = BACKS[int(recipe.get("back") or 0) % len(BACKS)]
        self.r = float(recipe.get("ball_r") or 9.5)
        self.bg = self._background(back, inner)
        self.bg_np = np.asarray(self.bg).copy()
        # sprites: pixel offsets of the edge (blended) and the core (solid)
        rgbs, alpha, size = [], None, 0
        for col in self.pal:
            rgb, alpha, size = _sprite(self.r, col)
            rgbs.append(rgb)
        self.half = size // 2
        yy, xx = np.nonzero(alpha > 0.02)
        core = alpha[yy, xx] >= 0.98
        self.e_dy, self.e_dx = yy[~core], xx[~core]
        self.c_dy, self.c_dx = yy[core], xx[core]
        self.e_a = alpha[self.e_dy, self.e_dx][None, :, None].astype(np.float32)
        self.e_rgb = np.stack([r[self.e_dy, self.e_dx] for r in rgbs]).astype(np.float32)   # (ncol, k, 3)
        self.c_rgb = np.stack([r[self.c_dy, self.c_dx] for r in rgbs]).round().astype(np.uint8)
        self.f_count = _font(84)
        self.f_small = _font(40, "Inter-Bold.ttf")
        self.f_end = _font(132)
        self._ends = {}

    def _background(self, back, inner) -> Image.Image:
        S = self.SS
        im = Image.new("RGB", (W * S, H * S), back)
        d = ImageDraw.Draw(im)

        def P(x, y):
            return (x * S, y * S)

        Ri = R - RIM_T
        d.ellipse([*P(CX - Ri, CY - Ri), *P(CX + Ri, CY + Ri)], fill=inner)
        # the floor: everything under the V, in the plates' grey, with a lighter top edge
        a8, a4 = math.pi - RIM_A, RIM_A
        p8 = (CX + R * math.cos(a8), CY + R * math.sin(a8))
        p4 = (CX + R * math.cos(a4), CY + R * math.sin(a4))
        plate = (84, 94, 114)
        d.polygon([P(*q) for q in floor_poly()], fill=plate)
        for q in (p8, p4):
            d.line([P(*HUB), P(*q)], fill=(150, 160, 182), width=2 * RIM_T * S)
        # the rim: thick slate grey, open at the top, with round ends
        rim = (104, 116, 138)
        t0 = -math.pi / 2 + GAP_HALF
        t1 = -math.pi / 2 - GAP_HALF + 2 * math.pi
        Ro, Rn = R + 16, R - RIM_T
        # (Pillow's arc grows its width inward from the box: the box is the outer edge)
        d.arc([*P(CX - Ro, CY - Ro), *P(CX + Ro, CY + Ro)],
              math.degrees(t0), math.degrees(t1), fill=rim, width=int((Ro - Rn) * S))
        for a in (t0, t1):
            mx, my = CX + (Ro + Rn) / 2 * math.cos(a), CY + (Ro + Rn) / 2 * math.sin(a)
            rr = (Ro - Rn) / 2
            d.ellipse([*P(mx - rr, my - rr), *P(mx + rr, my + rr)], fill=rim)
        # tick marks round the outside, a longer one every 30 degrees
        for k in range(48):
            a = 2 * math.pi * k / 48
            if abs(_wrap(a + math.pi / 2)) < GAP_HALF + math.radians(4):
                continue
            long = k % 4 == 0
            r0, r1 = R + 26, R + (48 if long else 38)
            d.line([P(CX + r0 * math.cos(a), CY + r0 * math.sin(a)), P(CX + r1 * math.cos(a), CY + r1 * math.sin(a))],
                   fill=(96, 108, 130) if long else (66, 76, 96), width=(5 if long else 3) * S)
        # the hub: a half-disc where the plates meet
        hx, hy = HUB
        d.pieslice([*P(hx - HUB_R, hy - HUB_R), *P(hx + HUB_R, hy + HUB_R)], 180, 360, fill=(150, 160, 182))
        im = im.resize((W, H), Image.LANCZOS)
        # the titles never change: drawn once
        d = ImageDraw.Draw(im)
        hook = self.recipe.get("hook") or HOOKS[0]
        size = 92
        while size > 50 and d.textlength(hook, font=_font(size)) > W - 100:
            size -= 4
        d.text((W // 2, 200), hook, font=_font(size), fill=(250, 250, 255), anchor="mm")
        sub = self.recipe.get("sub") or SUBS[0]
        if "split" in hook.lower() and "split" in sub:      # the title may be swapped for another hook
            sub = "will it overflow?"
        d.text((W // 2, 292), sub, font=_font(46, "Inter-Bold.ttf"),
               fill=(168, 180, 206), anchor="mm")
        return im

    def _balls(self, fr: np.ndarray, xy: np.ndarray, col: np.ndarray):
        if not len(xy):
            return
        flat = fr.reshape(-1, 3)
        x0 = np.rint(xy[:, 0]).astype(np.int32) - self.half
        y0 = np.rint(xy[:, 1]).astype(np.int32) - self.half
        # edges first (blended over what's there), then the solid cores over them
        Y = y0[:, None] + self.e_dy[None, :]
        X = x0[:, None] + self.e_dx[None, :]
        ok = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
        idx = (Y * W + X)[ok]
        src = self.e_rgb[col][ok]
        a = np.broadcast_to(self.e_a, (len(xy), self.e_a.shape[1], 1))[ok]
        flat[idx] = (flat[idx] * (1 - a) + src * a).astype(np.uint8)
        Y = y0[:, None] + self.c_dy[None, :]
        X = x0[:, None] + self.c_dx[None, :]
        ok = (Y >= 0) & (Y < H) & (X >= 0) & (X < W)
        flat[(Y * W + X)[ok]] = self.c_rgb[col][ok]

    def _paddle(self, fr: np.ndarray, t: float):
        cv2 = self.cv2
        ang, _w = paddle_angle(self.recipe, t)
        sh = 16
        pts = np.array([[int(x * sh), int(y * sh)] for x, y in paddle_poly(self.recipe, ang)], np.int32)
        cv2.fillPoly(fr, [pts], (196, 204, 222), cv2.LINE_AA, 4)
        cv2.polylines(fr, [pts], True, (132, 142, 166), 3, cv2.LINE_AA, 4)
        (px, py), _L, _h0, _h1 = paddle_geom(self.recipe)
        cv2.circle(fr, (int(px * sh), int(py * sh)), int(19 * sh), (132, 142, 166), -1, cv2.LINE_AA, 4)
        cv2.circle(fr, (int(px * sh), int(py * sh)), int(9 * sh), (222, 228, 240), -1, cv2.LINE_AA, 4)

    def frame(self, sim: OverflowSim) -> Image.Image:
        fr = self.bg_np.copy()
        # the paddle over the balls: balls it pins against a plate sit under it
        self._balls(fr, sim.xy, sim.col)
        self._paddle(fr, sim.t)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        n = sim.spawned
        d.text((W // 2, 430), f"{n:,}", font=self.f_count, fill=(255, 255, 255), anchor="mm")
        d.text((W // 2 + d.textlength(f"{n:,}", font=self.f_count) / 2 + 14, 444), "ball" if n == 1 else "balls",
               font=self.f_small,
               fill=(168, 180, 206), anchor="lm")
        if sim.done_at is not None and 4 in sim.unlocked_at:          # it really overflowed
            age = sim.t - sim.done_at
            if age > 0.05:
                k = min(1.0, (age - 0.05) / 0.22)
                text = self.recipe.get("ending") or ENDINGS[0]
                full = 132
                while full > 64 and d.textlength(text, font=_font(full)) > W - 110:
                    full -= 6
                sz = int(full * (0.53 + 0.47 * (1 - (1 - k) ** 3)) + 10 * math.sin(min(1.0, k) * math.pi))
                f = self._ends.setdefault(sz, _font(sz))
                d.text((W // 2, CY - 30), text, font=f, fill=(255, 214, 102),
                       anchor="mm", stroke_width=max(4, sz // 14), stroke_fill=(12, 12, 18))
        return im


def publish_text(recipe: dict) -> dict:
    return {"title": recipe["hook"][:100],
            "description": ("One ball drops into the bowl. Every time a ball hits the paddle, it splits into two, "
                            "so the count keeps doubling until the bowl can't hold them all.\n\n"
                            "Did you think it would overflow? 👇\n\n#satisfying #asmr #physics #simulation")}
