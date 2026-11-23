"""Will it overflow? Every squish makes two (Nov 2026).

The idea of a reel with 1.5M likes, drawn our own way: a still, closed
circle whose rim is a ring of dark tiles. The bottom third is the floor:
two wide plates (sectors from the hub in the middle out to the rim, at 4 and
8 o'clock round to 6) with a half-disc hub. A wide paddle (a sector from the
hub to the rim, its outer end following the rim) sweeps back and forth over
the top, from one plate to the other, closing fully against each.

The balls never stop bouncing (no gravity, no friction, fully elastic, each
ball's speed held in a band with a tiny random turn so none loops forever)
and a plain hit is only a bounce. The one rule (Dean's favourite): a ball
caught in the closing gap between the paddle and a plate is squished, and
pops out as TWO balls on the other side of the paddle. The paddle is faster
than the balls far from the hub, so it herds the ones ahead of it into the
gap: every close squishes a strip of them and the count roughly doubles each
sweep. When the circle is packed the rim shatters: its tiles fly off and
the balls burst out in every direction (OVERFLOW!).

Sim + painter for ``format: "overflow"``, a ``NO_PACK`` kind
(``ball_evolution._parts`` dispatches here). Physics is pymunk with a
spatial hash; the squish is geometry (each substep: is the gap where a ball
sits narrower than the ball?). Balls are drawn by a numpy splat of
pre-made sprites. Sound is ours, numpy only (``music_track``): a
micro-grain crunch per squish (rate-limited, softer as they get denser) and
a crack + boom + scatter of grains when the rim shatters.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from clipper.ball_evolution import FPS, H, SR, W, _font

END_HOLD = 1.8                  # after the shatter (CLAUDE.md: 1.8 s everywhere)
SECONDS_CAP = 58.0
CX, CY, R = 540, 1010, 440      # the circle; it never moves
RIM_T = 8                       # the rim's physics half-thickness (segment radius)
RIM_OUT = 18                    # the tiles reach this far outside R
TILES = 44                      # tiles round the rim
HUB_R = 46                      # the half-disc hub in the middle (and the paddle's pivot)
FLOOR_A0, FLOOR_A1 = math.radians(30), math.radians(150)   # the floor: 4 o'clock round to 8 o'clock
FREE = 2 * math.pi - (FLOOR_A1 - FLOOR_A0)                  # the open part the paddle sweeps (240 deg)
PADDLE_W = math.radians(12)     # the paddle's half-width (24 deg: 1/15 of the circle; 1/7.5 seen from the rim)
PERIOD = 3.6                    # s for the paddle's full there-and-back
SPEED = 850.0                   # px/s: each ball's speed, held in a band
SPEED_BAND = (0.8, 1.25)
SQUISH_K = 1.0                  # squished once the gap where it sits is narrower than SQUISH_K x its width
FILL = 0.50                     # the circle counts as packed at this share of its open area covered
CAP = 2000                      # never more balls than this
MOTIONS = ("swing", "swing", "flick")
HOOKS = ["Will It Overflow?", "Will the circle overflow?", "Can the circle hold them all?",
         "How fast does it overflow?", "One ball. Will it overflow?", "Every squish makes two",
         "How many until it breaks?", "Squish = 2 balls"]
SUBS = ["every squish makes two", "squish = 2 balls"]
ENDINGS = ["OVERFLOW!", "IT BROKE!", "OVERFLOW!"]
PALETTES = [   # pastel balls: coral, lilac, mint, gold, sky -- and three more sets of our own
    [(255, 138, 128), (190, 160, 255), (120, 230, 190), (255, 214, 102), (120, 196, 255)],
    [(255, 160, 190), (255, 196, 140), (170, 230, 140), (140, 210, 240), (215, 170, 250)],
    [(250, 128, 114), (255, 190, 92), (130, 220, 210), (164, 180, 255), (240, 150, 220)],
    [(255, 230, 140), (150, 236, 200), (130, 190, 255), (255, 150, 150), (200, 170, 255)],
]
BACKS = [((14, 16, 26), (24, 28, 44)), ((20, 14, 28), (32, 24, 46)), ((10, 20, 24), (20, 34, 40))]
PLATE, PLATE_EDGE, GROOVE = (84, 94, 114), (150, 160, 182), (66, 74, 94)
TILE, TILE_EDGE = (58, 66, 84), (92, 104, 128)


def overflow_recipe(recipe: dict) -> dict:
    rng = random.Random(f"overflow-recipe-{recipe['seed']}")
    recipe.update(hook=rng.choice(HOOKS), sub=rng.choice(SUBS), palette=rng.randrange(len(PALETTES)),
                  back=rng.randrange(len(BACKS)), motion=rng.choice(MOTIONS),
                  speed=round(rng.uniform(0.92, 1.1), 3), ball_r=rng.choice((8.0, 8.5, 9.0)),
                  spin=rng.choice((1, -1)), ending=rng.choice(ENDINGS), bed="off")
    return recipe


def _u_to_beta(u):
    """Open-part angle u (0 at the 8 o'clock plate, FREE at the 4 o'clock one,
    through 12 o'clock) -> screen angle (radians, clockwise from 3 o'clock)."""
    return FLOOR_A1 + u


def _sweep(x: float, k: float = 0.3):
    """Position and speed along one sweep, x in 0..1: mostly even speed, eased a little at the ends."""
    return (1 - k) * x + k * x * x * (3 - 2 * x), (1 - k) + k * 6 * x * (1 - x)


def paddle_u(recipe: dict, t: float):
    """The paddle's middle as an open-part angle, and its angular speed (rad/s).
    ``swing``: steady back and forth; ``flick``: rests closed on each plate a moment."""
    sp = float(recipe.get("speed") or 1.0)
    lo, hi = PADDLE_W, FREE - PADDLE_W
    if int(recipe.get("spin") or 1) < 0:
        lo, hi = hi, lo
    T = PERIOD / sp
    hold = 0.12 if recipe.get("motion") == "flick" else 0.0
    half = 0.5 - hold
    v = (t / T + 0.25) % 1.0             # start mid-sweep, so the first close comes quickly
    if v < hold:
        return lo, 0.0
    if v < 0.5:
        s, ds = _sweep((v - hold) / half)
        return lo + (hi - lo) * s, (hi - lo) * ds / (half * T)
    if v < 0.5 + hold:
        return hi, 0.0
    s, ds = _sweep((v - 0.5 - hold) / half)
    return hi - (hi - lo) * s, -(hi - lo) * ds / (half * T)


def sector(u0: float, u1: float, r0: float, r1: float, n: int = 12) -> list:
    """Screen points of the ring sector between open-part angles u0..u1 and radii r0..r1 (r0 0 = to the middle)."""
    out = [(CX + r1 * math.cos(_u_to_beta(u)), CY + r1 * math.sin(_u_to_beta(u))) for u in np.linspace(u0, u1, n)]
    if r0 <= 0:
        return [(CX, CY)] + out
    return out + [(CX + r0 * math.cos(_u_to_beta(u)), CY + r0 * math.sin(_u_to_beta(u)))
                  for u in np.linspace(u1, u0, n)]


def full_count(r: float) -> int:
    """How many balls of radius r make the open part 'packed'."""
    Ri = R - RIM_T
    area = (FREE - 2 * PADDLE_W) / 2 * (Ri * Ri - HUB_R * HUB_R)
    return min(CAP - 50, int(area * FILL / (math.pi * r * r)))


class OverflowSim:
    def __init__(self, recipe: dict):
        import pymunk

        self.pm = pymunk
        self.recipe = recipe
        self.rng = random.Random(f"overflow-{recipe['seed']}")
        self.r = float(recipe.get("ball_r") or 8.5)
        self.full = full_count(self.r)
        self.ncol = len(PALETTES[int(recipe.get("palette") or 0) % len(PALETTES)])
        self.spd = SPEED * float(recipe.get("speed") or 1.0)
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
        self.hits: list = []            # (t, vol): one per squish
        self.squishes: list = []        # (t, count after)
        self.pops: list = []            # (t, x, y): where a squished ball popped out (the painter flashes them)
        self.tiles: list = []           # after the shatter: flying rim tiles
        self.n_final = 0

        sp = self.space = pymunk.Space()
        sp.gravity = (0, 0)
        sp.iterations = 8
        sp.use_spatial_hash(2 * self.r + 2, 8000)
        self.rim = []
        n = 120
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            s = pymunk.Segment(sp.static_body, (CX + R * math.cos(a0), CY + R * math.sin(a0)),
                               (CX + R * math.cos(a1), CY + R * math.sin(a1)), RIM_T)
            s.elasticity, s.friction = 1.0, 0.0
            self.rim.append(s)
            sp.add(s)
        for u0, u1 in ((FREE, FREE + (FLOOR_A1 - FLOOR_A0) / 2), (FREE + (FLOOR_A1 - FLOOR_A0) / 2, 2 * math.pi)):
            floor = pymunk.Poly(sp.static_body, sector(u0, u1, 0, R - 2, 8))
            floor.elasticity, floor.friction = 1.0, 0.0
            sp.add(floor)
        hub = pymunk.Circle(sp.static_body, HUB_R, (CX, CY))
        hub.elasticity, hub.friction = 1.0, 0.0
        sp.add(hub)

        self.paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.paddle.position = (CX, CY)
        L = R - RIM_T - 1
        pts = [(0, 0)] + [(L * math.cos(a), L * math.sin(a)) for a in np.linspace(-PADDLE_W, PADDLE_W, 7)]
        shape = pymunk.Poly(self.paddle, pts)
        shape.elasticity, shape.friction = 1.0, 0.0
        sp.add(self.paddle, shape)
        self.u_p, self.w_p = paddle_u(recipe, 0.0)
        self._set_paddle(0.0)

        self.bodies: list = []
        self.cols: list = []
        self.xy = np.zeros((0, 2), np.float32)
        self.col = np.zeros(0, np.int16)
        # one ball, somewhere in the top half, off in a random direction
        u = self.rng.uniform(FREE * 0.3, FREE * 0.7)
        d = self.rng.uniform(150, 300)
        a = self.rng.uniform(0, 2 * math.pi)
        self._add(CX + d * math.cos(_u_to_beta(u)), CY + d * math.sin(_u_to_beta(u)),
                  (self.spd * math.cos(a), self.spd * math.sin(a)), self.rng.randrange(self.ncol))
        self._sync()

    # ---- physics ---------------------------------------------------------
    def _set_paddle(self, t: float):
        u, w = paddle_u(self.recipe, t)
        self.u_p, self.w_p = u, w
        self.paddle.angle = _u_to_beta(u)
        self.paddle.angular_velocity = w
        self.space.reindex_shapes_for_body(self.paddle)

    def _add(self, x, y, v, c):
        pm = self.pm
        b = pm.Body(1.0, pm.moment_for_circle(1.0, 0, self.r))
        b.position = (x, y)
        b.velocity = v
        s = pm.Circle(b, self.r)
        s.elasticity, s.friction = 1.0, 0.0
        self.space.add(b, s)
        self.bodies.append(b)
        self.cols.append(c)

    def _remove(self, idx: list):
        drop = set(idx)
        for i in idx:
            b = self.bodies[i]
            self.space.remove(b, *b.shapes)
        keep = [i for i in range(len(self.bodies)) if i not in drop]
        self.bodies = [self.bodies[i] for i in keep]
        self.cols = [self.cols[i] for i in keep]

    def _polar(self):
        xy = np.array([tuple(b.position) for b in self.bodies], np.float64).reshape(-1, 2)
        dx, dy = xy[:, 0] - CX, xy[:, 1] - CY
        d = np.hypot(dx, dy)
        u = (np.arctan2(dy, dx) - FLOOR_A1) % (2 * math.pi)
        return xy, d, u

    def _squish(self):
        """Balls in the closing gap between the paddle and a plate: each pops out
        as two just behind the paddle's back face."""
        if self.done_at is not None or not self.bodies or abs(self.w_p) < 1e-3:
            return
        xy, d, u = self._polar()
        if self.w_p > 0:      # closing on the 4 o'clock plate (u = FREE)
            front, gap = self.u_p + PADDLE_W, FREE - (self.u_p + PADDLE_W)
            inside = (u > front - 0.02) & (u < FREE + 0.05)
        else:                 # closing on the 8 o'clock plate (u = 0)
            front, gap = self.u_p - PADDLE_W, self.u_p - PADDLE_W
            inside = (u < front + 0.02) | (u > 2 * math.pi - 0.05)
        caught = np.nonzero(inside & (d * max(gap, 0.0) < 2 * self.r * SQUISH_K) & (d > HUB_R))[0]
        if not len(caught):
            return
        back = self.u_p - math.copysign(PADDLE_W, self.w_p)      # the paddle's back face
        sgn = -math.copysign(1.0, self.w_p)                        # away from it
        Ri = R - RIM_T - self.r - 2
        born = []
        for i in caught:
            if len(self.bodies) - len(caught) + len(born) + 2 > CAP:
                break
            dd = float(min(max(d[i], HUB_R + self.r + 4), Ri))
            off = (self.r + 3) / dd
            for k, dr in enumerate((0.0, 2.3 * self.r)):
                rr = dd + dr if dd + dr < Ri else dd - dr
                ub = back + sgn * (off + (k * 0.6 * self.r / rr))
                bx, by = CX + rr * math.cos(_u_to_beta(ub)), CY + rr * math.sin(_u_to_beta(ub))
                # pushed away from the paddle (its tangent direction), a little spread
                tb = _u_to_beta(ub) + sgn * math.pi / 2 + self.rng.uniform(-0.5, 0.5)
                born.append((bx, by, (self.spd * 1.1 * math.cos(tb), self.spd * 1.1 * math.sin(tb)),
                             self.cols[i] if k == 0 else self.rng.randrange(self.ncol)))
            self.pops.append((self.t, float(xy[i, 0]), float(xy[i, 1])))
            self.hits.append((self.t, 1.0 / (1.0 + len(self.bodies) / 300.0)))
        n_caught = len(born) // 2
        self._remove([int(i) for i in caught[:n_caught]])
        for bx, by, v, c in born:
            self._add(bx, by, v, c)
        self.spawned += n_caught
        self.squishes.append((self.t, len(self.bodies)))

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def step(self):
        dt = 1 / FPS
        sub = 3
        h = dt / sub
        for k in range(sub):
            self._set_paddle(self.t + (k + 1) * h)
            self._squish()
            self.space.step(h)
        self.t += dt
        if self.done_at is None:
            self._keep_inside()
        else:
            self._fly(dt)
        self._sync()
        self._milestones()

    def _keep_inside(self):
        """Steady speeds (a band, a tiny random turn) and nobody inside the
        floor, the paddle or past the rim."""
        if not self.bodies:
            return
        xy, d, u = self._polar()
        lo, hi = self.spd * SPEED_BAND[0], self.spd * SPEED_BAND[1]
        Ri = R - RIM_T - self.r - 1
        in_floor = u > FREE + 0.002
        in_pad = np.abs(u - self.u_p) < PADDLE_W + self.r / np.maximum(d, 1) * 0.5
        for i, b in enumerate(self.bodies):
            vx, vy = b.velocity
            s = math.hypot(vx, vy)
            turn = self.rng.uniform(-0.04, 0.04)
            if s < 1:
                a = self.rng.uniform(0, 2 * math.pi)
                vx, vy, s = math.cos(a), math.sin(a), 1.0
            k = min(max(s, lo), hi) / s
            c, sn = math.cos(turn), math.sin(turn)
            b.velocity = ((vx * c - vy * sn) * k, (vx * sn + vy * c) * k)
            fix = None
            if d[i] > Ri:
                fix = (Ri, u[i])
            if in_floor[i]:
                fix = (min(d[i], Ri), 0.01 if u[i] - FREE > (2 * math.pi - FREE) / 2 else FREE - 0.01)
            elif in_pad[i] and d[i] > HUB_R:
                # behind the paddle (the side it's moving away from): never a pinch
                side = -1.0 if self.w_p >= 0 else 1.0
                fix = (min(d[i], Ri), self.u_p + side * (PADDLE_W + (self.r + 2) / max(d[i], 1)))
            if d[i] < HUB_R + self.r - 1:
                fix = (HUB_R + self.r + 1, u[i] if fix is None else fix[1])
            if fix is not None:
                rr, uu = fix
                uu = min(max(uu, 0.005), FREE - 0.005)
                b.position = (CX + rr * math.cos(_u_to_beta(uu)), CY + rr * math.sin(_u_to_beta(uu)))

    def _shatter(self):
        """The rim breaks: its tiles fly out, the balls burst out in every direction."""
        for s in self.rim:
            self.space.remove(s)
        self.rim = []
        rng = self.rng
        for k in range(TILES):
            a = 2 * math.pi * (k + 0.5) / TILES
            sp = rng.uniform(380, 820)
            self.tiles.append({"k": k, "x": CX + (R + 5) * math.cos(a), "y": CY + (R + 5) * math.sin(a),
                               "vx": sp * math.cos(a) + rng.uniform(-80, 80),
                               "vy": sp * math.sin(a) - rng.uniform(100, 300),
                               "rot": 0.0, "vr": rng.uniform(-7, 7)})
        for b in self.bodies:
            dx, dy = b.position.x - CX, b.position.y - CY
            dd = math.hypot(dx, dy) or 1.0
            out = rng.uniform(500, 1000) * (0.5 + 0.5 * dd / R)
            b.velocity = (b.velocity.x * 0.4 + out * dx / dd, b.velocity.y * 0.4 + out * dy / dd)

    def _fly(self, dt):
        for tl in self.tiles:
            tl["vy"] += 1600 * dt
            tl["x"] += tl["vx"] * dt
            tl["y"] += tl["vy"] * dt
            tl["rot"] += tl["vr"] * dt
        gone = [i for i, b in enumerate(self.bodies)
                if not (-80 < b.position.x < W + 80 and -80 < b.position.y < H + 80)]
        if gone:
            self._remove(gone)

    def _sync(self):
        if not self.bodies:
            self.xy = np.zeros((0, 2), np.float32)
            self.col = np.zeros(0, np.int16)
            return
        self.xy = np.array([tuple(b.position) for b in self.bodies], np.float32)
        self.col = np.array(self.cols, np.int16)

    def _milestones(self):
        n = self.spawned
        for k, need in ((1, 10), (2, 100), (3, 1000)):
            if k not in self.unlocked_at and n >= need:
                self.unlocked_at[k] = self.t
                self.best = k
        times = sorted(self.unlocked_at.values())
        if self.done_at is None and len(self.bodies) >= self.full:
            self.done_at = self.t
            self.unlocked_at[4] = self.t
            self.best = 4
            self.ok = True
            self.n_final = len(self.bodies)
            self.max_gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
            self.max_gap = max(self.max_gap, self.t - times[-1])
            self._shatter()
        elif self.done_at is None:
            self.max_gap = max(max((b - a for a, b in zip(times, times[1:])), default=0.0), self.t - times[-1])
            if self.t > SECONDS_CAP - END_HOLD - 0.1:
                self.done_at = self.t            # never filled: end anyway (not ok)

    # ---- sound -----------------------------------------------------------
    def music_track(self, n: int) -> np.ndarray:
        """A crunch per squish (denser and softer as they get more), the crack at the shatter."""
        g = np.random.default_rng(int(self.recipe.get("seed") or 0) + 77)
        mix = np.zeros(n + 2 * SR, np.float64)
        bank = [_crunch(g) for _ in range(28)]
        ts = np.array([t for t, _ in self.hits]) if self.hits else np.zeros(0)
        last = -1.0
        for t, _v in self.hits:
            i = int(t * SR)
            if i >= n:
                continue
            rate = (np.searchsorted(ts, t + 0.25) - np.searchsorted(ts, t - 0.25)) * 2.0
            gap = 1.0 / (8.0 + 50.0 * min(1.0, rate / 120.0))
            if t - last < gap:
                continue
            last = t
            h = bank[g.integers(len(bank))]
            gain = g.uniform(0.55, 1.0) / math.sqrt(1.0 + min(rate, 400) / 30.0)
            seg = mix[i:i + len(h)]
            seg += h[:len(seg)] * gain
        if self.ok and self.done_at is not None:
            i = int(self.done_at * SR)
            k = int(1.5 * SR)
            tt = np.arange(k) / SR
            crack = _band(g.standard_normal(k), 900, 12000) * np.exp(-tt / 0.045) * 0.9
            f = 70 + 60 * np.exp(-tt * 8)
            boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.35) * 0.8
            burst = crack + boom
            for _ in range(70):                 # the tiles and balls scattering: a shower of grains
                at = int(g.uniform(0.02, 0.8) ** 1.6 * SR)
                h = bank[g.integers(len(bank))] * g.uniform(0.15, 0.5)
                burst[at:at + len(h)] += h[:max(0, k - at)]
            seg = mix[i:i + k]
            seg += burst[:len(seg)]
        m = int(0.08 * SR)                      # a little room (FFT convolution)
        ir = g.standard_normal(m) * np.exp(-np.arange(m) / (0.015 * SR)) * 0.04
        ir[0] = 1.0
        L = 1 << int(math.ceil(math.log2(len(mix) + m)))
        mix = np.fft.irfft(np.fft.rfft(mix, L) * np.fft.rfft(ir, L), L)[:len(mix)]
        peak = float(np.abs(mix).max()) or 1.0
        mix = np.tanh(mix / peak * 1.6) * 0.89
        return mix[:n].astype(np.float32)


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


def _tile_poly(k: int) -> list:
    """Tile k of the rim, as screen points (small gaps between tiles)."""
    g = math.radians(0.9)
    a0, a1 = 2 * math.pi * k / TILES + g, 2 * math.pi * (k + 1) / TILES - g
    ri, ro = R - RIM_T, R + RIM_OUT
    out = [(CX + ro * math.cos(a), CY + ro * math.sin(a)) for a in np.linspace(a0, a1, 4)]
    return out + [(CX + ri * math.cos(a), CY + ri * math.sin(a)) for a in np.linspace(a1, a0, 4)]


class OverflowPainter:
    SS = 2

    def __init__(self, recipe: dict):
        import cv2

        self.cv2 = cv2
        self.recipe = recipe
        self.pal = PALETTES[int(recipe.get("palette") or 0) % len(PALETTES)]
        back, inner = BACKS[int(recipe.get("back") or 0) % len(BACKS)]
        self.r = float(recipe.get("ball_r") or 8.5)
        self.bg = self._background(back, inner, rim=True)
        self.bg_np = np.asarray(self.bg).copy()
        self.bg_open = None
        self._back, self._inner = back, inner
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
        self._ends = {}

    def _background(self, back, inner, rim: bool) -> Image.Image:
        S = self.SS
        im = Image.new("RGB", (W * S, H * S), back)
        d = ImageDraw.Draw(im)

        def P(x, y):
            return (x * S, y * S)

        Ri = R - RIM_T
        d.ellipse([*P(CX - Ri, CY - Ri), *P(CX + Ri, CY + Ri)], fill=inner)
        # the floor: two wide plates, 4 o'clock -> 6 and 6 -> 8 o'clock, each with a groove
        mid = FREE + (FLOOR_A1 - FLOOR_A0) / 2
        for u0, u1 in ((FREE, mid), (mid, 2 * math.pi)):
            pts = sector(u0 + 0.012, u1 - 0.012, 0, Ri, 24)
            d.polygon([P(*q) for q in pts], fill=PLATE)
            d.line([P(*q) for q in (pts[0], pts[1])], fill=PLATE_EDGE, width=3 * S)
            d.line([P(*q) for q in (pts[0], pts[-1])], fill=PLATE_EDGE, width=3 * S)
            um = (u0 + u1) / 2      # a groove along the middle of each plate, like the reel's
            a, b = Ri * 0.3, Ri * 0.8
            d.line([P(CX + a * math.cos(_u_to_beta(um)), CY + a * math.sin(_u_to_beta(um))),
                    P(CX + b * math.cos(_u_to_beta(um)), CY + b * math.sin(_u_to_beta(um)))], fill=GROOVE, width=5 * S)
        # the rim: a ring of dark tiles with small gaps
        if rim:
            for k in range(TILES):
                pts = [P(*q) for q in _tile_poly(k)]
                d.polygon(pts, fill=TILE, outline=TILE_EDGE)
        # tick marks round the outside, a longer one every 30 degrees
        for k in range(48):
            a = 2 * math.pi * k / 48
            long = k % 4 == 0
            r0, r1 = R + 30, R + (50 if long else 40)
            d.line([P(CX + r0 * math.cos(a), CY + r0 * math.sin(a)), P(CX + r1 * math.cos(a), CY + r1 * math.sin(a))],
                   fill=(96, 108, 130) if long else (60, 70, 90), width=(5 if long else 3) * S)
        # the hub: a half-disc on the plates' meeting point (flat side down)
        d.pieslice([*P(CX - HUB_R, CY - HUB_R), *P(CX + HUB_R, CY + HUB_R)], 180, 360, fill=PLATE_EDGE)
        d.rectangle([*P(CX - HUB_R, CY), *P(CX + HUB_R, CY + 6)], fill=PLATE_EDGE)
        im = im.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        hook = self.recipe.get("hook") or HOOKS[0]
        size = 92
        while size > 50 and d.textlength(hook, font=_font(size)) > W - 100:
            size -= 4
        d.text((W // 2, 200), hook, font=_font(size), fill=(250, 250, 255), anchor="mm")
        sub = self.recipe.get("sub") or SUBS[0]
        if any(w in hook.lower() for w in ("squish", "split", "makes two")):   # the title may be swapped
            sub = "will it overflow?"
        d.text((W // 2, 292), sub, font=_font(46, "Inter-Bold.ttf"), fill=(168, 180, 206), anchor="mm")
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

    def _paddle(self, fr: np.ndarray, u: float):
        cv2 = self.cv2
        sh = 16
        L = R - RIM_T - 1

        def at(dist, du):          # a point on the paddle: distance from the hub, angle off its middle
            b = _u_to_beta(u + du)
            return (CX + dist * math.cos(b), CY + dist * math.sin(b))

        def q(p):
            return (int(p[0] * sh), int(p[1] * sh))
        pts = np.array([q((CX, CY))] + [q(at(L, a)) for a in np.linspace(-PADDLE_W, PADDLE_W, 16)], np.int32)
        cv2.fillPoly(fr, [pts], (196, 204, 222), cv2.LINE_AA, 4)
        cv2.polylines(fr, [pts], True, (132, 142, 166), 3, cv2.LINE_AA, 4)
        for f in (0.38, 0.5, 0.62, 0.74, 0.86):     # ruler marks across it
            dist = L * f
            cv2.line(fr, q(at(dist, -PADDLE_W * 0.7)), q(at(dist, PADDLE_W * 0.25)), (120, 130, 156), 4, cv2.LINE_AA, 4)
        hx_, hy_ = at(L * 0.93, PADDLE_W * 0.45)     # the little hole near the tip
        cv2.circle(fr, q((hx_, hy_)), int(8 * sh), (120, 130, 156), 2, cv2.LINE_AA, 4)
        cv2.circle(fr, q((CX, CY)), int(19 * sh), (132, 142, 166), -1, cv2.LINE_AA, 4)
        cv2.circle(fr, q((CX, CY)), int(9 * sh), (222, 228, 240), -1, cv2.LINE_AA, 4)

    def _pops(self, fr: np.ndarray, sim: OverflowSim):
        """A small white ring where a squished ball just popped (0.18 s)."""
        cv2 = self.cv2
        for t, x, y in reversed(sim.pops[-400:]):
            age = sim.t - t
            if age > 0.18:
                break
            if age < 0:
                continue
            k = age / 0.18
            cv2.circle(fr, (int(x * 16), int(y * 16)), int((self.r + 2 + 10 * k) * 16),
                       (255, 255, 255), max(1, int(3 * (1 - k))), cv2.LINE_AA, 4)

    def _tiles(self, fr: np.ndarray, sim: OverflowSim):
        cv2 = self.cv2
        for tl in sim.tiles:
            pts = _tile_poly(tl["k"])
            a = 2 * math.pi * (tl["k"] + 0.5) / TILES
            cx0, cy0 = CX + (R + 5) * math.cos(a), CY + (R + 5) * math.sin(a)
            c, s = math.cos(tl["rot"]), math.sin(tl["rot"])
            out = []
            for x, y in pts:
                dx, dy = x - cx0, y - cy0
                out.append((int((tl["x"] + dx * c - dy * s) * 16), int((tl["y"] + dx * s + dy * c) * 16)))
            p = np.array(out, np.int32)
            cv2.fillPoly(fr, [p], TILE, cv2.LINE_AA, 4)
            cv2.polylines(fr, [p], True, TILE_EDGE, 2, cv2.LINE_AA, 4)

    def frame(self, sim: OverflowSim) -> Image.Image:
        if sim.tiles:
            if self.bg_open is None:
                self.bg_open = np.asarray(self._background(self._back, self._inner, rim=False)).copy()
            fr = self.bg_open.copy()
        else:
            fr = self.bg_np.copy()
        self._balls(fr, sim.xy, sim.col)
        self._paddle(fr, sim.u_p)
        self._pops(fr, sim)
        if sim.tiles:
            self._tiles(fr, sim)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        n = len(sim.bodies) if sim.done_at is None else (sim.n_final or len(sim.bodies))
        n = max(n, 1)
        d.text((W // 2, 430), f"{n:,}", font=self.f_count, fill=(255, 255, 255), anchor="mm")
        d.text((W // 2 + d.textlength(f"{n:,}", font=self.f_count) / 2 + 14, 444), "ball" if n == 1 else "balls",
               font=self.f_small, fill=(168, 180, 206), anchor="lm")
        if sim.done_at is not None and sim.ok:
            age = sim.t - sim.done_at
            if age > 0.08:
                k = min(1.0, (age - 0.08) / 0.22)
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
            "description": ("One ball bounces around the circle. A plain hit is just a bounce, but when the paddle "
                            "squishes a ball against the floor, it pops out as two on the other side. When the "
                            "circle is packed, the rim breaks.\n\n"
                            "Did you think it would overflow? 👇\n\n#satisfying #asmr #physics #simulation")}
