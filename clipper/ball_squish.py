"""Squish machines (Nov 2026): the overflow bowl's rule in three new machines.

Dean liked the overflow bowl ("it looks good") and picked three more ways to
squish balls from a board of six drawings: a hydraulic press, a gear grinder
and a syringe. The rule stays his: the balls never stop bouncing (no gravity,
steady speed, a tiny random turn) and a plain hit is only a bounce; a ball
caught where the machine pinches pops out as TWO balls on the other side. When
the machine is packed its walls break and the balls burst out.

- ``press``: a box with a ruler-marked plate on a rod going up and down. The
  plate is narrower than the box, so balls slip round its ends and only the
  ones caught under (or over) it at each stroke are squished.
- ``syringe``: a tall tube with a plunger pumping up and down inside it, a cap
  on top; same rule, its cap and walls fly off at the end.
- ``gears``: two big gears side by side, turning inward, the only way between
  the top and the bottom half is the gap where their teeth meet. A ball that
  goes through it is squished and comes out the other side as two.

Sims subclass ``ball_overflow.OverflowSim`` (ball bodies, sound, milestones)
and painters subclass ``OverflowPainter`` (the ball splat, pops, the five
looks in ``STYLES``). ``NO_PACK`` kinds: ``ball_evolution._parts`` dispatches
here.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

from clipper import ball_overflow as ov
from clipper.ball_evolution import FPS, H, W, _font

KINDS = ("press", "gears", "syringe")
END_HOLD = ov.END_HOLD
SECONDS_CAP = 58.0
SPEED = 640.0
SQUISH_K = 1.6
FILL = 0.40
START_N = 12
CAP = 2000
WALL = 14                    # wall half-thickness (physics) and drawn tile depth
PT = 30                      # press plate / plunger thickness

# arenas: inner box (x0, x1, y0, y1)
BOXES = {"press": (150, 930, 560, 1640), "syringe": (330, 750, 540, 1660), "gears": (120, 960, 540, 1660)}
PLATE_SHARE = {"press": 0.62, "syringe": 0.70}     # plate width / box width: balls slip round the ends
PERIOD = {"press": 2.4, "syringe": 2.0}            # s for a full down-and-up
GEAR_Y = 1100
NIP = 22                                           # px between the two gears' rims
GEAR_TEETH = 18
GEAR_PULL = 0.10                                   # how hard a ball near a gear is turned toward the gap

HOOKS = {"press": ["Will the press crush them all?", "Every press makes more", "How many until the box breaks?"],
         "syringe": ["Will it overflow the tube?", "Every push makes more", "How many until the cap pops?"],
         "gears": ["How many come out of the gears?", "Through the gears = 2 balls",
                   "How many until it breaks?"]}
SUBS = {"press": "squish = 2 balls", "syringe": "pressed = 2 balls", "gears": "through the gears = 2"}
ENDINGS = {"press": ["CRUSHED!", "IT BROKE!", "TOO FULL!"], "syringe": ["OVERFLOW!", "IT POPPED!", "TOO FULL!"],
           "gears": ["JAMMED!", "IT BROKE!", "TOO FULL!"]}


def squish_recipe(recipe: dict) -> dict:
    fmt = recipe["format"]
    rng = random.Random(f"squish-recipe-{recipe['seed']}")
    recipe.update(hook=rng.choice(HOOKS[fmt]), sub=SUBS[fmt], palette=rng.randrange(len(ov.PALETTES)),
                  back=rng.randrange(len(ov.BACKS)), speed=round(rng.uniform(0.92, 1.1), 3),
                  ball_r=rng.choice((8.0, 8.5, 9.0)), spin=rng.choice((1, -1)),
                  ending=rng.choice(ENDINGS[fmt]), bed="off")
    recipe["style"] = random.Random(f"squish-style-{recipe['seed']}").choice(ov.STYLE_NAMES)
    return recipe


def gear_centres():
    x0, x1, _, _ = BOXES["gears"]
    gr = ((x1 - x0) / 2 - NIP / 2) / 2
    mid = (x0 + x1) / 2
    return gr, (mid - NIP / 2 - gr, GEAR_Y), (mid + NIP / 2 + gr, GEAR_Y)


def plate_y(recipe: dict, t: float):
    """Top of the plate (y) and its speed (px/s). The press starts high and
    comes down first; the plunger starts low and goes up first."""
    fmt = recipe["format"]
    x0, x1, y0, y1 = BOXES[fmt]
    lo, hi = y0 + 6, y1 - PT - 6                       # top and bottom of its travel (the plate's top)
    T = PERIOD[fmt] / float(recipe.get("speed") or 1.0)
    v = (t / T + (0.0 if fmt == "press" else 0.5)) % 1.0
    x = v * 2 if v < 0.5 else 2 - v * 2               # 0 -> 1 -> 0
    s = 0.5 - 0.5 * math.cos(math.pi * x)
    ds = 0.5 * math.pi * math.sin(math.pi * x) * (2 if v < 0.5 else -2) / T
    return lo + (hi - lo) * s, (hi - lo) * ds


def full_count(fmt: str, r: float) -> int:
    x0, x1, y0, y1 = BOXES[fmt]
    area = (x1 - x0) * (y1 - y0)
    if fmt == "gears":
        gr = gear_centres()[0]
        area -= 2 * math.pi * gr * gr
    else:
        area -= (x1 - x0) * PLATE_SHARE[fmt] * PT
    return min(CAP - 50, int(area * FILL / (math.pi * r * r)))


class SquishSim(ov.OverflowSim):
    def __init__(self, recipe: dict):
        import pymunk

        self.pm = pymunk
        self.recipe = recipe
        self.fmt = recipe["format"]
        self.rng = random.Random(f"squish-{recipe['seed']}")
        self.r = float(recipe.get("ball_r") or 8.5)
        self.full = full_count(self.fmt, self.r)
        self.ncol = len(ov._palette(recipe))
        self.spd = SPEED * float(recipe.get("speed") or 1.0)
        self.t = 0.0
        self.chain = [(None, "start"), (None, "10 balls"), (None, "100 balls"), (None, "1,000 balls"),
                      (None, "full")]
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.done_at = None
        self.max_gap = 0.0
        self.ok = False
        self.events, self.hits, self.squishes, self.pops = [], [], [], []
        self.tiles: list = []
        self.shards: list = []
        self.n_final = 0
        self.u_p = 0.0
        self.angle = 0.0                 # the gears' turn (drawing)

        sp = self.space = pymunk.Space()
        sp.gravity = (0, 0)
        sp.iterations = 8
        sp.use_spatial_hash(2 * self.r + 2, 8000)
        x0, x1, y0, y1 = BOXES[self.fmt]
        self.walls = []
        for a, b in (((x0 - WALL, y0 - WALL), (x1 + WALL, y0 - WALL)), ((x1 + WALL, y0 - WALL), (x1 + WALL, y1 + WALL)),
                     ((x1 + WALL, y1 + WALL), (x0 - WALL, y1 + WALL)), ((x0 - WALL, y1 + WALL), (x0 - WALL, y0 - WALL))):
            s = pymunk.Segment(sp.static_body, a, b, WALL)
            s.elasticity, s.friction = 1.0, 0.0
            self.walls.append(s)
            sp.add(s)
        self.plate = None
        if self.fmt == "gears":
            gr, c1, c2 = gear_centres()
            self.gr, self.gears = gr, (c1, c2)
            for c in (c1, c2):
                s = pymunk.Circle(sp.static_body, gr, c)
                s.elasticity, s.friction = 1.0, 0.0
                sp.add(s)
        else:
            w = (x1 - x0) * PLATE_SHARE[self.fmt]
            self.pw = w
            self.plate = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            shape = pymunk.Poly.create_box(self.plate, (w, PT))
            shape.elasticity, shape.friction = 1.0, 0.0
            sp.add(self.plate, shape)
            self._set_plate(0.0)

        self.bodies, self.cols = [], []
        self.xy = np.zeros((0, 2), np.float32)
        self.col = np.zeros(0, np.int16)
        for k in range(START_N):
            while True:
                x, y = self.rng.uniform(x0 + 30, x1 - 30), self.rng.uniform(y0 + 30, y1 - 30)
                if not self._blocked(x, y):
                    break
            a = self.rng.uniform(0, 2 * math.pi)
            self._add(x, y, (self.spd * math.cos(a), self.spd * math.sin(a)), k % self.ncol)
        self.spawned = START_N
        self._sync()

    # ---- geometry -------------------------------------------------------
    def _blocked(self, x, y) -> bool:
        pad = self.r + 4
        if self.fmt == "gears":
            return any(math.hypot(x - cx, y - cy) < self.gr + pad for cx, cy in self.gears)
        top = self.py
        return top - pad < y < top + PT + pad and abs(x - (BOXES[self.fmt][0] + BOXES[self.fmt][1]) / 2) < self.pw / 2 + pad

    def _set_plate(self, t: float):
        y, vy = plate_y(self.recipe, t)
        self.py, self.vy = y, vy
        x0, x1, _, _ = BOXES[self.fmt]
        self.plate.position = ((x0 + x1) / 2, y + PT / 2)
        self.plate.velocity = (0, vy)
        self.space.reindex_shapes_for_body(self.plate)

    # ---- the rule -------------------------------------------------------
    def _born(self, caught, xy, place):
        """Replace each caught ball by two from ``place(i, k)`` -> (x, y, vx, vy)."""
        born = []
        for i in caught:
            if len(self.bodies) - len(caught) + len(born) + 2 > CAP:
                break
            for k in range(2):
                x, y, vx, vy = place(int(i), k)
                born.append((x, y, (vx, vy), self.cols[i] if k == 0 else self.rng.randrange(self.ncol)))
            self.pops.append((self.t, float(xy[i, 0]), float(xy[i, 1])))
            self.hits.append((self.t, 1.0 / (1.0 + len(self.bodies) / 300.0)))
        n = len(born) // 2
        if not n:
            return
        self._remove([int(i) for i in caught[:n]])
        for x, y, v, c in born:
            self._add(x, y, v, c)
        self.spawned += n
        self.squishes.append((self.t, len(self.bodies)))

    def _squish(self):
        if self.done_at is not None or not self.bodies:
            return
        xy = np.array([tuple(b.position) for b in self.bodies], np.float64).reshape(-1, 2)
        x0, x1, y0, y1 = BOXES[self.fmt]
        r = self.r
        spd = self.spd * 1.1
        rng = self.rng
        if self.fmt == "gears":
            (c1x, cy), (c2x, _) = self.gears
            mid = (c1x + c2x) / 2
            caught = np.nonzero((np.abs(xy[:, 0] - mid) < NIP / 2 + r * 0.6) & (np.abs(xy[:, 1] - cy) < r * 1.2))[0]

            def place(i, k):
                down = 1.0 if self.bodies[i].velocity.y >= 0 else -1.0
                y = cy + down * (r * 3.2 + k * 2.2 * r)
                a = math.pi / 2 * down + rng.uniform(-0.7, 0.7)
                return mid + rng.uniform(-r, r), y, spd * math.cos(a), spd * math.sin(a)
            self._born(caught, xy, place)
            return
        if abs(self.vy) < 1:
            return
        cx = (x0 + x1) / 2
        under = np.abs(xy[:, 0] - cx) < self.pw / 2 + r * 0.3
        top, bot = self.py, self.py + PT
        if self.vy > 0:        # coming down: the gap under the plate
            caught = np.nonzero(under & (xy[:, 1] > bot - 2) & ((y1 - bot) < 2 * r * SQUISH_K))[0]
            back, sgn = top, -1.0
        else:                  # going up: the gap over it
            caught = np.nonzero(under & (xy[:, 1] < top + 2) & ((top - y0) < 2 * r * SQUISH_K))[0]
            back, sgn = bot, 1.0

        def place(i, k):
            x = float(min(max(xy[i, 0] + (k - 0.5) * 2.4 * r, x0 + r + 2), x1 - r - 2))
            y = back + sgn * (r + 3 + k * 0.8 * r)
            a = math.atan2(sgn, 0) + rng.uniform(-0.8, 0.8)
            return x, y, spd * math.cos(a), spd * math.sin(a)
        self._born(caught, xy, place)

    # ---- every frame ----------------------------------------------------
    def step(self):
        dt = 1 / FPS
        sub = 3
        h = dt / sub
        for k in range(sub):
            if self.plate is not None:
                self._set_plate(self.t + (k + 1) * h)
            self._squish()
            self.space.step(h)
        self.t += dt
        self.angle += dt * 1.4 * float(self.recipe.get("speed") or 1.0)
        if self.done_at is None:
            self._keep_inside()
        else:
            self._fly(dt)
        self._sync()
        self._milestones()

    def _keep_inside(self):
        if not self.bodies:
            return
        x0, x1, y0, y1 = BOXES[self.fmt]
        r = self.r
        lo, hi = self.spd * 0.8, self.spd * 1.25
        cxm = (x0 + x1) / 2
        for b in self.bodies:
            vx, vy = b.velocity
            s = math.hypot(vx, vy)
            if s < 1:
                a = self.rng.uniform(0, 2 * math.pi)
                vx, vy, s = math.cos(a), math.sin(a), 1.0
            turn = self.rng.uniform(-0.04, 0.04)
            x, y = b.position
            if self.fmt == "gears":
                # near a gear: the teeth turn it toward the gap between them
                for (gx, gy), side in zip(self.gears, (1.0, -1.0)):
                    dd = math.hypot(x - gx, y - gy)
                    if dd < self.gr + 4 * r:
                        want = math.atan2(GEAR_Y - y, cxm - x)
                        have = math.atan2(vy, vx)
                        turn += ((want - have + math.pi) % (2 * math.pi) - math.pi) * GEAR_PULL
            k = min(max(s, lo), hi) / s
            c, sn = math.cos(turn), math.sin(turn)
            b.velocity = ((vx * c - vy * sn) * k, (vx * sn + vy * c) * k)
            nx = min(max(x, x0 + r + 1), x1 - r - 1)
            ny = min(max(y, y0 + r + 1), y1 - r - 1)
            if self.fmt == "gears":
                for gx, gy in self.gears:
                    dd = math.hypot(nx - gx, ny - gy)
                    if dd < self.gr + r - 1:
                        f = (self.gr + r + 1) / max(dd, 1e-6)
                        nx, ny = gx + (nx - gx) * f, gy + (ny - gy) * f
            elif abs(nx - cxm) < self.pw / 2 and self.py - r + 1 < ny < self.py + PT + r - 1:
                # inside the plate: out on the side it's moving away from
                ny = self.py - r - 1 if self.vy >= 0 else self.py + PT + r + 1
                ny = min(max(ny, y0 + r + 1), y1 - r - 1)
            if (nx, ny) != (x, y):
                b.position = (nx, ny)

    def _shatter(self):
        """The walls break into pieces that fly off; the balls burst out."""
        for s in self.walls:
            self.space.remove(s)
        self.walls = []
        if self.plate is not None:
            self.space.remove(self.plate, *self.plate.shapes)
        x0, x1, y0, y1 = BOXES[self.fmt]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rng = self.rng
        for (ax, ay), (bx, by) in _wall_lines(self.fmt):
            n = max(2, int(math.hypot(bx - ax, by - ay) / 70))
            for i in range(n):
                px, py = ax + (bx - ax) * (i + 0.5) / n, ay + (by - ay) * (i + 0.5) / n
                dx, dy = px - cx, py - cy
                dd = math.hypot(dx, dy) or 1.0
                sp = rng.uniform(380, 820)
                self.shards.append({"x": px, "y": py, "len": math.hypot(bx - ax, by - ay) / n - 4,
                                    "a": math.atan2(by - ay, bx - ax), "vx": sp * dx / dd + rng.uniform(-80, 80),
                                    "vy": sp * dy / dd - rng.uniform(100, 300), "vr": rng.uniform(-7, 7)})
        self.tiles = self.shards        # OverflowSim._fly moves "tiles"; ours fly the same way
        for tl in self.shards:
            tl["rot"] = 0.0
        for b in self.bodies:
            dx, dy = b.position.x - cx, b.position.y - cy
            dd = math.hypot(dx, dy) or 1.0
            out = rng.uniform(500, 1000)
            b.velocity = (b.velocity.x * 0.4 + out * dx / dd, b.velocity.y * 0.4 + out * dy / dd)


def _wall_lines(fmt):
    x0, x1, y0, y1 = BOXES[fmt]
    o = WALL
    return [((x0 - o, y0 - o), (x1 + o, y0 - o)), ((x1 + o, y0 - o), (x1 + o, y1 + o)),
            ((x1 + o, y1 + o), (x0 - o, y1 + o)), ((x0 - o, y1 + o), (x0 - o, y0 - o))]


class SquishPainter(ov.OverflowPainter):
    def __init__(self, recipe: dict):
        self.fmt = recipe["format"]
        super().__init__(recipe)

    def _background(self, back, inner, rim: bool) -> Image.Image:
        S = self.SS
        st = self.st
        im = Image.new("RGB", (W * S, H * S), back)
        d = ImageDraw.Draw(im)
        x0, x1, y0, y1 = BOXES[self.fmt]

        def P(x, y):
            return (x * S, y * S)
        d.rectangle([*P(x0, y0), *P(x1, y1)], fill=inner)
        if self.fmt == "syringe":           # measuring marks along the tube
            for k, y in enumerate(range(y1 - 60, y0 + 20, -70)):
                w = 46 if k % 2 == 0 else 26
                d.line([P(x1 - w, y), P(x1 - 4, y)], fill=st["groove"], width=4 * S)
        if self.fmt == "press":             # the press frame above the box
            d.rectangle([*P(x0 - 40, y0 - 64), *P(x1 + 40, y0 - 34)], fill=st["plate"], outline=st["plate_edge"],
                        width=3 * S)
        if rim:
            for (ax, ay), (bx, by) in _wall_lines(self.fmt):
                n = max(2, int(math.hypot(bx - ax, by - ay) / 70))
                for i in range(n):
                    p0 = (ax + (bx - ax) * i / n, ay + (by - ay) * i / n)
                    p1 = (ax + (bx - ax) * (i + 1) / n, ay + (by - ay) * (i + 1) / n)
                    d.line([P(*p0), P(*p1)], fill=st["tile_edge"], width=(2 * WALL + 2) * S)
                    d.line([P(*p0), P(*p1)], fill=st["tile"], width=(2 * WALL - 4) * S)
                    d.ellipse([*P(p1[0] - 2, p1[1] - 2), *P(p1[0] + 2, p1[1] + 2)], fill=back)
            if self.fmt == "syringe":       # the cap
                d.rounded_rectangle([*P(x0 - 40, y0 - WALL - 26), *P(x1 + 40, y0 - WALL + 4)], 10 * S,
                                    fill=st["paddle"][0], outline=st["paddle"][1], width=4 * S)
        im = im.resize((W, H), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        hook = self.recipe.get("hook") or HOOKS[self.fmt][0]
        size = 92
        while size > 50 and d.textlength(hook, font=_font(size)) > W - 100:
            size -= 4
        d.text((W // 2, 200), hook, font=_font(size), fill=st["title"], anchor="mm")
        sub = self.recipe.get("sub") or SUBS[self.fmt]
        if "=" in hook or "doubles" in hook:          # the title already says the rule
            sub = "how many until it breaks?"
        d.text((W // 2, 292), sub, font=_font(46, "Inter-Bold.ttf"),
               fill=st["sub"], anchor="mm")
        return im

    def _machine(self, fr: np.ndarray, sim: SquishSim):
        cv2 = self.cv2
        st = self.st
        sh = 16

        def q(x, y):
            return (int(x * sh), int(y * sh))
        x0, x1, y0, y1 = BOXES[self.fmt]
        if self.fmt == "gears":
            for (gx, gy), dirn in zip(sim.gears, (1.0, -1.0)):
                rot = sim.angle * dirn * int(self.recipe.get("spin") or 1)
                pts = []
                n = GEAR_TEETH * 2
                for i in range(n * 2):
                    a = rot + math.pi * i / n
                    rr = sim.gr if (i // 2) % 2 == 0 else sim.gr - 22
                    pts.append(q(gx + rr * math.cos(a), gy + rr * math.sin(a)))
                p = np.array(pts, np.int32)
                cv2.fillPoly(fr, [p], st["paddle"][0], cv2.LINE_AA, 4)
                cv2.polylines(fr, [p], True, st["paddle"][1], 3, cv2.LINE_AA, 4)
                cv2.circle(fr, q(gx, gy), int(sim.gr * 0.55 * sh), st["paddle"][2], 3, cv2.LINE_AA, 4)
                for k in range(5):
                    a = rot + 2 * math.pi * k / 5
                    cv2.line(fr, q(gx + 30 * math.cos(a), gy + 30 * math.sin(a)),
                             q(gx + sim.gr * 0.55 * math.cos(a), gy + sim.gr * 0.55 * math.sin(a)),
                             st["paddle"][2], 6, cv2.LINE_AA, 4)
                cv2.circle(fr, q(gx, gy), 26 * sh, st["paddle"][1], -1, cv2.LINE_AA, 4)
                cv2.circle(fr, q(gx, gy), 12 * sh, st["hub"], -1, cv2.LINE_AA, 4)
            return
        if sim.done_at is not None:
            return
        cx = (x0 + x1) / 2
        top = sim.py
        # the rod: from above for the press, from below for the plunger
        if self.fmt == "press":
            cv2.rectangle(fr, q(cx - 20, y0 - 60), q(cx + 20, top), st["paddle"][1], -1, cv2.LINE_AA, 4)
        else:
            cv2.rectangle(fr, q(cx - 16, top + PT), q(cx + 16, H - 40), st["paddle"][1], -1, cv2.LINE_AA, 4)
            cv2.rectangle(fr, q(cx - 70, H - 70), q(cx + 70, H - 40), st["paddle"][1], -1, cv2.LINE_AA, 4)
        p0, p1 = q(cx - sim.pw / 2, top), q(cx + sim.pw / 2, top + PT)
        cv2.rectangle(fr, p0, p1, st["paddle"][0], -1, cv2.LINE_AA, 4)
        cv2.rectangle(fr, p0, p1, st["paddle"][1], 3, cv2.LINE_AA, 4)
        for k in range(1, 6):
            x = cx - sim.pw / 2 + sim.pw * k / 6
            cv2.line(fr, q(x, top + 8), q(x, top + PT - 8), st["paddle"][2], 4, cv2.LINE_AA, 4)

    def _shards(self, fr: np.ndarray, sim: SquishSim):
        cv2 = self.cv2
        for tl in sim.shards:
            a = tl["a"] + tl["rot"]
            hx, hy = math.cos(a) * tl["len"] / 2, math.sin(a) * tl["len"] / 2
            p0 = (int((tl["x"] - hx) * 16), int((tl["y"] - hy) * 16))
            p1 = (int((tl["x"] + hx) * 16), int((tl["y"] + hy) * 16))
            cv2.line(fr, p0, p1, self.st["tile_edge"], 2 * WALL + 2, cv2.LINE_AA, 4)
            cv2.line(fr, p0, p1, self.st["tile"], 2 * WALL - 4, cv2.LINE_AA, 4)

    def frame(self, sim: SquishSim) -> Image.Image:
        if sim.shards:
            if self.bg_open is None:
                self.bg_open = np.asarray(self._background(self._back, self._inner, rim=False)).copy()
            fr = self.bg_open.copy()
        else:
            fr = self.bg_np.copy()
        self._balls(fr, sim.xy, sim.col)
        self._machine(fr, sim)
        self._pops(fr, sim)
        if sim.shards:
            self._shards(fr, sim)
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        n = len(sim.bodies) if sim.done_at is None else (sim.n_final or len(sim.bodies))
        n = max(n, 1)
        d.text((W // 2, 430), f"{n:,}", font=self.f_count, fill=self.st["count"], anchor="mm")
        d.text((W // 2 + d.textlength(f"{n:,}", font=self.f_count) / 2 + 14, 444), "ball" if n == 1 else "balls",
               font=self.f_small, fill=self.st["sub"], anchor="lm")
        if sim.done_at is not None and sim.ok:
            age = sim.t - sim.done_at
            if age > 0.08:
                k = min(1.0, (age - 0.08) / 0.22)
                text = self.recipe.get("ending") or ENDINGS[self.fmt][0]
                full = 132
                while full > 64 and d.textlength(text, font=_font(full)) > W - 110:
                    full -= 6
                sz = int(full * (0.53 + 0.47 * (1 - (1 - k) ** 3)) + 10 * math.sin(min(1.0, k) * math.pi))
                f = self._ends.setdefault(sz, _font(sz))
                x0, x1, y0, y1 = BOXES[self.fmt]
                d.text((W // 2, (y0 + y1) // 2), text, font=f, fill=self.st["ending"],
                       anchor="mm", stroke_width=max(4, sz // 14), stroke_fill=self.st["ending_stroke"])
        return im


_DESC = {"press": "A press goes up and down in a box of bouncing balls. A plain hit is just a bounce, but a ball "
                  "squished under the plate pops out as two on the other side. When the box is packed, it breaks.",
         "syringe": "A plunger pumps up and down a tube of bouncing balls. A ball pressed against the cap or the "
                    "bottom pops out as two on the other side. When the tube is packed, the cap pops.",
         "gears": "Two gears turn in a box of bouncing balls. The only way through is the gap where their teeth "
                  "meet, and every ball that squeezes through comes out as two. When the box is packed, it breaks."}


def publish_text(recipe: dict) -> dict:
    return {"title": recipe["hook"][:100],
            "description": _DESC[recipe["format"]] + "\n\nGuess how many balls it takes 👇\n\n"
                           "#satisfying #asmr #physics #simulation"}
