"""Ball Evolution: a satisfying physics Short, rendered from scratch.

Small items drip slowly out of a hole at the top and fall through a
"course". Hitting a gold peg sends a copy back out of the hole, so the drip
turns into a flood. In the jar at the bottom, two of the same item that
touch merge into the next, bigger one -- bee > mouse > ... > unicorn, or
cookie > donut > ... > cake, and so on. The ladder at the top hides each
item until it's first made; the video ends on the last one.

Every video is a different *recipe* so a daily channel doesn't post the
same video twice (YouTube won't monetize template-looking repeats):

* theme -- the evolution chain, its colours, hook line and starter item
* course -- how the drop is built: peg grid, triangle, spinners, ramps,
  bumpers (layout details also vary with the seed)
* jar -- box, bowl or flask
* sound -- soft noise-based ASMR: every bounce a marble-like tap, merges
  a puff that deepens with size, reveals a whoosh, over a quiet bed of
  noise (rain, air or hush, picked per video)

Everything on screen and every sound is made here (bundled MIT emoji, OFL
fonts, generated audio), so there is nothing to license or get claimed.

Runs differ by seed and some stall, so ``pick_seed`` simulates candidates
without drawing (fast, in parallel) and keeps one that reaches the last
item at a good pace. ``pick_recipe`` avoids themes and combinations used
recently (a small JSON history file).

    python -m clipper.ball_evolution out.mp4                 # new recipe, good seed
    python -m clipper.ball_evolution out.mp4 --theme food --course ramps --jar bowl
    python -m clipper.ball_evolution out.mp4 --history runs.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 60, 44100
ASSETS = Path(__file__).parent / "assets"

# ---- fixed frame of the machine ----------------------------------------
HOLE = (540, 330)
WALL_L, WALL_R = 70, 1010
COURSE_TOP, COURSE_BOT = 400, 1030
NECK_L, NECK_R, NECK_Y = 400, 680, 1215
JAR_L, JAR_R, JAR_TOP, JAR_BOT = 110, 970, 1320, 1830
PEG_R = 9
GRAVITY = 520            # low gravity: a slow, readable fall
MAX_ABOVE_NECK = 90      # the hole waits while this many are still falling
SPAWN_CAP = 2500
PULL_FROM_TIER = 2       # third item and up attract their match in the jar
PULL = 900.0             # px/s^2, a bit more than gravity
SECONDS_CAP = 90.0
END_HOLD = 4.0
MAX_UNLOCK_GAP = 18.0

# ---- themes: (emoji file, name) small to big ---------------------------
THEMES = {
    "animals": {
        "chain": [("1f41d", "Bee"), ("1f400", "Mouse"), ("1f438", "Frog"),
                  ("1f414", "Chicken"), ("1f431", "Cat"), ("1f436", "Dog"),
                  ("1f43c", "Panda"), ("1f981", "Lion"), ("1f984", "Unicorn")],
        "unit": "bees", "bg": ((14, 12, 30), (34, 18, 52)),
        "hooks": ["Can a bee become the last animal?", "Can it make the last one?"],
    },
    "sports": {
        "chain": [("26be", "Baseball"), ("1f3be", "Tennis ball"), ("1f3d0", "Volleyball"),
                  ("26bd", "Football"), ("1f3c0", "Basketball"), ("1f3c8", "Rugby ball"),
                  ("1f3b3", "Bowling"), ("1f947", "Gold medal"), ("1f3c6", "Trophy")],
        "unit": "baseballs", "bg": ((8, 26, 20), (14, 52, 36)),
        "hooks": ["Can a baseball win the trophy?", "Will it reach the last ball?"],
    },
    "food": {
        "chain": [("1f36a", "Cookie"), ("1f369", "Donut"), ("1f37f", "Popcorn"),
                  ("1f35f", "Fries"), ("1f32d", "Hot dog"), ("1f354", "Burger"),
                  ("1f355", "Pizza"), ("1f382", "Cake")],
        "unit": "cookies", "bg": ((36, 14, 12), (64, 28, 20)),
        "hooks": ["Can cookies become the final food?", "What's the last food?"],
    },
    "space": {
        "chain": [("2728", "Stardust"), ("2b50", "Star"), ("1f31f", "Bright star"),
                  ("1f319", "Moon"), ("1f30d", "Earth"), ("2600-fe0f", "Sun"),
                  ("1f680", "Rocket"), ("1f6f8", "UFO"), ("1f47d", "Alien")],
        "unit": "sparks", "bg": ((4, 6, 22), (16, 20, 58)),
        "hooks": ["Can stardust make the last one?", "What's at the end of space?"],
    },
    "money": {
        "chain": [("1fa99", "Coin"), ("1f4b5", "Cash"), ("1f4b8", "Flying cash"),
                  ("1f4b3", "Card"), ("1f4b0", "Money bag"), ("1f48e", "Diamond"),
                  ("1f451", "Crown"), ("1f911", "Rich")],
        "unit": "coins", "bg": ((6, 22, 14), (22, 44, 22)),
        "hooks": ["Can one coin make you rich?", "Coin to... what?"],
    },
    "laughs": {
        "chain": [("1f610", "Meh"), ("1f642", "Smile"), ("1f60a", "Happy"),
                  ("1f604", "Grin"), ("1f606", "Laugh"), ("1f602", "Crying laughing"),
                  ("1f923", "Rolling"), ("1f929", "Starstruck"), ("1f973", "Party")],
        "unit": "faces", "bg": ((10, 24, 34), (16, 44, 60)),
        "hooks": ["Can a meh face become the happiest?", "How happy can it get?"],
    },
    "vehicles": {
        "chain": [("1f697", "Car"), ("1f693", "Police car"), ("1f68c", "Bus"),
                  ("1f682", "Train"), ("1f3ce-fe0f", "Race car"), ("2708-fe0f", "Plane"),
                  ("1f680", "Rocket"), ("1f6f8", "UFO")],
        "unit": "cars", "bg": ((16, 18, 28), (34, 38, 56)),
        "hooks": ["Can a car become the fastest thing?", "Car to... what?"],
    },
    "weather": {
        "chain": [("1f4a7", "Drop"), ("1f4a6", "Splash"), ("1f327-fe0f", "Rain"),
                  ("26c8-fe0f", "Storm"), ("26a1", "Lightning"), ("1f32a-fe0f", "Tornado"),
                  ("1f30a", "Wave"), ("1f308", "Rainbow"), ("2600-fe0f", "Sun")],
        "unit": "drops", "bg": ((6, 16, 34), (12, 34, 64)),
        "hooks": ["Can one drop make the sun?", "What's after the storm?"],
    },
}
COURSES = ["pegs", "triangle", "spinners", "ramps", "bumpers"]
JARS = ["box", "bowl", "flask"]
ENDINGS = ["EVOLVED!", "FINAL FORM!", "MAXED OUT!"]

KEYS = {"C": 261.63, "D": 293.66, "Eb": 311.13, "F": 349.23, "G": 392.00, "A": 220.00 * 2}
SCALES = {"major": (0, 2, 4, 7, 9), "minor": (0, 3, 5, 7, 10), "dreamy": (0, 2, 4, 7, 11)}


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


def _font(size: int, name: str = "Inter-Black.ttf"):
    try:
        return ImageFont.truetype(str(ASSETS / "fonts" / name), size)
    except OSError:
        return ImageFont.load_default()


# ---- recipes -----------------------------------------------------------

def make_recipe(seed: int, theme=None, course=None, jar=None) -> dict:
    """Everything that makes one video different, decided up front."""
    rng = random.Random(f"recipe-{seed}")
    theme = theme or rng.choice(sorted(THEMES))
    scale = rng.choice(sorted(SCALES))
    return {
        "seed": seed, "theme": theme,
        "course": course or rng.choice(COURSES),
        "jar": jar or rng.choice(JARS),
        "hook": rng.choice(THEMES[theme]["hooks"]),
        "ending": rng.choice(ENDINGS),
        "key": rng.choice(sorted(KEYS)), "scale": scale,
        "bpm": rng.choice([88, 96, 100, 108]),
    }


def _load_history(path) -> list:
    try:
        return json.loads(Path(path).read_text()) if path else []
    except (OSError, ValueError):
        return []


def pick_recipe(seed: int, history: list, theme=None, course=None, jar=None) -> dict:
    """A recipe whose theme wasn't used in the last 3 videos and whose
    theme+course+jar combination hasn't been used at all, where possible."""
    recent = [h.get("theme") for h in history[-3:]]
    combos = {(h.get("theme"), h.get("course"), h.get("jar")) for h in history}
    best = None
    for k in range(200):
        r = make_recipe(seed * 1000 + k, theme, course, jar)
        r["seed"] = seed
        fresh = (r["theme"], r["course"], r["jar"]) not in combos
        if fresh and (theme or r["theme"] not in recent):
            return r
        best = best or r
    return best


# ---- the course ---------------------------------------------------------

def _seg(space, static, a, b, r=6, kind="wall", elasticity=0.35, friction=0.5):
    import pymunk
    s = pymunk.Segment(static, a, b, r)
    s.elasticity, s.friction = elasticity, friction
    s.collision_type = 3
    s.part = kind
    space.add(s)
    return s


def _jar_points(jar: str) -> list:
    if jar == "bowl":
        pts = [(NECK_L, NECK_Y), (JAR_L, JAR_TOP), (JAR_L, JAR_BOT - 260)]
        cx, rx, ry = (JAR_L + JAR_R) / 2, (JAR_R - JAR_L) / 2, 260
        for i in range(1, 16):
            a = math.pi - math.pi * i / 16
            pts.append((cx + rx * math.cos(a), JAR_BOT - 260 + ry * math.sin(a)))
        pts += [(JAR_R, JAR_BOT - 260), (JAR_R, JAR_TOP), (NECK_R, NECK_Y)]
        return pts
    if jar == "flask":
        return [(NECK_L, NECK_Y), (260, JAR_TOP - 20), (80, JAR_BOT), (1000, JAR_BOT),
                (820, JAR_TOP - 20), (NECK_R, NECK_Y)]
    return [(NECK_L, NECK_Y), (JAR_L, JAR_TOP), (JAR_L, JAR_BOT), (JAR_R, JAR_BOT),
            (JAR_R, JAR_TOP), (NECK_R, NECK_Y)]


def _build(recipe: dict):
    import pymunk

    rng = random.Random(recipe["seed"])
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.iterations = 20
    static = space.static_body
    parts = []      # drawables: static segments
    pegs = []       # {"shape","pos","r","gold","flash","note"}
    spinners = []   # {"body","half"}

    for a, b in [((WALL_L, COURSE_TOP - 20), (WALL_L, COURSE_BOT)),
                 ((WALL_R, COURSE_TOP - 20), (WALL_R, COURSE_BOT)),
                 ((WALL_L, COURSE_BOT), (NECK_L, NECK_Y)), ((WALL_R, COURSE_BOT), (NECK_R, NECK_Y))]:
        parts.append(_seg(space, static, a, b))
    jar = _jar_points(recipe["jar"])
    for a, b in zip(jar, jar[1:]):
        parts.append(_seg(space, static, a, b, kind="jar"))

    def peg(x, y, gold_share, r=PEG_R, elasticity=0.55):
        # a peg too close to a wall makes a pocket narrower than a ball,
        # where one can rest forever; leave those out
        room = 6 + r + 38
        if not WALL_L + room <= x <= WALL_R - room:
            return
        p = pymunk.Circle(static, r, (x, y))
        p.elasticity, p.friction = elasticity, 0.3
        p.collision_type = 2
        space.add(p)
        note = int((x - WALL_L) / (WALL_R - WALL_L) * 10)     # left low, right high
        pegs.append({"shape": p, "pos": (x, y), "r": r, "gold": rng.random() < gold_share,
                     "flash": 0.0, "note": max(0, min(9, note))})

    course = recipe["course"]
    if course == "pegs":
        rows, gy, gx = rng.choice([7, 8, 9]), rng.choice([64, 72, 78]), rng.choice([84, 92, 100])
        share = rng.uniform(0.28, 0.32)
        for row in range(rows):
            y = 450 + row * gy
            if y > COURSE_BOT - 60:
                break
            x = WALL_L + 40 + (0 if row % 2 == 0 else gx / 2)
            while x < WALL_R - 30:
                peg(x, y, share)
                x += gx
    elif course == "triangle":
        gx, gy = rng.choice([84, 92]), rng.choice([66, 72])
        for row in range(9):
            y = 440 + row * gy
            if y > COURSE_BOT - 50:
                break
            n = row + 2
            for i in range(n):
                peg(HOLE[0] + (i - (n - 1) / 2) * gx, y, 0.34)
    elif course == "spinners":
        speed = rng.uniform(1.2, 2.0)
        for row, y in enumerate((540, 800)):
            xs = (230, 540, 850) if row == 0 else (385, 695)
            for k, x in enumerate(xs):
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = (x, y)
                body.angular_velocity = speed * (1 if (k + row) % 2 == 0 else -1)
                half = 95
                s = pymunk.Segment(body, (-half, 0), (half, 0), 8)
                s.elasticity, s.friction = 0.5, 0.4
                s.collision_type = 3
                s.part = "spinner"
                space.add(body, s)
                spinners.append({"body": body, "half": half})
        share = rng.uniform(0.5, 0.58)
        for y, off in ((430, 0), (475, 42), (655, 0), (700, 42), (930, 0), (975, 42)):
            x = WALL_L + 45 + off
            while x < WALL_R - 30:
                peg(x, y, share)
                x += 84
    elif course == "ramps":
        # Short, steep deflector ramps up top ("\ /" pairs) that throw the
        # flow sideways, then a peg field underneath for the multiplying.
        # (Long full-width ramps were tried: balls took ~16 s to roll down
        # three of them, far too slow for the multiply to keep up.)
        tilt = rng.choice([0.45, 0.55, 0.65])
        half = 105
        # (x, y, +1 = low end on the right). The top pair funnels to the
        # middle; the lower pair throws balls out toward the walls, with
        # a ball-wide gap left at every low end so nothing gets pinned.
        for x, y, d in ((330, 470, 1), (750, 470, -1), (270, 615, -1), (810, 615, 1)):
            dx, dy = half * math.cos(tilt), half * math.sin(tilt) * d
            parts.append(_seg(space, static, (x - dx, y - dy), (x + dx, y + dy), r=7,
                              kind="ramp", elasticity=0.3, friction=0.05))
        share = rng.uniform(0.7, 0.76)
        gx = rng.choice([84, 92])
        for row in range(6):
            y = 700 + row * 62
            x = WALL_L + 40 + (0 if row % 2 == 0 else gx / 2)
            while x < WALL_R - 30:
                peg(x, y, share)
                x += gx
    else:  # bumpers
        for y, xs in ((520, (250, 540, 830)), (760, (395, 685)), (960, (250, 830))):
            for x in xs:
                r = rng.choice([40, 46, 52])
                p = pymunk.Circle(static, r, (x, y))
                p.elasticity, p.friction = 0.9, 0.2
                p.collision_type = 3
                p.part = "bumper"
                space.add(p)
                parts.append(p)
        share = rng.uniform(0.5, 0.58)
        for y, off in ((430, 0), (475, 42), (630, 0), (675, 42), (855, 0), (900, 42)):
            x = WALL_L + 45 + off
            while x < WALL_R - 30:
                if all(math.dist((x, y), q.offset) > q.radius + 30 for q in parts
                       if getattr(q, "part", "") == "bumper"):
                    peg(x, y, share)
                x += 84
    return rng, space, parts, pegs, spinners


# ---- simulation ---------------------------------------------------------

class Sim:
    """One run of the machine. ``step()`` advances one video frame."""

    def __init__(self, recipe: dict):
        import pymunk

        self.pymunk = pymunk
        self.recipe = recipe
        self.chain = THEMES[recipe["theme"]]["chain"]
        self.last = len(self.chain) - 1
        self.radii = [17 * 1.29 ** k for k in range(len(self.chain))]
        self.rng, self.space, self.parts, self.pegs, self.spinners = _build(recipe)
        self.peg_by_shape = {p["shape"]: p for p in self.pegs}
        self.balls: list[dict] = []
        self.queue = [0]
        self.events: list[tuple] = []     # (t, kind, value, volume) for the audio
        self.effects: list[list] = []     # merge rings
        self.reveal_flash = [0.0] * len(self.chain)
        self.unlocked_at = {0: 0.0}
        self.banner = None
        self.spawned = self.best = 0
        self.t = 0.0
        self.done_at = None
        self.hole_pulse = 0.0
        self._release_cd = 0.0
        self._merges: list[tuple] = []
        self._multiplied: list[dict] = []
        self.space.on_collision(1, 2, begin=self._on_peg)
        self.space.on_collision(1, 3, begin=self._on_part)
        self.space.on_collision(1, 1, pre_solve=self._on_ball)

    @staticmethod
    def _impact(arbiter) -> float:
        n = arbiter.contact_point_set.normal
        v = arbiter.shapes[0].body.velocity - arbiter.shapes[1].body.velocity
        return abs(v.dot(n))

    def _add_ball(self, tier, pos, vel, r_start=None):
        pm = self.pymunk
        r0 = r_start or self.radii[tier]
        body = pm.Body()
        body.position, body.velocity = pos, vel
        shape = pm.Circle(body, r0)
        shape.density = 1.0
        shape.elasticity, shape.friction = 0.25, 0.5
        shape.collision_type = 1
        self.space.add(body, shape)
        b = {"body": body, "shape": shape, "tier": tier, "r": r0,
             "hit": set(), "landed": False, "alive": True}
        shape.ball = b
        self.balls.append(b)
        return b

    def _on_peg(self, arbiter, space, data):
        bs, ps = arbiter.shapes
        peg, b = self.peg_by_shape.get(ps), getattr(bs, "ball", None)
        if peg is None or b is None:
            return
        peg["flash"] = 1.0 if peg["gold"] else max(peg["flash"], 0.35)
        speed = self._impact(arbiter)
        if speed > 40 and self.t - peg.get("sounded", -1) > 0.1:
            peg["sounded"] = self.t
            self.events.append((self.t, "tick", peg["note"], min(1.0, speed / 420)))
        if peg["gold"] and b["tier"] == 0 and id(peg) not in b["hit"]:
            b["hit"].add(id(peg))
            self._multiplied.append(peg)

    def _on_part(self, arbiter, space, data):
        b = getattr(arbiter.shapes[0], "ball", None)
        speed = self._impact(arbiter)
        if b is None or speed < 60:
            return
        part = getattr(arbiter.shapes[1], "part", "wall")
        if part == "jar":
            if b["tier"] == 0 and not b.get("thudded"):
                b["thudded"] = True
                self.events.append((self.t, "thud", 0, min(1.0, speed / 500)))
        else:
            self.events.append((self.t, "knock", part, min(1.0, speed / 500)))

    def _on_ball(self, arbiter, space, data):
        a, c = (getattr(s, "ball", None) for s in arbiter.shapes)
        if (a and c and a["tier"] == c["tier"] and a["tier"] < self.last
                and a["body"].position.y > NECK_Y - 10
                and c["body"].position.y > NECK_Y - 10):
            self._merges.append((a, c))

    def _apply_merges(self):
        for a, c in self._merges:
            if not (a["alive"] and c["alive"]) or a["tier"] != c["tier"]:
                continue
            a["alive"] = c["alive"] = False
            self.space.remove(a["body"], a["shape"], c["body"], c["shape"])
            pa, pc = a["body"].position, c["body"].position
            mid = ((pa.x + pc.x) / 2, (pa.y + pc.y) / 2)
            vel = ((a["body"].velocity.x + c["body"].velocity.x) / 2,
                   (a["body"].velocity.y + c["body"].velocity.y) / 2 - 60)
            tier = a["tier"] + 1
            nb = self._add_ball(tier, mid, vel, r_start=self.radii[tier - 1])
            nb["landed"] = True
            self.effects.append([mid[0], mid[1], self.radii[tier], 1.0, tier])
            self.events.append((self.t, "pop", tier, 1.0))
            if tier > self.best:
                self.best = tier
                self.unlocked_at[tier] = self.t
                self.reveal_flash[tier] = 1.0
                self.banner = (f"NEW: {self.chain[tier][1]}!", self.t)
                self.events.append((self.t, "unlock", tier, 1.0))
        self._merges.clear()

    def _pull_pairs(self):
        """Matching items in the jar drift toward each other, so two big
        ones settling on opposite sides can't stall the run."""
        by_tier: dict[int, list] = {}
        for b in self.balls:
            if b["tier"] >= PULL_FROM_TIER and b["body"].position.y > JAR_TOP - 60:
                by_tier.setdefault(b["tier"], []).append(b)
        for group in by_tier.values():
            if len(group) < 2:
                continue
            for b in group:
                p = b["body"].position
                other = min((o for o in group if o is not b),
                            key=lambda o: (o["body"].position - p).length)
                v = other["body"].position - p
                if v.length > 1:
                    b["body"].apply_force_at_world_point(v.normalized() * b["body"].mass * PULL, p)

    def step(self):
        dt = 1 / FPS
        self._release_cd -= dt
        above = sum(1 for b in self.balls if b["body"].position.y < NECK_Y)
        if (self.queue and self._release_cd <= 0 and self.done_at is None
                and above < MAX_ABOVE_NECK):
            n = 1 if len(self.queue) < 6 else min(len(self.queue), 1 + len(self.queue) // 12)
            for _ in range(n):
                self.queue.pop()
                self._add_ball(0, (HOLE[0] + self.rng.uniform(-6, 6), HOLE[1] + 10),
                               (self.rng.uniform(-40, 40), 30))
                self.spawned += 1
            self.hole_pulse = 1.0
            self._release_cd = 0.7 if self.spawned < 4 else max(0.04, 0.45 - self.spawned * 0.004)

        for _ in range(4):
            self._pull_pairs()          # pymunk clears forces after every step
            self.space.step(dt / 4)
            self._apply_merges()        # never inside the collision callback
        self.balls[:] = [b for b in self.balls if b["alive"]]

        for b in self.balls:            # freshly merged items grow in ~0.2 s
            full = self.radii[b["tier"]]
            if b["r"] < full:
                b["r"] = min(full, b["r"] + (full - self.radii[max(0, b["tier"] - 1)]) / 12)
                b["shape"].unsafe_set_radius(b["r"])
            if not b["landed"] and b["body"].position.y > JAR_TOP - 40:
                b["landed"] = True
            # nothing may sit forever on a ramp end or a bumper's top
            if not b["landed"] and b["body"].velocity.length < 3 and self.rng.random() < 0.05:
                b["body"].apply_impulse_at_local_point((self.rng.uniform(-40, 40) * b["body"].mass, 0))

        for _ in self._multiplied:
            if self.spawned + len(self.queue) < SPAWN_CAP:
                self.queue.append(0)
                self.events.append((self.t, "note", self.spawned + len(self.queue), 1.0))
        self._multiplied.clear()
        if self.done_at is None and not self.queue and all(b["landed"] for b in self.balls):
            self.queue.append(0)        # the drip never dies out

        if self.done_at is None and self.best == self.last:
            self.done_at = self.t
            self.queue.clear()
            self.events.append((self.t, "fanfare", 0, 1.0))
        self.t += dt

    @property
    def finished(self) -> bool:
        return ((self.done_at is not None and self.t - self.done_at > END_HOLD)
                or self.t > SECONDS_CAP)


# ---- picking a good run --------------------------------------------------

def good_end(recipe: dict) -> tuple:
    n = len(THEMES[recipe["theme"]]["chain"])
    return (28.0, 52.0) if n <= 8 else (38.0, 62.0)


def simulate(recipe: dict) -> dict:
    """Physics only, no drawing: how this recipe+seed plays out."""
    sim = Sim(recipe)
    lo, hi = good_end(recipe)
    while not sim.finished and sim.done_at is None and sim.t < hi + 1:
        sim.step()
    times = sorted(sim.unlocked_at.values())
    gap = max((b - a for a, b in zip(times, times[1:])), default=99.0)
    return {"seed": recipe["seed"], "done_at": sim.done_at and round(sim.done_at, 1),
            "max_gap": round(gap, 1), "best": sim.best}


def is_good(recipe: dict, r: dict) -> bool:
    lo, hi = good_end(recipe)
    return r["done_at"] is not None and lo <= r["done_at"] <= hi and r["max_gap"] <= MAX_UNLOCK_GAP


def pick_seed(recipe: dict, tries: int = 48, workers: int | None = None, log=print) -> dict:
    """Simulate seeds of this recipe in parallel; return it with a good seed
    (and ``expected_end``, when the last item appears)."""
    import multiprocessing

    workers = workers or max(1, min(4, os.cpu_count() or 2))
    lo, hi = good_end(recipe)
    good = []
    start = recipe["seed"]
    # "spawn", not fork: forking a process that runs threads (the web app)
    # can deadlock the child.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for b0 in range(start, start + tries, workers):
            batch = [dict(recipe, seed=s) for s in range(b0, min(b0 + workers, start + tries))]
            for rec, r in zip(batch, pool.map(simulate, batch)):
                ok = is_good(rec, r)
                log(f"seed {r['seed']}: last item at {r['done_at']}, longest wait "
                    f"{r['max_gap']} s{'  <- good' if ok else ''}")
                if ok:
                    good.append(r)
            if good:
                break
    if not good:
        raise RuntimeError(f"no good run for {recipe['theme']}/{recipe['course']}/{recipe['jar']} "
                           f"in seeds {start}..{start + tries - 1}")
    best = min(good, key=lambda r: abs(r["done_at"] - (lo + hi) / 2))
    return dict(recipe, seed=best["seed"], expected_end=best["done_at"])


# ---- drawing -------------------------------------------------------------

def _tint(im: Image.Image) -> tuple:
    """A light bubble colour from the emoji's own average colour."""
    a = np.asarray(im.convert("RGBA").resize((32, 32)), dtype=np.float32)
    m = a[..., 3] > 128
    rgb = a[..., :3][m].mean(axis=0) if m.any() else np.array([200, 200, 200])
    return tuple(int(v) for v in (rgb * 0.55 + 255 * 0.45))


class Painter:
    def __init__(self, recipe: dict):
        theme = THEMES[recipe["theme"]]
        self.recipe = recipe
        self.chain = theme["chain"]
        top, bot = theme["bg"]
        g = np.linspace(0, 1, H)[:, None]
        rows = (np.array(top) * (1 - g) + np.array(bot) * g).astype(np.uint8)
        self.bg = Image.fromarray(np.repeat(rows[:, None, :], W, axis=1), "RGB")
        self.emoji = [Image.open(ASSETS / "emoji" / f"{c}.webp").convert("RGBA")
                      for c, _ in self.chain]
        self.colours = [_tint(e) for e in self.emoji]
        cell = min(104, (W - 40) // len(self.chain))
        self.cell, self.icon = cell, int(cell * 0.75)
        self.icons = [e.resize((self.icon, self.icon), Image.LANCZOS) for e in self.emoji]
        self.sils = []
        for e in self.icons:
            sil = Image.new("RGBA", e.size, (40, 32, 70, 0))
            sil.putalpha(e.getchannel("A").point(lambda v: int(v * 0.85)))
            self.sils.append(sil)
        self.sprites: dict = {}
        self.f_hook, self.f_banner = _font(54), _font(64)
        self.f_big, self.f_small = _font(130), _font(36, "Inter-Bold.ttf")

    def sprite(self, tier: int, r: float) -> Image.Image:
        r = max(4, int(round(r)))
        key = (tier, r)
        if key not in self.sprites:
            ss = 3
            s = (2 * r + 4) * ss
            im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            c, R = s / 2, r * ss
            col = self.colours[tier]
            d.ellipse((c - R, c - R, c + R, c + R), fill=col + (235,))
            d.ellipse((c - R, c - R, c + R, c + R),
                      outline=tuple(int(v * 0.6) for v in col) + (255,), width=ss * 2)
            e = int(R * 1.45)
            im.alpha_composite(self.emoji[tier].resize((e, e), Image.LANCZOS),
                               (int(c - e / 2), int(c - e / 2)))
            hr = R * 0.25
            d.ellipse((c - R * 0.5 - hr, c - R * 0.55 - hr, c - R * 0.5 + hr, c - R * 0.55 + hr),
                      fill=(255, 255, 255, 90))
            self.sprites[key] = im.resize((s // ss, s // ss), Image.LANCZOS)
        return self.sprites[key]

    def frame(self, sim: Sim) -> Image.Image:
        im = self.bg.copy()
        d = ImageDraw.Draw(im, "RGBA")
        glass = (200, 220, 255, 150)
        for s in sim.parts:
            if isinstance(s, sim.pymunk.Segment):
                w = 14 if s.part == "ramp" else 10
                d.line([s.a, s.b], fill=(255, 255, 255, 190) if s.part == "ramp" else glass, width=w)
            else:                                     # bumper
                (x, y), r = s.offset, s.radius
                d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 30),
                          outline=(140, 220, 255, 220), width=6)
        for sp in sim.spinners:
            b, h = sp["body"], sp["half"]
            a, c = b.local_to_world((-h, 0)), b.local_to_world((h, 0))
            d.line([a, c], fill=(255, 140, 200, 230), width=16)
            x, y = b.position
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(255, 255, 255, 230))
        hr = 34 + 8 * sim.hole_pulse
        d.ellipse((HOLE[0] - hr, HOLE[1] - hr * 0.55, HOLE[0] + hr, HOLE[1] + hr * 0.55),
                  fill=(0, 0, 0, 255), outline=(255, 210, 90, 220), width=5)
        sim.hole_pulse *= 0.85
        for p in sim.pegs:
            (x, y), R = p["pos"], p["r"]
            if p["gold"]:
                if p["flash"] > 0.05:
                    g = R + 10 * p["flash"]
                    d.ellipse((x - g - 6, y - g - 6, x + g + 6, y + g + 6),
                              fill=(255, 200, 60, int(110 * p["flash"])))
                d.ellipse((x - R - 2, y - R - 2, x + R + 2, y + R + 2), fill=(255, 205, 60, 255))
            else:
                v = int(120 + 120 * p["flash"])
                d.ellipse((x - R, y - R, x + R, y + R), fill=(v, v, v + 20, 255))
            p["flash"] *= 0.88

        for b in sorted(sim.balls, key=lambda b: b["tier"]):
            x, y = b["body"].position
            sp = self.sprite(b["tier"], b["r"])
            if b["tier"] >= 2:
                sp = sp.rotate(math.degrees(-b["body"].angle), resample=Image.BICUBIC)
            im.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)

        for e in sim.effects:            # expanding ring where two merged
            x, y, r, life, tier = e
            rr = r * (1.0 + (1 - life) * 0.9)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=self.colours[tier] + (int(230 * life),),
                      width=max(2, int(8 * life)))
            e[3] -= 0.06
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]

        def ctext(y, s, f, fill=(255, 255, 255, 255)):
            w = d.textlength(s, font=f)
            d.text(((W - w) / 2 + 3, y + 3), s, font=f, fill=(0, 0, 0, 150))
            d.text(((W - w) / 2, y), s, font=f, fill=fill)

        ctext(30, self.recipe["hook"], self.f_hook)
        n, cell, ic_s, top = len(self.chain), self.cell, self.icon, 128
        x0 = (W - cell * n) / 2
        for i in range(n):
            cx = x0 + i * cell + cell / 2
            on, fl = i <= sim.best, sim.reveal_flash[i]
            half = cell / 2 - 6
            d.rounded_rectangle((cx - half, top - 6, cx + half, top + ic_s + 8), 16,
                                fill=(255, 255, 255, 40 if on else 14),
                                outline=(255, 215, 100, int(80 + 175 * fl)) if on
                                else (255, 255, 255, 40), width=3 + int(4 * fl))
            ic = self.icons[i] if on else self.sils[i]
            if on and fl > 0:
                s = int(ic_s * (1 + 0.35 * fl))
                ic = self.emoji[i].resize((s, s), Image.LANCZOS)
            im.paste(ic, (int(cx - ic.width / 2), int(top + ic_s / 2 + 1 - ic.height / 2)), ic)
            if not on:
                d.text((cx - 10, top + ic_s / 2 - 20), "?", font=self.f_small,
                       fill=(255, 255, 255, 130))
            sim.reveal_flash[i] *= 0.93

        unit = THEMES[self.recipe["theme"]]["unit"]
        d.text((JAR_L, JAR_BOT + 18), f"{sim.spawned:,} {unit} dropped", font=self.f_small,
               fill=(220, 230, 255, 255))
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((JAR_R - d.textlength(mark, font=self.f_small), JAR_BOT + 18), mark,
                   font=self.f_small, fill=(255, 255, 255, 150))

        if sim.banner and sim.done_at is None and sim.t - sim.banner[1] < 1.6:
            age = sim.t - sim.banner[1]
            a = min(1.0, age / 0.15) * min(1.0, (1.6 - age) / 0.3)
            bw = d.textlength(sim.banner[0], font=self.f_banner) + 60
            d.rounded_rectangle(((W - bw) / 2, 236, (W + bw) / 2, 316), 40,
                                fill=(20, 14, 40, int(220 * a)))
            ctext(240, sim.banner[0], self.f_banner, (255, 225, 110, int(255 * a)))
        if sim.done_at is not None:
            age = sim.t - sim.done_at
            s = int(260 * min(1.0, age / 0.5))
            if s > 4:
                big = self.emoji[-1].resize((s, s), Image.LANCZOS)
                im.paste(big, (int(W / 2 - s / 2), int(740 - s / 2)), big)
            ending = self.recipe["ending"]
            f = self.f_big if d.textlength(ending, font=self.f_big) < W - 80 else _font(100)
            ctext(1000, ending, f, (255, 225, 120, int(255 * min(1.0, age / 0.4))))
        return im


# ---- audio (all generated) -----------------------------------------------

def _noise(n: int, lo: float, hi: float, seed, pink: float = 0.0) -> np.ndarray:
    """White noise band-limited to lo..hi Hz (soft edges), optionally tilted
    toward pink (pink=0.5) or brown (1.0). Normalised to peak 1."""
    x = np.random.default_rng(seed).standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1.0
    mask = 1 / (1 + (lo / f) ** 4) / (1 + (f / hi) ** 4)
    if pink:
        mask = mask * (f / 1000.0) ** (-pink)
    y = np.fft.irfft(np.fft.rfft(x) * mask, n)
    return (y / max(1e-9, float(np.max(np.abs(y))))).astype(np.float32)


def _burst(dur, lo, hi, decay, vol, seed, attack=0.002, pink=0.0) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    env = np.minimum(1, t / attack) * np.exp(-t * decay)
    return (_noise(n, lo, hi, seed, pink) * env * vol).astype(np.float32)


class Voice:
    """The video's sounds: soft, noise-based ASMR taps, puffs and whooshes
    over a quiet bed of noise (rain, air or hush, picked per video).
    Dean found the first musical version (marimba notes, chimes, a music
    loop) "horrible" and asked for "more white noise"."""

    BEDS = ("rain", "air", "hush")

    def __init__(self, recipe: dict):
        pick = random.Random(f"bed-{recipe.get('seed')}-{recipe.get('key')}-{recipe.get('scale')}")
        self.bed_kind = pick.choice(self.BEDS)
        self.tone = pick.uniform(0.85, 1.15)     # shifts every sound's pitch a little per video
        self._cache: dict = {}
        self._n = 0

    def _get(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def tick(self, peg_note: int) -> np.ndarray:
        """A marble tapping a peg: a short bright click, a little higher
        toward the right. Three takes per peg so repeats don't sound copied."""
        self._n += 1
        take = self._n % 3
        lo = (1400 + peg_note * 220) * self.tone
        return self._get(("tick", peg_note, take),
                         lambda: _burst(0.05, lo, lo * 2.3, 150, 0.22, (peg_note, take)))

    def knock(self, part: str) -> np.ndarray:
        lo, hi, decay, vol = {"spinner": (700, 1900, 120, 0.16), "bumper": (240, 900, 70, 0.2),
                              "ramp": (1100, 2800, 170, 0.12)}.get(part, (1300, 3200, 200, 0.10))
        return self._get(("knock", part), lambda: _burst(0.07, lo * self.tone, hi * self.tone, decay, vol, len(part)))

    def thud(self) -> np.ndarray:
        return self._get("thud", lambda: _burst(0.12, 60, 420, 45, 0.28, 7, attack=0.004, pink=0.5))

    def pling(self, k: int) -> np.ndarray:
        """A multiply: a faint airy "tsk"."""
        return self._get("pling", lambda: _burst(0.06, 5000, 11000, 90, 0.06, 11))

    def pop(self, tier: int) -> np.ndarray:
        """A merge: a soft puff, deeper and longer for bigger items, with a
        low thump under the big ones."""
        def make():
            c = 3000 / (1.35 ** tier) * self.tone
            dur = 0.12 + 0.03 * tier
            out = _burst(dur, c * 0.5, c * 1.8, 40 - 3 * tier, 0.3 + 0.03 * tier, 100 + tier, attack=0.004, pink=0.3)
            if tier >= 3:
                thump = _burst(dur, 40, 160, 22, 0.25 + 0.03 * tier, 200 + tier, attack=0.003)
                out = out + thump[:len(out)]
            return out
        return self._get(("pop", tier), make)

    def _whoosh(self, dur, rise, vol, seed, lo=300, hi=6000) -> np.ndarray:
        n = int(SR * dur)
        t = np.arange(n) / SR
        env = np.where(t < rise, (t / rise) ** 2, np.exp(-(t - rise) * 9))
        return (_noise(n, lo, hi, seed, pink=0.5) * env * vol).astype(np.float32)

    def chime(self, tier: int) -> np.ndarray:
        """A new item revealed: a whoosh that lands on a soft thump."""
        def make():
            w = self._whoosh(0.8, 0.55, 0.28, 300 + tier)
            thump = _burst(0.2, 40, 220, 18, 0.3, 400 + tier, attack=0.003)
            o = int(0.55 * SR)
            w[o:o + len(thump)] += thump[:len(w) - o]
            return w
        return self._get(("chime", tier), make)

    def fanfare(self) -> np.ndarray:
        """The last item: a long whoosh, a deep boom and an airy tail."""
        w = self._whoosh(3.2, 0.9, 0.4, 500, lo=150, hi=7000)
        boom = _burst(1.2, 30, 180, 4, 0.45, 501, attack=0.004)
        o = int(0.9 * SR)
        w[o:o + len(boom)] += boom[:len(w) - o]
        return w

    def music(self, seconds: float) -> np.ndarray:
        """The bed: steady soft noise under everything (kept at the name the
        mix calls)."""
        n = int(SR * seconds)
        t = np.arange(n) / SR
        if self.bed_kind == "rain":
            bed = _noise(n, 400, 9000, 900, pink=0.5)
            drops = np.zeros(n, dtype=np.float32)
            rng = np.random.default_rng(901)
            tick = _burst(0.02, 2500, 7000, 260, 1.0, 902)
            for at in rng.integers(0, max(1, n - len(tick)), int(seconds * 25)):
                drops[at:at + len(tick)] += tick * rng.uniform(0.15, 0.5)
            bed = bed * 0.8 + drops * 0.5
        elif self.bed_kind == "air":
            bed = _noise(n, 80, 2500, 910, pink=1.0) * (0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t))
        else:
            bed = _noise(n, 200, 5000, 920, pink=0.5)
        rms = float(np.sqrt(np.mean(bed ** 2))) or 1.0
        fade = np.minimum(1, t / 0.6) * np.minimum(1, (seconds - t) / 1.0)
        return (bed / rms * 0.06 * fade).astype(np.float32)


def mix_audio(sim: Sim) -> np.ndarray:
    v = Voice(sim.recipe)
    n = int(SR * (sim.t + 1))
    fx = np.zeros(n, dtype=np.float32)
    last = {"tick": -1.0, "knock": -1.0, "thud": -1.0, "note": -1.0, "pop": -1.0}
    gaps = {"tick": 0.035, "knock": 0.05, "thud": 0.06, "note": 0.06, "pop": 0.03}
    for et, kind, val, vol in sim.events:
        if kind in gaps:                  # a flood stays a soft patter, not a roar
            if et - last[kind] < gaps[kind] and not (kind == "pop" and val >= 3):
                continue
            last[kind] = et
        w = {"tick": lambda: v.tick(val), "knock": lambda: v.knock(val),
             "thud": v.thud, "note": lambda: v.pling(val), "pop": lambda: v.pop(val),
             "unlock": lambda: v.chime(val), "fanfare": v.fanfare}[kind]()
        i = int(et * SR)
        seg = fx[i:i + len(w)]
        seg += w[:len(seg)] * (0.35 + 0.65 * vol)
    fx = np.tanh(fx * 1.4) * 0.85
    bed = v.music(sim.t + 1)[:n]
    mix = fx + bed
    return mix / max(1e-6, float(np.max(np.abs(mix)))) * 0.89


# ---- publish text --------------------------------------------------------------

def emoji_char(code: str) -> str:
    return "".join(chr(int(part, 16)) for part in code.split("-"))


def publish_text(recipe: dict) -> dict:
    """A default title and description for YouTube. The title asks the hook
    question; nothing names the last item, so the reveal isn't spoiled."""
    theme = THEMES[recipe["theme"]]
    first = emoji_char(theme["chain"][0][0])
    title = f"{recipe['hook']} {first}"
    desc = (f"It starts with one {first}. Every gold peg sends out another, and two of the same "
            f"merge into something bigger. Can it reach the end?\n\n"
            f"Guess the last one before it shows up 👇\n\n#satisfying #physics #evolution #asmr")
    return {"title": title[:100], "description": desc}


# ---- render ----------------------------------------------------------------

def render(recipe: dict, out_path: str | os.PathLike, log=print, progress=None) -> dict:
    """Draw and encode the video. ``progress(fraction)`` is called as it
    goes (estimated from ``expected_end`` when pick_seed set it)."""
    out_path = Path(out_path)
    video = out_path.with_suffix(".video.mp4")
    wav = out_path.with_suffix(".wav")
    sim, painter = Sim(recipe), Painter(recipe)
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "23", "-pix_fmt", "yuv420p", str(video)],
        stdin=subprocess.PIPE)
    total = ((recipe.get("expected_end") or SECONDS_CAP - END_HOLD) + END_HOLD) * FPS
    frame = 0
    try:
        while not sim.finished:
            sim.step()
            proc.stdin.write(painter.frame(sim).tobytes())
            frame += 1
            if progress and frame % 30 == 0:
                progress(min(0.97, frame / total))
    finally:
        proc.stdin.close()
        proc.wait()
    pcm = (mix_audio(sim) * 32767).astype(np.int16)
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)],
                   check=True)
    video.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    chain = THEMES[recipe["theme"]]["chain"]
    info = dict(recipe, seconds=round(sim.t, 1), dropped=sim.spawned,
                done_at=sim.done_at and round(sim.done_at, 1),
                unlocks={chain[k][1]: round(t, 1) for k, t in sorted(sim.unlocked_at.items())})
    log(f"rendered {out_path}: {info}")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--seed", type=int, help="start seed (default: random)")
    ap.add_argument("--theme", choices=sorted(THEMES))
    ap.add_argument("--course", choices=COURSES)
    ap.add_argument("--jar", choices=JARS)
    ap.add_argument("--history", help="JSON file of past recipes; avoids repeats, gets appended")
    ap.add_argument("--exact", action="store_true", help="render this seed as is, no picking")
    ap.add_argument("--recipe", help="a full recipe as JSON (from pick_recipe); overrides the options above")
    ap.add_argument("--progress", action="store_true",
                    help="print machine-readable PROGRESS/RESULT lines (the web app reads them)")
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else random.randrange(1, 10 ** 6)
    history = _load_history(args.history)
    if args.recipe:
        recipe = json.loads(args.recipe)
    else:
        recipe = pick_recipe(seed, history, args.theme, args.course, args.jar)
    say = (lambda *a, **k: None) if args.progress else print
    say("recipe:", recipe)
    if not args.exact:
        tried = [0]

        def log(line):
            tried[0] += 1
            if args.progress:
                print(f"PROGRESS pick {tried[0]}", flush=True)
            else:
                print(line)
        recipe = pick_seed(recipe, log=log)
    report = (lambda f: print(f"PROGRESS render {f:.3f}", flush=True)) if args.progress else None
    info = render(recipe, args.out, log=say, progress=report)
    if args.progress:
        print("RESULT " + json.dumps(info), flush=True)
    if args.history:
        history.append({k: info[k] for k in ("seed", "theme", "course", "jar", "key", "scale")})
        Path(args.history).write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
