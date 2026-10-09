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

from clipper import ball_themes

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
GATE_DRIP = 1.5          # gates course: the hole drips every this many s at first

# ---- themes: (emoji file, name) small to big ---------------------------
# The evolution chains live in clipper/ball_themes.py (57 of them). The
# first eight keep the background colours their videos were made with;
# the rest get one from their own emoji (see Painter).
_LEGACY_BG = {'animals': ((14, 12, 30), (34, 18, 52)), 'sports': ((8, 26, 20), (14, 52, 36)), 'food': ((36, 14, 12), (64, 28, 20)), 'space': ((4, 6, 22), (16, 20, 58)), 'money': ((6, 22, 14), (22, 44, 22)), 'laughs': ((10, 24, 34), (16, 44, 60)), 'vehicles': ((16, 18, 28), (34, 38, 56)), 'weather': ((6, 16, 34), (12, 34, 64))}
THEMES = ball_themes.build()
for _k, _bg in _LEGACY_BG.items():
    if _k in THEMES:
        THEMES[_k]["bg"] = _bg
COURSES = ["pegs", "triangle", "spinners", "ramps", "bumpers", "gates", "wheel"]
SKINS = ["bubble", "glass", "plain", "neon"]                 # how each item is drawn
BACKDROPS = ["gradient", "glow", "stars", "grid", "dots"]    # behind the machine
SOUNDS = ["marble", "glass", "wood", "plastic", "rubber", "water", "metal", "pop"]   # the bounce sound
# Dean found the box course + narrow neck + jar hard to follow and picked
# the gumball (one round globe under a short neck) from five mockups; the
# old jars stay for re-rendering old recipes.
JARS = ["gumball"]
OLD_JARS = ["box", "bowl", "flask"]
GUMBALL = {"cx": 540, "cy": 1100, "r": 520, "neck": 130}
# The course band 400..1030 maps to 665..1085, leaving room under the neck
# (pegs right under it made pockets where items wedged and the run stalled).
_GUM_TOP, _GUM_SCALE = 665, 420 / 630
_GUM_GOLD = 1.3          # fewer pegs fit in the globe: more of them are gold


def _geo(recipe: dict) -> dict:
    """Where items count as "in the jar" (merge, landed, pull), where the
    hole's wait-count line is, and where the counter text goes."""
    if recipe.get("jar") == "gumball":
        cy, r = GUMBALL["cy"], GUMBALL["r"]
        return {"merge_y": cy - 30, "above_y": cy - 110, "land_y": cy - 70, "pull_y": cy - 90,
                "text_y": cy + r + 24, "text_l": 110, "text_r": 970}
    return {"merge_y": NECK_Y - 10, "above_y": NECK_Y, "land_y": JAR_TOP - 40, "pull_y": JAR_TOP - 60,
            "text_y": JAR_BOT + 18, "text_l": JAR_L, "text_r": JAR_R}


def _gumball_points() -> list:
    """The neck down from the hole, then the globe, as one open outline."""
    cx, cy, r, n = GUMBALL["cx"], GUMBALL["cy"], GUMBALL["r"], GUMBALL["neck"]
    a0 = math.asin(n / r)
    pts = [(cx - n, 380)]
    steps = 96
    for k in range(steps + 1):
        a = -math.pi / 2 - a0 - (2 * math.pi - 2 * a0) * k / steps
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((cx + n, 380))
    return pts
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

FORMATS = ("evolve", "escape", "touch")      # escape/touch: clipper/ball_circles.py


def make_recipe(seed: int, theme=None, course=None, jar=None, fmt=None) -> dict:
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
        "skin": rng.choice(SKINS), "backdrop": rng.choice(BACKDROPS), "sound": rng.choice(SOUNDS),
        "format": fmt or "evolve",
    }


def _load_history(path) -> list:
    try:
        return json.loads(Path(path).read_text()) if path else []
    except (OSError, ValueError):
        return []


def pick_recipe(seed: int, history: list, theme=None, course=None, jar=None, fmt=None, today=None) -> dict:
    """A recipe that looks and sounds different from what was posted lately:
    its theme isn't one of the last ~20, its theme+course+jar combination is
    new, and its course, ball style, background and sound differ from the
    last video's -- as far as the candidates allow (best score wins). In a
    season (ball_themes.SEASONS: Halloween in October) its themes score a
    bit more, so they come up about every third video instead of rarely."""
    import datetime
    season = ball_themes.SEASONS.get((today or datetime.date.today()).month, set())
    window = min(20, max(3, len(THEMES) // 2))
    recent = {h.get("theme") for h in history[-window:]}
    def combo(h):
        if (h.get("format") or "evolve") == "evolve":
            return ("evolve", h.get("theme"), h.get("course"), h.get("jar"))
        return (h.get("format"), h.get("theme"), h.get("skin"))
    combos = {combo(h) for h in history}
    last = history[-1] if history else {}
    best, best_score = None, -1
    for k in range(300):
        r = make_recipe(seed * 1000 + k, theme, course, jar, fmt)
        r["seed"] = seed
        score = (4 * (combo(r) not in combos)
                 + 4 * (bool(theme) or r["theme"] not in recent)
                 + sum(r.get(f) != last.get(f) for f in ("course", "skin", "backdrop", "sound"))
                 + 3 * (not theme and r["theme"] in season))
        if score > best_score:
            best, best_score = r, score
        if score == 12 + 3 * bool(season):
            break
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

    gum = recipe.get("jar") == "gumball"

    def Y(y):                    # course coordinates -> this machine's
        return _GUM_TOP + (y - COURSE_TOP) * _GUM_SCALE if gum else y

    def inside(x, y, m):         # is (x, y) at least m inside the walls?
        if not gum:
            return WALL_L + m <= x <= WALL_R - m
        cx, cy, r, n = GUMBALL["cx"], GUMBALL["cy"], GUMBALL["r"], GUMBALL["neck"]
        if math.hypot(x - cx, y - cy) <= r - m:
            return True
        return y < cy and abs(x - cx) <= n - m

    rng = random.Random(recipe["seed"])
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.iterations = 20
    static = space.static_body
    parts = []      # drawables: static segments
    pegs = []       # {"shape","pos","r","gold","flash","note"}
    spinners = []   # {"body","half"}

    if gum:
        outline = _gumball_points()
        for a, b in zip(outline, outline[1:]):
            parts.append(_seg(space, static, a, b, r=7, kind="jar"))
    else:
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
        y = Y(y)
        if not inside(x, y, room):
            return
        # squeezed into the globe, staggered rows can end up closer than an
        # item is wide and form a mesh nothing passes through
        if gum and any(math.dist((x, y), q["pos"]) < 2 * r + 40 for q in pegs):
            return
        p = pymunk.Circle(static, r, (x, y))
        p.elasticity, p.friction = elasticity, 0.3
        p.collision_type = 2
        space.add(p)
        note = int((x - WALL_L) / (WALL_R - WALL_L) * 10)     # left low, right high
        pegs.append({"shape": p, "pos": (x, y), "r": r,
                     "gold": rng.random() < (min(0.62, gold_share * _GUM_GOLD) if gum else gold_share),
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
                half = 80 if gum else 95
                if not inside(x, Y(y), half + 30):
                    continue
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                body.position = (x, Y(y))
                body.angular_velocity = speed * (1 if (k + row) % 2 == 0 else -1)
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
            if gum:
                dx, dy = dx * 0.8, dy * 0.8
                if not (inside(x - dx, Y(y) - dy, 45) and inside(x + dx, Y(y) + dy, 45)):
                    continue
            parts.append(_seg(space, static, (x - dx, Y(y) - dy), (x + dx, Y(y) + dy), r=7,
                              kind="ramp", elasticity=0.3, friction=0.05))
        share = rng.uniform(0.7, 0.76)
        gx = rng.choice([84, 92])
        for row in range(6):
            y = 700 + row * 62
            x = WALL_L + 40 + (0 if row % 2 == 0 else gx / 2)
            while x < WALL_R - 30:
                peg(x, y, share)
                x += gx
    elif course == "gates":
        # Multiplier gates: every item that falls through one comes out as
        # that many (copies appear right under the gate). Three rows, each
        # split across the width, with a few plain pegs to spread the flow.
        # No gold pegs: the hole drips steadily instead (see Sim.step).
        import pymunk as _pm
        mults = [[2, "+1"], ["+1", 2, "+2"], [2, "+2"]]
        rng.shuffle(mults[0]); rng.shuffle(mults[1]); rng.shuffle(mults[2])
        for row, (y, ms) in enumerate(zip((530, 750, 960), mults)):
            y = Y(y)
            n = len(ms)
            left, right = WALL_L, WALL_R
            if gum:          # as wide as the globe is at this height
                hw = math.sqrt(max(0.0, GUMBALL["r"] ** 2 - (y - GUMBALL["cy"]) ** 2)) - 30
                left, right = GUMBALL["cx"] - hw, GUMBALL["cx"] + hw
            span = (right - left - 40) / n
            for i, m in enumerate(ms):
                x0 = left + 20 + i * span + 14
                x1 = x0 + span - 28
                g = _pm.Segment(static, (x0, y), (x1, y), 10)
                g.sensor = True
                g.collision_type = 4
                g.part = "gate"
                g.mult = m
                g.flash = 0.0
                space.add(g)
                parts.append(g)
            for i in range(1, n):                     # dividers between gates
                x = left + 20 + i * span
                parts.append(_seg(space, static, (x, y - 34), (x, y + 10), r=6, kind="divider"))
        gx = rng.choice([92, 100])
        for y, off in ((430, 0), (640, gx / 2), (860, 0)):
            x = WALL_L + 50 + off
            while x < WALL_R - 30:
                peg(x, y, 0.0)
                x += gx
    elif course == "wheel":
        # A big spinning wheel (four bars through the hub = eight spokes)
        # that bats the items around, gold pegs above and beside it.
        speed = rng.uniform(0.9, 1.4) * rng.choice((-1, 1))
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = (540, Y(715))
        body.angular_velocity = speed
        half = 200 if gum else 235
        space.add(body)
        # Three bars = six spokes (pockets wide enough to spill), slick, and
        # a solid hub: with four bars and grippy spokes items rode the
        # pockets next to the hub forever and the run stalled.
        hub = pymunk.Circle(body, 70)
        hub.elasticity, hub.friction = 0.5, 0.1
        hub.collision_type = 3
        hub.part = "spinner"
        space.add(hub)
        for k in range(3):
            a = k * math.pi / 3
            dx, dy = half * math.cos(a), half * math.sin(a)
            s_ = pymunk.Segment(body, (-dx, -dy), (dx, dy), 9)
            s_.elasticity, s_.friction = 0.5, 0.08
            s_.collision_type = 3
            s_.part = "spinner"
            space.add(s_)
            spinners.append({"body": body, "half": half, "rot": a, "wheel": True})
        share = rng.uniform(0.5, 0.58)
        for y, off in ((420, 0), (455, 42)):
            x = WALL_L + 45 + off
            while x < WALL_R - 30:
                peg(x, y, share)
                x += 84
        for y in range(540, 960, 70):                 # columns either side of the wheel
            for x in (150, 220, 860, 930):
                peg(x + (35 if (y // 70) % 2 else 0) * (1 if x < 540 else -1), y, share)
        for x in range(200, 900, 84):
            peg(x, 985, share)
    else:  # bumpers
        for y, xs in ((520, (250, 540, 830)), (760, (395, 685)), (960, (250, 830))):
            for x in xs:
                r = rng.choice([40, 46, 52]) * (0.85 if gum else 1)
                if not inside(x, Y(y), r + 45):
                    continue
                p = pymunk.Circle(static, r, (x, Y(y)))
                p.elasticity, p.friction = 0.9, 0.2
                p.collision_type = 3
                p.part = "bumper"
                space.add(p)
                parts.append(p)
        share = rng.uniform(0.5, 0.58)
        for y, off in ((430, 0), (475, 42), (630, 0), (675, 42), (855, 0), (900, 42)):
            x = WALL_L + 45 + off
            while x < WALL_R - 30:
                if all(math.dist((x, Y(y)), q.offset) > q.radius + 30 for q in parts
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
        self.geo = _geo(recipe)
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
        self.space.on_collision(1, 4, begin=self._on_gate)
        self.gates = [p for p in self.parts if getattr(p, "part", "") == "gate"]
        self.gate_course = bool(self.gates)
        self._gated: list[tuple] = []
        self.pops: list[list] = []        # floating "x2" labels
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
             "hit": set(), "landed": False, "alive": True, "born": self.t}
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

    def _on_gate(self, arbiter, space, data):
        bs, gs = arbiter.shapes
        b = getattr(bs, "ball", None)
        if b is None or b["tier"] != 0 or id(gs) in b["hit"]:
            return
        b["hit"].add(id(gs))
        self._gated.append((b, gs))

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
                and a["body"].position.y > self.geo["merge_y"]
                and c["body"].position.y > self.geo["merge_y"]):
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
            if b["tier"] >= PULL_FROM_TIER and b["body"].position.y > self.geo["pull_y"]:
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
        above = sum(1 for b in self.balls if b["body"].position.y < self.geo["above_y"])
        if (self.queue and self._release_cd <= 0 and self.done_at is None
                and above < MAX_ABOVE_NECK):
            n = 1 if len(self.queue) < 6 else min(len(self.queue), 1 + len(self.queue) // 12)
            for _ in range(n):
                self.queue.pop()
                self._add_ball(0, (HOLE[0] + self.rng.uniform(-6, 6), HOLE[1] + 10),
                               (self.rng.uniform(-40, 40), 30))
                self.spawned += 1
            self.hole_pulse = 1.0
            # gates: the drip starts slow and speeds up, so the last items
            # don't keep the viewer waiting (a steady drip grows linearly)
            gate_cd = max(0.45, GATE_DRIP - 0.025 * self.t) * (1.15 if self.last <= 7 else 1.0)
            self._release_cd = (gate_cd if self.gate_course
                                else 0.7 if self.spawned < 4 else max(0.04, 0.45 - self.spawned * 0.004))

        for _ in range(4):
            self._pull_pairs()          # pymunk clears forces after every step
            self.space.step(dt / 4)
            self._apply_merges()        # never inside the collision callback
        top = 372 if self.recipe.get("jar") == "gumball" else 150
        for b in self.balls:            # anything flung out of the machine is gone
            x, y = b["body"].position     # (in the gumball: back up the neck = into the hole)
            if b["alive"] and not (-50 < x < W + 50 and (top if self.t - b["born"] > 0.6 else 150) < y < H + 50):
                b["alive"] = False
                self.space.remove(b["body"], b["shape"])
        self.balls[:] = [b for b in self.balls if b["alive"]]

        for b in self.balls:            # freshly merged items grow in ~0.2 s
            full = self.radii[b["tier"]]
            if b["r"] < full:
                b["r"] = min(full, b["r"] + (full - self.radii[max(0, b["tier"] - 1)]) / 12)
                b["shape"].unsafe_set_radius(b["r"])
            if not b["landed"] and b["body"].position.y > self.geo["land_y"]:
                b["landed"] = True
            # nothing may sit forever on a ramp end or a bumper's top
            if not b["landed"] and b["body"].velocity.length < 3 and self.rng.random() < 0.05:
                b["body"].apply_impulse_at_local_point((self.rng.uniform(-40, 40) * b["body"].mass, 0))

        for b, g in self._gated:
            m = g.mult
            extra = (int(m[1:]) if isinstance(m, str) else m - 1)
            above = sum(1 for x in self.balls if x["body"].position.y < self.geo["above_y"])
            if self.done_at is not None or above > MAX_ABOVE_NECK * 2 or self.spawned >= SPAWN_CAP:
                extra = 0
            p, v = b["body"].position, b["body"].velocity
            for k in range(extra):
                nb = self._add_ball(0, (p.x + self.rng.uniform(-18, 18), p.y + 26 + k * 4),
                                    (v.x + self.rng.uniform(-120, 120), max(60.0, v.y)))
                nb["hit"] = set(b["hit"])
                self.spawned += 1
            g.flash = 1.0
            if extra and self.t - getattr(g, "popped", -1.0) > 0.35:   # one label at a time per gate
                g.popped = self.t
                self.pops.append([p.x, g.a.y + 70, f"×{m}" if not isinstance(m, str) else m, 1.0])
                self.events.append((self.t, "note", self.spawned, 0.8))
        self._gated.clear()
        for p in self.pops:             # labels fade over ~0.33 s
            p[3] -= 0.05
        self.pops[:] = [p for p in self.pops if p[3] > 0][-24:]
        if self.gate_course and self.done_at is None and len(self.queue) < 2:
            self.queue.append(0)        # gates multiply; the hole just keeps dripping
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

def _parts(recipe: dict):
    """The simulation and painter classes for the recipe's format."""
    if recipe.get("format") in ("escape", "touch"):
        from clipper import ball_circles
        return ball_circles.CircleSim, ball_circles.CirclePainter
    return Sim, Painter


def good_end(recipe: dict) -> tuple:
    if recipe.get("format") in ("escape", "touch"):
        return (25.0, 60.0)
    n = len(THEMES[recipe["theme"]]["chain"])
    return (28.0, 52.0) if n <= 8 else (38.0, 62.0)


def simulate(recipe: dict) -> dict:
    """Physics only, no drawing: how this recipe+seed plays out."""
    sim = _parts(recipe)[0](recipe)
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


def _auto_bg(colour: tuple) -> tuple:
    """A dark two-tone background tinted by the theme's last item."""
    c = np.array(colour, dtype=float)
    return (tuple(int(v) for v in c * 0.08 + 8), tuple(int(v) for v in c * 0.2 + 12))


def _backdrop(bg: Image.Image, kind: str, colour: tuple, seed: int) -> Image.Image:
    """Decorates the plain gradient: a glow, stars, a grid or dots."""
    if kind == "gradient":
        return bg
    rng = random.Random(f"backdrop-{seed}")
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if kind == "glow":
        for i in range(24, 0, -1):
            r = 60 + i * 40
            d.ellipse((W / 2 - r, 1100 - r, W / 2 + r, 1100 + r), fill=tuple(colour) + (int(7 * (24 - i) / 24) + 2,))
    elif kind == "stars":
        for _ in range(260):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.choice([1, 1, 2, 2, 3])
            d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, rng.randint(40, 150)))
    elif kind == "grid":
        for x in range(0, W, 60):
            d.line([(x, 0), (x, H)], fill=(255, 255, 255, 14), width=2)
        for y in range(0, H, 60):
            d.line([(0, y), (W, y)], fill=(255, 255, 255, 14), width=2)
    elif kind == "dots":
        for y in range(30, H, 70):
            for x in range(30 + (35 if (y // 70) % 2 else 0), W, 70):
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=tuple(colour) + (40,))
    out = bg.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


class Painter:
    def __init__(self, recipe: dict):
        theme = THEMES[recipe["theme"]]
        self.recipe = recipe
        self.chain = theme["chain"]
        self.skin = recipe.get("skin") or "bubble"
        self.emoji = [Image.open(ASSETS / "emoji" / f"{c}.webp").convert("RGBA")
                      for c, _ in self.chain]
        self.colours = [_tint(e) for e in self.emoji]
        top, bot = theme.get("bg") or _auto_bg(self.colours[-1])
        g = np.linspace(0, 1, H)[:, None]
        rows = (np.array(top) * (1 - g) + np.array(bot) * g).astype(np.uint8)
        self.bg = _backdrop(Image.fromarray(np.repeat(rows[:, None, :], W, axis=1), "RGB"),
                            recipe.get("backdrop") or "gradient", self.colours[-1], recipe.get("seed") or 0)
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
        self.f_gate = _font(46)

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
            e = int(R * 1.45)
            if self.skin == "bubble":
                d.ellipse((c - R, c - R, c + R, c + R), fill=col + (235,))
                d.ellipse((c - R, c - R, c + R, c + R),
                          outline=tuple(int(v * 0.6) for v in col) + (255,), width=ss * 2)
            elif self.skin == "glass":
                d.ellipse((c - R, c - R, c + R, c + R), fill=col + (70,))
                d.ellipse((c - R, c - R, c + R, c + R), outline=(255, 255, 255, 200), width=ss * 2)
                e = int(R * 1.3)
            elif self.skin == "neon":
                d.ellipse((c - R, c - R, c + R, c + R), fill=(24, 22, 36, 235))
                d.ellipse((c - R, c - R, c + R, c + R), outline=col + (255,), width=ss * 3)
                e = int(R * 1.3)
            else:                                      # plain: just the emoji
                e = int(R * 2.0)
            im.alpha_composite(self.emoji[tier].resize((e, e), Image.LANCZOS),
                               (int(c - e / 2), int(c - e / 2)))
            # the shine goes on its own layer: drawn straight onto im it
            # would replace the pixels and leave a see-through grey spot
            hr = R * 0.25 if self.skin != "plain" else 0
            shine = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            ImageDraw.Draw(shine).ellipse((c - R * 0.5 - hr, c - R * 0.55 - hr, c - R * 0.5 + hr,
                                           c - R * 0.55 + hr), fill=(255, 255, 255, 90))
            im.alpha_composite(shine)
            self.sprites[key] = im.resize((s // ss, s // ss), Image.LANCZOS)
        return self.sprites[key]

    def frame(self, sim: Sim) -> Image.Image:
        im = self.bg.copy()
        d = ImageDraw.Draw(im, "RGBA")
        glass = (200, 220, 255, 150)
        for s in sim.parts:
            if getattr(s, "part", "") == "gate":
                self.gate(d, s)
            elif isinstance(s, sim.pymunk.Segment):
                w = 14 if s.part == "ramp" else 10
                d.line([s.a, s.b], fill=(255, 255, 255, 190) if s.part == "ramp" else glass, width=w)
            else:                                     # bumper
                (x, y), r = s.offset, s.radius
                d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 30),
                          outline=(140, 220, 255, 220), width=6)
        for sp in sim.spinners:
            b, h = sp["body"], sp["half"]
            rot = sp.get("rot", 0.0)
            dx, dy = h * math.cos(rot), h * math.sin(rot)
            a, c = b.local_to_world((-dx, -dy)), b.local_to_world((dx, dy))
            if sp.get("wheel"):
                x, y = b.position
                if rot == 0.0:                       # the rim, once per wheel
                    d.ellipse((x - h - 8, y - h - 8, x + h + 8, y + h + 8),
                              outline=self.colours[-1] + (90,), width=6)
                d.line([a, c], fill=self.colours[-1] + (235,), width=18)
                for e in (a, c):
                    d.ellipse((e[0] - 13, e[1] - 13, e[0] + 13, e[1] + 13), fill=(255, 255, 255, 235))
                if rot == 0.0:
                    d.ellipse((x - 70, y - 70, x + 70, y + 70), fill=self.colours[-1] + (255,),
                              outline=(255, 255, 255, 240), width=8)
                continue
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

        for p in getattr(sim, "pops", []):          # "x2" floating up from a gate
            x, y, label, life = p
            f = self.f_gate
            w = d.textlength(label, font=f)
            yy = y - 40 + (1 - life) * 40               # drifts down with the copies
            d.text((x - w / 2 + 2, yy + 2), label, font=f, fill=(0, 0, 0, int(140 * life)))
            d.text((x - w / 2, yy), label, font=f, fill=(140, 255, 160, int(255 * life)))

        self.header(im, d, sim)
        unit = THEMES[self.recipe["theme"]]["unit"]
        verb = "made" if getattr(sim, "gate_course", False) else "dropped"
        g = sim.geo
        d.text((g["text_l"], g["text_y"]), f"{sim.spawned:,} {unit} {verb}", font=self.f_small,
               fill=(220, 230, 255, 255))
        self.watermark(d, g["text_r"], g["text_y"])
        self.overlays(im, d, sim)
        return im

    def gate(self, d, g):
        """A multiplier gate: a glowing bar with its number on it."""
        (x0, y), (x1, _) = g.a, g.b
        plus = isinstance(g.mult, str)
        col = (90, 170, 255) if plus else (80, 230, 120)
        fl = g.flash
        d.rounded_rectangle((x0, y - 34, x1, y + 10), 12, fill=col + (int(60 + 90 * fl),),
                            outline=col + (230,), width=4)
        label = g.mult if plus else f"×{g.mult}"
        w = d.textlength(label, font=self.f_gate)
        d.text(((x0 + x1) / 2 - w / 2, y - 37), label, font=self.f_gate, fill=(255, 255, 255, 245))
        g.flash *= 0.85

    # Shared by every Ball Evolution format (see ball_circles.CirclePainter).
    def ctext(self, d, y, s, f, fill=(255, 255, 255, 255)):
        w = d.textlength(s, font=f)
        d.text(((W - w) / 2 + 3, y + 3), s, font=f, fill=(0, 0, 0, 150))
        d.text(((W - w) / 2, y), s, font=f, fill=fill)

    def header(self, im, d, sim):
        """The hook line and the ladder that reveals each item."""
        hook = self.recipe["hook"]
        f = self.f_hook
        if d.textlength(hook, font=f) > W - 50:        # a long hook shrinks to fit
            f = _font(max(34, int(54 * (W - 50) / d.textlength(hook, font=f))))
        self.ctext(d, 30, hook, f)
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

    def watermark(self, d, right, y):
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((right - d.textlength(mark, font=self.f_small), y), mark,
                   font=self.f_small, fill=(255, 255, 255, 150))

    def overlays(self, im, d, sim, item_y=740, text_y=1000):
        """The "NEW: ...!" banner, and the last item + ending line at the end."""
        if sim.banner and sim.done_at is None and sim.t - sim.banner[1] < 1.6:
            age = sim.t - sim.banner[1]
            a = min(1.0, age / 0.15) * min(1.0, (1.6 - age) / 0.3)
            bw = d.textlength(sim.banner[0], font=self.f_banner) + 60
            d.rounded_rectangle(((W - bw) / 2, 236, (W + bw) / 2, 316), 40,
                                fill=(20, 14, 40, int(220 * a)))
            self.ctext(d, 240, sim.banner[0], self.f_banner, (255, 225, 110, int(255 * a)))
        if sim.done_at is not None:
            age = sim.t - sim.done_at
            s = int(260 * min(1.0, age / 0.5))
            if s > 4:
                big = self.emoji[-1].resize((s, s), Image.LANCZOS)
                im.paste(big, (int(W / 2 - s / 2), int(item_y - s / 2)), big)
            ending = self.recipe["ending"]
            f = self.f_big if d.textlength(ending, font=self.f_big) < W - 80 else _font(100)
            self.ctext(d, text_y, ending, f, (255, 225, 120, int(255 * min(1.0, age / 0.4))))


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


def _ring(freqs, amps, dur, decay, vol, attack=0.001) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    w = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    return (w * np.exp(-t * decay) * np.minimum(1, t / attack) * vol).astype(np.float32)


def _add(*parts) -> np.ndarray:
    """Sum sound clips of different lengths (padded to the longest)."""
    out = np.zeros(max(len(p) for p in parts), dtype=np.float32)
    for p in parts:
        out[:len(p)] += p
    return out


def _pack_hit(pack: str, pos: float, take: int, tone: float) -> np.ndarray:
    """One bounce in a sound pack. pos 0..1 (left..right) nudges the pitch;
    take varies the noise so repeats don't sound identical. All short and
    percussive -- no melody (Dean found the tonal version horrible)."""
    seed = (SOUNDS.index(pack) if pack in SOUNDS else 0, int(pos * 9), take)
    j = 1 + (take - 1) * 0.03                      # tiny per-take pitch jitter
    if pack == "glass":      # a bright clink: inharmonic partials, quick ring
        f = (2600 + 1600 * pos) * tone * j
        return _add(_ring((f, f * 2.76, f * 5.4), (1, 0.4, 0.15), 0.18, 32, 0.07), _burst(0.01, 4000, 9000, 400, 0.05, seed))
    if pack == "wood":       # a dry tok
        f = (550 + 400 * pos) * tone * j
        return _add(_ring((f, f * 2.3), (1, 0.3), 0.08, 70, 0.14), _burst(0.03, 600, 1800, 140, 0.10, seed))
    if pack == "plastic":    # a tiny click
        return _burst(0.025, (1800 + 1500 * pos) * tone, 6000, 260, 0.20, seed)
    if pack == "rubber":     # a soft bonk with a little pitch drop
        t = np.arange(int(SR * 0.12)) / SR
        f = (170 + 140 * pos) * tone * j * (1 + 0.5 * np.exp(-t * 60))
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 32) * 0.22).astype(np.float32)
    if pack == "water":      # a bloop: pitch rising fast
        t = np.arange(int(SR * 0.09)) / SR
        f = (320 + 260 * pos) * tone * j * (1 + 1.6 * (1 - np.exp(-t * 45)))
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 34) * np.minimum(1, t / 0.004) * 0.16).astype(np.float32)
    if pack == "metal":      # a light ting
        f = (950 + 900 * pos) * tone * j
        return _ring((f, f * 1.48, f * 2.83, f * 4.1), (1, 0.6, 0.35, 0.2), 0.3, 18, 0.06)
    if pack == "pop":        # a bubble pop
        f = (650 + 500 * pos) * tone * j
        return _add(_ring((f,), (1,), 0.05, 90, 0.12), _burst(0.03, 1000, 4000, 180, 0.10, seed))
    lo = (1400 + 220 * pos * 9) * tone           # marble: the original noise tap
    return _burst(0.05, lo, lo * 2.3, 150, 0.22, seed)


class Voice:
    """The video's sounds: soft, noise-based ASMR taps, puffs and whooshes
    over a quiet bed of noise (rain, air or hush, picked per video).
    Dean found the first musical version (marimba notes, chimes, a music
    loop) "horrible" and asked for "more white noise"."""

    BEDS = ("rain", "air", "hush")
    BED_OPTIONS = ("auto",) + BEDS + ("off",)   # recipe["bed"]; "off" = only the balls' own sounds

    def __init__(self, recipe: dict):
        pick = random.Random(f"bed-{recipe.get('seed')}-{recipe.get('key')}-{recipe.get('scale')}")
        self.bed_kind = pick.choice(self.BEDS)
        if recipe.get("bed") in self.BEDS + ("off",):
            self.bed_kind = recipe["bed"]
        self.tone = pick.uniform(0.85, 1.15)     # shifts every sound's pitch a little per video
        self.sound = recipe.get("sound") if recipe.get("sound") in SOUNDS else "marble"
        self._cache: dict = {}
        self._n = 0

    def _get(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def tick(self, peg_note: int) -> np.ndarray:
        """A ball hitting a peg, in the video's sound pack (``recipe["sound"]``),
        a little higher toward the right. Three takes per peg so repeats
        don't sound copied."""
        self._n += 1
        take = self._n % 3
        pos = peg_note / 9.0                       # 0 left .. 1 right
        return self._get(("tick", peg_note, take), lambda: _pack_hit(self.sound, pos, take, self.tone))

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
        if self.bed_kind == "off":
            return np.zeros(n, dtype=np.float32)
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
    sim_cls, painter_cls = _parts(recipe)
    sim, painter = sim_cls(recipe), painter_cls(recipe)
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
    ap.add_argument("--jar", choices=JARS + OLD_JARS)
    ap.add_argument("--format", choices=FORMATS, default="evolve")
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
        recipe = pick_recipe(seed, history, args.theme, args.course, args.jar, args.format)
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
        history.append({k: info[k] for k in ("seed", "theme", "course", "jar", "key", "scale", "format", "skin", "backdrop", "sound")})
        Path(args.history).write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
