"""Ball Evolution: a satisfying physics Short, rendered from scratch.

Bees drip slowly out of a hole at the top and fall through a field of
pegs. Hitting a gold peg sends a copy of the bee back out of the hole, so
the drip turns into a flood. In the jar at the bottom, two of the same
animal that touch merge into the next, bigger one:

    bee > mouse > frog > chicken > cat > dog > panda > lion > unicorn

The ladder at the top hides each animal until it's first made; the video
ends on the unicorn. Everything on screen and every sound is made here
(bundled MIT emoji, OFL fonts, generated audio), so there is nothing to
license or get claimed.

Runs differ by seed, and some stall (two lions that never touch), so
``pick_seed`` simulates candidates without drawing (fast, in parallel)
and keeps one that reaches the unicorn at a good pace before rendering.

    python -m clipper.ball_evolution out.mp4            # pick a seed, render
    python -m clipper.ball_evolution out.mp4 --seed 7   # render one seed
"""
from __future__ import annotations

import argparse
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

# ---- the machine -----------------------------------------------------
HOLE = (540, 330)
WALL_L, WALL_R = 70, 1010
NECK_L, NECK_R, NECK_Y = 400, 680, 1215
JAR_L, JAR_R, JAR_TOP, JAR_BOT = 110, 970, 1320, 1830
PEG_R = 9
GRAVITY = 520            # low gravity: a slow, readable fall
GOLD_SHARE = 0.24        # share of pegs that multiply
MAX_ABOVE_NECK = 90      # the hole waits while this many are still falling
SPAWN_CAP = 2500
PULL_FROM_TIER = 2       # frogs and up attract their match in the jar
PULL = 900.0             # px/s^2, a bit more than gravity
SECONDS_CAP = 90.0
END_HOLD = 4.0

TIERS = [  # (emoji file in assets/emoji, name, bubble colour)
    ("1f41d", "Bee", (255, 214, 90)),
    ("1f400", "Mouse", (190, 190, 205)),
    ("1f438", "Frog", (120, 220, 110)),
    ("1f414", "Chicken", (255, 150, 120)),
    ("1f431", "Cat", (255, 190, 90)),
    ("1f436", "Dog", (210, 160, 110)),
    ("1f43c", "Panda", (235, 235, 245)),
    ("1f981", "Lion", (255, 170, 50)),
    ("1f984", "Unicorn", (230, 140, 255)),
]
RADII = [17 * 1.29 ** k for k in range(len(TIERS))]   # 17 .. ~131 px
LAST = len(TIERS) - 1

BG_TOP, BG_BOT = (14, 12, 30), (34, 18, 52)
HOOK = "Can it make the last one?"

# A good run: unicorn between these times, no long wait between reveals.
GOOD_END = (40.0, 62.0)
MAX_UNLOCK_GAP = 16.0


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


def _font(size: int, name: str = "Inter-Black.ttf"):
    try:
        return ImageFont.truetype(str(ASSETS / "fonts" / name), size)
    except OSError:
        return ImageFont.load_default()


# ---- simulation ------------------------------------------------------

def _build(seed: int):
    import pymunk

    rng = random.Random(seed)
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.iterations = 20
    static = space.static_body

    walls = []
    for a, b in [
        ((WALL_L, 380), (WALL_L, 1030)), ((WALL_R, 380), (WALL_R, 1030)),
        ((WALL_L, 1030), (NECK_L, NECK_Y)), ((WALL_R, 1030), (NECK_R, NECK_Y)),
        ((NECK_L, NECK_Y), (JAR_L, JAR_TOP)), ((NECK_R, NECK_Y), (JAR_R, JAR_TOP)),
        ((JAR_L, JAR_TOP), (JAR_L, JAR_BOT)), ((JAR_R, JAR_TOP), (JAR_R, JAR_BOT)),
        ((JAR_L, JAR_BOT), (JAR_R, JAR_BOT)),
    ]:
        s = pymunk.Segment(static, a, b, 6)
        s.elasticity, s.friction = 0.35, 0.5
        space.add(s)
        walls.append(s)

    pegs = []
    for row in range(8):
        y = 450 + row * 72
        x = WALL_L + 40 + (0 if row % 2 == 0 else 46)
        while x < WALL_R - 30:
            p = pymunk.Circle(static, PEG_R, (x, y))
            p.elasticity, p.friction = 0.55, 0.3
            p.collision_type = 2
            space.add(p)
            pegs.append({"shape": p, "pos": (x, y), "gold": rng.random() < GOLD_SHARE,
                         "flash": 0.0})
            x += 92
    return rng, space, walls, pegs


class Sim:
    """One run of the machine. ``step()`` advances one video frame."""

    def __init__(self, seed: int):
        import pymunk

        self.pymunk = pymunk
        self.rng, self.space, self.walls, self.pegs = _build(seed)
        self.peg_by_shape = {p["shape"]: p for p in self.pegs}
        self.balls: list[dict] = []
        self.queue = [0]
        self.events: list[tuple] = []     # (t, kind, value) for the audio
        self.effects: list[list] = []     # merge rings
        self.reveal_flash = [0.0] * len(TIERS)
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
        self.space.on_collision(1, 1, pre_solve=self._on_ball)

    def _add_ball(self, tier, pos, vel, r_start=None):
        pm = self.pymunk
        r0 = r_start or RADII[tier]
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
        if peg["gold"] and b["tier"] == 0 and id(peg) not in b["hit"]:
            b["hit"].add(id(peg))
            self._multiplied.append(peg)

    def _on_ball(self, arbiter, space, data):
        a, c = (getattr(s, "ball", None) for s in arbiter.shapes)
        # animals merge only once they're in the jar
        if (a and c and a["tier"] == c["tier"] and a["tier"] < LAST
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
            nb = self._add_ball(tier, mid, vel, r_start=RADII[tier - 1])
            nb["landed"] = True
            self.effects.append([mid[0], mid[1], RADII[tier], 1.0, TIERS[tier][2]])
            self.events.append((self.t, "pop", tier))
            if tier > self.best:
                self.best = tier
                self.unlocked_at[tier] = self.t
                self.reveal_flash[tier] = 1.0
                self.banner = (f"NEW: {TIERS[tier][1]}!", self.t)
                self.events.append((self.t, "unlock", tier))
        self._merges.clear()

    def _pull_pairs(self):
        """Matching animals in the jar drift toward each other.

        Without this, two lions can settle on opposite sides and the run
        stalls. Bees and mice are left alone; there are plenty of them.
        """
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
                    b["body"].apply_force_at_world_point(
                        v.normalized() * b["body"].mass * PULL, p)

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
            self._pull_pairs()     # pymunk clears forces after every step
            self.space.step(dt / 4)
            self._apply_merges()       # never inside the collision callback
        self.balls[:] = [b for b in self.balls if b["alive"]]

        for b in self.balls:           # freshly merged animals grow in ~0.2 s
            full = RADII[b["tier"]]
            if b["r"] < full:
                b["r"] = min(full, b["r"] + (full - RADII[max(0, b["tier"] - 1)]) / 12)
                b["shape"].unsafe_set_radius(b["r"])
            if not b["landed"] and b["body"].position.y > JAR_TOP - 40:
                b["landed"] = True

        for _ in self._multiplied:
            if self.spawned + len(self.queue) < SPAWN_CAP:
                self.queue.append(0)
                self.events.append((self.t, "note", self.spawned + len(self.queue)))
        self._multiplied.clear()
        if self.done_at is None and not self.queue and all(b["landed"] for b in self.balls):
            self.queue.append(0)       # the drip never dies out

        if self.done_at is None and self.best == LAST:
            self.done_at = self.t
            self.queue.clear()
            self.events.append((self.t, "fanfare", 0))
        self.t += dt

    @property
    def finished(self) -> bool:
        return ((self.done_at is not None and self.t - self.done_at > END_HOLD)
                or self.t > SECONDS_CAP)


# ---- picking a good run ----------------------------------------------

def simulate(seed: int) -> dict:
    """Physics only, no drawing: how this seed plays out."""
    sim = Sim(seed)
    while not sim.finished:
        sim.step()
        if sim.done_at is not None:
            break
        if sim.t > GOOD_END[1] + 1:
            break
    times = sorted(sim.unlocked_at.values())
    gap = max((b - a for a, b in zip(times, times[1:])), default=99.0)
    return {"seed": seed, "done_at": sim.done_at, "max_gap": round(gap, 1),
            "unlocks": {TIERS[k][1]: round(v, 1) for k, v in sorted(sim.unlocked_at.items())}}


def is_good(r: dict) -> bool:
    return (r["done_at"] is not None and GOOD_END[0] <= r["done_at"] <= GOOD_END[1]
            and r["max_gap"] <= MAX_UNLOCK_GAP)


def pick_seed(start: int = 1, tries: int = 48, workers: int | None = None,
              log=print) -> int:
    """Simulate seeds in parallel; return the good one closest to ~52 s."""
    workers = workers or max(1, (os.cpu_count() or 2))
    good = []
    with ProcessPoolExecutor(workers) as pool:
        for lo in range(start, start + tries, workers):
            batch = list(range(lo, min(lo + workers, start + tries)))
            for r in pool.map(simulate, batch):
                log(f"seed {r['seed']}: unicorn at {r['done_at']}, "
                    f"longest wait {r['max_gap']} s{'  <- good' if is_good(r) else ''}")
                if is_good(r):
                    good.append(r)
            if good:
                break
    if not good:
        raise RuntimeError(f"no seed in {start}..{start + tries - 1} made a good run")
    return min(good, key=lambda r: abs(r["done_at"] - 52))["seed"]


# ---- drawing ---------------------------------------------------------

class Painter:
    def __init__(self):
        g = np.linspace(0, 1, H)[:, None]
        rows = (np.array(BG_TOP) * (1 - g) + np.array(BG_BOT) * g).astype(np.uint8)
        self.bg = Image.fromarray(np.repeat(rows[:, None, :], W, axis=1), "RGB")
        self.emoji = [Image.open(ASSETS / "emoji" / f"{c}.webp").convert("RGBA")
                      for c, _, _ in TIERS]
        self.icons = [e.resize((78, 78), Image.LANCZOS) for e in self.emoji]
        self.sils = []
        for e in self.icons:
            sil = Image.new("RGBA", e.size, (40, 32, 70, 0))
            sil.putalpha(e.getchannel("A").point(lambda v: int(v * 0.85)))
            self.sils.append(sil)
        self.sprites: dict = {}
        self.f_hook, self.f_banner = _font(58), _font(64)
        self.f_big, self.f_small = _font(150), _font(36, "Inter-Bold.ttf")

    def sprite(self, tier: int, r: float) -> Image.Image:
        r = max(4, int(round(r)))
        key = (tier, r)
        if key not in self.sprites:
            ss = 3
            s = (2 * r + 4) * ss
            im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            c, R = s / 2, r * ss
            col = TIERS[tier][2]
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
        for s in sim.walls:
            d.line([s.a, s.b], fill=(200, 220, 255, 150), width=10)
        hr = 34 + 8 * sim.hole_pulse
        d.ellipse((HOLE[0] - hr, HOLE[1] - hr * 0.55, HOLE[0] + hr, HOLE[1] + hr * 0.55),
                  fill=(0, 0, 0, 255), outline=(255, 210, 90, 220), width=5)
        sim.hole_pulse *= 0.85
        R = PEG_R
        for p in sim.pegs:
            x, y = p["pos"]
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
            x, y, r, life, col = e
            rr = r * (1.0 + (1 - life) * 0.9)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=col + (int(230 * life),),
                      width=max(2, int(8 * life)))
            e[3] -= 0.06
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]

        def ctext(y, s, f, fill=(255, 255, 255, 255)):
            w = d.textlength(s, font=f)
            d.text(((W - w) / 2 + 3, y + 3), s, font=f, fill=(0, 0, 0, 150))
            d.text(((W - w) / 2, y), s, font=f, fill=fill)

        ctext(30, HOOK, self.f_hook)
        cell, top = 104, 128
        x0 = (W - cell * len(TIERS)) / 2
        for i in range(len(TIERS)):
            cx = x0 + i * cell + cell / 2
            on, fl = i <= sim.best, sim.reveal_flash[i]
            d.rounded_rectangle((cx - 46, top - 6, cx + 46, top + 86), 16,
                                fill=(255, 255, 255, 40 if on else 14),
                                outline=(255, 215, 100, int(80 + 175 * fl)) if on
                                else (255, 255, 255, 40), width=3 + int(4 * fl))
            ic = self.icons[i] if on else self.sils[i]
            if on and fl > 0:
                s = int(78 * (1 + 0.35 * fl))
                ic = self.emoji[i].resize((s, s), Image.LANCZOS)
            im.paste(ic, (int(cx - ic.width / 2), int(top + 40 - ic.height / 2)), ic)
            if not on:
                d.text((cx - 10, top + 20), "?", font=self.f_small, fill=(255, 255, 255, 130))
            if i < len(TIERS) - 1:
                d.text((cx + 44, top + 22), "›", font=self.f_small, fill=(255, 255, 255, 90))
            sim.reveal_flash[i] *= 0.93

        d.text((JAR_L, JAR_BOT + 18), f"{sim.spawned:,} bees dropped", font=self.f_small,
               fill=(220, 230, 255, 255))

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
                big = self.emoji[LAST].resize((s, s), Image.LANCZOS)
                im.paste(big, (int(W / 2 - s / 2), int(760 - s / 2)), big)
            ctext(1010, "EVOLVED!", self.f_big, (255, 225, 120, int(255 * min(1.0, age / 0.4))))
        return im


# ---- audio (all generated) -------------------------------------------

def _tone(freq, dur, decay, vol, harm=(1.0, 0.3, 0.1)):
    t = np.arange(int(SR * dur)) / SR
    w = sum(a * np.sin(2 * np.pi * freq * (i + 1) * t) for i, a in enumerate(harm))
    return (w * np.exp(-t * decay) * np.minimum(1, t / 0.005) * vol).astype(np.float32)


def _pling(k: int) -> np.ndarray:
    """Soft bell per multiply, climbing a pentatonic scale as bees pile up."""
    scale = [f * 2 ** o for o in range(3) for f in (261.63, 293.66, 329.63, 392.0, 440.0)]
    f = scale[min(len(scale) - 1, int(math.log2(max(1, k)) * 1.3))]
    return _tone(f, 0.55, 7, 0.10, harm=(1, 0.35, 0.12))


def _pop(tier: int) -> np.ndarray:
    """Bubbly pop + a note; bigger animals sound deeper and fuller."""
    t = np.arange(int(SR * 0.5)) / SR
    sweep = 880 / (1.22 ** tier) * (1 + 0.6 * np.exp(-t * 40))
    body = np.sin(2 * np.pi * np.cumsum(sweep) / SR) * np.exp(-t * (14 - tier)) * (0.22 + 0.03 * tier)
    click = np.random.default_rng(tier).standard_normal(len(t)) * np.exp(-t * 300) * 0.08
    return (body + click).astype(np.float32)


def _chime(tier: int) -> np.ndarray:
    base = 523.25 * 2 ** ((tier % 5) * 2 / 12)
    out = np.zeros(int(SR * 1.2), dtype=np.float32)
    for i, mult in enumerate((1, 1.25, 1.5, 2)):
        n = _tone(base * mult, 0.9, 5, 0.16)
        o = int(i * 0.08 * SR)
        out[o:o + len(n)] += n[:len(out) - o]
    return out


def _fanfare() -> np.ndarray:
    out = np.zeros(int(SR * 3.0), dtype=np.float32)
    for i, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
        n = _tone(f, 2.6, 1.6, 0.14, harm=(1, 0.5, 0.25, 0.1))
        o = int(i * 0.11 * SR)
        out[o:o + len(n)] += n[:len(out) - o]
    return out


def _music(seconds: float, bpm: int = 100) -> np.ndarray:
    """Soft I-vi-IV-V loop: warm pad, plucked arpeggio, gentle kick."""
    out = np.zeros(int(SR * seconds), dtype=np.float32)
    beat = 60 / bpm
    bar = beat * 4
    chords = [(261.63, 329.63, 392.00), (220.00, 261.63, 329.63),
              (174.61, 220.00, 261.63), (196.00, 246.94, 293.66)]
    tb = np.arange(int(SR * bar)) / SR
    tk = np.arange(int(SR * 0.18)) / SR
    kick = (np.sin(2 * np.pi * (55 + 90 * np.exp(-tk * 30)) * tk) * np.exp(-tk * 18) * 0.10).astype(np.float32)

    def add(at, w):
        seg = out[at:at + len(w)]
        seg += w[:len(seg)]

    k = 0
    while k * bar < seconds:
        ch, start = chords[k % 4], int(k * bar * SR)
        pad = sum(np.sin(2 * np.pi * f * tb) + 0.3 * np.sin(2 * np.pi * f * 2.005 * tb) for f in ch)
        add(start, (pad * np.minimum(1, tb / 0.4) * np.minimum(1, (bar - tb) / 0.4) * 0.025).astype(np.float32))
        for s in range(8):
            add(start + int(s * beat / 2 * SR),
                _tone(ch[s % 3] * (2 if s % 4 == 3 else 1) * 2, 0.35, 9, 0.035, harm=(1, 0.2)))
        for b in range(4):
            add(start + int(b * beat * SR), kick)
        k += 1
    return out


def mix_audio(sim: Sim) -> np.ndarray:
    n = int(SR * (sim.t + 1))
    fx = np.zeros(n, dtype=np.float32)
    last_note = last_pop = -1.0
    for et, kind, v in sim.events:
        if kind == "note":
            if et - last_note < 0.06:     # a flood stays musical, not noise
                continue
            last_note, w = et, _pling(v)
        elif kind == "pop":
            if et - last_pop < 0.03 and v < 3:
                continue
            last_pop, w = et, _pop(v)
        elif kind == "unlock":
            w = _chime(v)
        else:
            w = _fanfare()
        i = int(et * SR)
        seg = fx[i:i + len(w)]
        seg += w[:len(seg)]
    fx = np.tanh(fx * 1.4) * 0.85
    music = _music(sim.t + 1)[:n]
    if sim.done_at is not None:           # music dips under the fanfare
        s0 = int(sim.done_at * SR)
        music[s0:] *= np.maximum(0.25, 1 - (np.arange(n - s0) / SR)).astype(np.float32)
    mix = fx + music * 0.9
    return mix / max(1e-6, float(np.max(np.abs(mix)))) * 0.89


# ---- render ----------------------------------------------------------

def render(seed: int, out_path: str | os.PathLike, log=print) -> dict:
    out_path = Path(out_path)
    video = out_path.with_suffix(".video.mp4")
    wav = out_path.with_suffix(".wav")
    sim, painter = Sim(seed), Painter()
    proc = subprocess.Popen(
        [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "23", "-pix_fmt", "yuv420p", str(video)],
        stdin=subprocess.PIPE)
    try:
        while not sim.finished:
            sim.step()
            proc.stdin.write(painter.frame(sim).tobytes())
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
    info = {"seed": seed, "seconds": round(sim.t, 1), "bees": sim.spawned,
            "done_at": sim.done_at and round(sim.done_at, 1),
            "unlocks": {TIERS[k][1]: round(v, 1) for k, v in sorted(sim.unlocked_at.items())}}
    log(f"rendered {out_path}: {info}")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output .mp4")
    ap.add_argument("--seed", type=int, help="render this seed (default: pick a good one)")
    ap.add_argument("--start", type=int, default=1, help="first seed to try when picking")
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else pick_seed(args.start)
    render(seed, args.out)


if __name__ == "__main__":
    main()
