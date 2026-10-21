"""Ball Evolution's "merge into final form" format, with a song.

Dean sent a music.bouncingball reel (308K likes): small cut-out meme
pictures drop down a chute into a rainbow ring, two of the same merge into
the next, and it builds to a final form. Prototyped in the sandbox, then
built in here.

* Items drop one by one from a capped chute (the cap stops big items
  bouncing back out). Small ones at first; bigger ones join once they've
  been revealed, which keeps a 9-step chain to ~30-45 s.
* Two of the same touching merge into the next one; from the 4th step up
  the bigger ones drift toward their match, so a run never stalls.
* Every hit on the ring plays the next note of a public-domain tune on a
  soft piano (``recipe["tune"]``), at most ~7 notes a second, so the tune
  speeds up as more things bounce. Merges, reveals and the final boom keep
  the usual noise sounds; there's no noise bed.
* The pictures are a pack from clipper/ball_packs.py (``recipe["pack"]``):
  stickers drawn as cut-outs, plain photos in a circle with a white rim.
  Without a pack the theme's emoji are used.
"""
from __future__ import annotations

import colorsys
import math
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from clipper import ball_evolution as be

W, H, FPS, SR = be.W, be.H, be.FPS, be.SR
CX, CY, R = 540, 1060, 430
CHUTE_HW = 70
CHUTE_TOP = 330
RING_TOP = CY - math.sqrt(R ** 2 - CHUTE_HW ** 2)
R0, GROW = 30, 1.2
DROP_EVERY = 0.3
MAX_DROP_TIER = 4
# The pace per chain length (seconds between drops, biggest tier dropped),
# measured so 6-9 pictures all reach the final form in ~26-44 s: with one
# pace for all, a 7-picture pack finished in ~15 s and no seed was good.
PACE = {9: (0.3, 4), 8: (0.38, 3), 7: (0.5, 2), 6: (0.85, 1)}
BOUNCE = 0.92
NOTE_GAP = 0.14
END_HOLD = 4.0
SECONDS_CAP = 80.0


def _m(names: str) -> list:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    out = []
    for n in names.split():
        step, rest, acc = n[0], n[1:], 0
        while rest and rest[0] in "#b":
            acc += 1 if rest[0] == "#" else -1
            rest = rest[1:]
        out.append(12 * (int(rest) + 1) + base[step] + acc)
    return out


# All long out of copyright (the melodies themselves, written here note by
# note), so nothing can be claimed.
TUNES = {
    "mountain_king": _m("B3 C#4 D4 E4 F#4 D4 F#4 E#4 C#4 E#4 E4 C4 E4 "      # Grieg, 1875
                        "B3 C#4 D4 E4 F#4 D4 F#4 B4 A4 F#4 D4 F#4 A4 "
                        "F#4 G#4 A#4 B4 C#5 A#4 C#5 D5 A#4 D5 C#5 A#4 C#5 "
                        "F#4 G#4 A#4 B4 C#5 A#4 C#5 D5 A#4 D5 C#5"),
    "ode_to_joy": _m("E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4 "        # Beethoven, 1824
                     "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4 C4 C4 "
                     "D4 D4 E4 C4 D4 E4 F4 E4 C4 D4 E4 F4 E4 D4 C4 D4 G3 "
                     "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4 C4 C4"),
    "fur_elise": _m("E5 D#5 E5 D#5 E5 B4 D5 C5 A4 C4 E4 A4 B4 E4 G#4 B4 C5 "  # Beethoven, 1810
                    "E4 E5 D#5 E5 D#5 E5 B4 D5 C5 A4 C4 E4 A4 B4 E4 C5 B4 A4"),
    "turkish_march": _m("B4 A4 G#4 A4 C5 D5 C5 B4 C5 E5 F5 E5 D#5 E5 "       # Mozart, 1784
                        "B5 A5 G#5 A5 B5 A5 G#5 A5 C6 A5 C6 B5 A5 G5 A5 "
                        "B5 A5 G5 A5 B5 A5 G5 A5 B5 A5 G5 F#5 E5"),
    "twinkle": _m("C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4 "              # folk tune
                  "G4 G4 F4 F4 E4 E4 D4 G4 G4 F4 F4 E4 E4 D4 "
                  "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4"),
}
SEASON_TUNES = {12: ["jingle_bells"]}
TUNES["jingle_bells"] = _m("E4 E4 E4 E4 E4 E4 E4 G4 C4 D4 E4 "                # Pierpont, 1857
                           "F4 F4 F4 F4 F4 E4 E4 E4 E4 D4 D4 E4 D4 G4 "
                           "E4 E4 E4 E4 E4 E4 E4 G4 C4 D4 E4 "
                           "F4 F4 F4 F4 F4 E4 E4 E4 G4 G4 F4 D4 C4")
EVERYDAY_TUNES = ["mountain_king", "ode_to_joy", "fur_elise", "turkish_march", "twinkle"]


def piano(midi: int, dur: float = 0.9) -> np.ndarray:
    """A soft piano-like note: decaying harmonics and a little hammer noise."""
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 9):
        fk = f * k * (1 + 0.0004 * k * k)
        if fk > SR / 2.2:
            break
        out += (1 / k ** 1.6) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (2.2 + 1.3 * k))
    attack = np.minimum(1, t / 0.004)
    hammer = np.random.default_rng(midi).normal(0, 1, n) * np.exp(-t * 90) * 0.04
    return ((out * attack + hammer) * 0.32).astype(np.float32)


def chain_names(recipe: dict) -> list:
    pack = recipe.get("pack")
    if pack:
        return [i["name"] for i in pack["items"]]
    return [name for _, name in be.THEMES[recipe["theme"]]["chain"]]


class MergeSim:
    def __init__(self, recipe: dict):
        import pymunk

        self.pymunk = pymunk
        self.recipe = recipe
        self.chain = [(None, n) for n in chain_names(recipe)]
        self.last = len(self.chain) - 1
        self.drop_every, self.max_drop_tier = PACE.get(len(self.chain), (DROP_EVERY, MAX_DROP_TIER))
        self.rng = random.Random(f"merge-{recipe['seed']}")
        sp = self.space = pymunk.Space()
        sp.gravity = (0, 900)
        sp.iterations = 20
        a = math.asin(CHUTE_HW / R)
        a0, span, n = -math.pi / 2 + a, 2 * math.pi - 2 * a, 90
        self.ring_pts = [(CX + R * math.cos(a0 + span * i / n), CY + R * math.sin(a0 + span * i / n))
                         for i in range(n + 1)]
        segs = list(zip(self.ring_pts, self.ring_pts[1:]))
        segs += [((CX - CHUTE_HW, CHUTE_TOP), (CX - CHUTE_HW, RING_TOP + 2)),
                 ((CX + CHUTE_HW, CHUTE_TOP), (CX + CHUTE_HW, RING_TOP + 2)),
                 ((CX - CHUTE_HW, CHUTE_TOP), (CX + CHUTE_HW, CHUTE_TOP))]      # the cap: nothing flies out
        for p, q in segs:
            s = pymunk.Segment(sp.static_body, p, q, 6)
            s.elasticity, s.friction, s.collision_type = 1.0, 0.2, 3
            sp.add(s)
        self.balls: list = []          # "balls" so the shared code (render info) works
        self.events: list = []
        self.hits: list = []
        self.effects: list = []
        self.reveal_flash = [0.0] * len(self.chain)
        self.unlocked_at = {0: 0.0}
        self.banner = None
        self.best = 0
        self.t, self.done_at, self.next_drop = 0.0, None, 0.3
        self.spawned = 0
        self._merges: list = []
        sp.on_collision(1, 3, begin=self._on_ring)
        sp.on_collision(1, 1, begin=self._on_item)

    @staticmethod
    def radius(tier: int) -> float:
        return R0 * GROW ** tier

    def _add(self, pos, tier, vel=(0, 0)):
        pm = self.pymunk
        r = self.radius(tier)
        body = pm.Body()
        body.position, body.velocity = pos, vel
        shape = pm.Circle(body, r)
        shape.density, shape.elasticity, shape.friction, shape.collision_type = 1.0, BOUNCE, 0.2, 1
        self.space.add(body, shape)
        b = {"body": body, "shape": shape, "tier": tier, "r": r, "sounded": -1.0, "alive": True}
        shape.ball = b
        self.balls.append(b)
        self.spawned += 1
        return b

    def _on_ring(self, arbiter, space, data):
        b = arbiter.shapes[0].ball
        speed = abs(arbiter.shapes[0].body.velocity.dot(arbiter.contact_point_set.normal))
        if speed > 120 and self.t - b["sounded"] > 0.08:
            b["sounded"] = self.t
            self.hits.append((self.t, min(1.0, speed / 900)))

    def _on_item(self, arbiter, space, data):
        a, b = arbiter.shapes[0].ball, arbiter.shapes[1].ball
        if a["alive"] and b["alive"] and a["tier"] == b["tier"] and a["tier"] < self.last:
            a["alive"] = b["alive"] = False
            self._merges.append((a, b))

    def step(self):
        dt = 1 / FPS
        if self.done_at is None and self.t >= self.next_drop:
            # small ones, plus bigger ones once they've been revealed
            top = min(self.max_drop_tier, max(0, self.best - 1))
            tier = self.rng.choice([k for k in range(top + 1) for _ in range(top + 1 - k)])
            self._add((CX + self.rng.uniform(-8, 8), CHUTE_TOP + 12 + self.radius(tier)), tier,
                      (self.rng.uniform(-40, 40), 200))
            self.next_drop = self.t + self.drop_every
        for _ in range(3):
            # the bigger ones drift toward their nearest match (forces reset every substep)
            for b in self.balls:
                if b["tier"] >= 3:
                    mates = [o for o in self.balls if o is not b and o["tier"] == b["tier"]]
                    if mates:
                        o = min(mates, key=lambda o: (o["body"].position - b["body"].position).length)
                        d = o["body"].position - b["body"].position
                        if d.length > 1:
                            b["body"].apply_force_at_local_point(d.normalized() * b["body"].mass * 500)
            self.space.step(dt / 3)
        for a, b in self._merges:
            for x in (a, b):
                if x in self.balls:
                    self.space.remove(x["body"], x["shape"])
                    self.balls.remove(x)
            pa, pb = a["body"].position, b["body"].position
            mid = ((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
            nt = a["tier"] + 1
            self._add(mid, nt, ((a["body"].velocity.x + b["body"].velocity.x) / 2, -150))
            if len(self.effects) < 10:
                self.effects.append([mid[0], mid[1], self.radius(nt) * 1.3, 1.0, nt])
            self.events.append((self.t, "pop", min(nt, 6), 0.8))
            if nt > self.best:
                for k in range(self.best + 1, nt + 1):
                    self.unlocked_at[k] = self.t
                    self.reveal_flash[k] = 1.0
                self.best = nt
                self.banner = (f"NEW: {self.chain[nt][1]}!", self.t)
                self.events.append((self.t, "unlock", nt, 1.0))
                if nt == self.last:
                    self.done_at = self.t
                    self.events.append((self.t, "fanfare", 0, 1.0))
        self._merges = []
        self.t += dt

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def music_track(self, n: int) -> np.ndarray:
        """The tune, one note per ring hit (ball_evolution.mix_audio adds it);
        or the cursed meow / phonk riff when the recipe picked one."""
        kind = self.recipe.get("bounce_sound")
        if kind in ("meow", "phonk", "knock", "none"):
            from clipper import ball_sounds
            return ball_sounds.track(kind, self.hits, n)
        if kind == "song" and self.recipe.get("song"):
            from clipper import ball_sounds
            try:
                return ball_sounds.song_track(self.recipe["song"], self.hits, n)
            except (OSError, RuntimeError, subprocess.SubprocessError) as e:
                print(f"song failed, playing the tune instead: {e}", flush=True)
        out = np.zeros(n, dtype=np.float32)
        notes = TUNES.get(self.recipe.get("tune") or "", TUNES["mountain_king"])
        k, last, cache = 0, -1.0, {}
        for t, vol in self.hits:
            if t - last < NOTE_GAP:
                continue
            last = t
            m = notes[k % len(notes)]
            k += 1
            w = cache.get(m)
            if w is None:
                w = cache[m] = piano(m)
            i = int(t * SR)
            seg = out[i:i + len(w)]
            seg += w[:len(seg)] * (0.55 + 0.45 * vol)
        return out


class MergePainter(be.Painter):
    TRAIL_FADE = 0.78

    def __init__(self, recipe: dict):
        super().__init__(recipe)      # the pack's pictures, when it has one (be.Painter)
        self.bg = Image.new("RGB", (W, H), (0, 0, 0))
        self.trail = Image.new("RGB", (W, H))
        self.fade = [int(v * self.TRAIL_FADE) for v in range(256)] * 3
        self.trail_cols = [tuple(int(v * 0.6) for v in c) for c in self.colours]
        self.static = None
        self.big: dict = {}

    def item_sprite(self, tier: int, r: float) -> Image.Image:
        """The picture itself, no bubble, a bit wider than its circle."""
        key = (tier, int(r))
        if key not in self.big:
            s = int(r * 2.3)
            self.big[key] = self.emoji[tier].resize((s, s), Image.LANCZOS)
        return self.big[key]

    def _walls(self, sim: MergeSim) -> Image.Image:
        lines = []
        n = len(sim.ring_pts) - 1
        for i, (p, q) in enumerate(zip(sim.ring_pts, sim.ring_pts[1:])):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 1)
            lines.append(((p, q), (int(r * 255), int(g * 255), int(b * 255))))
        white = (240, 240, 250)
        for x in (CX - CHUTE_HW, CX + CHUTE_HW):
            lines.append((((x, CHUTE_TOP), (x, RING_TOP + 2)), white))
        lines.append((((CX - CHUTE_HW, CHUTE_TOP), (CX + CHUTE_HW, CHUTE_TOP)), white))
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for seg, c in lines:
            gd.line(seg, fill=c + (170,), width=26)
        im = glow.filter(ImageFilter.GaussianBlur(12))
        d = ImageDraw.Draw(im)
        for seg, c in lines:
            d.line(seg, fill=c + (255,), width=11)
        return im

    def frame(self, sim: MergeSim) -> Image.Image:
        if self.static is None:
            self.static = self._walls(sim)
        self.trail = self.trail.point(self.fade)
        td = ImageDraw.Draw(self.trail)
        for b in sim.balls:
            x, y = b["body"].position
            px, py = b.get("trail_at") or (x, y)
            if (px - x) ** 2 + (py - y) ** 2 > 4:
                be.draw_trail(td, self.recipe, b, self.trail_cols[b["tier"]], px, py, x, y, int(b["r"]),
                              fine=len(sim.balls) <= 40)
            b["trail_at"] = (x, y)
        im = ImageChops.add(self.bg, self.trail)
        im.paste(self.static, (0, 0), self.static)
        d = ImageDraw.Draw(im, "RGBA")
        for b in sim.balls:
            x, y = b["body"].position
            sp = self.item_sprite(b["tier"], b["r"]).rotate(-math.degrees(b["body"].angle), resample=Image.BILINEAR)
            im.paste(sp, (int(x - sp.width / 2), int(y - sp.height / 2)), sp)
        for e in sim.effects:
            x, y, r, life, tier = e
            rr = r * (1 + (1 - life) * 0.8)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=self.colours[tier] + (int(230 * life),),
                      width=max(2, int(6 * life)))
            e[3] -= 0.07
        sim.effects[:] = [e for e in sim.effects if e[3] > 0]
        self.header(im, d, sim)
        self.watermark(d, W - 40, H - 70)
        self.overlays(im, d, sim, item_y=CY - 60, text_y=CY + 120)
        return im
