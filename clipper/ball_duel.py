"""Two more Ball Evolution kinds, from reels Dean sent (Oct 2026):

* ``duel`` -- Ball Duel (a 1M-like reel): two balls with faces, "Cherry Red
  VS Ocean Blue", 10 hearts each, bounce in a ring. Weapons appear in grey
  rings; a ball that touches one picks it up. Dean: mostly guns (pistol,
  shotgun, minigun, rocket launcher), each fires once and is gone; a knife
  and a shield now and then. A hit costs hearts, the face reacts (happy,
  angry with a weapon, scared when low, X eyes when hit) and the ball
  cracks; the last one alive wins. Seeds are picked for a close fight: a
  comeback, the winner on 3 hearts or fewer, no long stretch without a hit.
* ``race`` -- First to N wins (a 149K-like reel): four plain coloured balls
  in a circle. Every wall hit ties a line from that spot to the ball; a
  ball that runs into another's line steals it. First to N lines wins.
  Slow enough to follow (Dean found the first try too fast to follow).

Neither uses a picture pack: the balls are the contestants. The guns are
drawn here (the emoji set only has a water pistol). Sounds are made here
(numpy); the music is Dean's song (played straight through, not gated by
bounces) or the recipe's own bounce tune.

ball_evolution dispatches here (``_parts``), so picking seeds, rendering,
audio mixing and publishing are shared.
"""
from __future__ import annotations

import math
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from clipper import ball_evolution as be

W, H, FPS, SR = be.W, be.H, be.FPS, be.SR
END_HOLD = 3.2
SECONDS_CAP = 70.0
KINDS = ("duel", "race")

# colour names for the contestants: (first word, second word, rgb)
COLOURS = [("Cherry", "Red", (228, 52, 64)), ("Ocean", "Blue", (84, 140, 245)),
           ("Lime", "Green", (96, 204, 84)), ("Royal", "Purple", (154, 96, 232)),
           ("Sunny", "Yellow", (246, 200, 46)), ("Hot", "Pink", (240, 92, 172)),
           ("Tangerine", "Orange", (250, 142, 44)), ("Ice", "Cyan", (70, 210, 230))]

DUEL_HOOKS = ["{a} VS {b}", "{a} VS {b} -- who wins?", "Who wins? {a} VS {b}"]
RACE_HOOKS = ["First to {n} wins", "Pick a colour: first to {n} wins", "Which colour gets {n} first?"]


def duel_recipe(recipe: dict) -> dict:
    """The duel's own draws (own rng, so other kinds' recipes don't change)."""
    rng = random.Random(f"duel-recipe-{recipe['seed']}")
    a, b = rng.sample(COLOURS, 2)
    recipe.update(players=[list(a), list(b)], bed="off",
                  hook=rng.choice(DUEL_HOOKS).format(a=f"{a[0]} {a[1]}", b=f"{b[0]} {b[1]}"),
                  ending="WINS!")
    return recipe


def race_recipe(recipe: dict) -> dict:
    rng = random.Random(f"race-recipe-{recipe['seed']}")
    players = rng.sample(COLOURS, 4)
    n = RACE_TARGET
    recipe.update(players=[list(p) for p in players], target=n, bed="off",
                  hook=rng.choice(RACE_HOOKS).format(n=n), ending="WINS!")
    return recipe


def _font(size, name="Inter-Black.ttf"):
    return be._font(size, name)


def _ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


# ---- the guns, drawn (pointing right, held at the left) ----------------------

def _gun(kind: str) -> Image.Image:
    k = 3
    if kind == "pistol":
        w, h = 120, 74
    elif kind == "shotgun":
        w, h = 214, 60
    elif kind == "minigun":
        w, h = 206, 96
    else:  # rocket
        w, h = 230, 76
    im = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (22, 22, 28, 255)
    steel, steel_hi, dark = (70, 75, 86, 255), (128, 136, 150, 255), (40, 43, 50, 255)
    wood, wood_hi = (132, 84, 44, 255), (170, 114, 64, 255)
    o = 3 * k

    def P(pts):
        return [(x * k, y * k) for x, y in pts]

    def box(x0, y0, x1, y1, fill, r=3):
        d.rounded_rectangle((x0 * k, y0 * k, x1 * k, y1 * k), r * k, fill=fill, outline=ink, width=o)

    if kind == "pistol":
        d.polygon(P([(22, 30), (52, 30), (46, 70), (18, 70)]), fill=wood, outline=ink, width=o)   # grip
        d.arc((40 * k, 30 * k, 66 * k, 54 * k), 0, 180, fill=ink, width=o)                        # trigger guard
        box(10, 10, 114, 34, steel)                                                                # slide
        d.line((18 * k, 16 * k, 104 * k, 16 * k), fill=steel_hi, width=3 * k)
        box(104, 14, 116, 28, dark, 2)                                                             # muzzle
    elif kind == "shotgun":
        d.polygon(P([(2, 26), (52, 14), (52, 42), (8, 56)]), fill=wood, outline=ink, width=o)     # stock
        box(48, 10, 98, 42, steel)                                                                 # receiver
        box(92, 14, 212, 26, dark, 2)                                                              # barrel
        box(92, 26, 196, 36, steel, 2)                                                             # tube
        box(118, 30, 168, 46, wood, 4)                                                             # pump
        d.line((124 * k, 36 * k, 162 * k, 36 * k), fill=wood_hi, width=2 * k)
        d.arc((60 * k, 34 * k, 84 * k, 56 * k), 0, 180, fill=ink, width=o)
    elif kind == "minigun":
        box(24, 22, 102, 74, steel, 8)                                                             # body
        d.rectangle((30 * k, 44 * k, 96 * k, 52 * k), fill=(240, 196, 40, 255))                    # stripe
        box(40, 6, 80, 24, dark, 4)                                                                # handle
        for y in (28, 42, 56):
            box(100, y, 202, y + 12, dark, 3)                                                      # barrels
        for x in (122, 170):
            box(x, 24, x + 10, 72, steel, 2)                                                       # rings
        # ammo belt
        for j in range(6):
            x = 50 + j * 9
            d.rounded_rectangle((x * k, (74 + j * 2.4) * k, (x + 7) * k, (86 + j * 2.4) * k), 2 * k,
                                fill=(214, 168, 50, 255), outline=ink, width=2 * k)
    else:  # rocket launcher
        olive, olive_hi = (86, 112, 62, 255), (126, 156, 92, 255)
        box(6, 20, 196, 52, olive, 14)                                                             # tube
        d.line((24 * k, 27 * k, 180 * k, 27 * k), fill=olive_hi, width=3 * k)
        box(70, 6, 110, 20, dark, 3)                                                               # sight
        d.polygon(P([(62, 52), (86, 52), (80, 74), (60, 74)]), fill=dark, outline=ink, width=o)    # grip
        d.polygon(P([(194, 22), (228, 36), (194, 50)]), fill=(222, 50, 46, 255), outline=ink, width=o)  # warhead
        box(186, 18, 198, 54, (200, 200, 205, 255), 3)
    return im.resize((w, h), Image.LANCZOS)


# name: (kind, damage, spawn weight); guns fire once, then they're gone
WEAPONS = {
    "pistol": ("gun", 1, 3),
    "shotgun": ("gun", 1, 3),       # 7 pellets, at most 2 hearts a blast
    "minigun": ("gun", 1, 2),       # a 1.2 s burst, at most 3 hearts
    "rocket": ("gun", 3, 2),        # explodes: 3 hearts close, 2 at the edge
    "knife": ("melee", 1, 1),       # spins, two hits
    "shield": ("shield", 1, 1),     # blocks two hits from the front, hurts on contact
}
KNIFE_HITS, SHIELD_BLOCKS = 2, 2


# ---- the duel -----------------------------------------------------------------

D_CX, D_CY, D_R = 540, 1130, 470
D_BALL = 64
D_SPEED = 560
HEARTS = 10


class DuelSim:
    """The fight. ``events`` stays empty (no stock ball sounds): the duel's
    own sounds are in ``sfx`` and come out through ``music_track``."""

    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"duel-{recipe['seed']}")
        self.t = 0.0
        self.balls = []
        for i, (w1, w2, col) in enumerate(recipe.get("players") or [list(c) for c in COLOURS[:2]]):
            a = self.rng.uniform(0, 2 * math.pi)
            self.balls.append({
                "i": i, "name": f"{w1} {w2}", "col": tuple(col),
                "x": D_CX + (-170 if i == 0 else 170), "y": D_CY + self.rng.uniform(-120, 120),
                "vx": D_SPEED * math.cos(a), "vy": D_SPEED * math.sin(a),
                "hp": HEARTS, "weapon": None, "uses": 0, "wang": self.rng.uniform(0, 6.28), "held": 0.0,
                "cool": 0.0, "hurt": 0.0, "squash": 0.0, "kick": 0.0, "alive": True, "weapons_had": 0,
                "burst": 0.0, "burst_hits": 0, "next_round": 0.0,
            })
        self.pickups: list = []
        self.bullets: list = []
        self.rockets: list = []
        self.effects: list = []
        self.sfx: list = []          # (t, kind, value)
        self.events: list = []       # the shared mixer's ball sounds: none
        self.hits: list = []         # wall bounces (a bounce tune, when there's no song)
        self.unlocked_at = {0: 0.0}
        self.best, self.spawned = 0, 2
        self.next_spawn = 1.4
        self.winner = None
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0
        self.shake = 0.0
        self._hp_log = []            # (t, hp0, hp1)
        self._last_damage = 0.0
        self._gaps = []

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def other(self, b):
        return self.balls[1 - b["i"]]

    # -- damage
    def damage(self, target, dmg, x, y, by) -> bool:
        if not target["alive"] or self.winner is not None:
            return False
        if target["weapon"] == "shield" and target["uses"] > 0:
            face = math.atan2(by["y"] - target["y"], by["x"] - target["x"])
            if abs(_ang_diff(face, target["wang"])) < 1.2:
                target["uses"] -= 1
                self.sfx.append((self.t, "clang", 0))
                self.effects.append({"k": "spark", "x": x, "y": y, "life": 1.0, "col": (255, 230, 150)})
                if target["uses"] <= 0:
                    self._drop(target)
                return False
        target["hp"] = max(0, target["hp"] - dmg)
        target["hurt"] = 0.3
        target["squash"] = 1.0
        self.shake = max(self.shake, 0.25 + 0.15 * dmg)
        self.sfx.append((self.t, "hit", dmg))
        self.effects.append({"k": "num", "x": target["x"], "y": target["y"] - 80, "text": f"-{dmg}", "life": 1.0,
                             "col": target["col"]})
        self.effects.append({"k": "spark", "x": x, "y": y, "life": 1.0, "col": (255, 255, 255)})
        self._gaps.append(self.t - self._last_damage)
        self._last_damage = self.t
        self._hp_log.append((self.t, self.balls[0]["hp"], self.balls[1]["hp"]))
        a = math.atan2(target["y"] - y, target["x"] - x)
        sp = math.hypot(target["vx"], target["vy"])
        target["vx"], target["vy"] = sp * math.cos(a), sp * math.sin(a)
        if target["hp"] <= 0:
            target["alive"] = False
            self._end(by)
            for _ in range(44):
                aa = self.rng.uniform(0, 6.28)
                s = self.rng.uniform(150, 700)
                self.effects.append({"k": "shard", "x": target["x"], "y": target["y"], "vx": s * math.cos(aa),
                                     "vy": s * math.sin(aa), "life": 1.0, "col": target["col"],
                                     "r": self.rng.uniform(6, 16), "spin": self.rng.uniform(-8, 8)})
        return True

    def _end(self, winner):
        self.winner = winner
        self.done_at = self.t
        self.sfx.append((self.t, "die", 0))
        w = winner["i"]
        behind = any(h[1 + w] - h[2 - w] <= -2 for h in self._hp_log)       # a comeback
        left = winner["hp"]
        had = min(b["weapons_had"] for b in self.balls)
        self.max_gap = max(self._gaps + [0.0])
        self.ok = behind and left <= 3 and had >= 2

    def _drop(self, b):
        """The weapon is used up: it falls away."""
        if b["weapon"]:
            self.effects.append({"k": "drop", "w": b["weapon"], "x": b["x"] + math.cos(b["wang"]) * 70,
                                 "y": b["y"] + math.sin(b["wang"]) * 70, "ang": b["wang"], "life": 1.0,
                                 "vy": -120.0})
        b["weapon"] = None

    # -- one frame
    def step(self):
        dt = 1 / FPS
        self.t += dt
        self.shake = max(0.0, self.shake - dt)
        if self.winner is not None:
            w = self.winner
            w["hurt"] = w["squash"] = 0.0
            w["x"] += (D_CX - w["x"]) * 0.06
            w["y"] += (D_CY - w["y"]) * 0.06
            self._effects(dt)
            return
        self._spawn()
        for _ in range(4):
            self._move(dt / 4)
        self._weapons(dt)
        self._effects(dt)
        if self.t > SECONDS_CAP - END_HOLD - 0.1 and self.done_at is None:
            self.done_at = self.t        # nobody won in time: a bad seed
            self.ok = False

    def _spawn(self):
        if self.t < self.next_spawn:
            return
        if len(self.pickups) < 3:
            names = list(WEAPONS)
            w = self.rng.choices(names, [WEAPONS[n][2] for n in names])[0]
            for _ in range(30):
                a = self.rng.uniform(0, 6.28)
                dd = self.rng.uniform(0, D_R - 120)
                x, y = D_CX + dd * math.cos(a), D_CY + dd * math.sin(a)
                if all(math.hypot(x - b["x"], y - b["y"]) > 190 for b in self.balls) and \
                        all(math.hypot(x - p["x"], y - p["y"]) > 170 for p in self.pickups):
                    self.pickups.append({"w": w, "x": x, "y": y, "born": self.t})
                    self.sfx.append((self.t, "spawn", 0))
                    break
        self.next_spawn = self.t + self.rng.uniform(2.0, 3.2)

    def _move(self, dt):
        for b in self.balls:
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            dx, dy = b["x"] - D_CX, b["y"] - D_CY
            dd = math.hypot(dx, dy)
            if dd > D_R - D_BALL:
                nx, ny = dx / dd, dy / dd
                vn = b["vx"] * nx + b["vy"] * ny
                if vn > 0:
                    vx, vy = b["vx"] - 2 * vn * nx, b["vy"] - 2 * vn * ny
                    a = math.atan2(vy, vx) + self.rng.uniform(-0.18, 0.18)
                    b["vx"], b["vy"] = D_SPEED * math.cos(a), D_SPEED * math.sin(a)
                    self.hits.append((self.t, 0.6))
                    self.sfx.append((self.t, "wall", 0))
                b["x"], b["y"] = D_CX + nx * (D_R - D_BALL), D_CY + ny * (D_R - D_BALL)
        a, c = self.balls
        dx, dy = c["x"] - a["x"], c["y"] - a["y"]
        dd = math.hypot(dx, dy)
        if 0 < dd < 2 * D_BALL:
            nx, ny = dx / dd, dy / dd
            rel = (c["vx"] - a["vx"]) * nx + (c["vy"] - a["vy"]) * ny
            if rel < 0:
                a["vx"] += rel * nx
                a["vy"] += rel * ny
                c["vx"] -= rel * nx
                c["vy"] -= rel * ny
                self.sfx.append((self.t, "bump", 0))
                for p, q in ((a, c), (c, a)):
                    if p["weapon"] == "shield" and p["cool"] <= 0:
                        p["cool"] = 0.6
                        self.damage(q, 1, (a["x"] + c["x"]) / 2, (a["y"] + c["y"]) / 2, p)
            over = 2 * D_BALL - dd
            a["x"] -= nx * over / 2
            a["y"] -= ny * over / 2
            c["x"] += nx * over / 2
            c["y"] += ny * over / 2
            for b in (a, c):
                s = math.hypot(b["vx"], b["vy"]) or 1
                b["vx"], b["vy"] = b["vx"] / s * D_SPEED, b["vy"] / s * D_SPEED
        for p in list(self.pickups):
            for b in self.balls:
                if b["alive"] and math.hypot(p["x"] - b["x"], p["y"] - b["y"]) < D_BALL + 55:
                    self._equip(b, p["w"])
                    self.pickups.remove(p)
                    break

    def _equip(self, b, w):
        if b["weapon"]:
            self._drop(b)
        b["weapon"] = w
        b["weapons_had"] += 1
        b["held"] = 0.0
        b["burst"] = 0.0
        b["burst_hits"] = 0
        b["uses"] = KNIFE_HITS if w == "knife" else SHIELD_BLOCKS if w == "shield" else 1
        self.sfx.append((self.t, "pickup", 0))
        self.effects.append({"k": "ring", "ball": b, "x": b["x"], "y": b["y"], "life": 1.0, "col": (255, 255, 255)})

    def _fire(self, b, w):
        """The one shot of a gun."""
        a = b["wang"]
        mx, my = b["x"] + math.cos(a) * (D_BALL + 90), b["y"] + math.sin(a) * (D_BALL + 90)
        b["kick"] = 1.0
        if w == "pistol":
            self._bullet(b, mx, my, a, 1450, 1.2, "pistol")
            self.sfx.append((self.t, "shot", 0))
        elif w == "shotgun":
            b["burst_hits"] = 0
            for k in range(7):
                self._bullet(b, mx, my, a + (k - 3) * 0.1 + self.rng.uniform(-0.03, 0.03), 1500, 0.5, "shotgun")
            self.sfx.append((self.t, "blast", 0))
        elif w == "rocket":
            self.rockets.append({"x": mx, "y": my, "a": a, "sp": 520.0, "owner": b, "smoke": 0.0})
            self.sfx.append((self.t, "launch", 0))
        self.effects.append({"k": "flash", "x": mx, "y": my, "life": 1.0, "col": (255, 214, 110)})
        self.shake = max(self.shake, 0.12)

    def _bullet(self, b, x, y, a, sp, life, kind):
        self.bullets.append({"x": x, "y": y, "vx": sp * math.cos(a), "vy": sp * math.sin(a), "owner": b,
                             "life": life, "kind": kind})

    def _weapons(self, dt):
        for b in self.balls:
            o = self.other(b)
            b["cool"] = max(0.0, b["cool"] - dt)
            b["hurt"] = max(0.0, b["hurt"] - dt)
            b["squash"] = max(0.0, b["squash"] - dt * 4)
            b["kick"] = max(0.0, b["kick"] - dt * 5)
            w = b["weapon"]
            if not w or not b["alive"]:
                continue
            kind, dmg, _wt = WEAPONS[w]
            aim = math.atan2(o["y"] - b["y"], o["x"] - b["x"])
            b["held"] += dt
            if kind == "melee":
                b["wang"] += 6.0 * dt
                tx = b["x"] + math.cos(b["wang"]) * (D_BALL + 62)
                ty = b["y"] + math.sin(b["wang"]) * (D_BALL + 62)
                if b["cool"] <= 0 and math.hypot(tx - o["x"], ty - o["y"]) < D_BALL + 18:
                    b["cool"] = 0.55
                    self.damage(o, dmg, tx, ty, b)
                    b["uses"] -= 1
                    if b["uses"] <= 0:
                        self._drop(b)
            elif kind == "shield":
                b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 5)
            elif w == "minigun" and b["burst"] > 0:
                b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 4)
                b["burst"] -= dt
                if self.t >= b["next_round"]:
                    b["next_round"] = self.t + 0.075
                    a = b["wang"] + self.rng.uniform(-0.1, 0.1)
                    mx = b["x"] + math.cos(b["wang"]) * (D_BALL + 100)
                    my = b["y"] + math.sin(b["wang"]) * (D_BALL + 100)
                    self._bullet(b, mx, my, a, 1600, 1.0, "minigun")
                    self.sfx.append((self.t, "rat", 0))
                    self.effects.append({"k": "flash", "x": mx, "y": my, "life": 0.6, "col": (255, 214, 110)})
                    b["kick"] = 0.6
                if b["burst"] <= 0:
                    self._drop(b)
            else:  # a gun: aim, then fire its one shot
                b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 6)
                if b["held"] > 0.45 and abs(_ang_diff(aim, b["wang"])) < 0.15:
                    if w == "minigun":
                        b["burst"], b["burst_hits"], b["next_round"] = 1.2, 0, self.t
                        self.sfx.append((self.t, "spin", 0))
                    else:
                        self._fire(b, w)
                        self._drop(b)
        for u in list(self.bullets):
            u["x"] += u["vx"] * dt
            u["y"] += u["vy"] * dt
            u["life"] -= dt
            o = self.other(u["owner"])
            cap = {"shotgun": 2, "minigun": 3}.get(u["kind"])
            if o["alive"] and math.hypot(u["x"] - o["x"], u["y"] - o["y"]) < D_BALL + 8:
                own = u["owner"]
                if cap is None or own["burst_hits"] < cap:
                    if self.damage(o, 1, u["x"], u["y"], own) and cap is not None:
                        own["burst_hits"] += 1
                self.bullets.remove(u)
            elif u["life"] <= 0 or math.hypot(u["x"] - D_CX, u["y"] - D_CY) > D_R:
                self.effects.append({"k": "spark", "x": u["x"], "y": u["y"], "life": 0.5, "col": (255, 220, 120)})
                self.bullets.remove(u)
        for r in list(self.rockets):
            o = self.other(r["owner"])
            r["sp"] = min(1300.0, r["sp"] + 1600 * dt)
            want = math.atan2(o["y"] - r["y"], o["x"] - r["x"])
            r["a"] += _ang_diff(want, r["a"]) * min(1, dt * 1.6)       # a lazy homing curve
            r["x"] += math.cos(r["a"]) * r["sp"] * dt
            r["y"] += math.sin(r["a"]) * r["sp"] * dt
            r["smoke"] -= dt
            if r["smoke"] <= 0:
                r["smoke"] = 0.03
                self.effects.append({"k": "smoke", "x": r["x"] - math.cos(r["a"]) * 30,
                                     "y": r["y"] - math.sin(r["a"]) * 30, "life": 1.0,
                                     "r": self.rng.uniform(10, 18)})
            hit = o["alive"] and math.hypot(r["x"] - o["x"], r["y"] - o["y"]) < D_BALL + 14
            if hit or math.hypot(r["x"] - D_CX, r["y"] - D_CY) > D_R - 10:
                self.rockets.remove(r)
                self._boom(r["x"], r["y"], r["owner"])

    def _boom(self, x, y, owner):
        self.sfx.append((self.t, "boom", 0))
        self.shake = max(self.shake, 0.6)
        self.effects.append({"k": "boom", "x": x, "y": y, "life": 1.0, "col": (255, 170, 60)})
        o = self.other(owner)
        dd = math.hypot(o["x"] - x, o["y"] - y)
        if dd < 120:
            self.damage(o, 3, x, y, owner)
        elif dd < 220:
            self.damage(o, 2, x, y, owner)

    def _effects(self, dt):
        for e in self.effects:
            k = e["k"]
            e["life"] -= dt * (1.6 if k in ("spark", "flash") else 1.3 if k == "smoke" else 0.9)
            if k == "num":
                e["y"] -= 70 * dt
            elif k == "ring":
                e["x"], e["y"] = e["ball"]["x"], e["ball"]["y"]
                e["life"] -= dt * 1.2
            elif k == "shard":
                e["x"] += e["vx"] * dt
                e["y"] += e["vy"] * dt
                e["vx"] *= 0.96
                e["vy"] *= 0.96
            elif k == "drop":
                e["vy"] += 900 * dt
                e["y"] += e["vy"] * dt
                e["ang"] += 4 * dt
            elif k == "smoke":
                e["r"] += 30 * dt
        self.effects = [e for e in self.effects if e["life"] > 0]

    # -- sound
    def music_track(self, n: int) -> np.ndarray:
        return _sfx_mix(self.recipe, self.sfx, self.hits, n, self)


def _heart(cx, cy, s):
    pts = []
    for k in range(80):
        t = k / 80 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * s / 34, cy + y * s / 34))
    return pts


def _heart_sprite(size, fill, outline=None):
    big = size * 3
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    poly = _heart(big / 2, big / 2 + big * 0.04, big * 0.95)
    if fill:
        d.polygon(poly, fill=fill)
        hl = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(hl).ellipse((big * 0.2, big * 0.2, big * 0.42, big * 0.4), fill=(255, 255, 255, 90))
        im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(big * 0.03)))
    if outline:
        d.line(poly + [poly[0]], fill=outline, width=int(big * 0.05))
    return im.resize((size, size), Image.LANCZOS)


def _arena_bg(cx, cy, r, grid=True, ring=(245, 245, 255)) -> Image.Image:
    bg = Image.new("RGB", (W, H), (7, 9, 24))
    if grid:
        d = ImageDraw.Draw(bg)
        for x in range(0, W, 60):
            d.line([(x, 0), (x, H)], fill=(20, 24, 48), width=2)
        for y in range(0, H, 60):
            d.line([(0, y), (W, y)], fill=(20, 24, 48), width=2)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    bg = Image.composite(bg, Image.new("RGB", (W, H), (5, 6, 16)), mask)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    ImageDraw.Draw(glow).ellipse((cx - r, cy - r, cx + r, cy + r), outline=(160, 170, 255), width=16)
    glow = glow.filter(ImageFilter.GaussianBlur(14))
    bg = Image.fromarray(np.clip(np.asarray(bg, dtype=np.int16) + np.asarray(glow, dtype=np.int16), 0, 255)
                         .astype(np.uint8))
    ImageDraw.Draw(bg).ellipse((cx - r, cy - r, cx + r, cy + r), outline=ring, width=6)
    return bg


class DuelPainter:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.bg = _arena_bg(D_CX, D_CY, D_R)
        players = recipe.get("players") or [list(c) for c in COLOURS[:2]]
        self.cols = [tuple(p[2]) for p in players]
        self.title = []
        for k, (w1, w2, col) in enumerate(players):
            if k:
                self.title.append(("VS ", (255, 255, 255)))
            self.title += [(w1 + " ", tuple(col)), (w2 + (" " if k == 0 else ""), (255, 255, 255))]
        self.heart_full = [_heart_sprite(60, c + (255,)) for c in self.cols]
        self.heart_empty = _heart_sprite(60, None, (70, 74, 100, 255))
        self.heart_lost = [_heart_sprite(60, tuple(int(v * 0.45) for v in c) + (255,)) for c in self.cols]
        self.guns = {w: _gun(w) for w in ("pistol", "shotgun", "minigun", "rocket")}
        self.knife = _emoji("1f52a", 118)
        self.shield = _emoji("1f6e1", 118)
        self.icons = {**self.guns, "knife": self.knife, "shield": self.shield}
        self.rocket = Image.new("RGBA", (90, 40), (0, 0, 0, 0))          # a rocket in flight
        rd = ImageDraw.Draw(self.rocket)
        rd.rounded_rectangle((4, 10, 70, 30), 8, fill=(220, 220, 228, 255), outline=(22, 22, 28, 255), width=3)
        rd.polygon([(68, 8), (88, 20), (68, 32)], fill=(222, 50, 46, 255), outline=(22, 22, 28, 255))
        rd.polygon([(4, 10), (0, 2), (18, 10)], fill=(222, 50, 46, 255))
        rd.polygon([(4, 30), (0, 38), (18, 30)], fill=(222, 50, 46, 255))
        self.f_title = _font(60)
        self.f_num = _font(64)
        self.f_win = _font(96)
        self.f_small = _font(34, "Inter-Bold.ttf")
        self.cache: dict = {}
        self.shown_hp = [HEARTS, HEARTS]
        self.lost_at = [[None] * HEARTS, [None] * HEARTS]

    def ball_sprite(self, b, look, mood, blink, flash, cracks):
        key = (b["i"], round(look / (math.pi / 12)), mood, blink, flash, cracks)
        if key in self.cache:
            return self.cache[key]
        r = D_BALL
        S3 = 3
        big = (r * 2 + 8) * S3
        c = big / 2
        R = r * S3
        im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        col = (255, 255, 255) if flash else b["col"]
        d.ellipse((c - R, c - R, c + R, c + R), fill=col + (255,), outline=(255, 255, 255, 255), width=int(5 * S3))
        rng = np.random.default_rng(b["i"] * 7 + 3)
        for _ in range(cracks):
            a0 = rng.uniform(0, 2 * math.pi)
            x, y = c + math.cos(a0) * R * 0.95, c + math.sin(a0) * R * 0.95
            pts = [(x, y)]
            ang = a0 + math.pi + rng.uniform(-0.5, 0.5)
            for _ in range(3):                  # stays near the rim, off the face
                step = R * rng.uniform(0.11, 0.16)
                ang += rng.uniform(-0.6, 0.6)
                x, y = x + math.cos(ang) * step, y + math.sin(ang) * step
                pts.append((x, y))
            d.line(pts, fill=(30, 30, 50, 200), width=int(2.4 * S3))
        ink = (20, 20, 30, 255)
        ex, ey = math.cos(look), math.sin(look)
        for side in (-1, 1):
            cx_e, cy_e = c + side * R * 0.32 + ex * R * 0.08, c - 0.18 * R + ey * R * 0.06
            er = R * (0.27 if mood == "scared" else 0.22)
            if mood == "dead" or flash:
                w_ = er * 0.8
                d.line([(cx_e - w_, cy_e - w_), (cx_e + w_, cy_e + w_)], fill=ink, width=int(4 * S3))
                d.line([(cx_e - w_, cy_e + w_), (cx_e + w_, cy_e - w_)], fill=ink, width=int(4 * S3))
                continue
            if blink:
                d.line([(cx_e - er, cy_e), (cx_e + er, cy_e)], fill=ink, width=int(4 * S3))
                continue
            d.ellipse((cx_e - er, cy_e - er, cx_e + er, cy_e + er), fill=(255, 255, 255, 255), outline=ink,
                      width=int(2.5 * S3))
            pr = er * (0.42 if mood == "scared" else 0.55)
            px, py = cx_e + ex * er * 0.4, cy_e + ey * er * 0.4
            d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=ink)
            d.ellipse((px - pr * 0.05, py - pr * 0.65, px + pr * 0.65, py - pr * 0.05), fill=(255, 255, 255, 255))
            if mood == "angry":
                # outer end high, inner end low: a frown
                d.line([(cx_e + side * er * 1.1, cy_e - er * 1.5), (cx_e - side * er * 0.9, cy_e - er * 0.85)],
                       fill=ink, width=int(5 * S3))
            elif mood == "scared":
                # inner end high: worried
                d.line([(cx_e + side * er * 1.0, cy_e - er * 1.05), (cx_e - side * er * 0.9, cy_e - er * 1.55)],
                       fill=ink, width=int(4 * S3))
        my = c + R * 0.38
        if mood == "happy":
            d.arc((c - R * 0.3, my - R * 0.25, c + R * 0.3, my + R * 0.12), 20, 160, fill=ink, width=int(4 * S3))
        elif mood == "angry":
            d.arc((c - R * 0.28, my - R * 0.02, c + R * 0.28, my + R * 0.3), 200, 340, fill=ink, width=int(4 * S3))
        elif mood == "scared":
            pts = [(c - R * 0.28 + k * R * 0.07, my + (R * 0.05 if k % 2 else -R * 0.03)) for k in range(9)]
            d.line(pts, fill=ink, width=int(3.5 * S3))
        elif mood == "dead":
            d.ellipse((c - R * 0.12, my - R * 0.08, c + R * 0.12, my + R * 0.16), fill=ink)
        sp = im.resize((r * 2 + 8, r * 2 + 8), Image.LANCZOS)
        if len(self.cache) > 600:
            self.cache.clear()
        self.cache[key] = sp
        return sp

    def mood(self, sim, b):
        if sim.winner is b:
            return "happy"
        if not b["alive"]:
            return "dead"
        if b["weapon"] and b["weapon"] != "shield":
            return "angry"
        if b["hp"] <= 4:
            return "scared"
        return "happy"

    def frame(self, sim: DuelSim) -> Image.Image:
        im = self.bg.copy().convert("RGBA")
        d = ImageDraw.Draw(im)
        self.hud(im, d, sim)
        for p in sim.pickups:
            age = sim.t - p["born"]
            s = min(1.0, age / 0.25)
            rr = 70 * s
            bob = math.sin(age * 4) * 6
            d.ellipse((p["x"] - rr, p["y"] - rr + bob, p["x"] + rr, p["y"] + rr + bob), outline=(150, 150, 165, 255),
                      width=7)
            if s > 0.2:
                ic = self.icons[p["w"]]
                f = min(1.0, 124 / max(ic.size)) * s
                ic = ic.resize((max(4, int(ic.width * f)), max(4, int(ic.height * f))), Image.LANCZOS)
                im.alpha_composite(ic, (int(p["x"] - ic.width / 2), int(p["y"] - ic.height / 2 + bob)))
        for e in sim.effects:
            if e["k"] == "smoke":
                L = max(0.0, e["life"])
                r = e["r"]
                d.ellipse((e["x"] - r, e["y"] - r, e["x"] + r, e["y"] + r), fill=(190, 190, 200, int(120 * L)))
        for u in sim.bullets:
            sp = math.hypot(u["vx"], u["vy"])
            ln = 34 if u["kind"] == "shotgun" else 46
            tx, ty = u["x"] - u["vx"] / sp * ln, u["y"] - u["vy"] / sp * ln
            d.line([(tx, ty), (u["x"], u["y"])], fill=(255, 210, 90, 170), width=7)
            rr = 6 if u["kind"] == "shotgun" else 8
            d.ellipse((u["x"] - rr, u["y"] - rr, u["x"] + rr, u["y"] + rr), fill=(255, 245, 200, 255))
        for r in sim.rockets:
            rot = self.rocket.rotate(-math.degrees(r["a"]), resample=Image.BICUBIC, expand=True)
            im.alpha_composite(rot, (int(r["x"] - rot.width / 2), int(r["y"] - rot.height / 2)))
        for b in sim.balls:
            if not b["alive"]:
                continue
            o = sim.other(b)
            look = math.atan2(o["y"] - b["y"], o["x"] - b["x"])
            blink = (int(sim.t * 10 + b["i"] * 17) % 37) == 0
            flash = b["hurt"] > 0.18
            cracks = min(4, (HEARTS - b["hp"]) // 2)
            sp = self.ball_sprite(b, look, self.mood(sim, b), blink, flash, cracks)
            if sim.winner is b:
                scale = 1 + min(0.5, (sim.t - sim.done_at) * 0.4)
            else:
                scale = 1 + 0.25 * b["squash"]
            if abs(scale - 1) > 0.01:
                sp = sp.resize((int(sp.width * scale), int(sp.height * scale)), Image.LANCZOS)
            im.alpha_composite(sp, (int(b["x"] - sp.width / 2), int(b["y"] - sp.height / 2)))
            self.held(im, b)
        self.effects(im, d, sim)
        if sim.winner is not None:
            self.ending(im, d, sim)
        self.watermark(d)
        out = im.convert("RGB")
        if sim.shake > 0:
            amp = int(14 * min(1.0, sim.shake))
            rng = random.Random(int(sim.t * 1000))
            out = Image.fromarray(np.roll(np.asarray(out), (rng.randint(-amp, amp), rng.randint(-amp, amp)), (0, 1)))
        return out

    def held(self, im, b):
        w = b["weapon"]
        if not w:
            return
        a = b["wang"]
        if w == "knife":
            img, dist, rot = self.knife, D_BALL + 34, -math.degrees(a) - 45
        elif w == "shield":
            img, dist, rot = self.shield, D_BALL + 20, 0
        else:
            img = self.guns[w]
            if math.cos(a) < 0:              # keep the gun upright on the left side
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            kick = b["kick"] * 14
            dist = D_BALL + img.width * 0.42 - kick
            rot = -math.degrees(a)
        x = b["x"] + math.cos(a) * dist
        y = b["y"] + math.sin(a) * dist
        r_img = img.rotate(rot, resample=Image.BICUBIC, expand=True)
        im.alpha_composite(r_img, (int(x - r_img.width / 2), int(y - r_img.height / 2)))

    def hud(self, im, d, sim):
        for i, b in enumerate(sim.balls):
            y = 110 + i * 150
            d.ellipse((30, y - 48, 126, y + 48), fill=b["col"] + (255,), outline=(255, 255, 255, 255), width=5)
            if b["hp"] < self.shown_hp[i]:
                for k in range(b["hp"], self.shown_hp[i]):
                    self.lost_at[i][k] = sim.t
                self.shown_hp[i] = b["hp"]
            for k in range(HEARTS):
                x = 150 + k * 88
                if k < b["hp"]:
                    im.alpha_composite(self.heart_full[i], (x, y - 30))
                    continue
                la = self.lost_at[i][k]
                if la is not None and sim.t - la < 0.5:
                    s = 1 + (sim.t - la) * 1.2
                    hs = self.heart_lost[i].resize((int(60 * s), int(60 * s)))
                    fade = 1 - (sim.t - la) / 0.5
                    hs.putalpha(hs.getchannel("A").point(lambda v, f=fade: int(v * f)))
                    im.alpha_composite(hs, (int(x + 30 - hs.width / 2), int(y - hs.height / 2)))
                else:
                    im.alpha_composite(self.heart_empty, (x, y - 30))
        f = self.f_title
        total = sum(d.textlength(t, font=f) for t, _ in self.title)
        while total > W - 80 and f.size > 30:
            f = _font(f.size - 4)
            total = sum(d.textlength(t, font=f) for t, _ in self.title)
        x = (W - total) / 2
        for t, col in self.title:
            d.text((x, 480), t, font=f, fill=col, anchor="lm")
            x += d.textlength(t, font=f)

    def effects(self, im, d, sim):
        for e in sim.effects:
            L = max(0.0, e["life"])
            k = e["k"]
            if k == "num":
                a = int(255 * min(1, L * 2))
                d.text((e["x"], e["y"]), e["text"], font=self.f_num, fill=(255, 255, 255, a), anchor="mm",
                       stroke_width=6, stroke_fill=e["col"] + (a,))
            elif k == "spark":
                for j in range(8):
                    ang = j * math.pi / 4 + 0.3
                    r0, r1 = 10 + (1 - L) * 40, 22 + (1 - L) * 70
                    d.line([(e["x"] + math.cos(ang) * r0, e["y"] + math.sin(ang) * r0),
                            (e["x"] + math.cos(ang) * r1, e["y"] + math.sin(ang) * r1)],
                           fill=e["col"] + (int(255 * L),), width=5)
            elif k == "flash":
                r = 30 * L + 8
                d.ellipse((e["x"] - r, e["y"] - r, e["x"] + r, e["y"] + r), fill=e["col"] + (int(230 * L),))
            elif k == "ring":
                r = D_BALL + 8 + (1 - L) * 50
                d.ellipse((e["x"] - r, e["y"] - r, e["x"] + r, e["y"] + r), outline=e["col"] + (int(200 * L),), width=6)
            elif k == "boom":
                r = 70 + (1 - L) * 170
                d.ellipse((e["x"] - r, e["y"] - r, e["x"] + r, e["y"] + r), fill=(255, 140, 40, int(160 * L)))
                r2 = r * 0.6
                d.ellipse((e["x"] - r2, e["y"] - r2, e["x"] + r2, e["y"] + r2), fill=(255, 240, 180, int(210 * L)))
            elif k == "shard":
                r = e["r"] * L + 2
                d.regular_polygon((e["x"], e["y"], r), 3, rotation=(1 - L) * 400, fill=e["col"] + (int(255 * L),))
            elif k == "drop":
                img = self.icons.get(e["w"])
                if img is not None:
                    g = img.rotate(-math.degrees(e["ang"]), resample=Image.BICUBIC, expand=True)
                    g.putalpha(g.getchannel("A").point(lambda v, f=L: int(v * f)))
                    im.alpha_composite(g, (int(e["x"] - g.width / 2), int(e["y"] - g.height / 2)))

    def ending(self, im, d, sim):
        w = sim.winner
        age = sim.t - sim.done_at
        if age < 0.6:
            return
        k = min(1.0, (age - 0.6) / 0.3)
        name = w["name"].upper() + " WINS!"
        f = self.f_win
        while d.textlength(name, font=f) > W - 180:
            f = _font(f.size - 4)
        s = 0.6 + 0.4 * k
        txt = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((W / 2, 100), name, font=f, fill=(255, 255, 255, 255), anchor="mm",
                                 stroke_width=10, stroke_fill=w["col"] + (255,))
        txt = txt.resize((int(W * s), int(200 * s)), Image.LANCZOS)
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(D_CY + 230 - txt.height / 2)))

    def watermark(self, d):
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((W - 60 - d.textlength(mark, font=self.f_small), H - 70), mark, font=self.f_small,
                   fill=(255, 255, 255, 150))


def _emoji(code: str, size: int) -> Image.Image:
    for name in (f"{code}.webp", f"{code}-fe0f.webp"):
        p = be.ASSETS / "emoji" / name
        if p.is_file():
            return Image.open(p).convert("RGBA").resize((size, size), Image.LANCZOS)
    return Image.new("RGBA", (size, size), (0, 0, 0, 0))


# ---- the race -----------------------------------------------------------------

R_CX, R_CY, R_R = 540, 1110, 470
R_BALL = 36
R_SPEED0, R_SPEED_MAX, R_SPEED_UP = 360.0, 640.0, 1.006
RACE_TARGET = 40
STEAL_REST = 1.5


class RaceSim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"race-{recipe['seed']}")
        self.target = int(recipe.get("target") or RACE_TARGET)
        self.t = 0.0
        self.balls = []
        for i, (w1, w2, col) in enumerate(recipe.get("players") or [list(c) for c in COLOURS[:4]]):
            a = self.rng.uniform(0, 6.28)
            p = i * math.pi / 2 + 0.6
            self.balls.append({"i": i, "name": w2, "col": tuple(col),
                               "x": R_CX + 160 * math.cos(p), "y": R_CY + 160 * math.sin(p),
                               "vx": R_SPEED0 * math.cos(a), "vy": R_SPEED0 * math.sin(a), "sp": R_SPEED0,
                               "anchors": np.zeros((0, 2)), "rest": np.zeros(0)})
        self.sfx: list = []
        self.events: list = []
        self.hits: list = []
        self.marks: list = []        # [x, y, i, life]
        self.unlocked_at = {0: 0.0}
        self.best, self.spawned = 0, 4
        self.winner = None
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0
        self.lead_changes = 0
        self._leader = None
        self._last_change = 0.0
        self._gaps: list = []

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def score(self, b) -> int:
        return len(b["anchors"])

    def step(self):
        dt = 1 / FPS
        self.t += dt
        for m in self.marks:
            m[3] -= dt * 2.2
        self.marks = [m for m in self.marks if m[3] > 0]
        if self.winner is not None:
            return
        for _ in range(4):
            self._move(dt / 4)
        self._steal()
        best = max(self.balls, key=self.score)
        top = self.score(best)
        if top >= self.target:
            self.winner, self.done_at = best, self.t
            self.sfx.append((self.t, "win", 0))
            self._gaps.append(self.t - self._last_change)
            second = sorted((self.score(b) for b in self.balls), reverse=True)[1]
            self.max_gap = max(self._gaps)
            # a close race with real lead changes
            self.ok = self.lead_changes >= 3 and top - second <= self.target * 0.3
            return
        if sum(self.score(b) == top for b in self.balls) == 1 and best["i"] != self._leader:
            if self._leader is not None and top >= 5:
                self.lead_changes += 1
                self._gaps.append(self.t - self._last_change)
                self._last_change = self.t
            self._leader = best["i"]
        if self.t > SECONDS_CAP - END_HOLD - 0.1:
            self.done_at = self.t
            self.ok = False

    def _move(self, dt):
        for b in self.balls:
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            dx, dy = b["x"] - R_CX, b["y"] - R_CY
            d = math.hypot(dx, dy)
            if d > R_R - R_BALL:
                nx, ny = dx / d, dy / d
                vn = b["vx"] * nx + b["vy"] * ny
                if vn > 0:
                    b["sp"] = min(R_SPEED_MAX, b["sp"] * R_SPEED_UP)
                    vx, vy = b["vx"] - 2 * vn * nx, b["vy"] - 2 * vn * ny
                    a = math.atan2(vy, vx) + self.rng.uniform(-0.12, 0.12)
                    b["vx"], b["vy"] = b["sp"] * math.cos(a), b["sp"] * math.sin(a)
                    ax, ay = R_CX + nx * R_R, R_CY + ny * R_R
                    b["anchors"] = np.vstack([b["anchors"], [ax, ay]])
                    b["rest"] = np.append(b["rest"], 0.0)
                    self.marks.append([ax, ay, b["i"], 1.0])
                    self.hits.append((self.t, 0.6))
                    self.sfx.append((self.t, "tie", b["i"]))
                b["x"], b["y"] = R_CX + nx * (R_R - R_BALL), R_CY + ny * (R_R - R_BALL)
        bs = self.balls
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, c = bs[i], bs[j]
                dx, dy = c["x"] - a["x"], c["y"] - a["y"]
                d = math.hypot(dx, dy)
                if 0 < d < 2 * R_BALL:
                    nx, ny = dx / d, dy / d
                    rel = (c["vx"] - a["vx"]) * nx + (c["vy"] - a["vy"]) * ny
                    if rel < 0:
                        a["vx"] += rel * nx
                        a["vy"] += rel * ny
                        c["vx"] -= rel * nx
                        c["vy"] -= rel * ny
                        for b in (a, c):
                            s = math.hypot(b["vx"], b["vy"]) or 1
                            b["vx"], b["vy"] = b["vx"] / s * b["sp"], b["vy"] / s * b["sp"]
                    over = 2 * R_BALL - d
                    a["x"] -= nx * over / 2
                    a["y"] -= ny * over / 2
                    c["x"] += nx * over / 2
                    c["y"] += ny * over / 2

    def _steal(self):
        """A ball that runs into another ball's line steals it: the line lets
        go of its owner and ties onto the thief (same spot on the wall). A
        stolen line can't change hands again for STEAL_REST seconds."""
        for a in self.balls:
            p = np.array([a["x"], a["y"]])
            for b in self.balls:
                if b is a or not len(b["anchors"]):
                    continue
                A = b["anchors"]
                AB = np.array([b["x"], b["y"]]) - A
                L2 = np.maximum(1e-6, np.sum(AB ** 2, axis=1))
                tt = np.clip(np.sum((p - A) * AB, axis=1) / L2, 0, 0.9)
                closest = A + AB * tt[:, None]
                dist = np.hypot(*(closest - p).T)
                take = (dist < R_BALL * 0.8) & (b["rest"] <= self.t)
                if take.any():
                    k = int(take.sum())
                    a["anchors"] = np.vstack([a["anchors"], A[take]])
                    a["rest"] = np.append(a["rest"], np.full(k, self.t + STEAL_REST))
                    b["anchors"], b["rest"] = A[~take], b["rest"][~take]
                    self.sfx.append((self.t, "steal", k))
                    self.marks.append([a["x"], a["y"], a["i"], 1.0])

    def music_track(self, n: int) -> np.ndarray:
        return _sfx_mix(self.recipe, self.sfx, self.hits, n, self)


class RacePainter:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.bg = Image.new("RGB", (W, H), (0, 0, 0))
        ImageDraw.Draw(self.bg).ellipse((R_CX - R_R - 3, R_CY - R_R - 3, R_CX + R_R + 3, R_CY + R_R + 3),
                                        outline=(240, 240, 245), width=5)
        players = recipe.get("players") or [list(c) for c in COLOURS[:4]]
        self.cols = [tuple(p[2]) for p in players]
        self.sprites, self.dots = [], []
        for col in self.cols:
            big = R_BALL * 2 * 3
            im = Image.new("RGBA", (big + 60, big + 60), (0, 0, 0, 0))
            ImageDraw.Draw(im).ellipse((30, 30, 30 + big, 30 + big), fill=col + (110,))
            im = im.filter(ImageFilter.GaussianBlur(14))
            ImageDraw.Draw(im).ellipse((30, 30, 30 + big, 30 + big), fill=col + (255,), outline=(255, 255, 255, 255),
                                       width=12)
            s = im.resize(((big + 60) // 3, (big + 60) // 3), Image.LANCZOS)
            self.sprites.append(s)
            dot = Image.new("RGBA", (156, 156), (0, 0, 0, 0))
            ImageDraw.Draw(dot).ellipse((6, 6, 150, 150), fill=col + (255,), outline=(255, 255, 255, 255), width=9)
            self.dots.append(dot.resize((50, 50), Image.LANCZOS))
        self.f_title = _font(78)
        self.f_score = _font(40, "Inter-Bold.ttf")
        self.f_label = _font(30, "Inter-Bold.ttf")
        self.f_win = _font(110)
        self.f_small = _font(34, "Inter-Bold.ttf")

    def frame(self, sim: RaceSim) -> Image.Image:
        lines = Image.new("RGB", (W, H), (0, 0, 0))
        dl = ImageDraw.Draw(lines)
        for b in sorted(sim.balls, key=sim.score):
            col = tuple(int(v * 0.85) for v in b["col"])
            if sim.winner is b and int((sim.t - sim.done_at) * 6) % 2 == 0:
                col = (255, 255, 255)
            for ax, ay in b["anchors"]:
                dl.line([(ax, ay), (b["x"], b["y"])], fill=col, width=3)
        im = Image.fromarray(np.maximum(np.asarray(self.bg), np.asarray(lines))).convert("RGBA")
        d = ImageDraw.Draw(im)
        for x, y, i, life in sim.marks:
            r = 26 + (1 - life) * 22
            d.ellipse((x - r, y - r, x + r, y + r), outline=self.cols[i] + (int(255 * life),), width=6)
        for b in sim.balls:
            sp = self.sprites[b["i"]]
            im.alpha_composite(sp, (int(b["x"] - sp.width / 2), int(b["y"] - sp.height / 2)))
            n = str(sim.score(b))
            tw = d.textlength(n, font=self.f_label)
            bx, by = b["x"], b["y"] - R_BALL - 32
            d.rounded_rectangle((bx - tw / 2 - 12, by - 21, bx + tw / 2 + 12, by + 21), 9, fill=(40, 40, 46, 225),
                                outline=(150, 150, 160, 255), width=2)
            d.text((bx, by), n, font=self.f_label, fill=(255, 255, 255), anchor="mm")
        order = sorted(sim.balls, key=lambda b: -sim.score(b))
        top = 430
        d.rounded_rectangle((16, top, 236, top + 66 * len(order) + 74), 12, fill=(18, 18, 22, 235))
        for k, b in enumerate(order):
            y = top + 38 + k * 66
            im.alpha_composite(self.dots[b["i"]], (30, y - 25))
            d.text((96, y), str(sim.score(b)), font=self.f_score, fill=(255, 255, 255), anchor="lm")
        yl = top + 38 + len(order) * 66 - 28
        d.line([(30, yl), (222, yl)], fill=(120, 120, 130), width=2)
        d.text((96, yl + 34), str(sum(sim.score(b) for b in sim.balls)), font=self.f_score, fill=(190, 190, 196),
               anchor="lm")
        lead = order[0]["col"] if sim.score(order[0]) else (255, 255, 255)
        d.text((W / 2, 320), f"First to {sim.target} wins", font=self.f_title, fill=lead, anchor="mm")
        if sim.winner is not None and sim.t - sim.done_at > 0.3:
            name = sim.winner["name"].upper() + " WINS!"
            d.text((W / 2, R_CY), name, font=self.f_win, fill=(255, 255, 255), anchor="mm", stroke_width=12,
                   stroke_fill=sim.winner["col"])
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((W - 60 - d.textlength(mark, font=self.f_small), H - 70), mark, font=self.f_small,
                   fill=(255, 255, 255, 150))
        return im.convert("RGB")


# ---- sound (numpy) -------------------------------------------------------------

def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


def _lp(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def _sound(kind: str, seed: int = 0) -> np.ndarray:
    def tt(dur):
        return np.arange(int(SR * dur)) / SR
    if kind == "hit":                 # a punchy thud with a crack
        t = tt(0.28)
        body = np.sin(2 * np.pi * np.cumsum(90 + 160 * np.exp(-t * 40)) / SR) * np.exp(-t * 14)
        crack = _lp(_noise(len(t), seed), 3) * np.exp(-t * 60) * 0.6
        return np.tanh((body + crack) * 2.2) * 0.8
    if kind == "shot":                # a pistol: crack + low punch
        t = tt(0.3)
        crack = _lp(_noise(len(t), seed), 2) * np.exp(-t * 45)
        punch = np.sin(2 * np.pi * np.cumsum(140 + 300 * np.exp(-t * 50)) / SR) * np.exp(-t * 20)
        return np.tanh((crack * 0.9 + punch) * 2.0) * 0.75
    if kind == "blast":               # a shotgun: bigger, longer
        t = tt(0.55)
        crack = _lp(_noise(len(t), seed), 4) * np.exp(-t * 18)
        punch = np.sin(2 * np.pi * np.cumsum(70 + 250 * np.exp(-t * 30)) / SR) * np.exp(-t * 9)
        return np.tanh((crack + punch * 1.3) * 2.5) * 0.9
    if kind == "rat":                 # one minigun round
        t = tt(0.07)
        return np.tanh(_lp(_noise(len(t), seed), 2) * np.exp(-t * 70) * 2.5) * 0.45
    if kind == "spin":                # the minigun spinning up
        t = tt(0.4)
        f = 300 + 900 * t / 0.4
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * np.sin(2 * np.pi * 40 * t)) * 0.15
    if kind == "launch":              # a rocket leaving the tube
        t = tt(0.6)
        return _lp(_noise(len(t), seed), 8) * np.minimum(1, t / 0.03) * np.exp(-t * 4) * 0.5
    if kind == "boom":
        t = tt(1.1)
        rumble = _lp(_noise(len(t), seed), 40) * 6
        thump = np.sin(2 * np.pi * np.cumsum(45 + 130 * np.exp(-t * 10)) / SR)
        return np.tanh((rumble + thump) * np.exp(-t * 3.5) * 2.2) * 0.95
    if kind == "clang":
        t = tt(0.5)
        return sum(np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, k in ((820, 9), (1310, 12), (2240, 16))) * 0.24
    if kind == "pickup":
        t = tt(0.2)
        return np.sin(2 * np.pi * np.cumsum(500 + 7000 * t) / SR) * np.minimum(1, t / 0.01) * np.exp(-t * 9) * 0.35
    if kind == "spawn":
        t = tt(0.25)
        return np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 20) * 0.12
    if kind == "die":
        t = tt(1.4)
        glass = _noise(len(t), seed) * np.exp(-t * 6) * 0.5
        thump = np.sin(2 * np.pi * np.cumsum(70 + 200 * np.exp(-t * 8)) / SR) * np.exp(-t * 3)
        return np.tanh(glass + thump * 1.5) * 0.9
    if kind == "win":
        t = tt(1.2)
        return np.sin(2 * np.pi * np.cumsum(60 + 150 * np.exp(-t * 8)) / SR) * np.exp(-t * 3) * 0.9
    if kind == "wall":
        t = tt(0.08)
        return _lp(_noise(len(t), seed), 8) * np.exp(-t * 70) * 0.12
    if kind == "bump":
        t = tt(0.15)
        return np.sin(2 * np.pi * 160 * t) * np.exp(-t * 30) * 0.4
    if kind == "tie":                 # a line tied on (race)
        t = tt(0.06)
        return np.sin(2 * np.pi * (880 + 120 * (seed % 4)) * t) * np.exp(-t * 70) * 0.16
    if kind == "steal":               # lines stolen (race)
        t = tt(0.2)
        return _lp(_noise(len(t), seed), 4) * np.exp(-t * 22) * 0.35
    return np.zeros(1)


def _sfx_mix(recipe: dict, sfx: list, hits: list, n: int, sim) -> np.ndarray:
    """The kind's own sounds over the music: Dean's song played straight
    through (a fight shouldn't stop and start with the bounces), else the
    recipe's bounce tune on the wall hits."""
    out = np.zeros(n, dtype=np.float32)
    cache: dict = {}
    last = {}
    for k, (t, kind, val) in enumerate(sfx):
        gap = {"rat": 0.05, "tie": 0.03, "steal": 0.06, "wall": 0.05}.get(kind, 0.0)
        if gap and t - last.get(kind, -9) < gap:
            continue
        last[kind] = t
        key = (kind, val if kind == "tie" else k % 3)
        w = cache.get(key)
        if w is None:
            w = cache[key] = _sound(kind, val if kind == "tie" else k % 3).astype(np.float32)
        i = int(t * SR)
        seg = out[i:i + len(w)]
        seg += w[:len(seg)]
    song = recipe.get("song") if recipe.get("bounce_sound") == "song" else None
    music = None
    if song:
        from clipper import ball_sounds
        try:
            music = ball_sounds.load_song(song["path"], float(song.get("start") or 0.0), n / SR)
            music = music / (float(np.max(np.abs(music))) or 1.0) * 0.6
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            print(f"song failed, playing the tune instead: {e}", flush=True)
            music = None
    if music is None:
        from clipper import ball_merge as bm
        music = bm.MergeSim.music_track(sim, n) * 0.8
    music = np.pad(music, (0, max(0, n - len(music))))[:n]
    return out * 0.8 + music


PARTS = {"duel": (DuelSim, DuelPainter), "race": (RaceSim, RacePainter)}
