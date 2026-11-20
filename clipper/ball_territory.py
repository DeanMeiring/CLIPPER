"""Which country owns the map? A territory war (Nov 2026).

From the "colour war" / Pong-wars Shorts: a grid of tiles split between 2-4
countries. Each country's ball bounces around its own colour; every tile of
another colour it touches flips to its colour and the ball bounces off it,
so the borders swing back and forth. Viewers pick a side in the comments.

``format: "territory"``, a ``NO_PACK`` kind (the balls are the countries);
``ball_evolution._parts`` dispatches here. The teams are drawn once in
``territory_recipe`` (own rng ``territory-recipe-<seed>``); ``pick_seed``
only changes the seed of the run (``territory-<seed>``).

How it ends: from `T_ESC` s a country gets a bonus ball now and then (the
ball splits) and everyone speeds up; a country whose share drops under
`OUT_SHARE` is out (its ball pops, its tiles go grey for anyone to take);
at the deadline (`T_END_MIN`..`T_END_MAX` s, a timer on screen) the
country with the most tiles wins, and its colour floods the rest of the map.
A lead change only counts with a margin (`LEAD_MARGIN`), so the
"NEW LEADER" banner and its whoosh don't flicker on a near tie.

Drawing is from the numpy owner grid (palette lookup, ``np.repeat`` up to
pixels, a precomputed tile bevel), not one ImageDraw call per tile. Sound
(``music_track``) is ours, numpy only: a soft tick per flipped tile (rate
limited, pitched per team), a pop when a country is out, a whoosh on a lead
change, a thump at the end.
"""
from __future__ import annotations

import math
import random
import subprocess

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from clipper import ball_duel as bd
from functools import lru_cache

from clipper.ball_evolution import FPS, H, SR, W
from clipper.ball_evolution import _font as _be_font


@lru_cache(maxsize=256)
def _font(size: int, name: str = "Inter-Black.ttf"):
    return _be_font(size, name)

END_HOLD = 1.8
SECONDS_CAP = 60.0
G = 48                                  # tiles per side
TILE = 20                               # px
X0, Y0 = (W - G * TILE) // 2, 520       # the map's top-left corner (960 px square)
BALL_R = 24
SUBSTEPS = 3
V0 = 560.0                              # px/s
T_ESC = 3.0                             # bonus balls + speed-up from here
T_ESC_2 = 1.5                           # ... with only 2 countries (a straight border is slow to move)
BONUS_EVERY = (1.6, 0.8)                # s between bonus balls: at T_ESC, at the deadline
SPEEDUP = 0.018                         # +1.8% speed per second after T_ESC
MAX_BALLS_TEAM = 8
PIERCE = (3, 14)                        # extra tiles a ball digs through before bouncing: start, deadline
BEHIND_W = 0.0                          # bonus odds: + this much for the country furthest behind
BOMB_SHARE = 0.3                        # share of the bonuses that are a bomb instead of a ball
BOMB_R = 5.5                            # tiles
OUT_SHARE = 0.04                        # under this share of the map you're out
OUT_AFTER = 5.0
T_END_MIN, T_END_MAX = 26.0, 32.0       # the deadline (seeded per run)
WIN_SHARE = 0.70                        # ... or a country holding this much wins early (after 14 s)
FLOOD = 0.8                             # s for the winner's colour to flood the map
LEAD_MARGIN = 0.008                     # a new leader must be this far ahead (share)
POP_S = 0.18                            # a flipped tile flashes this long
MIN_LEADS = {2: 2, 3: 3, 4: 5}          # a good run: this many lead changes (by team count) ...
LATE_S = 6.0                            # ... the winner took the lead in the last 6 s or came from behind ...
MIN_MARGIN = 0.03                       # ... and wins by a visible margin ...
MAX_GAP = 12.0                          # ... with no 12 s stretch without a lead change
NEUTRAL = (70, 72, 86)                  # tiles of a country that's out
BG = (10, 11, 24)

HOOKS = ["Which country owns the map?", "Pick a country before it starts", "Who takes the whole map?",
         "{n} countries, 1 map. Pick one", "Every tile it hits turns its colour", "Pick a side. Only one owns the map"]
VS_HOOKS = ["{a} VS {b}: who owns the map?", "{a} or {b}? Pick one", "Who takes the map? {a} VS {b}"]
ENDINGS = ["OWNS THE MAP!", "TAKES THE MAP!", "WINS THE MAP!"]
# fallback colours if the flags' main colours are too close
FALLBACK = [(230, 50, 60), (70, 130, 245), (250, 205, 40), (40, 180, 90), (250, 140, 40), (160, 90, 230)]


def _dist(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _country_colour(code: str, taken: list):
    """The flag's main colour, else its second colour, if far enough from the
    colours already taken."""
    for col in (bd.COUNTRIES[code][1], bd.ALT_COLOURS.get(code)):
        if col and all(_dist(col, c) > 150 for c in taken):
            return col
    return None


def territory_recipe(recipe: dict) -> dict:
    rng = random.Random(f"territory-recipe-{recipe['seed']}")
    n = rng.choices((2, 3, 4), weights=(2, 2, 6))[0]
    teams: list = []
    if n == 2:
        pairs = bd.RIVALRIES[:]
        rng.shuffle(pairs)
        for a, b in pairs:
            ca = bd.COUNTRIES[a][1]
            cb = _country_colour(b, [ca])
            if cb:
                teams = [[bd.COUNTRIES[a][0], a, list(ca)], [bd.COUNTRIES[b][0], b, list(cb)]]
                break
    if not teams:
        codes = sorted(bd.COUNTRIES)
        rng.shuffle(codes)
        for code in codes:
            col = _country_colour(code, [tuple(t[2]) for t in teams])
            if col:
                teams.append([bd.COUNTRIES[code][0], code, list(col)])
            if len(teams) == n:
                break
        for i, t in enumerate(teams):           # (never needed with 46 countries; kept as a guard)
            if any(_dist(t[2], o[2]) < 120 for o in teams[:i]):
                t[2] = list(FALLBACK[i])
    names = [t[0] for t in teams]
    hook = (rng.choice(VS_HOOKS).format(a=names[0], b=names[1]) if n == 2 and rng.random() < 0.6
            else rng.choice(HOOKS).format(n=n))
    recipe.update(look="flags", teams=teams, hook=hook, ending=rng.choice(ENDINGS), bed="off")
    return recipe


def flag_emoji(code: str) -> str:
    """The flag as an emoji for the title (England / Scotland are tag sequences)."""
    if code.startswith("gb-"):
        tag = "gb" + code[3:]
        return "\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in tag) + "\U000E007F"
    return "".join(chr(0x1F1E6 + ord(c) - ord("a")) for c in code[:2])


def publish_text(recipe: dict, info: dict | None = None) -> dict:
    teams = recipe.get("teams") or []
    flags = " ".join(flag_emoji(t[1]) for t in teams)
    title = f"{recipe['hook']} {flags}".strip()[:100]
    names = ", ".join(t[0] for t in teams)
    desc = (f"{names}: each country's ball bounces around the map, and every tile it hits turns its colour. "
            f"The one with the most tiles when the time runs out owns the map.\n\n"
            f"Which one did you pick? 👇\n\n#satisfying #countryballs #flags #bouncingball\n\n"
            f"Flags: flag-icons (MIT)")
    return {"title": title, "description": desc}


def _start_owners(n: int) -> np.ndarray:
    o = np.zeros((G, G), np.int16)            # [row, col]
    yy, xx = np.mgrid[0:G, 0:G]
    if n == 2:
        o[:, G // 2:] = 1
    elif n == 3:
        # 3 slices of a pie, cut by tile count so each starts with exactly a third
        a = ((np.degrees(np.arctan2(yy - (G - 1) / 2, xx - (G - 1) / 2)) + 90) % 360).ravel()
        rank = np.empty(G * G, np.int64)
        rank[np.argsort(a, kind="stable")] = np.arange(G * G)
        o[:] = (rank * 3 // (G * G)).reshape(G, G).astype(np.int16)
    else:
        o[:G // 2, G // 2:] = 1
        o[G // 2:, :G // 2] = 2
        o[G // 2:, G // 2:] = 3
    return o


class TerritorySim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"territory-{recipe['seed']}")
        self.teams = [{"i": i, "name": t[0], "flag": t[1], "col": tuple(t[2]), "out_at": None}
                      for i, t in enumerate(recipe.get("teams") or [])]
        self.n = len(self.teams)
        self.owner = _start_owners(self.n)
        self.flip_t = np.full((G, G), -9.0)
        self.flip_by = np.zeros((G, G), np.int16)
        self.t = 0.0
        self.t_end = self.rng.uniform(T_END_MIN, T_END_MAX)
        self.balls: list = []
        for tm in self.teams:
            ys, xs = np.nonzero(self.owner == tm["i"])
            cx, cy = X0 + (xs.mean() + 0.5) * TILE, Y0 + (ys.mean() + 0.5) * TILE
            self._add_ball(tm["i"], cx, cy)
        self.chain = [(None, "start"), (None, "map owned")]   # (render's info: unlocks = start, result)
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.spawned = len(self.balls)
        self.events: list = []            # the shared mixer's ball sounds: none (music_track instead)
        self.sfx: list = []               # (t, kind, value)
        self.shards: list = []            # (t, x, y, colour) of popped balls
        self.share = self._shares()
        self.history: list = [(0.0, self.share.copy())]
        self.leader = None                # nobody leads a tied start
        self.leads: list = []             # (t, team) every time the lead changes hands
        self.esc = T_ESC if self.n > 2 else T_ESC_2
        self.next_bonus = self.esc
        self.bonus: list = []             # (t, team, x, y)
        self.bombs: list = []             # (t, team, x, y)
        self.decided_at = None
        self.winner = None
        self.flood_from = None
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0
        self.lead_changes = 0
        self.half_leader = None
        self.half_seen = False

    # ---- the sim --------------------------------------------------------
    def _add_ball(self, team, x, y, ang=None):
        a = self.rng.uniform(0, 2 * math.pi) if ang is None else ang
        # never straight along an axis (it could slide along a border forever)
        if min(abs(math.sin(a)), abs(math.cos(a))) < 0.25:
            a += 0.4
        self.balls.append({"team": team, "x": float(x), "y": float(y), "vx": math.cos(a), "vy": math.sin(a),
                           "px": float(x), "py": float(y), "born": self.t, "dig": 0})

    def _shares(self) -> np.ndarray:
        c = np.bincount(self.owner.ravel() + 1, minlength=self.n + 1)[1:self.n + 1].astype(float)   # -1 = grey
        c[[t["i"] for t in self.teams if t["out_at"] is not None]] = 0
        return c / (G * G)

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def speed(self) -> float:
        return V0 * (1 + SPEEDUP * max(0.0, self.t - self.esc))

    def pierce(self) -> int:
        """How many enemy tiles in a row a ball flips before it bounces:
        1 at first (plain Pong wars), more as the deadline nears."""
        if self.t < self.esc:
            return PIERCE[0]
        k = min(1.0, (self.t - self.esc) / max(1.0, self.t_end - self.esc))
        return int(round(PIERCE[0] + (PIERCE[1] - PIERCE[0]) * k))

    def alive(self) -> list:
        return [t for t in self.teams if t["out_at"] is None]

    def step(self):
        dt = 1 / FPS
        self.t += dt
        for b in self.balls:
            b["px"], b["py"] = b["x"], b["y"]
        if self.decided_at is not None:
            self._flood()
            return
        sp = self.speed()
        for _ in range(SUBSTEPS):
            for b in self.balls:
                self._move(b, sp * dt / SUBSTEPS)
        self.share = self._shares()
        self.history.append((self.t, self.share.copy()))
        self._lead()
        if not self.half_seen and self.t >= self.t_end / 2:
            self.half_seen, self.half_leader = True, self.leader
        self._outs()
        self._bonus()
        alive = self.alive()
        if (self.t >= self.t_end or len(alive) == 1
                or (self.t > 14 and float(self.share.max()) >= WIN_SHARE)):
            self._decide()

    def _move(self, b, d):
        b["x"] += b["vx"] * d
        b["y"] += b["vy"] * d
        if b["x"] - BALL_R < X0:
            b["x"], b["vx"] = X0 + BALL_R, abs(b["vx"])
        elif b["x"] + BALL_R > X0 + G * TILE:
            b["x"], b["vx"] = X0 + G * TILE - BALL_R, -abs(b["vx"])
        if b["y"] - BALL_R < Y0:
            b["y"], b["vy"] = Y0 + BALL_R, abs(b["vy"])
        elif b["y"] + BALL_R > Y0 + G * TILE:
            b["y"], b["vy"] = Y0 + G * TILE - BALL_R, -abs(b["vy"])
        team = b["team"]
        nx = ny = 0.0
        hit = False
        for k in range(8):
            a = k * math.pi / 4
            ca, sa = math.cos(a), math.sin(a)
            col = int((b["x"] + ca * BALL_R - X0) // TILE)
            row = int((b["y"] + sa * BALL_R - Y0) // TILE)
            if not (0 <= col < G and 0 <= row < G):
                continue
            if self.owner[row, col] != team:
                self.owner[row, col] = team
                self.flip_t[row, col] = self.t
                self.flip_by[row, col] = team
                nx += ca
                ny += sa
                hit = True
        if hit:
            self.sfx.append((self.t, "flip", team))
            if b["dig"] > 0:                  # still digging through: no bounce yet
                b["dig"] -= 1
                return
            b["dig"] = self.pierce()
            m = math.hypot(nx, ny)
            if m > 1e-6:
                nx, ny = nx / m, ny / m
                dot = b["vx"] * nx + b["vy"] * ny
                if dot > 0:
                    b["vx"] -= 2 * dot * nx
                    b["vy"] -= 2 * dot * ny
            a = math.atan2(b["vy"], b["vx"]) + self.rng.uniform(-0.08, 0.08)   # tiny turns: no loops
            if min(abs(math.sin(a)), abs(math.cos(a))) < 0.15:
                a += self.rng.choice((-0.2, 0.2))
            b["vx"], b["vy"] = math.cos(a), math.sin(a)

    def _lead(self):
        top = int(np.argmax(self.share))
        if self.leader is not None and self.teams[self.leader]["out_at"] is not None:
            self.leader = None
        ref = (float(np.partition(self.share, -2)[-2]) if self.leader is None and self.n > 1
               else float(self.share[self.leader]) if self.leader is not None else 0.0)
        if top != self.leader and self.share[top] - ref >= LEAD_MARGIN:
            self.leader = top
            self.leads.append((self.t, top))
            self.lead_changes += 1
            self.sfx.append((self.t, "lead", top))

    def _outs(self):
        if self.t < OUT_AFTER:
            return
        for tm in self.alive():
            if self.share[tm["i"]] < OUT_SHARE and len(self.alive()) > 1:
                tm["out_at"] = self.t
                for b in [b for b in self.balls if b["team"] == tm["i"]]:
                    self.shards.append((self.t, b["x"], b["y"], tm["col"]))
                    self.balls.remove(b)
                m = self.owner == tm["i"]
                self.owner[m] = -1
                self.flip_t[m] = self.t
                self.flip_by[m] = -1
                self.sfx.append((self.t, "out", tm["i"]))
                self.share = self._shares()

    def _bonus(self):
        if self.t < self.next_bonus:
            return
        k = min(1.0, (self.t - self.esc) / max(1.0, self.t_end - self.esc))
        self.next_bonus = self.t + BONUS_EVERY[0] + (BONUS_EVERY[1] - BONUS_EVERY[0]) * k
        alive = [t for t in self.alive()
                 if sum(1 for b in self.balls if b["team"] == t["i"]) < MAX_BALLS_TEAM]
        if not alive:
            return
        w = [1.0 + BEHIND_W * (1.0 - self.share[t["i"]] / max(1e-6, float(self.share.max()))) for t in alive]
        tm = self.rng.choices(alive, weights=w)[0]
        src = self.rng.choice([b for b in self.balls if b["team"] == tm["i"]])
        if self.rng.random() < BOMB_SHARE:
            # a bomb: every tile near the ball turns its colour at once
            col = (src["x"] - X0) / TILE
            row = (src["y"] - Y0) / TILE
            yy, xx = np.mgrid[0:G, 0:G]
            m = (np.hypot(yy + 0.5 - row, xx + 0.5 - col) <= BOMB_R) & (self.owner != tm["i"])
            self.owner[m] = tm["i"]
            self.flip_t[m] = self.t
            self.flip_by[m] = tm["i"]
            self.bombs.append((self.t, tm["i"], src["x"], src["y"]))
            self.sfx.append((self.t, "bomb", tm["i"]))
            return
        a = math.atan2(src["vy"], src["vx"]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.6, 1.4)
        self._add_ball(tm["i"], src["x"], src["y"], a)
        self.spawned += 1
        self.bonus.append((self.t, tm["i"], src["x"], src["y"]))
        self.sfx.append((self.t, "bonus", tm["i"]))

    def _decide(self):
        sh = self.share
        self.final_share = sh.copy()          # the score the HUD keeps showing while the map floods
        w = int(np.argmax(sh))
        self.winner = self.teams[w]
        self.decided_at = self.t
        ys, xs = np.nonzero(self.owner == w)
        self.flood_from = (ys.mean(), xs.mean()) if len(xs) else (G / 2, G / 2)
        yy, xx = np.mgrid[0:G, 0:G]
        d = np.hypot(yy - self.flood_from[0], xx - self.flood_from[1])
        self._flood_d = d / max(1e-6, d.max())
        order = sorted(sh, reverse=True)
        margin = order[0] - (order[1] if len(order) > 1 else 0)
        times = [0.0] + [t for t, _ in self.leads] + [self.t]
        self.max_gap = max(b - a for a, b in zip(times, times[1:]))
        self.done_at = self.t + FLOOD        # the result shows once the map is all theirs
        comeback = self.half_leader is not None and self.half_leader != w
        late = bool(self.leads) and self.t - self.leads[-1][0] < LATE_S
        self.ok = (self.lead_changes >= MIN_LEADS.get(self.n, 3) and (comeback or late) and margin >= MIN_MARGIN
                   and self.max_gap <= MAX_GAP)
        self.margin = margin
        self.sfx.append((self.t, "win", w))
        self.sfx.append((self.t + FLOOD, "thump", w))
        self.unlocked_at[1] = self.t + FLOOD

    def _flood(self):
        k = (self.t - self.decided_at) / FLOOD
        w = self.winner["i"]
        m = (self._flood_d <= k) & (self.owner != w)
        if m.any():
            self.owner[m] = w
            self.flip_t[m] = self.t
            self.flip_by[m] = w
            self.sfx.append((self.t, "flip", w))
        self.share = self._shares()
        for b in self.balls:                  # the balls drift to a stop
            b["x"] += b["vx"] * V0 / FPS * max(0.0, 1 - k)
            b["y"] += b["vy"] * V0 / FPS * max(0.0, 1 - k)
            b["x"] = min(max(b["x"], X0 + BALL_R), X0 + G * TILE - BALL_R)
            b["y"] = min(max(b["y"], Y0 + BALL_R), Y0 + G * TILE - BALL_R)

    # ---- sound ----------------------------------------------------------
    def music_track(self, n: int) -> np.ndarray:
        g = np.random.default_rng(int(self.recipe.get("seed") or 0))
        out = np.zeros(n, np.float32)
        last: dict = {}
        cache: dict = {}
        base = [520, 660, 780, 940]
        for et, kind, v in self.sfx:
            i = int(et * SR)
            if i >= n:
                continue
            if kind == "flip":
                if et - last.get(("flip", v), -9) < 0.07 or et - last.get("flip", -9) < 0.025:
                    continue
                last[("flip", v)] = last["flip"] = et
                key = ("flip", v, len(cache) % 3)
                w = cache.get(key)
                if w is None:
                    ts = np.arange(int(SR * 0.05)) / SR
                    f = base[v % 4] * g.uniform(0.96, 1.04)
                    w = cache[key] = ((np.sin(2 * np.pi * f * ts) * 0.7 + g.normal(0, 1, len(ts)) * 0.12)
                                      * np.exp(-ts * 90) * 0.22).astype(np.float32)
                vol = 0.5 if self.decided_at is not None and et > self.decided_at else 1.0
                w = w * vol
            elif kind == "bonus":
                ts = np.arange(int(SR * 0.16)) / SR
                f = base[v % 4] * (1 + ts * 5)
                w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ts * 18) * 0.3
            elif kind == "out":
                ts = np.arange(int(SR * 0.35)) / SR
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(8) / 8, "same")
                f = 300 * np.exp(-ts * 9) + 60
                w = (noise * 0.5 + np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.8) * np.exp(-ts * 10) * 0.6
            elif kind == "lead":
                d = 0.55
                ts = np.arange(int(SR * d)) / SR
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(30) / 30, "same")
                w = noise * np.sin(np.pi * ts / d) ** 2 * 1.4
            elif kind == "win":
                d = 0.8
                ts = np.arange(int(SR * d)) / SR
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(20) / 20, "same")
                w = noise * np.sin(np.pi * ts / d) * 1.2
            elif kind == "bomb":
                ts = np.arange(int(SR * 0.6)) / SR
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(12) / 12, "same")
                f = 90 * np.exp(-ts * 6) + 45
                w = (noise * 0.9 + np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.9) * np.exp(-ts * 6) * 0.75
            else:                               # "thump"
                ts = np.arange(int(SR * 1.2)) / SR
                f = 120 * np.exp(-ts * 5) + 40
                w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ts * 3.5) * 0.9
            seg = out[i:i + len(w)]
            seg += np.asarray(w[:len(seg)], np.float32)
        peak = float(np.abs(out).max()) or 1.0
        out = out / max(1.0, peak / 0.9)
        song = self.recipe.get("song") if self.recipe.get("bounce_sound") == "song" else None
        if song:
            from clipper import ball_sounds
            try:
                m = ball_sounds.load_song(song["path"], float(song.get("start") or 0.0), n / SR)
                m = np.pad(m, (0, max(0, n - len(m))))[:n]
                out = out * 0.8 + m / (float(np.max(np.abs(m))) or 1.0) * 0.55
            except (OSError, RuntimeError, subprocess.SubprocessError) as e:
                print(f"song failed, keeping the tile sounds: {e}", flush=True)
        return out


# ---- drawing ---------------------------------------------------------------

def _bevel() -> Image.Image:
    """The tile seams, tiled over the map: multiplied over the colours (a
    dark line between tiles)."""
    t = np.full((TILE, TILE), 255, np.uint8)
    t[-2:, :] = 158
    t[:, -2:] = 158
    t[0, :] = 235
    t[:, 0] = 235
    return Image.fromarray(np.tile(t, (G, G))).convert("RGB")


class TerritoryPainter:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.teams = recipe.get("teams") or []
        self.cols = np.array([t[2] for t in self.teams] + [NEUTRAL], np.float32)   # index -1 = grey
        self.bg = Image.new("RGBA", (W, H), BG + (255,))
        d = ImageDraw.Draw(self.bg)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).rectangle((X0 - 10, Y0 - 10, X0 + G * TILE + 10, Y0 + G * TILE + 10),
                                       outline=(140, 160, 255, 160), width=14)
        self.bg.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)))
        d.rectangle((X0 - 6, Y0 - 6, X0 + G * TILE + 5, Y0 + G * TILE + 5), outline=(235, 238, 255, 255), width=5)
        self.bevel = _bevel()
        # the hook: its own layer, not baked into bg (an art look swaps bg out)
        self.hook_layer = Image.new("RGBA", (W, 150), (0, 0, 0, 0))
        hd = ImageDraw.Draw(self.hook_layer)
        hook = recipe.get("hook") or HOOKS[0]
        f = _font(58)
        while hd.textlength(hook, font=f) > W - 80 and f.size > 30:
            f = _font(f.size - 4)
        hd.text((W / 2, 110), hook, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=5,
                stroke_fill=BG + (255,))
        self.f_hook = _font(58)
        self.f_sub = _font(40, "Inter-Bold.ttf")
        self.f_pct = _font(44)
        self.f_name = _font(28, "Inter-Bold.ttf")
        self.f_banner = _font(54)
        self.f_win = _font(96)
        self.f_small = _font(34, "Inter-Bold.ttf")
        self.sprites: dict = {}
        self.crown = None
        try:
            from clipper.ball_evolution import ASSETS
            self.crown = Image.open(ASSETS / "emoji" / "1f451.webp").convert("RGBA").resize((54, 54), Image.LANCZOS)
        except OSError:
            pass

    def sprite(self, i, size):
        key = (i, size)
        im = self.sprites.get(key)
        if im is None:
            s3 = size * 3
            im = Image.new("RGBA", (s3 + 18, s3 + 18), (0, 0, 0, 0))
            dd = ImageDraw.Draw(im)
            dd.ellipse((0, 0, s3 + 17, s3 + 17), fill=(10, 11, 24, 230))           # a dark rim: seen on any colour
            im.alpha_composite(bd._flag(self.teams[i][1], s3), (9, 9))
            dd.ellipse((9, 9, 9 + s3, 9 + s3), outline=(255, 255, 255, 255), width=10)
            im = self.sprites[key] = im.resize((size + 6, size + 6), Image.LANCZOS)
        return im

    def grid(self, sim: TerritorySim) -> Image.Image:
        col = self.cols[sim.owner]                               # (G, G, 3); -1 picks the grey
        pop = np.clip(1 - (sim.t - sim.flip_t) / POP_S, 0, 1)[..., None]
        if pop.any():                     # a flipped tile flashes lighter for a moment (never fully white)
            col = col + (255 - col) * pop * 0.55
        small = Image.fromarray(col.astype(np.uint8), "RGB").resize((G * TILE, G * TILE), Image.NEAREST)
        return ImageChops.multiply(small, self.bevel)

    def frame(self, sim: TerritorySim) -> Image.Image:
        im = self.bg.copy()
        im.paste(self.grid(sim), (X0, Y0))
        # popped balls: a ring of shards flying out
        d = ImageDraw.Draw(im)
        for t0, x, y, c in sim.shards:
            age = sim.t - t0
            if age > 0.6:
                continue
            for k in range(10):
                a = k * math.pi / 5 + 0.3
                r = 20 + age * 360
                q = 9 * (1 - age / 0.6)
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                d.ellipse((px - q, py - q, px + q, py + q), fill=tuple(c) + (255,), outline=(255, 255, 255, 255), width=2)
        for t0, i, x, y in sim.bombs:          # a bomb: a shockwave ring + "BOOM!"
            age = sim.t - t0
            if age < 0.8 and sim.decided_at is None:
                a = int(255 * min(1.0, (0.8 - age) / 0.35))
                r = BOMB_R * TILE * min(1.0, age / 0.18) + 16
                c = tuple(self.teams[i][2])
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, a), width=9)
                d.ellipse((x - r + 9, y - r + 9, x + r - 9, y + r - 9), outline=c + (a,), width=5)
                ly = y - r - 30 if y - r - 30 > Y0 + 30 else y + r + 30   # near the top: under the ring
                d.text((x, ly), "BOOM!", font=self.f_banner, fill=(255, 255, 255, a), anchor="mm",
                       stroke_width=6, stroke_fill=bd._stroke(c) + (a,))
        for t0, i, x, y in sim.bonus:
            age = sim.t - t0
            if age < 0.7:
                a = int(255 * min(1.0, (0.7 - age) / 0.3))
                r = 24 + age * 90
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, a), width=5)
                d.text((x, y - 40 - age * 60), "+1", font=self.f_banner, fill=(255, 255, 255, a), anchor="mm",
                       stroke_width=5, stroke_fill=bd._stroke(self.teams[i][2]) + (a,))
        fade = None
        if sim.decided_at is not None:              # the losers' balls fade out as the map floods
            fade = max(0.0, 1 - (sim.t - sim.decided_at) / (FLOOD * 0.8))
        for b in sim.balls:
            sp = self.sprite(b["team"], 2 * BALL_R)
            if fade is not None and b["team"] != sim.winner["i"]:
                if fade <= 0:
                    continue
                sp = sp.copy()
                sp.putalpha(sp.getchannel("A").point(lambda v, f=fade: int(v * f)))
            im.alpha_composite(sp, (int(b["x"] - sp.width / 2), int(b["y"] - sp.height / 2)))
        im.alpha_composite(self.hook_layer, (0, 0))
        self.hud(im, d, sim)
        if sim.decided_at is not None:
            self.ending(im, sim)
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d.text((W - 60 - d.textlength(mark, font=self.f_small), H - 70), mark, font=self.f_small,
                   fill=(255, 255, 255, 150))
        return im.convert("RGB")

    def hud(self, im, d, sim):
        share = sim.final_share if sim.decided_at is not None else sim.share
        # "Pick one" with a drawn arrow (the fonts have no emoji), then the clock
        if sim.t < 3.0 and sim.decided_at is None:
            txt = "Pick one"
            tw = d.textlength(txt, font=self.f_sub)
            x = W / 2 - (tw + 46) / 2
            d.text((x, 182), txt, font=self.f_sub, fill=(255, 214, 80, 255), anchor="lm")
            ax = x + tw + 30
            d.polygon([(ax - 14, 172), (ax + 14, 172), (ax + 14, 172), (ax, 198)], fill=(255, 214, 80, 255))
            d.rectangle((ax - 5, 160, ax + 5, 176), fill=(255, 214, 80, 255))
        elif sim.decided_at is None:
            left = max(0, math.ceil(sim.t_end - sim.t))
            col = (255, 90, 90, 255) if left <= 5 else (200, 205, 230, 255)
            d.text((W / 2, 182), f"{left}s LEFT", font=self.f_sub, fill=col, anchor="mm")
        # the teams: flag, name, share; the leader wears the crown
        n = len(self.teams)
        cw = (W - 80) / n
        for i, t in enumerate(self.teams):
            cx = 40 + cw * (i + 0.5)
            out = sim.teams[i]["out_at"] is not None
            sp = self.sprite(i, 64)
            if out:
                sp = sp.copy()
                sp.putalpha(sp.getchannel("A").point(lambda v: int(v * 0.35)))
            im.alpha_composite(sp, (int(cx - sp.width / 2), int(262 - sp.height / 2)))
            if out:
                d.line([(cx - 30, 232), (cx + 30, 292)], fill=(255, 70, 80, 255), width=8)
                d.line([(cx - 30, 292), (cx + 30, 232)], fill=(255, 70, 80, 255), width=8)
            lead = (sim.winner["i"] == i) if sim.winner is not None else (sim.leader == i)
            if lead and not out and self.crown is not None:
                im.alpha_composite(self.crown, (int(cx + 14), 214))
            name = t[0].upper()
            fn = self.f_name
            while d.textlength(name, font=fn) > cw - 16 and fn.size > 18:
                fn = _font(fn.size - 2, "Inter-Bold.ttf")
            d.text((cx, 318), name, font=fn, fill=(255, 255, 255, 120 if out else 255), anchor="mm")
            pct = "OUT" if out else f"{share[i] * 100:.0f}%"
            d.text((cx, 362), pct, font=self.f_pct, fill=tuple(t[2]) + (255,) if not out else (255, 70, 80, 255),
                   anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, 255))
        # the stacked bar
        bx0, bx1, by0, by1 = X0, X0 + G * TILE, 410, 452
        d.rounded_rectangle((bx0 - 4, by0 - 4, bx1 + 4, by1 + 4), radius=12, fill=(30, 32, 52, 255))
        x = bx0
        tot = float(share.sum()) or 1.0
        for i, t in enumerate(self.teams):
            wpx = (bx1 - bx0) * share[i] / tot
            if wpx >= 1:
                d.rectangle((x, by0, x + wpx, by1), fill=tuple(t[2]) + (255,))
            x += wpx
        # NEW LEADER banner, under the bar
        if sim.decided_at is None and sim.leads:
            t0, i = sim.leads[-1]
            age = sim.t - t0
            if age < 1.1:
                a = int(255 * min(1.0, (1.1 - age) / 0.3, age / 0.08 if age > 0 else 0))
                text = f"NEW LEADER: {self.teams[i][0].upper()}"
                s = 1 + 0.25 * max(0.0, 1 - age / 0.15)
                fb = _font(int(self.f_banner.size * s))
                while d.textlength(text, font=fb) > W - 60:
                    fb = _font(fb.size - 3)
                d.text((W / 2, 486), text, font=fb, fill=(255, 255, 255, a), anchor="mm", stroke_width=6,
                       stroke_fill=bd._stroke(self.teams[i][2]) + (a,))

    def ending(self, im, sim):
        w = sim.winner
        age = sim.t - sim.decided_at
        if age < 0.2:
            return
        k = min(1.0, (age - 0.2) / 0.25)
        line1 = w["name"].upper()
        line2 = self.recipe.get("ending") or ENDINGS[0]
        txt = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
        dd = ImageDraw.Draw(txt)
        stroke = (20, 22, 40, 255)
        for text, y, size in ((line1, 90, 120), (line2, 220, 84)):
            f = _font(size)
            while dd.textlength(text, font=f) > W - 100:
                f = _font(f.size - 4)
            dd.text((W / 2, y), text, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=12,
                    stroke_fill=stroke)
        s = 0.6 + 0.4 * k
        txt = txt.resize((int(W * s), int(300 * s)), Image.LANCZOS)
        cy = Y0 + G * TILE / 2
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(cy - txt.height / 2)))
