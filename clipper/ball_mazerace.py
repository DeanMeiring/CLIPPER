"""Pick one before it starts: a maze race of flag balls (Nov 2026).

From reels that broke out in this niche: "Horse Race Tests" (icons bouncing
through a maze to a goal, 1.8-5M views) and "Pick one 🏁 No switching
allowed" marble races (15.7M views, 15k comments). The point is the pick:
the viewer chooses a racer in the first second and stays to see if they
were right.

4-6 flag balls start together in a room at the top of a maze and race to
the trophy in the middle. Each keeps a steady speed with no gravity,
bounces off the walls and off each other, and is pulled along the maze's
shortest way to the goal (a distance field on a fine grid) -- but every
racer's sense of direction drifts on its own (a slow random wander), so a
leader can take a wrong turn and the race turns around. First to touch the
trophy wins.

``format: "mazerace"``, a ``NO_PACK`` kind (the balls are the countries);
``ball_evolution._parts`` dispatches here. The maze is built from the
recipe's seed (`build_maze`), so the sim and the painter draw the same one
and ``pick_seed`` can try seeds. A good run (`ok`) has 3+ lead changes and a
winner that wasn't leading at half time. The sound (``music_track``) is
ours, numpy only: a tick per wall bounce (each racer its own pitch), a
whoosh on a lead change, a bright hit + thump at the finish.
"""
from __future__ import annotations

import heapq
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from clipper import ball_duel as bd
from clipper.ball_evolution import ASSETS, FPS, H, SR, W

END_HOLD = 1.8
SECONDS_CAP = 70.0
# ---- the maze -----------------------------------------------------------
COLS, ROWS, CELL = 7, 9, 136
MX0 = (W - COLS * CELL) // 2                      # 64
MY0 = 372
MX1, MY1 = MX0 + COLS * CELL, MY0 + ROWS * CELL   # 1016, 1596
WALL_T = 14                                       # wall thickness
GOAL_CELL = (3, 4)
GOAL = (MX0 + (GOAL_CELL[0] + 0.5) * CELL, MY0 + (GOAL_CELL[1] + 0.5) * CELL)
GOAL_R = 46
START_CELLS = [(2, 0), (3, 0), (4, 0)]            # one open room at the top
PATH_MIN, PATH_MAX = 20, 30                       # cells on the shortest way start -> goal
BRAID = 0.16                                      # share of the spare walls taken out (loops)
GRID = 12                                         # the distance field's step (px)
LOOKAHEAD = 8                                     # grid steps ahead a racer aims
# ---- the racers ---------------------------------------------------------
R = 30
SPEED = 210.0
TURN = 3.6                                        # rad/s a racer turns toward where it wants to go
WANDER_TAU = 1.4                                  # s: how long a wrong idea lasts
WANDER_SD = 0.6                                   # rad: how wrong it gets (stationary sd)
START_T = 0.5                                     # the race starts here
LEAD_MARGIN = 26.0                                # px ahead to take the lead ...
LEAD_HOLD = 0.3                                   # ... for this long
MIN_CHANGES = 3
TRAIL_FADE = 0.95                                 # per frame
MAX_QUIET = 13.0                                  # s: never longer than this without a lead change
WRONG_TURN = 0.5                                  # chance a racer takes a wrong branch at a junction
WRONG_DEPTH = (1, 3)                              # how far down it goes before it turns back
DETOUR_MAX = 4.0                                  # s: a wrong turn never lasts longer
SIDESTEP = 0.5                                    # rad: two racers that meet both keep right
BOUNCE_AIM = 0.6                                  # a wall bounce turns this share toward the way on
PALETTES = [((80, 220, 255), (210, 248, 255)), ((255, 90, 200), (255, 215, 245)),
            ((120, 255, 150), (220, 255, 228)), ((255, 170, 70), (255, 232, 200)),
            ((170, 130, 255), (232, 222, 255))]
HOOKS = ["Pick one before it starts", "Which country reaches the trophy first?", "{n} countries, 1 maze, 1 trophy",
         "First one to the trophy wins", "Pick a country. Watch it race", "Who finds the way out first?",
         "{n} countries race through a maze", "Your country vs the maze"]


def _pick_cast(rng: random.Random, n: int) -> list:
    """n country codes, no two whose colours look alike (their trails and
    the standings would blur together); often a football rivalry in it."""
    codes: list = []
    if rng.random() < 0.6:
        a, b = rng.choice(bd.RIVALRIES)
        codes = [a] if bd._close(bd.COUNTRIES[a][1], bd.COUNTRIES[b][1]) else [a, b]
    pool = sorted(bd.COUNTRIES)
    rng.shuffle(pool)
    for c in pool:
        if len(codes) >= n:
            break
        if c not in codes and not any(bd._close(bd.COUNTRIES[c][1], bd.COUNTRIES[o][1]) for o in codes):
            codes.append(c)
    rng.shuffle(codes)
    return codes


def mazerace_recipe(recipe: dict) -> dict:
    rng = random.Random(f"mazerace-recipe-{recipe['seed']}")
    n = rng.choice((4, 5, 5))     # more than 5 rarely have colours far enough apart
    codes = _pick_cast(rng, n)
    recipe.update(look="flags", players=[bd._country(c) for c in codes], bed="off",
                  hook=rng.choice(HOOKS).format(n=len(codes)), ending="WINS!",
                  maze_colour=rng.randrange(len(PALETTES)))
    return recipe


# ---- the maze, from the seed ----------------------------------------------

def _neighbours(c, r):
    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= c + dc < COLS and 0 <= r + dr < ROWS:
            yield c + dc, r + dr


def _cell_path(open_: set) -> int:
    """Cells on the shortest way from the start room to the goal."""
    dist = {s: 0 for s in START_CELLS}
    q = list(START_CELLS)
    while q:
        cur = q.pop(0)
        if cur == GOAL_CELL:
            return dist[cur]
        for nb in _neighbours(*cur):
            if nb not in dist and frozenset((cur, nb)) in open_:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return 999


def build_maze(seed) -> dict:
    """The walls of this seed's maze: a random spanning tree with a few
    extra openings (loops, so there's more than one way), redrawn until the
    way from the start room to the trophy is a good length."""
    rng = random.Random(f"mazerace-maze-{seed}")
    for _ in range(400):
        open_: set = set()
        seen = {GOAL_CELL}
        stack = [GOAL_CELL]
        while stack:
            cur = stack[-1]
            nbs = [nb for nb in _neighbours(*cur) if nb not in seen]
            if not nbs:
                stack.pop()
                continue
            nb = rng.choice(nbs)
            open_.add(frozenset((cur, nb)))
            seen.add(nb)
            stack.append(nb)
        for a, b in zip(START_CELLS, START_CELLS[1:]):
            open_.add(frozenset((a, b)))
        spare = []
        for r in range(ROWS):
            for c in range(COLS):
                for nb in ((c + 1, r), (c, r + 1)):
                    if nb[0] < COLS and nb[1] < ROWS and frozenset(((c, r), nb)) not in open_:
                        spare.append(frozenset(((c, r), nb)))
        for e in spare:
            if GOAL_CELL in e:
                continue
            if rng.random() < BRAID:
                open_.add(e)
        goal_open = sum(1 for nb in _neighbours(*GOAL_CELL) if frozenset((GOAL_CELL, nb)) in open_)
        if PATH_MIN <= _cell_path(open_) <= PATH_MAX and goal_open <= 2:
            break
    segs = []
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = MX0 + c * CELL, MY0 + r * CELL
            if c + 1 < COLS and frozenset(((c, r), (c + 1, r))) not in open_:
                segs.append((x0 + CELL, y0, x0 + CELL, y0 + CELL))
            if r + 1 < ROWS and frozenset(((c, r), (c, r + 1))) not in open_:
                segs.append((x0, y0 + CELL, x0 + CELL, y0 + CELL))
    segs += [(MX0, MY0, MX1, MY0), (MX0, MY1, MX1, MY1), (MX0, MY0, MX0, MY1), (MX1, MY0, MX1, MY1)]
    links = {(c, r): [nb for nb in _neighbours(c, r) if frozenset(((c, r), nb)) in open_]
             for r in range(ROWS) for c in range(COLS)}
    return {"segs": segs, "path": _cell_path(open_), "links": links}


def _cell_bfs(links: dict, src, banned=None, depth=None) -> dict:
    """Steps from ``src`` to every cell (not through ``banned``)."""
    dist = {src: 0}
    q = [src]
    while q:
        cur = q.pop(0)
        if depth is not None and dist[cur] >= depth:
            continue
        for nb in links[cur]:
            if nb not in dist and nb != banned:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def _near_segments(segs: list) -> dict:
    """For each maze cell, the walls a ball in it can touch (its own and
    its neighbours' edges)."""
    near = {}
    for r in range(ROWS):
        for c in range(COLS):
            x0, y0 = MX0 + (c - 1) * CELL - 1, MY0 + (r - 1) * CELL - 1
            x1, y1 = MX0 + (c + 2) * CELL + 1, MY0 + (r + 2) * CELL + 1
            near[(c, r)] = [s for s in segs if max(s[0], s[2]) >= x0 and min(s[0], s[2]) <= x1
                            and max(s[1], s[3]) >= y0 and min(s[1], s[3]) <= y1]
    return near


def distance_field(segs: list):
    """Distance along the maze to the trophy on a GRID px grid (for a ball's
    centre: walls padded by its radius), and for each point the next one on
    the shortest way there (``hop``)."""
    nx, ny = (MX1 - MX0) // GRID, (MY1 - MY0) // GRID
    xs = MX0 + (np.arange(nx) + 0.5) * GRID
    ys = MY0 + (np.arange(ny) + 0.5) * GRID
    px, py = np.meshgrid(xs, ys)                    # [ny, nx]
    clear = np.full(px.shape, 1e9)
    for x0, y0, x1, y1 in segs:
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy or 1.0
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0, 1)
        clear = np.minimum(clear, np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))
    free = clear > R + WALL_T / 2 - 2
    dist = np.full(px.shape, np.inf)
    hop = np.full(px.shape + (2,), -1, dtype=np.int32)     # the next grid point on the way to the trophy
    heap = []
    goal = np.hypot(px - GOAL[0], py - GOAL[1]) <= GOAL_R + R
    for j, i in zip(*np.nonzero(goal & free)):
        dist[j, i] = 0.0
        heap.append((0.0, int(j), int(i)))
    heapq.heapify(heap)
    steps = [(dj, di, GRID * math.hypot(dj, di)) for dj in (-1, 0, 1) for di in (-1, 0, 1) if dj or di]
    while heap:
        d0, j, i = heapq.heappop(heap)
        if d0 > dist[j, i]:
            continue
        for dj, di, w in steps:
            jj, ii = j + dj, i + di
            if 0 <= jj < ny and 0 <= ii < nx and free[jj, ii] and d0 + w < dist[jj, ii]:
                if dj and di and not (free[j + dj, i] and free[j, i + di]):
                    continue                        # no cutting a wall's corner
                dist[jj, ii] = d0 + w
                hop[jj, ii] = (j, i)
                heapq.heappush(heap, (d0 + w, jj, ii))
    reach = np.isfinite(dist)
    worst = float(np.max(dist[reach])) if reach.any() else 1.0
    dist = np.where(reach, dist, worst + 200.0)
    return dist, hop, reach


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MazeRaceSim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.rng = random.Random(f"mazerace-{recipe['seed']}")
        self.t = 0.0
        maze = build_maze(recipe["seed"])
        self.segs = maze["segs"]
        self.path_cells = maze["path"]
        self.near = _near_segments(self.segs)
        self.links = maze["links"]
        self.gdist = _cell_bfs(self.links, GOAL_CELL)
        self.dist, self.hop, self.reach = distance_field(self.segs)
        players = recipe.get("players") or [bd._country(c) for c in sorted(bd.COUNTRIES)[:5]]
        n = len(players)
        x0 = MX0 + START_CELLS[0][0] * CELL + R + 10
        x1 = MX0 + (START_CELLS[-1][0] + 1) * CELL - R - 10
        slots = [x0 + (x1 - x0) * k / max(1, n - 1) for k in range(n)]
        self.balls = []
        for i, p in enumerate(players):
            h = math.pi / 2 + self.rng.uniform(-0.6, 0.6)
            self.balls.append({"i": i, "name": p[0], "col": tuple(p[2]), "flag": p[3] if len(p) > 3 else None,
                               "x": slots[i], "y": MY0 + CELL / 2, "h": h, "wander": self.rng.gauss(0, 0.5),
                               "px": slots[i], "py": MY0 + CELL / 2, "d": 0.0, "last_tick": -1.0,
                               "cell": None, "prev": None, "detour": None, "tdist": None, "since": 0.0})
        for b in self.balls:
            b["d"] = self.d_at(b["x"], b["y"])
        self.start_d = max(b["d"] for b in self.balls)
        self.n = n
        self.chain = [(None, "start"), (None, "finish")]   # (render's info: start, finish)
        self.events: list = []         # the shared mixer's ball sounds: none (music_track has them)
        self.sfx: list = []            # (t, kind, value)
        self.unlocked_at = {0: 0.0}
        self.best = 0                  # lead changes so far
        self.spawned = n               # (render's info: how many racers)
        self.order = list(range(n))    # standings, leader first
        self.leader = None
        self._cand, self._cand_since = None, 0.0
        self.changes: list = []        # (t, new leader)
        self.leads: list = []          # (t, leader) every time it changes, incl. the first
        self.winner = None
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0
        self.stuck = 0
        self.stuck_log: list = []
        self._still = {b["i"]: (0.0, b["x"], b["y"]) for b in self.balls}

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    # -- the field --------------------------------------------------------
    def _idx(self, x, y):
        i = int((x - MX0) // GRID)
        j = int((y - MY0) // GRID)
        ny, nx = self.dist.shape
        return min(max(j, 0), ny - 1), min(max(i, 0), nx - 1)

    def d_at(self, x, y) -> float:
        j, i = self._idx(x, y)
        if self.reach[j, i]:
            return float(self.dist[j, i])
        ny, nx = self.dist.shape
        best = None
        for jj in range(max(0, j - 2), min(ny, j + 3)):
            for ii in range(max(0, i - 2), min(nx, i + 3)):
                if self.reach[jj, ii] and (best is None or self.dist[jj, ii] < best):
                    best = float(self.dist[jj, ii])
        return best if best is not None else float(self.dist[j, i])

    def flow_at(self, x, y):
        """Toward a point a little way ahead on the shortest way to the
        trophy (following the local slope alone pins a ball against a
        wall's end: around a corner every slope points into it)."""
        j, i = self._idx(x, y)
        if not self.reach[j, i]:
            best = None
            ny, nx = self.dist.shape
            for jj in range(max(0, j - 2), min(ny, j + 3)):
                for ii in range(max(0, i - 2), min(nx, i + 3)):
                    if self.reach[jj, ii] and (best is None or self.dist[jj, ii] < self.dist[best]):
                        best = (jj, ii)
            if best is None:
                return None
            j, i = best
        for _ in range(LOOKAHEAD):
            nj, ni = self.hop[j, i]
            if nj < 0:
                break
            j, i = int(nj), int(ni)
        tx, ty = MX0 + (i + 0.5) * GRID, MY0 + (j + 0.5) * GRID
        if abs(tx - x) < 1 and abs(ty - y) < 1:
            tx, ty = GOAL
        return math.atan2(ty - y, tx - x)

    # -- stepping ---------------------------------------------------------
    def step(self):
        dt = 1 / FPS
        self.t += dt
        for b in self.balls:
            b["px"], b["py"] = b["x"], b["y"]
        if self.done_at is not None:
            w = self.winner
            w["x"] += (GOAL[0] - w["x"]) * 0.12
            w["y"] += (GOAL[1] - w["y"]) * 0.12
            return
        if self.t < START_T:
            return
        for _ in range(2):
            if self._move(dt / 2):
                break
        for b in self.balls:
            b["d"] = self.d_at(b["x"], b["y"])
        if self.done_at is None:
            self._standings()
            self._check_stuck()
        if self.t > SECONDS_CAP - END_HOLD - 0.1 and self.done_at is None:
            self.done_at = self.t
            self.winner = min(self.balls, key=lambda b: b["d"])
            self.ok = False

    def _move(self, dt) -> bool:
        k = math.exp(-dt / WANDER_TAU)
        sd = WANDER_SD * math.sqrt(1 - k * k)
        for b in self.balls:
            b["wander"] = b["wander"] * k + self.rng.gauss(0, sd)
            want = self._detour_dir(b) if b["detour"] else self.flow_at(b["x"], b["y"])
            if want is not None:
                diff = _wrap_angle(want + b["wander"] - b["h"])
                b["h"] += max(-TURN * dt, min(TURN * dt, diff))
            b["h"] += self.rng.uniform(-0.02, 0.02)
            b["x"] += SPEED * math.cos(b["h"]) * dt
            b["y"] += SPEED * math.sin(b["h"]) * dt
        # racers bump each other
        for a_i, a in enumerate(self.balls):
            for c in self.balls[a_i + 1:]:
                dx, dy = c["x"] - a["x"], c["y"] - a["y"]
                dd = math.hypot(dx, dy)
                if 0 < dd < 2 * R:
                    nx, ny = dx / dd, dy / dd
                    avx, avy = math.cos(a["h"]), math.sin(a["h"])
                    cvx, cvy = math.cos(c["h"]), math.sin(c["h"])
                    rel = (cvx - avx) * nx + (cvy - avy) * ny
                    if rel < 0:
                        avx, avy = avx + rel * nx, avy + rel * ny
                        cvx, cvy = cvx - rel * nx, cvy - rel * ny
                        # both keep right a little, so two meeting in a corridor
                        # slip past each other instead of bouncing forever
                        a["h"] = (math.atan2(avy, avx) if (avx or avy) else a["h"] + math.pi) + SIDESTEP
                        c["h"] = (math.atan2(cvy, cvx) if (cvx or cvy) else c["h"] + math.pi) + SIDESTEP
                        self.sfx.append((self.t, "bump", 0))
                        a["detour"] = c["detour"] = None      # a racer that meets another gives up its wrong turn
                    over = 2 * R - dd
                    a["x"] -= nx * over / 2
                    a["y"] -= ny * over / 2
                    c["x"] += nx * over / 2
                    c["y"] += ny * over / 2
        # walls (after the pushes, so nobody ends up inside one)
        for b in self.balls:
            self._walls(b)
            self._junction(b)
            if math.hypot(b["x"] - GOAL[0], b["y"] - GOAL[1]) < GOAL_R + R * 0.4:
                self._finish(b)
                return True
        return False

    @staticmethod
    def cell_of(x, y):
        return (min(COLS - 1, max(0, int((x - MX0) // CELL))), min(ROWS - 1, max(0, int((y - MY0) // CELL))))

    def _junction(self, b):
        """A wrong turn now and then: at a junction, down a branch that
        doesn't lead to the trophy, a cell or a few deep, then back."""
        cell = self.cell_of(b["x"], b["y"])
        if cell == b["cell"]:
            return
        b["prev"], b["cell"] = b["cell"], cell
        if b["detour"] is not None or cell == GOAL_CELL or len(self.links[cell]) < 3:
            return
        g = self.gdist.get(cell, 99)
        wrong = [nb for nb in self.links[cell] if self.gdist.get(nb, 99) >= g and nb != b["prev"]]
        if wrong and self.rng.random() < WRONG_TURN:
            nb = self.rng.choice(sorted(wrong))
            depth = self.rng.randint(*WRONG_DEPTH)
            sub = _cell_bfs(self.links, nb, banned=cell, depth=depth - 1)
            busy = {o["cell"] for o in self.balls if o is not b} | {o["detour"] for o in self.balls if o is not b}
            if busy & set(sub):
                return                              # someone's down there: no jams in a dead end
            target = max(sorted(sub), key=lambda c: sub[c])
            b["detour"], b["tdist"], b["since"] = target, _cell_bfs(self.links, target), self.t

    def _detour_dir(self, b):
        cell = self.cell_of(b["x"], b["y"])
        tgt = b["detour"]
        cx, cy = MX0 + (tgt[0] + 0.5) * CELL, MY0 + (tgt[1] + 0.5) * CELL
        if (cell == tgt and math.hypot(b["x"] - cx, b["y"] - cy) < CELL * 0.3) or self.t - b["since"] > DETOUR_MAX:
            b["detour"] = None
            return self.flow_at(b["x"], b["y"])
        if cell != tgt:
            td = b["tdist"]
            nxt = min(self.links[cell], key=lambda c: (td.get(c, 99), c))
            cx, cy = MX0 + (nxt[0] + 0.5) * CELL, MY0 + (nxt[1] + 0.5) * CELL
        return math.atan2(cy - b["y"], cx - b["x"])

    def _walls(self, b):
        c = min(COLS - 1, max(0, int((b["x"] - MX0) // CELL)))
        r = min(ROWS - 1, max(0, int((b["y"] - MY0) // CELL)))
        lim = R + WALL_T / 2
        for x0, y0, x1, y1 in self.near[(c, r)]:
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = max(0.0, min(1.0, ((b["x"] - x0) * dx + (b["y"] - y0) * dy) / L2))
            qx, qy = x0 + t * dx, y0 + t * dy
            ex, ey = b["x"] - qx, b["y"] - qy
            d = math.hypot(ex, ey)
            if d >= lim:
                continue
            if d < 1e-6:
                ex, ey, d = -math.sin(b["h"]), math.cos(b["h"]), 1.0
            nx, ny = ex / d, ey / d
            b["x"], b["y"] = qx + nx * lim, qy + ny * lim
            vx, vy = math.cos(b["h"]), math.sin(b["h"])
            vn = vx * nx + vy * ny
            if vn < 0:
                vx, vy = vx - 2 * vn * nx, vy - 2 * vn * ny
                h = math.atan2(vy, vx) + self.rng.uniform(-0.15, 0.15)
                want = self._detour_dir(b) if b["detour"] else self.flow_at(b["x"], b["y"])
                if want is not None and math.cos(want) * nx + math.sin(want) * ny > 0:
                    # off the wall partly the way it wants to go: in a corridor
                    # a plain mirror bounce zigzags across it and gets nowhere
                    h += BOUNCE_AIM * _wrap_angle(want - h)
                b["h"] = h
                if self.t - b["last_tick"] > 0.07:
                    b["last_tick"] = self.t
                    self.sfx.append((self.t, "wall", b["i"]))

    def _standings(self):
        self.order = sorted(range(self.n), key=lambda i: (self.balls[i]["d"], i))
        top = self.order[0]
        if self.leader is None:
            self.leader = top
            self.leads.append((self.t, top))
            return
        if top == self.leader:
            self._cand = None
            return
        if self.balls[top]["d"] < self.balls[self.leader]["d"] - LEAD_MARGIN:
            if self._cand != top:
                self._cand, self._cand_since = top, self.t
            elif self.t - self._cand_since >= LEAD_HOLD:
                self.leader = top
                self._cand = None
                self.changes.append((self.t, top))
                self.leads.append((self.t, top))
                self.best = len(self.changes)
                self.sfx.append((self.t, "lead", top))
        else:
            self._cand = None

    def leader_at(self, t: float):
        who = None
        for lt, i in self.leads:
            if lt <= t:
                who = i
        return who

    def _check_stuck(self):
        for b in self.balls:
            t0, x0, y0 = self._still[b["i"]]
            if math.hypot(b["x"] - x0, b["y"] - y0) > 60:
                self._still[b["i"]] = (self.t, b["x"], b["y"])
            elif self.t - t0 > 3.0:
                self.stuck += 1
                self.stuck_log.append((round(self.t, 1), b['i'], round(b['x']), round(b['y']), b['detour'], b['cell']))
                self._still[b["i"]] = (self.t, b["x"], b["y"])

    def _finish(self, b):
        self.winner = b
        self.done_at = self.t
        self.unlocked_at[1] = self.t
        self.order = [b["i"]] + [i for i in sorted(range(self.n), key=lambda i: self.balls[i]["d"]) if i != b["i"]]
        self.sfx.append((self.t, "finish", 0))
        if self.leader != b["i"]:
            # a last-moment pass counts as a lead change too
            self.changes.append((self.t, b["i"]))
            self.leads.append((self.t, b["i"]))
            self.best = len(self.changes)
            self.leader = b["i"]
        times = [START_T] + [t for t, _ in self.changes] + [self.t]
        self.max_gap = round(max(y - x for x, y in zip(times, times[1:])), 2)
        half = self.leader_at((START_T + self.t) / 2)
        self.ok = (len(self.changes) >= MIN_CHANGES and half != b["i"] and self.stuck == 0
                   and self.max_gap <= MAX_QUIET)

    # -- sound ------------------------------------------------------------
    def music_track(self, n: int) -> np.ndarray:
        """Our own sounds, numpy only: ticks, whooshes, the finish."""
        mix = np.zeros(n, np.float32)
        g = np.random.default_rng(int(self.recipe.get("seed") or 0))
        pitch = [523.0, 659.0, 784.0, 988.0, 1175.0, 1397.0]
        last_any = -1.0
        events = list(self.sfx)
        events.append((START_T, "go", 0))
        for et, kind, v in events:
            a0 = int(et * SR)
            if a0 >= n:
                continue
            if kind == "wall":
                if et - last_any < 0.03:
                    continue
                last_any = et
                f = pitch[int(v) % len(pitch)]
                ts = np.arange(int(SR * 0.08)) / SR
                w = (np.sin(2 * np.pi * f * ts) * 0.55 + np.sin(2 * np.pi * 2 * f * ts) * 0.12
                     + g.normal(0, 1, len(ts)) * 0.12) * np.exp(-ts * 60) * 0.32
            elif kind == "bump":
                ts = np.arange(int(SR * 0.07)) / SR
                noise = np.convolve(g.normal(0, 1, len(ts)), np.ones(12) / 12, "same")
                w = (noise * 1.6 + np.sin(2 * np.pi * 180 * ts) * 0.4) * np.exp(-ts * 70) * 0.3
            elif kind == "lead":
                d = 0.55
                ts = np.arange(int(SR * d)) / SR
                noise = g.normal(0, 1, len(ts))
                # a whoosh: noise through a sweeping smoother, swelling and fading
                k = 6 + (40 * (1 - ts / d)).astype(int)
                cs = np.cumsum(np.concatenate([[0.0], noise]))
                idx = np.arange(len(ts))
                lo = np.maximum(0, idx - k)
                sm = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
                w = sm * np.sin(np.pi * ts / d) ** 2 * 1.4
                w += np.sin(2 * np.pi * (500 + 500 * ts / d) * ts) * np.exp(-ts * 8) * 0.12
            elif kind == "go":
                ts = np.arange(int(SR * 0.25)) / SR
                w = np.sin(2 * np.pi * 880 * ts) * np.exp(-ts * 12) * 0.35
            else:                                         # finish
                d = 1.4
                ts = np.arange(int(SR * d)) / SR
                hit = g.normal(0, 1, len(ts)) * np.exp(-ts * 18)
                hit = hit - np.convolve(hit, np.ones(6) / 6, "same")   # bright: the highs of the noise
                f = 120 * np.exp(-ts * 5) + 45
                thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-ts * 4)
                bell = sum(np.sin(2 * np.pi * fr * ts) * a for fr, a in ((1046.5, 0.25), (1318.5, 0.18),
                                                                          (1568.0, 0.14)))
                w = hit * 1.3 + thump * 0.9 + bell * np.exp(-ts * 3) * 0.6
            seg = mix[a0:a0 + len(w)]
            seg += w[:len(seg)].astype(np.float32)
        peak = float(np.abs(mix).max()) or 1.0
        return mix / max(1.0, peak / 0.9)


# ---- drawing --------------------------------------------------------------

def _emoji(code: str, size: int) -> Image.Image:
    return Image.open(ASSETS / "emoji" / f"{code}.webp").convert("RGBA").resize((size, size), Image.LANCZOS)


def _ordinal(k: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


class MazeRacePainter:
    BG = (7, 9, 22)
    FLOOR = (13, 15, 36)

    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.players = recipe.get("players") or []
        maze = build_maze(recipe["seed"])
        neon, core = PALETTES[int(recipe.get("maze_colour") or 0) % len(PALETTES)]
        self.neon = neon
        bg = Image.new("RGBA", (W, H), self.BG + (255,))
        d = ImageDraw.Draw(bg)
        d.rounded_rectangle((MX0 - 8, MY0 - 8, MX1 + 8, MY1 + 8), radius=18, fill=self.FLOOR + (255,))
        # the start room: a faint chequered strip
        sx0 = MX0 + START_CELLS[0][0] * CELL
        sx1 = MX0 + (START_CELLS[-1][0] + 1) * CELL
        sq = 17
        for k, x in enumerate(range(sx0 + 8, sx1 - 8, sq)):
            for row in range(2):
                if (k + row) % 2 == 0:
                    d.rectangle((x, MY0 + 10 + row * sq, x + sq - 1, MY0 + 10 + row * sq + sq - 1),
                                fill=(40, 44, 70, 255))
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for x0, y0, x1, y1 in maze["segs"]:
            gd.line((x0, y0, x1, y1), fill=neon + (210,), width=WALL_T + 14)
        glow = glow.filter(ImageFilter.GaussianBlur(9))
        bg.alpha_composite(glow)
        d = ImageDraw.Draw(bg)
        for x0, y0, x1, y1 in maze["segs"]:
            d.line((x0, y0, x1, y1), fill=neon + (255,), width=WALL_T)
            for x, y in ((x0, y0), (x1, y1)):
                d.ellipse((x - WALL_T / 2, y - WALL_T / 2, x + WALL_T / 2, y + WALL_T / 2), fill=neon + (255,))
        for x0, y0, x1, y1 in maze["segs"]:
            d.line((x0, y0, x1, y1), fill=core + (255,), width=4)
        # the goal: a gold glow (pulses per frame) and the trophy
        self.goal_glow = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        ImageDraw.Draw(self.goal_glow).ellipse((150 - 70, 150 - 70, 150 + 70, 150 + 70), fill=(255, 200, 60, 200))
        self.goal_glow = self.goal_glow.filter(ImageFilter.GaussianBlur(28))
        ImageDraw.Draw(bg).ellipse((GOAL[0] - GOAL_R, GOAL[1] - GOAL_R, GOAL[0] + GOAL_R, GOAL[1] + GOAL_R),
                                   fill=(70, 52, 18, 255), outline=(255, 214, 90, 255), width=5)
        self.bg = bg
        self.trophy = _emoji("1f3c6", 76)
        self.trophy_big = _emoji("1f3c6", 64)
        self.trails = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.f_hook = bd._font(50)
        self.f_pick = bd._font(62)
        self.f_lead = bd._font(46)
        self.f_rank = bd._font(26)
        self.f_go = bd._font(96)
        self.f_win = bd._font(96)
        self.f_small = bd._font(34, "Inter-Bold.ttf")
        self.f_foot = bd._font(36)
        self.trophy_small = _emoji("1f3c6", 44)
        self.f_name = bd._font(28, "Inter-Bold.ttf")
        self.sprites: dict = {}
        self.slot_x: dict = {}

    def sprite(self, b, size, ring=(255, 255, 255)):
        key = (b["i"], size, ring)
        im = self.sprites.get(key)
        if im is None:
            s3 = size * 3
            im = Image.new("RGBA", (s3 + 12, s3 + 12), (0, 0, 0, 0))
            if b.get("flag"):
                im.alpha_composite(bd._flag(b["flag"], s3), (6, 6))
            else:
                ImageDraw.Draw(im).ellipse((6, 6, 6 + s3, 6 + s3), fill=b["col"] + (255,))
            ImageDraw.Draw(im).ellipse((6, 6, 6 + s3, 6 + s3), outline=ring + (255,), width=max(9, s3 // 16))
            im = self.sprites[key] = im.resize((size + 4, size + 4), Image.LANCZOS)
        return im

    def frame(self, sim: MazeRaceSim) -> Image.Image:
        im = self.bg.copy()
        # the trophy's glow pulses
        pulse = 0.75 + 0.25 * math.sin(sim.t * 5)
        g = self.goal_glow.copy()
        g.putalpha(g.getchannel("A").point(lambda v: int(v * pulse)))
        im.alpha_composite(g, (int(GOAL[0] - 150), int(GOAL[1] - 150)))
        if sim.done_at is None or sim.t - sim.done_at < 0.15:
            im.alpha_composite(self.trophy, (int(GOAL[0] - 38), int(GOAL[1] - 40)))
        # trails, each racer's colour, fading
        a = np.asarray(self.trails).copy()
        a[..., 3] = (a[..., 3] * TRAIL_FADE).astype(np.uint8)
        self.trails = Image.fromarray(a)
        td = ImageDraw.Draw(self.trails)
        if sim.done_at is None:
            for b in sim.balls:
                td.line([(b["px"], b["py"]), (b["x"], b["y"])], fill=b["col"] + (200,), width=18)
        im.alpha_composite(self.trails)
        lead = sim.leader
        for b in sorted(sim.balls, key=lambda b: b is sim.winner):
            size = 2 * R
            ring = (255, 214, 80) if b["i"] == lead and sim.t >= START_T else (255, 255, 255)
            if sim.winner is b:
                size = int(2 * R * (1 + min(1.2, (sim.t - sim.done_at) * 3)))
            sp = self.sprite(b, size, ring)
            im.alpha_composite(sp, (int(b["x"] - sp.width / 2), int(b["y"] - sp.height / 2)))
        d = ImageDraw.Draw(im)
        if START_T <= sim.t < START_T + 0.7:
            k = (sim.t - START_T) / 0.7
            a_ = int(255 * (1 - k))
            d.text((W / 2, MY0 + CELL * 1.5), "GO!", font=self.f_go, fill=(120, 255, 140, a_), anchor="mm",
                   stroke_width=8, stroke_fill=(0, 0, 0, a_))
        self.hud(im, d, sim)
        foot = "FIRST TO THE TROPHY WINS"
        fw = d.textlength(foot, font=self.f_foot)
        d.text(((W - fw - 52) / 2, 1660), foot, font=self.f_foot, fill=(150, 150, 175, 255), anchor="lm")
        im.alpha_composite(self.trophy_small, (int((W + fw - 52) / 2 + 12), 1660 - 22))
        if sim.done_at is not None:
            self.ending(im, sim)
        mark = (self.recipe.get("watermark") or "").strip()
        if mark:
            d = ImageDraw.Draw(im)
            d.text((W - 60 - d.textlength(mark, font=self.f_small), H - 70), mark, font=self.f_small,
                   fill=(255, 255, 255, 150))
        return im.convert("RGB")

    def _arrow(self, d, x, y, s, a):
        """A down arrow (the fonts have no 👇)."""
        d.rectangle((x - s * 0.16, y - s * 0.5, x + s * 0.16, y + s * 0.05), fill=(255, 222, 70, a),
                    outline=(0, 0, 0, a), width=4)
        d.polygon([(x - s * 0.45, y), (x + s * 0.45, y), (x, y + s * 0.5)], fill=(255, 222, 70, a),
                  outline=(0, 0, 0, a), width=4)

    def hud(self, im, d, sim):
        hook = self.recipe.get("hook") or HOOKS[0]
        f = self.f_hook
        while d.textlength(hook, font=f) > W - 80 and f.size > 30:
            f = bd._font(f.size - 4)
        d.text((W / 2, 62), hook, font=f, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=4, stroke_fill=self.BG + (255,))
        # the pick line for the first seconds, then the lead changes in the same row
        if sim.t < 3.0:
            a = int(255 * min(1.0, (3.0 - sim.t) / 0.4))
            left, right = "PICK ONE", "NO SWITCHING!"
            wl = d.textlength(left, font=self.f_pick)
            wr = d.textlength(right, font=self.f_pick)
            total = wl + 80 + wr
            x = (W - total) / 2
            d.text((x, 140), left, font=self.f_pick, fill=(255, 222, 70, a), anchor="lm",
                   stroke_width=6, stroke_fill=(0, 0, 0, a))
            self._arrow(d, x + wl + 35, 138, 64, a)
            d.text((x + wl + 80, 140), right, font=self.f_pick, fill=(255, 255, 255, a), anchor="lm",
                   stroke_width=6, stroke_fill=(0, 0, 0, a))
        elif sim.changes and sim.done_at is None:
            t, i = sim.changes[-1]
            age = sim.t - t
            if age < 1.3:
                b = sim.balls[i]
                a = int(255 * min(1.0, (1.3 - age) / 0.3))
                s = 0.7 + 0.3 * min(1.0, age / 0.12)
                text = f"{b['name'].upper()} TAKES THE LEAD!"
                fl = bd._font(int(self.f_lead.size * s))
                while d.textlength(text, font=fl) > W - 140:
                    fl = bd._font(fl.size - 3)
                tw = d.textlength(text, font=fl)
                fs = int(56 * s)
                x0 = (W - tw - fs - 16) / 2
                sp = self.sprite(b, fs).copy()
                sp.putalpha(sp.getchannel("A").point(lambda v: v * a // 255))
                im.alpha_composite(sp, (int(x0), int(140 - sp.height / 2)))
                d.text((x0 + fs + 16, 140), text, font=fl, fill=(255, 255, 255, a), anchor="lm",
                       stroke_width=5, stroke_fill=(0, 0, 0, a))
        # the standings strip: leader on the left, sliding as they pass
        n = sim.n
        slot = (W - 60) / n
        fsize = 28
        while fsize > 18 and max(d.textlength(b["name"], font=bd._font(fsize, "Inter-Bold.ttf"))
                                 for b in sim.balls) > slot - 10:
            fsize -= 2
        fname = bd._font(fsize, "Inter-Bold.ttf")
        for rank, i in enumerate(sim.order):
            target = 30 + slot * (rank + 0.5)
            x = self.slot_x.get(i)
            x = target if x is None else x + (target - x) * 0.18
            self.slot_x[i] = x
            b = sim.balls[i]
            gold = rank == 0 and sim.t >= START_T
            sp = self.sprite(b, 84, (255, 214, 80) if gold else (255, 255, 255))
            im.alpha_composite(sp, (int(x - sp.width / 2), int(242 - sp.height / 2)))
            d.text((x, 306), b["name"], font=fname, fill=(255, 255, 255, 255), anchor="mm",
                   stroke_width=3, stroke_fill=(0, 0, 0, 255))
            if sim.t >= START_T:
                bx, by = x - 40, 204
                d.ellipse((bx - 21, by - 21, bx + 21, by + 21), fill=(255, 214, 80, 255) if gold else (30, 34, 60, 255),
                          outline=(0, 0, 0, 255), width=3)
                d.text((bx, by), str(rank + 1), font=self.f_rank, fill=(0, 0, 0, 255) if gold else (255, 255, 255, 255),
                       anchor="mm")

    def ending(self, im, sim):
        age = sim.t - sim.done_at
        if age < 0.3:                                    # a white flash on the finish
            fl = Image.new("RGBA", (W, H), (255, 255, 255, int(170 * (1 - age / 0.3))))
            im.alpha_composite(fl)
        if age < 0.2:
            return
        w = sim.winner
        k = min(1.0, (age - 0.2) / 0.25)
        text = f"{w['name'].upper()} {self.recipe.get('ending') or 'WINS!'}"
        f = self.f_win
        dd = ImageDraw.Draw(im)
        while dd.textlength(text, font=f) > W - 120:
            f = bd._font(f.size - 4)
        s = 0.6 + 0.4 * k
        txt = Image.new("RGBA", (W, 240), (0, 0, 0, 0))
        td = ImageDraw.Draw(txt)
        td.rounded_rectangle((40, 30, W - 40, 210), radius=30, fill=(0, 0, 0, 170))
        td.text((W / 2, 120), text, font=f, fill=(255, 255, 255, 255), anchor="mm",
                stroke_width=10, stroke_fill=bd._stroke(w["col"]) + (255,))
        txt = txt.resize((int(W * s), int(240 * s)), Image.LANCZOS)
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(GOAL[1] + 230 - txt.height / 2)))
        tr = self.trophy_big
        im.alpha_composite(tr, (int(GOAL[0] - 32), int(GOAL[1] - R * (1 + min(1.2, age * 3)) - 70)))


# ---- publishing ------------------------------------------------------------

def flag_emoji(code: str) -> str:
    if code.startswith("gb-"):
        tag = "gb" + code[3:]
        return "\U0001F3F4" + "".join(chr(0xE0000 + ord(ch)) for ch in tag) + "\U000E007F"
    return "".join(chr(0x1F1E6 + ord(ch) - ord("a")) for ch in code.lower()[:2])


def publish_text(recipe: dict) -> dict:
    players = recipe.get("players") or []
    flags = "".join(flag_emoji(p[3]) for p in players if len(p) > 3 and p[3])
    title = f"{recipe.get('hook') or HOOKS[0]} 🏁 {flags}".strip()
    if len(title) > 100:
        title = f"{recipe.get('hook') or HOOKS[0]} 🏁"
    names = ", ".join(p[0] for p in players)
    desc = (f"{len(players)} countries race through a maze to the trophy: {names}. Same speed for all of them; "
            f"the walls and the bumps decide. Pick one before it starts. No switching!\n\n"
            f"Which one did you pick? 👇\n\n#satisfying #marblerace #countryballs #bouncingball\n\n"
            f"Flags: flag-icons (MIT)")
    return {"title": title[:100], "description": desc}


GOOD_END = (20.0, 36.0)     # suggested ball_evolution.good_end for "mazerace"
