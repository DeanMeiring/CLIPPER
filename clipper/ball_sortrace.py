"""Sorting race: "Who unscrambles it first?" (Dean, 17 Nov).

From a reel Dean sent (12K likes, 625 saves: quick sort vs merge sort on
65,536 pixels of Starry Night). One picture is cut into 128x128 = 16,384
pixels, scrambled the same way twice, and two sorting algorithms put their
copy back together side by side, with a clock each; the first one done wins.

The race is fair: every comparison and every move counts as one step, and
both run at the same steps per frame (`rate`, set so the slower one takes
about `S_TARGET` s). Only close pairs race (`PAIRS`: quick vs merge, heap vs
comb, each within a few percent on random data), and ``pick_seed`` keeps a
scramble whose finish is close but not a tie (`S_GAP_MIN`..`S_GAP_MAX`),
so who wins changes from video to video.

Pictures (``picture_for``): famous public-domain paintings and free animal
photos from Openverse (CC0 / public domain only, the same source as the
photo packs), saved once under the store and credited in the description;
the bundled CC0 cat and coffee photos (``clipper/assets/pictures``) when
nothing can be downloaded. The sound is ours: a short tone per frame for
each side, pitched by the pixel just moved (the "sound of sorting"), and a
rising sweep when one finishes. ``format: "sortrace"``, a ``NO_PACK`` kind.
"""
from __future__ import annotations

import io
import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from clipper.ball_evolution import ASSETS, FPS, H, SR, W, _font

END_HOLD = 2.0
SECONDS_CAP = 60.0
S_N = 128                      # 128 x 128 pixels
S_TARGET = 28.0                # the slower one finishes about here
S_LOOK = 0.3                   # a short look at the scrambled picture first
S_GAP_MIN, S_GAP_MAX = 0.15, 2.5
PANEL = 700
NAMES = {"quick": "QUICK SORT", "merge": "MERGE SORT", "heap": "HEAP SORT", "comb": "COMB SORT"}
COLOURS = {"quick": (247, 160, 70), "merge": (110, 200, 245), "heap": (170, 225, 110), "comb": (240, 130, 190)}
PAIRS = [("quick", "merge"), ("heap", "comb")]
TITLES = ["WHO UNSCRAMBLES IT FIRST?", "WHO FIXES IT FIRST?", "WHICH SORT WINS?"]
HOOKS = ["{a} vs {b}: who unscrambles it first?", "Who puts {pic} back together first?",
         "{a} vs {b} on {pic}", "16,384 pixels of {pic}. Which sort wins?", "Two sorts, one scrambled {pic}"]
FALLBACKS = [{"name": "a cat", "path": "cat.jpg", "credit": "Picture: \"Chelsea\" by Stefan van der Walt, CC0"},
             {"name": "a cup of coffee", "path": "coffee.jpg", "credit": "Picture: coffee by Rachel Michetti, CC0"}]
# (what the video calls it, Openverse search, words its title must have; None = any)
PICTURES = [
    ("Starry Night", "starry night van gogh", ("starry",)),
    ("The Great Wave", "great wave off kanagawa hokusai", ("wave",)),
    ("the Mona Lisa", "mona lisa leonardo da vinci", ("mona", "lisa")),
    ("Girl with a Pearl Earring", "girl with a pearl earring vermeer", ("pearl",)),
    ("Van Gogh's Sunflowers", "sunflowers van gogh painting", ("sunflower",)),
    ("Monet's Water Lilies", "water lilies monet", ("lil",)),
    ("The Scream", "the scream edvard munch", ("scream",)),
    ("The Kiss", "the kiss gustav klimt", ("kiss",)),
    ("Impression, Sunrise", "impression sunrise monet", ("impression",)),
    ("Wheat Field with Cypresses", "wheat field with cypresses van gogh", ("cypress",)),
    ("a fox", "red fox portrait", None), ("a tiger", "tiger face close up", None),
    ("a puppy", "puppy portrait", None), ("a parrot", "colorful parrot", None),
    ("a kitten", "kitten portrait", None), ("an owl", "owl portrait", None),
]


# ---- the algorithms: each yields None for a comparison, (index, value) for a move ----

def _quick(a):
    st = [(0, len(a) - 1)]
    while st:
        lo, hi = st.pop()
        if lo >= hi:
            continue
        p = a[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while True:
                yield None
                if a[i] < p:
                    i += 1
                else:
                    break
            while True:
                yield None
                if a[j] > p:
                    j -= 1
                else:
                    break
            if i <= j:
                a[i], a[j] = a[j], a[i]
                yield i, a[i]
                yield j, a[j]
                i += 1
                j -= 1
        st.append((i, hi))
        st.append((lo, j))


def _merge(a):
    n, w = len(a), 1
    while w < n:
        for lo in range(0, n, 2 * w):
            mid, hi = min(lo + w, n), min(lo + 2 * w, n)
            left, right = a[lo:mid], a[mid:hi]
            i = j = 0
            k = lo
            while i < len(left) and j < len(right):
                yield None
                if left[i] <= right[j]:
                    a[k] = left[i]
                    i += 1
                else:
                    a[k] = right[j]
                    j += 1
                yield k, a[k]
                k += 1
            for v in left[i:] + right[j:]:
                a[k] = v
                yield k, v
                k += 1
        w *= 2


def _heap(a):
    n = len(a)

    def sift(s, e):
        r = s
        while 2 * r + 1 <= e:
            c = 2 * r + 1
            if c + 1 <= e:
                yield None
                if a[c] < a[c + 1]:
                    c += 1
            yield None
            if a[r] < a[c]:
                a[r], a[c] = a[c], a[r]
                yield r, a[r]
                yield c, a[c]
                r = c
            else:
                return
    for s in range(n // 2 - 1, -1, -1):
        yield from sift(s, n - 1)
    for e in range(n - 1, 0, -1):
        a[0], a[e] = a[e], a[0]
        yield 0, a[0]
        yield e, a[e]
        yield from sift(0, e - 1)


def _comb(a):
    n, g, done = len(a), len(a), False
    while not done:
        g = int(g / 1.3)
        if g <= 1:
            g, done = 1, True
        for i in range(n - g):
            yield None
            if a[i] > a[i + g]:
                a[i], a[i + g] = a[i + g], a[i]
                yield i, a[i]
                yield i + g, a[i + g]
                done = False


ALGOS = {"quick": _quick, "merge": _merge, "heap": _heap, "comb": _comb}


def sortrace_recipe(recipe: dict) -> dict:
    rng = random.Random(f"sortrace-recipe-{recipe['seed']}")
    pair = list(rng.choice(PAIRS))
    rng.shuffle(pair)                         # which one is on top
    recipe.update(algos=pair, title=rng.choice(TITLES), bed="off", ending="WINS!",
                  hook_tpl=rng.choice(HOOKS))
    name_hook(recipe)
    return recipe


def name_hook(recipe: dict) -> dict:
    """The hook (= YouTube title), once the picture is known."""
    a, b = (NAMES[x].title() for x in recipe["algos"])
    pic = (recipe.get("picture") or {}).get("name") or "a picture"
    recipe["hook"] = recipe.get("hook_tpl", HOOKS[0]).format(a=a, b=b, pic=pic)
    recipe["hook"] = recipe["hook"][0].upper() + recipe["hook"][1:]
    return recipe


def _scramble(recipe: dict) -> list:
    perm = list(range(S_N * S_N))
    random.Random(f"sortrace-{recipe['seed']}").shuffle(perm)
    return perm


class SortSim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.algos = list(recipe.get("algos") or PAIRS[0])
        perm = _scramble(recipe)
        steps = [sum(1 for _ in ALGOS[k](list(perm))) for k in self.algos]
        self.steps = steps
        self.rate = max(steps) / (S_TARGET * FPS)
        self.arrs = [list(perm) for _ in self.algos]
        self.gens = [ALGOS[k](self.arrs[i]) for i, k in enumerate(self.algos)]
        self.budget = [0.0, 0.0]
        self.done = [None, None]          # when each finished (video time)
        self.tones: list = []             # (t, side, value 0..1 or -1 = finished)
        self.events: list = []
        self.t = 0.0
        self.unlocked_at = {0: 0.0}
        self.best = 0
        self.spawned = S_N * S_N
        self.done_at = None
        self.winner = None
        self.ok = False
        self.chain = [(None, "start"), (None, "first done"), (None, "second done")]   # (render's info)
        self.max_gap = 0.0                # (simulate: no "unlock" gaps to judge here; ok covers it)

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD) or self.t > SECONDS_CAP

    def step(self):
        self.t += 1 / FPS
        if self.t < S_LOOK or self.done_at is not None:
            return
        n = S_N * S_N
        for k in range(2):
            if self.done[k] is not None:
                continue
            self.budget[k] += self.rate
            v = None
            while self.budget[k] >= 1:
                try:
                    st = next(self.gens[k])
                except StopIteration:
                    self.done[k] = self.t
                    self.tones.append((self.t, k, -1.0))
                    self.unlocked_at[len(self.unlocked_at)] = self.t
                    break
                self.budget[k] -= 1
                if st is not None:
                    v = st[1]
            if v is not None:
                self.tones.append((self.t, k, v / n))
        if None not in self.done:
            self.done_at = self.t
            self.winner = 0 if self.done[0] <= self.done[1] else 1
            gap = abs(self.done[0] - self.done[1])
            self.ok = S_GAP_MIN <= gap <= S_GAP_MAX
            self.best = 1

    def clock(self, k: int) -> float:
        end = self.done[k] if self.done[k] is not None else self.t
        return max(0.0, end - S_LOOK)

    def music_track(self, n: int) -> np.ndarray:
        mix = np.zeros(n, np.float32)
        seen = set()
        for t, k, v in self.tones:
            a0 = int(t * SR)
            if a0 >= n:
                continue
            if v < 0:                          # one is done: a rising sweep
                d = 0.6
                ts = np.arange(int(SR * d)) / SR
                f = 300 + 900 * ts / d
                w = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, (d - ts) / 0.1)
            else:
                if (round(t, 4), k) in seen:
                    continue
                seen.add((round(t, 4), k))
                f = 220 + 880 * v
                d = 1.6 / FPS
                ts = np.arange(int(SR * d)) / SR
                tone = np.sin(2 * np.pi * f * ts) if k == 0 else 2 / np.pi * np.arcsin(np.sin(2 * np.pi * f * ts))
                w = 0.12 * tone * np.sin(np.pi * ts / d)
            seg = mix[a0:a0 + len(w)]
            seg += w[:len(seg)].astype(np.float32)
        peak = float(np.abs(mix).max()) or 1.0
        return mix / max(1.0, peak / 0.9)


def _load_picture(recipe: dict) -> Image.Image:
    pic = recipe.get("picture") or {}
    path = pic.get("path") or FALLBACKS[0]["path"]
    p = Path(path)
    if not p.is_absolute():
        p = ASSETS / "pictures" / path
    try:
        im = Image.open(p).convert("RGB")
    except OSError:
        im = Image.open(ASSETS / "pictures" / FALLBACKS[0]["path"]).convert("RGB")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    return im.resize((S_N, S_N), Image.LANCZOS)


class SortPainter:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.px = np.asarray(_load_picture(recipe)).reshape(-1, 3)
        self.bg = Image.new("RGB", (W, H), (14, 16, 26))
        self.f_big = _font(68)
        self.f_lab = _font(54)
        self.f_t = _font(50)
        self.algos = list(recipe.get("algos") or PAIRS[0])

    def frame(self, sim: SortSim) -> Image.Image:
        im = self.bg.copy()
        d = ImageDraw.Draw(im)
        if sim.winner is not None:
            title = f"{NAMES[self.algos[sim.winner]]} WINS!"
            d.text((W // 2, 170), title, font=self.f_big, fill=(250, 215, 90), anchor="mm", stroke_width=5,
                   stroke_fill=(0, 0, 0))
        else:
            d.text((W // 2, 170), self.recipe.get("title") or TITLES[0], font=self.f_big, fill=(255, 255, 255),
                   anchor="mm", stroke_width=5, stroke_fill=(0, 0, 0))
        x0 = (W - PANEL) // 2
        for k, key in enumerate(self.algos):
            y0 = 330 + k * (PANEL + 130)
            pic = Image.fromarray(self.px[np.array(sim.arrs[k])].reshape(S_N, S_N, 3).astype(np.uint8))
            pic = pic.resize((PANEL, PANEL), Image.NEAREST)
            if sim.winner == k:
                d.rounded_rectangle([x0 - 8, y0 - 8, x0 + PANEL + 8, y0 + PANEL + 8], 18, fill=(250, 215, 90))
            im.paste(pic, (x0, y0))
            d.rounded_rectangle([x0 + 10, y0 - 50, x0 + 380, y0 + 30], 22, fill=COLOURS[key], outline=(0, 0, 0),
                                width=3)
            d.text((x0 + 195, y0 - 10), NAMES[key], font=self.f_lab, fill=(20, 20, 20), anchor="mm")
            if sim.done[k] is None:
                lab, first = f"{sim.clock(k):4.1f} s", False
            else:
                other = sim.done[1 - k]
                first = other is None or sim.done[k] <= other
                lab = ("1ST " if first else "2ND ") + f"{sim.clock(k):.1f} s"
            d.rounded_rectangle([x0 + 400, y0 - 50, x0 + 690, y0 + 30], 22,
                                fill=(250, 215, 90) if first else (235, 235, 240), outline=(0, 0, 0), width=3)
            d.text((x0 + 545, y0 - 10), lab, font=self.f_t, fill=(20, 20, 20), anchor="mm")
        return im


# ---- pictures --------------------------------------------------------------

OPENVERSE = "https://api.openverse.org/v1/images/"
UA = "Dropvolve/1.0 (sorting race pictures)"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def picture_for(store: Path, used: list, log=print, http=None) -> dict:
    """A picture for the next race: the one used longest ago (never used
    first), downloaded once from Openverse (CC0 / public domain only) and
    kept under ``store``; the bundled cat / coffee when nothing works."""
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    order = sorted(PICTURES, key=lambda p: (used.count(p[0]), -used[::-1].index(p[0]) if p[0] in used else 0))
    for name, query, words in order[:4]:
        path = store / f"{_slug(name)}.jpg"
        meta = store / f"{_slug(name)}.json"
        if path.is_file() and meta.is_file():
            try:
                return json.loads(meta.read_text())
            except ValueError:
                pass
        try:
            got = _download(name, query, words, path, http)
        except Exception as e:                       # network, bad image: try the next one
            log(f"[sortrace] {name}: {e}")
            continue
        if got:
            meta.write_text(json.dumps(got))
            return got
    fb = FALLBACKS[len(used) % len(FALLBACKS)]
    return dict(fb, path=str(ASSETS / "pictures" / fb["path"]))


def _download(name: str, query: str, words, path: Path, http=None) -> Optional[dict]:
    if http is None:
        import requests as http
    r = http.get(OPENVERSE, params={"q": query, "license": "cc0,pdm", "page_size": 20, "mature": "false"},
                 headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    for x in r.json().get("results") or []:
        title = (x.get("title") or "").lower()
        if words and not all(w in title for w in words):
            continue
        if min(int(x.get("width") or 0), int(x.get("height") or 0)) < 400 or not x.get("url"):
            continue
        img = http.get(x["url"], headers={"User-Agent": UA}, timeout=40)
        img.raise_for_status()
        if len(img.content) > 25_000_000:
            continue
        im = Image.open(io.BytesIO(img.content))
        im.draft("RGB", (640, 640))
        im = im.convert("RGB")
        im.thumbnail((640, 640))
        im.save(path, quality=90)
        creator = (x.get("creator") or "").strip()
        lic = "public domain" if (x.get("license") or "").lower() == "pdm" else "CC0"
        credit = f"Picture: {x.get('title') or name}" + (f" by {creator}" if creator else "") + f", {lic} via Openverse"
        return {"name": name, "path": str(path), "credit": credit[:300]}
    return None


def publish_text(recipe: dict) -> dict:
    a, b = (NAMES[x].title() for x in recipe.get("algos") or PAIRS[0])
    pic = recipe.get("picture") or {}
    desc = (f"16,384 pixels of {pic.get('name') or 'a picture'}, scrambled the same way twice. {a} and {b} put "
            f"them back together side by side: every comparison and every move counts as one step, at the same "
            f"speed for both.\n\nWho did you pick? 👇\n\n#satisfying #sorting #algorithm #coding")
    if pic.get("credit"):
        desc += "\n\n" + pic["credit"]
    return {"title": recipe["hook"][:100], "description": desc}
