"""Brand League: the 1v1 duels as one running series of brand fights.

31 Oct: the colour duels became a league (champion, records, rematches)
after the colour duel was the channel's one breakout. 1 Nov: Dean saw
"Royal Purple VS Cherry Red" again and said nobody relates to colours,
"even I don't watch the video" -- use the brand icons instead. So the cast
is the bundled logo packs (clipper/assets/packs/logos_*: Simple Icons, the
credit line says the logos belong to their owners), and every fight is two
brands from one category -- Toyota VS BMW, Apple VS Samsung, TikTok VS
Instagram -- a side people already argue about.

* The category rotates fight by fight (``CATEGORIES`` order); each has its
  own champion and first matchup (``FIRST``).
* Within it: Dean's pick from the comments (used when its category comes
  up) > a rematch after a close fight (winner on <= 2 hearts; never a
  rematch of a rematch) > a brand that hasn't fought yet > the best record.
* The ball is the brand's badge (``disc``), rebuilt from the logo image
  without the crown the packs' last picture wears.
* Nothing is stored apart from the videos: each league video's recipe has
  ``league`` (+ ``winner`` / ``winner_hp`` after render); the standings are
  the brand fights posted or slotted earlier. The old colour-league fights
  (no ``cat``) are left out, so brand fight #1 starts the count.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

PACKS = Path(__file__).resolve().parent / "assets" / "packs"
CATEGORIES = ("phones", "cars", "apps", "gaming", "sports")
CAT_LABEL = {"phones": "PHONES", "cars": "CARS", "apps": "APPS", "gaming": "GAMING", "sports": "SPORTS"}
CAT_EMOJI = {"phones": "📱", "cars": "🚗", "apps": "📲", "gaming": "🎮", "sports": "👟"}
FIRST = {"phones": ("Apple", "Samsung"), "cars": ("BMW", "Audi"), "apps": ("TikTok", "Instagram"),
         "gaming": ("PlayStation", "Steam"), "sports": ("Nike", "Adidas")}
CREDIT = "Logos: Simple Icons. The logos belong to their owners; this video isn't made or endorsed by them."
CLOSE_HP = 2
TAGS = ("REMATCH", "NEW CHALLENGER", "TOP CONTENDER", "YOUR PICK", "FIRST FIGHT")

HOOKS = {
    "FIRST FIGHT": ["{a} VS {b}: who wins?", "Team {a} or team {b}?", "{a} VS {b}. Pick a side"],
    "REMATCH": ["{b} wants revenge on {a}", "Rematch! {a} VS {b}", "{a} VS {b}: the rematch"],
    "NEW CHALLENGER": ["{b} takes on {a}", "Can {b} beat {a}?", "{a} VS {b}: who wins?"],
    "TOP CONTENDER": ["{a} VS {b} for the crown", "Can {b} take {a}'s crown?"],
    "YOUR PICK": ["You asked for it: {a} VS {b}", "{a} VS {b}: you picked this one"],
}
STREAK_HOOKS = ["Can anyone stop {a}? {n} wins in a row", "{a} goes for {m} in a row"]


@lru_cache(maxsize=1)
def roster() -> Dict[str, dict]:
    """Brand name -> {cat, file, col} from the bundled logo packs."""
    out = {}
    for cat in CATEGORIES:
        d = PACKS / f"logos_{cat}"
        pack = json.loads((d / "pack.json").read_text(encoding="utf-8"))
        for it in pack["items"]:
            im = Image.open(d / it["file"]).convert("RGB")
            col = im.getpixel((94, 256))           # the badge's own colour, left of the logo
            out[it["name"]] = {"cat": cat, "file": f"logos_{cat}/{it['file']}", "col": list(col)}
    return out


BY_NAME = roster()       # 40 small images, read once at import


def disc(rel: str, size: int) -> Image.Image:
    """A brand's badge as a round ball picture, ``size`` px: the badge's
    colour with the logo from its middle (which leaves out the crown the
    packs' final picture wears above the badge)."""
    return _disc(rel, size)


@lru_cache(maxsize=256)
def _disc(rel: str, size: int) -> Image.Image:
    src = Image.open(PACKS / rel).convert("RGBA")       # 512 px, badge r 190 around (256, 256)
    col = src.convert("RGB").getpixel((94, 256))
    big = Image.new("RGBA", (380, 380), (0, 0, 0, 0))
    ImageDraw.Draw(big).ellipse((0, 0, 379, 379), fill=col + (255,))
    inner = Image.new("L", (380, 380), 0)
    # the pack's last picture wears a crown that reaches into the badge's
    # rim: its (compact) logo is cut tighter
    ri = 118 if rel.endswith("08.webp") else 136
    ImageDraw.Draw(inner).ellipse((190 - ri, 190 - ri, 190 + ri, 190 + ri), fill=255)
    logo = src.crop((66, 66, 446, 446))
    big.paste(logo, (0, 0), Image.fromarray(np.minimum(np.asarray(inner), np.asarray(logo.getchannel("A")))))
    mask = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size * 3 - 1, size * 3 - 1), fill=255)
    out = big.resize((size, size), Image.LANCZOS)
    out.putalpha(mask.resize((size, size), Image.LANCZOS))
    return out


def _brand_fights(history: List[dict]) -> List[dict]:
    return [f for f in history if f.get("cat") and all(n in BY_NAME for n in f.get("names", []))]


def standings(history: List[dict], cat: Optional[str] = None) -> dict:
    """Records (all brand fights), plus the champion and streak in ``cat``."""
    fights = _brand_fights(history)
    rec: Dict[str, List[int]] = {n: [0, 0] for n in BY_NAME}
    champ, streak = None, 0
    for f in fights:
        a, b = f["names"]
        w = f.get("winner")
        if w not in (a, b):
            continue
        lose = b if w == a else a
        rec[w][0] += 1
        rec[lose][1] += 1
        if f["cat"] == cat:
            streak = streak + 1 if w == champ else 1
            champ = w
    return {"records": rec, "champ": champ, "streak": streak, "fights": len(fights)}


def next_cat(history: List[dict]) -> str:
    return CATEGORIES[len(_brand_fights(history)) % len(CATEGORIES)]


def next_match(history: List[dict], seed, pick: Optional[str] = None) -> dict:
    """The next brand fight: {no, cat, names [champion, challenger],
    records, tag, streak, champ, hook}."""
    fights = _brand_fights(history)
    cat = next_cat(history)
    st = standings(history, cat)
    rec = st["records"]
    champ = st["champ"]
    in_cat = [f for f in fights if f["cat"] == cat]
    brands = [n for n, v in roster().items() if v["cat"] == cat]
    pick_rng = random.Random(f"league-pick-{st['fights']}")
    rng = random.Random(f"league-{seed}")
    if not champ:
        a, b = FIRST[cat]
        tag = "FIRST FIGHT"
    else:
        a = champ
        last = in_cat[-1]
        loser = next(n for n in last["names"] if n != last.get("winner"))
        if pick and pick in brands and pick != a:
            b, tag = pick, "YOUR PICK"
        elif (last.get("winner_hp") or 99) <= CLOSE_HP and last.get("tag") != "REMATCH" and loser != a:
            b, tag = loser, "REMATCH"
        else:
            pool = [n for n in brands if n != a and n not in last["names"]] or [n for n in brands if n != a]
            fresh = [n for n in pool if sum(rec.get(n, [0, 0])) == 0]
            if fresh:
                b, tag = pick_rng.choice(fresh), "NEW CHALLENGER"
            else:
                def pct(n):
                    w, l = rec.get(n, [0, 0])
                    return (w + 1) / (w + l + 2)
                best = max(pct(n) for n in pool)
                b, tag = pick_rng.choice([n for n in pool if pct(n) >= best - 1e-9]), "TOP CONTENDER"
    streak = st["streak"] if champ else 0
    if streak >= 2:
        hook = rng.choice(STREAK_HOOKS).format(a=a, n=streak, m=streak + 1)
    else:
        hook = rng.choice(HOOKS[tag]).format(a=a, b=b)
    return {"no": st["fights"] + 1, "cat": cat, "names": [a, b],
            "records": [list(rec.get(a, [0, 0])), list(rec.get(b, [0, 0]))],
            "tag": tag, "streak": streak, "champ": bool(champ), "hook": hook}


def apply(recipe: dict, match: dict) -> dict:
    """Make a duel recipe (ball_duel.duel_recipe) this brand fight: the
    champion on the left, the brands' badges as the balls."""
    r = roster()
    players = []
    for n in match["names"]:
        col = list(r[n]["col"])
        if sum(col) < 150:              # Apple, Nike, TikTok...: black badges vanish on the dark arena
            col = [225, 228, 238]
        players.append([n, "", col, None, "brand:" + r[n]["file"]])
    recipe.pop("team_names", None)
    recipe.update(look="brands", players=players, hook=match["hook"],
                  league={k: match[k] for k in ("no", "cat", "names", "records", "tag", "streak", "champ")})
    return recipe


def record(r) -> str:
    return f"{r[0]}-{r[1]}"


def publish_text(recipe: dict) -> dict:
    lg = recipe["league"]
    a, b = lg["names"]
    ra, rb = (record(x) for x in lg["records"])
    cat = lg.get("cat") or ""
    title = f"{recipe['hook']} {CAT_EMOJI.get(cat, '⚔️')}"
    who_a = "the champion" if lg.get("champ") else ""
    desc = (f"{a} ({ra}{', ' + who_a if who_a else ''}) VS {b} ({rb}). Ten hearts each, weapons drop into the "
            f"ring, every gun fires once. The winner keeps the {CAT_LABEL.get(cat, '').lower()} crown.\n\n"
            f"Team {a} or team {b}? Comment 👇\n\n{CREDIT}\n\n#satisfying #bouncingball #battle")
    return {"title": title[:100], "description": desc}


def table(history: List[dict]) -> List[dict]:
    """The standings for the page: each category's champion first, then by wins."""
    fights = _brand_fights(history)
    st = standings(history)
    champs = {}
    for cat in CATEGORIES:
        s = standings(history, cat)
        if s["champ"]:
            champs[s["champ"]] = s["streak"]
    rows = []
    for n, (w, l) in st["records"].items():
        if not (w or l):
            continue
        c = BY_NAME[n]["col"]
        rows.append({"name": n, "w": w, "l": l, "col": "#%02x%02x%02x" % tuple(c), "cat": BY_NAME[n]["cat"],
                     "champ": n in champs, "streak": champs.get(n, 0)})
    rows.sort(key=lambda r: (not r["champ"], -r["w"], r["l"], r["name"]))
    return rows if fights else []
