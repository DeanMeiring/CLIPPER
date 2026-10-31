"""Ball League: the colour duels as one running series (31 Oct).

Dean asked how to grow Dropvolve. The one breakout so far was a colour duel
(Royal Purple VS Cherry Red, 14.6k views, 68% kept watching), and the
biggest ball-fight channel (Earclacks' Weapon Balls) grew on a fixed cast of
named balls with records and leagues, not one-off fights. So every colour
duel is now a league fight: the champion (left) against a challenger, both
with their win-loss record on screen, and the winner holds the belt.

Nothing is stored apart from the videos themselves: each league video's
recipe carries its fight (``recipe["league"]``) and, once made, the winner.
The standings before a new fight are the fights of every league video
posted or made for an earlier slot, in slot order, so a Remake of the
upcoming videos re-runs the story in order.

Who fights next (``next_match``):
* the first fight is the rematch of the breakout duel, Purple VS Red;
* Dean's pick from the comments, when he set one ("YOUR PICK");
* after a close fight (the winner on 2 hearts or fewer) the loser gets a
  rematch ("REMATCH"), but never a rematch of a rematch;
* else a colour that hasn't fought yet ("NEW CHALLENGER"), and once
  everyone has, the best record ("TOP CONTENDER").
Nobody who fought in the last two fights, and no colour too close to the
champion's (two greens) is picked.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from clipper.ball_duel import COLOURS, EXTRA_COLOURS, _close

# the cast: the duel's 8 colours + five more (Coal Black would vanish on the
# dark arenas and Snow White on the notebook's paper; Steel Grey reads as
# "no colour")
ROSTER = COLOURS + [c for c in EXTRA_COLOURS if c[1] not in ("Black", "White", "Grey")]
FIRST = ("Royal Purple", "Cherry Red")          # the breakout duel
CLOSE_HP = 2                                    # the winner on this or fewer: a close fight
TAGS = ("REMATCH", "NEW CHALLENGER", "TOP CONTENDER", "YOUR PICK")

HOOKS = {
    "REMATCH": ["{b} wants revenge", "The rematch: {a} VS {b}", "Rematch! Can {b} get revenge?"],
    "NEW CHALLENGER": ["New challenger: {b} VS {a}", "Can {b} beat the champ?", "{b} steps up to {a}"],
    "TOP CONTENDER": ["Title fight: {a} VS {b}", "{a} VS {b} for the belt", "Can {b} take the belt?"],
    "YOUR PICK": ["You asked for it: {a} VS {b}", "You picked {b}. Can it win?"],
}
STREAK_HOOKS = ["Can anyone stop {a}? {n} wins in a row", "{a} goes for {m} in a row"]


def name(c) -> str:
    return f"{c[0]} {c[1]}"


BY_NAME = {name(c): c for c in ROSTER}


def standings(history: List[dict]) -> dict:
    """Records, the champion and their streak after these fights (oldest
    first, each a recipe's ``league`` with its ``winner``)."""
    rec: Dict[str, List[int]] = {n: [0, 0] for n in BY_NAME}
    champ, streak = None, 0
    for f in history:
        a, b = f["names"]
        w = f.get("winner")
        if w not in (a, b):
            continue
        lose = b if w == a else a
        rec.setdefault(w, [0, 0])[0] += 1
        rec.setdefault(lose, [0, 0])[1] += 1
        streak = streak + 1 if w == champ else 1
        champ = w
    return {"records": rec, "champ": champ, "streak": streak, "fights": len(history)}


def next_match(history: List[dict], seed, pick: Optional[str] = None) -> dict:
    """The next fight: {no, names [champion, challenger], records, tag,
    streak, hook}."""
    st = standings(history)
    # the challenger by the fight's number (so the page can show who's next),
    # the hook's wording by the video's seed
    pick_rng = random.Random(f"league-pick-{st['fights']}")
    rng = random.Random(f"league-{seed}")
    rec = st["records"]
    champ = st["champ"]
    if not champ:
        a, b, tag = FIRST[0], FIRST[1], "REMATCH"
    else:
        a = champ
        last = history[-1]
        recent = {n for f in history[-2:] for n in f["names"]}
        col = BY_NAME[a][2] if a in BY_NAME else (0, 0, 0)

        def fits(n):
            # not two of a colour (Lime / Mint Green, Ocean / Navy Blue)
            return (n != a and n in BY_NAME and not _close(BY_NAME[n][2], col)
                    and BY_NAME[n][1] != BY_NAME.get(a, ("", ""))[1])

        loser = next(n for n in last["names"] if n != last.get("winner"))
        if pick and fits(pick):
            b, tag = pick, "YOUR PICK"
        elif (last.get("winner_hp") or 99) <= CLOSE_HP and last.get("tag") != "REMATCH" and fits(loser):
            b, tag = loser, "REMATCH"
        else:
            pool = [n for n in BY_NAME if fits(n) and n not in recent] or [n for n in BY_NAME if fits(n)]
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
    if not champ:
        hook = rng.choice(["The rematch: {a} VS {b}", "{a} VS {b}: the rematch"]).format(a=a, b=b)
    elif streak >= 2:
        hook = rng.choice(STREAK_HOOKS).format(a=a, n=streak, m=streak + 1)
    else:
        hook = rng.choice(HOOKS[tag]).format(a=a, b=b)
    return {"no": st["fights"] + 1, "names": [a, b], "records": [list(rec.get(a, [0, 0])), list(rec.get(b, [0, 0]))],
            "tag": tag, "streak": streak, "champ": bool(champ), "hook": hook}


def apply(recipe: dict, match: dict) -> dict:
    """Make a duel recipe (ball_duel.duel_recipe) this league fight: the
    champion on the left, colours, the hook. Weapons and arena stay as the
    duel drew them."""
    a, b = (BY_NAME[n] for n in match["names"])
    recipe.pop("team_names", None)
    recipe.update(look="colours", players=[list(a), list(b)], hook=match["hook"],
                  league={k: match[k] for k in ("no", "names", "records", "tag", "streak", "champ")})
    return recipe


def record(r) -> str:
    return f"{r[0]}-{r[1]}"


def publish_text(recipe: dict) -> dict:
    """Title and description for a league fight."""
    lg = recipe["league"]
    a, b = lg["names"]
    ra, rb = (record(r) for r in lg["records"])
    title = f"{recipe['hook']} ⚔️ Ball League #{lg['no']}"
    if len(title) > 100:
        title = f"{recipe['hook']} ⚔️"[:100]
    who_a = "the champion" if lg.get("champ") else "the last winner"
    tag = {"REMATCH": "the rematch", "NEW CHALLENGER": "a new challenger", "TOP CONTENDER": "the top contender",
           "YOUR PICK": "picked in the comments"}.get(lg["tag"], "")
    others = [n for n in BY_NAME if n not in (a, b)]
    rng = random.Random(f"league-desc-{recipe.get('seed')}")
    rng.shuffle(others)
    desc = (f"Ball League, fight #{lg['no']}: {a} ({ra}, {who_a}) VS {b} ({rb}, {tag}). Ten hearts each, "
            f"weapons drop into the ring, every gun fires once. The winner holds the belt.\n\n"
            f"Who should fight the champion next? Comment a colour 👇 "
            f"({', '.join(others[:4])}...)\n\n#satisfying #physics #bouncingball #battle")
    return {"title": title, "description": desc}


def table(history: List[dict]) -> List[dict]:
    """The standings for the page: champion first, then by wins."""
    st = standings(history)
    rows = []
    for n, (w, l) in st["records"].items():
        rows.append({"name": n, "w": w, "l": l, "col": "#%02x%02x%02x" % tuple(BY_NAME[n][2]) if n in BY_NAME else "#888",
                     "champ": n == st["champ"], "streak": st["streak"] if n == st["champ"] else 0})
    rows.sort(key=lambda r: (not r["champ"], -r["w"], r["l"], r["name"]))
    return rows
