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

* ``tourney`` -- a 16-ball tournament (a viewer asked under a duel, 25 Oct):
  a bracket on top; each round's fights play at once in smaller arenas (8,
  4, 2, then the final full size). Each fight is a DuelSim; fewer hearts
  early on and sudden death (one heart, a rocket each, the ring closes in)
  keep it under a minute.

None of them uses a picture pack: the balls are the contestants. The guns are
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
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from clipper import ball_evolution as be

W, H, FPS, SR = be.W, be.H, be.FPS, be.SR
END_HOLD = 1.8              # see ball_evolution.END_HOLD
SECONDS_CAP = 70.0
KINDS = ("duel", "race", "tourney", "royale")

# colour names for the contestants: (first word, second word, rgb)
COLOURS = [("Cherry", "Red", (228, 52, 64)), ("Ocean", "Blue", (84, 140, 245)),
           ("Lime", "Green", (96, 204, 84)), ("Royal", "Purple", (154, 96, 232)),
           ("Sunny", "Yellow", (246, 200, 46)), ("Hot", "Pink", (240, 92, 172)),
           ("Tangerine", "Orange", (250, 142, 44)), ("Ice", "Cyan", (70, 210, 230))]

DUEL_HOOKS = ["{a} VS {b}", "{a} VS {b} -- who wins?", "Who wins? {a} VS {b}"]

# Flag balls (Dean, 25 Oct: more kinds of fights after the first duel took
# off). The flags are flag-icons' square SVGs (MIT), drawn to 256 px in the
# sandbox only (cairosvg isn't an app dependency), in assets/flags. The
# colour is the country's for its hearts and name. Countries in an armed
# conflict with each other are left out on purpose: these balls shoot each
# other, so it stays a football-style rivalry.
COUNTRIES = {
    "br": ("Brazil", (250, 214, 30)), "ar": ("Argentina", (116, 172, 223)), "fr": ("France", (80, 130, 240)),
    "gb-eng": ("England", (235, 235, 240)), "de": ("Germany", (255, 206, 0)), "nl": ("Netherlands", (255, 120, 20)),
    "es": ("Spain", (230, 40, 50)), "pt": ("Portugal", (40, 170, 70)), "it": ("Italy", (40, 120, 230)),
    "be": ("Belgium", (240, 50, 60)), "us": ("USA", (90, 120, 230)), "mx": ("Mexico", (20, 160, 90)),
    "ca": ("Canada", (230, 40, 50)), "au": ("Australia", (255, 200, 40)), "nz": ("New Zealand", (220, 220, 225)),
    "jp": ("Japan", (230, 40, 70)), "kr": ("South Korea", (220, 50, 70)), "ng": ("Nigeria", (30, 170, 90)),
    "gh": ("Ghana", (250, 200, 30)), "ma": ("Morocco", (220, 40, 50)), "eg": ("Egypt", (220, 40, 50)),
    "za": ("South Africa", (40, 170, 90)), "sn": ("Senegal", (40, 170, 90)), "cm": ("Cameroon", (40, 170, 90)),
    "uy": ("Uruguay", (110, 170, 240)), "co": ("Colombia", (255, 210, 0)), "cl": ("Chile", (230, 40, 50)),
    "se": ("Sweden", (255, 205, 0)), "no": ("Norway", (230, 40, 60)), "dk": ("Denmark", (230, 40, 60)),
    "ie": ("Ireland", (30, 170, 90)), "gb-sct": ("Scotland", (60, 130, 230)), "ch": ("Switzerland", (230, 40, 50)),
    "pl": ("Poland", (230, 40, 70)), "hr": ("Croatia", (230, 50, 60)), "tr": ("Turkey", (230, 40, 50)),
    "in": ("India", (255, 150, 50)), "ph": ("Philippines", (60, 110, 220)), "id": ("Indonesia", (230, 40, 50)),
    "th": ("Thailand", (80, 80, 200)), "vn": ("Vietnam", (230, 40, 50)), "ke": ("Kenya", (40, 160, 80)),
    "jm": ("Jamaica", (255, 200, 30)), "gr": ("Greece", (60, 120, 220)), "at": ("Austria", (230, 40, 50)),
}
RIVALRIES = [("br", "ar"), ("gb-eng", "fr"), ("gb-eng", "de"), ("de", "nl"), ("es", "pt"), ("us", "mx"),
             ("us", "ca"), ("au", "nz"), ("ng", "gh"), ("ma", "eg"), ("it", "fr"), ("es", "it"), ("be", "nl"),
             ("uy", "ar"), ("br", "de"), ("ar", "fr"), ("pt", "ar"), ("za", "ng"), ("gb-eng", "gb-sct"),
             ("se", "no"), ("dk", "se"), ("jm", "us"), ("co", "br"), ("cl", "ar"), ("sn", "cm"), ("gr", "tr")]
ALT_COLOURS = {"fr": (240, 70, 80), "it": (40, 170, 80), "ar": (250, 214, 30), "uy": (250, 214, 30),
               "eg": (235, 235, 240), "ma": (40, 160, 80), "de": (230, 40, 50), "se": (60, 120, 220),
               "no": (60, 90, 200), "dk": (235, 235, 240), "gb-sct": (235, 235, 240), "cm": (230, 50, 50)}
FLAG_DUEL_HOOKS = ["{a} VS {b} -- who wins?", "Who wins? {a} VS {b}", "{a} VS {b}", "Pick a side: {a} or {b}?"]
FLAG_TOURNEY_HOOKS = ["16 countries, 1 winner", "Which country wins it all?", "16 countries enter. Only 1 wins.",
                      "Pick your country: 16 enter, 1 wins"]
_FLAGS: dict = {}


def _flag(code: str, size: int) -> Image.Image:
    """A country's flag as a round disc, ``size`` px."""
    key = (code, size)
    im = _FLAGS.get(key)
    if im is None:
        src = Image.open(be.ASSETS / "flags" / f"{code}.webp").convert("RGBA").resize((size, size), Image.LANCZOS)
        mask = Image.new("L", (size * 3, size * 3), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size * 3 - 1, size * 3 - 1), fill=255)
        src.putalpha(mask.resize((size, size), Image.LANCZOS))
        im = _FLAGS[key] = src
    return im


def _close(a, b) -> bool:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 < 110


def _country(code: str) -> list:
    """A player entry for a country: [name, "", colour, flag code]."""
    name, col = COUNTRIES[code]
    return [name, "", list(col), code]


def _stroke(col) -> tuple:
    """The outline for white text in a player's colour: a light colour
    (England, Snow White) would vanish, so those get a dark one."""
    return (40, 42, 64) if min(col) > 170 else tuple(col)


def _name(p) -> str:
    return " ".join(x for x in (p[0], p[1]) if x)


def _name_parts(p, last: bool) -> list:
    """(text, colour) runs of a player's name: the first word in its colour,
    the second in white; a country's whole name in its colour."""
    w1, w2, col = p[0], p[1], tuple(p[2])
    if not w2:
        return [(w1 + ("" if last else " "), col)]
    return [(w1 + " ", col), (w2 + ("" if last else " "), (255, 255, 255))]
RACE_HOOKS = ["First to {n} wins", "Pick a colour: first to {n} wins", "Which colour gets {n} first?"]


def duel_recipe(recipe: dict, force: Optional[str] = None) -> dict:
    """The duel's own draws (own rng, so other kinds' recipes don't change).
    `force` sets the look (colours / flags / teams) the learning picked; the
    draws stay in the same order, so a forced look draws what that look
    would have drawn anyway."""
    recipe.pop("look", None)
    recipe.pop("team_names", None)
    recipe["arsenal"] = 2            # the new weapons (WEAPONS2); not a draw, so the rest is unchanged
    recipe["style"] = "clean"        # health bars, a bigger arena (DuelPainter)
    mix = random.Random(f"weapons-{recipe['seed']}")      # own rng: the draws below stay as they were
    recipe["weapons"] = mix.sample(CLASSICS, 4) + mix.sample(NEWER, 6)
    recipe["arena"] = mix.choice(ARENAS)
    rng = random.Random(f"duel-recipe-{recipe['seed']}")
    a, b = rng.sample(COLOURS, 2)
    recipe.update(players=[list(a), list(b)], bed="off",
                  hook=rng.choice(DUEL_HOOKS).format(a=f"{a[0]} {a[1]}", b=f"{b[0]} {b[1]}"),
                  ending="WINS!")
    look = random.Random(f"duel-look-{recipe['seed']}")      # own rng: the draws above stay as they were
    x = look.random()
    x = {"flags": 0.2, "teams": 0.6, "colours": 0.9}.get(force, x)
    if x < 0.45:
        # flag balls: a known rivalry most of the time, else any two countries
        pair = list(look.choice(RIVALRIES)) if look.random() < 0.7 else look.sample(sorted(COUNTRIES), 2)
        look.shuffle(pair)
        players = [_country(c) for c in pair]
        if _close(players[0][2], players[1][2]):
            # two blues (Italy, France): the second one's hearts and name take another colour
            players[1][2] = list(ALT_COLOURS.get(pair[1], (235, 235, 240)))
        recipe.update(look="flags", players=players,
                      hook=look.choice(FLAG_DUEL_HOOKS).format(a=players[0][0], b=players[1][0]))
    elif x < 0.7:
        # 2 VS 2: a warm team and a cool team
        teams = [look.sample(WARM, 2), look.sample(COOL, 2)]
        look.shuffle(teams)
        players = [list(c) for c in teams[0] + teams[1]]
        names = [_team_name(teams[0]), _team_name(teams[1])]
        recipe.update(look="teams", players=players, team_names=names,
                      hook=look.choice(TEAM_HOOKS).format(a=names[0], b=names[1]))
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
    elif kind == "laser":
        w, h = 200, 70
    elif kind == "freeze":
        w, h = 184, 84
    elif kind == "flamethrower":
        w, h = 200, 88
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
    elif kind == "laser":
        white, grey, red = (232, 236, 244, 255), (150, 156, 172, 255), (255, 50, 70, 255)
        d.polygon(P([(30, 34), (58, 34), (50, 68), (24, 68)]), fill=grey, outline=ink, width=o)     # grip
        box(10, 12, 150, 38, white, 10)                                                             # body
        box(146, 18, 186, 32, grey, 4)                                                              # barrel
        d.ellipse((176 * k, 14 * k, 198 * k, 36 * k), fill=red, outline=ink, width=o)               # emitter
        for x in (40, 62, 84):
            d.rounded_rectangle((x * k, 20 * k, (x + 14) * k, 30 * k), 2 * k, fill=(255, 90, 110, 255))
    elif kind == "freeze":
        ice, ice_hi, white = (120, 200, 250, 255), (190, 236, 255, 255), (236, 244, 252, 255)
        d.polygon(P([(30, 44), (60, 44), (52, 80), (24, 80)]), fill=white, outline=ink, width=o)     # grip
        box(10, 14, 128, 48, white, 12)                                                             # body
        d.ellipse((60 * k, 0, 104 * k, 30 * k), fill=ice, outline=ink, width=o)                     # tank
        d.ellipse((70 * k, 6 * k, 84 * k, 16 * k), fill=ice_hi)
        box(124, 20, 168, 42, ice, 8)                                                               # nozzle
        box(160, 16, 182, 46, ice_hi, 6)
    elif kind == "flamethrower":
        red, grey = (214, 60, 46, 255), (120, 126, 140, 255)
        d.ellipse((4 * k, 34 * k, 70 * k, 84 * k), fill=red, outline=ink, width=o)                   # tank
        d.rectangle((20 * k, 44 * k, 54 * k, 52 * k), fill=(250, 214, 60, 255))
        box(40, 18, 150, 40, grey, 6)                                                               # body
        box(146, 22, 196, 36, (70, 75, 86, 255), 4)                                                 # nozzle
        d.polygon(P([(186, 14), (200, 29), (186, 44)]), fill=(255, 150, 40, 255), outline=ink, width=2 * k)
        d.polygon(P([(64, 40), (88, 40), (82, 66), (60, 66)]), fill=grey, outline=ink, width=o)     # grip
    else:  # rocket launcher
        olive, olive_hi = (86, 112, 62, 255), (126, 156, 92, 255)
        box(6, 20, 196, 52, olive, 14)                                                             # tube
        d.line((24 * k, 27 * k, 180 * k, 27 * k), fill=olive_hi, width=3 * k)
        box(70, 6, 110, 20, dark, 3)                                                               # sight
        d.polygon(P([(62, 52), (86, 52), (80, 74), (60, 74)]), fill=dark, outline=ink, width=o)    # grip
        d.polygon(P([(194, 22), (228, 36), (194, 50)]), fill=(222, 50, 46, 255), outline=ink, width=o)  # warhead
        box(186, 18, 198, 54, (200, 200, 205, 255), 3)
    return im.resize((w, h), Image.LANCZOS)


def _saw() -> Image.Image:
    """A circular saw blade (drawn at 3x, reduced)."""
    k, r = 3, 56
    size = (2 * r + 8) * k
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size / 2
    pts = []
    for j in range(32):                      # 16 teeth
        a = j * math.pi / 16
        rr = r * k if j % 2 == 0 else (r - 12) * k
        aa = a + (0.12 if j % 2 == 0 else 0)
        pts.append((c + rr * math.cos(aa), c + rr * math.sin(aa)))
    d.polygon(pts, fill=(178, 184, 196, 255), outline=(22, 22, 28, 255), width=3 * k)
    d.ellipse((c - (r - 22) * k, c - (r - 22) * k, c + (r - 22) * k, c + (r - 22) * k), fill=(206, 212, 222, 255))
    d.ellipse((c - 14 * k, c - 14 * k, c + 14 * k, c + 14 * k), fill=(230, 60, 60, 255), outline=(22, 22, 28, 255),
              width=3 * k)
    return im.resize((2 * r + 8, 2 * r + 8), Image.LANCZOS)


def _sticker(img: Image.Image, outline: int = 6) -> Image.Image:
    """A picture as a die-cut sticker: a white outline around it (the look
    the idea board used for the newer weapons)."""
    off = 2 * outline
    big = Image.new("RGBA", (img.width + 2 * off, img.height + 2 * off), (0, 0, 0, 0))
    m = Image.new("L", big.size, 0)
    m.paste(img.getchannel("A"), (off, off))
    m = m.filter(ImageFilter.MaxFilter(outline * 2 + 1))
    white = Image.new("RGBA", big.size, (255, 255, 255, 255))
    white.putalpha(m)
    big.alpha_composite(white)
    big.alpha_composite(img, (off, off))
    return big


# the newer weapons' pictures (Fluent Emoji 3D) and where each one's business
# end points in its picture (degrees, y down), so it can be turned to aim
EMOJI_ARMS = {"katana": "1f5e1", "hammer": "1f528", "boomerang": "1fa83", "axe": "1fa93", "bow": "1f3f9",
              "lightning": "26a1", "magnet": "1f9f2", "banana": "1f34c", "glove": "1f94a", "pan": "1f373",
              "tornado": "1f32a", "blackhole": "1f300", "boulder": "1f5ff", "wand": "1fa84", "potion": "1f9ea"}
HELD_ANGLE = {"katana": 135, "hammer": -135, "axe": -135, "wand": -45, "bow": -45, "magnet": 90, "glove": -90}


def _wobble(d, pts_fn, col, width, seed=0, passes=2, amp=0.012, n=150, closed=True):
    """A hand-drawn line: the same shape drawn ``passes`` times, each with
    its own slow wobble (the notebook and chalk arenas)."""
    rng = random.Random(seed)
    for p in range(passes):
        ph = [rng.uniform(0, 6.28) for _ in range(3)]
        pts = []
        for k in range(n + (6 if closed else 1)):
            t = k / n
            j = amp * (math.sin(18.8 * t + ph[0]) + 0.6 * math.sin(31.4 * t + ph[1]) + 0.3 * math.sin(56 * t + ph[2]))
            pts.append(pts_fn(t, j, p))
        d.line(pts, fill=col, width=width, joint="curve")


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
# Dean (29 Oct): "new weapons". Recipes with ``arsenal: 2`` (every new fight)
# draw from these too; older recipes keep the list above, so they play out
# exactly as they did.
WEAPONS2 = {
    **WEAPONS,
    "laser": ("laser", 2, 2),       # aims (a red sight line), then a beam across the ring: 2 hearts
    "saw": ("melee", 1, 1),         # a spinning blade around the ball, three hits
    "freeze": ("gun", 1, 2),        # an ice shot: 1 heart and the target freezes for FREEZE_S
    "bomb": ("bomb", 3, 1),         # dropped where the ball is; goes off after BOMB_FUSE
}
SAW_HITS, FREEZE_S, BOMB_FUSE, LASER_CHARGE = 3, 1.5, 1.4, 0.5
# Then (29 Oct, from the idea board Dean liked): sixteen more. Each new
# fight gets its own mix (``recipe["weapons"]``: 4 of the classics + 6 of
# the new ones), so the fights differ from video to video.
WEAPONS2.update({
    "katana": ("dash", 2, 2),       # dashes at the enemy: 2 hearts if it connects
    "hammer": ("slam", 2, 2),       # slams the floor: a shockwave, 2 hearts close, 1 further
    "boomerang": ("throw", 1, 2),   # thrown, curves back: can hit going out and coming back
    "axe": ("throw", 2, 2),         # thrown, spinning: 2 hearts
    "bow": ("bow", 2, 2),           # draws, then an arrow that goes through everyone
    "lightning": ("chain", 2, 2),   # strikes the nearest enemy, jumps to the next ones
    "flamethrower": ("flame", 1, 2),  # a cone of fire for FLAME_S: a heart every 0.4 s, 3 at most
    "magnet": ("magnet", 1, 1),     # pulls the enemy in; contact = 1 heart
    "banana": ("trap", 1, 2),       # drops a peel: 1 heart and a spin-out
    "glove": ("glove", 1, 2),       # a punch that sends them flying; a wall after it = 1 more
    "pan": ("pan", 1, 1),           # bats bullets back, bonks on contact
    "tornado": ("summon", 1, 1),    # roams the ring, flings whoever it touches
    "blackhole": ("summon", 1, 1),  # pulls everyone near it in, hurts at the middle
    "boulder": ("summon", 2, 1),    # a Moai head rolls across: 2 hearts
    "wand": ("gun", 1, 2),          # a spark: 1 heart, and they're tiny (take +1 a hit) for SMALL_S
    "potion": ("potion", 0, 1),     # heal 2, speed or giant, at random
})
CLASSICS = ("pistol", "shotgun", "minigun", "rocket", "knife", "shield")
NEWER = tuple(k for k in WEAPONS2 if k not in CLASSICS)
FLAME_S, SMALL_S, PAN_BLOCKS, GLOVE_HITS = 1.5, 3.5, 3, 2


# ---- the duel -----------------------------------------------------------------

D_CX, D_CY, D_R = 540, 1130, 470
# The "clean" look (Dean, 29 Oct: "make it look better or cleaner"):
# recipes with ``style: "clean"`` draw the same fight a bit bigger and
# higher, under fighting-game health bars, with colour trails and a white
# flash on the knockout. Physics are untouched.
C_CY, C_K = 1050, 1.08
# The arenas a new duel / 2v2 rotates through (Dean picked these from the
# idea board, 29 Oct): the neon ring, a notebook page, a chalkboard (both
# rings, drawn by hand), an ice rink and a boxing ring (rounded boxes:
# ARENA_SHAPES, in the duel's own coordinates).
ARENAS = ("neon", "notebook", "chalk", "ice", "boxing")
ARENA_SHAPES = {"ice": (540, 1130, 420, 480, 170), "boxing": (540, 1130, 440, 440, 14)}
NEW_FX = {"wave": 2.2, "bolt": 3.2, "flame": 2.6, "slashfx": 3.0, "fall": 1.5}      # how fast the newer effects fade
NEW_KINDS = ("dash", "slam", "throw", "bow", "chain", "flame", "magnet", "trap", "glove", "pan", "summon",
             "potion")
D_BALL = 64
D_SPEED = 560
HEARTS = 10


class DuelSim:
    """The fight. ``events`` stays empty (no stock ball sounds): the duel's
    own sounds are in ``sfx`` and come out through ``music_track``."""

    def __init__(self, recipe: dict, players=None, hearts: int = HEARTS, key: str | None = None):
        # players / hearts / key: a match in the tournament (TourneySim)
        self.recipe = recipe
        self.rng = random.Random(key or f"duel-{recipe['seed']}")
        self.t = 0.0
        self.hearts = hearts
        self.rush_at = None          # tournament: from this second on, weapons drop much faster
        self.keep = False            # tournament: a ball holding a weapon walks past the others
        self.ring = D_R              # tournament: sudden death closes the ring in
        self.max_drops = 4           # weapons waiting at once while rushing (battle royale: more)
        self.new_arms = recipe.get("arsenal") == 2
        self.arms = WEAPONS2 if self.new_arms else WEAPONS
        if recipe.get("weapons"):
            self.arms = {k: WEAPONS2[k] for k in recipe["weapons"] if k in WEAPONS2}
        self.bombs: list = []
        self.things: list = []       # thrown / summoned things (the newer weapons)
        # the arena's walls: None = the ring (every older fight); else a rounded
        # box (cx, cy, half width, half height, corner radius)
        self.shape = ARENA_SHAPES.get(recipe.get("arena")) if not players else None
        self.balls = []
        for i, p in enumerate(players or recipe.get("players") or [list(c) for c in COLOURS[:2]]):
            w1, w2, col = p[:3]
            a = self.rng.uniform(0, 2 * math.pi)
            self.balls.append({
                "i": i, "name": f"{w1} {w2}" if w2 else w1, "col": tuple(col), "flag": p[3] if len(p) > 3 else None,
                "x": D_CX + (-170 if i == 0 else 170), "y": D_CY + self.rng.uniform(-120, 120),
                "vx": D_SPEED * math.cos(a), "vy": D_SPEED * math.sin(a),
                "hp": hearts, "weapon": None, "uses": 0, "wang": self.rng.uniform(0, 6.28), "held": 0.0,
                "cool": 0.0, "hurt": 0.0, "squash": 0.0, "kick": 0.0, "alive": True, "weapons_had": 0,
                "burst": 0.0, "burst_hits": 0, "next_round": 0.0, "frozen": 0.0, "charge": 0.0,
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
        if target.get("small", 0) > 0:
            dmg += 1                    # shrunk by the wand: every hit hurts more
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
        self._hp_log.append((self.t,) + self._hp_row())
        a = math.atan2(target["y"] - y, target["x"] - x)
        sp = math.hypot(target["vx"], target["vy"])
        target["vx"], target["vy"] = sp * math.cos(a), sp * math.sin(a)
        if target["hp"] <= 0:
            target["alive"] = False
            self._ko(target, by)
            for _ in range(44):
                aa = self.rng.uniform(0, 6.28)
                s = self.rng.uniform(150, 700)
                self.effects.append({"k": "shard", "x": target["x"], "y": target["y"], "vx": s * math.cos(aa),
                                     "vy": s * math.sin(aa), "life": 1.0, "col": target["col"],
                                     "r": self.rng.uniform(6, 16), "spin": self.rng.uniform(-8, 8)})
        return True

    # -- hooks the 2v2 (TeamDuelSim) changes; for two balls they are the duel's own
    def _hp_row(self) -> tuple:
        return self.balls[0]["hp"], self.balls[1]["hp"]

    def _ko(self, target, by):
        self._end(by)

    def _movers(self):
        return self.balls

    def _pairs(self):
        return [tuple(self.balls)]

    def _bullet_hit(self, u):
        o = self.other(u["owner"])
        return o if o["alive"] and math.hypot(u["x"] - o["x"], u["y"] - o["y"]) < D_BALL + 8 else None

    def _boom_targets(self, owner):
        return [self.other(owner)]

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
        self._bombs()
        if self.new_arms:
            self._things(dt)
            self._status(dt)
        self._effects(dt)
        if self.t > SECONDS_CAP - END_HOLD - 0.1 and self.done_at is None:
            self.done_at = self.t        # nobody won in time: a bad seed
            self.ok = False

    def _spawn(self):
        if self.t < self.next_spawn:
            return
        rush = self.rush_at is not None and self.t >= self.rush_at
        if len(self.pickups) < (self.max_drops if rush else 3):
            names = list(self.arms)
            w = self.rng.choices(names, [self.arms[n][2] for n in names])[0]
            for _ in range(30):
                a = self.rng.uniform(0, 6.28)
                dd = self.rng.uniform(0, self.ring - 120)
                x, y = D_CX + dd * math.cos(a), D_CY + dd * math.sin(a)
                if self.shape:                          # a box: anywhere inside it
                    cx, cy, hw, hh, _cr = self.shape
                    x = cx + (2 * a / 6.28 - 1) * (hw - 130)
                    y = cy + (2 * dd / max(1.0, self.ring - 120) - 1) * (hh - 130)
                if all(math.hypot(x - b["x"], y - b["y"]) > 190 for b in self.balls) and \
                        all(math.hypot(x - p["x"], y - p["y"]) > 170 for p in self.pickups):
                    self.pickups.append({"w": w, "x": x, "y": y, "born": self.t})
                    self.sfx.append((self.t, "spawn", 0))
                    break
        lo, hi = (1.5, 2.5) if self.new_arms else (2.0, 3.2)     # the new weapons are slower to land
        self.next_spawn = self.t + (self.rng.uniform(0.5, 0.9) if rush else self.rng.uniform(lo, hi))

    def _move(self, dt):
        for b in self._movers():
            if b["frozen"] > 0:
                continue                 # frozen solid: it stays where it is
            b["x"] += b["vx"] * dt
            b["y"] += b["vy"] * dt
            if self.shape:
                self._box_bounce(b)
                continue
            dx, dy = b["x"] - D_CX, b["y"] - D_CY
            dd = math.hypot(dx, dy)
            if dd > self.ring - D_BALL:
                nx, ny = dx / dd, dy / dd
                vn = b["vx"] * nx + b["vy"] * ny
                if vn > 0:
                    vx, vy = b["vx"] - 2 * vn * nx, b["vy"] - 2 * vn * ny
                    a = math.atan2(vy, vx) + self.rng.uniform(-0.18, 0.18)
                    sp = self._spd(b)
                    b["vx"], b["vy"] = sp * math.cos(a), sp * math.sin(a)
                    self.hits.append((self.t, 0.6))
                    self.sfx.append((self.t, "wall", 0))
                    self._wall_slam(b)
                b["x"], b["y"] = D_CX + nx * (self.ring - D_BALL), D_CY + ny * (self.ring - D_BALL)
        for a, c in self._pairs():
            self._collide(a, c)
        for p in list(self.pickups):
            for b in self.balls:
                if b["alive"] and not (self.keep and self.arms.get(b["weapon"], ("",))[0] == "gun") and \
                        math.hypot(p["x"] - b["x"], p["y"] - b["y"]) < D_BALL + 55:
                    self._equip(b, p["w"])
                    self.pickups.remove(p)
                    break

    def _collide(self, a, c):
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
                    elif self.new_arms and (p["weapon"] == "pan" or p.get("giant", 0) > 0) and p["cool"] <= 0 \
                            and self.damage(q, 1, (a["x"] + c["x"]) / 2, (a["y"] + c["y"]) / 2, p):
                        p["cool"] = 0.6
                        self.sfx.append((self.t, "bonk", 0))
            over = 2 * D_BALL - dd
            a["x"] -= nx * over / 2
            a["y"] -= ny * over / 2
            c["x"] += nx * over / 2
            c["y"] += ny * over / 2
            for b in (a, c):
                s = math.hypot(b["vx"], b["vy"]) or 1
                sp = self._spd(b)
                b["vx"], b["vy"] = b["vx"] / s * sp, b["vy"] / s * sp

    # -- the arena's walls and speeds (newer fights)
    def _spd(self, b):
        """A ball's speed: D_SPEED, faster while dashing / boosted / punched
        (only the newer weapons set those; older fights always get D_SPEED)."""
        if not self.new_arms:
            return D_SPEED
        k = 1.0
        if b.get("dash", 0) > 0:
            k *= 2.4
        if b.get("fast", 0) > 0:
            k *= 1.5
        if b.get("knock", 0) > 0:
            k *= 1.9
        return D_SPEED * k

    def _box_wall(self, x, y, r):
        """Where a circle of radius r at (x, y) pokes out of the box, as
        (contact x, contact y, outward nx, ny), or None if it's inside."""
        cx, cy, hw, hh, cr = self.shape
        hw2, hh2, c2 = hw - r, hh - r, max(0.0, cr - r)
        px, py = x - cx, y - cy
        ax, ay = abs(px), abs(py)
        qx, qy = ax - (hw2 - c2), ay - (hh2 - c2)
        if qx > 0 and qy > 0:
            d = math.hypot(qx, qy)
            if d <= c2:
                return None
            nx, ny = qx / d, qy / d
            ox, oy = (hw2 - c2) + nx * c2, (hh2 - c2) + ny * c2
        elif qx > c2:
            nx, ny, ox, oy = 1.0, 0.0, hw2, ay
        elif qy > c2:
            nx, ny, ox, oy = 0.0, 1.0, ax, hh2
        else:
            return None
        sx, sy = (1 if px >= 0 else -1), (1 if py >= 0 else -1)
        return cx + sx * ox, cy + sy * oy, sx * nx, sy * ny

    def _box_bounce(self, b):
        w = self._box_wall(b["x"], b["y"], D_BALL)
        if w is None:
            return
        px, py, nx, ny = w
        vn = b["vx"] * nx + b["vy"] * ny
        if vn > 0:
            vx, vy = b["vx"] - 2 * vn * nx, b["vy"] - 2 * vn * ny
            a = math.atan2(vy, vx) + self.rng.uniform(-0.18, 0.18)
            sp = self._spd(b)
            b["vx"], b["vy"] = sp * math.cos(a), sp * math.sin(a)
            self.hits.append((self.t, 0.6))
            self.sfx.append((self.t, "wall", 0))
            self._wall_slam(b)
        b["x"], b["y"] = px, py

    def _outside(self, x, y, margin=0.0):
        if self.shape:
            return self._box_wall(x, y, -margin) is not None
        return math.hypot(x - D_CX, y - D_CY) > self.ring + margin

    def _wall_slam(self, b):
        """Punched into a wall (the glove): one more heart."""
        if b.get("knock", 0) > 0 and b.get("slam_by"):
            by, b["slam_by"] = b["slam_by"], None
            self.damage(b, 1, b["x"], b["y"], by)

    def _equip(self, b, w):
        if b["weapon"]:
            self._drop(b)
        b["weapon"] = w
        b["weapons_had"] += 1
        b["held"] = 0.0
        b["burst"] = 0.0
        b["burst_hits"] = 0
        b["uses"] = KNIFE_HITS if w == "knife" else SHIELD_BLOCKS if w == "shield" else SAW_HITS if w == "saw" \
            else PAN_BLOCKS if w == "pan" else GLOVE_HITS if w == "glove" else 1
        b["charge"] = 0.0
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
        elif w == "freeze":
            self._bullet(b, mx, my, a, 1150, 1.4, "ice")
            self.sfx.append((self.t, "icefire", 0))
        elif w == "wand":
            self._bullet(b, mx, my, a, 1300, 1.4, "magic")
            self.sfx.append((self.t, "sparkle", 0))
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
            if b["frozen"] > 0:
                b["frozen"] = max(0.0, b["frozen"] - dt)
                continue                 # frozen: can't use its weapon either
            w = b["weapon"]
            if not w or not b["alive"]:
                continue
            kind, dmg, _wt = self.arms[w]
            aim = math.atan2(o["y"] - b["y"], o["x"] - b["x"])
            b["held"] += dt
            if kind == "melee":
                saw = w == "saw"
                b["wang"] += (9.0 if saw else 6.0) * dt
                reach = D_BALL + (74 if saw else 62)
                tx = b["x"] + math.cos(b["wang"]) * reach
                ty = b["y"] + math.sin(b["wang"]) * reach
                if b["cool"] <= 0 and math.hypot(tx - o["x"], ty - o["y"]) < D_BALL + (34 if saw else 18):
                    b["cool"] = 0.55
                    self.damage(o, dmg, tx, ty, b)
                    b["uses"] -= 1
                    if b["uses"] <= 0:
                        self._drop(b)
            elif kind == "shield":
                b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 5)
            elif kind == "bomb":
                b["wang"] += 3.0 * dt
                near = math.hypot(o["x"] - b["x"], o["y"] - b["y"]) < 300
                if b["held"] > 0.3 and (near or b["held"] > 2.0):     # put it down near them, and run
                    self.bombs.append({"x": b["x"], "y": b["y"], "t0": self.t, "owner": b})
                    self.sfx.append((self.t, "fuse", 0))
                    b["weapon"] = None
            elif kind == "laser":
                if b["charge"] > 0:
                    b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 5)        # it keeps tracking
                    b["charge"] += dt
                    if b["charge"] >= LASER_CHARGE:
                        self._laser(b)
                        self._drop(b)
                else:
                    b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 6)
                    if b["held"] > 0.4 and (abs(_ang_diff(aim, b["wang"])) < 0.25 or b["held"] > 1.4):
                        b["charge"] = 1e-6
                        self.sfx.append((self.t, "charge", 0))
            elif kind in NEW_KINDS:
                self._use(b, o, w, kind, dmg, aim, dt)
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
                # (tournament: circling balls can outrun the aim, so after 1.2 s it
                # fires near enough; with the ring closed in, it just fires)
                tol = 0.15
                if self.keep and b["held"] > 1.2:
                    tol = 0.6
                if self.ring < D_R and b["held"] > 0.6:
                    tol = 9.0
                if b["held"] > 0.45 and abs(_ang_diff(aim, b["wang"])) < tol:
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
            o = self._bullet_hit(u)
            cap = {"shotgun": 2, "minigun": 3}.get(u["kind"])
            if o is not None and o["weapon"] == "pan" and self._facing(o, u["owner"]):
                # batted straight back: it's the pan holder's bullet now
                u["vx"], u["vy"], u["owner"] = -u["vx"], -u["vy"], o
                o["uses"] -= 1
                self.sfx.append((self.t, "bonk", 0))
                self.effects.append({"k": "spark", "x": u["x"], "y": u["y"], "life": 1.0, "col": (255, 255, 255)})
                if o["uses"] <= 0:
                    self._drop(o)
                continue
            if o is not None:
                own = u["owner"]
                if cap is None or own["burst_hits"] < cap:
                    hurt = self.damage(o, 1, u["x"], u["y"], own)
                    if hurt and cap is not None:
                        own["burst_hits"] += 1
                    if hurt and u["kind"] == "ice" and o["alive"]:
                        o["frozen"] = FREEZE_S
                        self.sfx.append((self.t, "freeze", 0))
                        self.effects.append({"k": "ice", "ball": o, "x": o["x"], "y": o["y"], "life": 1.0})
                    if hurt and u["kind"] == "magic" and o["alive"]:
                        o["small"] = SMALL_S
                self.bullets.remove(u)
            elif u["life"] <= 0 or (self._outside(u["x"], u["y"]) if self.shape else
                                    math.hypot(u["x"] - D_CX, u["y"] - D_CY) > self.ring):
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
            if hit or (self._outside(r["x"], r["y"], -10) if self.shape else
                       math.hypot(r["x"] - D_CX, r["y"] - D_CY) > self.ring - 10):
                self.rockets.remove(r)
                self._boom(r["x"], r["y"], r["owner"])

    def _laser(self, b):
        """The beam: from the ball to the ring's edge along its aim; it hurts
        every enemy it passes through."""
        a = b["wang"]
        x0, y0 = b["x"] + math.cos(a) * (D_BALL + 40), b["y"] + math.sin(a) * (D_BALL + 40)
        # where the beam leaves the ring
        dx, dy = math.cos(a), math.sin(a)
        fx, fy = x0 - D_CX, y0 - D_CY
        bq = fx * dx + fy * dy
        L = -bq + math.sqrt(max(0.0, bq * bq - (fx * fx + fy * fy - self.ring * self.ring)))
        if self.shape:
            L = 0.0
            while L < 2400 and not self._outside(x0 + dx * L, y0 + dy * L):
                L += 8
        x1, y1 = x0 + dx * L, y0 + dy * L
        self.effects.append({"k": "beam", "x": x0, "y": y0, "x1": x1, "y1": y1, "life": 1.0})
        self.sfx.append((self.t, "laser", 0))
        self.shake = max(self.shake, 0.3)
        b["kick"] = 1.0
        for o in self._boom_targets(b):
            if not o["alive"]:
                continue
            t = max(0.0, min(L, (o["x"] - x0) * dx + (o["y"] - y0) * dy))
            if math.hypot(o["x"] - (x0 + dx * t), o["y"] - (y0 + dy * t)) < D_BALL + 12:
                self.damage(o, self.arms["laser"][1], x0 + dx * t, y0 + dy * t, b)

    # -- the newer weapons (WEAPONS2 from the idea board)
    def _facing(self, b, by):
        return abs(_ang_diff(math.atan2(by["y"] - b["y"], by["x"] - b["x"]), b["wang"])) < 1.2

    def _use(self, b, o, w, kind, dmg, aim, dt):
        dist = math.hypot(o["x"] - b["x"], o["y"] - b["y"])
        if kind not in ("throw", "bow"):
            b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 7)
        if kind == "dash":                                   # katana
            if b.get("dash", 0) <= 0 and b["held"] > 0.35 and dist < 560:
                b["dash"] = 0.42
                b["vx"], b["vy"] = 2.4 * D_SPEED * math.cos(aim), 2.4 * D_SPEED * math.sin(aim)
                self.sfx.append((self.t, "slash", 0))
            if b.get("dash", 0) > 0 and dist < 2 * D_BALL + 34 and b["cool"] <= 0:
                b["cool"] = 0.5
                b["dash"] = 0.0
                self.damage(o, dmg, (b["x"] + o["x"]) / 2, (b["y"] + o["y"]) / 2, b)
                self.effects.append({"k": "slashfx", "x": o["x"], "y": o["y"], "ang": aim, "life": 1.0})
                self._drop(b)
            elif b["held"] > 2.4:
                self._drop(b)                                # missed: it's gone
        elif kind == "slam":                                 # hammer
            if b["held"] > 0.3 and (dist < 330 or b["held"] > 2.2):
                self.effects.append({"k": "wave", "x": b["x"], "y": b["y"], "life": 1.0, "r": 330})
                self.sfx.append((self.t, "slam", 0))
                self.shake = max(self.shake, 0.5)
                for t in self._boom_targets(b):
                    dd = math.hypot(t["x"] - b["x"], t["y"] - b["y"])
                    if dd < 330 and self.damage(t, 2 if dd < 180 else 1, t["x"], t["y"], b) and t["alive"]:
                        self._fling(t, math.atan2(t["y"] - b["y"], t["x"] - b["x"]))
                self._drop(b)
        elif kind in ("throw", "bow"):
            b["wang"] += _ang_diff(aim, b["wang"]) * min(1, dt * 6)
            ready = b["held"] > (0.55 if kind == "bow" else 0.4)
            if ready and (abs(_ang_diff(aim, b["wang"])) < 0.25 or b["held"] > 1.6):
                a = b["wang"]
                x, y = b["x"] + math.cos(a) * (D_BALL + 40), b["y"] + math.sin(a) * (D_BALL + 40)
                sp = {"boomerang": 1050, "axe": 1100, "bow": 1900}[w]
                self.things.append({"k": w, "x": x, "y": y, "vx": sp * math.cos(a), "vy": sp * math.sin(a),
                                    "owner": b, "t0": self.t, "hit": set(), "spin": 0.0, "dmg": dmg})
                self.sfx.append((self.t, "twang" if kind == "bow" else "throw", 0))
                self._drop(b)
        elif kind == "chain":                                # lightning
            if b["held"] > 0.3:
                pts, last, hit, power = [(b["x"], b["y"])], b, [], dmg
                for _ in range(3):
                    cand = [t for t in self._boom_targets(b) if t["alive"] and t not in hit and
                            (last is b or math.hypot(t["x"] - last["x"], t["y"] - last["y"]) < 460)]
                    if not cand:
                        break
                    t = min(cand, key=lambda t: math.hypot(t["x"] - last["x"], t["y"] - last["y"]))
                    hit.append(t)
                    pts.append((t["x"], t["y"]))
                    self.damage(t, power, t["x"], t["y"], b)
                    power, last = 1, t
                self.effects.append({"k": "bolt", "pts": pts, "x": b["x"], "y": b["y"], "life": 1.0,
                                     "seed": int(self.t * 100)})
                self.sfx.append((self.t, "thunder", 0))
                self.shake = max(self.shake, 0.35)
                self._drop(b)
        elif kind == "flame":                                # flamethrower
            if b["held"] > 0.3:
                if b.get("fire", 0) <= 0:
                    b["fire"], b["burst_hits"] = FLAME_S, 0
                    self.sfx.append((self.t, "fire", 0))
                b["fire"] -= dt
                a = b["wang"]
                if self.rng.random() < 0.9:
                    sp = self.rng.uniform(500, 760)
                    aa = a + self.rng.uniform(-0.3, 0.3)
                    self.effects.append({"k": "flame", "x": b["x"] + math.cos(a) * (D_BALL + 70),
                                         "y": b["y"] + math.sin(a) * (D_BALL + 70), "vx": sp * math.cos(aa),
                                         "vy": sp * math.sin(aa), "life": 1.0, "r": self.rng.uniform(14, 24)})
                for t in self._boom_targets(b):
                    dd = math.hypot(t["x"] - b["x"], t["y"] - b["y"])
                    ang = math.atan2(t["y"] - b["y"], t["x"] - b["x"])
                    if (dd < 340 and abs(_ang_diff(ang, a)) < 0.4 and b["cool"] <= 0 and b["burst_hits"] < 3
                            and self.damage(t, 1, t["x"], t["y"], b)):
                        b["cool"], b["burst_hits"] = 0.4, b["burst_hits"] + 1
                if b["fire"] <= 0:
                    self._drop(b)
        elif kind == "magnet":
            if b["held"] > 0.3 and not o.get("pulled"):
                o["pulled"], o["puller"] = 1.4, b
                self.sfx.append((self.t, "magnet", 0))
            if o.get("pulled", 0) > 0 and dist < 2 * D_BALL + 12 and b["cool"] <= 0:
                b["cool"] = 0.6
                o["pulled"] = 0.0
                self.damage(o, dmg, (b["x"] + o["x"]) / 2, (b["y"] + o["y"]) / 2, b)
                self._drop(b)
            elif b["held"] > 2.2:
                o["pulled"] = 0.0
                self._drop(b)
        elif kind == "trap":                                 # banana
            if b["held"] > 0.3 and (dist < 340 or b["held"] > 1.6):
                self.things.append({"k": "peel", "x": b["x"], "y": b["y"], "owner": b, "t0": self.t,
                                    "ang": self.rng.uniform(0, 6.28)})
                self.sfx.append((self.t, "pickup", 0))
                b["weapon"] = None
        elif kind == "glove":
            if dist < 2 * D_BALL + 56 and b["cool"] <= 0:
                b["cool"] = 0.6
                if self.damage(o, dmg, (b["x"] + o["x"]) / 2, (b["y"] + o["y"]) / 2, b) and o["alive"]:
                    self._fling(o, aim)
                    o["slam_by"] = b
                self.sfx.append((self.t, "punch", 0))
                b["kick"] = 1.0
                b["uses"] -= 1
                if b["uses"] <= 0:
                    self._drop(b)
        elif kind == "pan":
            pass                                             # held up front: bats bullets back (bullet loop)
        elif kind == "summon":
            if b["held"] > 0.35:
                if w == "tornado":
                    v = self.rng.uniform(0, 6.28)
                    self.things.append({"k": "tornado", "x": b["x"] + math.cos(aim) * 150,
                                        "y": b["y"] + math.sin(aim) * 150, "vx": 340 * math.cos(v),
                                        "vy": 340 * math.sin(v), "owner": b, "t0": self.t, "life": 4.5,
                                        "cool": {}})
                    self.sfx.append((self.t, "wind", 0))
                elif w == "blackhole":
                    self.things.append({"k": "hole", "x": (b["x"] + o["x"]) / 2, "y": (b["y"] + o["y"]) / 2,
                                        "owner": b, "t0": self.t, "life": 2.4, "cool": {}})
                    self.sfx.append((self.t, "hole", 0))
                else:                                        # the Moai rolls in from behind
                    x, y = b["x"] - math.cos(aim) * 120, b["y"] - math.sin(aim) * 120
                    self.things.append({"k": "boulder", "x": x, "y": y, "vx": 980 * math.cos(aim),
                                        "vy": 980 * math.sin(aim), "owner": b, "t0": self.t, "hit": set(),
                                        "spin": 0.0, "dmg": dmg})
                    self.sfx.append((self.t, "rumble", 0))
                self._drop(b)
        elif kind == "potion":
            if b["held"] > 0.4:
                pick = self.rng.choice(("heal", "fast", "giant"))
                if pick == "heal":
                    b["hp"] = min(self.hearts, b["hp"] + 2)
                    self.effects.append({"k": "num", "x": b["x"], "y": b["y"] - 80, "text": "+2", "life": 1.0,
                                         "col": (60, 200, 90)})
                else:
                    b[pick] = 3.0
                    self.effects.append({"k": "num", "x": b["x"], "y": b["y"] - 80,
                                         "text": "SPEED!" if pick == "fast" else "GIANT!", "life": 1.0,
                                         "col": (150, 90, 230)})
                self.sfx.append((self.t, "gulp", 0))
                self._drop(b)

    def _fling(self, t, ang):
        t["knock"] = 0.55
        sp = 1.9 * D_SPEED
        t["vx"], t["vy"] = sp * math.cos(ang), sp * math.sin(ang)

    def _things(self, dt):
        for m in list(self.things):
            k = m["k"]
            age = self.t - m["t0"]
            gone = False
            if k in ("boomerang", "axe", "bow", "boulder"):
                if k == "boomerang" and age > 0.5:            # curves back to its owner
                    ow = m["owner"]
                    want = math.atan2(ow["y"] - m["y"], ow["x"] - m["x"])
                    sp = math.hypot(m["vx"], m["vy"])
                    cur = math.atan2(m["vy"], m["vx"])
                    cur += _ang_diff(want, cur) * min(1, dt * 7)
                    m["vx"], m["vy"] = sp * math.cos(cur), sp * math.sin(cur)
                    if math.hypot(ow["x"] - m["x"], ow["y"] - m["y"]) < D_BALL + 20 or age > 2.6:
                        gone = True
                m["x"] += m["vx"] * dt
                m["y"] += m["vy"] * dt
                m["spin"] += dt * (18 if k in ("boomerang", "axe") else 6)
                rad = 95 if k == "boulder" else 30
                leg = 1 if (k == "boomerang" and age > 0.5) else 0
                for t in self._boom_targets(m["owner"]):
                    key = (t["i"], leg)
                    if t["alive"] and key not in m["hit"] and math.hypot(t["x"] - m["x"], t["y"] - m["y"]) < D_BALL + rad:
                        m["hit"].add(key)
                        if self.damage(t, m["dmg"], m["x"], m["y"], m["owner"]) and k == "boulder" and t["alive"]:
                            self._fling(t, math.atan2(m["vy"], m["vx"]))
                        if k == "axe":
                            gone = True
                if k != "boomerang" and self._outside(m["x"], m["y"], 60 if k == "boulder" else 0):
                    gone = True
                    if k == "axe":
                        self.effects.append({"k": "spark", "x": m["x"], "y": m["y"], "life": 0.6,
                                             "col": (255, 220, 160)})
                if k == "boomerang" and self._outside(m["x"], m["y"], 40):
                    gone = True
            elif k == "peel":
                for t in self._boom_targets(m["owner"]):
                    if t["alive"] and t["frozen"] <= 0 and math.hypot(t["x"] - m["x"], t["y"] - m["y"]) < D_BALL + 26:
                        self.damage(t, 1, m["x"], m["y"], m["owner"])
                        t["frozen"], t["spin"] = 1.2, 1.2
                        self.sfx.append((self.t, "slip", 0))
                        gone = True
                        break
                if age > 7.0:
                    gone = True
            elif k == "tornado":
                m["x"] += m["vx"] * dt
                m["y"] += m["vy"] * dt
                if self._outside(m["x"], m["y"], -150):
                    ang = math.atan2(D_CY - m["y"], D_CX - m["x"]) + self.rng.uniform(-0.6, 0.6)
                    m["vx"], m["vy"] = 340 * math.cos(ang), 340 * math.sin(ang)
                for t in self._boom_targets(m["owner"]):
                    if t["alive"] and math.hypot(t["x"] - m["x"], t["y"] - m["y"]) < D_BALL + 70 and \
                            self.t >= m["cool"].get(t["i"], 0):
                        m["cool"][t["i"]] = self.t + 0.9
                        if self.damage(t, 1, t["x"], t["y"], m["owner"]) and t["alive"]:
                            self._fling(t, self.rng.uniform(0, 6.28))
                gone = age > m["life"]
            elif k == "hole":
                for t in self._boom_targets(m["owner"]):
                    if not t["alive"] or t["frozen"] > 0:
                        continue
                    dd = math.hypot(t["x"] - m["x"], t["y"] - m["y"])
                    if dd < 430:
                        want = math.atan2(m["y"] - t["y"], m["x"] - t["x"])
                        cur = math.atan2(t["vy"], t["vx"])
                        cur += _ang_diff(want, cur) * min(1, dt * 5)
                        sp = self._spd(t)
                        t["vx"], t["vy"] = sp * math.cos(cur), sp * math.sin(cur)
                    if dd < D_BALL + 30 and self.t >= m["cool"].get(t["i"], 0):
                        m["cool"][t["i"]] = self.t + 0.5
                        self.damage(t, 1, m["x"], m["y"], m["owner"])
                gone = age > m["life"]
            if gone:
                self.things.remove(m)

    def _status(self, dt):
        for b in self.balls:
            for key in ("dash", "fast", "giant", "small", "knock", "spin"):
                v = b.get(key, 0)
                if v > 0:
                    b[key] = max(0.0, v - dt)
                    if b[key] == 0 and key in ("dash", "fast", "knock"):
                        s = math.hypot(b["vx"], b["vy"]) or 1     # back to normal speed
                        sp = self._spd(b)
                        b["vx"], b["vy"] = b["vx"] / s * sp, b["vy"] / s * sp
            if b.get("pulled", 0) > 0 and b["alive"] and b["frozen"] <= 0:
                b["pulled"] = max(0.0, b["pulled"] - dt)
                pu = b["puller"]
                want = math.atan2(pu["y"] - b["y"], pu["x"] - b["x"])
                cur = math.atan2(b["vy"], b["vx"])
                cur += _ang_diff(want, cur) * min(1, dt * 8)
                sp = self._spd(b)
                b["vx"], b["vy"] = sp * math.cos(cur), sp * math.sin(cur)

    def _bombs(self):
        for m in list(self.bombs):
            if self.t - m["t0"] >= BOMB_FUSE:
                self.bombs.remove(m)
                self._boom(m["x"], m["y"], m["owner"])

    def _boom(self, x, y, owner):
        self.sfx.append((self.t, "boom", 0))
        self.shake = max(self.shake, 0.6)
        self.effects.append({"k": "boom", "x": x, "y": y, "life": 1.0, "col": (255, 170, 60)})
        for o in self._boom_targets(owner):
            dd = math.hypot(o["x"] - x, o["y"] - y)
            if dd < 120:
                self.damage(o, 3, x, y, owner)
            elif dd < 220:
                self.damage(o, 2, x, y, owner)

    def _effects(self, dt):
        for e in self.effects:
            k = e["k"]
            e["life"] -= dt * (1.6 if k in ("spark", "flash") else 1.3 if k == "smoke" else 3.0 if k == "beam" else
                               NEW_FX.get(k, 0.9))
            if k == "num":
                e["y"] -= 70 * dt
            elif k in ("ring", "ice"):
                e["x"], e["y"] = e["ball"]["x"], e["ball"]["y"]
                e["life"] -= dt * (1.2 if k == "ring" else 0.0)
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
            elif k == "flame":
                e["x"] += e["vx"] * dt
                e["y"] += e["vy"] * dt
                e["vx"] *= 0.93
                e["vy"] *= 0.93
                e["r"] += 60 * dt
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
        for k, pl in enumerate(players):
            if k:
                self.title.append(("VS ", (255, 255, 255)))
            self.title += _name_parts(pl, last=k > 0)
        self.heart_full = [_heart_sprite(60, c + (255,)) for c in self.cols]
        self.heart_empty = _heart_sprite(60, None, (70, 74, 100, 255))
        self.heart_lost = [_heart_sprite(60, tuple(int(v * 0.45) for v in c) + (255,)) for c in self.cols]
        self.guns = {w: _gun(w) for w in ("pistol", "shotgun", "minigun", "rocket")}
        self.knife = _emoji("1f52a", 118)
        self.shield = _emoji("1f6e1", 118)
        self.icons = {**self.guns, "knife": self.knife, "shield": self.shield}
        if recipe.get("arsenal") == 2:
            self.guns.update(laser=_gun("laser"), freeze=_gun("freeze"), flamethrower=_gun("flamethrower"))
            self.saw = _saw()
            self.bomb = _emoji("1f4a3", 104)
            self.snow = _emoji("2744", 52)
            self.star = _emoji("2b50", 34)
            self.icons.update(laser=self.guns["laser"], freeze=self.guns["freeze"], saw=self.saw, bomb=self.bomb,
                              flamethrower=self.guns["flamethrower"])
            for w, code in EMOJI_ARMS.items():
                self.icons[w] = _sticker(_emoji(code, 100), 5)
            self.moai = _emoji("1f5ff", 200)
            self.swirl = _emoji("1f300", 150)
            self.twister = _emoji("1f32a", 170)
            self.peel = _emoji("1f34c", 76)
            self.spin_pics = {w: _emoji(EMOJI_ARMS[w], 92) for w in ("boomerang", "axe")}
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
        # where draw_match puts the arena: the duel's own place (plain), or a
        # smaller arena somewhere else (the tournament: TourneyPainter.place)
        self.plain, self.k, self.cx, self.cy = True, 1.0, D_CX, D_CY
        self.scaled: dict = {}
        self.fonts: dict = {}
        self.end_y = D_CY + 230
        self.clean = recipe.get("style") == "clean"
        self.arena = recipe.get("arena") or "neon"
        self.ball_ring = None                 # the balls' outline (the hand-drawn arenas: ink / chalk)
        self.ink, self.ink_stroke = (255, 255, 255), (7, 9, 24)
        self.trail_col = None                 # None: each ball's own colour
        self.hand_font = None
        if self.clean:
            self.bg = _arena_bg(D_CX, C_CY, int(D_R * C_K)) if self.arena == "neon" else self.arena_bg()
            self.place(D_CX, C_CY, C_K)
            self.end_y = C_CY + 250
            self.trail_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            self.prev: dict = {}
            self.bar_shown: dict = {}       # ball -> the hp its bar shows (drains after a hit)
            hf = self.hand_font
            big = 1.18 if hf and hf.startswith("Gochi") else 1.0      # Gochi Hand runs small
            self.f_name = _font(int((50 if hf else 46) * big), hf or "Inter-Black.ttf")
            self.f_name_s = _font(int((34 if hf else 32) * big), hf or "Inter-Black.ttf")
            self.f_vs = _font(50)
            self.f_hook = _font(int((58 if hf else 52) * big), hf or "Inter-Black.ttf")

    # -- placing a smaller arena (the tournament); plain = the duel, untouched
    def place(self, cx, cy, k):
        self.plain, self.k, self.cx, self.cy = False, k, cx, cy

    def P(self, x, y):
        if self.plain:
            return x, y
        return self.cx + (x - D_CX) * self.k, self.cy + (y - D_CY) * self.k

    def sl(self, v):
        """A length."""
        return v if self.plain else v * self.k

    def sw(self, v):
        """A line width."""
        return v if self.plain else max(1, int(round(v * self.k)))

    def img(self, key, im):
        if self.plain:
            return im
        k2 = (key, round(self.k, 4))
        out = self.scaled.get(k2)
        if out is None:
            out = self.scaled[k2] = im.resize((max(2, int(im.width * self.k)), max(2, int(im.height * self.k))),
                                              Image.LANCZOS)
        return out

    def font(self, f):
        if self.plain:
            return f
        size = max(10, int(f.size * self.k))
        out = self.fonts.get(size)
        if out is None:
            out = self.fonts[size] = _font(size)
        return out

    def ball_sprite(self, b, look, mood, blink, flash, cracks):
        r = D_BALL if self.plain else max(8, int(round(D_BALL * self.k)))
        q = 1 if self.plain else r / D_BALL
        key = (b["i"], b["col"], b.get("flag"), b.get("ring"), r, round(look / (math.pi / 12)), mood, blink, flash,
               cracks)
        if key in self.cache:
            return self.cache[key]
        S3 = 3
        big = (r * 2 + 8) * S3
        c = big / 2
        R = r * S3
        im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        col = (255, 255, 255) if flash else b["col"]
        ring = tuple(b.get("ring") or self.ball_ring or (255, 255, 255)) + (255,)   # 2v2: the team's colour
        rw = max(1, int((8 if b.get("ring") else 5) * S3 * q))
        if b.get("flag") and not flash:
            im.alpha_composite(_flag(b["flag"], int(2 * R)), (int(c - R), int(c - R)))
            d.ellipse((c - R, c - R, c + R, c + R), outline=ring, width=rw)
        else:
            d.ellipse((c - R, c - R, c + R, c + R), fill=col + (255,), outline=ring, width=rw)
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
            d.line(pts, fill=(30, 30, 50, 200), width=max(1, int(2.4 * S3 * q)))
        ink = (20, 20, 30, 255)
        ex, ey = math.cos(look), math.sin(look)
        for side in (-1, 1):
            cx_e, cy_e = c + side * R * 0.32 + ex * R * 0.08, c - 0.18 * R + ey * R * 0.06
            er = R * (0.27 if mood == "scared" else 0.22)
            if mood == "dead" or flash:
                w_ = er * 0.8
                d.line([(cx_e - w_, cy_e - w_), (cx_e + w_, cy_e + w_)], fill=ink, width=max(1, int(4 * S3 * q)))
                d.line([(cx_e - w_, cy_e + w_), (cx_e + w_, cy_e - w_)], fill=ink, width=max(1, int(4 * S3 * q)))
                continue
            if blink:
                d.line([(cx_e - er, cy_e), (cx_e + er, cy_e)], fill=ink, width=max(1, int(4 * S3 * q)))
                continue
            d.ellipse((cx_e - er, cy_e - er, cx_e + er, cy_e + er), fill=(255, 255, 255, 255), outline=ink,
                      width=max(1, int(2.5 * S3 * q)))
            pr = er * (0.42 if mood == "scared" else 0.55)
            px, py = cx_e + ex * er * 0.4, cy_e + ey * er * 0.4
            d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=ink)
            d.ellipse((px - pr * 0.05, py - pr * 0.65, px + pr * 0.65, py - pr * 0.05), fill=(255, 255, 255, 255))
            if mood == "angry":
                # outer end high, inner end low: a frown
                d.line([(cx_e + side * er * 1.1, cy_e - er * 1.5), (cx_e - side * er * 0.9, cy_e - er * 0.85)],
                       fill=ink, width=max(1, int(5 * S3 * q)))
            elif mood == "scared":
                # inner end high: worried
                d.line([(cx_e + side * er * 1.0, cy_e - er * 1.05), (cx_e - side * er * 0.9, cy_e - er * 1.55)],
                       fill=ink, width=max(1, int(4 * S3 * q)))
        my = c + R * 0.38
        if mood == "happy":
            d.arc((c - R * 0.3, my - R * 0.25, c + R * 0.3, my + R * 0.12), 20, 160, fill=ink,
                  width=max(1, int(4 * S3 * q)))
        elif mood == "angry":
            d.arc((c - R * 0.28, my - R * 0.02, c + R * 0.28, my + R * 0.3), 200, 340, fill=ink,
                  width=max(1, int(4 * S3 * q)))
        elif mood == "scared":
            pts = [(c - R * 0.28 + k * R * 0.07, my + (R * 0.05 if k % 2 else -R * 0.03)) for k in range(9)]
            d.line(pts, fill=ink, width=max(1, int(3.5 * S3 * q)))
        elif mood == "dead":
            d.ellipse((c - R * 0.12, my - R * 0.08, c + R * 0.12, my + R * 0.16), fill=ink)
        sp = im.resize((r * 2 + 8, r * 2 + 8), Image.LANCZOS)
        if len(self.cache) > 1500:
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
        if b["hp"] <= 4 * sim.hearts / HEARTS:
            return "scared"
        return "happy"

    def frame(self, sim: DuelSim) -> Image.Image:
        im = self.bg.copy().convert("RGBA")
        if self.clean:
            self.trails(im, sim)
        d = ImageDraw.Draw(im)
        if self.clean:
            self.hud_clean(im, d, sim)
        else:
            self.hud(im, d, sim)
        self.draw_match(im, d, sim)
        self.watermark(d)
        if self.clean and sim.done_at is not None and sim.t - sim.done_at < 0.25:
            a = int(110 * (1 - (sim.t - sim.done_at) / 0.25))          # the knockout: a white flash
            im.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, a)))
        out = im.convert("RGB")
        if sim.shake > 0:
            amp = int(14 * min(1.0, sim.shake))
            rng = random.Random(int(sim.t * 1000))
            out = Image.fromarray(np.roll(np.asarray(out), (rng.randint(-amp, amp), rng.randint(-amp, amp)), (0, 1)))
        return out

    def draw_match(self, im, d, sim: DuelSim, ending: bool = True):
        """Everything inside the arena (where ``place`` put it)."""
        P, sl, sw = self.P, self.sl, self.sw
        for p in sim.pickups:
            age = sim.t - p["born"]
            s = min(1.0, age / 0.25)
            rr = sl(70) * s
            bob = math.sin(age * 4) * sl(6)
            x, y = P(p["x"], p["y"])
            d.ellipse((x - rr, y - rr + bob, x + rr, y + rr + bob), outline=(150, 150, 165, 255),
                      width=sw(7))
            if s > 0.2:
                ic = self.icons[p["w"]]
                f = min(1.0, 124 / max(ic.size)) * s * (1 if self.plain else self.k)
                ic = ic.resize((max(4, int(ic.width * f)), max(4, int(ic.height * f))), Image.LANCZOS)
                im.alpha_composite(ic, (int(x - ic.width / 2), int(y - ic.height / 2 + bob)))
        for e in sim.effects:
            if e["k"] == "smoke":
                L = max(0.0, e["life"])
                r = sl(e["r"])
                x, y = P(e["x"], e["y"])
                d.ellipse((x - r, y - r, x + r, y + r), fill=(190, 190, 200, int(120 * L)))
        for u in sim.bullets:
            sp = math.hypot(u["vx"], u["vy"])
            ln = 34 if u["kind"] == "shotgun" else 46
            tx, ty = P(u["x"] - u["vx"] / sp * ln, u["y"] - u["vy"] / sp * ln)
            x, y = P(u["x"], u["y"])
            if u["kind"] == "ice":
                d.line([(tx, ty), (x, y)], fill=(150, 220, 255, 170), width=sw(12))
                rr = sl(14)
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(220, 245, 255, 255), outline=(110, 190, 250, 255),
                          width=sw(4))
                continue
            d.line([(tx, ty), (x, y)], fill=(255, 210, 90, 170), width=sw(7))
            rr = sl(6 if u["kind"] == "shotgun" else 8)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(255, 245, 200, 255))
        for m in getattr(sim, "bombs", ()):
            age = sim.t - m["t0"]
            x, y = P(m["x"], m["y"])
            # the blast zone, pulsing faster as the fuse runs down
            pulse = 0.5 + 0.5 * math.sin(age * (8 + 30 * age / BOMB_FUSE))
            rr = sl(220)
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(255, 60, 60, int(26 + 30 * pulse)),
                      outline=(255, 80, 80, int(120 + 100 * pulse)), width=sw(4))
            g = self.img("bomb", self.bomb)
            s2 = 1 + 0.12 * pulse
            g = g.resize((int(g.width * s2), int(g.height * s2)), Image.LANCZOS)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
        for m in getattr(sim, "things", ()):
            self.thing(im, d, sim, m)
        for b in sim.balls:                           # the magnet's pull: a line between them
            if b.get("pulled", 0) > 0 and b["alive"]:
                pu = b["puller"]
                x0, y0 = P(pu["x"], pu["y"])
                x1, y1 = P(b["x"], b["y"])
                n = 14
                for j in range(n):
                    if j % 2 == 0:
                        d.line([(x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * j / n),
                                (x0 + (x1 - x0) * (j + 1) / n, y0 + (y1 - y0) * (j + 1) / n)],
                               fill=(230, 60, 80, 220) if j % 4 == 0 else (120, 160, 255, 220), width=sw(8))
        for r in sim.rockets:
            rot = self.img("rocket", self.rocket).rotate(-math.degrees(r["a"]), resample=Image.BICUBIC, expand=True)
            x, y = P(r["x"], r["y"])
            im.alpha_composite(rot, (int(x - rot.width / 2), int(y - rot.height / 2)))
        for b in sim.balls:
            if not b["alive"]:
                continue
            o = sim.other(b)
            look = math.atan2(o["y"] - b["y"], o["x"] - b["x"])
            blink = (int(sim.t * 10 + b["i"] * 17) % 37) == 0
            flash = b["hurt"] > 0.18
            cracks = min(4, (sim.hearts - b["hp"]) * HEARTS // sim.hearts // 2)
            sp = self.ball_sprite(b, look, self.mood(sim, b), blink, flash, cracks)
            if sim.winner is b:
                scale = 1 + min(0.5, (sim.t - sim.done_at) * 0.4)
            else:
                scale = 1 + 0.25 * b["squash"]
            if b.get("small", 0) > 0:
                scale *= 0.65                         # the wand
            if b.get("giant", 0) > 0:
                scale *= 1.45                         # the potion
            if b.get("fast", 0) > 0 or b.get("dash", 0) > 0:
                sp_ = math.hypot(b["vx"], b["vy"]) or 1
                ux, uy = -b["vx"] / sp_, -b["vy"] / sp_
                for j in (-1, 0, 1):
                    ox, oy = -uy * j * 30, ux * j * 30
                    x0, y0 = P(b["x"] + ux * 80 + ox, b["y"] + uy * 80 + oy)
                    x1, y1 = P(b["x"] + ux * 190 + ox, b["y"] + uy * 190 + oy)
                    d.line([(x0, y0), (x1, y1)], fill=(255, 255, 255, 170), width=sw(6))
            if abs(scale - 1) > 0.01:
                sp = sp.resize((int(sp.width * scale), int(sp.height * scale)), Image.LANCZOS)
            x, y = P(b["x"], b["y"])
            im.alpha_composite(sp, (int(x - sp.width / 2), int(y - sp.height / 2)))
            if b.get("spin", 0) > 0:                   # slipped on a banana: stars go round
                for j in range(3):
                    ang = sim.t * 9 + j * 2.09
                    g = self.img("star", self.star)
                    sx, sy = P(b["x"] + math.cos(ang) * 82, b["y"] - 70 + math.sin(ang) * 28)
                    im.alpha_composite(g, (int(sx - g.width / 2), int(sy - g.height / 2)))
            elif b.get("frozen", 0) > 0:
                rr = sl(D_BALL + 6)
                a = int(150 * min(1.0, b["frozen"] / 0.3))
                d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=(170, 225, 255, a // 2),
                          outline=(225, 248, 255, a), width=sw(6))
                g = self.img("snow", self.snow)
                im.alpha_composite(g, (int(x + rr * 0.55 - g.width / 2), int(y - rr * 0.75 - g.height / 2)))
            self.held(im, b)
        self.effects(im, d, sim)
        if ending and sim.winner is not None:
            self.ending(im, d, sim)

    # -- the clean look
    def trails(self, im, sim):
        a = np.asarray(self.trail_layer).copy()
        a[..., 3] = (a[..., 3] * 0.86).astype(np.uint8)
        self.trail_layer = Image.fromarray(a)
        td = ImageDraw.Draw(self.trail_layer)
        for b in sim.balls:
            p = self.prev.get(b["i"])
            x, y = self.P(b["x"], b["y"])
            if p and b["alive"] and b.get("frozen", 0) <= 0:
                col = self.trail_col or b["col"] + (170,)
                wd = int(D_BALL * C_K * (0.18 if self.arena == "notebook" else 0.32))
                td.line([p, (x, y)], fill=col, width=wd)
            self.prev[b["i"]] = (x, y)
        im.alpha_composite(self.trail_layer)

    def hud_clean(self, im, d, sim):
        """Fighting-game health bars: one side per fighter (2v2: per team,
        a bar per ball), a VS badge between them, the question under."""
        teams = "team" in sim.balls[0]
        sides = [[b for b in sim.balls if (b["team"] if teams else b["i"]) == k] for k in (0, 1)]
        top = 150
        for k, side in enumerate(sides):
            x0, x1 = (32, W // 2 - 66) if k == 0 else (W // 2 + 66, W - 32)
            if teams:
                names = self.recipe.get("team_names") or ["", ""]
                col = side[0].get("ring") or (255, 255, 255)
                d.text((x0 if k == 0 else x1, top - 30), names[k].upper(), font=self.f_name_s, fill=col + (255,),
                       anchor="ls" if k == 0 else "rs")
            rows = len(side)
            for j, b in enumerate(side):
                big = rows == 1
                size = 120 if big else 78
                y = top + (j * 104 if not big else 0)
                px = x0 if k == 0 else x1 - size
                self.portrait(im, d, b, px, y, size)
                bx0, bx1 = (px + size + 16, x1) if k == 0 else (x0, px - 16)
                bh = 42 if big else 28
                by = y + size - bh - (8 if big else 2)
                name = b["name"].upper()
                f = self.f_name if big else self.f_name_s
                while d.textlength(name, font=f) > bx1 - bx0 and f.size > 18:
                    f = _font(f.size - 2, self.hand_font or "Inter-Black.ttf")
                d.text((bx0 if k == 0 else bx1, by - 10), name, font=f, fill=self.ink + (255,),
                       anchor="ls" if k == 0 else "rs")
                self.bar(d, b, sim, bx0, by, bx1, by + bh, mirror=k == 1)
        # VS badge
        cy = top + (60 if not teams else 90)
        d.ellipse((W / 2 - 52, cy - 52, W / 2 + 52, cy + 52), fill=(230, 40, 64, 255), outline=(255, 255, 255, 255),
                  width=6)
        d.text((W / 2, cy + 2), "VS", font=self.f_vs, fill=(255, 255, 255, 255), anchor="mm")
        lg = self.recipe.get("league")
        if lg:
            self.league_hud(im, d, lg, top)
        hook = self.recipe.get("hook") or ""
        f = self.f_hook
        while d.textlength(hook, font=f) > W - 80 and f.size > 28:
            f = _font(f.size - 2, self.hand_font or "Inter-Black.ttf")
        d.text((W / 2, 450 if lg else 440 if teams else 410), hook, font=f, fill=self.ink + (255,), anchor="mm",
               stroke_width=4, stroke_fill=self.ink_stroke + (255,))

    LEAGUE_TAG_COL = {"REMATCH": (230, 40, 64), "NEW CHALLENGER": (40, 150, 240), "TOP CONTENDER": (150, 80, 230),
                      "YOUR PICK": (40, 170, 90)}

    def league_hud(self, im, d, lg, top):
        """Ball League (ball_league): each fighter's record under their bar,
        a crown on the champion, the fight's tag + number in a pill."""
        gold = (196, 132, 0) if self.arena == "notebook" else (255, 200, 50)     # paper: a darker gold
        f = self.fonts.get("rec")
        if f is None:
            f = self.fonts["rec"] = _font(30 if not self.hand_font else 34, self.hand_font or "Inter-Black.ttf")
        y = top + 120 + 34
        for k in (0, 1):
            w, l = lg["records"][k]
            champ = k == 0 and lg.get("champ")
            txt = (f"CHAMPION  {w}-{l}" if champ else f"{w}-{l}") if (w or l or champ) else "NEW"
            col = gold if champ else self.ink
            d.text((32 if k == 0 else W - 32, y), txt, font=f, fill=col + (255,), anchor="lm" if k == 0 else "rm",
                   stroke_width=3, stroke_fill=self.ink_stroke + (255,))
        if lg.get("champ"):
            c = self.scaled.get("crown")
            if c is None:
                c = self.scaled["crown"] = _emoji("1f451", 70).rotate(18, resample=Image.BICUBIC, expand=True)
            im.alpha_composite(c, (32 - 18, top - 46))
        pill = f"{lg['tag']}  ·  FIGHT #{lg['no']}"
        fp = self.fonts.get("pill")
        if fp is None:
            fp = self.fonts["pill"] = _font(32)
        tw = d.textlength(pill, font=fp)
        cy = 362
        d.rounded_rectangle((W / 2 - tw / 2 - 26, cy - 28, W / 2 + tw / 2 + 26, cy + 28), 28,
                            fill=self.LEAGUE_TAG_COL.get(lg["tag"], (230, 40, 64)) + (255,),
                            outline=(255, 255, 255, 255), width=4)
        d.text((W / 2, cy + 1), pill, font=fp, fill=(255, 255, 255, 255), anchor="mm")

    def portrait(self, im, d, b, x, y, size):
        ring = tuple(b.get("ring") or (255, 255, 255))
        if b.get("flag"):
            im.alpha_composite(_flag(b["flag"], size), (int(x), int(y)))
        else:
            d.ellipse((x, y, x + size, y + size), fill=b["col"] + (255,))
        d.ellipse((x, y, x + size, y + size), outline=ring + (255,), width=5)
        if not b["alive"]:
            d.ellipse((x, y, x + size, y + size), fill=(10, 12, 26, 170))
            q = size * 0.28
            cx, cy = x + size / 2, y + size / 2
            d.line([(cx - q, cy - q), (cx + q, cy + q)], fill=(255, 70, 80, 255), width=8)
            d.line([(cx - q, cy + q), (cx + q, cy - q)], fill=(255, 70, 80, 255), width=8)

    def bar(self, d, b, sim, x0, y0, x1, y1, mirror):
        """A segmented health bar: a segment per heart, the ones just lost
        turn red and drain away."""
        n = sim.hearts
        shown = self.bar_shown.get(b["i"], float(n))
        shown = max(float(b["hp"]), shown - 3.0 / FPS)          # drains ~3 hearts a second
        self.bar_shown[b["i"]] = shown
        d.rounded_rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), 10, fill=(14, 16, 36, 255),
                            outline=(255, 255, 255, 60), width=2)
        gap = 4
        seg = (x1 - x0 - gap * (n - 1)) / n
        for s in range(n):
            k = n - 1 - s if mirror else s
            sx = x0 + k * (seg + gap)
            if s < b["hp"]:
                col = b["col"] + (255,)
            elif s < shown:
                col = (255, 70, 80, 255)          # just lost: red, then it drains
            else:
                col = (44, 48, 76, 255)
            d.rounded_rectangle((sx, y0, sx + seg, y1), 5, fill=col)

    def thing(self, im, d, sim, m):
        """A thrown / summoned thing (the newer weapons)."""
        P, sl, sw = self.P, self.sl, self.sw
        k = m["k"]
        x, y = P(m["x"], m["y"])
        age = sim.t - m["t0"]
        if k in ("boomerang", "axe"):
            g = self.img(k + "-spin", self.spin_pics[k]).rotate(-math.degrees(m["spin"]) % 360, resample=Image.BICUBIC,
                                                               expand=True)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
        elif k == "bow":
            a = math.atan2(m["vy"], m["vx"])
            tx, ty = P(m["x"] - math.cos(a) * 90, m["y"] - math.sin(a) * 90)
            d.line([(tx, ty), (x, y)], fill=(120, 80, 40, 255), width=sw(7))
            hx, hy = math.cos(a), math.sin(a)
            q = sl(22)
            d.polygon([(x + hx * q, y + hy * q), (x - hy * q * 0.6, y + hx * q * 0.6), (x + hy * q * 0.6, y - hx * q * 0.6)],
                      fill=(200, 205, 215, 255))
            for f in (-1, 1):
                d.line([(tx, ty), (tx - hx * q + f * hy * q * 0.7, ty - hy * q - f * hx * q * 0.7)],
                       fill=(230, 60, 60, 255), width=sw(5))
        elif k == "boulder":
            g = self.img("moai", self.moai).rotate(-math.degrees(m["spin"]) % 360, resample=Image.BICUBIC, expand=True)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
        elif k == "peel":
            g = self.img("peel", self.peel).rotate(math.degrees(m["ang"]), resample=Image.BICUBIC, expand=True)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
        elif k == "tornado":
            r = sl(110)
            for j in range(3):
                a0 = (age * 400 + j * 120) % 360
                d.arc((x - r, y - r * 0.5 + sl(40), x + r, y + r * 0.5 + sl(40)), a0, a0 + 70,
                      fill=(200, 200, 210, 160), width=sw(6))
            s2 = 1 + 0.06 * math.sin(age * 20)
            g = self.img("twister", self.twister)
            g = g.resize((int(g.width * s2), int(g.height * s2)), Image.LANCZOS)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
        elif k == "hole":
            grow = min(1.0, age / 0.3) * min(1.0, (m["life"] - age) / 0.3)
            r = sl(170) * grow
            d.ellipse((x - r, y - r, x + r, y + r), fill=(120, 60, 200, 60))
            for j in range(4):                         # rings pulled inward
                f = ((age * 1.5 + j / 4) % 1)
                rr = r * (1 - f)
                d.ellipse((x - rr, y - rr, x + rr, y + rr), outline=(190, 140, 255, int(160 * f)), width=sw(4))
            g = self.img("swirl", self.swirl).rotate(age * 400 % 360, resample=Image.BICUBIC, expand=True)
            g = g.resize((max(4, int(g.width * grow)), max(4, int(g.height * grow))), Image.LANCZOS)
            im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
            c = sl(40) * grow
            d.ellipse((x - c, y - c, x + c, y + c), fill=(0, 0, 0, 255))

    # -- the arenas (Dean picked notebook, chalkboard, ice rink, boxing ring)
    def arena_bg(self) -> Image.Image:
        cx, cy, r = D_CX, C_CY, D_R * C_K
        rng = np.random.default_rng(3)
        if self.arena == "notebook":
            im = Image.new("RGBA", (W, H), (250, 247, 236, 255))
            d = ImageDraw.Draw(im)
            for y in range(70, H, 52):
                d.line([(0, y), (W, y)], fill=(170, 196, 232, 255), width=3)
            d.line([(112, 0), (112, H)], fill=(235, 120, 120, 255), width=3)
            grain = Image.fromarray((rng.random((H, W)) * 40).astype(np.uint8))
            im = Image.composite(Image.new("RGBA", im.size, (226, 220, 202, 255)), im, grain.point(lambda v: v * 2))
            d = ImageDraw.Draw(im)
            _wobble(d, lambda t, j, p: (cx + r * (1 + j) * math.cos(t * 6.283) + p, cy + r * (1 + j) * math.sin(t * 6.283)),
                    (40, 40, 48, 255), 8, seed=4, passes=3, amp=0.006)
            self.ball_ring, self.ink, self.ink_stroke = (28, 28, 34), (30, 30, 36), (250, 247, 236)
            self.trail_col = (70, 70, 84, 150)
            self.hand_font = "PermanentMarker-Regular.woff"
            return im
        if self.arena == "chalk":
            base = np.zeros((H, W, 3), np.float32) + (34, 60, 46)
            base += rng.normal(0, 5, (H, W, 1))
            im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
            smudge = Image.new("RGBA", im.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(smudge)
            for _ in range(8):
                x, y = rng.uniform(0, W), rng.uniform(0, H)
                sd.ellipse((x - 260, y - 120, x + 260, y + 120), fill=(255, 255, 255, 16))
            im.alpha_composite(smudge.filter(ImageFilter.GaussianBlur(60)))
            d = ImageDraw.Draw(im)
            _wobble(d, lambda t, j, p: (cx + r * (1 + j) * math.cos(t * 6.283) + p, cy + r * (1 + j) * math.sin(t * 6.283)),
                    (232, 238, 228, 230), 10, seed=9, passes=2, amp=0.008)
            self.ball_ring = (236, 240, 232)
            self.trail_col = (230, 236, 226, 70)
            self.hand_font = "GochiHand-Regular.woff"
            return im
        im = Image.new("RGBA", (W, H), (12, 14, 26, 255))
        d = ImageDraw.Draw(im)
        bx, by, hw, hh, cr = ARENA_SHAPES[self.arena]
        X, Y = D_CX, C_CY + (by - D_CY) * C_K       # (the painter isn't placed yet: map by hand)
        hw, hh, cr = hw * C_K, hh * C_K, cr * C_K
        if self.arena == "ice":
            d.rounded_rectangle((X - hw - 14, Y - hh - 14, X + hw + 14, Y + hh + 14), cr + 14, fill=(230, 60, 70, 255))
            d.rounded_rectangle((X - hw, Y - hh, X + hw, Y + hh), cr, fill=(226, 240, 250, 255))
            ice = Image.new("RGBA", im.size, (0, 0, 0, 0))
            idr = ImageDraw.Draw(ice)
            r2 = random.Random(4)
            for _ in range(70):                       # skate scratches
                sx, sy = r2.uniform(X - hw + 40, X + hw - 40), r2.uniform(Y - hh + 40, Y + hh - 40)
                rr = r2.uniform(60, 260)
                a0 = r2.uniform(0, 360)
                idr.arc((sx - rr, sy - rr, sx + rr, sy + rr), a0, a0 + r2.uniform(15, 60), fill=(180, 206, 228, 200),
                        width=3)
            m = Image.new("L", im.size, 0)
            ImageDraw.Draw(m).rounded_rectangle((X - hw, Y - hh, X + hw, Y + hh), cr, fill=255)
            im.paste(ice, (0, 0), Image.fromarray(np.minimum(np.asarray(m), np.asarray(ice.getchannel("A")))))
            d = ImageDraw.Draw(im)
            d.line([(X - hw + 8, Y), (X + hw - 8, Y)], fill=(225, 60, 70, 255), width=10)
            d.ellipse((X - 110, Y - 110, X + 110, Y + 110), outline=(70, 110, 220, 255), width=8)
            for yy in (Y - hh * 0.45, Y + hh * 0.45):
                d.line([(X - hw + 8, yy), (X + hw - 8, yy)], fill=(70, 110, 220, 255), width=8)
            self.trail_col = (150, 186, 220, 170)
            self.ball_ring = (255, 255, 255)
            return im
        # boxing
        r2 = random.Random(2)
        for _ in range(1800):                         # the crowd
            x, y = r2.uniform(0, W), r2.uniform(480, H)
            c = r2.choice([(56, 56, 76), (78, 68, 60), (48, 58, 70), (88, 88, 108), (70, 50, 60)])
            d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=c + (255,))
        d.rectangle((X - hw - 26, Y - hh - 26, X + hw + 26, Y + hh + 26), fill=(18, 18, 28, 255))
        d.rectangle((X - hw, Y - hh, X + hw, Y + hh), fill=(64, 104, 186, 255))
        for k, col in enumerate(((230, 50, 60), (245, 245, 245), (60, 100, 220))):
            o = 6 + k * 16
            d.rectangle((X - hw + o, Y - hh + o, X + hw - o, Y + hh - o), outline=col + (255,), width=7)
        for (px, py), col in zip(((X - hw, Y - hh), (X + hw, Y - hh), (X - hw, Y + hh), (X + hw, Y + hh)),
                                 ((230, 50, 60), (60, 100, 220), (60, 100, 220), (230, 50, 60))):
            d.rectangle((px - 26, py - 26, px + 26, py + 26), fill=col + (255,), outline=(255, 255, 255, 255), width=5)
        self.trail_col = None
        return im

    def held(self, im, b):
        w = b["weapon"]
        if not w:
            return
        a = b["wang"]
        flip = False
        if w == "knife":
            img, dist, rot = self.knife, D_BALL + 34, -math.degrees(a) - 45
        elif w == "shield":
            img, dist, rot = self.shield, D_BALL + 20, 0
        elif w == "saw":
            img, dist, rot = self.saw, D_BALL + 74, -math.degrees(a) * 4
        elif w == "bomb":
            img, dist, rot = self.bomb, D_BALL + 30, -math.degrees(a) * 0.3
        elif w in EMOJI_ARMS:
            img = self.icons[w]
            base = HELD_ANGLE.get(w)
            rot = 0 if base is None else -(math.degrees(a) - base)
            dist = D_BALL + img.width * 0.38 - b["kick"] * 14
        else:
            img = self.guns[w]
            if math.cos(a) < 0:              # keep the gun upright on the left side
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                flip = True
            kick = b["kick"] * 14
            dist = D_BALL + img.width * 0.42 - kick
            rot = -math.degrees(a)
        x, y = self.P(b["x"] + math.cos(a) * dist, b["y"] + math.sin(a) * dist)
        r_img = self.img((w, flip), img).rotate(rot, resample=Image.BICUBIC, expand=True)
        im.alpha_composite(r_img, (int(x - r_img.width / 2), int(y - r_img.height / 2)))
        if w == "laser" and b.get("charge", 0) > 0:
            # the sight line while it charges, brighter as it gets close
            k = min(1.0, b["charge"] / LASER_CHARGE)
            mx, my = b["x"] + math.cos(a) * (D_BALL + 150), b["y"] + math.sin(a) * (D_BALL + 150)
            ex, ey = b["x"] + math.cos(a) * 1500, b["y"] + math.sin(a) * 1500
            d = ImageDraw.Draw(im)
            d.line([self.P(mx, my), self.P(ex, ey)], fill=(255, 40, 60, int(60 + 150 * k)),
                   width=self.sw(3 + 4 * k))
            gx, gy = self.P(b["x"] + math.cos(a) * (D_BALL + 182), b["y"] + math.sin(a) * (D_BALL + 182))
            gr = self.sl(10 + 16 * k)
            d.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=(255, 120, 140, int(140 + 100 * k)))

    def hud(self, im, d, sim):
        for i, b in enumerate(sim.balls):
            y = 110 + i * 150
            if b.get("flag"):
                im.alpha_composite(_flag(b["flag"], 96), (30, y - 48))
                d.ellipse((30, y - 48, 126, y + 48), outline=(255, 255, 255, 255), width=5)
            else:
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
        P, sl, sw = self.P, self.sl, self.sw
        for e in sim.effects:
            L = max(0.0, e["life"])
            k = e["k"]
            ex, ey = P(e["x"], e["y"])
            if k == "num":
                a = int(255 * min(1, L * 2))
                d.text((ex, ey), e["text"], font=self.font(self.f_num), fill=(255, 255, 255, a), anchor="mm",
                       stroke_width=sw(6), stroke_fill=e["col"] + (a,))
            elif k == "spark":
                for j in range(8):
                    ang = j * math.pi / 4 + 0.3
                    r0, r1 = sl(10 + (1 - L) * 40), sl(22 + (1 - L) * 70)
                    d.line([(ex + math.cos(ang) * r0, ey + math.sin(ang) * r0),
                            (ex + math.cos(ang) * r1, ey + math.sin(ang) * r1)],
                           fill=e["col"] + (int(255 * L),), width=sw(5))
            elif k == "flash":
                r = sl(30 * L + 8)
                d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=e["col"] + (int(230 * L),))
            elif k == "ring":
                r = sl(D_BALL + 8 + (1 - L) * 50)
                d.ellipse((ex - r, ey - r, ex + r, ey + r), outline=e["col"] + (int(200 * L),), width=sw(6))
            elif k == "wave":
                r = sl(60 + (e["r"] - 60) * (1 - L))
                d.ellipse((ex - r, ey - r, ex + r, ey + r), outline=(255, 255, 255, int(230 * L)), width=sw(10))
                r2 = r * 0.82
                d.ellipse((ex - r2, ey - r2, ex + r2, ey + r2), outline=(255, 220, 120, int(150 * L)), width=sw(5))
            elif k == "bolt":
                rng = random.Random(e["seed"] + int(sim.t * 30))
                pts = [P(*p) for p in e["pts"]]
                path = [pts[0]]
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    nx, ny = -(y1 - y0), x1 - x0
                    ln = math.hypot(nx, ny) or 1
                    for j in range(1, 9):
                        o = rng.uniform(-1, 1) * sl(26)
                        path.append((x0 + (x1 - x0) * j / 9 + nx / ln * o, y0 + (y1 - y0) * j / 9 + ny / ln * o))
                    path.append((x1, y1))
                d.line(path, fill=(255, 230, 90, int(140 * L)), width=sw(22), joint="curve")
                d.line(path, fill=(255, 255, 255, int(255 * L)), width=sw(7), joint="curve")
            elif k == "flame":
                r = sl(e["r"])
                col = (255, int(120 + 120 * L), int(40 * L), int(220 * L))
                d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=col)
            elif k == "slashfx":
                a0 = math.degrees(e["ang"]) - 60
                r = sl(110 + (1 - L) * 30)
                d.arc((ex - r, ey - r, ex + r, ey + r), a0, a0 + 120, fill=(255, 255, 255, int(255 * L)), width=sw(12))
            elif k == "beam":
                x1, y1 = P(e["x1"], e["y1"])
                for wd, col in ((46, (255, 40, 60, int(110 * L))), (24, (255, 120, 140, int(200 * L))),
                                (9, (255, 255, 255, int(255 * L)))):
                    d.line([(ex, ey), (x1, y1)], fill=col, width=sw(max(2, int(wd * (0.5 + 0.5 * L)))))
            elif k == "ice":
                for j in range(10):
                    ang = j * math.pi / 5 + 0.2
                    r0, r1 = sl(D_BALL + 4 + (1 - L) * 30), sl(D_BALL + 30 + (1 - L) * 60)
                    d.line([(ex + math.cos(ang) * r0, ey + math.sin(ang) * r0),
                            (ex + math.cos(ang) * r1, ey + math.sin(ang) * r1)],
                           fill=(200, 240, 255, int(255 * L)), width=sw(6))
            elif k == "boom":
                r = sl(70 + (1 - L) * 170)
                d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(255, 140, 40, int(160 * L)))
                r2 = r * 0.6
                d.ellipse((ex - r2, ey - r2, ex + r2, ey + r2), fill=(255, 240, 180, int(210 * L)))
            elif k == "shard":
                r = sl(e["r"] * L + 2)
                d.regular_polygon((ex, ey, r), 3, rotation=(1 - L) * 400, fill=e["col"] + (int(255 * L),))
            elif k == "drop":
                img = self.icons.get(e["w"])
                if img is not None:
                    g = self.img(e["w"], img).rotate(-math.degrees(e["ang"]), resample=Image.BICUBIC, expand=True)
                    g.putalpha(g.getchannel("A").point(lambda v, f=L: int(v * f)))
                    im.alpha_composite(g, (int(ex - g.width / 2), int(ey - g.height / 2)))

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
                                 stroke_width=10, stroke_fill=_stroke(w["col"]) + (255,))
        txt = txt.resize((int(W * s), int(200 * s)), Image.LANCZOS)
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(self.end_y - txt.height / 2)))
        lg = self.recipe.get("league")
        if lg and age >= 0.9:
            # what the win means for the belt
            if w["i"] == 1 and lg.get("champ"):
                sub = "NEW CHAMPION!"
            elif w["i"] == 0 and lg.get("champ"):
                n = lg.get("streak", 0) + 1
                sub = f"KEEPS THE BELT · {n} IN A ROW" if n >= 2 else "KEEPS THE BELT"
            else:
                sub = "TAKES THE BELT!"
            fs = self.fonts.get("sub")
            if fs is None:
                fs = self.fonts["sub"] = _font(54)
            a = int(255 * min(1.0, (age - 0.9) / 0.25))
            d.text((W / 2, self.end_y + 120), sub, font=fs, fill=(255, 200, 50, a), anchor="mm",
                   stroke_width=7, stroke_fill=(20, 16, 6, a))

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


# ---- the tournament (a viewer asked for 16 balls, 25 Oct) ----------------------
#
# 16 duel balls in a bracket. All the fights of a round play at once, in
# smaller arenas (8, then 4, then 2), and the final fills the screen. Every
# match is a DuelSim in the duel's own coordinates, so the fighting is the
# duel's; the painter only draws each arena smaller somewhere else. Fewer
# hearts in the early rounds and faster weapon drops once a fight runs long
# keep the whole thing under a minute.

EXTRA_COLOURS = [("Snow", "White", (236, 238, 244)), ("Coal", "Black", (52, 52, 62)),
                 ("Choco", "Brown", (134, 86, 52)), ("Mint", "Green", (140, 236, 186)),
                 ("Deep", "Maroon", (134, 26, 52)), ("Navy", "Blue", (36, 52, 150)),
                 ("Steel", "Grey", (150, 156, 172)), ("Shiny", "Gold", (206, 160, 40))]
TOURNEY_HOOKS = ["16 balls, 1 winner", "16 ball tournament -- who wins?", "Pick your ball: 16 enter, 1 wins",
                 "16 balls enter. Only 1 wins."]
ROUND_NAMES = ("ROUND OF 16", "QUARTER-FINALS", "SEMI-FINALS", "FINAL")
T_HEARTS = (2, 2, 3, 5)           # hearts per ball, by round
T_RUSH = (0.0, 0.0, 0.0, 4.0)     # seconds into a fight when the weapons start dropping fast
T_SUDDEN = (6.0, 7.0, 8.0, None)  # then sudden death: one heart each and a rocket each (the final plays out)
T_RING_MIN, T_RING_SPEED = 190, 110   # sudden death: the ring shrinks to this, this fast (px/s, duel scale)
T_INTRO = 0.6                     # the bracket fills in, then round 1 starts
T_WAIT = 0.9                      # after a round's last knockout
T_BREAK = 1.6                     # the winners move up the bracket
T_CAP = 95.0


def tourney_recipe(recipe: dict, force: Optional[str] = None) -> dict:
    recipe.pop("look", None)
    recipe["arsenal"] = 2
    rng = random.Random(f"tourney-recipe-{recipe['seed']}")
    a, b = rng.sample(COLOURS, 8), rng.sample(EXTRA_COLOURS, 8)
    players = []
    for x, y in zip(a, b):           # every first fight: a bright colour vs one of the extra ones
        players += [x, y] if rng.random() < 0.5 else [y, x]
    recipe.update(players=[list(p) for p in players], bed="off", hook=rng.choice(TOURNEY_HOOKS),
                  ending="CHAMPION!")
    look = random.Random(f"tourney-look-{recipe['seed']}")
    x = look.random()
    if {"flags": 0.0, "colours": 1.0}.get(force, x) < 0.5:
        recipe.update(look="flags", players=[_country(c) for c in look.sample(sorted(COUNTRIES), 16)],
                      hook=look.choice(FLAG_TOURNEY_HOOKS))
    return recipe


def _arenas(rnd: int) -> list:
    """(cx, cy, r) of each arena in a round, in bracket order."""
    if rnd == 0:
        return [(cx, 828 + j * 295, 118) for cx in (270, 810) for j in range(4)]
    if rnd == 1:
        return [(cx, 975 + j * 590, 222) for cx in (270, 810) for j in range(2)]
    if rnd == 2:
        return [(540, 975 + j * 590, 262) for j in range(2)]
    return [(540, 1300, 420)]


class TourneySim:
    def __init__(self, recipe: dict):
        self.recipe = recipe
        self.players = [tuple(p[:2]) + (tuple(p[2]),) + tuple(p[3:4]) for p in
                        (recipe.get("players") or [list(c) for c in COLOURS + EXTRA_COLOURS])]
        self.t = 0.0
        self.round = -1
        self.phase = "intro"
        self.phase_at = 0.0
        self.slots = [list(range(16))]           # who is in each bracket level
        self.matches: list = []
        self.pairs: list = []
        self.old: list = []                     # the last round's fights, shown while the bracket moves
        self.old_pairs: list = []
        self.sfx: list = []
        self.events: list = []
        self.hits: list = []
        self.unlocked_at = {0: 0.0}
        self.best, self.spawned = 0, 16
        self.winner = None                      # the champion's player index
        self.done_at = None
        self.ok = False
        self.max_gap = 99.0
        self.shake = 0.0
        self._ko: list = []                     # (t, round) of every knockout
        self._hit_times: list = []
        self._natural = True                    # every fight ended in a knockout

    @property
    def finished(self) -> bool:
        return (self.done_at is not None and self.t - self.done_at > END_HOLD + 0.6) or self.t > T_CAP

    def _start(self, rnd):
        self.round = rnd
        self.phase, self.phase_at = "fight", self.t
        ids = self.slots[rnd]
        self.pairs = [(ids[2 * j], ids[2 * j + 1]) for j in range(len(ids) // 2)]
        self.matches = []
        for j, (a, b) in enumerate(self.pairs):
            m = DuelSim(self.recipe, players=[self.players[a], self.players[b]], hearts=T_HEARTS[rnd],
                        key=f"tourney-{self.recipe['seed']}-{rnd}-{j}")
            m.rush_at = T_RUSH[rnd]
            m.next_spawn = 0.5               # a weapon right away
            m.keep = True                    # fire what you've got (swapping stalled the fights)
            self.matches.append(m)
        self.sfx.append((self.t, "whoosh", rnd))

    def _won(self, m):
        if m.winner is not None:
            return m.winner["i"]
        self._natural = False                   # out of time: more hearts wins
        return 0 if m.balls[0]["hp"] >= m.balls[1]["hp"] else 1

    def step(self):
        dt = 1 / FPS
        self.t += dt
        if self.phase == "intro":
            if self.t >= T_INTRO:
                self._start(0)
            return
        for m in self.matches:
            sd = T_SUDDEN[self.round]
            if sd is not None and m.winner is None and m.t >= sd and not getattr(m, "sudden", False):
                m.sudden = True
                for b in m.balls:
                    b["hp"] = min(b["hp"], 1)
                    m._equip(b, "rocket")        # homing
                self.sfx.append((self.t, "sudden", 0, self.round))
            if getattr(m, "sudden", False) and m.winner is None and m.ring > T_RING_MIN:
                # and the ring closes in until nothing can miss
                m.ring = max(T_RING_MIN, m.ring - T_RING_SPEED / FPS)
                m.pickups = [p for p in m.pickups if math.hypot(p["x"] - D_CX, p["y"] - D_CY) < m.ring - 60]
            elif getattr(m, "sudden", False) and m.winner is None and m.t >= getattr(m, "rearm", 0.0):
                # closed in: no room for drops, so a ball without a gun gets a rocket
                m.rearm = m.t + 1.0
                for b in m.balls:
                    if b["alive"] and WEAPONS.get(b["weapon"], ("",))[0] != "gun":
                        m._equip(b, "rocket")
        for m in self.matches + self.old:
            n0 = len(m.sfx)
            had = m.winner
            m.step()
            for _t, kind, val in m.sfx[n0:]:
                if kind == "hit":
                    self._hit_times.append(self.t)
                if m in self.old and kind in ("wall", "bump"):
                    continue
                self.sfx.append((self.t, kind, val, self.round if m in self.matches else self.round - 1))
            if m in self.matches and had is None and m.winner is not None:
                self._ko.append((self.t, self.round))
            if self.round == 3 and m in self.matches:
                self.hits += [(self.t, v) for _t, v in m.hits[len(self.hits):]]
        final = self.round == 3
        self.shake = self.matches[0].shake if final and self.matches else 0.0
        if self.phase == "fight":
            over = [m.done_at for m in self.matches]
            if final and self.matches[0].done_at is not None and self.done_at is None:
                m = self.matches[0]
                self.winner = self.pairs[0][self._won(m)]
                self.done_at = self.t
                self.slots.append([self.winner])
                self._finish(m)
                self.phase = "done"
            elif all(x is not None for x in over) and self.t - self.phase_at - max(over) >= T_WAIT:
                self.slots.append([pair[self._won(m)] for pair, m in zip(self.pairs, self.matches)])
                self.old, self.old_pairs = self.matches, self.pairs
                self.matches, self.pairs = [], []
                self.phase, self.phase_at = "break", self.t
        elif self.phase == "break" and self.t - self.phase_at >= T_BREAK:
            self.old, self.old_pairs = [], []
            self._start(self.round + 1)

    def _finish(self, final):
        times = [0.0] + self._hit_times + [self.t]
        self.max_gap = max(b - a for a, b in zip(times, times[1:]))
        w = final.winner["i"] if final.winner else 0
        behind = any(h[1 + w] - h[2 - w] <= -2 for h in final._hp_log)
        left = final.balls[w]["hp"]
        had = min(b["weapons_had"] for b in final.balls)
        long_enough = final.t >= 8.0
        self.ok = self._natural and (behind or left <= 2) and had >= 2 and long_enough
        self.sfx.append((self.t + 0.7, "win", 0, 3))

    def music_track(self, n: int) -> np.ndarray:
        # eight fights at once would be a wall of noise: the early rounds keep
        # the hits, shots and knockouts, and drop the wall taps and bumps
        sfx = [(t, kind, val) for t, kind, val, *rnd in self.sfx
               if not (rnd and rnd[0] < 3 and kind in ("wall", "bump", "spawn", "pickup"))]
        return _sfx_mix(self.recipe, sfx, self.hits, n, self)


def _tourney_bg(arenas) -> Image.Image:
    bg = Image.new("RGB", (W, H), (7, 9, 24))
    d = ImageDraw.Draw(bg)
    for x in range(0, W, 60):
        d.line([(x, 0), (x, H)], fill=(20, 24, 48), width=2)
    for y in range(0, H, 60):
        d.line([(0, y), (W, y)], fill=(20, 24, 48), width=2)
    mask = Image.new("L", (W, H), 0)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    md, gd = ImageDraw.Draw(mask), ImageDraw.Draw(glow)
    for cx, cy, r in arenas:
        md.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(160, 170, 255), width=max(6, r // 30))
    bg = Image.composite(bg, Image.new("RGB", (W, H), (5, 6, 16)), mask)
    glow = glow.filter(ImageFilter.GaussianBlur(12))
    bg = Image.fromarray(np.clip(np.asarray(bg, dtype=np.int16) + np.asarray(glow, dtype=np.int16), 0, 255)
                         .astype(np.uint8))
    d = ImageDraw.Draw(bg)
    for cx, cy, r in arenas:
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(245, 245, 255), width=max(3, r // 80))
    return bg


# the bracket: left half x going right, right half mirrored; rows per level
B_X = (52, 150, 248, 346)
B_Y0, B_DY = 178, 62
B_CHAMP = (540, 395)


def _bracket_pos(level: int, k: int) -> tuple:
    """Where the k-th ball of a bracket level sits (level 4: the champion)."""
    if level >= 4:
        return B_CHAMP
    per_side = 8 >> level
    side, j = divmod(k, per_side)
    span = 8 // per_side
    y = B_Y0 + (j * span + (span - 1) / 2) * B_DY
    x = B_X[level]
    return (W - x if side else x), y


class TourneyPainter(DuelPainter):
    def __init__(self, recipe: dict):
        super().__init__(dict(recipe, players=(recipe.get("players") or [])[:2] or None))
        self.recipe = recipe
        self.bgs = [_tourney_bg(_arenas(r)) for r in range(4)]
        self.bg_empty = _tourney_bg([])
        self.f_sudden, self.f_sudden2, self.f_sudden3 = _font(24), _font(34), _font(40)
        self.f_round = _font(64)
        self.f_names = _font(46)
        self.f_champ = _font(110)
        self.trophy = _emoji("1f3c6", 120)
        self.crown = _emoji("1f451", 64)
        self.hearts: dict = {}

    def heart(self, col, size, kind):
        key = (col, size, kind)
        h = self.hearts.get(key)
        if h is None:
            if kind == "full":
                h = _heart_sprite(size, col + (255,))
            else:
                h = _heart_sprite(size, None, (70, 74, 100, 255))
            self.hearts[key] = h
        return h

    def frame(self, sim: TourneySim) -> Image.Image:
        rnd = max(0, sim.round)
        if sim.phase == "break":
            # the finished round's arenas fade out, then the next round's fade in
            k = (sim.t - sim.phase_at) / T_BREAK
            if k < 0.5:
                im = Image.blend(self.bgs[rnd], self.bg_empty, min(1.0, k / 0.45)).convert("RGBA")
            else:
                im = Image.blend(self.bg_empty, self.bgs[min(3, rnd + 1)], min(1.0, (k - 0.5) / 0.45)).convert("RGBA")
        else:
            im = self.bgs[rnd].copy().convert("RGBA")
        d = ImageDraw.Draw(im)
        if sim.phase == "break":
            fade = 1 - min(1.0, (sim.t - sim.phase_at) / (T_BREAK * 0.5))
            if fade > 0:
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                self.round_matches(layer, ImageDraw.Draw(layer), sim, sim.old, sim.old_pairs, rnd)
                layer.putalpha(layer.getchannel("A").point(lambda v, f=fade: int(v * f)))
                im.alpha_composite(layer)
        else:
            self.round_matches(im, d, sim, sim.matches, sim.pairs, rnd)
        self.bracket(im, d, sim)
        self.header(im, d, sim)
        if sim.winner is not None:
            self.champion(im, d, sim)
        self.watermark(d)
        out = im.convert("RGB")
        if sim.shake > 0:
            amp = int(12 * min(1.0, sim.shake))
            rng = random.Random(int(sim.t * 1000))
            out = Image.fromarray(np.roll(np.asarray(out), (rng.randint(-amp, amp), rng.randint(-amp, amp)), (0, 1)))
        return out

    def round_matches(self, im, d, sim, matches, pairs, rnd):
        arenas = _arenas(rnd)
        hs = (30, 40, 44, 52)[rnd]
        for m, pair, (cx, cy, r) in zip(matches, pairs, arenas):
            self.place(cx, cy, r / D_R)
            if m.ring < D_R:
                # sudden death: the ring closing in, in red
                rr = m.ring * r / D_R
                q = r - max(3, r // 80)
                d.ellipse((cx - q, cy - q, cx + q, cy + q), fill=(5, 6, 16, 255))
                d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(12, 10, 26, 255), outline=(255, 70, 80, 255),
                          width=max(3, int(r / 40)))
            if getattr(m, "sudden", False) and m.winner is None:
                blink = int(m.t * 4) % 2 == 0
                d.text((cx, cy - r - 4 - (14 if rnd == 0 else 26)), "SUDDEN DEATH", font=self.font_sized(rnd),
                       fill=(255, 90, 100, 255 if blink else 170), anchor="mm", stroke_width=3,
                       stroke_fill=(7, 9, 24, 255))
            self.draw_match(im, d, m, ending=False)
            # each ball's hearts in a column beside the arena, on its side
            for i, b in enumerate(m.balls):
                x = cx + (-1 if i == 0 else 1) * (r + 16 + hs / 2)
                if rnd == 3:
                    x = 58 if i == 0 else W - 58
                n = m.hearts
                y0 = cy - (n - 1) * (hs + 6) / 2
                for k in range(n):
                    full = k < b["hp"]
                    h = self.heart(b["col"], hs, "full" if full else "empty")
                    im.alpha_composite(h, (int(x - hs / 2), int(y0 + k * (hs + 6) - hs / 2)))
        if rnd == 3 and matches:
            m = matches[0]
            parts = []
            for k, b in enumerate(m.balls):
                if k:
                    parts.append(("VS ", (255, 255, 255)))
                parts += _name_parts(sim.players[pairs[0][k]], last=k > 0)
            f = self.f_names
            total = sum(d.textlength(t, font=f) for t, _ in parts)
            x = (W - total) / 2
            for t, col in parts:
                d.text((x, 815), t, font=f, fill=col, anchor="lm", stroke_width=3, stroke_fill=(7, 9, 24))
                x += d.textlength(t, font=f)
        self.plain = True

    def font_sized(self, rnd):
        return self.f_sudden if rnd == 0 else self.f_sudden2 if rnd < 3 else self.f_sudden3

    def bracket(self, im, d, sim):
        line, dim = (58, 64, 100, 255), (34, 38, 62)
        # the lines: each ball to the next level
        for level in range(4):
            for k in range(8 >> level << 1):
                x0, y0 = _bracket_pos(level, k)
                x1, y1 = _bracket_pos(level + 1, k // 2)
                mx = (x0 + x1) / 2
                won = len(sim.slots) > level + 1 and sim.slots[level][k] == sim.slots[level + 1][k // 2]
                col = line
                width = 3
                if won:
                    p = self.players(sim, sim.slots[level][k])
                    col, width = p[2] + (255,), 5
                pts = [(x0, y0), (mx, y0), (mx, y1), (x1, y1)] if level < 3 else \
                    [(x0, y0), (x1 + (-62 if x0 < W / 2 else 62), y1)]
                d.line(pts, fill=col, width=width, joint="curve")
        # the balls
        for level, ids in enumerate(sim.slots[:4]):
            born = 0.0 if level == 0 else self.level_at(sim, level)
            for k, pid in enumerate(ids):
                x, y = _bracket_pos(level, k)
                col = self.players(sim, pid)[2]
                out = len(sim.slots) > level + 1 and pid not in sim.slots[level + 1]
                age = sim.t - (k * 0.03 if level == 0 else born)
                s = min(1.0, max(0.0, age / 0.25))
                if s <= 0:
                    continue
                r = (22 + 2 * level) * (1 + 0.3 * math.sin(min(1.0, age / 0.4) * math.pi)) * s
                flag = self.players(sim, pid)[3] if len(self.players(sim, pid)) > 3 else None
                if flag and r >= 2:
                    disc = _flag(flag, max(4, int(2 * r)))
                    if out:
                        disc = Image.blend(Image.new("RGBA", disc.size, (7, 9, 24, 0)), disc, 0.35)
                        disc.putalpha(_flag(flag, disc.width).getchannel("A"))
                    im.alpha_composite(disc, (int(x - disc.width / 2), int(y - disc.height / 2)))
                    d.ellipse((x - r, y - r, x + r, y + r), outline=dim if out else (255, 255, 255, 255),
                              width=3 if out else 4)
                    if out:
                        q = r * 0.5
                        d.line([(x - q, y - q), (x + q, y + q)], fill=(230, 60, 70, 255), width=5)
                        d.line([(x - q, y + q), (x + q, y - q)], fill=(230, 60, 70, 255), width=5)
                elif out:
                    c2 = tuple(int(v * 0.3 + 20) for v in col)
                    d.ellipse((x - r, y - r, x + r, y + r), fill=c2 + (255,), outline=dim, width=3)
                    q = r * 0.5
                    d.line([(x - q, y - q), (x + q, y + q)], fill=(230, 60, 70, 255), width=5)
                    d.line([(x - q, y + q), (x + q, y - q)], fill=(230, 60, 70, 255), width=5)
                else:
                    d.ellipse((x - r, y - r, x + r, y + r), fill=col + (255,), outline=(255, 255, 255, 255), width=4)
        tx, ty = B_CHAMP
        if sim.winner is None:
            im.alpha_composite(self.trophy, (int(tx - 60), int(ty - 60)))

    def level_at(self, sim, level):
        """When a bracket level was decided (when its round ended)."""
        ends = [t for t, r in sim._ko if r == level - 1]
        return (max(ends) + T_WAIT) if ends else sim.t

    def players(self, sim, pid):
        return sim.players[pid]

    def header(self, im, d, sim):
        rnd = max(0, sim.round)
        name = ROUND_NAMES[rnd]
        if sim.phase == "break":
            k = (sim.t - sim.phase_at) / T_BREAK
            name = ROUND_NAMES[min(3, rnd + 1)] if k > 0.45 else name
        if sim.winner is not None:
            name = "CHAMPION"
        d.text((W / 2, 92), name, font=self.f_round, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=6, stroke_fill=(70, 60, 160, 255))

    def champion(self, im, d, sim):
        p = sim.players[sim.winner]
        age = sim.t - sim.done_at
        x, y = B_CHAMP
        s = min(1.0, age / 0.3)
        r = 56 * s * (1 + 0.15 * math.sin(min(1.0, age / 0.5) * math.pi))
        if r > 1:
            if len(p) > 3 and p[3]:
                im.alpha_composite(_flag(p[3], max(4, int(2 * r))), (int(x - r), int(y - r)))
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 215, 80, 255), width=7)
            else:
                d.ellipse((x - r, y - r, x + r, y + r), fill=p[2] + (255,), outline=(255, 215, 80, 255), width=7)
            if s >= 1:
                im.alpha_composite(self.crown, (int(x - 32), int(y - r - 52)))
        if age < 0.7:
            return
        k = min(1.0, (age - 0.7) / 0.3)
        sc = 0.6 + 0.4 * k
        for text, yy, size in ((_name(p).upper(), 1490, 92), ("CHAMPION!", 1590, 80)):
            f = _font(size)
            while d.textlength(text, font=f) > W - 120:
                f = _font(f.size - 4)
            txt = Image.new("RGBA", (W, 160), (0, 0, 0, 0))
            ImageDraw.Draw(txt).text((W / 2, 80), text, font=f, fill=(255, 255, 255, 255), anchor="mm",
                                     stroke_width=9, stroke_fill=_stroke(p[2]) + (255,))
            txt = txt.resize((int(W * sc), int(160 * sc)), Image.LANCZOS)
            im.alpha_composite(txt, (int((W - txt.width) / 2), int(yy - txt.height / 2)))


# ---- 2 VS 2 (25 Oct: more kinds of fights) --------------------------------------
#
# Four duel balls, two teams: a warm pair and a cool pair, each ball ringed in
# its team's colour. Every ball aims at the nearest enemy; teammates can't
# hurt each other (a shield bump on a teammate does nothing). A team is out
# when both its balls are. The duel's own code with its hooks overridden, so
# the 1v1 duel is unchanged.

WARM = [c for c in COLOURS if c[1] in ("Red", "Yellow", "Pink", "Orange")]
COOL = [c for c in COLOURS if c[1] in ("Blue", "Green", "Purple", "Cyan")]
TEAM_RINGS = ((255, 160, 40), (80, 200, 255))
TEAM_HEARTS = 6
TEAM_HOOKS = ["2 VS 2 -- which team wins?", "{a} VS {b} -- 2 on 2", "Pick a team: {a} or {b}?"]


def _team_name(pair) -> str:
    return f"{pair[0][1]} & {pair[1][1]}"


class TeamDuelSim(DuelSim):
    def __init__(self, recipe: dict):
        super().__init__(recipe, hearts=TEAM_HEARTS)
        for b in self.balls:
            b["team"] = b["i"] // 2
            b["ring"] = TEAM_RINGS[b["team"]]
            if recipe.get("style") == "clean":
                # the ring in the team's own colours (warm: orange, cool: blue), not by its place
                first_warm = any(tuple(c[2]) == self.balls[b["team"] * 2]["col"] for c in WARM)
                b["ring"] = TEAM_RINGS[0] if first_warm else TEAM_RINGS[1]
            b["x"] = D_CX + (-230 if b["team"] == 0 else 230)
            b["y"] = D_CY + (-170 if b["i"] % 2 == 0 else 170)

    def team(self, k):
        return [b for b in self.balls if b["team"] == k]

    def other(self, b):
        foes = [o for o in self.balls if o["team"] != b["team"]]
        live = [o for o in foes if o["alive"]] or foes
        return min(live, key=lambda o: (o["x"] - b["x"]) ** 2 + (o["y"] - b["y"]) ** 2)

    def damage(self, target, dmg, x, y, by) -> bool:
        if target["team"] == by["team"]:
            return False
        return super().damage(target, dmg, x, y, by)

    def _hp_row(self) -> tuple:
        return tuple(sum(b["hp"] for b in self.team(k)) for k in (0, 1))

    def _ko(self, target, by):
        if not any(b["alive"] for b in self.team(target["team"])):
            self._end(by)

    def _movers(self):
        return [b for b in self.balls if b["alive"]]

    def _pairs(self):
        live = self._movers()
        return [(a, c) for k, a in enumerate(live) for c in live[k + 1:]]

    def _bullet_hit(self, u):
        own = u["owner"]
        for o in self.balls:
            if o["alive"] and o["team"] != own["team"] and math.hypot(u["x"] - o["x"], u["y"] - o["y"]) < D_BALL + 8:
                return o
        return None

    def _boom_targets(self, owner):
        return [o for o in self.balls if o["alive"] and o["team"] != owner["team"]]

    def _end(self, winner):
        self.winner = winner
        self.done_at = self.t
        self.sfx.append((self.t, "die", 0))
        w = winner["team"]
        behind = any(h[1 + w] - h[2 - w] <= -3 for h in self._hp_log)      # a comeback
        left = sum(b["hp"] for b in self.team(w))
        had = min(b["weapons_had"] for b in self.balls)
        self.max_gap = max(self._gaps + [0.0])
        self.ok = behind and left <= 5 and had >= 1


class TeamDuelPainter(DuelPainter):
    def __init__(self, recipe: dict):
        super().__init__(dict(recipe, players=(recipe.get("players") or [])[:2] or None))
        self.recipe = recipe
        names = recipe.get("team_names") or ["Warm", "Cool"]
        self.title = [(names[0] + " ", TEAM_RINGS[0]), ("VS ", (255, 255, 255)), (names[1], TEAM_RINGS[1])]
        self.small_full: dict = {}
        self.small_empty = _heart_sprite(40, None, (70, 74, 100, 255))

    def mood(self, sim, b):
        if sim.winner is not None and b["team"] == sim.winner["team"]:
            return "happy" if b["alive"] else "dead"
        return super().mood(sim, b)

    def hud(self, im, d, sim):
        # one half per team, a row per ball: its colour, then its hearts
        for b in sim.balls:
            x0 = 24 if b["team"] == 0 else W // 2 + 12
            y = 120 + (b["i"] % 2) * 112
            d.ellipse((x0, y - 36, x0 + 72, y + 36), fill=(b["col"] if b["alive"] else (60, 62, 80)) + (255,),
                      outline=b["ring"] + (255,), width=6)
            h = self.small_full.get(b["col"])
            if h is None:
                h = self.small_full[b["col"]] = _heart_sprite(40, b["col"] + (255,))
            for k in range(sim.hearts):
                im.alpha_composite(h if k < b["hp"] else self.small_empty, (x0 + 84 + k * 70, y - 20))
        f = self.f_title
        total = sum(d.textlength(t, font=f) for t, _ in self.title)
        while total > W - 80 and f.size > 30:
            f = _font(f.size - 4)
            total = sum(d.textlength(t, font=f) for t, _ in self.title)
        x = (W - total) / 2
        for t, col in self.title:
            d.text((x, 480), t, font=f, fill=col, anchor="lm")
            x += d.textlength(t, font=f)

    def ending(self, im, d, sim):
        w = sim.winner
        age = sim.t - sim.done_at
        if age < 0.6:
            return
        k = min(1.0, (age - 0.6) / 0.3)
        names = self.recipe.get("team_names") or ["Warm", "Cool"]
        name = names[w["team"]].upper() + " WIN!"
        f = self.f_win
        while d.textlength(name, font=f) > W - 180:
            f = _font(f.size - 4)
        s = 0.6 + 0.4 * k
        txt = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        ImageDraw.Draw(txt).text((W / 2, 100), name, font=f, fill=(255, 255, 255, 255), anchor="mm",
                                 stroke_width=10, stroke_fill=tuple(w.get("ring") or TEAM_RINGS[w["team"]]) + (255,))
        txt = txt.resize((int(W * s), int(200 * s)), Image.LANCZOS)
        im.alpha_composite(txt, (int((W - txt.width) / 2), int(self.end_y - txt.height / 2)))


def _duel_sim(recipe: dict):
    return (TeamDuelSim if recipe.get("look") == "teams" else DuelSim)(recipe)


def _duel_painter(recipe: dict):
    return (TeamDuelPainter if recipe.get("look") == "teams" else DuelPainter)(recipe)


# ---- battle royale (26 Oct: Dean, "16 balls, one circle, 1v1v1...") ------------
#
# 16 duel balls in one ring, every ball for itself: it aims at the nearest
# ball, bullets hit whoever they reach, a rocket hurts everyone near it.
# R_HEARTS each. After R_ZONE_AT the ring closes in like a battle-royale zone
# (red on screen), so the last fights can't drag on; once it is smaller
# than the duel's ring, guns fire without waiting to aim. The fight runs in
# a bigger ring (R_START) in the duel's own units and is drawn scaled down,
# so 16 balls have room. Last ball standing wins.

# 29 Oct: the first royale lost half its viewers 5-10 s in, before the
# knockouts started, so it starts closer, with 3 hearts and the zone at 3 s:
# first knockout ~3 s (was ~6), third ~7 s (was ~11), ~22 s long.
R_START, R_MIN = 800, 250
R_ZONE_AT, R_ZONE_SPEED = 3.0, 16.0
R_HEARTS = 3
R_SPAWN = 420                     # where the 16 start, from the middle
R_FIRST_KO = 3.5                  # a seed needs its first knockout by then
# The falling floor (Dean picked it from the idea board, 29 Oct): half the new
# royales stand on hex tiles instead of a closing ring. From F_START the tiles
# crack (F_CRACK s of warning) and drop, outside first (F_SPEED px/s inward,
# a bit of jitter); the middle (F_KEEP) never drops. A ball over a hole falls
# and is out. Balls see a hole coming and turn back F_DODGE of the time, so
# falls are a moment, not a constant.
F_TILE, F_START, F_SPEED, F_CRACK, F_KEEP, F_DODGE = 70, 4.0, 12.0, 1.0, 260, 0.95
FLOOR_HOOKS = ["The floor falls away. Last one standing wins", "16 balls, the floor is falling. Who's left?",
               "Don't fall: last one standing wins", "Pick one before the floor goes"]
ROYALE_HOOKS = ["16 balls, 1 circle -- who survives?", "Last ball standing wins", "16 balls enter. 1 leaves.",
                "Pick a ball: last one alive wins"]
FLAG_ROYALE_HOOKS = ["16 countries, 1 circle -- who survives?", "Last country standing wins",
                     "16 countries enter. 1 leaves.", "Pick a country: last one alive wins"]


def royale_recipe(recipe: dict, force: Optional[str] = None) -> dict:
    recipe.pop("look", None)
    recipe["arsenal"] = 2
    rng = random.Random(f"royale-recipe-{recipe['seed']}")
    x = rng.random()
    if {"flags": 0.0, "colours": 1.0}.get(force, x) < 0.5:
        players = [_country(c) for c in rng.sample(sorted(COUNTRIES), 16)]
        recipe.update(look="flags", hook=rng.choice(FLAG_ROYALE_HOOKS))
    else:
        players = [list(c) for c in rng.sample(COLOURS + EXTRA_COLOURS, 16)]
        recipe.update(hook=rng.choice(ROYALE_HOOKS))
    recipe.update(players=players, bed="off", ending="WINS!")
    fl = random.Random(f"floor-{recipe['seed']}")               # own rng: the draws above stay as they were
    recipe["floor"] = fl.random() < 0.5
    if recipe["floor"]:
        recipe["hook"] = fl.choice(FLOOR_HOOKS)
    return recipe


class RoyaleSim(TeamDuelSim):
    def __init__(self, recipe: dict):
        DuelSim.__init__(self, recipe, hearts=R_HEARTS)
        self.ring = R_START
        self.rush_at = 0.0
        self.keep = True
        self.max_drops = 6
        self.next_spawn = 0.4
        n = len(self.balls)
        for b in self.balls:
            a = 2 * math.pi * b["i"] / n - math.pi / 2
            b["team"] = b["i"]                       # everyone for themselves
            b["x"], b["y"] = D_CX + math.cos(a) * R_SPAWN, D_CY + math.sin(a) * R_SPAWN
        self.kos: list = []                          # (t, ball index)
        self.tiles: dict = {}                        # the falling floor: (q, r) -> tile
        if recipe.get("floor"):
            frng = random.Random(f"floor-tiles-{recipe['seed']}")
            n = int(R_START / F_TILE) + 2
            for q in range(-n, n + 1):
                for r in range(-n, n + 1):
                    x, y = self._hex_xy(q, r)
                    dd = math.hypot(x - D_CX, y - D_CY)
                    if dd > R_START + F_TILE * 0.3:
                        continue
                    drop = None if dd < F_KEEP else F_START + (R_START - dd) / F_SPEED + frng.uniform(-0.9, 0.9)
                    self.tiles[(q, r)] = {"x": x, "y": y, "drop": drop}

    # -- the falling floor
    @staticmethod
    def _hex_xy(q, r):
        return D_CX + F_TILE * 1.5 * q, D_CY + F_TILE * math.sqrt(3) * (r + q / 2)

    def _tile_at(self, x, y):
        fq = (x - D_CX) / (F_TILE * 1.5)
        fr = (y - D_CY) / (F_TILE * math.sqrt(3)) - fq / 2
        # cube rounding
        cx, cz = fq, fr
        cy = -cx - cz
        rx, ry, rz = round(cx), round(cy), round(cz)
        dx, dy, dz = abs(rx - cx), abs(ry - cy), abs(rz - cz)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dy <= dz:
            rz = -rx - ry
        return self.tiles.get((rx, rz))

    def _hole(self, x, y, ahead=0.0):
        t = self._tile_at(x, y)
        return t is None or (t["drop"] is not None and self.t + ahead >= t["drop"])

    def _floor(self):
        for b in self.balls:
            if not b["alive"]:
                continue
            if self._hole(b["x"], b["y"]):
                b["alive"], b["hp"] = False, 0
                self.effects.append({"k": "fall", "x": b["x"], "y": b["y"], "life": 1.0, "col": b["col"],
                                     "flag": b.get("flag")})
                self.sfx.append((self.t, "out", b["i"] % 3))
                self._ko(b, b)
                if self.winner is not None:
                    return
                continue
            if b["frozen"] > 0 or self.t < b.get("edge_cd", 0):
                continue
            sp = math.hypot(b["vx"], b["vy"]) or 1
            if self._hole(b["x"] + b["vx"] / sp * (D_BALL + 50), b["y"] + b["vy"] / sp * (D_BALL + 50), 0.15):
                b["edge_cd"] = self.t + 0.5
                if self.rng.random() < F_DODGE:          # it sees the hole and turns back
                    a = math.atan2(-b["vy"], -b["vx"]) + self.rng.uniform(-0.6, 0.6)
                    b["vx"], b["vy"] = sp * math.cos(a), sp * math.sin(a)
        self.pickups = [p for p in self.pickups if not self._hole(p["x"], p["y"], F_CRACK)]

    def left(self) -> int:
        return sum(1 for b in self.balls if b["alive"])

    def _hp_row(self) -> tuple:
        return (self.left(),)

    def _ko(self, target, by):
        self.kos.append((self.t, target["i"]))
        self.sfx.append((self.t, "die", 0))
        live = [b for b in self.balls if b["alive"]]
        if len(live) == 1:
            self._end(live[0])

    def _end(self, winner):
        self.winner = winner
        self.done_at = self.t
        self.sfx.append((self.t + 0.3, "win", 0))
        times = [0.0] + [t for t, _ in self.kos]
        self.max_gap = max(b - a for a, b in zip(times, times[1:]))     # longest wait between knockouts
        self.ok = winner["hp"] <= 2 and times[1] <= R_FIRST_KO

    def step(self):
        super().step()
        if self.tiles:
            if self.winner is None:
                self._floor()
        elif self.winner is None and self.t > R_ZONE_AT:
            self.ring = max(R_MIN, R_START - (self.t - R_ZONE_AT) * R_ZONE_SPEED)
            self.pickups = [p for p in self.pickups if math.hypot(p["x"] - D_CX, p["y"] - D_CY) < self.ring - 60]
        for b in self.balls:
            if not b["alive"] and not b.get("gone"):
                b["gone"] = True                     # out of the way of drops (its shards are already flying)
                b["x"] = b["y"] = -1e5

    def music_track(self, n: int) -> np.ndarray:
        # 16 balls tapping the wall is a wall of noise: hits, shots and knockouts only
        sfx = [e for e in self.sfx if e[1] not in ("wall", "bump", "spawn")]
        return _sfx_mix(self.recipe, sfx, self.hits, n, self)


class RoyalePainter(DuelPainter):
    def __init__(self, recipe: dict):
        super().__init__(dict(recipe, players=(recipe.get("players") or [])[:2] or None))
        self.recipe = recipe
        self.players = recipe.get("players") or [list(c) for c in COLOURS + EXTRA_COLOURS]
        self.k = D_R / R_START
        self.f_left = _font(64)
        self.f_hook = _font(52)
        self.pip_full: dict = {}
        self.pip_empty = _heart_sprite(18, None, (90, 94, 120, 255))
        self.dead_at: dict = {}

    def floor(self, im, sim):
        """The hex floor: blue tiles, orange (blinking) when cracked, gone
        when dropped. Redrawn only when a tile changes."""
        state = tuple(0 if t["drop"] is None or sim.t < t["drop"] - F_CRACK else
                      1 if sim.t < t["drop"] else 2 for t in sim.tiles.values())
        blink = int(sim.t * 8) % 2
        key = (state, blink if 1 in state else 0)
        if getattr(self, "_floor_key", None) != key:
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            k = self.k
            for t, st in zip(sim.tiles.values(), state):
                if st == 2:
                    continue
                x, y = D_CX + (t["x"] - D_CX) * k, D_CY + (t["y"] - D_CY) * k
                rr = F_TILE * k * 0.95
                pts = [(x + rr * math.cos(j * math.pi / 3), y + rr * math.sin(j * math.pi / 3)) for j in range(6)]
                fill = (70, 92, 205, 255) if st == 0 else ((255, 130, 80, 255) if blink else (220, 90, 60, 255))
                d.polygon(pts, fill=fill, outline=(150, 176, 255, 255) if st == 0 else (255, 210, 160, 255))
            self._floor_layer, self._floor_key = lay, key
        im.alpha_composite(self._floor_layer)

    def frame(self, sim: RoyaleSim) -> Image.Image:
        im = self.bg.copy().convert("RGBA")
        d = ImageDraw.Draw(im)
        if sim.tiles:
            q = D_R - 4
            d.ellipse((D_CX - q, D_CY - q, D_CX + q, D_CY + q), fill=(4, 5, 12, 255))       # the void under it
            self.floor(im, sim)
        if sim.ring < R_START - 0.5:
            # the zone closing in, in red
            rr = sim.ring * self.k
            q = D_R - 4
            d.ellipse((D_CX - q, D_CY - q, D_CX + q, D_CY + q), fill=(5, 6, 16, 255))
            d.ellipse((D_CX - rr, D_CY - rr, D_CX + rr, D_CY + rr), fill=(12, 10, 26, 255),
                      outline=(255, 70, 80, 255), width=6)
        self.hud(im, d, sim)
        self.place(D_CX, D_CY, self.k)
        self.draw_match(im, d, sim, ending=False)
        for e in sim.effects:                        # falling through the floor: it shrinks away
            if e["k"] == "fall":
                x, y = self.P(e["x"], e["y"])
                r = D_BALL * self.k * max(0.0, e["life"])
                if r > 2:
                    if e.get("flag"):
                        g = _flag(e["flag"], max(4, int(2 * r)))
                        im.alpha_composite(g, (int(x - g.width / 2), int(y - g.height / 2)))
                    else:
                        d.ellipse((x - r, y - r, x + r, y + r), fill=e["col"] + (255,))
                    d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, int(255 * e["life"])), width=3)
        for b in sim.balls:                          # each ball's hearts as pips above it
            if not b["alive"] or sim.winner is not None:
                continue
            x, y = self.P(b["x"], b["y"])
            full = self.pip_full.get(b["col"])
            if full is None:
                full = self.pip_full[b["col"]] = _heart_sprite(18, b["col"] + (255,))
            x0 = x - (sim.hearts * 20 - 2) / 2
            for k in range(sim.hearts):
                im.alpha_composite(full if k < b["hp"] else self.pip_empty, (int(x0 + k * 20), int(y - 62)))
        self.plain = True
        if sim.winner is not None:
            self.ending(im, d, sim)
        self.watermark(d)
        out = im.convert("RGB")
        if sim.shake > 0:
            amp = int(10 * min(1.0, sim.shake))
            rng = random.Random(int(sim.t * 1000))
            out = Image.fromarray(np.roll(np.asarray(out), (rng.randint(-amp, amp), rng.randint(-amp, amp)), (0, 1)))
        return out

    def hud(self, im, d, sim):
        f = self.f_hook
        hook = self.recipe.get("hook") or ""
        while d.textlength(hook, font=f) > W - 80 and f.size > 30:
            f = _font(f.size - 4)
        d.text((W / 2, 92), hook, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=5,
               stroke_fill=(70, 60, 160, 255))
        # everyone in two rows: crossed out when they're knocked out
        for b in sim.balls:
            x, y = 76 + (b["i"] % 8) * 132, 210 + (b["i"] // 8) * 108
            r = 40
            if not b["alive"]:
                self.dead_at.setdefault(b["i"], sim.t)
            if b.get("flag"):
                disc = _flag(b["flag"], 2 * r)
                if not b["alive"]:
                    disc = Image.blend(Image.new("RGBA", disc.size, (7, 9, 24, 0)), disc, 0.35)
                    disc.putalpha(_flag(b["flag"], 2 * r).getchannel("A"))
                im.alpha_composite(disc, (x - r, y - r))
                d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, 255) if b["alive"] else (60, 64, 90),
                          width=4)
            else:
                col = b["col"] if b["alive"] else tuple(int(v * 0.3 + 20) for v in b["col"])
                d.ellipse((x - r, y - r, x + r, y + r), fill=col + (255,),
                          outline=(255, 255, 255, 255) if b["alive"] else (60, 64, 90), width=4)
            if not b["alive"]:
                age = sim.t - self.dead_at[b["i"]]
                q = r * 0.55 * min(1.0, age / 0.15 + 0.3)
                d.line([(x - q, y - q), (x + q, y + q)], fill=(230, 60, 70, 255), width=6)
                d.line([(x - q, y + q), (x + q, y - q)], fill=(230, 60, 70, 255), width=6)
        left = sim.left()
        d.text((W / 2, 430), f"{left} LEFT" if left > 1 else "1 LEFT", font=self.f_left, fill=(255, 255, 255, 255),
               anchor="mm", stroke_width=6, stroke_fill=(200, 50, 70, 255))


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
    if kind == "sudden":              # sudden death: a low warning blip
        t = tt(0.35)
        return np.sin(2 * np.pi * 220 * t) * np.exp(-t * 9) * (0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 8 * t))) * 0.3
    if kind == "whoosh":              # a tournament round starts
        t = tt(0.7)
        sweep = _lp(_noise(len(t), seed), 6) * np.sin(np.pi * np.minimum(1, t / 0.7)) ** 2
        gong = np.sin(2 * np.pi * (110 + 30 * seed) * t) * np.exp(-t * 5) * np.minimum(1, t / 0.01)
        return (sweep * 0.35 + gong * 0.5) * 0.8
    if kind == "charge":              # the laser charging: a rising hum
        t = tt(LASER_CHARGE)
        f = 180 + 900 * (t / LASER_CHARGE) ** 2
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.05) * 0.22
    if kind == "laser":               # the beam: a bright zap with a low body
        t = tt(0.45)
        zap = np.sin(2 * np.pi * np.cumsum(2400 * np.exp(-t * 9) + 300) / SR) * np.exp(-t * 7)
        body = np.sin(2 * np.pi * np.cumsum(90 + 60 * np.exp(-t * 12)) / SR) * np.exp(-t * 6)
        return np.tanh((zap * 0.8 + body + _lp(_noise(len(t), seed), 2) * np.exp(-t * 30) * 0.4) * 1.8) * 0.75
    if kind == "icefire":             # the freeze gun: a soft puff
        t = tt(0.25)
        return _lp(_noise(len(t), seed), 3) * np.exp(-t * 20) * 0.4
    if kind == "freeze":              # frozen: a glassy crackle + a high chime
        t = tt(0.7)
        chime = sum(np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, k in ((1760, 6), (2637, 8), (3520, 11)))
        crackle = _noise(len(t), seed) * (np.random.default_rng(seed + 9).random(len(t)) > 0.995) * 3
        return (chime * 0.18 + _lp(crackle, 2) * np.exp(-t * 8) * 0.5) * 0.9
    if kind == "fuse":                # a bomb put down: beeps that speed up over BOMB_FUSE
        t = tt(BOMB_FUSE)
        beeps = np.zeros(len(t))
        bt = 0.0
        while bt < BOMB_FUSE - 0.05:
            i = int(bt * SR)
            n = int(0.05 * SR)
            beeps[i:i + n] += np.sin(2 * np.pi * 1500 * t[:n]) * np.exp(-t[:n] * 40)
            bt += 0.32 * (1 - bt / BOMB_FUSE) + 0.06
        return beeps * 0.25
    if kind == "slash":               # katana dash: a sharp swish
        t = tt(0.3)
        sw = _lp(_noise(len(t), seed), 2) * np.sin(np.pi * np.minimum(1, t / 0.3)) ** 2
        return sw * np.exp(-t * 4) * 0.55
    if kind == "slam":                # hammer: a deep thud + crunch
        t = tt(0.8)
        thud = np.sin(2 * np.pi * np.cumsum(55 + 120 * np.exp(-t * 14)) / SR) * np.exp(-t * 5)
        return np.tanh((thud * 1.6 + _lp(_noise(len(t), seed), 6) * np.exp(-t * 12) * 1.2) * 2) * 0.9
    if kind == "throw":               # boomerang / axe leaving the hand
        t = tt(0.35)
        return _lp(_noise(len(t), seed), 5) * np.sin(np.pi * t / 0.35) * 0.35
    if kind == "twang":               # the bow
        t = tt(0.45)
        f = 220 * (1 + 0.04 * np.exp(-t * 30))
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) + _lp(_noise(len(t), seed), 2)
                * np.exp(-t * 60) * 0.4) * 0.55
    if kind == "thunder":             # lightning: a crack and a rumble
        t = tt(1.0)
        crack = _noise(len(t), seed) * np.exp(-t * 30)
        rumble = _lp(_noise(len(t), seed + 1), 60) * 8 * np.exp(-t * 3)
        return np.tanh((crack + rumble) * 1.6) * 0.85
    if kind == "fire":                # the flamethrower's roar, FLAME_S long
        t = tt(FLAME_S)
        roar = _lp(_noise(len(t), seed), 8) * 3 * np.minimum(1, t / 0.08) * np.minimum(1, (FLAME_S - t) / 0.2)
        return np.tanh(roar) * 0.45
    if kind == "magnet":              # a rising electric hum
        t = tt(0.9)
        f = 110 + 120 * t
        return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.minimum(1, t / 0.05) * np.exp(-t * 2) * 0.12
    if kind == "slip":                # a banana spin-out: a slide whistle down
        t = tt(0.5)
        f = 1400 * np.exp(-t * 3) + 300
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3) * 0.3
    if kind == "punch":
        t = tt(0.3)
        body = np.sin(2 * np.pi * np.cumsum(70 + 220 * np.exp(-t * 35)) / SR) * np.exp(-t * 16)
        return np.tanh((body * 1.4 + _lp(_noise(len(t), seed), 3) * np.exp(-t * 50)) * 2.4) * 0.85
    if kind == "bonk":                # the frying pan
        t = tt(0.6)
        return sum(np.sin(2 * np.pi * f * t) * np.exp(-t * k) for f, k in ((310, 6), (620, 9), (1240, 14))) * 0.3
    if kind == "wind":                # a tornado
        t = tt(1.4)
        w = _lp(_noise(len(t), seed), 12) * 4 * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        return np.tanh(w) * np.minimum(1, t / 0.2) * np.exp(-t * 1.2) * 0.4
    if kind == "hole":                # a black hole opening: a low falling swirl
        t = tt(1.2)
        f = 180 * np.exp(-t * 1.5) + 40
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.1) * np.exp(-t * 1.5) * 0.5
    if kind == "rumble":              # the boulder rolling in
        t = tt(1.0)
        return np.tanh(_lp(_noise(len(t), seed), 50) * 10) * np.minimum(1, t / 0.1) * np.exp(-t * 2) * 0.6
    if kind == "sparkle":             # the wand
        t = tt(0.5)
        return sum(np.sin(2 * np.pi * f * t) * np.exp(-t * 10) * (t > d)
                   for f, d in ((1568, 0), (2093, 0.06), (2637, 0.12))) * 0.15
    if kind == "gulp":                # a potion
        t = tt(0.4)
        return sum(np.sin(2 * np.pi * (300 + 200 * k) * t) * np.exp(-((t - 0.1 * k) * 30) ** 2)
                   for k in range(3)) * 0.35
    if kind == "out":                 # a country falls out of the ring (survivor): a quick drop
        t = tt(0.32)
        f = 900 * np.exp(-t * 7) + 160 + 40 * seed
        tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
        return (tone * 0.5 + _lp(_noise(len(t), seed), 6) * np.exp(-t * 25) * 0.25) * 0.8
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
    if music is None and recipe.get("bounce_sound") == "none":
        music = np.zeros(n, dtype=np.float32)        # just the fight's own sounds
    if music is None:
        from clipper import ball_merge as bm
        music = bm.MergeSim.music_track(sim, n) * 0.8
    music = np.pad(music, (0, max(0, n - len(music))))[:n]
    return out * 0.8 + music


PARTS = {"duel": (_duel_sim, _duel_painter), "race": (RaceSim, RacePainter),
         "tourney": (TourneySim, TourneyPainter), "royale": (RoyaleSim, RoyalePainter)}
