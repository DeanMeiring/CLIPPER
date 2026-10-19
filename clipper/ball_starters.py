"""Starter picture packs for Ball Evolution (Dean asked for 10 templates:
cursed, chad, sigma and meme chains that evolve to "the best one", and
countries that evolve to "the best country", from a source we're allowed
to use or made ourselves).

All of them ship with the code in ``clipper/assets/packs/<key>/`` (the
pictures + a pack.json) and are copied into the pack store the first time
it's listed:

* 5 flag packs (Europe, world, football; Africa and Asia since Oct 19):
  countryballs (a flag in a circle with two eyes, the final one wearing a
  crown), flags from flag-icons (MIT).
* 5 logo packs (Oct 19): phones, apps, cars, gaming, sportswear -- logo
  balls from Simple Icons (CC0; the logos are their owners' trademarks).
  Dean wants pictures people know rather than emoji, so these and the flag
  packs are picked before any other (``ball_packs.is_known``).
* 7 cursed packs made by ``ball_cursed`` from the bundled Fluent Emoji
  (MIT): Cat > Long Cat > Cool Cat > Buff Cat > Sigma Cat > Cursed Cat >
  Deep Fried Cat > Cat God. The app makes new ones the same way when the
  packs run out.

Not used on purpose: GIPHY stickers and meme art belong to the people who
made them, real people (Gigachad is a photo of a real man), Pepe, company
characters (mascots -- the logo packs are logos only, Dean's call).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .ball_packs import PackStore

ASSETS = Path(__file__).resolve().parent / "assets" / "packs"

# the cursed starter packs: folder name -> (face, seed) for ball_cursed.make
CURSED_STARTERS = {"cursed_cat": ("1f431", 1), "cursed_dog": ("1f436", 2), "cursed_frog": ("1f438", 3),
                   "cursed_panda": ("1f43c", 4), "cursed_alien": ("1f47d", 5), "cursed_pumpkin": ("1f383", 6),
                   "cursed_monkey": ("1f435", 7)}


def seed_bundled(store: PackStore) -> list:
    """Copy the bundled flag packs into the store, once each. Returns the
    keys added."""
    have = {p.get("template") for p in store.packs()}
    added = []
    if not ASSETS.is_dir():
        return added
    for d in sorted(ASSETS.iterdir()):
        meta = d / "pack.json"
        if not meta.is_file() or d.name in have:
            continue
        info = json.loads(meta.read_text())
        items = [(it["name"], Image.open(d / it["file"]).convert("RGBA")) for it in info["items"]]
        store.save(info["title"], items, "starter",
                   {"template": d.name, "hook": info.get("hook"), "credit": info.get("credit"),
                    **({"cursed_base": info["cursed_base"]} if info.get("cursed_base") else {})})
        added.append(d.name)
    return added
