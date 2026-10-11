"""Starter picture packs for Ball Evolution (Dean asked for 10 templates:
cursed, chad, sigma and meme chains that evolve to "the best one", and
countries that evolve to "the best country").

* Flag packs ship with the code: ``clipper/assets/packs/<key>/`` holds the
  pictures (countryballs: a flag in a circle with two eyes, the final one
  wearing a crown; flags from flag-icons, MIT) and a pack.json. They're
  copied into the pack store the first time it's listed.
* Meme packs can't ship as pictures: their stickers come from GIPHY, which
  only the app can reach. Each one is a ready plan here (names + GIPHY
  sticker searches); the app's automatic pack uses the next unbuilt plan
  before asking Claude for a new one, and Claude still picks the stickers
  (no real people, nothing unsafe).

Left out on purpose: real people (Gigachad is a photo of a real man,
sigma edits use actors), Pepe (its artist enforces his copyright),
company characters (Skibidi Toilet, Grimace, Labubu, Minions), and the
"chud" wojak (a political insult).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from .ball_packs import PackStore

ASSETS = Path(__file__).resolve().parent / "assets" / "packs"


def _q(*qs: str) -> list:
    return list(qs)


# name the viewer sees, then three GIPHY sticker searches (two specific, one broad)
MEME_PLANS = {
    "sigma_cat": {"title": "Cat to Sigma Cat", "hook": "What does the cat turn into?", "steps": [
        {"name": "Cat", "queries": _q("cute cat", "kitten sticker", "cat")},
        {"name": "Chubby Cat", "queries": _q("chubby cat", "fat cat", "cat")},
        {"name": "Cool Cat", "queries": _q("cat sunglasses", "cool cat", "cat")},
        {"name": "Buff Cat", "queries": _q("buff cat", "muscle cat", "strong cat")},
        {"name": "Chad Cat", "queries": _q("chad cat", "cat smirk", "cat meme")},
        {"name": "Sigma Cat", "queries": _q("sigma cat", "cat suit", "boss cat")},
        {"name": "Cosmic Cat", "queries": _q("space cat", "cat galaxy", "astronaut cat")},
        {"name": "Cat God", "queries": _q("cat king", "cat crown", "king cat")}]},
    "doge": {"title": "Doge Ascension", "hook": "What does the doge become?", "steps": [
        {"name": "Shiba", "queries": _q("shiba inu", "shiba", "dog")},
        {"name": "Cheems", "queries": _q("cheems", "sad doge", "shiba meme")},
        {"name": "Doge", "queries": _q("doge", "doge meme", "much wow")},
        {"name": "Cool Doge", "queries": _q("doge sunglasses", "cool shiba", "cool dog")},
        {"name": "Swole Doge", "queries": _q("swole doge", "buff doge", "buff dog")},
        {"name": "Doge Astronaut", "queries": _q("doge moon", "doge space", "dog astronaut")},
        {"name": "Doge King", "queries": _q("doge king", "shiba crown", "king dog")},
        {"name": "Doge God", "queries": _q("doge god", "golden doge", "doge")}]},
    "wojak_chad": {"title": "Wojak to Chad", "hook": "Who's the final form?", "steps": [
        {"name": "Wojak", "queries": _q("wojak", "feels guy", "sad wojak")},
        {"name": "Doomer", "queries": _q("doomer", "doomer wojak", "sad guy")},
        {"name": "NPC", "queries": _q("npc meme", "npc wojak", "npc")},
        {"name": "Bloomer", "queries": _q("bloomer wojak", "happy wojak", "wojak smile")},
        {"name": "Zoomer", "queries": _q("zoomer wojak", "zoomer", "wojak")},
        {"name": "Chad", "queries": _q("chad meme", "chad wojak", "chad")},
        {"name": "Yes Chad", "queries": _q("yes chad", "nordic gamer", "chad")},
        {"name": "Sigma", "queries": _q("sigma", "sigma male", "sigma meme")}]},
    "brainrot": {"title": "Italian Brainrot", "hook": "Which brainrot is last?", "steps": [
        {"name": "Tung Tung Sahur", "queries": _q("tung tung tung sahur", "tung tung", "brainrot")},
        {"name": "Ballerina Cappuccina", "queries": _q("ballerina cappuccina", "ballerina cappuccino", "italian brainrot")},
        {"name": "Brr Brr Patapim", "queries": _q("brr brr patapim", "patapim", "italian brainrot")},
        {"name": "Lirili Larila", "queries": _q("lirili larila", "cactus elephant", "italian brainrot")},
        {"name": "Chimpanzini", "queries": _q("chimpanzini bananini", "chimpanzini", "italian brainrot")},
        {"name": "Bombardiro", "queries": _q("bombardiro crocodilo", "bombardiro", "italian brainrot")},
        {"name": "Tralalero", "queries": _q("tralalero tralala", "tralalero", "italian brainrot")}]},
    "frog_king": {"title": "Frog to Frog King", "hook": "What will the frog become?", "steps": [
        {"name": "Frog", "queries": _q("cute frog", "frog sticker", "frog")},
        {"name": "Cursed Frog", "queries": _q("cursed frog", "weird frog", "frog meme")},
        {"name": "Frog in a Hat", "queries": _q("frog hat", "frog cowboy", "frog")},
        {"name": "Dancing Frog", "queries": _q("dancing frog", "frog dance", "frog")},
        {"name": "Buff Frog", "queries": _q("buff frog", "muscle frog", "strong frog")},
        {"name": "Wizard Frog", "queries": _q("wizard frog", "frog magic", "frog wizard")},
        {"name": "Frog King", "queries": _q("frog king", "frog crown", "frog prince")},
        {"name": "Frog God", "queries": _q("giant frog", "frog god", "frog")}]},
    "capybara": {"title": "Capybara Ascension", "hook": "What does the capybara become?", "steps": [
        {"name": "Capybara", "queries": _q("capybara", "cute capybara", "capybara sticker")},
        {"name": "Orange Capybara", "queries": _q("capybara orange", "capybara chill", "capybara")},
        {"name": "Chill Capybara", "queries": _q("capybara relax", "chill capybara", "capybara")},
        {"name": "Cool Capybara", "queries": _q("capybara sunglasses", "cool capybara", "capybara")},
        {"name": "Buff Capybara", "queries": _q("buff capybara", "capybara gym", "capybara")},
        {"name": "Capybara Chef", "queries": _q("capybara chef", "capybara cook", "capybara")},
        {"name": "Capybara King", "queries": _q("capybara king", "capybara crown", "capybara")},
        {"name": "Capybara God", "queries": _q("capybara space", "capybara galaxy", "capybara")}]},
    "cursed_smiley": {"title": "Cursed Smiley", "hook": "How cursed does it get?", "steps": [
        {"name": "Smiley", "queries": _q("smiley face", "happy face", "smile")},
        {"name": "Sweaty Smiley", "queries": _q("sweating emoji", "nervous face", "sweat")},
        {"name": "Cursed Smiley", "queries": _q("cursed emoji", "cursed smiley", "weird face")},
        {"name": "Melting Face", "queries": _q("melting face", "melting smiley", "melt")},
        {"name": "Clown", "queries": _q("clown face", "clown emoji", "clown")},
        {"name": "Skull", "queries": _q("skull emoji", "dancing skull", "skull")},
        {"name": "Ghost", "queries": _q("cute ghost", "ghost sticker", "ghost")},
        {"name": "Demon", "queries": _q("devil emoji", "cute devil", "demon")}]},
}


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
                   {"template": d.name, "hook": info.get("hook"), "credit": info.get("credit")})
        added.append(d.name)
    return added


def _tries_path(store: PackStore) -> Path:
    return store.root / "_starter_tries.json"


def next_meme_plan(store: PackStore) -> Optional[tuple]:
    """(key, plan) of the meme starter pack to build next: one not built yet,
    the least-tried first. None once they're all built."""
    have = {p.get("template") for p in store.packs()}
    try:
        tries = json.loads(_tries_path(store).read_text())
    except (OSError, ValueError):
        tries = {}
    todo = [k for k in MEME_PLANS if k not in have]
    if not todo:
        return None
    key = min(todo, key=lambda k: (tries.get(k, {}).get("n", 0), list(MEME_PLANS).index(k)))
    return key, MEME_PLANS[key]


def note_try(store: PackStore, key: str, ok: bool) -> None:
    try:
        tries = json.loads(_tries_path(store).read_text())
    except (OSError, ValueError):
        tries = {}
    t = tries.setdefault(key, {"n": 0})
    t["n"] += 1
    t["last"] = time.time()
    t["ok"] = ok
    store.root.mkdir(parents=True, exist_ok=True)
    _tries_path(store).write_text(json.dumps(tries, indent=1))
