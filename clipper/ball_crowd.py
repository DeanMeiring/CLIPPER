"""Comment fights (31 Oct): the viewers who comment are the balls.

Dean wanted something viewers relate to, not "Royal Purple VS Cherry Red":
the people who comment on the channel's videos get into a next video as a
ball with their own name and a character picture. Every day's 15:00 video is
one (the autopilot's ``crowd_slot``); the 08:00 one keeps following the
learned mix.

* Comments: the channel's own videos from the last ``LOOKBACK_DAYS`` are read
  with YouTube's commentThreads.list (the app's ``youtube.readonly`` scope, 1
  quota unit per video), top-level comments only, not the channel's own.
* Check: Claude looks at every new name and comment once (``moderate``):
  the name is fine to show, a short line from the comment can be shown as
  trash talk, or the person asked to be left out. Without a Claude key a word
  list checks the names and no lines are shown.
* Cast (``cast``): 3+ people waiting -> a battle royale of 16 (the champion,
  the waiting ones oldest first, then people who were in before, then bots);
  fewer -> a 1v1 duel, the champion against the next one waiting, each with
  their comment as a speech card. The champion is whoever won the last
  comment fight; they defend the crown.
* Each person always gets the same character (one of ``CHARACTERS``, the
  bundled Fluent 3D emoji) and colour, from their channel id.
* Call-outs (1 Nov, Dean): a comment like "me vs levi" asks for a fight.
  Claude spots them in ``moderate`` (``challenge``: the name as written, or
  "CHAMPION"); ``callouts`` matches the name to someone who commented
  (``match_name``) and ``callout_cast`` makes that 1v1. The app checks once
  an evening, 5 hours after the 15:00 fight, and posts the call-out right
  away; without one nothing extra posts and the next 15:00 fight is king of
  the hill as before. The crown only changes hands in a call-out when the
  champion is in it.

YouTube's fake-engagement policy bans content whose only purpose is to reward
comments; a full fight that stars its commenters isn't that (Marble Drop Daily
runs the same idea with marbles), but there are no prizes, nothing asks for
likes or subscriptions, and "remove me" takes a person out.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageDraw

from clipper import ball_duel as bd

API = "https://www.googleapis.com/youtube/v3/commentThreads"
LOOKBACK_DAYS = 21
ROYALE_MIN = 3                    # people waiting for a 16-ball royale; fewer: a 1v1
ROYALE_N = 16
NAME_MAX = 18
QUOTE_MAX = 48

# one character per person (their channel id picks it), all bundled Fluent 3D
CHARACTERS = ["1f436", "1f431", "1f42d", "1f439", "1f430", "1f98a", "1f43b", "1f43c", "1f428", "1f42f", "1f981",
              "1f42e", "1f437", "1f438", "1f435", "1f414", "1f427", "1f424", "1f984", "1f432", "1f47d", "1f47b",
              "1f383", "1f480", "1f921", "1f47e", "1f60e", "1f920", "1f996", "1f419", "1f988", "1f989", "1f43a",
              "1f99d", "1f417", "1f434", "1f98b", "1f41d", "1f422", "1f98e", "1f99c", "1f9a9", "1f9a5", "1f42c",
              "1f9d9", "1f977", "1f9db", "1f9df", "1f9de", "1f911", "1f913", "1f973", "1f976", "1f9cc", "1f985"]
BOT = "1f916"
BOT_COL = (150, 156, 172)
PALETTE = [c[2] for c in bd.COLOURS + bd.EXTRA_COLOURS if c[1] not in ("Black", "Grey")]

# nobody has commented yet (Dean, 1 Nov: the 15:00 fight still goes out,
# and says "comment to join the league")
EMPTY_HOOKS = ["The league is empty. Comment to join!", "Bots hold every spot. Comment to join the league!",
               "Your name could be on this ball. Comment to join!"]
ROYALE_HOOKS = ["16 commenters. 1 ring. 1 winner.", "You commented. Now you fight.",
                "Comments become fighters. Who wins?"]
ROYALE_CHAMP_HOOKS = ["Can anyone take {c}'s crown?", "{c} defends the crown against 15 of you"]
DUEL_HOOKS = ["{a} VS {b}", "{a} VS {b}: who wins?"]
DUEL_CHAMP_HOOKS = ["Champ {a} VS {b}", "{b} comes for {a}'s crown"]
# call-outs: {a} asked for the fight, {b} is who they called out
CALLOUT_HOOKS = ["{a} called out {b}", "{a} wanted {b}. Here it is", "You asked for it: {a} VS {b}"]
CALLOUT_CHAMP_HOOKS = ["{a} called out the champ {b}", "{a} comes for {b}'s crown"]

_BAD = re.compile(r"(fuck|shit|bitch|cunt|nigg|fag|retard|whore|slut|porn|sex|nazi|hitler|rape|kill ?yourself|kys|"
                  r"https?://|www\.|\.com\b|onlyfans|t\.me/|whatsapp|telegram)", re.I)
_OPT_OUT = re.compile(r"(remove me|take me out|don'?t (use|put|show) my name|leave me out|not in (the|your) video)",
                      re.I)


class Handle(str):
    """A name that keeps its own case where a painter upper()s names."""
    def upper(self):
        return self


def _h(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest(), 16)


def character_for(key: str) -> str:
    return CHARACTERS[_h(key) % len(CHARACTERS)]


def colour_for(key: str) -> tuple:
    return tuple(PALETTE[_h("col-" + key) % len(PALETTE)])


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    return name if len(name) <= NAME_MAX else name[:NAME_MAX - 1] + "…"


# ---- comments ----------------------------------------------------------------

def fetch(token: Optional[str], video_ids: List[str], own_channel: Optional[str], http=None,
          log: Optional[Callable[[str], None]] = None, api_key: Optional[str] = None,
          stats: Optional[dict] = None) -> List[dict]:
    """Top-level comments on these videos (newest 100 each).

    With ``api_key`` (YOUTUBE_API_KEY) the public comments are read without
    the channel's login: on 1 Nov every read with the channel's token came
    back 403 -- its youtube.readonly scope doesn't cover commentThreads.list,
    which wants youtube.force-ssl -- while a key reads public comments with
    no account permission at all."""
    import requests
    http = http or requests
    out = []
    skipped = []
    for vid in video_ids:
        params = {"part": "snippet,replies", "videoId": vid, "maxResults": 100, "order": "time", "textFormat": "plainText"}
        if api_key:
            r = http.get(API, params={**params, "key": api_key}, timeout=30)
        else:
            r = http.get(API, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
        if r.status_code in (403, 404):
            reason = ""
            try:
                reason = ((r.json().get("error") or {}).get("errors") or [{}])[0].get("reason") or ""
            except ValueError:
                pass
            skipped.append(f"{vid}:{r.status_code}{' ' + reason if reason else ''}")
            continue                # comments off, the video is gone, or no access
        r.raise_for_status()
        for item in r.json().get("items") or []:
            top = (item.get("snippet") or {}).get("topLevelComment") or {}
            # the replies YouTube sends along (up to 5 a thread): "me vs levi"
            # is often written under Levi's own comment
            for c in [top] + list((item.get("replies") or {}).get("comments") or []):
                sn = c.get("snippet") or {}
                ch = (sn.get("authorChannelId") or {}).get("value")
                if not ch or ch == own_channel:
                    continue
                out.append({"id": c.get("id") or item.get("id"), "video": vid, "channel": ch,
                            "name": sn.get("authorDisplayName") or "",
                            "text": (sn.get("textOriginal") or sn.get("textDisplay") or "")[:500],
                            "at": sn.get("publishedAt") or ""})
    if stats is not None:
        stats.update(refused=len(skipped), first=skipped[:1])
    if log:
        log(f"[crowd] read {len(out)} comments on {len(video_ids)} videos"
            + (f" ({len(skipped)} refused: {', '.join(skipped[:5])})" if skipped else ""))
    return out


def update_people(store: dict, comments: List[dict]) -> int:
    """Add the comments' authors to the store (one entry per channel, their
    newest comment). Returns how many are new or changed."""
    people = store.setdefault("people", {})
    changed = 0
    for c in sorted(comments, key=lambda c: c["at"]):
        p = people.get(c["channel"])
        if p is None:
            people[c["channel"]] = {"channel": c["channel"], "name": clean_name(c["name"]), "text": c["text"],
                                    "comment": c["id"], "video": c["video"], "first_at": c["at"],
                                    "last_at": c["at"], "checked": False}
            changed += 1
        elif c["at"] > p.get("last_at", "") and c["id"] != p.get("comment"):
            p.update(name=clean_name(c["name"]) or p["name"], text=c["text"], comment=c["id"], video=c["video"],
                     last_at=c["at"], checked=False)
            changed += 1
    return changed


_MOD_PROMPT = """These people commented on a YouTube Shorts channel of bouncing-ball \
battles. Each may be put in a next video as a ball with their name, and a short line \
from their comment may be shown as their trash talk. For each one decide:

- "name_ok": false if the name is a slur, sexual, hateful, a scam/link/phone number, \
or pretends to be a famous person or brand. Ordinary handles are fine.
- "quote": a short line from THEIR comment (at most {qmax} characters, their own words, \
you may shorten it), fun as trash talk or a cheer, or null. Null when the comment is \
just an emoji, a question to the channel, a complaint, personal info, a link, insults \
at a person or group, anything sexual, or anything you wouldn't show a 12-year-old.
- "opt_out": true if they ask not to be in a video or to be removed.
- "challenge": if they ask to fight one particular person ("me vs levi", "1v1 me @sam", \
"put me against blueman", "levi vs me"), that person's name exactly as they wrote it \
(without the @); "CHAMPION" if they want to fight the champion / the winner / the king; \
else null. Null when they pair up two other people or just cheer for someone.

{rows}

Answer with JSON only: {{"people": [{{"id": <number>, "name_ok": true/false, "quote": "..." or null, \
"opt_out": true/false, "challenge": "..." or null}}]}}"""


_LINK = re.compile(r"(https?://|www\.|t\.me/|\b[\w-]+\.(com|net|org|io|ly|me|xyz|shop)\b)", re.I)


_ME_VS = re.compile(r"(?:^|\b)(?:me|i)\s*(?:vs\.?|versus|v\.?|against)\s*@?([\w][\w.\-]{1,29})|"
                    r"1\s*v\s*1\s+me\s+@?([\w][\w.\-]{1,29})|"
                    r"@?([\w][\w.\-]{1,29})\s+(?:vs\.?|versus|v\.?)\s+me\b", re.I)
_CHAMP_WORDS = {"champ", "champion", "the", "winner", "king", "you"}


def challenge_in(text: str) -> Optional[str]:
    """The word list's call-out spotting (no Claude key): "me vs levi" -> "levi"."""
    m = _ME_VS.search(text or "")
    if not m:
        return None
    name = next(g for g in m.groups() if g)
    if name.lower() in ("me", "i"):
        return None
    return "CHAMPION" if name.lower() in _CHAMP_WORDS else name


def _fallback(p: dict) -> dict:
    # a link in the comment reads as spam: that person stays out
    return {"name_ok": not _BAD.search(p["name"]) and not _LINK.search(p["text"]), "quote": None,
            "opt_out": bool(_OPT_OUT.search(p["text"])), "challenge": challenge_in(p["text"])}


def _set_challenge(p: dict, target) -> None:
    """Keep a call-out on the person until it's fought: a later comment that
    isn't one ("lol") doesn't wipe it."""
    target = target.strip().lstrip("@").strip() if isinstance(target, str) else ""
    if target and len(target) <= 40:
        p["challenge"] = {"target": target, "comment": p.get("comment"), "at": p.get("last_at") or ""}


def moderate(store: dict, api_key: Optional[str], claude_json=None,
             log: Optional[Callable[[str], None]] = None) -> int:
    """Check everyone not checked yet; returns how many were checked."""
    todo = [p for p in (store.get("people") or {}).values() if not p.get("checked")]
    for k in range(0, len(todo), 40):
        batch = todo[k:k + 40]
        verdicts: Dict[int, dict] = {}
        if api_key:
            rows = "\n".join(f'{i}. name: {json.dumps(p["name"])} comment: {json.dumps(p["text"][:300])}'
                             for i, p in enumerate(batch))
            try:
                if claude_json is None:
                    from clipper.ball_packs import _claude_json as claude_json
                ans = claude_json(_MOD_PROMPT.format(qmax=QUOTE_MAX, rows=rows), api_key, max_tokens=6000)
                for v in ans.get("people") or []:
                    if isinstance(v.get("id"), int):
                        verdicts[v["id"]] = v
            except Exception as e:        # a failed check: names by the word list, no lines
                if log:
                    log(f"[crowd] Claude's check failed, using the word list: {e}")
        for i, p in enumerate(batch):
            v = verdicts.get(i)
            fb = _fallback(p)
            if v is None:
                ch = fb.pop("challenge")
                p.update(fb)
                _set_challenge(p, ch)
            else:
                q = v.get("quote")
                q = q.strip().strip('"').strip() if isinstance(q, str) else None
                if q and (len(q) > QUOTE_MAX or _BAD.search(q)):
                    q = None
                # the word list still has a say: either one saying no is a no
                p.update(name_ok=bool(v.get("name_ok")) and fb["name_ok"], quote=q or None,
                         opt_out=bool(v.get("opt_out")) or fb["opt_out"])
                _set_challenge(p, v.get("challenge"))
            p["checked"] = True
    return len(todo)


# ---- who's in it ---------------------------------------------------------------

def _usable(p: dict, blocked) -> bool:
    return p.get("checked") and p.get("name_ok") and not p.get("opt_out") and p["channel"] not in blocked


def waiting(store: dict, used: set, blocked=()) -> List[dict]:
    """People who commented and haven't been in a video yet, oldest first."""
    ps = [p for p in (store.get("people") or {}).values() if _usable(p, blocked) and p["channel"] not in used]
    return sorted(ps, key=lambda p: p.get("first_at", ""))


def returning(store: dict, used: set, blocked=()) -> List[dict]:
    """People who were in a video before, most recent comment first."""
    ps = [p for p in (store.get("people") or {}).values() if _usable(p, blocked) and p["channel"] in used]
    return sorted(ps, key=lambda p: p.get("last_at", ""), reverse=True)


def _member(p: dict, quote: bool = False) -> dict:
    return {"name": p["name"], "channel": p["channel"], "char": character_for(p["channel"]),
            "col": list(colour_for(p["channel"])), "quote": p.get("quote") if quote else None, "bot": False}


def _bot(k: int) -> dict:
    return {"name": f"Bot {k}", "channel": None, "char": BOT, "col": list(BOT_COL), "quote": None, "bot": True}


def cast(store: dict, used: set, blocked=(), champ: Optional[dict] = None) -> dict:
    """The next comment fight: {look, cast, champ (index or None), fresh}."""
    fresh = waiting(store, used, blocked)
    back = returning(store, used, blocked)
    people = store.get("people") or {}
    if champ and champ.get("channel"):
        p = people.get(champ["channel"])
        if not p or not _usable(p, blocked):
            champ = None                    # left, or blocked: the crown is open
    if champ:
        back = [p for p in back if p["channel"] != champ.get("channel")]
        fresh = [p for p in fresh if p["channel"] != champ.get("channel")]
    look = "royale" if len(fresh) >= ROYALE_MIN else "duel"
    n = ROYALE_N if look == "royale" else 2
    duel = look == "duel"
    members: List[dict] = []
    if champ:
        p = people.get(champ.get("channel") or "")
        members.append(_member(p, duel) if p else dict(champ, quote=None))
    for p in fresh + back:
        if len(members) >= n:
            break
        members.append(_member(p, duel))
    k = 1
    while len(members) < n:
        members.append(_bot(k))
        k += 1
    return {"look": look, "cast": members, "champ": 0 if champ else None,
            "fresh": sum(1 for m in members if m["channel"] and m["channel"] in {p["channel"] for p in fresh})}


# ---- call-outs ("me vs levi") -----------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def match_name(store: dict, target: str, blocked=(), exclude: Optional[str] = None) -> Optional[dict]:
    """The person a call-out names. YouTube handles carry a random tail
    ("@Levi-k4z8j"), and people write "levi": the whole handle, the handle
    without its tail, the part before the first - / _ / . or the start of the
    handle all count (3+ letters). Several match: the one who commented last."""
    t = _norm(target)
    if len(t) < 3:
        return None
    hits = []
    for p in (store.get("people") or {}).values():
        if p["channel"] == exclude or not _usable(p, blocked):
            continue
        raw = (p.get("name") or "").lstrip("@").lower().rstrip("…")
        full = _norm(raw)
        head = _norm(re.split(r"[-_.\s]", raw)[0])
        notail = _norm(re.sub(r"-[a-z0-9]{3,6}$", "", raw))
        if t in (full, head, notail):
            hits.append((0, p))
        elif full.startswith(t) and len(t) >= 4:
            hits.append((1, p))
    if not hits:
        return None
    best = min(h[0] for h in hits)
    return max((p for k, p in hits if k == best), key=lambda p: p.get("last_at", ""))


def callouts(store: dict, done: set, blocked=(), champ: Optional[dict] = None, fought: Optional[dict] = None) -> List[dict]:
    """Call-outs still to be fought, oldest first: [{by, vs, comment, at}].
    ``done``: comment ids already fought; ``fought``: frozenset of the two
    channels -> when (ISO) they last had a call-out fight -- asking again for
    a pair that already fought since doesn't count twice."""
    out = []
    people = store.get("people") or {}
    for p in people.values():
        c = p.get("challenge")
        if not c or not _usable(p, blocked) or c.get("comment") in done:
            continue
        if c["target"] == "CHAMPION":
            q = people.get((champ or {}).get("channel") or "")
            q = q if q and q["channel"] != p["channel"] and _usable(q, blocked) else None
        else:
            q = match_name(store, c["target"], blocked, exclude=p["channel"])
        if not q:
            continue
        last = (fought or {}).get(frozenset((p["channel"], q["channel"])))
        if last and last >= (c.get("at") or ""):
            continue
        out.append({"by": p["channel"], "vs": q["channel"], "comment": c.get("comment"), "at": c.get("at") or ""})
    return sorted(out, key=lambda c: c["at"])


def callout_cast(store: dict, callout: dict, blocked=(), champ: Optional[dict] = None) -> Optional[dict]:
    """The call-out's 1v1 (a ``cast`` block): the champion on the left when
    they're in it (then it's for the crown), else the one who asked."""
    people = store.get("people") or {}
    by, vs = people.get(callout["by"]), people.get(callout["vs"])
    if not (by and vs and _usable(by, blocked) and _usable(vs, blocked)):
        return None
    pair = [by, vs]
    champ_ch = (champ or {}).get("channel")
    if champ_ch == vs["channel"]:
        pair = [vs, by]
    members = [_member(p, True) for p in pair]
    return {"look": "duel", "cast": members, "champ": 0 if champ_ch in (by["channel"], vs["channel"]) else None,
            "fresh": 0, "callout": {"by": by["channel"], "vs": vs["channel"], "comment": callout.get("comment"),
                                    "at": callout.get("at") or "", "by_i": pair.index(by)}}


# ---- the recipe -------------------------------------------------------------------

def apply(recipe: dict, block: dict) -> dict:
    """Make a duel / royale recipe (already drawn by ball_duel.duel_recipe /
    royale_recipe) this comment fight."""
    rng = random.Random(f"crowd-{recipe['seed']}")
    members = block["cast"]
    recipe["players"] = [[m["name"], "", list(m["col"]), None, m["char"]] for m in members]
    recipe.pop("look", None)
    recipe.pop("team_names", None)
    champ = members[0]["name"] if block.get("champ") is not None else None
    co = block.get("callout")
    if co:
        a, b = members[co["by_i"]]["name"], members[1 - co["by_i"]]["name"]
        champ_called = block.get("champ") is not None and co["by_i"] == 1
        hook = rng.choice(CALLOUT_CHAMP_HOOKS if champ_called else CALLOUT_HOOKS).format(a=a, b=b)
    elif not any(not m.get("bot") for m in members):
        hook = rng.choice(EMPTY_HOOKS)
    elif block["look"] == "royale":
        # "16 commenters" / "15 of you" only when nobody is a bot
        bots = any(m.get("bot") for m in members)
        pool = (ROYALE_CHAMP_HOOKS[:1] if bots else ROYALE_CHAMP_HOOKS) if champ else \
            (ROYALE_HOOKS[1:] if bots else ROYALE_HOOKS)
        hook = rng.choice(pool).format(c=champ)
    else:
        a, b = members[0]["name"], members[1]["name"]
        hook = (rng.choice(DUEL_CHAMP_HOOKS) if champ else rng.choice(DUEL_HOOKS)).format(a=a, b=b)
    recipe.update(format="crowd", crowd=block, hook=hook, ending="WINS!", bed="off")
    return recipe


def base_format(block: dict) -> str:
    return "royale" if block["look"] == "royale" else "duel"


def publish_text(recipe: dict) -> dict:
    block = recipe["crowd"]
    royale = block["look"] == "royale"
    title = f"{recipe['hook']} {'💬' if royale else '⚔️'}"
    real = [m["name"] for m in block["cast"] if not m.get("bot")]
    co = block.get("callout")
    if co:
        a, b = block["cast"][co["by_i"]]["name"], block["cast"][1 - co["by_i"]]["name"]
        crown = " The champion is in it, so the crown is on the line." if block.get("champ") is not None else ""
        who = f"{a} called out {b} in the comments, so here it is.{crown} Want a fight? Comment \"me vs\" and a name"
    elif not real:
        who = "Nobody has joined the league yet, so bots are holding the spots"
    elif len(real) == len(block["cast"]):
        who = "Everyone in this video commented on one of our videos"
    else:
        who = "The named balls in this video commented on one of our videos; bots hold the open spots"
    desc = (f"{who}. Comment anything here on YouTube to join the league: your name goes on a ball in a next fight 👇 "
            f"(just your name, no prizes). The winner keeps the crown until someone beats them.\n\n"
            f"Don't want to be in one? Comment \"remove me\".\n\n"
            f"#satisfying #bouncingball #battle #comments")
    return {"title": title[:100], "description": desc}


def _real_wins(sim) -> None:
    """A bot never takes the crown while real commenters are in the fight:
    such a run doesn't count as a good one (pick_seed tries another)."""
    cast_ = sim.recipe["crowd"]["cast"]
    if sim.winner is not None and cast_[sim.winner["i"]].get("bot") and any(not m.get("bot") for m in cast_):
        sim.ok = False


class CrowdRoyaleSim(bd.RoyaleSim):
    def _end(self, winner):
        super()._end(winner)
        _real_wins(self)


class CrowdDuelSim(bd.DuelSim):
    def _end(self, winner):
        super()._end(winner)
        _real_wins(self)


def parts(recipe: dict):
    if recipe["crowd"]["look"] == "royale":
        return CrowdRoyaleSim, CrowdRoyalePainter
    return CrowdDuelSim, CrowdDuelPainter


def good_end(recipe: dict) -> tuple:
    if recipe["crowd"]["look"] == "royale":
        return (15.0, 34.0) if recipe.get("floor") else (18.0, 38.0)
    return (26.0, 50.0)


# ---- drawing -------------------------------------------------------------------------

W, H = bd.W, bd.H
CTA = "Comment to join the league"
GOLD = (255, 205, 60)
CARD_S = 4.0                      # how long the comment cards stay up
# Dean (31 Oct): say it at the start too -- most viewers swipe before the
# end button. A red banner for the first INVITE_S seconds, then it fades.
INVITE = "Comment to join the league"
INVITE_S = 4.0
# what the other videos' descriptions say (``invite``), so their viewers
# know they can get in
INVITE_LINE = ("💬 Want your name on a ball? Comment on any of our YouTube Shorts to join the league: "
               "you'll be in one of the next fights.")
_INVITE_IMG: dict = {}


def invite(desc: str) -> str:
    """The invite line in a description, before its hashtags."""
    if INVITE_LINE in desc:
        return desc
    head, sep, tags = desc.rpartition("\n\n#")
    if sep:
        return f"{head}\n\n{INVITE_LINE}{sep}{tags}"
    return f"{desc}\n\n{INVITE_LINE}" if desc else INVITE_LINE


def _invite_img() -> Image.Image:
    im = _INVITE_IMG.get("pill")
    if im is None:
        f = bd._font(34)
        tw = int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(INVITE, font=f))
        w, h = tw + 46 + 56, 64
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((2, 2, w - 3, h - 3), 30, fill=(230, 40, 64, 255), outline=(255, 255, 255, 255), width=4)
        im.alpha_composite(bd._emoji("1f4ac", 40), (24, 12))
        d.text((24 + 40 + 14, h / 2), INVITE, font=f, fill=(255, 255, 255, 255), anchor="lm")
        im = _INVITE_IMG["pill"] = im
    return im


def _invite(im: Image.Image, t: float, y: int, keep: bool = False) -> None:
    """The start banner, centred on y, while t < INVITE_S (+ a 0.4 s fade);
    ``keep``: the whole fight (bots hold spots: there's room to join)."""
    a = 1.0 if t < INVITE_S or keep else max(0.0, 1.0 - (t - INVITE_S) / 0.4)
    if a <= 0:
        return
    g = _invite_img()
    if a < 1:
        g = g.copy()
        g.putalpha(g.getchannel("A").point(lambda v: int(v * a)))
    im.alpha_composite(g, (int(W / 2 - g.width / 2), int(y - g.height / 2)))


def _has_bots(recipe: dict) -> bool:
    return any(m.get("bot") for m in recipe["crowd"]["cast"])


def _names(recipe: dict) -> None:
    for p in recipe.get("players") or []:
        p[0] = Handle(p[0])


def _cta(d, y, font):
    tw = d.textlength(CTA, font=font)
    d.rounded_rectangle((W / 2 - tw / 2 - 40, y - 44, W / 2 + tw / 2 + 40, y + 44), 44,
                        fill=(230, 40, 64, 255), outline=(255, 255, 255, 255), width=5)
    d.text((W / 2, y), CTA, font=font, fill=(255, 255, 255, 255), anchor="mm")


def _ending(painter, im, d, sim, y_win, y_sub, y_cta):
    w = sim.winner
    age = sim.t - sim.done_at
    if age < 0.5:
        return
    k = min(1.0, (age - 0.5) / 0.3)
    name = f"{w['name']} WINS!"
    f = bd._font(96)
    while d.textlength(name, font=f) > W - 120 and f.size > 40:
        f = bd._font(f.size - 4)
    s = 0.6 + 0.4 * k
    txt = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text((W / 2, 100), name, font=f, fill=(255, 255, 255, 255), anchor="mm",
                             stroke_width=10, stroke_fill=bd._stroke(w["col"]) + (255,))
    txt = txt.resize((int(W * s), int(200 * s)), Image.LANCZOS)
    im.alpha_composite(txt, (int((W - txt.width) / 2), int(y_win - txt.height / 2)))
    if age > 0.8:
        crowd = painter.recipe["crowd"]
        champ = crowd.get("champ") is not None
        bot = crowd["cast"][w["i"]].get("bot")
        callout = crowd.get("callout") and not champ
        sub = ("Comment to take its spot!" if bot else
               "Wins the call-out!" if callout else
               "Keeps the crown. Defends it next fight." if champ and w["i"] == 0 else
               "Takes the crown! Defends it next fight." if champ else "Wins the crown. Defends it next fight.")
        d.text((W / 2, y_sub), sub, font=painter.f_sub, fill=GOLD + (255,), anchor="mm", stroke_width=5,
               stroke_fill=(20, 16, 6, 255))
        _cta(d, y_cta, painter.f_cta)


class CrowdRoyalePainter(bd.RoyalePainter):
    def __init__(self, recipe: dict):
        _names(recipe)
        super().__init__(recipe)
        self.f_tag = bd._font(22)
        self.f_lab = bd._font(24)
        self.f_out = bd._font(56)
        self.f_cta = bd._font(40)
        self.f_sub = bd._font(40)
        self.out_feed: list = []
        self.crown = bd._emoji("1f451", 44)
        self.pics: dict = {}

    def hud(self, im, d, sim):
        hook = self.recipe.get("hook") or ""
        f = bd._font(50)
        while d.textlength(hook, font=f) > W - 70 and f.size > 30:
            f = bd._font(f.size - 2)
        d.text((W / 2, 70), hook, font=f, fill=(255, 255, 255, 255), anchor="mm", stroke_width=5,
               stroke_fill=(70, 60, 160, 255))
        d.text((W / 2, 122), "every name here is someone who commented", font=self.f_tag,
               fill=(190, 196, 230, 255), anchor="mm")
        champ = self.recipe["crowd"].get("champ")
        for b in sim.balls:
            x, y = 76 + (b["i"] % 8) * 132, 200 + (b["i"] // 8) * 122
            r = 36
            if not b["alive"] and b["i"] not in self.dead_at:
                self.dead_at[b["i"]] = sim.t
                self.out_feed.append((sim.t, b["name"]))
            col = b["col"] if b["alive"] else tuple(int(v * 0.3 + 20) for v in b["col"])
            d.ellipse((x - r, y - r, x + r, y + r), fill=col + (255,),
                      outline=(255, 255, 255, 255) if b["alive"] else (60, 64, 90), width=3)
            if b.get("pic"):
                key = (b["pic"], b["alive"])
                g = self.pics.get(key)
                if g is None:
                    g = self.pics[key] = self._pic(b["pic"], 54, b["alive"])
                im.alpha_composite(g, (int(x - 27), int(y - 27)))
            if champ is not None and b["i"] == champ:
                im.alpha_composite(self.crown, (int(x - 22 + 18), int(y - r - 30)))
            d.text((x, y + 52), b["name"], font=self.f_tag, fill=(235, 238, 255, 255 if b["alive"] else 90),
                   anchor="mm")
            if not b["alive"]:
                q = 22
                d.line([(x - q, y - q), (x + q, y + q)], fill=(230, 60, 70, 255), width=6)
                d.line([(x - q, y + q), (x + q, y - q)], fill=(230, 60, 70, 255), width=6)
        d.text((W / 2, 470), f"{sim.left()} LEFT", font=self.f_left, fill=(255, 255, 255, 255), anchor="mm",
               stroke_width=6, stroke_fill=(200, 50, 70, 255))

    @staticmethod
    def _pic(code, size, alive):
        g = bd._emoji(code, size)
        if not alive:
            a = g.getchannel("A").point(lambda v: v * 35 // 100)
            g.putalpha(a)
        return g

    def ending(self, im, d, sim):
        # the button above the bottom ~20% YouTube covers with the title and buttons
        _ending(self, im, d, sim, bd.D_CY + 230, bd.D_CY + 330, bd.D_CY + 440)

    def frame(self, sim):
        im = super().frame(sim).convert("RGBA")
        d = ImageDraw.Draw(im)
        if sim.winner is None:
            for b in sim.balls:            # each ball's name under it
                if b["alive"]:
                    x = bd.D_CX + (b["x"] - bd.D_CX) * self.k
                    y = bd.D_CY + (b["y"] - bd.D_CY) * self.k
                    d.text((x, y + 50), b["name"], font=self.f_lab, fill=(255, 255, 255, 255), anchor="mm",
                           stroke_width=4, stroke_fill=(10, 12, 30, 255))
            recent = [n for t, n in self.out_feed if sim.t - t < 1.2]
            if recent:
                d.text((W / 2, 1720), f"{recent[-1]} IS OUT", font=self.f_out, fill=(255, 255, 255, 255),
                       anchor="mm", stroke_width=7, stroke_fill=(200, 40, 60, 255))
        _invite(im, sim.t, 128, keep=sim.winner is None and _has_bots(self.recipe))
        return im.convert("RGB")


class CrowdDuelPainter(bd.DuelPainter):
    UPPER_NAMES = False

    def __init__(self, recipe: dict):
        _names(recipe)
        super().__init__(recipe)
        self.f_cta = bd._font(40)
        self.f_sub = bd._font(40)
        self.f_q = bd._font(26)
        self.f_qn = bd._font(22)
        self.crown = bd._emoji("1f451", 64).rotate(18, resample=Image.BICUBIC, expand=True)
        self.cards = None

    def hud_clean(self, im, d, sim):
        super().hud_clean(im, d, sim)
        if self.recipe["crowd"].get("champ") is not None:
            im.alpha_composite(self.crown, (12, 104))

    def _card(self, k, b, quote):
        """A comment card like YouTube's: the character, the name, their line."""
        wmax = W // 2 - 44
        f = self.f_q
        lines, cur = [], ""
        for word in f"“{quote}”".split():
            t = (cur + " " + word).strip()
            if ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(t, font=f) > wmax - 96 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = t
        lines.append(cur)
        lines = lines[:2]
        h = 40 + 32 * len(lines) + 14
        card = Image.new("RGBA", (wmax, h + 22), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((0, 22 if k == 0 else 22, wmax - 1, h + 21), 22, fill=(255, 255, 255, 245))
        tx = (34, 58) if k == 0 else (wmax - 58, wmax - 34)
        cd.polygon([(tx[0], 23), (tx[1], 23), ((tx[0] + tx[1]) / 2 + (-14 if k == 0 else 14), 0)],
                   fill=(255, 255, 255, 245))
        cd.ellipse((14, 36, 82, 104), fill=tuple(b["col"]) + (255,))
        card.alpha_composite(bd._emoji(b["pic"], 54), (21, 43))
        cd.text((96, 50), b["name"], font=self.f_qn, fill=(100, 104, 130, 255), anchor="lm")
        for j, line in enumerate(lines):
            cd.text((96, 80 + j * 32), line, font=f, fill=(20, 22, 40, 255), anchor="lm")
        return card

    def frame(self, sim):
        im = super().frame(sim).convert("RGBA")
        if self.cards is None:
            self.cards = []
            for k, b in enumerate(sim.balls[:2]):
                q = self.recipe["crowd"]["cast"][k].get("quote")
                self.cards.append(self._card(k, b, q) if q and b.get("pic") else None)
        # their trash talk at the start, then out of the way of the fight
        a = 1.0 if sim.t < CARD_S else max(0.0, 1.0 - (sim.t - CARD_S) / 0.4)
        if a > 0:
            for k, c in enumerate(self.cards):
                if c is not None:
                    if a < 1:
                        c = c.copy()
                        c.putalpha(c.getchannel("A").point(lambda v: int(v * a)))
                    x = 30 if k == 0 else W - 30 - c.width
                    im.alpha_composite(c, (x, 440 + k * 18))
        _invite(im, sim.t, 92, keep=sim.winner is None and _has_bots(self.recipe))
        return im.convert("RGB")

    def ending(self, im, d, sim):
        _ending(self, im, d, sim, self.end_y, self.end_y + 100, self.end_y + 220)
