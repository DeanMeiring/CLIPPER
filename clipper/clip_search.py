"""Find streamer clips for a Caught On Code episode: real Twitch moments of
the thing the episode explains (a streamer dying behind a wall, a lag
spike, a desync), to play as "moment" scenes before the narration explains
what happened.

Twitch can't search clips by keyword -- only list a game's (or a
streamer's) top clips by views. So:

  search_terms   Claude names the games where players run into the topic
                 on stream, and words their clip titles would use
  game_clips     the games' top English clips: all time, plus the last
                 year in two-month windows (a few hundred titles)
  shortlist      keep the titles with those words
  pick           Claude picks up to 8 from the titles
  (download)     documentary.build_library downloads and transcribes them
  check          Claude reads what is said in each and marks whether it
                 really shows the topic, with a line on what happens

Dean can also paste Twitch clip links (resolve_links); those are always
used. Everything costs nothing but a few small Claude calls: the Twitch
API is free with the login the documentaries already use.
"""
from __future__ import annotations

import datetime
import re
from typing import Callable, Dict, List

from .documentary import _helix, _transcript_marked, clip_words

MAX_PICK = 8
WINDOWS = 6           # two-month windows over the last year, plus all time
_LINK = re.compile(r"(?:clips\.twitch\.tv/(?:embed\?clip=)?|twitch\.tv/[A-Za-z0-9_]+/clip/)([A-Za-z0-9_-]{6,})")


def _clean(s, n: int) -> str:
    return " ".join(str(s or "").split())[:n]


def _entry(c: dict, games: Dict[str, str]) -> dict:
    name = c.get("broadcaster_name") or ""
    return {
        "twitch_id": c["id"], "url": c.get("url"), "title": _clean(c.get("title"), 140),
        "views": int(c.get("view_count") or 0), "date": (c.get("created_at") or "")[:10],
        "duration": float(c.get("duration") or 0), "clipped_by": c.get("creator_name") or "",
        "streamer": name, "game": games.get(c.get("game_id"), ""), "thumbnail": c.get("thumbnail_url"),
    }


def credit(clip: dict) -> str:
    """On-screen and in-description credit for a clip's streamer."""
    name = clip.get("streamer") or ""
    return f"twitch.tv/{name.lower()}" if re.fullmatch(r"[A-Za-z0-9_]{2,25}", name) else name


# ------------------------------------------------------------ the search ---

def search_terms(topic: str) -> dict:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    prompt = f"""A YouTube channel explains the tech behind gaming, streaming and the
internet. This episode: "{topic}".

We want short Twitch clips where a streamer runs into this on stream (for
example dying after reaching cover, for an episode about netcode). Twitch
only lets us list a game's top clips, so we search clip titles.

Give:
- "games": 1 to 3 Twitch game names (exactly as Twitch spells them) where
  this happens on stream a lot. An empty list if the topic never shows up
  in a game on stream.
- "keywords": 8 to 15 short lowercase words or phrases that clip titles
  about it would contain ("lag", "desync", "how did i die", "behind the
  wall", "peek"...).

Respond with ONLY a JSON array holding one object:
[{{"games": ["..."], "keywords": ["..."]}}]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    obj = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, dict) else {})
    games = [_clean(g, 60) for g in obj.get("games") or [] if _clean(g, 60)][:3]
    words = [_clean(k, 30).lower() for k in obj.get("keywords") or [] if len(_clean(k, 30)) >= 3][:15]
    return {"games": games, "keywords": words}


def game_ids(names: List[str]) -> Dict[str, str]:
    if not names:
        return {}
    return {g["id"]: g["name"] for g in _helix("games", {"name": names}).get("data") or []}


def game_clips(game_id: str, now: datetime.datetime = None) -> List[dict]:
    """The game's top clips: all time, plus each of the last WINDOWS
    two-month windows (so recent moments get a chance next to old hits)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    found: dict = {}
    spans = [None] + [(now - datetime.timedelta(days=60 * (i + 1)), now - datetime.timedelta(days=60 * i)) for i in range(WINDOWS)]
    for span in spans:
        params = {"game_id": game_id, "first": 100}
        if span:
            params.update(started_at=span[0].strftime("%Y-%m-%dT%H:%M:%SZ"), ended_at=span[1].strftime("%Y-%m-%dT%H:%M:%SZ"))
        try:
            for c in _helix("clips", params).get("data") or []:
                found.setdefault(c["id"], c)
        except Exception as e:
            print(f"[clip_search] clips for game {game_id} failed: {e}", flush=True)
    return [c for c in found.values() if (c.get("language") or "en").startswith("en")]


def shortlist(clips: List[dict], keywords: List[str], limit: int = 60) -> List[dict]:
    """Clips whose titles contain the keywords, most matches then most views first."""
    pats = [re.compile(r"(?<![a-z0-9])" + re.escape(k)) for k in keywords if k]
    scored = []
    for c in clips:
        title = " ".join(re.sub(r"[^a-z0-9' ]+", " ", str(c.get("title") or "").lower()).split())
        score = sum(1 for p in pats if p.search(title))
        if score:
            scored.append((score, int(c.get("view_count") or c.get("views") or 0), c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _s, _v, c in scored[:limit]]


def pick(topic: str, candidates: List[dict], n: int = MAX_PICK) -> List[dict]:
    """Claude picks the titles most likely to show the topic happening."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    if not candidates:
        return []
    lines = [f"{c['twitch_id']} | {c['title']} | {c['streamer']} | {c['game']} | {c['views']:,} views | {c['duration']:.0f}s"
             for c in candidates]
    prompt = f"""A YouTube channel explains the tech behind gaming. This episode: "{topic}".

Twitch clips whose titles might show this happening to a streamer
(id | title | streamer | game | views | length):
{chr(10).join(lines)}

Pick up to {n} that most likely SHOW the thing the episode explains
happening on stream (not just a word in common). Skip clips that look like
jokes, reactions to other things, or anything mean about a person.

Respond with ONLY a JSON array, best first:
[{{"id": "...", "why": "one short line"}}]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    by_id = {c["twitch_id"]: c for c in candidates}
    out = []
    for it in data if isinstance(data, list) else []:
        cid = str((it or {}).get("id") or "") if isinstance(it, dict) else ""
        if cid in by_id and by_id[cid] not in out:
            out.append({**by_id[cid], "why": _clean(it.get("why"), 120)})
        if len(out) >= n:
            break
    return out


def check(topic: str, library: List[dict], project_dir) -> Dict[str, dict]:
    """After download: Claude reads what is said in each clip and says
    whether it really shows the topic, and what happens in it."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    if not library:
        return {}
    lines = [f"{c['id']} | \"{c['title']}\" by {c.get('streamer') or '?'} ({c['duration']:.0f}s)\n"
             f"    transcript: {_transcript_marked(clip_words(project_dir, c['id']), 500) or '(nobody talks)'}"
             for c in library]
    prompt = f"""A YouTube channel explains the tech behind gaming. This episode: "{topic}".

Streamer clips (id | title, then what is said with [seconds] markers):
{chr(10).join(lines)}

For each clip, say whether it really shows the thing the episode explains
happening on stream, and in one plain line what happens in it.

Respond with ONLY a JSON array:
[{{"id": "C01", "fits": true, "what": "..."}}]
"""
    try:
        data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    except Exception as e:
        print(f"[clip_search] check failed: {e}", flush=True)
        return {}
    out = {}
    for it in data if isinstance(data, list) else []:
        if isinstance(it, dict) and it.get("id"):
            out[str(it["id"]).upper()] = {"fits": bool(it.get("fits")), "what": _clean(it.get("what"), 140)}
    return out


def resolve_links(links: List[str]) -> List[dict]:
    """Twitch clip links Dean pasted, as clip entries."""
    slugs = []
    for link in links:
        m = _LINK.search(str(link))
        if m and m.group(1) not in slugs:
            slugs.append(m.group(1))
    if not slugs:
        return []
    clips = _helix("clips", {"id": slugs[:100]}).get("data") or []
    games = {}
    gids = sorted({c.get("game_id") for c in clips if c.get("game_id")})
    if gids:
        try:
            games = {g["id"]: g["name"] for g in _helix("games", {"id": gids}).get("data") or []}
        except Exception:
            pass
    return [_entry(c, games) for c in clips]


def find(topic: str, on_progress: Callable[[str], None] = lambda m: None) -> dict:
    """Search terms, the games' clips, the shortlist and Claude's picks
    (not downloaded yet). {"games", "keywords", "scanned", "picked"}."""
    on_progress("Choosing the games and words to search for...")
    terms = search_terms(topic)
    games = game_ids(terms["games"])
    if not games:
        return {**terms, "scanned": 0, "picked": []}
    clips = []
    for gid, name in games.items():
        on_progress(f"Reading the top {name} clips on Twitch...")
        clips += game_clips(gid)
    entries = [_entry(c, games) for c in clips]
    short = shortlist(entries, terms["keywords"])
    on_progress(f"Picking from {len(short)} clips whose titles match...")
    picked = pick(topic, short)
    return {"games": list(games.values()), "keywords": terms["keywords"], "scanned": len(entries), "picked": picked}
