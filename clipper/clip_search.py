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

YouTube is searched too (youtube_search): unlike Twitch it searches by
words, so it finds clips of the actual situation. Only short videos
(Shorts, stream clips) are kept, and Claude skips other channels'
explainers, reactions and news. A search costs about 400 of the 10,000
free daily YouTube API units, with the channel's own OAuth token.

The first version picked clips unrelated to the episode (Dean): the title
words were too generic and every step gave clips the benefit of the doubt.
Now the words must be specific to the situation, and both Claude steps
keep a clip only when it clearly shows the topic -- when unsure, it's out.

Dean can also paste Twitch or YouTube links (resolve_links); those are
always used. Everything is free but a few small Claude calls.
"""
from __future__ import annotations

import datetime
import re
from typing import Callable, Dict, List

from .documentary import _helix, _transcript_marked, clip_words

MAX_PICK = 8
WINDOWS = 6           # two-month windows over the last year, plus all time
_LINK = re.compile(r"(?:clips\.twitch\.tv/(?:embed\?clip=)?|twitch\.tv/[A-Za-z0-9_]+/clip/)([A-Za-z0-9_-]{6,})")
_YT_LINK = re.compile(r"(?:youtube\.com/(?:watch\?(?:[^ ]*&)?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})")
YT_MAX_SECONDS = 240  # Shorts and stream clips, not whole videos


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
    if clip.get("source") == "youtube":
        return f"YouTube · {name}" if name else "YouTube"
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
- "keywords": 8 to 15 lowercase phrases that clip titles about THIS
  situation would contain ("died behind the wall", "desync", "how did
  that hit", "peekers advantage"...). Be specific: no single generic words
  that also fit unrelated clips ("server", "game", "lag", "100", "players").
- "youtube": 3 or 4 YouTube searches that would find short clips of a
  streamer or player running into it (game name + the situation + "clip",
  e.g. "valorant died behind wall desync clip"). Not searches for
  explanations of it.

Respond with ONLY a JSON array holding one object:
[{{"games": ["..."], "keywords": ["..."], "youtube": ["..."]}}]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    obj = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, dict) else {})
    games = [_clean(g, 60) for g in obj.get("games") or [] if _clean(g, 60)][:3]
    words = [_clean(k, 30).lower() for k in obj.get("keywords") or [] if len(_clean(k, 30)) >= 3][:15]
    queries = [_clean(q, 100) for q in obj.get("youtube") or [] if len(_clean(q, 100)) >= 6][:4]
    return {"games": games, "keywords": words, "youtube": queries}


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
    lines = [f"{c['twitch_id']} | {c['title']} | {c['streamer']} | {c.get('game') or '-'} | {c['views']:,} views | {c['duration']:.0f}s"
             + (f"\n    description: {c['description']}" if c.get("description") else "") for c in candidates]
    prompt = f"""A YouTube channel explains the tech behind gaming. This episode: "{topic}".

Clips whose titles might show this happening to a streamer or player
(id | title | channel | game | views | length):
{chr(10).join(lines)}

Pick up to {n} whose title (and description, if given) CLEARLY says the
situation this episode explains happens in the clip. A word in common is
not enough: a clip titled "100 players dropped" doesn't show how a server
handles 100 players. When unsure, leave it out -- fewer good clips beat
many weak ones, and an empty list is fine. Skip jokes, reactions to other
things, anything mean about a person, and every video that explains,
reviews, reports on or reacts to the topic (other channels' explainers,
news, tutorials, compilations with commentary).

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

For each clip, say in one plain line what happens in it, and whether it
really shows the thing this episode explains happening to the player.
"fits" is true ONLY when the transcript or title make that clear. If it
could be about something else, or you're unsure, "fits" is false. A clip
that explains, reviews or reports on the topic (rather than showing it
happen) is false too.

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


# --------------------------------------------------------------- youtube ---

def _yt_get(path: str, params: dict, token: str) -> dict:
    import requests

    r = requests.get(f"https://www.googleapis.com/youtube/v3/{path}", params=params,
                     headers={"Authorization": f"Bearer {token}"}, timeout=20)
    r.raise_for_status()
    return r.json()


def _iso_seconds(d: str) -> float:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0.0
    dd, h, mi, se = (int(x or 0) for x in m.groups())
    return float(dd * 86400 + h * 3600 + mi * 60 + se)


def youtube_videos(ids: List[str], token: str) -> List[dict]:
    """Clip entries for YouTube video ids (title, channel, length, views)."""
    out = []
    for i in range(0, len(ids), 50):
        data = _yt_get("videos", {"part": "snippet,contentDetails,statistics", "id": ",".join(ids[i:i + 50])}, token)
        for v in data.get("items") or []:
            sn, st = v.get("snippet") or {}, v.get("statistics") or {}
            thumbs = sn.get("thumbnails") or {}
            out.append({
                "twitch_id": f"yt:{v['id']}", "source": "youtube", "url": f"https://www.youtube.com/watch?v={v['id']}",
                "title": _clean(sn.get("title"), 140), "views": int(st.get("viewCount") or 0),
                "date": (sn.get("publishedAt") or "")[:10], "duration": _iso_seconds((v.get("contentDetails") or {}).get("duration")),
                "clipped_by": "", "streamer": _clean(sn.get("channelTitle"), 60), "game": "",
                "thumbnail": ((thumbs.get("medium") or thumbs.get("default") or {}).get("url")),
                "description": _clean(sn.get("description"), 200),
            })
    return out


def youtube_search(queries: List[str], token: str, per_query: int = 25) -> List[dict]:
    """Short videos (<= YT_MAX_SECONDS) for each search, best match first.
    100 quota units per search, 1 per 50 videos looked up."""
    ids: List[str] = []
    for q in queries:
        try:
            data = _yt_get("search", {"part": "id", "q": q, "type": "video", "videoDuration": "short", "maxResults": per_query,
                                      "relevanceLanguage": "en", "safeSearch": "moderate"}, token)
        except Exception as e:
            print(f"[clip_search] YouTube search {q!r} failed: {e}", flush=True)
            continue
        for it in data.get("items") or []:
            vid = (it.get("id") or {}).get("videoId")
            if vid and vid not in ids:
                ids.append(vid)
    vids = {v["twitch_id"]: v for v in youtube_videos(ids, token)} if ids else {}
    return [vids[f"yt:{i}"] for i in ids if f"yt:{i}" in vids and 0 < vids[f"yt:{i}"]["duration"] <= YT_MAX_SECONDS]


def resolve_links(links: List[str], yt_token: str = "") -> List[dict]:
    """Twitch clip and YouTube links Dean pasted, as clip entries."""
    yt_ids = []
    for link in links:
        m = _YT_LINK.search(str(link))
        if m and m.group(1) not in yt_ids:
            yt_ids.append(m.group(1))
    # whole long videos would take ages to download and transcribe
    found = [v for v in (youtube_videos(yt_ids, yt_token) if yt_ids and yt_token else []) if v["duration"] <= 900]
    return _twitch_links(links) + found


def _twitch_links(links: List[str]) -> List[dict]:
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


def find(topic: str, on_progress: Callable[[str], None] = lambda m: None, yt_token: str = "") -> dict:
    """Search terms, the Twitch games' clips and YouTube's short videos,
    the shortlist and Claude's picks (not downloaded yet).
    {"games", "keywords", "scanned", "picked"}."""
    on_progress("Choosing what to search for...")
    terms = search_terms(topic)
    games = game_ids(terms["games"])
    candidates: List[dict] = []
    scanned = 0
    for gid, name in games.items():
        on_progress(f"Reading the top {name} clips on Twitch...")
        entries = [_entry(c, games) for c in game_clips(gid)]
        scanned += len(entries)
        candidates += shortlist(entries, terms["keywords"], limit=30)
    if yt_token and terms.get("youtube"):
        on_progress("Searching YouTube...")
        yt = youtube_search(terms["youtube"], yt_token)
        scanned += len(yt)
        candidates += yt[:40]
    if not candidates:
        return {**terms, "games": list(games.values()), "scanned": scanned, "picked": []}
    on_progress(f"Picking from {len(candidates)} clips...")
    picked = pick(topic, candidates)
    return {**terms, "games": list(games.values()), "scanned": scanned, "picked": picked}
