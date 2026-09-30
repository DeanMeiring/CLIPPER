"""Streamer documentaries ("The Story of <streamer>") for the Caught On Stream
channel: a bi-weekly long-form series, a different streamer each episode.

The shape is the one that works for creator documentaries on YouTube: a cold
open on the most gripping moment, a promise of what the video answers, then
the story in chapters (origins, breakthrough, peak, turning point, today),
with the streamer's real clips playing between the narration. The creator's
own voice and take carry it; the clips are the evidence.

This module does the research and the story:

  build_dossier    real facts: Twitch profile, the streamer's most-viewed
                   clips across their whole career (year by year), their
                   Wikipedia article if there is one, any article links the
                   creator pastes, and the creator's own notes
  build_library    downloads those clips and transcribes them (word timings,
                   for subtitles and for cutting a moment on a full sentence)
  write_script     Claude drafts the episode from the dossier: narrated
                   scenes over clip footage, clip "moments" that play with
                   their own sound, and chapter cards

Recording (longform.py) and rendering (longform_video.py) are shared with
the rest of the long-form code.
"""
from __future__ import annotations

import datetime
import html
import json
import os
import re
from pathlib import Path
from typing import Callable, List, Optional

SCENE_KINDS = ("narrate", "moment", "title")
MAX_LIBRARY_CLIPS = 24
MOMENT_MIN, MOMENT_MAX = 3.0, 25.0
_UA = "clipper-documentaries/1.0 (personal YouTube documentary tool)"


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- twitch ---

def _helix(path: str, params: dict) -> dict:
    import requests

    from .trending import _get_twitch_token

    client_id = os.environ.get("TWITCH_CLIENT_ID")
    token = _get_twitch_token()
    if not client_id or not token:
        raise RuntimeError("Twitch isn't connected (TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET are not set).")
    resp = requests.get(f"https://api.twitch.tv/helix/{path}", params=params,
                        headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def twitch_profile(login: str) -> dict:
    login = login.strip().lower().lstrip("@")
    users = _helix("users", {"login": login}).get("data") or []
    if not users:
        raise RuntimeError(f"No Twitch account called {login!r} -- check the spelling of their Twitch login.")
    u = users[0]
    profile = {
        "id": u["id"], "login": u["login"], "display_name": u.get("display_name") or u["login"],
        "description": u.get("description") or "", "created_at": u.get("created_at"),
        "broadcaster_type": u.get("broadcaster_type") or "", "profile_image_url": u.get("profile_image_url"),
    }
    try:
        ch = (_helix("channels", {"broadcaster_id": u["id"]}).get("data") or [{}])[0]
        profile.update(current_game=ch.get("game_name"), current_title=ch.get("title"), tags=ch.get("tags") or [])
    except Exception:
        pass
    return profile


def _clips(broadcaster_id: str, first: int, start: Optional[str] = None, end: Optional[str] = None) -> List[dict]:
    params = {"broadcaster_id": broadcaster_id, "first": first}
    if start and end:
        params.update(started_at=start, ended_at=end)
    return _helix("clips", params).get("data") or []


def career_clips(profile: dict, all_time: int = 14, per_year: int = 4, max_years: int = 8) -> List[dict]:
    """The streamer's most-viewed clips of all time, plus the top few from
    each year they've been active -- so the story has footage from every
    era, not only their biggest recent moments. Oldest first."""
    found: dict = {}
    for c in _clips(profile["id"], all_time):
        found[c["id"]] = c
    now = datetime.datetime.now(datetime.timezone.utc)
    first_year = now.year - max_years + 1
    if profile.get("created_at"):
        try:
            first_year = max(first_year, int(profile["created_at"][:4]))
        except ValueError:
            pass
    for year in range(first_year, now.year + 1):
        start = f"{year}-01-01T00:00:00Z"
        end = f"{year}-12-31T23:59:59Z"
        try:
            for c in _clips(profile["id"], per_year, start, end):
                found.setdefault(c["id"], c)
        except Exception as e:
            print(f"[documentary] clips for {year} failed: {e}", flush=True)
    clips = sorted(found.values(), key=lambda c: c.get("view_count") or 0, reverse=True)[:MAX_LIBRARY_CLIPS]
    game_ids = sorted({c.get("game_id") for c in clips if c.get("game_id")})
    games = {}
    if game_ids:
        try:
            games = {g["id"]: g["name"] for g in _helix("games", {"id": game_ids}).get("data") or []}
        except Exception:
            pass
    out = []
    for c in sorted(clips, key=lambda c: c.get("created_at") or ""):
        out.append({
            "twitch_id": c["id"], "url": c.get("url"), "title": " ".join(str(c.get("title") or "").split()),
            "views": int(c.get("view_count") or 0), "date": (c.get("created_at") or "")[:10],
            "duration": float(c.get("duration") or 0), "clipped_by": c.get("creator_name") or "",
            "game": games.get(c.get("game_id"), ""), "thumbnail": c.get("thumbnail_url"),
        })
    for i, c in enumerate(out, start=1):
        c["id"] = f"C{i:02d}"
    return out


# ------------------------------------------------------------- wikipedia ---

def wikipedia(names: List[str]) -> Optional[dict]:
    """The streamer's English Wikipedia article (plain text), if they have
    one -- the most reliable source for dates and events."""
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": _UA}
    for name in [n for n in names if n]:
        try:
            r = requests.get(api, params={"action": "query", "list": "search", "srsearch": f"{name} streamer",
                                          "format": "json", "srlimit": 5}, headers=headers, timeout=15)
            r.raise_for_status()
            hits = r.json().get("query", {}).get("search") or []
        except Exception as e:
            print(f"[documentary] wikipedia search failed: {e}", flush=True)
            return None
        key = name.lower().replace(" ", "")
        match = next((h for h in hits if key in h["title"].lower().replace(" ", "")), None)
        if not match:
            continue
        try:
            r = requests.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                          "titles": match["title"], "format": "json"}, headers=headers, timeout=20)
            r.raise_for_status()
            pages = (r.json().get("query") or {}).get("pages") or {}
            page = next(iter(pages.values()), {})
        except Exception as e:
            print(f"[documentary] wikipedia extract failed: {e}", flush=True)
            return None
        text = (page.get("extract") or "").strip()
        if len(text) < 300:
            continue
        title = page.get("title") or match["title"]
        return {"title": title, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), "text": text[:30000]}
    return None


# -------------------------------------------------------------- articles ---

def fetch_article(url: str) -> dict:
    """Readable text of a web page the creator pasted (news article,
    interview, fan wiki), for the research pack."""
    import requests

    if not re.match(r"^https?://", url):
        raise ValueError("not a web link")
    m = re.match(r"^https?://en\.(?:m\.)?wikipedia\.org/wiki/([^#?]+)", url)
    if m:
        from urllib.parse import unquote

        wiki = wikipedia([unquote(m.group(1)).replace("_", " ")])
        if wiki:
            return {"url": url, "title": wiki["title"], "text": wiki["text"][:12000]}
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 " + _UA}, timeout=20)
    r.raise_for_status()
    page = r.text
    title_m = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    page = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</li>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if len(ln) > 40)
    return {"url": url, "title": html.unescape(" ".join((title_m.group(1) if title_m else url).split()))[:150],
            "text": text[:12000]}


def build_dossier(login: str, notes: str, links: List[str], on_progress: Callable[[str], None] = lambda m: None) -> dict:
    on_progress("Looking up their Twitch channel...")
    profile = twitch_profile(login)
    on_progress("Finding their most-viewed clips across their career...")
    clips = career_clips(profile)
    on_progress("Checking Wikipedia...")
    wiki = wikipedia([profile["display_name"], profile["login"]])
    articles, failed = [], []
    for url in links[:5]:
        on_progress(f"Reading {url[:60]}...")
        try:
            a = fetch_article(url)
            if len(a["text"]) > 200:
                articles.append(a)
            else:
                failed.append(url)
        except Exception as e:
            print(f"[documentary] article {url} failed: {e}", flush=True)
            failed.append(url)
    return {"profile": profile, "clips": clips, "wikipedia": wiki, "articles": articles,
            "failed_links": failed, "notes": notes.strip()[:6000]}


# --------------------------------------------------------------- library ---

def build_library(clips: List[dict], clips_dir: Path, on_progress: Callable[[int, int], None] = lambda i, n: None) -> List[dict]:
    """Download every clip and transcribe it. A clip that fails to download
    is dropped (Twitch sometimes deletes old clips) rather than failing the
    whole research."""
    from .download import download_video
    from .longform import audio_duration, transcribe_words

    out = []
    for i, c in enumerate(clips):
        on_progress(i, len(clips))
        d = clips_dir / c["id"]
        words_file = d / "words.json"
        try:
            existing = sorted(d.glob("*.mp4")) if d.exists() else []
            if existing and words_file.exists():
                path = existing[0]
            else:
                dl = download_video(c["url"], d)
                path = Path(dl.video_path)
            dur = audio_duration(path)
            if not words_file.exists():
                words_file.write_text(json.dumps(transcribe_words(path)), encoding="utf-8")
            out.append({**c, "file": str(path.relative_to(clips_dir.parent)), "duration": round(dur, 2)})
        except Exception as e:
            print(f"[documentary] clip {c.get('url')} failed: {e}", flush=True)
    on_progress(len(clips), len(clips))
    return out


def clip_words(project_dir: Path, clip_id: str) -> List[dict]:
    try:
        return json.loads((project_dir / "clips" / clip_id / "words.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _transcript_marked(words: List[dict], limit: int = 700) -> str:
    """Transcript with [seconds] markers every few words, so Claude can say
    exactly where a moment should start and end."""
    out, last_mark, n = [], -99.0, 0
    for w in words:
        if w["s"] - last_mark >= 4:
            out.append(f"[{w['s']:.0f}s]")
            last_mark = w["s"]
        out.append(w["w"])
        n += len(w["w"]) + 1
        if n > limit:
            out.append("...")
            break
    return " ".join(out)


# ---------------------------------------------------------------- script ---

def _script_prompt(dossier: dict, library: List[dict], project_dir: Path, channel: str) -> str:
    p = dossier["profile"]
    facts = [
        f"Twitch login: {p['login']} (display name {p['display_name']})",
        f"Twitch account created: {(p.get('created_at') or '')[:10] or 'unknown'}",
        f"Twitch status: {p.get('broadcaster_type') or 'regular'}",
        f"Channel bio: {p.get('description') or '-'}",
        f"Currently streaming: {p.get('current_game') or '-'} (title: {p.get('current_title') or '-'})",
    ]
    clip_lines = []
    for c in library:
        clip_lines.append(
            f"{c['id']} | {c['date']} | {c['views']:,} views | {c['duration']:.0f}s | {c.get('game') or '-'} | "
            f"\"{c['title']}\"\n    transcript: {_transcript_marked(clip_words(project_dir, c['id']))}"
        )
    sources = []
    if dossier.get("wikipedia"):
        sources.append(f"WIKIPEDIA ARTICLE \"{dossier['wikipedia']['title']}\":\n{dossier['wikipedia']['text'][:20000]}")
    for a in dossier.get("articles") or []:
        sources.append(f"ARTICLE \"{a['title']}\" ({a['url']}):\n{a['text'][:8000]}")
    if dossier.get("notes"):
        sources.append(f"THE CREATOR'S OWN NOTES (things they know as a long-time viewer):\n{dossier['notes']}")
    name = p["display_name"]
    return f"""You are writing an episode of "The Story Of", a bi-weekly documentary series
on the YouTube channel "{channel}", which posts streamer clips. This episode
is about the Twitch streamer {name}. The channel's creator narrates it in
their own voice; between the narration, {name}'s real clips play with their
own sound.

RESEARCH (use ONLY facts found here):
{chr(10).join(facts)}

{chr(10).join(sources) if sources else "(No articles found -- rely on the Twitch facts and the clips.)"}

CLIPS YOU CAN USE (id | date | views | length | game | title, then what is said with [seconds] markers):
{chr(10).join(clip_lines) if clip_lines else "(none)"}

Write the episode as a list of scenes. Three kinds:
- "narrate": the creator speaks. Give "narration" (30 to 120 words) and a
  "clip" to show underneath (footage from the same period or topic; clips
  can be reused as background). Its sound plays quietly under the voice.
- "moment": a clip plays at full volume with no narration. Give "clip",
  "start" and "end" in seconds from that clip's transcript markers, cut on
  full sentences, 4 to 20 seconds long. This is the payoff -- the funniest,
  most surprising or most famous bit, not filler.
- "title": a chapter card. Give "title" (under 32 characters).

Every scene may have a short "caption" for the corner of the screen (under
40 characters): a date, a place or a stat from the research, e.g.
"March 2023 · 1.2M views". Use "" when nothing fits.

Structure (this is what keeps viewers watching):
1. Cold open: a "moment" with the most gripping clip, then a short
   "narrate" scene that promises what the video will answer (the question
   at the heart of {name}'s story). Then a "title" scene: "The Story of {name}".
   That card also opens the first chapter, so follow it with narration,
   never with another "title".
2. Four to six chapters in time order -- for example the start, the
   breakthrough, the peak, a turning point, where they are now. Every
   chapter after the first starts with a "title" scene. Fit them to what
   actually happened.
3. Keep momentum: never more than two "narrate" scenes in a row without a
   "moment". Aim for 8 to 12 moments in total.
4. End with a "narrate" scene that ties the story together, asks viewers a
   question to answer in the comments, and says more of {name}'s best
   moments are on this channel.

Total narration: 1,100 to 1,500 words. Rules:
- Facts ONLY from the research above. Dates come from the Twitch data, the
  clip dates or the articles. If you're not sure of something, leave it
  out. Never invent quotes -- quote {name} only from a clip transcript.
- No speculation about private life, health or relationships, and no
  accusations. If a source reports a controversy, you may mention it only
  as the source reports it ("according to ...").
- Voice: a fan who knows this world, calm and clear, with a bit of wit.
  Spoken sentences that are easy to read aloud. Numbers as digits.
- Plain words only. Don't use hype words like "insane", "crazy", "chaos",
  "legendary" or "iconic".

Respond with ONLY a JSON array of scenes in order, no other text:
[
  {{"kind": "moment", "clip": "C05", "start": 3, "end": 14, "caption": "May 2023"}},
  {{"kind": "narrate", "narration": "...", "clip": "C05", "caption": ""}},
  {{"kind": "title", "title": "The Story of {name}"}}
]
"""


def normalize_scenes(items, library: List[dict]) -> List[dict]:
    by_id = {c["id"]: c for c in library}
    scenes = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind") or "").lower()
        caption = " ".join(str(it.get("caption") or "").split())[:48]
        clip = str(it.get("clip") or "").strip().upper() or None
        if clip not in by_id:
            clip = None
        if kind == "narrate":
            text = " ".join(str(it.get("narration") or "").split())
            if text:
                scenes.append({"kind": "narrate", "narration": text, "clip": clip, "start": 0.0, "caption": caption, "take": None})
        elif kind == "moment" and clip:
            dur = by_id[clip]["duration"]
            start = max(0.0, min(_num(it.get("start")) or 0.0, max(0.0, dur - MOMENT_MIN)))
            end = _num(it.get("end"))
            end = min(dur, end if end is not None and end > start else start + 12.0)
            if end - start < MOMENT_MIN:
                end = min(dur, start + MOMENT_MIN)
            end = min(end, start + MOMENT_MAX)
            if end - start >= 1.0:
                scenes.append({"kind": "moment", "clip": clip, "start": round(start, 2), "end": round(end, 2), "caption": caption})
        elif kind == "title":
            title = " ".join(str(it.get("title") or "").split())[:40]
            if title:
                scenes.append({"kind": "title", "title": title})
    return scenes


def write_script(dossier: dict, library: List[dict], project_dir: Path, channel: str) -> List[dict]:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    data = _ask_claude_for_json(_script_prompt(dossier, library, project_dir, channel), None, DEFAULT_MODEL)
    scenes = normalize_scenes(data, library)
    if not any(s["kind"] == "narrate" for s in scenes):
        raise RuntimeError("Claude didn't return a usable script -- try Write the story again.")
    return scenes


# ------------------------------------------------------------ the series ---

SERIES_EVERY_DAYS = 14


def series_slots(start: str, count: int = 6, today: Optional[datetime.date] = None) -> List[str]:
    """Upcoming release dates, every two weeks from the series' first date."""
    today = today or datetime.date.today()
    try:
        d = datetime.date.fromisoformat(start)
    except (TypeError, ValueError):
        d = today
    while d < today:
        d += datetime.timedelta(days=SERIES_EVERY_DAYS)
    return [(d + datetime.timedelta(days=SERIES_EVERY_DAYS * i)).isoformat() for i in range(count)]
