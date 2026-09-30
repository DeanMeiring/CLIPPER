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
    site_m = re.search(r'<meta[^>]+property=["\']og:site_name["\'][^>]+content=["\']([^"\']+)', page, re.I)
    date_m = re.search(r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|date|pubdate|publish-date)["\'][^>]+content=["\'](\d{4}-\d{2}-\d{2})', page, re.I)
    page = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header|aside)[^>]*>.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</li>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if len(ln) > 40)
    return {"url": url, "title": html.unescape(" ".join((title_m.group(1) if title_m else url).split()))[:150],
            "site": html.unescape(site_m.group(1)).strip()[:40] if site_m else "",
            "published": date_m.group(1) if date_m else "", "text": text[:12000]}


def build_dossier(login: str, notes: str, links: List[str], on_progress: Callable[[str], None] = lambda m: None) -> dict:
    on_progress("Looking up their Twitch channel...")
    profile = twitch_profile(login)
    on_progress("Finding their most-viewed clips across their career...")
    clips = career_clips(profile)
    on_progress("Checking Wikipedia...")
    wiki = wikipedia([profile["display_name"], profile["login"]])
    from .visual_sources import is_x_post, x_post

    articles, posts, failed = [], [], []
    for url in links[:8]:
        on_progress(f"Reading {url[:60]}...")
        if is_x_post(url):
            try:
                post = x_post(url)
                (posts if post else failed).append(post or url)
            except Exception as e:
                print(f"[documentary] post {url} failed: {e}", flush=True)
                failed.append(url)
            continue
        if len(articles) >= 5:
            continue
        try:
            a = fetch_article(url)
            if len(a["text"]) > 200:
                articles.append(a)
            else:
                failed.append(url)
        except Exception as e:
            print(f"[documentary] article {url} failed: {e}", flush=True)
            failed.append(url)
    return {"profile": profile, "clips": clips, "wikipedia": wiki, "articles": articles, "posts": posts,
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
    exactly where a moment should start and end -- and "(silence Ns)" where
    nobody talks for a while, so a moment doesn't start on a silent setup
    or trail off into dead air (YouTube's review of the first episode)."""
    out, last_mark, n, prev_end = [], -99.0, 0, None
    for w in words:
        gap = w["s"] - (prev_end if prev_end is not None else 0.0)
        if gap >= 1.5:
            out.append(f"(silence {gap:.0f}s)")
            last_mark = -99.0
        prev_end = w["e"]
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

def _script_prompt(dossier: dict, library: List[dict], project_dir: Path, channel: str,
                   lessons: Optional[List[str]] = None) -> str:
    from .longform_lessons import prompt_block

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
    for post in dossier.get("posts") or []:
        sources.append(f"POST ON X by @{post['handle']} ({post.get('date') or 'undated'}): \"{post['text']}\"")
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
  most surprising or most famous bit, not filler. "(silence Ns)" in a
  transcript marks a stretch where nobody talks.
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

{prompt_block(lessons or [])}Respond with ONLY a JSON array of scenes in order, no other text:
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
                cues = normalize_cues(it.get("cues") or [], text, library) if it.get("cues") else []
                scenes.append({"kind": "narrate", "narration": text, "clip": clip, "start": 0.0, "caption": caption,
                               "take": None, "cues": cues})
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


def write_script(dossier: dict, library: List[dict], project_dir: Path, channel: str,
                 lessons: Optional[List[str]] = None) -> List[dict]:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    data = _ask_claude_for_json(_script_prompt(dossier, library, project_dir, channel, lessons), None, DEFAULT_MODEL)
    scenes = tighten_moments(normalize_scenes(data, library), project_dir)
    if not any(s["kind"] == "narrate" for s in scenes):
        raise RuntimeError("Claude didn't return a usable script -- try Write the story again.")
    return scenes


COLD_OPEN_MAX = 10.0


def tighten_moments(scenes: List[dict], project_dir: Path) -> List[dict]:
    """Trim dead air off the moments Claude picked (its times come from
    markers a few seconds apart): a moment starts at most 0.8 s before the
    first word in it and ends at most 1.5 s after the last one, and the
    cold open (a moment as the first scene) stops by about 10 s. Only
    trims -- a moment is never made longer, never shorter than
    MOMENT_MIN, and one with no words in it is left alone (a visual gag).
    Applied when the story is written, not to Dean's own edits."""
    out = []
    for i, sc in enumerate(scenes):
        if sc.get("kind") != "moment":
            out.append(sc)
            continue
        start, end = float(sc["start"]), float(sc["end"])
        ws = [w for w in clip_words(project_dir, sc["clip"]) if w["e"] > start + 0.05 and w["s"] < end - 0.05]
        if ws:
            new_start = max(start, ws[0]["s"] - 0.8)
            new_end = min(end, ws[-1]["e"] + 1.5)
            if i == 0 and new_end - new_start > COLD_OPEN_MAX:
                ends = [w["e"] for w in ws if w["e"] <= new_start + COLD_OPEN_MAX and re.search(r"[.!?]$", w["w"])]
                if ends:
                    new_end = min(new_end, max(ends) + 0.8)
            if new_end - new_start >= MOMENT_MIN:
                sc = {**sc, "start": round(new_start, 2), "end": round(new_end, 2)}
        out.append(sc)
    return out


# --------------------------------------------------------------- visuals ---
# Keyword visuals for narrated scenes (rendered by longform_beats.py): what
# appears on screen when the narrator says a given phrase. Claude plans
# them; everything is checked against the narration and the research here,
# so a stat card can only show a number that is written in the research.

MAX_CUES_PER_SCENE = 8


def _tokens(text: str) -> List[str]:
    from .longform import _norm

    return [t for t in (_norm(w) for w in str(text).split()) if t]


def _contains(hay: List[str], needle: List[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def research_text(dossier: dict, library: List[dict]) -> str:
    """Every fact the visuals may quote, as one lower-case blob."""
    p = dossier.get("profile") or {}
    parts = [p.get("display_name") or "", p.get("description") or "", (p.get("created_at") or "")[:10],
             p.get("broadcaster_type") or "", dossier.get("notes") or ""]
    parts += [(dossier.get("wikipedia") or {}).get("text") or ""]
    parts += [f"{a.get('title', '')} {a.get('published', '')} {a.get('text', '')}" for a in dossier.get("articles") or []]
    parts += [f"{x.get('text', '')} {x.get('date', '')} {x.get('likes', '')}" for x in dossier.get("posts") or []]
    parts += [f"{c.get('date', '')} {c.get('views', '')} {int(c.get('views') or 0):,} {c.get('title', '')}" for c in library]
    return " ".join(parts).lower()


def _numbers(text: str) -> set:
    return {m.replace(",", "").rstrip(".") for m in re.findall(r"\d[\d,.]*", text)}


def _value_ok(value: str, blob: str, nums: set) -> bool:
    v = value.strip().lower()
    if not v:
        return False
    if v in blob:
        return True
    digits = re.findall(r"\d[\d,.]*", v)
    return bool(digits) and all(d.replace(",", "").rstrip(".") in nums for d in digits)


def normalize_cues(cues, narration: str, library: List[dict], dossier: Optional[dict] = None) -> List[dict]:
    """Keep only cues that can be drawn and are true: the phrase is in the
    narration, clips/posts/articles exist, numbers and years come from the
    research. Without a dossier (a story edit being saved) the research
    checks are skipped -- the cues were checked when they were planned."""
    from .longform_beats import CUE_TYPES
    from .visual_sources import emoji_file

    narr = _tokens(narration)
    by_id = {c["id"]: c for c in library}
    blob = research_text(dossier, library) if dossier is not None else None
    nums = _numbers(blob) if blob is not None else set()
    posts = (dossier or {}).get("posts")
    articles = (dossier or {}).get("articles")
    out: List[dict] = []
    for c in cues if isinstance(cues, list) else []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("type") or "").lower()
        at = " ".join(str(c.get("at") or "").split())[:60]
        if kind not in CUE_TYPES or not _contains(narr, _tokens(at)):
            continue
        cue = {"at": at, "type": kind}
        if kind == "words":
            cue["text"] = " ".join(str(c.get("text") or at).split())[:32]
        elif kind == "emoji":
            e = str(c.get("emoji") or "").strip()
            if not emoji_file(e):
                continue
            cue["emoji"] = e
            cue["text"] = " ".join(str(c.get("text") or "").split())[:24]
        elif kind == "stat":
            value = " ".join(str(c.get("value") or "").split())[:24]
            if not value or (blob is not None and not _value_ok(value, blob, nums)):
                continue
            cue.update(value=value, label=" ".join(str(c.get("label") or "").split())[:30],
                       sub=" ".join(str(c.get("sub") or "").split())[:48])
        elif kind in ("stock", "photo"):
            q = " ".join(str(c.get("query") or "").split())[:60]
            if not q:
                continue
            cue["query"] = q
            for k in ("asset", "credit"):
                if c.get(k):
                    cue[k] = c[k]
        elif kind == "clip":
            cid = str(c.get("clip") or "").upper()
            if cid not in by_id:
                continue
            start = _num(c.get("start")) or 0.0
            cue.update(clip=cid, start=round(max(0.0, min(start, float(by_id[cid].get("duration") or 0) - 2)), 2))
        elif kind == "post":
            try:
                i = int(c.get("post"))
            except (TypeError, ValueError):
                continue
            if posts is not None and not 0 <= i < len(posts):
                continue
            hl = " ".join(str(c.get("highlight") or "").split())[:120]
            if posts is not None and hl and not _contains(_tokens(posts[i]["text"]), _tokens(hl)):
                hl = ""
            cue.update(post=i, highlight=hl)
        elif kind == "headline":
            try:
                i = int(c.get("article"))
            except (TypeError, ValueError):
                continue
            if articles is not None and not 0 <= i < len(articles):
                continue
            hl = " ".join(str(c.get("highlight") or "").split())[:100]
            if articles is not None and hl and not _contains(_tokens(articles[i]["title"]), _tokens(hl)):
                hl = ""
            cue.update(article=i, highlight=hl)
        elif kind == "timeline":
            pts = []
            for pt in c.get("points") or []:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                year = str(pt[0]).strip()[:6]
                if blob is not None and not (re.fullmatch(r"\d{4}", year) and year in blob):
                    continue
                pts.append([year, " ".join(str(pt[1]).split())[:22]])
            if len(pts) < 2:
                continue
            cue["points"] = pts[:5]
        out.append(cue)
    return out[:MAX_CUES_PER_SCENE]


def _visuals_prompt(dossier: dict, library: List[dict], scenes: List[dict], emoji: str,
                    lessons: Optional[List[str]] = None) -> str:
    from .longform_lessons import prompt_block

    p = dossier["profile"]
    name = p.get("display_name") or p.get("login")
    facts = [f"Twitch account created: {(p.get('created_at') or '')[:10] or 'unknown'}",
             f"Twitch status: {p.get('broadcaster_type') or 'regular'}"]
    if dossier.get("wikipedia"):
        facts.append(f"WIKIPEDIA:\n{dossier['wikipedia']['text'][:12000]}")
    for a in dossier.get("articles") or []:
        facts.append(f"ARTICLE \"{a['title']}\":\n{a['text'][:4000]}")
    if dossier.get("notes"):
        facts.append(f"CREATOR'S NOTES:\n{dossier['notes'][:3000]}")
    posts = "\n".join(f"{i}: @{x['handle']} ({x.get('date') or 'undated'}): \"{x['text']}\""
                      for i, x in enumerate(dossier.get("posts") or [])) or "(none)"
    arts = "\n".join(f"{i}: \"{a['title']}\" ({a.get('site') or a['url']}, {a.get('published') or 'undated'})"
                     for i, a in enumerate(dossier.get("articles") or [])) or "(none)"
    clips = "\n".join(f"{c['id']} | {c['date']} | {c['views']:,} views | {c['duration']:.0f}s | \"{c['title']}\"" for c in library)
    narrated = "\n".join(f"[{i}] {sc['narration']}" for i, sc in enumerate(scenes) if sc.get("kind") == "narrate")
    return f"""You are planning the on-screen visuals for a YouTube documentary about
the Twitch streamer {name}. The narrator's own voice tells the story; the
picture changes on key phrases, every few seconds, the way well-edited
documentary channels do it.

For each narrated scene below, pick key phrases and what appears the
moment each phrase is spoken. Visual types:
- "words": the phrase or a short line (max 4 words) in big letters. For a
  short, striking statement.
- "emoji": one emoji from this list: {emoji}
  For a feeling or an object (money, a trophy, a late night). Optional
  "text": a 1-3 word label.
- "stat": a number card with {name}'s picture: "label" (e.g. "Clip views"),
  "value" written exactly as in the research, optional "sub". Only numbers
  that appear in the research, clip list, posts or articles below.
- "stock": free stock video: "query" of 2-4 plain words for a generic
  scene ("gaming setup at night", "crowd cheering"). Never a person's name,
  a brand, a game title or a logo -- stock libraries don't have those.
- "photo": a free photo, same "query" rules. Good for places and objects.
- "clip": cut to one of {name}'s clips: "clip" id and "start" second. When
  the narration mentions an event that clip shows.
- "post": show one of the posts below: "post" number and "highlight"
  (exact words from the post).
- "headline": show one of the article headlines below: "article" number
  and "highlight" (exact words from its title).
- "timeline": "points", 2 to 5 of ["year", "event"] from the research. For
  a stretch of time.

Rules:
- "at" is copied exactly from the scene's narration: 1 to 5 consecutive
  words. The visual appears when they are said.
- About one visual per 12 to 18 words. At least 6 words between two
  "at" phrases. Give cards (stat, post, headline, timeline) at least 10
  words of narration before the next visual, so there is time to read.
- Mix the types; never the same type twice in a row.
- Facts only from the research. If a number isn't there, don't make a stat.
- Plain labels; no hype words.

RESEARCH:
{chr(10).join(facts)}

POSTS:
{posts}

ARTICLES:
{arts}

CLIPS (id | date | views | length | title):
{clips}

NARRATED SCENES ([scene number] narration):
{narrated}

{prompt_block(lessons or [])}Respond with ONLY a JSON array, one object per narrated scene:
[{{"scene": 3, "cues": [{{"at": "twelve viewers", "type": "words", "text": "12 viewers"}}, {{"at": "late at night", "type": "stock", "query": "bedroom desk at night"}}]}}]
"""


def plan_visuals(dossier: dict, library: List[dict], scenes: List[dict], lessons: Optional[List[str]] = None) -> List[dict]:
    """Keyword visuals for every narrated scene. Returns the scenes with
    "cues" set (other scenes unchanged)."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json
    from .visual_sources import emoji_list

    if not any(sc.get("kind") == "narrate" for sc in scenes):
        return scenes
    data = _ask_claude_for_json(_visuals_prompt(dossier, library, scenes, emoji_list(), lessons), None, DEFAULT_MODEL)
    planned = {}
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            try:
                planned[int(item.get("scene"))] = item.get("cues")
            except (TypeError, ValueError):
                continue
    out = []
    for i, sc in enumerate(scenes):
        sc = dict(sc)
        if sc.get("kind") == "narrate":
            sc["cues"] = normalize_cues(planned.get(i) or [], sc["narration"], library, dossier)
        out.append(sc)
    return out


def fetch_visuals(scenes: List[dict], dossier: dict, project_dir: Path,
                  on_progress: Callable[[str], None] = lambda m: None) -> List[dict]:
    """Download what the cues need into visuals/: stock footage and photos
    (a stock video that can't be found falls back to a photo; one with no
    photo either is dropped), the streamer's avatar and post avatars."""
    from .visual_sources import fetch_image, find_stock

    vdir = project_dir / "visuals"
    vdir.mkdir(parents=True, exist_ok=True)
    fetch_image((dossier.get("profile") or {}).get("profile_image_url") or "", vdir / "avatar.jpg")
    for i, post in enumerate(dossier.get("posts") or []):
        fetch_image(post.get("avatar") or "", vdir / f"post_{i}_avatar.jpg")
    out = []
    for sc in scenes:
        sc = dict(sc)
        cues = []
        for c in sc.get("cues") or []:
            c = dict(c)
            if c["type"] in ("stock", "photo") and not (c.get("asset") and (vdir / c["asset"]).is_file()):
                on_progress(f"Finding free {'footage' if c['type'] == 'stock' else 'photos'}: {c['query']}...")
                found = find_stock("video" if c["type"] == "stock" else "photo", c["query"], vdir)
                if not found and c["type"] == "stock":
                    found = find_stock("photo", c["query"], vdir)
                if not found:
                    continue
                c["asset"] = found["file"]
                c["credit"] = {k: found.get(k, "") for k in ("source", "author", "page")}
            cues.append(c)
        if sc.get("kind") == "narrate":
            sc["cues"] = cues
        out.append(sc)
    return out


def visual_credits(scenes: List[dict]) -> List[str]:
    from .visual_sources import credits

    return credits([c.get("credit") for sc in scenes for c in sc.get("cues") or [] if c.get("credit")])


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
