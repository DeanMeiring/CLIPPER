"""Caught On Code: explainers about the tech behind gaming, streaming and the
internet (netcode, anti-cheat, matchmaking, how a stream reaches you, how AI
voice cloning works...). Dean narrates in his own voice, no face, like "The
Story Of"; the picture is animated diagrams drawn from code
(explainer_visuals.py) instead of clips.

The niche was picked from research (Oct 2026): the big science topics are
owned by large channels (3Blue1Brown on neural networks, Kurzgesagt on
space, Branch Education on hardware, Real/Practical Engineering and The B1M
on engineering), while gaming-tech questions players actually search for
(netcode, SBMM, kernel anti-cheat, stream latency) have scattered one-off
videos and no channel that owns them. It's also the same audience as
Caught On Stream, so the two channels can point at each other.

This module does the research and the script:

  build_dossier     Wikipedia (the topic's top articles), links Dean pastes,
                    and his notes
  write_script      Claude writes the episode: narrated scenes, each with 1-3
                    diagrams from a fixed set of templates, "pause and guess"
                    quiz scenes, and chapter cards
  normalize_scenes  keeps only what's true and drawable: a number on screen
                    must be in the research (or be simple maths on numbers
                    that are); a phrase a diagram waits for must be in the
                    narration
  quiz_suggestions  the quizzes as YouTube Studio in-video quizzes (Studio >
                    Video elements > Add a quiz; there's no API for it)
  write_publish_text, make_thumbnails / redraw  for the Render & post step

Projects live in the long-form store with kind "explainer"; recording,
rendering, music and upload are shared with the documentaries.
"""
from __future__ import annotations

import ast
import operator
import re
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from .explainer_visuals import TEMPLATES

CHANNEL = "Caught On Code"
KIND = "explainer"
END_LINE = "New explainers every month"
PROFILE = "code"  # its own YouTube account (see webapp/main.py)

TOPIC_IDEAS = [
    "Why you died behind the wall: netcode, tick rate and peeker's advantage",
    "How kernel anti-cheat works, and why cheaters still get through",
    "How skill-based matchmaking builds your lobby",
    "How a Twitch stream reaches viewers in 2 seconds",
    "How AI clones a voice",
    "How DLSS and frame generation make frames that were never rendered",
    "How Fortnite fits 100 players on one server",
    "How the YouTube algorithm picks what you watch",
    "How open-world games load without loading screens",
    "How robots learn to walk",
]

_UA = "caught-on-code/1.0 (personal YouTube explainer tool)"


# --------------------------------------------------------------- research ---

def wikipedia_topic(topic: str, max_articles: int = 2) -> List[dict]:
    """The topic's top English Wikipedia articles, as plain text."""
    import requests

    api = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": _UA}
    query = re.sub(r"[:,?]", " ", topic)
    try:
        r = requests.get(api, params={"action": "query", "list": "search", "srsearch": query, "format": "json",
                                      "srlimit": 5}, headers=headers, timeout=15)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search") or []
    except Exception as e:
        print(f"[explainer] wikipedia search failed: {e}", flush=True)
        return []
    out = []
    for h in hits:
        if len(out) >= max_articles:
            break
        try:
            r = requests.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                                          "titles": h["title"], "format": "json"}, headers=headers, timeout=20)
            r.raise_for_status()
            page = next(iter(((r.json().get("query") or {}).get("pages") or {}).values()), {})
        except Exception as e:
            print(f"[explainer] wikipedia extract failed: {e}", flush=True)
            continue
        text = (page.get("extract") or "").strip()
        if len(text) >= 500:
            title = page.get("title") or h["title"]
            out.append({"title": title, "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), "text": text[:20000]})
    return out


def build_dossier(topic: str, notes: str, links: List[str], on_progress: Callable[[str], None] = lambda m: None) -> dict:
    from .documentary import fetch_article

    on_progress("Reading Wikipedia...")
    wiki = wikipedia_topic(topic)
    articles, failed = [], []
    for i, url in enumerate(links):
        on_progress(f"Reading link {i + 1} of {len(links)}...")
        try:
            a = fetch_article(url)
            if len(a.get("text") or "") >= 200:
                articles.append(a)
            else:
                failed.append(url)
        except Exception as e:
            print(f"[explainer] article {url} failed: {e}", flush=True)
            failed.append(url)
    if not wiki and not articles and len((notes or "").strip()) < 200:
        raise RuntimeError("Couldn't find enough to go on -- paste a few article links or write more notes, then run the research again.")
    return {"topic": topic, "notes": notes, "wikipedia": wiki, "articles": articles, "failed_links": failed}


def research_text(dossier: dict) -> str:
    """Every fact a diagram may show, as one lower-case blob."""
    parts = [dossier.get("topic") or "", dossier.get("notes") or ""]
    parts += [w.get("text") or "" for w in dossier.get("wikipedia") or []]
    parts += [f"{a.get('title', '')} {a.get('text', '')}" for a in dossier.get("articles") or []]
    return " ".join(parts).lower()


def sources(dossier: dict) -> List[dict]:
    out = [{"url": w["url"], "title": w["title"]} for w in dossier.get("wikipedia") or []]
    out += [{"url": a["url"], "title": a.get("title") or a["url"]} for a in dossier.get("articles") or []]
    return out


# ----------------------------------------------------------------- script ---

DIAGRAMS = """DIAGRAM TYPES (every field optional unless marked *):
- {"type": "flow", "title", "nodes"*: [{"label"*, "emoji", "sub", "at"}] (2-6)}
    boxes left to right joined by arrows, a packet moving along them. For a
    path something travels (your PC -> ingest -> transcoder -> viewers).
- {"type": "network", "title", "center": {"label", "emoji"}, "clients"*: [{"label"*, "emoji", "at"}] (2-8), "tick_rate"}
    players around a server, packets flying both ways, the server pulsing.
- {"type": "race", "title", "lanes"*: ["Enemy", "Server", "You"] (2-3), "unit": "ms",
   "events"*: [{"lane"* (0-based), "ms"*, "label"*, "at", "highlight"}] (2-8)}
    what happens when, on a timeline: delays between machines.
- {"type": "bars", "title", "unit", "items"*: [{"label"*, "value"* (number), "highlight", "at"}] (2-6)}
- {"type": "bignum", "value"*, "label"*, "sub", "math"}
    one number, big. "math" ("1000 / 128") when the value is worked out from
    numbers in the research.
- {"type": "grid", "count"*, "highlight", "label", "highlight_at"}   up to 400 dots, some lit.
- {"type": "layers", "title", "items"*: [{"label"*, "sub", "at"}] (top to bottom, 2-6), "highlight" (index), "pin", "highlight_at"}
- {"type": "neural", "layers": [4, 6, 6, 3], "outputs": ["..."], "label"}
- {"type": "compare", "left"*: {"title"*, "items"*: ["...", ...]}, "right"*: {"title"*, "items"*: [...]}}
- {"type": "quiz", "question"*, "options"* (2-4 short answers), "answer"* (0-based index), "reveal_at"}
- {"type": "words", "text"*, "highlight"}   one short line in big type. Use rarely.
"at" (on a visual or an element) is a short phrase, copied EXACTLY from that
scene's narration, at which it appears. The first visual of a scene starts
with the scene; a second or third needs "at". "title" is a 1-4 word label
for the top corner. Emoji: one plain emoji like 🎮 🖥️ 💻 📺 🔫 👀 🧑 🔒 ⚡ 🧠 🤖 🛡️ 🏆 🎯 🚀."""


def _script_prompt(dossier: dict, channel: str, minutes: int = 8) -> str:
    srcs = []
    for w in dossier.get("wikipedia") or []:
        srcs.append(f"WIKIPEDIA \"{w['title']}\":\n{w['text'][:15000]}")
    for a in dossier.get("articles") or []:
        srcs.append(f"ARTICLE \"{a['title']}\" ({a['url']}):\n{a['text'][:8000]}")
    if dossier.get("notes"):
        srcs.append(f"THE CREATOR'S OWN NOTES:\n{dossier['notes']}")
    topic = dossier.get("topic") or ""
    words = minutes * 150
    return f"""You are writing an episode of "{channel}", a YouTube channel that explains
the tech behind gaming, streaming and the internet to gamers. The creator
narrates in their own voice; the picture is animated diagrams drawn from
code, one or more per scene.

TOPIC: {topic}

RESEARCH (use ONLY facts found here):
{chr(10).join(srcs) if srcs else "(none)"}

Write the episode as a list of scenes of two kinds:
- "narrate": the creator speaks. "narration" (25 to 110 words) and
  "visuals": 1 to 3 diagrams that show what is being said, at that moment.
- "title": a chapter card, "title" under 32 characters.

{DIAGRAMS}

Structure (this is what keeps viewers watching):
1. Cold open (one "narrate" scene, under 60 words): a moment every gamer
   has lived through ("you shoot first, and you still die"), then the
   question the video answers.
2. A "title" scene with the episode title, then 3 to 5 chapters, each
   starting with a "title" scene. Build up: what's going on, why, the
   trade-offs, what it means for the viewer.
3. Two or three "pause and guess" moments: a "narrate" scene whose
   narration asks the question and lists the options ("Pause and guess:
   ... A, B or C?"), with a "quiz" visual; the next scene gives the answer.
   Put the first one in the first 2 minutes.
4. Change the picture often: every narrate scene gets its own diagrams,
   and the same type should not appear in more than two scenes in a row.
5. End with a "narrate" scene that sums it up in one line, asks viewers a
   question to answer in the comments, and says what the next video is.

Total narration: about {words} words. Rules:
- Facts ONLY from the research. Numbers on screen (bars, bignum, grid,
  tick_rate, race ms) must appear in the research, or be simple maths on
  numbers that do (then give "math"). If timings in a "race" are an
  illustration rather than measured, set "example": true on it.
- Say clearly when something is a simplification.
- Voice: a gamer who gets the tech, calm and clear, a bit of wit, talking
  to "you". Short spoken sentences, easy to read aloud. Numbers as digits.
- Plain words only. Don't use hype words like "insane", "crazy", "chaos",
  "mind-blowing" or "game-changer".

Respond with ONLY a JSON array of scenes in order, no other text:
[
  {{"kind": "narrate", "narration": "...", "visuals": [{{"type": "race", "lanes": ["Enemy", "Server", "You"], "events": [...]}}]}},
  {{"kind": "title", "title": "..."}}
]
"""


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _calc(expr: str) -> Optional[float]:
    """Value of plain arithmetic ("1000 / 128"), or None."""
    try:
        node = ast.parse(expr.replace(",", "").replace("x", "*").replace("×", "*").replace("÷", "/"), mode="eval").body
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError
    try:
        return ev(node)
    except (ValueError, ZeroDivisionError):
        return None


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# Unit conversions a calculation may use without a source (ms in a second,
# seconds in a minute, percent, bytes in a kilobyte...).
_CONSTANTS = {"2", "8", "10", "24", "60", "100", "1000", "1024", "3600"}


def _supported(value, blob: str, nums: set, math: str = "") -> bool:
    """A number may go on screen if it's in the research, or if it's simple
    maths on numbers that are."""
    from .documentary import _value_ok

    if _value_ok(str(value), blob, nums):
        return True
    v = _num(value)
    if math and v is not None:
        res = _calc(math)
        operands = re.findall(r"\d[\d,.]*", math)
        if res is not None and operands and all(o.replace(",", "") in _CONSTANTS or _value_ok(o, blob, nums) for o in operands):
            return abs(res - v) <= max(0.051, abs(res) * 0.01)
    return False


def _clean(s, n: int) -> str:
    return " ".join(str(s or "").split())[:n]


def _phrase(at, narration_tokens: List[str]) -> str:
    from .documentary import _contains, _tokens

    at = _clean(at, 80)
    return at if at and _contains(narration_tokens, _tokens(at)) else ""


def _emoji_ok(e) -> str:
    from .visual_sources import emoji_file

    e = _clean(e, 8)
    return e if e and emoji_file(e) else ""


def normalize_visual(v, narration: str, blob: str, nums: set) -> Optional[dict]:
    """A diagram as it will be drawn, or None if it can't be drawn truthfully."""
    from .documentary import _tokens

    if not isinstance(v, dict) or v.get("type") not in TEMPLATES:
        return None
    toks = _tokens(narration)
    kind = v["type"]
    out = {"type": kind}
    if _clean(v.get("title"), 40):
        out["title"] = _clean(v.get("title"), 40)
    if _phrase(v.get("at"), toks):
        out["at"] = _phrase(v.get("at"), toks)

    def element(it, extra=()):
        if not isinstance(it, dict):
            it = {"label": it}
        e = {"label": _clean(it.get("label"), 40)}
        for k in ("sub",):
            if _clean(it.get(k), 40):
                e[k] = _clean(it.get(k), 40)
        if _emoji_ok(it.get("emoji")):
            e["emoji"] = _emoji_ok(it.get("emoji"))
        if _phrase(it.get("at"), toks):
            e["at"] = _phrase(it.get("at"), toks)
        for k in extra:
            if k in it:
                e[k] = it[k]
        return e if e["label"] else None

    if kind == "flow":
        nodes = [e for e in (element(n) for n in (v.get("nodes") or [])[:6]) if e]
        if len(nodes) < 2:
            return None
        out.update(nodes=nodes, packet=v.get("packet", True) is not False)
    elif kind == "network":
        clients = [e for e in (element(n) for n in (v.get("clients") or [])[:8]) if e]
        if len(clients) < 2:
            return None
        center = element(v.get("center") or {"label": "Server", "emoji": "🖥️"}) or {"label": "Server"}
        out.update(center=center, clients=clients)
        if v.get("tick_rate") is not None and _supported(v["tick_rate"], blob, nums):
            out["tick_rate"] = _clean(v["tick_rate"], 8)
    elif kind == "race":
        lanes = [_clean(x, 16) for x in (v.get("lanes") or [])[:3] if _clean(x, 16)]
        events = []
        for ev in (v.get("events") or [])[:8]:
            if not isinstance(ev, dict) or _num(ev.get("ms")) is None or not _clean(ev.get("label"), 40):
                continue
            e = {"lane": int(_num(ev.get("lane")) or 0) % max(1, len(lanes)), "ms": _num(ev["ms"]),
                 "label": _clean(ev["label"], 40)}
            if ev.get("highlight"):
                e["highlight"] = True
            if _phrase(ev.get("at"), toks):
                e["at"] = _phrase(ev.get("at"), toks)
            events.append(e)
        if len(lanes) < 2 or len(events) < 2:
            return None
        measured = all(_supported(_fmt(e["ms"]), blob, nums) for e in events if e["ms"])
        out.update(lanes=lanes, events=events, unit=_clean(v.get("unit") or "ms", 6),
                   example=bool(v.get("example")) or not measured)
    elif kind == "bars":
        items = []
        for it in (v.get("items") or [])[:6]:
            if not isinstance(it, dict) or _num(it.get("value")) is None or not _clean(it.get("label"), 28):
                continue
            if not _supported(_fmt(_num(it["value"])), blob, nums, str(it.get("math") or "")):
                continue
            e = {"label": _clean(it["label"], 28), "value": _num(it["value"])}
            if it.get("highlight"):
                e["highlight"] = True
            if _phrase(it.get("at"), toks):
                e["at"] = _phrase(it.get("at"), toks)
            items.append(e)
        if len(items) < 2:
            return None
        out.update(items=items, unit=_clean(v.get("unit"), 10))
    elif kind == "bignum":
        value, label = _clean(v.get("value"), 14), _clean(v.get("label"), 80)
        if not value or not label:
            return None
        if not _supported(value, blob, nums, str(v.get("math") or "")):
            return {"type": "words", "text": label, **({"at": out["at"]} if "at" in out else {})}
        out.update(value=value, label=label)
        if _clean(v.get("sub"), 80):
            out["sub"] = _clean(v.get("sub"), 80)
    elif kind == "grid":
        count, hl = _num(v.get("count")), _num(v.get("highlight")) or 0
        if not count or count < 2 or count > 400 or not _supported(_fmt(count), blob, nums):
            return None
        out.update(count=int(count), label=_clean(v.get("label"), 60))
        if 0 < hl <= count and _supported(_fmt(hl), blob, nums):
            out["highlight"] = int(hl)
        if _phrase(v.get("highlight_at"), toks):
            out["highlight_at"] = _phrase(v.get("highlight_at"), toks)
    elif kind == "layers":
        items = [e for e in (element(n) for n in (v.get("items") or [])[:6]) if e]
        if len(items) < 2:
            return None
        out["items"] = items
        h = _num(v.get("highlight"))
        if h is not None and 0 <= h < len(items):
            out["highlight"] = int(h)
            if _clean(v.get("pin"), 24):
                out["pin"] = _clean(v.get("pin"), 24)
            if _phrase(v.get("highlight_at"), toks):
                out["highlight_at"] = _phrase(v.get("highlight_at"), toks)
    elif kind == "neural":
        layers = [max(1, min(8, int(_num(x) or 1))) for x in (v.get("layers") or [4, 6, 6, 3])[:5]]
        out.update(layers=layers if len(layers) >= 2 else [4, 6, 3],
                   outputs=[_clean(o, 16) for o in (v.get("outputs") or [])[:8] if _clean(o, 16)],
                   label=_clean(v.get("label"), 70))
    elif kind == "compare":
        sides = {}
        for side in ("left", "right"):
            part = v.get(side) or {}
            items = [_clean(x.get("text") if isinstance(x, dict) else x, 50) for x in (part.get("items") or [])[:5]]
            items = [x for x in items if x]
            if not _clean(part.get("title"), 24) or not items:
                return None
            sides[side] = {"title": _clean(part["title"], 24), "items": items}
        out.update(sides)
    elif kind == "quiz":
        q = _clean(v.get("question"), 90)
        opts = [_clean(o, 50) for o in (v.get("options") or [])[:4] if _clean(o, 50)]
        ans = _num(v.get("answer"))
        if not q or len(opts) < 2 or ans is None or not 0 <= ans < len(opts):
            return None
        out.update(question=q, options=opts, answer=int(ans))
        if _phrase(v.get("reveal_at"), toks):
            out["reveal_at"] = _phrase(v.get("reveal_at"), toks)
    elif kind == "words":
        text = _clean(v.get("text"), 60)
        if not text:
            return None
        out["text"] = text
        if _clean(v.get("highlight"), 30):
            out["highlight"] = _clean(v.get("highlight"), 30)
    return out


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def normalize_scenes(items, dossier: dict) -> List[dict]:
    from .documentary import _numbers

    blob = research_text(dossier)
    nums = _numbers(blob)
    scenes = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind") or "").lower()
        if kind == "narrate":
            text = " ".join(str(it.get("narration") or "").split())
            if not text:
                continue
            vis = [x for x in (normalize_visual(v, text, blob, nums) for v in (it.get("visuals") or [])[:3]) if x]
            if not vis:
                vis = [{"type": "words", "text": " ".join(text.split()[:7])}]
            vis = [vis[0]] + [x for x in vis[1:] if x.get("at")]  # a later visual needs its cue
            scenes.append({"kind": "narrate", "narration": text, "visuals": vis, "take": None})
        elif kind == "title":
            title = _clean(it.get("title"), 40)
            if title:
                scenes.append({"kind": "title", "title": title})
    return scenes


def write_script(dossier: dict, channel: str = CHANNEL, minutes: int = 8) -> List[dict]:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    data = _ask_claude_for_json(_script_prompt(dossier, channel, minutes), None, DEFAULT_MODEL)
    scenes = normalize_scenes(data, dossier)
    if not any(s["kind"] == "narrate" for s in scenes):
        raise RuntimeError("Claude didn't return a usable script -- try Write the script again.")
    return scenes


# ------------------------------------------------------------ interaction ---

def quiz_suggestions(scenes: List[dict], starts: List[float]) -> List[dict]:
    """The quiz moments as YouTube Studio in-video quizzes: when, the
    question, the answers and which is right."""
    from .longform_video import _fmt_ts

    out = []
    for i, sc in enumerate(scenes):
        if i >= len(starts) or sc.get("kind") != "narrate":
            continue
        for v in sc.get("visuals") or []:
            if v.get("type") != "quiz":
                continue
            words = (sc.get("narration") or "").split()
            offset = 0.0
            if v.get("at"):
                head = v["at"].split()[0].lower().strip(".,!?")
                idx = next((k for k, w in enumerate(words) if w.lower().strip(".,!?") == head), 0)
                offset = idx / 2.5
            out.append({"time": _fmt_ts(starts[i] + offset + 2), "question": v["question"], "options": v["options"],
                        "answer": v["answer"]})
    return out


def write_publish_text(project: dict, dossier: dict, scenes: List[dict], starts: List[float]) -> dict:
    """Three title options, a description with chapters and sources, and the
    quizzes to add in YouTube Studio."""
    from .longform_video import _fmt_ts, chapters
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    script = "\n".join((f"[{sc['title']}]" if sc.get("kind") == "title" else sc.get("narration", ""))
                       for sc in scenes if sc.get("kind") in ("title", "narrate"))[:9000]
    prompt = f"""You write YouTube titles and descriptions for "{CHANNEL}", a channel that
explains the tech behind gaming, streaming and the internet.

The episode's narration:
{script}

Write:
- "titles": 3 title options under 70 characters: the question the video
  answers, or the surprising fact at its heart. Factual, no ALL CAPS, no
  hype words like "insane", "crazy" or "mind-blowing".
- "description": 2 short paragraphs (under 90 words total), plain and factual.

Respond with ONLY a JSON array holding one object:
[{{"titles": ["...", "...", "..."], "description": "..."}}]
"""
    try:
        data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    except Exception as e:
        print(f"[explainer] publish text failed: {e}", flush=True)
        data = []
    obj = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, dict) else {})
    titles = [_clean(t, 100) for t in obj.get("titles") or [] if str(t).strip()][:3] or [project.get("title") or dossier.get("topic") or CHANNEL]
    lines = [_clean(obj.get("description"), 1200)]
    chs = chapters(scenes, starts)
    if chs:
        lines += ["", "Chapters"] + [f"{_fmt_ts(t)} {n}" for t, n in chs]
    src = sources(dossier)
    if src:
        lines += ["", "Sources"] + [f"- {s['title']}: {s['url']}" for s in src]
    lines += ["", "Diagrams made with code. Narrated by a human."]
    return {"titles": titles, "description": "\n".join(lines).strip(), "quiz": quiz_suggestions(scenes, starts)}


# ------------------------------------------------------------- thumbnails ---

def hook_options(project: dict, dossier: Optional[dict] = None) -> List[str]:
    from .documentary import _numbers
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    script = " ".join(sc.get("narration", "") for sc in project.get("scenes") or [] if sc.get("kind") == "narrate")
    prompt = f"""You write the big text on a YouTube thumbnail for "{CHANNEL}", a channel
explaining the tech behind gaming and streaming. The video: "{project.get('title') or ''}".

What the narrator says:
{script[:3000]}

Write 3 options, each 1 to 3 words and at most 20 characters: a question
or a short claim that makes a gamer curious ("YOU SHOT FIRST?"). Plain
words, no hype words, no emojis.

Respond with ONLY a JSON array of 3 strings.
"""
    try:
        data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    except Exception as e:
        print(f"[explainer] hooks failed: {e}", flush=True)
        data = []
    blob = research_text(dossier or {"topic": project.get("topic") or ""})
    nums = _numbers(blob)
    out = []
    for h in data if isinstance(data, list) else []:
        h = _clean(h, 22).upper().replace('"', "")
        # a number on a thumbnail has to be true, like one in a diagram
        if any(not _supported(n, blob, nums) for n in re.findall(r"\d[\d,.]*", h)):
            continue
        if h and len(h.split()) <= 4 and h not in out:
            out.append(h)
    for f in ("HOW DOES IT WORK?", "THE REAL REASON", "EXPLAINED"):
        if len(out) >= 3:
            break
        if f not in out:
            out.append(f)
    return out[:3]


THUMB_LAYOUTS = ("left", "center", "bottom")
_VISUAL_RANK = {"race": 0, "network": 1, "flow": 2, "layers": 3, "bars": 4, "grid": 5, "neural": 6, "compare": 7,
                "bignum": 8, "quiz": 9, "words": 10}


def _thumb_scenes(scenes: List[dict], n: int = 3) -> List[int]:
    ranked = sorted((i for i, s in enumerate(scenes) if s.get("kind") == "narrate" and s.get("visuals")),
                    key=lambda i: (_VISUAL_RANK.get(scenes[i]["visuals"][0]["type"], 9), i))
    return ranked[:n] or [i for i, s in enumerate(scenes) if s.get("kind") == "narrate"][:n]


def draw_thumb(layout: str, frame_path: Path, hook: str, out: Path) -> Path:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    from .longform_thumbnail import TH, TW, _draw_hook, _hook_lines

    base = Image.open(frame_path).convert("RGB").resize((TW, TH), Image.LANCZOS)
    if layout == "left":
        pic = base.resize((int(TW * 0.62), int(TH * 0.62)), Image.LANCZOS)
        img = ImageEnhance.Brightness(base.filter(ImageFilter.GaussianBlur(18))).enhance(0.35)
        img.paste(pic, (TW - pic.width - 30, (TH - pic.height) // 2))
        fnt, rows = _hook_lines(hook, 500)
        _draw_hook(img, hook, 44, (TH - len(rows) * int(fnt.size * 1.06)) // 2, 500)
    elif layout == "center":
        img = ImageEnhance.Brightness(base).enhance(0.45)
        fnt, rows = _hook_lines(hook, 1150, 190)
        _draw_hook(img, hook, 65, (TH - len(rows) * int(fnt.size * 1.06)) // 2, 1150, "center", 190)
    else:
        img = base.copy()
        shade = Image.new("L", (TW, TH))
        for y in range(TH // 2, TH):
            ImageDraw.Draw(shade).line((0, y, TW, y), fill=int(230 * ((y - TH / 2) / (TH / 2)) ** 1.4))
        img = Image.composite(Image.new("RGB", (TW, TH)), img, shade)
        fnt, rows = _hook_lines(hook, 1180, 150, max_rows=1)
        _draw_hook(img, hook, 50, TH - 40 - int(fnt.size * 1.06), 1180, "center", 150, max_rows=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=90, optimize=True)
    return out


def make_thumbnails(project_dir: Path, project: dict, dossier: Optional[dict] = None) -> dict:
    """Three thumbnails from the episode's own diagrams, each with its own
    hook. Same record shape as the documentaries' ("items", "chosen",
    "frames"), so choosing, downloading and setting it on YouTube are
    shared."""
    from .explainer_visuals import still

    scenes = project.get("scenes") or []
    picks = _thumb_scenes(scenes)
    if not picks:
        raise RuntimeError("write the script first")
    tdir = project_dir / "thumbs"
    fdir = tdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    frames = []
    for k, i in enumerate(picks):
        name = f"diagram_{k}_{stamp}.png"
        still(scenes[i], which=0).save(fdir / name)
        frames.append({"file": name, "scene": i})
    hooks = hook_options(project, dossier)
    items = []
    for k, layout in enumerate(THUMB_LAYOUTS):
        fr = frames[k % len(frames)]
        name = f"thumb_{k}_{stamp}.jpg"
        draw_thumb(layout, fdir / fr["file"], hooks[k % len(hooks)], tdir / name)
        items.append({"layout": layout, "hook": hooks[k % len(hooks)], "file": name, "frames": [fr["file"]]})
    for old in list(tdir.glob("thumb_*.jpg")) + list(fdir.glob("diagram_*.png")):
        if old.name not in {it["file"] for it in items} | {f["file"] for f in frames}:
            old.unlink(missing_ok=True)
    return {"items": items, "chosen": 0, "frames": frames}


def redraw(project_dir: Path, record: dict, index: int, hook: str) -> dict:
    tdir = project_dir / "thumbs"
    item = record["items"][index]
    frame = tdir / "frames" / item["frames"][0]
    if not frame.is_file():
        raise RuntimeError("the diagram for this thumbnail is gone -- make new thumbnails")
    name = f"thumb_{index}_{uuid.uuid4().hex[:8]}.jpg"
    draw_thumb(item["layout"], frame, hook, tdir / name)
    if item["file"] != name:
        (tdir / item["file"]).unlink(missing_ok=True)
    item.update(hook=hook, file=name)
    return record
