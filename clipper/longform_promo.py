"""Cliffhanger promo Shorts for a finished long-form episode: two vertical
clips cut from final.mp4 that stop right before a payoff, so the only way
to find out is the full video (linked in the description, and as the
Short's "Related video").

Built from the episode's own structure, not the clip pipeline's picks:

  opener   the cold-open moment and the narration that asks the question
           the video answers, cut as the title card would start
  turning  a narrated setup and the first second or so of the clip it
           leads into, cut before the payoff -- the turning point Claude
           judges strongest, else the most-viewed clip's

Each is 1080x1920: the episode footage in the middle over a blurred copy of
itself, a question in big letters on top, the channel at the bottom, and a
short end card ("What happened next?") on a freeze of the last frame.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from .longform_beats import F, text_w, wrap

VW, VH = 1080, 1920
FG_H = 608            # the 16:9 episode frame scaled to the Short's width
FG_Y = 620            # where it sits (a little above centre)
MAX_SETUP = 26.0      # at most this much narration before the cut
GLIMPSE = 1.4         # seconds of the payoff clip shown before the cut
END_CARD = 2.4
MAX_CUT = 56.0        # + END_CARD stays under the 60 s the app uploads as a Short
YEL = (250, 204, 21)
WHITE = (255, 255, 255)


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


def _take_words(project_dir: Path, scene: dict) -> List[dict]:
    f = (scene.get("take") or {}).get("file")
    if not f:
        return []
    try:
        return json.loads((project_dir / "takes" / Path(f).with_suffix(".words.json").name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _setup_start(project_dir: Path, scene: dict, start: float, end: float) -> float:
    """Where the narration before the cut should begin: the whole scene if
    it's short, else the sentence start that leaves at most MAX_SETUP."""
    if end - start <= MAX_SETUP:
        return start
    words = _take_words(project_dir, scene)
    starts = [0.0] + [w2["s"] for w1, w2 in zip(words, words[1:]) if re.search(r"[.!?]$", w1["w"])]
    ok = [s for s in starts if end - (start + s) <= MAX_SETUP]
    return start + (min(ok) if ok else (end - start - MAX_SETUP))


def candidates(project: dict, project_dir: Path) -> List[dict]:
    """Every narrated-setup -> clip-moment pair in the episode, as cut
    points in final.mp4: {"kind", "start", "end", "scene", "narration", "clip"}."""
    scenes = project.get("scenes") or []
    starts = (project.get("render") or {}).get("starts") or []
    lib = {c["id"]: c for c in project.get("library") or []}
    out = []
    if len(scenes) >= 3 and len(starts) >= 3 and scenes[0].get("kind") == "moment" and scenes[1].get("kind") == "narrate":
        # Opener: cold open + the question, cut where the title card starts.
        # Too long for a Short (the app uploads Shorts up to 60 s): drop the
        # cold open and keep the end of the question.
        end, start = starts[2], starts[0]
        if end - start > MAX_CUT:
            start = _setup_start(project_dir, scenes[1], starts[1], end)
        out.append({"kind": "opener", "start": start, "end": end, "scene": 1,
                    "narration": scenes[1].get("narration", ""), "clip": scenes[0].get("clip")})
    for i in range(1, len(scenes) - 1):
        if i + 1 >= len(starts) or scenes[i].get("kind") != "narrate" or scenes[i + 1].get("kind") != "moment":
            continue
        if out and out[0]["kind"] == "opener" and i <= 1:
            continue
        s0 = _setup_start(project_dir, scenes[i], starts[i], starts[i + 1])
        cid = scenes[i + 1].get("clip")
        out.append({"kind": "turning", "start": s0, "end": starts[i + 1] + GLIMPSE, "scene": i,
                    "narration": scenes[i].get("narration", ""), "clip": cid,
                    "views": int((lib.get(cid) or {}).get("views") or 0),
                    "clip_title": (lib.get(cid) or {}).get("title") or ""})
    return out


def _ask(project: dict, turning: List[dict], opener: Optional[dict]) -> dict:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    name = (project.get("streamer") or {}).get("display_name") or project.get("login") or "this streamer"
    opts = "\n".join(f"[{k}] narration before the cut: \"{c['narration'][-400:]}\" -> then the clip \"{c['clip_title']}\" ({c['views']:,} views) starts"
                     for k, c in enumerate(turning)) or "(none)"
    prompt = f"""Two promo Shorts are being cut from a YouTube documentary, "{project.get('title') or f'The Story of {name}'}",
about the Twitch streamer {name}. Each stops right before a payoff, so viewers
tap through to the full video to find out what happened.

Short A is the opening: {'the narration that sets up the question: "' + opener['narration'][-500:] + '"' if opener else '(no opener)'}

For short B, pick the strongest cliffhanger among these moments in the story
(a turning point, a surprise, a question left open):
{opts}

For each short write:
- "hook": the line shown in big letters on top, 3 to 7 words, a question or
  an open statement that the short doesn't answer.
- "title": the YouTube title, under 70 characters, honest (it must not
  promise anything the full video doesn't show), with {name}'s name.
- "description": one sentence.
Only facts from the narration. Plain words, no hype words, no emojis.

Respond with ONLY JSON:
{{"pick": 0, "a": {{"hook": "...", "title": "...", "description": "..."}}, "b": {{"hook": "...", "title": "...", "description": "..."}}}}
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else {}
    return data if isinstance(data, dict) else {}


def _text(x, n: int) -> str:
    return " ".join(str(x or "").replace('"', "").split())[:n]


def plan(project: dict, project_dir: Path) -> List[dict]:
    """The two cliffhangers: [{"kind","start","end","hook","title","description"}]."""
    cands = candidates(project, project_dir)
    opener = next((c for c in cands if c["kind"] == "opener"), None)
    turning = [c for c in cands if c["kind"] == "turning"]
    name = (project.get("streamer") or {}).get("display_name") or project.get("login") or "this streamer"
    try:
        ans = _ask(project, turning, opener)
    except Exception as e:
        print(f"[promo] Claude failed, using defaults: {e}", flush=True)
        ans = {}
    try:
        pick = int(ans.get("pick"))
    except (TypeError, ValueError):
        pick = -1
    if not 0 <= pick < len(turning):
        pick = max(range(len(turning)), key=lambda k: turning[k]["views"]) if turning else -1
    chosen = []
    if opener:
        chosen.append(("a", opener))
    if pick >= 0:
        chosen.append(("b", turning[pick]))
    if len(chosen) < 2:  # no opener: take a second turning point
        rest = sorted((c for k, c in enumerate(turning) if k != pick), key=lambda c: -c["views"])
        chosen += [("b2", c) for c in rest[:2 - len(chosen)]]
    out = []
    for key, c in chosen[:2]:
        meta = ans.get(key if key != "b2" else "b") or {}
        out.append({
            "kind": c["kind"], "start": round(c["start"], 2), "end": round(c["end"], 2),
            "hook": _text(meta.get("hook"), 60).upper() or ("WHAT HAPPENED NEXT?" if c["kind"] == "turning" else f"WHO IS {name.upper()}?"),
            "title": _text(meta.get("title"), 100) or f"The Story of {name}",
            "description": _text(meta.get("description"), 300),
        })
    return out


# ---------------------------------------------------------------- render ---

def _overlay(hook: str, brand: str, out: Path) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    size = 104
    while True:
        fnt = F("Anton", size)
        rows = wrap(fnt, hook, VW - 120)
        if len(rows) <= 3 or size <= 64:
            break
        size -= 8
    y = FG_Y - 60 - len(rows) * int(size * 1.1)
    for i, row in enumerate(rows):
        d.text(((VW - text_w(fnt, row)) / 2, y), row, font=fnt, fill=YEL if i == len(rows) - 1 else WHITE,
               stroke_width=8, stroke_fill=(0, 0, 0))
        y += int(size * 1.1)
    bf = F("Inter-Black", 40)
    s = brand.upper()
    d.text(((VW - text_w(bf, s)) / 2, FG_Y + FG_H + 70), s, font=bf, fill=WHITE, stroke_width=3, stroke_fill=(0, 0, 0))
    img.save(out)
    return out


def _end_card(out: Path) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (VW, VH), (0, 0, 0, 170))
    d = ImageDraw.Draw(img)
    f1 = F("Anton", 130)
    y = 700
    for i, row in enumerate(["WHAT HAPPENED", "NEXT?"]):
        d.text(((VW - text_w(f1, row)) / 2, y), row, font=f1, fill=YEL if i else WHITE, stroke_width=8, stroke_fill=(0, 0, 0))
        y += 150
    f2 = F("Inter-Black", 50)
    for row in ["Full story on the channel", "tap the video below"]:
        d.text(((VW - text_w(f2, row)) / 2, y + 40), row, font=f2, fill=WHITE)
        y += 66
    cx, cy = VW // 2, y + 120
    d.polygon([(cx - 60, cy - 30), (cx + 60, cy - 30), (cx, cy + 40)], fill=YEL)
    img.save(out)
    return out


def render_short(final: Path, cut: dict, brand: str, out: Path) -> float:
    """Cut [start, end] of the episode into a vertical cliffhanger Short
    with its end card. Returns its length in seconds."""
    tmpdir = out.parent
    ov = _overlay(cut["hook"], brand, tmpdir / f".{out.stem}_ov.png")
    ec = _end_card(tmpdir / f".{out.stem}_end.png")
    a, b = float(cut["start"]), float(cut["end"])
    length = b - a
    total = length + END_CARD
    fc = (f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,fps=30,tpad=stop_mode=clone:stop_duration={END_CARD},split[s1][s2];"
          f"[s1]scale={VW}:{VH}:force_original_aspect_ratio=increase,crop={VW}:{VH},boxblur=24:3,eq=brightness=-0.12[bg];"
          f"[s2]scale={VW}:{FG_H}[fg];[bg][fg]overlay=0:{FG_Y}[v1];[v1][1:v]overlay=0:0[v2];"
          f"[2:v]format=rgba,fade=t=in:st={length:.3f}:d=0.3:alpha=1[ec];[v2][ec]overlay=0:0:enable='gte(t,{length:.3f})',format=yuv420p[v];"
          f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,afade=t=out:st={max(0.0, length - 0.35):.3f}:d=0.35,"
          f"apad=pad_dur={END_CARD}[a]")
    cmd = [_ffmpeg(), "-y", "-v", "error", "-i", str(final), "-loop", "1", "-i", str(ov), "-loop", "1", "-i", str(ec),
           "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
    finally:
        ov.unlink(missing_ok=True)
        ec.unlink(missing_ok=True)
    return round(total, 2)
