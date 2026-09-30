"""Rendering for the long-form streamer documentaries (see documentary.py).

Every scene becomes its own 1080p30 segment with sound, then the segments
are joined:

  narrate  the creator's recorded take over the streamer's clip footage
           (dimmed a touch, its own sound kept very low underneath), or
           over a plain title card when the scene has no clip
  moment   a stretch of a clip at full volume with burned-in subtitles
           from its transcript -- the payoff the narration builds to
  title    a short chapter card (these become the YouTube chapters)

plus a corner caption (date / stat) and a small channel watermark on every
clip scene, an end card for YouTube's end screens, and optional music
ducked under everything. Pillow draws the cards and text; ffmpeg does all
the video work. Segments are cached by a hash of what went into them, so
re-rendering after one change only redoes that scene.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

W, H, FPS = 1920, 1080, 30
TITLE_SECONDS = 3.0
END_CARD_SECONDS = 12.0
LAST_SCENE_TAIL = 0.8

BG = (8, 12, 22)
TEXT = (243, 244, 246)
MUTED = (156, 163, 175)
AMBER = (245, 158, 11)
RED = (239, 68, 68)

_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


# ----------------------------------------------------------------- fonts ---

_FONT_FILES = {
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
_FC_NAMES = {"sans": "DejaVu Sans", "bold": "DejaVu Sans:bold"}
_font_cache: dict = {}


def font(size: int, kind: str = "sans"):
    from PIL import ImageFont

    key = (kind, size)
    if key in _font_cache:
        return _font_cache[key]
    path = next((p for p in _FONT_FILES[kind] if Path(p).exists()), None)
    if path is None:
        try:
            out = subprocess.run(["fc-match", "-f", "%{file}", _FC_NAMES[kind]], capture_output=True, text=True, timeout=10)
            path = out.stdout.strip() or None
        except Exception:
            path = None
    f = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    _font_cache[key] = f
    return f


def _wrap(draw, text: str, fnt, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------- images ---

def overlay_layer(caption: str, brand: str):
    """Corner caption (top-left) + channel watermark (top-right), as one
    transparent full-frame layer. Top corners keep the bottom free for
    subtitles."""
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    wm = brand.upper()
    wf = font(26, "bold")
    d.text((W - 60 - d.textlength(wm, font=wf), 52), wm, font=wf, fill=(255, 255, 255, 170),
           stroke_width=2, stroke_fill=(0, 0, 0, 120))
    if caption:
        cf = font(36, "bold")
        cw = d.textlength(caption, font=cf)
        x, y = 60, 44
        d.rounded_rectangle([x, y, x + cw + 60, y + 70], radius=10, fill=(8, 12, 22, 210))
        d.rectangle([x, y, x + 8, y + 70], fill=RED + (255,))
        d.text((x + 32, y + 14), caption, font=cf, fill=TEXT + (255,))
    return layer


def subtitle_layer(text: str):
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font(54, "bold")
    rows = _wrap(d, text, f, W - 360)[:2]
    y = H - 110 - 70 * len(rows)
    for row in rows:
        d.text(((W - d.textlength(row, font=f)) / 2, y), row, font=f, fill=(255, 255, 255, 255),
               stroke_width=5, stroke_fill=(0, 0, 0, 255))
        y += 70
    return layer


def subtitle_lines(words: List[dict], start: float, end: float) -> List[Tuple[float, float, str]]:
    """Group a moment's words into short on-screen lines (times relative to
    the moment's start)."""
    ws = [w for w in words if w["e"] > start + 0.05 and w["s"] < end - 0.05]
    lines, cur = [], []
    for w in ws:
        if cur and (len(cur) >= 7 or w["s"] - cur[-1]["e"] > 0.7 or len(" ".join(x["w"] for x in cur)) > 40):
            lines.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?]$", w["w"]) and len(cur) >= 3:
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        a = max(0.0, ln[0]["s"] - start)
        b = min(end - start, ln[-1]["e"] - start + 0.25)
        if b - a > 0.2:
            out.append((round(a, 2), round(b, 2), " ".join(x["w"] for x in ln)))
    return out


def card_image(title: str, kicker: str = ""):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r in range(900, 0, -90):
        c = int(20 * (1 - r / 900))
        d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=(8 + c, 12 + c, 22 + c * 2))
    tf = font(96, "bold")
    rows = _wrap(d, title.upper(), tf, W - 360)[:3]
    y = H / 2 - len(rows) * 58
    if kicker:
        kf = font(34, "bold")
        d.text(((W - d.textlength(kicker.upper(), font=kf)) / 2, y - 80), kicker.upper(), font=kf, fill=AMBER)
    for row in rows:
        d.text(((W - d.textlength(row, font=tf)) / 2, y), row, font=tf, fill=TEXT)
        y += 116
    d.rectangle([W / 2 - 110, y + 24, W / 2 + 30, y + 32], fill=RED)
    d.rectangle([W / 2 + 44, y + 24, W / 2 + 110, y + 32], fill=AMBER)
    return img


def end_card_image(brand: str, line: str = "A new story every two weeks"):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r in range(900, 0, -90):
        c = int(18 * (1 - r / 900))
        d.ellipse([W / 2 - r, H / 2 - r, W / 2 + r, H / 2 + r], fill=(8 + c, 12 + c, 22 + c * 2))
    bf = font(84, "bold")
    t = brand.upper()
    d.text(((W - d.textlength(t, font=bf)) / 2, 120), t, font=bf, fill=TEXT)
    d.rectangle([W / 2 - 110, 236, W / 2 + 30, 244], fill=RED)
    d.rectangle([W / 2 + 44, 236, W / 2 + 110, 244], fill=AMBER)
    sf = font(34)
    d.text(((W - d.textlength(line, font=sf)) / 2, 272), line, font=sf, fill=MUTED)
    return img


# ------------------------------------------------------------- encoding ---

def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


def _run(cmd: list, timeout: int = 900) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-500:]}")


def has_audio(path: Path) -> bool:
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60)
    return "Audio:" in r.stderr


def media_duration(path: Path) -> float:
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise RuntimeError(f"couldn't read the length of {path.name}")
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])


def _vchain(dim: bool) -> str:
    chain = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1"
    if dim:
        chain += ",eq=brightness=-0.07:saturation=0.85"
    return chain


def _encode_still(img_path: Path, audio: Optional[Path], duration: float, out: Path, fade_out: bool = False) -> None:
    cmd = [_ffmpeg(), "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-i", str(img_path)]
    if audio is not None:
        cmd += ["-i", str(audio)]
        afilt = "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[a]"
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        afilt = "[1:a]anull[a]"
    vf = "[0:v]format=yuv420p,fade=t=in:st=0:d=0.3" + (f",fade=t=out:st={max(0.0, duration - 0.3):.2f}:d=0.3" if fade_out else "") + "[v]"
    cmd += ["-filter_complex", f"{vf};{afilt}", "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}", *_ENC, str(out)]
    _run(cmd)


def _encode_narrate_clip(clip: Path, start: float, take: Path, overlay_png: Path, duration: float, out: Path) -> None:
    cmd = [_ffmpeg(), "-y", "-v", "error", "-stream_loop", "-1", "-ss", f"{start:.2f}", "-i", str(clip),
           "-i", str(take), "-i", str(overlay_png)]
    fc = f"[0:v]{_vchain(True)}[bv];[bv][2:v]overlay=0:0,fade=t=in:st=0:d=0.25[v];"
    fc += "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[vo];"
    if has_audio(clip):
        fc += "[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.10[bg];[vo][bg]amix=inputs=2:duration=first:normalize=0[a]"
    else:
        fc += "[vo]anull[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{duration:.3f}", *_ENC, str(out)]
    _run(cmd)


def _encode_moment(clip: Path, start: float, end: float, overlay_png: Path, subs: List[Tuple[float, float, Path]], out: Path) -> None:
    length = end - start
    cmd = [_ffmpeg(), "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{length:.3f}", "-i", str(clip), "-i", str(overlay_png)]
    for _a, _b, p in subs:
        cmd += ["-i", str(p)]
    fc = f"[0:v]{_vchain(False)}[b];[b][1:v]overlay=0:0[v0];"
    for k, (a, b, _p) in enumerate(subs):
        fc += f"[v{k}][{k + 2}:v]overlay=0:0:enable='between(t,{a:.2f},{b:.2f})'[v{k + 1}];"
    fc += f"[v{len(subs)}]fade=t=in:st=0:d=0.2,fade=t=out:st={max(0.0, length - 0.2):.2f}:d=0.2[v];"
    if has_audio(clip):
        fc += "[0:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,apad[a]"
        cmd_audio = []
    else:
        cmd_audio = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        fc += f"[{len(subs) + 2}:a]anull[a]"
    cmd += cmd_audio + ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}", *_ENC, str(out)]
    _run(cmd)


# --------------------------------------------------------------- scenes ---

def scene_duration(scene: dict, take_duration: Optional[float], last: bool) -> float:
    kind = scene.get("kind")
    if kind == "title":
        return TITLE_SECONDS
    if kind == "moment":
        return max(1.0, float(scene["end"]) - float(scene["start"]))
    from .longform import SCENE_GAP_SECONDS

    return max(1.0, float(take_duration or 0) + (LAST_SCENE_TAIL if last else SCENE_GAP_SECONDS))


def _scene_key(scene: dict, duration: float, brand: str, clip_file: Optional[str]) -> str:
    keep = {k: scene.get(k) for k in ("kind", "narration", "clip", "start", "end", "caption", "title")}
    keep["take"] = (scene.get("take") or {}).get("file")
    blob = json.dumps({"s": keep, "d": round(duration, 3), "b": brand, "c": clip_file, "v": 2}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def render_scene(project_dir: Path, scene: dict, index: int, duration: float, library: dict, brand: str,
                 chapter_no: int) -> Path:
    from .documentary import clip_words

    rdir = project_dir / "render"
    rdir.mkdir(parents=True, exist_ok=True)
    clip = library.get(scene.get("clip") or "")
    clip_path = project_dir / clip["file"] if clip else None
    out = rdir / f"scene{index:02d}_{_scene_key(scene, duration, brand, clip and clip['file'])}.mp4"
    if out.exists():
        return out
    tmp = out.with_name(f".{out.name}")
    work = []
    try:
        kind = scene.get("kind")
        if kind == "title":
            img = rdir / f".card{index:02d}.png"
            work.append(img)
            card_image(scene["title"], f"Chapter {chapter_no}" if chapter_no else "").save(img)
            _encode_still(img, None, duration, tmp, fade_out=True)
        elif kind == "moment" and clip_path and clip_path.exists():
            ov = rdir / f".ov{index:02d}.png"
            work.append(ov)
            overlay_layer(scene.get("caption") or "", brand).save(ov)
            subs = []
            for k, (a, b, text) in enumerate(subtitle_lines(clip_words(project_dir, clip["id"]), scene["start"], scene["end"])):
                sp = rdir / f".sub{index:02d}_{k:02d}.png"
                subtitle_layer(text).save(sp)
                work.append(sp)
                subs.append((a, b, sp))
            _encode_moment(clip_path, float(scene["start"]), float(scene["end"]), ov, subs, tmp)
        else:
            take = project_dir / "takes" / scene["take"]["file"]
            if clip_path and clip_path.exists():
                ov = rdir / f".ov{index:02d}.png"
                work.append(ov)
                overlay_layer(scene.get("caption") or "", brand).save(ov)
                _encode_narrate_clip(clip_path, float(scene.get("start") or 0), take, ov, duration, tmp)
            else:
                img = rdir / f".card{index:02d}.png"
                work.append(img)
                card_image(scene.get("caption") or brand).save(img)
                _encode_still(img, take, duration, tmp)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
        for p in work:
            p.unlink(missing_ok=True)
    return out


def preview_still(project_dir: Path, scene: dict, library: dict, brand: str, out: Path) -> Path:
    """One frame of what a scene will look like, for the story editor."""
    from PIL import Image

    from .documentary import clip_words

    out.parent.mkdir(parents=True, exist_ok=True)
    clip = library.get(scene.get("clip") or "")
    if scene.get("kind") == "title" or not clip or not (project_dir / clip["file"]).exists():
        img = card_image(scene.get("title") or scene.get("caption") or brand).convert("RGBA")
    else:
        t = float(scene.get("start") or 0) + (1.5 if scene.get("kind") == "moment" else 1.0)
        tmp = out.with_suffix(".src.jpg")
        _run([_ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(project_dir / clip["file"]), "-frames:v", "1",
              "-vf", _vchain(scene.get("kind") == "narrate"), str(tmp)], timeout=120)
        img = Image.open(tmp).convert("RGBA")
        tmp.unlink(missing_ok=True)
        img = Image.alpha_composite(img, overlay_layer(scene.get("caption") or "", brand))
        if scene.get("kind") == "moment":
            lines = subtitle_lines(clip_words(project_dir, clip["id"]), scene["start"], scene["end"])
            if lines:
                img = Image.alpha_composite(img, subtitle_layer(lines[0][2]))
    img.convert("RGB").resize((640, 360), Image.LANCZOS).save(out, quality=85)
    return out


# ------------------------------------------------------------- assembly ---

def render_documentary(
    project_dir: Path, scenes: List[dict], library: dict, take_durations: List[Optional[float]], brand: str,
    music: Optional[Path], on_progress: Callable[[float, str], None] = lambda p, m: None,
) -> Tuple[Path, List[float], float]:
    """Render every scene, add the end card, join, and lay optional music
    under it. Returns (final video, each scene's start time, total length)."""
    rdir = project_dir / "render"
    rdir.mkdir(parents=True, exist_ok=True)
    n = len(scenes)
    parts, starts, t, titles_seen = [], [], 0.0, 0
    for i, sc in enumerate(scenes):
        on_progress(0.03 + 0.85 * i / max(n, 1), f"Rendering scene {i + 1} of {n}...")
        chapter = 0
        if sc.get("kind") == "title":
            titles_seen += 1
            chapter = titles_seen - 1  # the first card is the episode title, then Chapter 1, 2, ...
        dur = scene_duration(sc, take_durations[i], last=(i == n - 1))
        path = render_scene(project_dir, sc, i, dur, library, brand, chapter)
        real = media_duration(path)
        parts.append(path)
        starts.append(round(t, 2))
        t += real
    on_progress(0.9, "Adding the end card...")
    end = rdir / f"endcard_{hashlib.sha1(brand.encode()).hexdigest()[:8]}.mp4"
    if not end.exists():
        img = rdir / ".endcard.png"
        end_card_image(brand).save(img)
        _encode_still(img, None, END_CARD_SECONDS, end)
        img.unlink(missing_ok=True)
    parts.append(end)
    total = t + END_CARD_SECONDS

    on_progress(0.93, "Joining scenes...")
    listing = rdir / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    joined = rdir / "joined.mp4"
    _run([_ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])

    final = project_dir / "final.mp4"
    tmp = project_dir / ".final.partial.mp4"
    if music is not None and music.exists():
        on_progress(0.96, "Mixing in the music...")
        fc = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},volume=0.16,"
              f"afade=t=in:d=2,afade=t=out:st={max(0.0, total - 4):.3f}:d=4[bed];"
              f"[0:a]asplit=2[main][key];[bed][key]sidechaincompress=threshold=0.02:ratio=10:attack=30:release=500[ducked];"
              f"[main][ducked]amix=inputs=2:duration=first:normalize=0[a]")
        _run([_ffmpeg(), "-y", "-v", "error", "-i", str(joined), "-stream_loop", "-1", "-i", str(music),
              "-filter_complex", fc, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
              "-ac", "2", "-t", f"{total:.3f}", "-movflags", "+faststart", str(tmp)])
    else:
        _run([_ffmpeg(), "-y", "-v", "error", "-i", str(joined), "-c", "copy", "-movflags", "+faststart", str(tmp)])
    tmp.replace(final)
    joined.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    on_progress(1.0, "Done.")
    return final, starts, round(total, 2)


# ------------------------------------------------------- title & chapters ---

def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters(scenes: List[dict], starts: List[float]) -> List[Tuple[float, str]]:
    """YouTube chapters from the chapter cards: the first must be at 0:00,
    there must be at least 3, each at least 10 seconds long."""
    marks = [(starts[i], sc["title"]) for i, sc in enumerate(scenes) if sc.get("kind") == "title" and i < len(starts)]
    if not marks:
        return []
    if marks[0][0] > 0:
        marks.insert(0, (0.0, "Intro"))
    out = []
    for t, name in marks:
        if out and t - out[-1][0] < 10:
            # Two cards back to back (e.g. the episode title, then chapter
            # 1): one chapter, named after the later, more specific card.
            out[-1] = (out[-1][0], name)
            continue
        out.append((0.0 if not out else t, name))
    return out if len(out) >= 3 else []


def write_publish_text(streamer: dict, channel: str, scenes: List[dict], starts: List[float], sources: List[str]) -> dict:
    """Three title options and a description with chapters and credits."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    name = streamer.get("display_name") or streamer.get("login") or "this streamer"
    story = "\n".join(
        (f"[{sc['title']}]" if sc.get("kind") == "title" else sc.get("narration", ""))
        for sc in scenes if sc.get("kind") in ("title", "narrate")
    )[:9000]
    prompt = f"""You write YouTube titles and descriptions for "The Story Of", a documentary
series on the channel "{channel}". This episode is about the Twitch streamer {name}.

The episode's narration:
{story}

Write:
- "titles": 3 title options under 70 characters. Built on the real story
  (the rise, the turning point, the question the video answers), factual,
  with {name}'s name in each. No ALL CAPS and no hype words like "insane",
  "crazy" or "chaos".
- "description": 2 short paragraphs (under 90 words total), plain and factual.

Respond with ONLY a JSON array holding one object:
[{{"titles": ["...", "...", "..."], "description": "..."}}]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    obj = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else (data if isinstance(data, dict) else {})
    titles = [" ".join(str(t).split())[:100] for t in obj.get("titles") or [] if str(t).strip()][:3] or [f"The Story of {name}"]
    lines = [" ".join(str(obj.get("description") or "").split())]
    chs = chapters(scenes, starts)
    if chs:
        lines += ["", "Chapters"] + [f"{_fmt_ts(t)} {n}" for t, n in chs]
    login = streamer.get("login")
    lines += ["", f"All clips: {name}" + (f" on Twitch (twitch.tv/{login})" if login else "") + ", clipped by the Twitch community."]
    if sources:
        lines += ["Sources:"] + [f"- {s}" for s in sources]
    return {"titles": titles, "description": "\n".join(lines).strip()}
