"""Thumbnails for the long-form documentaries (1280x720 JPEGs): real frames
from the streamer's own clips -- faces first -- with a short hook line in big
letters, in three layouts to pick from:

  face   the best face shot zoomed in on the right, the hook on a dark
         fade on the left (the classic creator-documentary thumbnail)
  full   one strong frame edge to edge, a red circle on the face, the hook
         across the bottom
  split  "then vs now": the oldest good frame (faded) next to a recent one,
         their years on each side and the hook across the top

The hook is written by Claude from the episode's story (2-4 words); a
number in it must appear in the story or the research, like the stat
cards. Frames are scored with OpenCV's face detector (the same Haar cascade
reframe.py uses), sharpness and brightness, so a blurry or black frame
doesn't win.
"""
from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import List

from .longform_beats import F, text_w, wrap

TW, TH = 1280, 720
YEL = (250, 204, 21)
WHITE = (255, 255, 255)
RED = (239, 68, 68)
LAYOUTS = ("face", "full", "split")


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


# ----------------------------------------------------------------- frames ---

def _grab(clip: Path, t: float, out: Path) -> bool:
    r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
                        "-vf", f"scale={TW}:{TH}:force_original_aspect_ratio=increase,crop={TW}:{TH}", "-q:v", "2", str(out)],
                       capture_output=True, timeout=60)
    return r.returncode == 0 and out.is_file() and out.stat().st_size > 0


def _score(path: Path) -> dict:
    """Biggest face (in 1280x720 coordinates), sharpness and brightness of
    a frame, and an overall score."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return {"score": -1, "face": None}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (640, 360))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=6, minSize=(36, 36))
    face = None
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = [int(x * 2), int(y * 2), int(w * 2), int(h * 2)]
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    face_part = min(1.0, (face[2] * face[3]) / (TW * TH * 0.06)) if face else 0.0
    sharp_part = min(1.0, sharp / 400.0)
    bright_part = 1.0 if 45 <= bright <= 200 else 0.3
    return {"score": round(0.55 * face_part + 0.3 * sharp_part + 0.15 * bright_part, 4), "face": face}


def candidate_frames(project_dir: Path, library: List[dict], out_dir: Path, max_clips: int = 6, per_clip: int = 6) -> List[dict]:
    """The best frame of each of the streamer's most-viewed clips, best
    first: {"clip", "t", "file", "face", "date", "score"}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top = sorted((c for c in library if (project_dir / c["file"]).is_file()),
                 key=lambda c: int(c.get("views") or 0), reverse=True)[:max_clips]
    best: List[dict] = []
    for c in top:
        dur = float(c.get("duration") or 0) or 10.0
        found = None
        for k in range(per_clip):
            t = dur * (k + 1) / (per_clip + 1)
            f = out_dir / f"{c['id']}_{k}.jpg"
            if not f.is_file() and not _grab(project_dir / c["file"], t, f):
                continue
            s = _score(f)
            if found is None or s["score"] > found["score"]:
                found = {"clip": c["id"], "t": round(t, 2), "file": f.name, "face": s["face"],
                         "date": c.get("date") or "", "score": s["score"]}
        if found:
            best.append(found)
    best.sort(key=lambda f: f["score"], reverse=True)
    return best


# ------------------------------------------------------------------ hooks ---

def hook_options(project: dict, dossier: dict, library: List[dict]) -> List[str]:
    """Three 2-4 word hook lines for the thumbnail."""
    from .documentary import _numbers, _value_ok, research_text
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    name = (project.get("streamer") or {}).get("display_name") or project.get("login") or "this streamer"
    story = " ".join(sc.get("narration", "") for sc in project.get("scenes") or [] if sc.get("kind") == "narrate")
    prompt = f"""You write the big text on a YouTube thumbnail for a documentary about the
Twitch streamer {name}. The video's title: "{project.get('title') or f'The Story of {name}'}".

The story (what the narrator says):
{story[:3500]}

Write 3 different options. Each is 2 to 4 words, the kind of short line that
makes someone curious to click: a turning point, a contrast, a question the
video answers. Only facts from the story; a number only if the story has it.
Plain words, no hype words, no emojis, no hashtags.

Respond with ONLY a JSON array of 3 strings: ["...", "...", "..."]
"""
    try:
        data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    except Exception as e:
        print(f"[thumbnail] hooks failed: {e}", flush=True)
        data = []
    blob = (research_text(dossier, library) + " " + story).lower()
    nums = _numbers(blob)
    out = []
    for h in data if isinstance(data, list) else []:
        h = " ".join(str(h).replace('"', "").split()).upper()
        if not h or len(h.split()) > 5 or len(h) > 30:
            continue
        if any(ch.isdigit() for ch in h) and not _value_ok(h.lower(), blob, nums) \
                and not all(_value_ok(d, blob, nums) for d in re.findall(r"\d[\d,.]*[kmb]?", h.lower())):
            continue
        if h not in out:
            out.append(h)
    for fallback in (f"WHO IS {name.upper()}?", "HOW IT STARTED", "THE FULL STORY"):
        if len(out) >= 3:
            break
        if fallback not in out and len(fallback) <= 30:
            out.append(fallback)
    return out[:3]


# ---------------------------------------------------------------- drawing ---

def _open(path: Path):
    from PIL import Image, ImageEnhance

    img = Image.open(path).convert("RGB").resize((TW, TH))
    img = ImageEnhance.Color(img).enhance(1.25)
    return ImageEnhance.Contrast(img).enhance(1.08)


def _vignette(img, strength: float = 0.55):
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (TW, TH), 0)
    ImageDraw.Draw(mask).ellipse((-TW * 0.15, -TH * 0.25, TW * 1.15, TH * 1.25), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGB", (TW, TH), (0, 0, 0))
    return Image.composite(img, Image.blend(img, dark, strength), mask)


def _hook_lines(hook: str, max_w: int, start: int = 150, min_size: int = 70):
    size = start
    while True:
        fnt = F("Anton", size)
        rows = wrap(fnt, hook, max_w)
        if (len(rows) <= 2 and all(text_w(fnt, r) <= max_w for r in rows)) or size <= min_size:
            return fnt, rows[:3]
        size -= 8


def _draw_hook(draw, hook: str, x: int, y: int, max_w: int, align: str = "left", start: int = 150) -> int:
    fnt, rows = _hook_lines(hook, max_w, start)
    for i, row in enumerate(rows):
        w = text_w(fnt, row)
        rx = x if align == "left" else x + (max_w - w) / 2
        color = YEL if i == len(rows) - 1 else WHITE
        draw.text((rx, y), row, font=fnt, fill=color, stroke_width=9, stroke_fill=(0, 0, 0))
        y += int(fnt.size * 1.08)
    return y


def _face_zoom(img, face, target_x: float, target_y: float = 0.45, face_h: float = 0.42, max_zoom: float = 2.2):
    """Crop so the face lands at (target_x, target_y) of the frame, about
    face_h of its height."""
    from PIL import Image

    if not face:
        return img
    x, y, w, h = face
    z = max(1.0, min(max_zoom, face_h * TH / max(h, 1)))
    cw, ch = TW / z, TH / z
    fx, fy = x + w / 2, y + h / 2
    x0 = min(max(0.0, fx - target_x * cw), TW - cw)
    y0 = min(max(0.0, fy - target_y * ch), TH - ch)
    return img.resize((TW, TH), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))


def draw(layout: str, frames: List[dict], frames_dir: Path, hook: str, out: Path) -> Path:
    from PIL import Image, ImageDraw

    main = frames[0]
    if layout == "face":
        img = _face_zoom(_open(frames_dir / main["file"]), main.get("face"), target_x=0.7)
        shade = Image.new("L", (TW, TH))
        for x in range(TW):
            a = int(235 * max(0.0, 1 - x / (TW * 0.62)) ** 1.3)
            ImageDraw.Draw(shade).line((x, 0, x, TH), fill=a)
        img = Image.composite(Image.new("RGB", (TW, TH), (0, 0, 0)), img, shade)
        d = ImageDraw.Draw(img)
        _draw_hook(d, hook, 56, 190, 640)
    elif layout == "full":
        img = _vignette(_open(frames_dir / main["file"]))
        d = ImageDraw.Draw(img)
        face = main.get("face")
        if face:
            x, y, w, h = face
            pad = 0.3
            box = [x - w * pad, y - h * pad, x + w * (1 + pad), y + h * (1 + pad)]
            box = [max(8, box[0]), max(8, box[1]), min(TW - 8, box[2]), min(TH - 8, box[3])]
            d.ellipse(box, outline=RED, width=12)
        fnt, rows = _hook_lines(hook, 1160, 140)
        y = TH - 40 - len(rows) * int(fnt.size * 1.08)
        _draw_hook(d, hook, 60, y, 1160, "center", 140)
    else:  # split
        old = min(frames, key=lambda f: f.get("date") or "9999")
        new = max(frames, key=lambda f: f.get("date") or "")
        left = _face_zoom(_open(frames_dir / old["file"]), old.get("face"), target_x=0.5, face_h=0.38)
        right = _face_zoom(_open(frames_dir / new["file"]), new.get("face"), target_x=0.5, face_h=0.38)
        left = Image.blend(left.convert("L").convert("RGB"), left, 0.35)
        img = Image.new("RGB", (TW, TH))
        img.paste(left.resize((TW // 2, TH), box=(TW // 4, 0, TW * 3 // 4, TH)), (0, 0))
        img.paste(right.resize((TW // 2, TH), box=(TW // 4, 0, TW * 3 // 4, TH)), (TW // 2, 0))
        img = _vignette(img, 0.35)
        d = ImageDraw.Draw(img)
        d.rectangle((TW // 2 - 5, 0, TW // 2 + 5, TH), fill=WHITE)
        yf = F("Anton", 64)
        for f, x in ((old, 30), (new, TW // 2 + 30)):
            yr = (f.get("date") or "")[:4]
            if yr:
                d.text((x, TH - 110), yr, font=yf, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
        _draw_hook(d, hook, 40, 24, TW - 80, "center", 120)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=90, optimize=True)
    return out


# ------------------------------------------------------------------ build ---

def make_thumbnails(project_dir: Path, project: dict, dossier: dict, library: List[dict]) -> dict:
    """Three thumbnails, one per layout, each with its own hook. Returns
    the project's "thumbnails" record."""
    tdir = project_dir / "thumbs"
    frames = candidate_frames(project_dir, library, tdir / "frames")
    if not frames:
        raise RuntimeError("couldn't grab any frames from the clips")
    hooks = hook_options(project, dossier, library)
    items = []
    stamp = uuid.uuid4().hex[:8]
    for i, layout in enumerate(LAYOUTS):
        use = frames[i:] + frames[:i] if layout != "split" else frames
        if layout == "split" and len({(f.get("date") or "")[:4] for f in frames}) < 2:
            layout = "face"
            use = frames[min(2, len(frames) - 1):] + frames[:min(2, len(frames) - 1)]
        name = f"thumb_{i}_{stamp}.jpg"
        draw(layout, use, tdir / "frames", hooks[i % len(hooks)], tdir / name)
        items.append({"layout": layout, "hook": hooks[i % len(hooks)], "file": name,
                      "frames": [f["file"] for f in use[:6]]})
    for old in tdir.glob("thumb_*.jpg"):
        if old.name not in {it["file"] for it in items}:
            old.unlink(missing_ok=True)
    return {"items": items, "chosen": 0, "frames": frames}


def redraw(project_dir: Path, record: dict, index: int, hook: str) -> dict:
    """Same thumbnail, new hook text."""
    tdir = project_dir / "thumbs"
    item = record["items"][index]
    by_file = {f["file"]: f for f in record.get("frames") or []}
    frames = [by_file[f] for f in item["frames"] if f in by_file]
    if not frames:
        raise RuntimeError("the frames for this thumbnail are gone -- make new thumbnails")
    # A fresh name every time (a browser shows the new image, and the old
    # file can be removed without any chance of it being the new one).
    name = f"thumb_{index}_{uuid.uuid4().hex[:8]}.jpg"
    draw(item["layout"], frames, tdir / "frames", hook, tdir / name)
    if item["file"] != name:
        (tdir / item["file"]).unlink(missing_ok=True)
    item.update(hook=hook, file=name)
    return record
