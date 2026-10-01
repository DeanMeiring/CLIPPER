"""Thumbnails for the long-form documentaries (1280x720 JPEGs): real frames
from the streamer's own clips with one big, clear face, cut out from a
darkened background, and a short hook line -- in three layouts to pick from:

  face   the face big on the right, the hook on the left
  full   the face big in the middle, the hook along the bottom
  split  "then vs now": the streamer in their oldest clip next to their
         newest, the years on each side and the hook across the top

What makes a frame: a face, as big and sharp as the clips have. Frames
without one (gameplay, a hand, an empty room) are never used while a face
frame exists. Frames are grabbed at the clip's own resolution and cropped
tight on the face, which also crops out the stream's chat, timers and HUD.
The person is cut out with MediaPipe's selfie-segmentation model (Apache
2.0, 250 KB, bundled in assets/models and run with OpenCV's dnn module --
no extra packages), the background blurred and darkened, and a white
outline added: the look streamer documentaries use. If the cut-out fails
on a frame, it falls back to a dark vignette.

The hook is written by Claude from the episode's story (1-3 words, no
in-jokes); a number in it must appear in the story or the research, like
the stat cards. Faces are found with OpenCV's Haar cascades (the same one
reframe.py uses), checked for eyes.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from .longform_beats import F, text_w, wrap

TW, TH = 1280, 720
YEL = (250, 204, 21)
WHITE = (255, 255, 255)
LAYOUTS = ("face", "full", "split")
MAX_SOURCE_W = 1920
# A face this tall (source pixels) fills the "face" layout without being
# blown up more than ~1.2x; smaller faces still work, they just score lower.
GOOD_FACE_PX = 300
# Never blow a frame up more than this: a small webcam face then sits a
# bit smaller in the picture instead of turning to mush.
MAX_ZOOM = 2.5
SEG_MODEL = Path(__file__).parent / "assets" / "models" / "selfie_segmentation_landscape.tflite"
FACE_MODEL = Path(__file__).parent / "assets" / "models" / "face_detection_short_range.tflite"
# BlazeFace's score (a logit) on a square around a Haar face: real faces,
# webcam-sized ones included, came out at 2.1-2.7 in testing; a cartoon
# character's face at 0.3-0.9.
FACE_MIN_LOGIT = 1.4


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


# ----------------------------------------------------------------- frames ---

def _grab(clip: Path, t: float, out: Path) -> bool:
    """One frame at the clip's own resolution (capped at 1920 wide), so a
    webcam face in a corner still has pixels to zoom into."""
    r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
                        "-vf", f"scale='min({MAX_SOURCE_W},iw)':-2", "-q:v", "2", str(out)],
                       capture_output=True, timeout=60)
    return r.returncode == 0 and out.is_file() and out.stat().st_size > 0


_cascades: dict = {}
_face_net = None
_nets_lock = threading.Lock()


def _cascade(name: str):
    import cv2

    if name not in _cascades:
        _cascades[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _cascades[name]


def _face_logit(img, face) -> Optional[float]:
    """MediaPipe BlazeFace's confidence that face ([x, y, w, h]) in img (a
    BGR array) really is a human face -- the Haar cascade alone also fires
    on game characters, emotes and posters. None when the model can't run
    (then the Haar face is taken as it is)."""
    global _face_net
    import cv2

    if not FACE_MODEL.is_file():
        return None
    H, W = img.shape[:2]
    x, y, w, h = face
    cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 0.9
    crop = img[int(max(0, cy - half)):int(min(H, cy + half)), int(max(0, cx - half)):int(min(W, cx + half))]
    if crop.size == 0:
        return None
    try:
        with _nets_lock:
            if _face_net is None:
                _face_net = cv2.dnn.readNetFromTFLite(str(FACE_MODEL))
            names = _face_net.getUnconnectedOutLayersNames()
            _face_net.setInput(cv2.dnn.blobFromImage(crop, 1 / 127.5, (128, 128), (127.5,) * 3, swapRB=True))
            outs = dict(zip(names, _face_net.forward(names)))
        return float(outs["classificators"].max())
    except Exception as e:
        print(f"[thumbnail] face check unavailable: {e}", flush=True)
        return None


def _score(path: Path) -> dict:
    """The best face in a frame ([x, y, w, h] in the frame's own pixels)
    and a score for it: size, sharpness of the face itself, eyes found,
    lighting. No face: score 0 and face None."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return {"score": -1, "face": None, "sharp": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    k = min(1.0, 960 / W)
    small = cv2.resize(gray, (int(W * k), int(H * k))) if k < 1 else gray
    faces = _cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        small, scaleFactor=1.08, minNeighbors=7, minSize=(32, 32))
    frame_sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    best, best_s = None, 0.0
    for fx, fy, fw, fh in faces:
        x, y, w, h = (int(round(v / k)) for v in (fx, fy, fw, fh))
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        logit = _face_logit(img, [x, y, w, h])
        if logit is not None and logit < FACE_MIN_LOGIT:
            continue
        eyes = _cascade("haarcascade_eye.xml").detectMultiScale(
            roi[: int(h * 0.6)], scaleFactor=1.1, minNeighbors=6, minSize=(max(8, w // 10),) * 2)
        sharp = float(cv2.Laplacian(cv2.resize(roi, (160, 160)), cv2.CV_64F).var())
        bright = float(roi.mean())
        s = (0.45 * min(1.0, h / GOOD_FACE_PX) + 0.30 * min(1.0, sharp / 250.0)
             + 0.15 * (1.0 if len(eyes) else 0.0) + 0.10 * (1.0 if 60 <= bright <= 215 else 0.3))
        if s > best_s:
            best, best_s = [x, y, w, h], s
    return {"score": round(best_s, 4), "face": best, "sharp": frame_sharp}


def candidate_frames(project_dir: Path, library: List[dict], out_dir: Path, max_clips: int = 10, per_clip: int = 8) -> List[dict]:
    """The best frame of each of the streamer's most-viewed clips, best
    first: {"clip", "t", "file", "face", "date", "score"}. Frames with a
    face come first; a frame without one only makes the list when no clip
    has a face at all."""
    out_dir.mkdir(parents=True, exist_ok=True)
    top = sorted((c for c in library if (project_dir / c["file"]).is_file()),
                 key=lambda c: int(c.get("views") or 0), reverse=True)[:max_clips]
    with_face: List[dict] = []
    without: List[dict] = []
    for c in top:
        dur = float(c.get("duration") or 0) or 10.0
        found = blank = None
        for k in range(per_clip):
            t = dur * (0.05 + 0.9 * (k + 0.5) / per_clip)
            f = out_dir / f"{c['id']}_{k}.jpg"
            if not f.is_file() and not _grab(project_dir / c["file"], t, f):
                continue
            s = _score(f)
            entry = {"clip": c["id"], "t": round(t, 2), "file": f.name, "face": s["face"],
                     "date": c.get("date") or "", "score": s["score"]}
            if s["face"] and (found is None or s["score"] > found["score"]):
                found = entry
            if blank is None or s["sharp"] > blank["sharp"]:
                blank = {**entry, "face": None, "score": 0.0, "sharp": s["sharp"]}
        if found:
            with_face.append(found)
        elif blank:
            blank.pop("sharp", None)
            without.append(blank)
    with_face.sort(key=lambda f: f["score"], reverse=True)
    return with_face or without


# ------------------------------------------------------------------ hooks ---

MAX_HOOK_CHARS = 22


def hook_options(project: dict, dossier: dict, library: List[dict]) -> List[str]:
    """Three 1-3 word hook lines for the thumbnail."""
    from .documentary import _numbers, _value_ok, research_text
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    name = (project.get("streamer") or {}).get("display_name") or project.get("login") or "this streamer"
    story = " ".join(sc.get("narration", "") for sc in project.get("scenes") or [] if sc.get("kind") == "narrate")
    prompt = f"""You write the big text on a YouTube thumbnail for a documentary about the
Twitch streamer {name}. The video's title: "{project.get('title') or f'The Story of {name}'}".
The thumbnail is the streamer's face, large; the text sits next to it.

The story (what the narrator says):
{story[:3500]}

Write 3 different options, each 1 to 3 words and at most {MAX_HOOK_CHARS} characters.
- It has to make sense to someone who has never watched {name}: no
  in-jokes, nicknames, catchphrases, quotes from clips or game slang.
- It adds to the title rather than repeating it, and leaves out the name.
- A turning point, a contrast or a question the video answers.
- Only facts from the story; a number only if the story has it.
- Plain words, no hype words, no emojis, no hashtags.

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
        if not h or len(h.split()) > 4 or len(h) > MAX_HOOK_CHARS:
            continue
        if any(ch.isdigit() for ch in h) and not _value_ok(h.lower(), blob, nums) \
                and not all(_value_ok(d, blob, nums) for d in re.findall(r"\d[\d,.]*[kmb]?", h.lower())):
            continue
        if h not in out:
            out.append(h)
    for fallback in ("HOW IT STARTED", "THE FULL STORY", "WHAT HAPPENED?"):
        if len(out) >= 3:
            break
        if fallback not in out:
            out.append(fallback)
    return out[:3]


# ---------------------------------------------------------------- cut-out ---

_seg_net = None
_seg_lock = threading.Lock()


def _person_mask(img, face_box):
    """A soft mask (Pillow "L", img's size) of the person whose face is
    face_box ([x, y, w, h] in img's pixels), or None when the model is
    missing or the result doesn't look like a person."""
    global _seg_net
    import cv2
    import numpy as np
    from PIL import Image

    if not SEG_MODEL.is_file():
        return None
    W, H = img.size
    try:
        with _seg_lock:
            if _seg_net is None:
                _seg_net = cv2.dnn.readNetFromTFLite(str(SEG_MODEL))
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            _seg_net.setInput(cv2.dnn.blobFromImage(bgr, 1 / 255.0, (256, 144), swapRB=True))
            prob = _seg_net.forward().squeeze().astype(np.float32)
    except Exception as e:  # an OpenCV without the TFLite importer, a bad model file
        print(f"[thumbnail] cut-out unavailable: {e}", flush=True)
        return None
    prob = np.clip(cv2.resize(prob, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
    # Keep only the blob the face belongs to: the model also lights up
    # other people, posters and furniture.
    n, labels = cv2.connectedComponents((prob > 0.5).astype(np.uint8))
    x, y, w, h = face_box
    box = labels[max(0, y):y + h, max(0, x):x + w]
    ids, counts = np.unique(box[box > 0], return_counts=True)
    if not len(ids):
        return None
    keep = labels == ids[int(np.argmax(counts))]
    coverage = keep.mean()
    if not 0.04 <= coverage <= 0.85:
        return None
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8))
    soft = np.clip((prob - 0.3) / 0.4, 0, 1) * keep
    soft = cv2.GaussianBlur(soft, (0, 0), max(1.0, W / 640))
    return Image.fromarray((soft * 255).astype(np.uint8), "L")


def _pop(img, mask):
    """The person sharp and bright with a white outline, over a blurred,
    darkened copy of the frame."""
    from PIL import Image, ImageEnhance, ImageFilter

    W = img.width
    bg = img.filter(ImageFilter.GaussianBlur(max(8, W // 70)))
    bg = ImageEnhance.Brightness(ImageEnhance.Color(bg).enhance(0.7)).enhance(0.42)
    subject = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.3)).enhance(1.1)).enhance(1.4)
    edge = max(5, W // 150)
    stroke = mask.filter(ImageFilter.MaxFilter(edge * 2 + 1)).filter(ImageFilter.GaussianBlur(1.2))
    glow = mask.filter(ImageFilter.MaxFilter(edge * 4 + 1)).filter(ImageFilter.GaussianBlur(edge * 3))
    out = Image.composite(Image.new("RGB", img.size, (255, 255, 255)), bg, glow.point(lambda v: int(v * 0.35)))
    out = Image.composite(Image.new("RGB", img.size, (255, 255, 255)), out, stroke)
    return Image.composite(subject, out, mask)


def _vignette(img, strength: float = 0.6):
    from PIL import Image, ImageDraw, ImageFilter

    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse((-W * 0.1, -H * 0.2, W * 1.1, H * 1.2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(W // 10))
    return Image.composite(img, Image.blend(img, Image.new("RGB", (W, H)), strength), mask)


# ---------------------------------------------------------------- drawing ---

def _crop(src, face, out_w: int, out_h: int, face_frac: float, tx: float, ty: float):
    """The part of src where the face is face_frac of the height, centred at
    (tx, ty) of the picture, resized to out_w x out_h. Also returns the
    face box in the new picture's pixels."""
    from PIL import Image

    W, H = src.size
    aspect = out_w / out_h
    if face:
        x, y, w, h = face
        ch = min(H, max(h / face_frac, out_h / MAX_ZOOM))
        cw = ch * aspect
        if cw > W:
            cw, ch = W, W / aspect
        fx, fy = x + w / 2, y + h / 2
        x0 = min(max(0.0, fx - tx * cw), W - cw)
        y0 = min(max(0.0, fy - ty * ch), H - ch)
    else:
        cw, ch = (W, W / aspect) if W / H < aspect else (H * aspect, H)
        x0, y0 = (W - cw) / 2, (H - ch) / 2
    pic = src.resize((out_w, out_h), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
    k = out_h / ch
    box = [int((x - x0) * k), int((y - y0) * k), int(w * k), int(h * k)] if face else None
    return pic, box


def _subject(src, face, out_w: int, out_h: int, face_frac: float, tx: float, ty: float):
    pic, box = _crop(src, face, out_w, out_h, face_frac, tx, ty)
    mask = _person_mask(pic, box) if box else None
    return _pop(pic, mask) if mask is not None else _vignette(pic)


def _hook_lines(hook: str, max_w: int, start: int = 170, min_size: int = 80, max_rows: int = 2):
    size = start
    while True:
        fnt = F("Anton", size)
        rows = wrap(fnt, hook, max_w)
        if (len(rows) <= max_rows and all(text_w(fnt, r) <= max_w for r in rows)) or size <= min_size:
            return fnt, rows[:3]
        size -= 6


def _draw_hook(img, hook: str, x: int, y: int, max_w: int, align: str = "left", start: int = 170, max_rows: int = 2) -> int:
    """The hook in big letters, white with the last line yellow, a thick
    black edge and a soft shadow so it reads on any picture."""
    from PIL import Image, ImageDraw, ImageFilter

    fnt, rows = _hook_lines(hook, max_w, start, max_rows=max_rows)
    line_h = int(fnt.size * 1.06)
    shadow = Image.new("L", img.size, 0)
    sd = ImageDraw.Draw(shadow)
    spots = []
    for i, row in enumerate(rows):
        w = text_w(fnt, row)
        rx = x if align == "left" else x + (max_w - w) / 2
        spots.append((rx, y + i * line_h, row, YEL if i == len(rows) - 1 and len(rows) > 1 else WHITE))
        sd.text((rx + 6, y + i * line_h + 8), row, font=fnt, fill=200, stroke_width=12, stroke_fill=200)
    img.paste((0, 0, 0), (0, 0), shadow.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(img)
    for rx, ry, row, color in spots:
        d.text((rx, ry), row, font=fnt, fill=color, stroke_width=10, stroke_fill=(0, 0, 0))
    return y + len(rows) * line_h


def _open(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def draw(layout: str, frames: List[dict], frames_dir: Path, hook: str, out: Path) -> Path:
    from PIL import Image, ImageDraw, ImageEnhance

    main = frames[0]
    if layout == "face":
        img = _subject(_open(frames_dir / main["file"]), main.get("face"), TW, TH, 0.52, 0.70, 0.42)
        fnt, rows = _hook_lines(hook, 600)
        y = (TH - len(rows) * int(fnt.size * 1.06)) // 2
        _draw_hook(img, hook, 48, y, 600)
    elif layout == "full":
        img = _subject(_open(frames_dir / main["file"]), main.get("face"), TW, TH, 0.46, 0.5, 0.34)
        fnt, rows = _hook_lines(hook, 1180, 150, max_rows=1)
        _draw_hook(img, hook, 50, TH - 36 - int(fnt.size * 1.06), 1180, "center", 150, max_rows=1)
    else:  # split
        old = min(frames, key=lambda f: f.get("date") or "9999")
        new = max(frames, key=lambda f: f.get("date") or "")
        half = TW // 2
        left = _subject(_open(frames_dir / old["file"]), old.get("face"), half, TH, 0.44, 0.5, 0.42)
        right = _subject(_open(frames_dir / new["file"]), new.get("face"), half, TH, 0.44, 0.5, 0.42)
        left = ImageEnhance.Color(left).enhance(0.55)
        img = Image.new("RGB", (TW, TH))
        img.paste(left, (0, 0))
        img.paste(right, (half, 0))
        d = ImageDraw.Draw(img)
        d.rectangle((half - 4, 0, half + 4, TH), fill=WHITE)
        yf = F("Anton", 84)
        for f, x in ((old, 34), (new, half + 34)):
            yr = (f.get("date") or "")[:4]
            if yr:
                d.text((x, 18), yr, font=yf, fill=WHITE, stroke_width=8, stroke_fill=(0, 0, 0))
        fnt, rows = _hook_lines(hook, 1180, 130, max_rows=1)
        _draw_hook(img, hook, 50, TH - 32 - int(fnt.size * 1.06), 1180, "center", 130, max_rows=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=90, optimize=True)
    return out


# ------------------------------------------------------------------ build ---

def _then_and_now(frames: List[dict]) -> Optional[List[dict]]:
    """The oldest and newest face frames, from different years, or None."""
    dated = [f for f in frames if f.get("face") and (f.get("date") or "")[:4]]
    if not dated:
        return None
    old = min(dated, key=lambda f: f["date"])
    new = max(dated, key=lambda f: f["date"])
    return [old, new] if old["date"][:4] != new["date"][:4] else None


def make_thumbnails(project_dir: Path, project: dict, dossier: dict, library: List[dict]) -> dict:
    """Three thumbnails, one per layout, each with its own hook and (where
    the clips allow) its own frame. Returns the project's "thumbnails"
    record."""
    tdir = project_dir / "thumbs"
    frames = candidate_frames(project_dir, library, tdir / "frames")
    if not frames:
        raise RuntimeError("couldn't grab any frames from the clips")
    hooks = hook_options(project, dossier, library)
    items = []
    stamp = uuid.uuid4().hex[:8]
    pair = _then_and_now(frames)
    plan = [("face", frames[:1]), ("full", frames[1:2] or frames[:1]),
            ("split", pair) if pair else ("face", frames[2:3] or frames[-1:])]
    for i, (layout, use) in enumerate(plan):
        name = f"thumb_{i}_{stamp}.jpg"
        draw(layout, use, tdir / "frames", hooks[i % len(hooks)], tdir / name)
        items.append({"layout": layout, "hook": hooks[i % len(hooks)], "file": name, "frames": [f["file"] for f in use]})
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
