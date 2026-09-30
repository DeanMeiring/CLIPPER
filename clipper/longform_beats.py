"""Keyword visuals for the long-form documentaries: a narrated scene is cut
into short visual beats that change on the words the creator says, instead
of one clip looping under the whole scene.

Each narrated scene carries "cues" (planned by documentary.plan_visuals,
editable in the story editor). A cue names a phrase from the narration and
what appears when that phrase is spoken:

  words     the phrase (or a short line) in big letters over the footage
  emoji     a 3D emoji popping in (bundled Fluent Emoji, MIT)
  stat      a stat card with the streamer's avatar; the number counts up
  stock     free stock footage (Pexels / Pixabay)
  photo     a free photo with a slow zoom (Pexels / Pixabay / Openverse CC0)
  clip      a quick cut to another of the streamer's clips, with its date
  post      a post the creator pasted (X), with a highlighter sweep
  headline  an article the creator pasted, headline highlighted
  timeline  years and milestones lighting up one by one

Between cues the streamer's clips play with slow zooms, cutting to a new
clip every few seconds, and word-by-word captions run along the bottom.
When the phrase is spoken comes from Whisper word timings of the take,
aligned to the script, so the captions use the script's spelling and a
visual lands on its word.

Frames are drawn with Pillow and piped straight into ffmpeg.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .visual_sources import FONT_DIR, emoji_file

W, H, FPS = 1920, 1080, 30
CUE_TYPES = ("words", "emoji", "stat", "stock", "photo", "clip", "post", "headline", "timeline")
# How long each kind of visual stays up at most before the footage returns.
MAX_HOLD = {"words": 2.4, "emoji": 2.6, "stat": 4.5, "stock": 4.5, "photo": 4.0, "clip": 4.5,
            "post": 5.5, "headline": 5.0, "timeline": 6.0}
MIN_GAP = 1.3          # two cues closer than this: the later one is dropped
BASE_CUT = 4.5         # between cues, cut to another clip at least this often
LEAD = 0.12            # a visual appears this much before its word

YEL = (250, 204, 21)
WHITE = (255, 255, 255)
INK = (15, 17, 23)
MUTED = (148, 155, 170)
RED = (239, 68, 68)
ACC = (168, 85, 247)
PINK = (236, 72, 153)

_fonts: dict = {}


def F(name: str, size: int):
    from PIL import ImageFont

    key = (name, size)
    if key not in _fonts:
        path = FONT_DIR / f"{name}.ttf"
        _fonts[key] = ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)
    return _fonts[key]


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def ease_out(x: float) -> float:
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp(x)
    return 3 * x * x - 2 * x * x * x


def back(x: float, s: float = 1.7) -> float:
    x = clamp(x) - 1
    return x * x * ((s + 1) * x + s) + 1


def prog(t: float, start: float, length: float) -> float:
    return clamp((t - start) / length)


# ---------------------------------------------------------------- timing ---

def _norm(tok: str) -> str:
    from .longform import _norm as n

    return n(tok)


def script_timings(script: str, words: List[dict], duration: float) -> List[Tuple[str, float, float]]:
    """(word, start, end) for every word of the script as written, timed
    from Whisper's words of the take. Words Whisper missed or misheard get
    times interpolated from their neighbours."""
    disp = script.split()
    if not disp:
        return []
    if not words:
        step = max(0.2, duration / len(disp))
        return [(w, i * step, (i + 1) * step) for i, w in enumerate(disp)]
    sn = [_norm(w) for w in disp]
    wn = [_norm(w["w"]) for w in words]
    times: List[Optional[Tuple[float, float]]] = [None] * len(disp)
    for a, b, size in SequenceMatcher(None, sn, wn, autojunk=False).get_matching_blocks():
        for k in range(size):
            times[a + k] = (float(words[b + k]["s"]), float(words[b + k]["e"]))
    known = [i for i, t in enumerate(times) if t]
    if not known:
        first, last = float(words[0]["s"]), float(words[-1]["e"])
        step = max(0.2, (last - first) / len(disp))
        return [(w, first + i * step, first + (i + 1) * step) for i, w in enumerate(disp)]
    out = []
    for i, w in enumerate(disp):
        if times[i]:
            out.append((w, *times[i]))
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is None:
            t0 = max(0.0, times[nxt][0] - 0.3 * (nxt - i))
            out.append((w, t0, t0 + 0.25))
        elif nxt is None:
            t0 = times[prev][1] + 0.3 * (i - prev - 1)
            out.append((w, t0, t0 + 0.25))
        else:
            span = times[nxt][0] - times[prev][1]
            f0 = (i - prev) / (nxt - prev)
            t0 = times[prev][1] + span * f0
            out.append((w, t0, t0 + max(0.1, span / (nxt - prev))))
    return out


def find_phrase(phrase: str, timed: List[Tuple[str, float, float]]) -> Optional[float]:
    """When the first word of `phrase` is said (exact match first, then the
    closest similar run of words)."""
    target = [t for t in (_norm(w) for w in phrase.split()) if t]
    toks = [_norm(w) for w, _, _ in timed]
    if not target or not toks:
        return None
    n = len(target)
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == target:
            return timed[i][1]
    best, best_i = 0.0, None
    for i in range(len(toks) - n + 1):
        r = SequenceMatcher(None, toks[i:i + n], target).ratio()
        if r > best:
            best, best_i = r, i
    return timed[best_i][1] if best_i is not None and best >= 0.6 else None


def schedule(cues: List[dict], timed: List[Tuple[str, float, float]], duration: float) -> List[dict]:
    """Cues with start/end times, in order, not overlapping. A cue whose
    phrase isn't in the narration, or that lands right on top of another,
    is dropped."""
    placed = []
    for c in cues or []:
        if c.get("type") not in CUE_TYPES:
            continue
        t = find_phrase(str(c.get("at") or ""), timed)
        if t is None:
            continue
        placed.append({**c, "t0": max(0.0, t - LEAD)})
    placed.sort(key=lambda c: c["t0"])
    kept: List[dict] = []
    for c in placed:
        if kept and c["t0"] - kept[-1]["t0"] < MIN_GAP:
            continue
        if c["t0"] > duration - 0.8:
            continue
        kept.append(c)
    for i, c in enumerate(kept):
        nxt = kept[i + 1]["t0"] if i + 1 < len(kept) else duration
        c["t1"] = min(nxt, c["t0"] + MAX_HOLD[c["type"]], duration)
    return kept


def segments(cues: List[dict], duration: float) -> List[dict]:
    """The whole scene as back-to-back segments: base footage between cues
    (split so the picture changes at least every BASE_CUT seconds), and
    one segment per cue."""
    out, t = [], 0.0
    for c in cues + [None]:
        end = c["t0"] if c else duration
        gap = end - t
        if gap > 0.05:
            n = max(1, math.ceil(gap / BASE_CUT - 0.15))
            for k in range(n):
                out.append({"kind": "base", "t0": t + gap * k / n, "t1": t + gap * (k + 1) / n})
        if c:
            out.append({"kind": "cue", "cue": c, "t0": c["t0"], "t1": c["t1"]})
            t = c["t1"]
    # overlay cues (words, emoji) keep the footage underneath running
    return [s for s in out if s["t1"] - s["t0"] > 0.04]


# --------------------------------------------------------------- footage ---

class Footage:
    """Frames of a video, played from `start`, looped, scaled to fill w x h."""

    def __init__(self, ffmpeg: str, path: Path, start: float = 0.0, w: int = W, h: int = H):
        self.w, self.h = w, h
        self.p = subprocess.Popen(
            [ffmpeg, "-v", "error", "-stream_loop", "-1", "-ss", f"{max(0.0, start):.2f}", "-i", str(path),
             "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},setsar=1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last = None

    def next(self):
        from PIL import Image

        buf = self.p.stdout.read(self.w * self.h * 3)
        if len(buf) < self.w * self.h * 3:
            return self.last if self.last is not None else Image.new("RGB", (self.w, self.h), (8, 10, 16))
        self.last = Image.frombuffer("RGB", (self.w, self.h), buf, "raw", "RGB", 0, 1)
        return self.last

    def close(self):
        try:
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass


class Still:
    def __init__(self, path: Path):
        from PIL import Image

        img = Image.open(path).convert("RGB")
        scale = max(W * 1.25 / img.width, H * 1.25 / img.height)
        self.img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        self.w, self.h = self.img.size

    def next(self):
        return self.img

    def close(self):
        pass


def kenburns(img, z: float, fx: float = 0.5, fy: float = 0.5):
    from PIL import Image

    w, h = img.size
    cw, ch = w / z, h / z
    if cw / ch > W / H:
        cw = ch * W / H
    else:
        ch = cw * H / W
    x0, y0 = (w - cw) * fx, (h - ch) * fy
    return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))


def blurred(small, dark: float = 0.5):
    from PIL import Image, ImageFilter

    b = small.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    return Image.blend(b, Image.new("RGB", (W, H), (6, 8, 14)), dark)


def dim(img, amount: float):
    from PIL import Image

    return Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), amount) if amount > 0 else img


# ---------------------------------------------------------------- drawing ---

def text_w(fnt, s: str) -> float:
    return fnt.getlength(s)


def wrap(fnt, text: str, max_w: float) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if text_w(fnt, trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def shadow_card(base, box, r: int = 28, fill=(255, 255, 255), alpha: int = 255) -> None:
    from PIL import Image, ImageDraw, ImageFilter

    x0, y0, x1, y1 = [int(v) for v in box]
    sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(sh).rounded_rectangle((x0 + 8, y0 + 22, x1 + 8, y1 + 22), r, fill=int(130 * alpha / 255))
    base.paste((0, 0, 0), (0, 0), sh.filter(ImageFilter.GaussianBlur(22)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((x0, y0, x1, y1), r, fill=tuple(fill) + (alpha,))
    base.paste(layer, (0, 0), layer)


def highlighted(draw, xy, text: str, fnt, highlight: str, sweep: float, max_w: float, color=INK, marker=(253, 224, 71)) -> float:
    """Wrapped text; the words of `highlight` get a marker that sweeps
    across them as `sweep` goes 0 -> 1. Returns the y below the text."""
    x0, y0 = xy
    words = text.split()
    space = text_w(fnt, " ")
    line_h = int(fnt.size * 1.3)
    x, y, pos = x0, y0, []
    for w in words:
        ww = text_w(fnt, w)
        if x + ww > x0 + max_w and x > x0:
            x, y = x0, y + line_h
        pos.append((x, y, ww))
        x += ww + space
    hl = [_norm(w) for w in highlight.split() if _norm(w)]
    idx: List[int] = []
    toks = [_norm(w) for w in words]
    for i in range(len(toks) - len(hl) + 1):
        if hl and toks[i:i + len(hl)] == hl:
            idx = list(range(i, i + len(hl)))
            break
    total = sum(pos[i][2] + space for i in idx)
    done = total * sweep
    for i in idx:
        px, py, pw = pos[i]
        seg = min(pw + space, max(0.0, done))
        if seg > 0:
            draw.rectangle((px - 4, py + fnt.size * 0.1, px + seg - 2, py + fnt.size * 1.12), fill=marker)
        done -= pw + space
    for (px, py, _), w in zip(pos, words):
        draw.text((px, py), w, font=fnt, fill=color)
    return y + line_h


def circle_image(src: Optional[Path], size: int, initials: str):
    from PIL import Image, ImageDraw

    img = None
    if src and src.is_file():
        try:
            img = Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS)
        except Exception:
            img = None
    if img is None:
        img = Image.new("RGB", (size, size), ACC)
        d = ImageDraw.Draw(img)
        for x in range(size):
            f = x / size
            d.line((x, 0, x, size), fill=tuple(int(a * (1 - f) + b * f) for a, b in zip(ACC, PINK)))
        fnt = F("Inter-Black", int(size * 0.38))
        d.text(((size - text_w(fnt, initials)) / 2, size * 0.26), initials, font=fnt, fill=WHITE)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    return img, mask


def watermark(img, brand: str) -> None:
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    s = brand.upper()
    fnt = F("Inter-Black", 26)
    d.text((W - 56 - text_w(fnt, s), 44), s, font=fnt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))


def lower_third(img, label: str, t: float) -> None:
    from PIL import ImageDraw

    if not label:
        return
    d = ImageDraw.Draw(img)
    fnt = F("Inter-Bold", 38)
    w = text_w(fnt, label) + 72
    k = ease_out(prog(t, 0.15, 0.4))
    x = -w - 20 + (w + 68) * k
    d.rounded_rectangle((x, H - 300, x + w, H - 222), 14, fill=(0, 0, 0))
    d.rectangle((x, H - 300, x + 12, H - 222), fill=YEL)
    d.text((x + 36, H - 286), label, font=fnt, fill=WHITE)


class Captions:
    """Word-by-word captions: the line being said, current word in yellow."""

    def __init__(self, timed: List[Tuple[str, float, float]]):
        self.lines: List[List[Tuple[str, float, float]]] = []
        cur: List[Tuple[str, float, float]] = []
        for w in timed:
            if cur and (len(cur) >= 6 or len(" ".join(x[0] for x in cur + [w])) > 34 or w[1] - cur[-1][2] > 0.7):
                self.lines.append(cur)
                cur = []
            cur.append(w)
            if re.search(r"[.!?]$", w[0]) and len(cur) >= 2:
                self.lines.append(cur)
                cur = []
        if cur:
            self.lines.append(cur)

    def draw(self, img, t: float) -> None:
        from PIL import ImageDraw

        line = None
        for i, ln in enumerate(self.lines):
            end = self.lines[i + 1][0][1] if i + 1 < len(self.lines) else ln[-1][2] + 0.4
            if ln[0][1] - 0.05 <= t < min(end, ln[-1][2] + 0.6):
                line = ln
                break
        if not line:
            return
        d = ImageDraw.Draw(img)
        fnt = F("Inter-Black", 56)
        x = (W - text_w(fnt, " ".join(w for w, _, _ in line))) / 2
        for w, a, b in line:
            col = YEL if a <= t < max(b, a + 0.15) else WHITE
            d.text((x, H - 158), w, font=fnt, fill=col, stroke_width=5, stroke_fill=(0, 0, 0))
            x += text_w(fnt, w + " ")


# ------------------------------------------------------------------ beats ---

def _count_up(value: str, p: float) -> str:
    """'1,240,000' counts up; anything else ('1.2 million', '2020') shows as is."""
    if re.fullmatch(r"\d{1,3}(,\d{3})+|\d{5,}", value.strip()):
        n = int(value.replace(",", ""))
        return f"{int(n * ease_out(p)):,}"
    return value


class Ctx:
    """Everything a beat needs to draw itself."""

    def __init__(self, ffmpeg: str, project_dir: Path, library: dict, profile: dict, posts: List[dict],
                 articles: List[dict], brand: str):
        self.ffmpeg, self.project_dir, self.library = ffmpeg, project_dir, library
        self.profile, self.posts, self.articles, self.brand = profile, posts, articles, brand
        self.avatar = project_dir / "visuals" / "avatar.jpg"

    def clip_path(self, cid: str) -> Optional[Path]:
        c = self.library.get(cid or "")
        p = self.project_dir / c["file"] if c else None
        return p if p and p.is_file() else None

    def clip_label(self, cid: str) -> str:
        c = self.library.get(cid or "") or {}
        if not c:
            return ""
        from datetime import date

        try:
            d = date.fromisoformat(c["date"]).strftime("%b %Y")
        except (KeyError, ValueError):
            d = ""
        views = int(c.get("views") or 0)
        v = f"{views / 1e6:.1f}M views" if views >= 1e6 else f"{views:,} views" if views else ""
        return " · ".join(x for x in (d, v) if x)


def _base(ctx: Ctx, src, dur: float, idx: int, label: str = "") -> Iterator:
    fx, fy = [(0.5, 0.45), (0.35, 0.4), (0.65, 0.5), (0.5, 0.35)][idx % 4]
    zoom_in = idx % 2 == 0
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        k = ease_in_out(t / max(dur, 0.01))
        z = 1.03 + 0.09 * (k if zoom_in else 1 - k)
        f = kenburns(src.next(), z, fx, fy)
        if label:
            lower_third(f, label, t)
        yield f


def _words(ctx: Ctx, src, dur: float, cue: dict) -> Iterator:
    from PIL import ImageDraw

    text = (cue.get("text") or cue.get("at") or "").upper()[:40]
    fnt_full = F("Anton", 200)
    rows = wrap(fnt_full, text, W - 260)[:3]
    size = 200 if len(rows) <= 2 else 150
    fnt_full = F("Anton", size)
    rows = wrap(fnt_full, text, W - 260)[:3]
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = dim(kenburns(src.next(), 1.05 + 0.08 * (t / dur)), 0.35)
        d = ImageDraw.Draw(f)
        y = H / 2 - len(rows) * size * 0.62
        for r, row in enumerate(rows):
            k = back(prog(t, 0.08 * r, 0.3))
            fnt = F("Anton", max(8, int(size * (0.6 + 0.4 * k))))
            x = (W - text_w(fnt, row)) / 2
            d.text((x + 6, y + 8), row, font=fnt, fill=(0, 0, 0))
            d.text((x, y), row, font=fnt, fill=YEL if r == len(rows) - 1 else WHITE)
            y += size * 1.12
        if t < 0.1:
            from PIL import Image
            f = Image.blend(f, Image.new("RGB", (W, H), WHITE), 0.5 * (1 - t / 0.1))
        yield f


def _emoji(ctx: Ctx, src, dur: float, cue: dict) -> Iterator:
    from PIL import Image, ImageDraw

    ef = emoji_file(cue.get("emoji") or "")
    em = Image.open(ef).convert("RGBA") if ef else None
    label = (cue.get("text") or "").strip()[:28]
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = dim(kenburns(src.next(), 1.04 + 0.06 * (t / dur)), 0.25)
        if em is not None:
            k = back(prog(t, 0, 0.35), 2.2)
            wob = math.sin(t * 5) * 4 * clamp(t / 0.4)
            size = max(8, int(360 * k))
            e = em.resize((size, size), Image.LANCZOS).rotate(wob, resample=Image.BICUBIC, expand=True)
            cx, cy = W / 2, H / 2 - (60 if label else 20) + math.sin(t * 3) * 10
            f.paste(e, (int(cx - e.width / 2), int(cy - e.height / 2)), e)
        if label:
            d = ImageDraw.Draw(f)
            fnt = F("Anton", 110)
            k = ease_out(prog(t, 0.2, 0.3))
            x = (W - text_w(fnt, label.upper())) / 2
            d.text((x, H / 2 + 150 + (1 - k) * 40), label.upper(), font=fnt, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
        yield f


def _stat(ctx: Ctx, bg, dur: float, cue: dict) -> Iterator:
    from PIL import ImageDraw

    name = ctx.profile.get("display_name") or ctx.profile.get("login") or ""
    av, am = circle_image(ctx.avatar, 220, (name[:2] or "?").upper())
    label = str(cue.get("label") or "").upper()[:30]
    value = str(cue.get("value") or "")[:24]
    sub = str(cue.get("sub") or "")[:50]
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = blurred(bg.next(), 0.5)
        k = ease_out(prog(t, 0, 0.45))
        x0, y0 = 300, 280 + (1 - k) * 70
        shadow_card(f, (x0, y0, W - 300, y0 + 460), 36, fill=(18, 20, 28), alpha=int(240 * k))
        if k > 0.3:
            f.paste(av, (x0 + 80, int(y0 + 80)), am)
            d = ImageDraw.Draw(f)
            nf = F("Inter-Black", 50)
            while text_w(nf, name) > 300 and nf.size > 26:
                nf = F("Inter-Black", nf.size - 4)
            d.text((x0 + 80, y0 + 330), name, font=nf, fill=WHITE)
            p = prog(t, 0.35, 1.2)
            vf = F("Inter-Black", 118 if len(value) <= 11 else 84)
            d.text((x0 + 420, y0 + 100), label, font=F("Inter-Bold", 34), fill=MUTED)
            d.text((x0 + 420 + (1 - ease_out(p)) * 30, y0 + 160), _count_up(value, p), font=vf, fill=YEL)
            if sub:
                d.text((x0 + 420, y0 + 300), sub, font=F("Inter-Bold", 32), fill=(210, 214, 222))
        yield f


def _post(ctx: Ctx, bg, dur: float, cue: dict) -> Iterator:
    from PIL import ImageDraw

    try:
        post = ctx.posts[int(cue.get("post", -1))]
    except (ValueError, IndexError, TypeError):
        post = None
    if not post:
        yield from _base(ctx, bg, dur, 0)
        return
    av_path = ctx.project_dir / "visuals" / f"post_{int(cue['post'])}_avatar.jpg"
    av, am = circle_image(av_path, 96, (post.get("name") or "?")[:2].upper())
    fnt = F("Inter-Regular", 50 if len(post["text"]) < 160 else 40)
    rows = wrap(fnt, post["text"], 920)[:6]
    card_h = 250 + len(rows) * int(fnt.size * 1.3) + 90
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = blurred(bg.next(), 0.55)
        k = back(prog(t, 0, 0.5), 1.2)
        x0 = (W - 1060) / 2
        y0 = (H - card_h) / 2 - 40 + (1 - k) * 700
        shadow_card(f, (x0, y0, x0 + 1060, y0 + card_h), 30)
        d = ImageDraw.Draw(f)
        f.paste(av, (int(x0 + 50), int(y0 + 50)), am)
        d.text((x0 + 170, y0 + 56), post.get("name") or "", font=F("Inter-Black", 38), fill=INK)
        meta = f"@{post.get('handle')}" + (f" · {post['date']}" if post.get("date") else "")
        d.text((x0 + 170, y0 + 102), meta, font=F("Inter-Regular", 30), fill=(100, 106, 120))
        yb = highlighted(d, (x0 + 50, y0 + 185), "\n".join(rows).replace("\n", " "), fnt, str(cue.get("highlight") or ""),
                         ease_in_out(prog(t, 1.0, 1.1)), 920)
        if post.get("likes"):
            d.text((x0 + 50, yb + 20), f"{post['likes']:,} likes", font=F("Inter-Bold", 30), fill=(100, 106, 120))
        yield f


def _headline(ctx: Ctx, bg, dur: float, cue: dict) -> Iterator:
    from PIL import Image, ImageDraw

    try:
        art = ctx.articles[int(cue.get("article", -1))]
    except (ValueError, IndexError, TypeError):
        art = None
    if not art:
        yield from _base(ctx, bg, dur, 1)
        return
    site = (art.get("site") or re.sub(r"^www\.", "", re.sub(r"^https?://([^/]+).*", r"\1", art.get("url") or ""))).upper()[:40]
    date = art.get("published") or ""
    title = (art.get("title") or "")[:140]
    hf = F("SourceSerif-Bold", 74 if len(title) < 70 else 60)
    rows = wrap(hf, title, 1060)[:4]
    ch = 230 + len(rows) * int(hf.size * 1.3) + 60
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = blurred(bg.next(), 0.5)
        k = ease_out(prog(t, 0, 0.55))
        sc = 0.92 + 0.08 * k + 0.02 * (t / dur)
        card = Image.new("RGB", (1200, ch), (250, 249, 246))
        d = ImageDraw.Draw(card)
        d.rectangle((0, 0, 1200, 12), fill=RED)
        d.text((70, 58), site, font=F("Inter-Black", 30), fill=RED)
        if date:
            d.text((70, 100), date, font=F("Inter-Regular", 28), fill=(110, 110, 110))
        highlighted(d, (70, 170), " ".join(rows), hf, str(cue.get("highlight") or ""), ease_in_out(prog(t, 1.0, 1.0)), 1060)
        cw, chh = int(1200 * sc), int(ch * sc)
        card = card.resize((cw, chh), Image.BICUBIC)
        x, y = (W - cw) // 2, (H - chh) // 2 - 40
        shadow_card(f, (x, y, x + cw, y + chh), 8, alpha=int(255 * k))
        if k > 0.02:
            f.paste(card, (x, y), Image.new("L", (cw, chh), int(255 * k)))
        yield f


def _timeline(ctx: Ctx, bg, dur: float, cue: dict) -> Iterator:
    from PIL import ImageDraw

    pts = [(str(p[0])[:6], str(p[1])[:22]) for p in (cue.get("points") or []) if isinstance(p, (list, tuple)) and len(p) >= 2][:5]
    if len(pts) < 2:
        yield from _base(ctx, bg, dur, 2)
        return
    n = len(pts)
    span = W - 520
    xs = [260 + span * k / (n - 1) for k in range(n)]
    reveal = [0.2 + (dur - 1.2) * k / max(1, n - 1) * 0.8 for k in range(n)]
    for i in range(max(1, round(dur * FPS))):
        t = i / FPS
        f = blurred(bg.next(), 0.62)
        d = ImageDraw.Draw(f)
        y = 540
        grow = ease_in_out(prog(t, 0, reveal[-1] + 0.3))
        d.line((xs[0], y, xs[0] + (xs[-1] - xs[0]) * grow, y), fill=WHITE, width=8)
        shown = [k for k in range(n) if t >= reveal[k]]
        for k in shown:
            p = back(prog(t, reveal[k], 0.4))
            active = k == shown[-1]
            col = YEL if active else WHITE
            r = 22 * p
            d.ellipse((xs[k] - r, y - r, xs[k] + r, y + r), fill=col)
            fy = F("Anton", max(8, int(96 * clamp(p, 0, 1.2))))
            d.text((xs[k] - text_w(fy, pts[k][0]) / 2, y - 190), pts[k][0], font=fy, fill=col)
            fl = F("Inter-Bold", 34)
            for j, row in enumerate(wrap(fl, pts[k][1], 380)[:2]):
                d.text((xs[k] - text_w(fl, row) / 2, y + 50 + j * 42), row, font=fl, fill=col if active else (210, 214, 222))
        yield f


# ----------------------------------------------------------------- render ---

def _base_clips(ctx: Ctx, scene: dict) -> List[Tuple[str, float]]:
    """Clips to cut between under the narration: the scene's own clip
    first, then the others closest to it in date."""
    lib = [c for c in ctx.library.values() if ctx.clip_path(c["id"])]
    if not lib:
        return []
    own = ctx.library.get(scene.get("clip") or "")
    ref = (own or {}).get("date") or ""
    others = sorted((c for c in lib if not own or c["id"] != own["id"]), key=lambda c: abs(_days(c.get("date"), ref)))
    out = []
    if own and ctx.clip_path(own["id"]):
        out.append((own["id"], float(scene.get("start") or 0)))
    out += [(c["id"], min(5.0, float(c.get("duration") or 10) * 0.2)) for c in others[:6]]
    return out


def _days(a: Optional[str], b: str) -> int:
    from datetime import date

    try:
        return (date.fromisoformat(a or "") - date.fromisoformat(b)).days
    except ValueError:
        return 99999


def render_narrate(ctx: Ctx, scene: dict, duration: float, take: Path, take_words: List[dict], out: Path) -> None:
    """One narrated scene, keyword visuals and all, with the take as sound."""
    from PIL import Image

    timed = script_timings(scene.get("narration") or "", take_words, duration)
    cues = schedule(scene.get("cues") or [], timed, duration)
    segs = segments(cues, duration)
    caps = Captions(timed)
    bases = _base_clips(ctx, scene) or []
    used = {}

    def base_src(k: int, small: bool = False):
        if not bases:
            return None
        cid, st = bases[k % len(bases)]
        length = float((ctx.library.get(cid) or {}).get("duration") or 0)
        pos = used.get(cid, st)
        if length > 2 and pos > length - 1.5:
            pos = st if st < length - 1.5 else 0.0
        path = ctx.clip_path(cid)
        return cid, pos, Footage(ctx.ffmpeg, path, pos, 480 if small else W, 270 if small else H)

    enc = subprocess.Popen(
        [ctx.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(take), "-filter_complex", "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[a]",
         "-map", "0:v", "-map", "[a]", "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_total = max(1, round(duration * FPS))
    written = 0
    frame = Image.new("RGB", (W, H), (0, 0, 0))
    base_i = 0
    try:
        for seg in segs:
            n = max(0, round(seg["t1"] * FPS) - written)
            if n == 0:
                continue
            dur = n / FPS
            cue = seg.get("cue") or {}
            kind = cue.get("type") if seg["kind"] == "cue" else "base"
            src = None
            label = ""
            if kind in ("stock", "photo") and cue.get("asset"):
                p = ctx.project_dir / "visuals" / cue["asset"]
                if p.is_file():
                    src = Footage(ctx.ffmpeg, p, 0.0) if p.suffix == ".mp4" else Still(p)
            elif kind == "clip" and ctx.clip_path(cue.get("clip")):
                src = Footage(ctx.ffmpeg, ctx.clip_path(cue["clip"]), float(cue.get("start") or 0))
                label = ctx.clip_label(cue["clip"])
            if src is not None:
                gen = _base(ctx, src, dur, base_i, label)
            else:
                small = kind in ("stat", "post", "headline", "timeline")
                got = base_src(base_i, small)
                if got:
                    cid, pos, src = got
                    used[cid] = pos + dur
                    if kind == "base" and base_i == 0 and scene.get("caption"):
                        label = scene["caption"]
                    elif kind == "base" and base_i > 0 and cid != (scene.get("clip") or ""):
                        label = ctx.clip_label(cid)
                else:
                    src = _Blank(small)
                base_i += 1
                gen = {"words": _words, "emoji": _emoji, "stat": _stat, "post": _post, "headline": _headline,
                       "timeline": _timeline}.get(kind)
                gen = gen(ctx, src, dur, cue) if gen else _base(ctx, src, dur, base_i, label)
            try:
                for k, frame in enumerate(gen):
                    if k >= n:
                        break
                    frame = frame.copy() if frame.mode == "RGB" else frame.convert("RGB")
                    watermark(frame, ctx.brand)
                    caps.draw(frame, (written + 0.5) / FPS)
                    if written < 8:
                        frame = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), frame, (written + 1) / 9)
                    enc.stdin.write(frame.tobytes())
                    written += 1
            finally:
                src.close()
        while written < frames_total:  # rounding: hold the last frame
            enc.stdin.write(frame.tobytes())
            written += 1
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    except BaseException:
        enc.kill()
        raise


class _Blank:
    def __init__(self, small: bool):
        from PIL import Image

        self.img = Image.new("RGB", (480, 270) if small else (W, H), (14, 16, 24))

    def next(self):
        return self.img

    def close(self):
        pass


def render_title(ffmpeg: str, title: str, kicker: str, duration: float, bg_path: Optional[Path], bg_start: float,
                 brand: str, out: Path) -> None:
    """Chapter card: the title sliding in over blurred footage from the
    chapter, a yellow underline drawing itself."""
    src = Footage(ffmpeg, bg_path, bg_start, 480, 270) if bg_path else _Blank(True)
    enc = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a", "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        n = max(1, round(duration * FPS))
        for i in range(n):
            f = title_frame(src.next(), i / FPS, title, kicker, brand)
            fade = min(1.0, (n - i) / 9, (i + 1) / 9)
            if fade < 1:
                from PIL import Image
                f = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), f, fade)
            enc.stdin.write(f.tobytes())
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    except BaseException:
        enc.kill()
        raise
    finally:
        src.close()


def title_frame(small_bg, t: float, title: str, kicker: str, brand: str):
    """One frame of a chapter card, `t` seconds in."""
    from PIL import ImageDraw

    title = title.upper()
    fnt = F("Anton", 170)
    rows = wrap(fnt, title, W - 400)
    if len(rows) > 2:
        fnt = F("Anton", 120)
        rows = wrap(fnt, title, W - 400)[:3]
    f = blurred(small_bg, 0.68)
    d = ImageDraw.Draw(f)
    k = ease_out(prog(t, 0, 0.6))
    y = H / 2 - len(rows) * fnt.size * 0.6
    if kicker:
        d.text((200 - (1 - k) * 100, y - 70), kicker.upper(), font=F("Inter-Black", 44), fill=YEL)
    for r, row in enumerate(rows):
        d.text((190 + (1 - ease_out(prog(t, 0.08 * r, 0.6))) * 160, y), row, font=fnt, fill=WHITE)
        y += fnt.size * 1.1
    d.rectangle((200, y + 20, 200 + 700 * ease_out(prog(t, 0.3, 0.8)), 32 + y), fill=YEL)
    watermark(f, brand)
    return f


def take_words(take: Path) -> List[dict]:
    """Whisper word timings of a take, cached next to it."""
    cache = take.with_suffix(".words.json")
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    from .longform import transcribe_words

    words = transcribe_words(take)
    cache.write_text(json.dumps(words), encoding="utf-8")
    return words
