"""Dean's take at the end of a clip: the clip plays exactly as rendered, then
its last frame holds while Pixel -- the channel's retro-TV mascot -- pops in
small in the corner and says one short line in Dean's cloned voice, with a
few words of it on screen.

Why: since 1 Oct 2026 the Shorts feed shows fewer re-posted clips "without
significant changes"; the creator's own commentary is what counts. The line
is Dean's opinion or a question to viewers, never a description of the clip
(YouTube named describing voice-overs as not enough). Claude suggests it; he
can change it on Home.

  draw_pixel()  the mascot, one frame: yellow TV, black screen with the
                channel's viewfinder corners, a white pixel face with a red
                mouth (the channel's colours, same as the shocked-face
                badge). The mouth opens with the voice's loudness.
  suggest()     Claude's line: `say` (spoken, <= 10 words) and `text` (on
                screen, <= 4 words).
  apply()       clip + tail as one file, in place; the clip as it was is
                kept next to it so the take can be removed or redone.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional

VW, VH = 1080, 1920
SIZE = 380                 # the mascot's box on the 1080x1920 frame
MARGIN_X, BOTTOM_Y = 40, 1650
LEAD, HOLD = 0.35, 0.6     # pop-in before the voice, hold after it
SAY_MAX_WORDS, TEXT_MAX_WORDS = 10, 4
VOICE_LUFS = -16.0

YEL, YEL_D = (255, 204, 0, 255), (224, 166, 0, 255)
INK = (17, 17, 17, 255)
SCREEN = (17, 17, 22, 255)
FACE = (255, 255, 255, 255)
RED = (227, 18, 27, 255)
FONT_DIR = Path(__file__).parent / "assets" / "fonts"


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


# ----------------------------------------------------------------- mascot ---

def draw_pixel(mouth: float = 0.0, blink: bool = False, look: float = 0.0, size: int = SIZE):
    """Pixel, one frame, as an RGBA image `size` square. Drawn at 3x and
    scaled down for smooth edges."""
    from PIL import Image, ImageDraw, ImageFilter

    S = 3
    W = size * S
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    out = max(2, int(W * 0.016))

    def c(v):
        return v * W / 100

    def rr(x0, y0, x1, y1, r, fill, w=out):
        d.rounded_rectangle([c(x0), c(y0), c(x1), c(y1)], radius=c(r), fill=fill, outline=INK, width=w)

    def ell(cx, cy, r, fill, w=out):
        d.ellipse([c(cx - r), c(cy - r), c(cx + r), c(cy + r)], fill=fill, outline=INK, width=w)

    def ln(pts, fill=INK, w=out):
        d.line([(c(x), c(y)) for x, y in pts], fill=fill, width=w, joint="curve")

    # antennas
    ln([(42, 22), (34, 8)], w=int(out * 0.9)); ell(34, 7, 3, RED, int(out * 0.7))
    ln([(58, 22), (68, 10)], w=int(out * 0.9)); ell(68, 9, 3, YEL, int(out * 0.7))
    # stand + feet
    rr(40, 80, 60, 88, 3, YEL_D)
    rr(30, 87, 70, 94, 4, (40, 40, 48, 255))
    # case
    rr(14, 20, 86, 82, 10, YEL)
    d.rounded_rectangle([c(18), c(70), c(82), c(79)], radius=c(6), fill=YEL_D)
    shine = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shine).ellipse([c(17), c(23), c(35), c(29)], fill=(255, 255, 255, 120))
    img.alpha_composite(shine.filter(ImageFilter.GaussianBlur(S * 2)))
    # screen
    rr(21, 27, 71, 70, 7, SCREEN)
    # the channel's viewfinder corners on the screen
    arm, t = 6, int(out * 0.7)
    for (x, y, sx, sy) in ((24, 30, 1, 1), (68, 30, -1, 1), (24, 67, 1, -1), (68, 67, -1, -1)):
        ln([(x, y + sy * arm), (x, y), (x + sx * arm, y)], fill=YEL, w=t)
    # knobs
    ell(78, 40, 3.5, RED, int(out * 0.7))
    ell(78, 52, 3.5, INK, int(out * 0.7))
    # pixel face
    px = 2.6

    def pix(x, y, col=FACE):
        d.rectangle([c(x), c(y), c(x + px * 0.9), c(y + px * 0.9)], fill=col)

    off = look * 2
    for ex in (33, 52):
        if blink:
            for i in range(3):
                pix(ex + off + i * px, 42)
        else:
            for i in range(3):
                for j in range(3):
                    pix(ex + off + i * px, 37 + j * px)
    rows = 1 + int(round(max(0.0, min(1.0, mouth)) * 3))
    for i in range(6):
        for j in range(rows):
            if rows > 1 and i in (0, 5) and j in (0, rows - 1):
                continue
            inside = rows > 1 and 0 < j < rows - 1
            pix(37 + i * px, 54 + j * px, RED if inside else FACE)
    for y in range(28, 70, 2):
        d.line([(c(22), c(y)), (c(70), c(y))], fill=(255, 255, 255, 14), width=S)
    return img.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------ line ---

_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍#\"“”]")


def _clean(s, max_words: int) -> str:
    words = _EMOJI.sub("", str(s or "")).split()
    return " ".join(words[:max_words]).strip()


def normalize(say, text) -> dict:
    say = _clean(say, SAY_MAX_WORDS)
    text = _clean(text, TEXT_MAX_WORDS) or _clean(say, TEXT_MAX_WORDS)
    return {"say": say, "text": text}


def suggest(title: str, reason: str, transcript: str, source_title: str = "") -> dict:
    """Claude's take for one clip: {"say", "text"}; empty strings if it
    can't (the clip then goes out without a take rather than with a stock
    line -- the same line on every clip is the kind of template YouTube
    doesn't count)."""
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    prompt = f"""A clip from a live stream is about to go out as a YouTube Short on a clip channel. When it ends, the
channel owner adds one short line of his own, said in his voice: his take on the moment, or a question to the
people watching. It has to be his opinion or reaction -- never a description of what just happened, the viewer
just saw that.

The clip:
- Stream: {source_title or "unknown"}
- Title: {title}
- Why it was picked: {reason}
- What was said in it:
{transcript or "(no speech)"}

Write:
- say: what he says, 4 to {SAY_MAX_WORDS} words, casual and plain, like he's talking to a friend. A question to the
  viewers works well ("Would you have kept streaming after that?").
- text: the heart of it in 1 to {TEXT_MAX_WORDS} words, shown on screen ("Would you stay?").
No emoji, no hashtags, no all caps.

Return only JSON: {{"say": "...", "text": "..."}}"""
    try:
        ans = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
        if isinstance(ans, list):
            ans = ans[0] if ans and isinstance(ans[0], dict) else {}
        if not isinstance(ans, dict):
            ans = {}
    except Exception as e:  # noqa: BLE001
        print(f"[take] Claude couldn't suggest a line: {e}", flush=True)
        ans = {}
    out = normalize(ans.get("say"), ans.get("text"))
    return out if len(out["say"].split()) >= 2 else {"say": "", "text": ""}


# ---------------------------------------------------------------- render ---

def probe(path: Path) -> dict:
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60)
    err = r.stderr
    fps = re.search(r"(\d+(?:\.\d+)?) fps", err)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return {
        "fps": float(fps.group(1)) if fps else 30.0,
        "duration": int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0.0,
        "audio": "Audio:" in err,
    }


def _voice_levels(wav: Path, fps: float) -> List[float]:
    """Mouth openness per video frame from the voice's loudness."""
    import numpy as np

    with wave.open(str(wav)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    hop = max(1, int(sr / fps))
    rms = np.array([np.sqrt((pcm[i:i + hop] ** 2).mean() + 1e-9) for i in range(0, len(pcm), hop)])
    if not len(rms):
        return []
    m = np.clip(rms / (np.percentile(rms, 90) + 1e-6) * 1.4, 0, 1) ** 0.7
    m[rms < rms.max() * 0.06] = 0
    return [float(x) for x in np.convolve(m, [0.25, 0.5, 0.25], mode="same")]


def _bubble(text: str):
    """The few words, in a small speech bubble pointing down-left at Pixel."""
    from PIL import Image, ImageDraw, ImageFont

    fnt = ImageFont.truetype(str(FONT_DIR / "Inter-Black.ttf"), 54)
    pad, max_w = 34, VW - (MARGIN_X + SIZE - 20) - 60
    words, rows, cur = text.split(), [], ""
    probe_d = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    for w in words:
        t = f"{cur} {w}".strip()
        if probe_d.textlength(t, font=fnt) > max_w - 2 * pad and cur:
            rows.append(cur)
            cur = w
        else:
            cur = t
    rows.append(cur)
    tw = max(probe_d.textlength(r, font=fnt) for r in rows)
    bw, bh = int(tw + 2 * pad), int(len(rows) * 66 + 2 * pad - 10)
    img = Image.new("RGBA", (bw + 40, bh + 50), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 0, 20 + bw, bh], 30, fill=(255, 255, 255, 255), outline=INK, width=6)
    d.polygon([(40, bh - 4), (6, bh + 44), (90, bh - 4)], fill=(255, 255, 255, 255))
    d.line([(40, bh - 2), (6, bh + 44), (90, bh - 2)], fill=INK, width=6, joint="curve")
    y = pad - 6
    for r in rows:
        d.text((20 + pad, y), r, font=fnt, fill=INK)
        y += 66
    return img


def render_tail(still: Path, voice_wav: Path, text: str, out: Path, fps: float) -> float:
    """The take on its own: `still` (the clip's last frame) held while Pixel
    pops in and says `voice_wav`. Returns its length."""
    from PIL import Image, ImageEnhance

    levels = _voice_levels(voice_wav, fps)
    voice_len = len(levels) / fps
    total = LEAD + voice_len + HOLD
    n = int(math.ceil(total * fps))
    bg = ImageEnhance.Brightness(Image.open(still).convert("RGB").resize((VW, VH))).enhance(1.0).convert("RGBA")
    bubble = _bubble(text) if text else None
    proc = subprocess.Popen(
        [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{VW}x{VH}", "-r", f"{fps:g}",
         "-i", "-", "-i", str(voice_wav),
         "-filter_complex", f"[1:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I={VOICE_LUFS}:TP=-1.5:LRA=11,"
                            f"aresample=48000,adelay={int(LEAD * 1000)}|{int(LEAD * 1000)},apad[a]",
         "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in range(n):
            t = f / fps
            img = bg.copy()
            p = min(1.0, t / 0.3)
            s = (1 + 0.18 * math.sin(p * math.pi)) * (p ** 0.5) if p < 1 else 1 + 0.03 * math.sin((t - 0.3) * 9) * math.exp(-(t - 0.3) * 3)
            sx = s * (1 + 0.06 * math.sin(p * math.pi * 2))
            sy = s * (1 - 0.06 * math.sin(p * math.pi * 2))
            vi = int((t - LEAD) * fps)
            m = levels[vi] if 0 <= vi < len(levels) else 0.0
            talking = 0 <= vi < len(levels)
            ch = draw_pixel(m, (t % 2.6) < 0.1, 0.5 * math.sin(t * 1.3))
            if sx > 0.03:
                ch = ch.resize((max(1, int(SIZE * sx)), max(1, int(SIZE * sy))))
                bob = int(6 * math.sin(t * 5)) if talking else 0
                img.alpha_composite(ch, (MARGIN_X + (SIZE - ch.width) // 2, BOTTOM_Y - ch.height + bob))
            if bubble is not None and t >= LEAD:
                q = min(1.0, (t - LEAD) / 0.18)
                b = bubble if q >= 1 else bubble.resize((max(1, int(bubble.width * q)), max(1, int(bubble.height * q))))
                img.alpha_composite(b, (MARGIN_X + SIZE - 30, BOTTOM_Y - SIZE + 10 - b.height + 30))
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-400:]}")
    except BaseException:
        proc.kill()
        raise
    return round(total, 2)


def apply(clip: Path, voice_wav: Path, text: str, backup: Path) -> dict:
    """Put the take on the end of `clip`, in place. `backup` keeps the clip
    as it was without a take (made once; reused when the take is redone, so
    takes never stack). Returns {"duration", "tail"}."""
    if not backup.is_file():
        shutil.copy2(clip, backup)
    base = backup
    info = probe(base)
    fps = info["fps"] if 10 <= info["fps"] <= 120 else 30.0
    work = clip.parent / f".{clip.stem}_take"
    work.mkdir(exist_ok=True)
    try:
        still, tail, tmp = work / "last.png", work / "tail.mp4", work / "joined.mp4"
        subprocess.run([_ffmpeg(), "-v", "error", "-y", "-sseof", "-0.08", "-i", str(base), "-frames:v", "1",
                        "-vf", f"scale={VW}:{VH}", str(still)], check=True, capture_output=True, timeout=60)
        tail_len = render_tail(still, voice_wav, text, tail, fps)
        a0 = "[0:a]" if info["audio"] else None
        inputs = [_ffmpeg(), "-v", "error", "-y", "-i", str(base), "-i", str(tail)]
        if a0 is None:
            inputs += ["-f", "lavfi", "-t", f"{info['duration']:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            a0 = "[2:a]"
        fc = (f"[0:v]scale={VW}:{VH},setsar=1,fps={fps:g}[v0];[1:v]setsar=1,fps={fps:g}[v1];"
              f"{a0}aresample=48000,aformat=channel_layouts=stereo[a0];[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
              f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
        subprocess.run(inputs + ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
                                 "-movflags", "+faststart", str(tmp)], check=True, capture_output=True, timeout=1200)
        tmp.replace(clip)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"duration": round(probe(clip)["duration"], 2), "tail": tail_len}


def remove(clip: Path, backup: Path) -> Optional[float]:
    """Put the clip back as it was before the take. Its length, or None if
    there was no take on it."""
    if not backup.is_file():
        return None
    backup.replace(clip)
    return round(probe(clip)["duration"], 2)
