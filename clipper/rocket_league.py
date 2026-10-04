"""Dean's own Rocket League clips as daily Shorts on his RL channel ("The
Deniboy"): a backlog of gameplay clips, each trimmed, given his own text
and a montage track whose beat drop lands on the goal, then scheduled on
YouTube one per day.

  find_goal()   the goal moment: the loudest burst in the game audio (the
                goal explosion and horn), editable on the page.
  find_drop()   where a track's drop is: the biggest jump in bass energy.
  render()      1080x1920: the gameplay full screen (the middle of the
                frame; ball cam keeps the car and ball there) or the whole
                frame over a blurred copy, his text on top, the channel
                name at the bottom, a white flash and a quick zoom on the
                goal, the music lined up so its drop hits the goal, the
                game sound mixed under it.
  next_slot()   the next free day at his posting time (SA time).

Everything is his own gameplay, so it's original content. Music is his
own pick: free tracks (NCS, free phonk) keep a Short earning; a label's
song is claimed by Content ID and the label gets that Short's ad money.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

from .longform_beats import F, text_w, wrap

VW, VH = 1080, 1920
MAX_SECONDS = 59.0            # Shorts with a claimed song over 60 s get blocked
GAME_LEVELS = {"low": -30.0, "medium": -24.0, "high": -19.0}   # LUFS; music sits at MUSIC_LUFS
MUSIC_LUFS = -14.0
FLASH = 0.08
ZOOM = 0.5                    # seconds of the punch-in on the goal
YEL = (250, 204, 21)
WHITE = (255, 255, 255)
TZ = "Africa/Johannesburg"
DEFAULT_SETTINGS = {
    "post_time": "17:00",
    "watermark": "The Deniboy",
    "description": "Rocket League clips every day from a South African Grand Champ 🇿🇦\n\n#rocketleague #rl #rocketleagueclips",
    "game_level": "medium",
}
_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _ffmpeg() -> str:
    return os.environ.get("CLIPPER_FFMPEG", "ffmpeg")


# ----------------------------------------------------------------- store ---

class Store:
    """BASE_DIR/_rl/: clips/<id>/ (source, clip.json, short.mp4),
    music/<id>.<ext> + music.json, settings.json."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()

    # clips
    def clip_dir(self, cid: str) -> Path:
        if not _ID_RE.match(cid or ""):
            raise KeyError(cid)
        return self.root / "clips" / cid

    def new_clip(self, name: str) -> dict:
        cid = uuid.uuid4().hex[:12]
        d = self.root / "clips" / cid
        d.mkdir(parents=True, exist_ok=True)
        clip = {"id": cid, "name": name, "uploaded_at": time.time(), "status": "uploading",
                "text": "", "description": None, "layout": "full", "music": "auto", "game_level": None}
        self._write(clip)
        return clip

    def load(self, cid: str) -> dict:
        p = self.clip_dir(cid) / "clip.json"
        if not p.is_file():
            raise KeyError(cid)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write(self, clip: dict) -> None:
        d = self.root / "clips" / clip["id"]
        tmp = d / "clip.json.tmp"
        tmp.write_text(json.dumps(clip), encoding="utf-8")
        tmp.replace(d / "clip.json")

    def update(self, cid: str, fn) -> dict:
        with self.lock:
            clip = self.load(cid)
            fn(clip)
            self._write(clip)
            return clip

    def clips(self) -> List[dict]:
        base = self.root / "clips"
        out = []
        if base.is_dir():
            for d in base.iterdir():
                try:
                    out.append(self.load(d.name))
                except (KeyError, OSError, ValueError):
                    continue
        out.sort(key=lambda c: c.get("uploaded_at") or 0)
        return out

    def delete(self, cid: str) -> None:
        import shutil
        with self.lock:
            shutil.rmtree(self.clip_dir(cid), ignore_errors=True)

    # music
    def music_dir(self) -> Path:
        d = self.root / "music"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def tracks(self) -> List[dict]:
        p = self.music_dir() / "music.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def save_tracks(self, tracks: List[dict]) -> None:
        p = self.music_dir() / "music.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(tracks), encoding="utf-8")
        tmp.replace(p)

    # settings
    def settings(self) -> dict:
        try:
            data = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return {**DEFAULT_SETTINGS, **data}

    def save_settings(self, data: dict) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        merged = {**self.settings(), **data}
        (self.root / "settings.json").write_text(json.dumps(merged), encoding="utf-8")
        return merged


# ----------------------------------------------------------------- audio ---

def probe(path: Path) -> dict:
    """{"duration", "width", "height", "fps", "audio"} from ffmpeg's banner."""
    r = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60)
    err = r.stderr
    out = {"duration": 0.0, "width": 0, "height": 0, "fps": 30.0, "audio": "Audio:" in err}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?) fps", err)
    if m:
        out["fps"] = float(m.group(1))
    if not out["width"] or out["duration"] <= 0:
        raise ValueError("not a video")
    return out


def _envelope(path: Path, step: float = 0.1, lowpass: Optional[int] = None) -> List[float]:
    """RMS loudness per `step` seconds of the file's audio (mono, 8 kHz)."""
    import numpy as np

    af = f"lowpass=f={lowpass}," if lowpass else ""
    r = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-af", af + "aresample=8000",
                        "-ac", "1", "-f", "f32le", "-"], capture_output=True, timeout=300)
    pcm = np.frombuffer(r.stdout, dtype=np.float32)
    n = int(8000 * step)
    if len(pcm) < n:
        return []
    frames = pcm[: len(pcm) // n * n].reshape(-1, n)
    return [float(x) for x in np.sqrt((frames ** 2).mean(axis=1))]


def find_goal(path: Path, start: float = 0.0, end: Optional[float] = None) -> Optional[float]:
    """The goal moment in a Rocket League clip: the loudest burst in the
    game audio (explosion + horn + crowd) that jumps clearly above what came
    before it. Seconds from the start of the file; None with no audio."""
    env = _envelope(path)
    if not env:
        return None
    lo = int(start / 0.1)
    hi = min(len(env), int(end / 0.1) if end else len(env))
    best, best_t = 0.0, None
    for i in range(max(lo, 1), hi):
        now = sum(env[i:i + 3]) / len(env[i:i + 3])
        before = sum(env[max(0, i - 20):i]) / max(1, len(env[max(0, i - 20):i]))
        score = now * 2 - before
        if score > best:
            best, best_t = score, i * 0.1
    return round(best_t, 1) if best_t is not None else None


def find_drop(path: Path) -> Optional[float]:
    """Where a track drops: the biggest jump in bass energy (the next 4 s
    against the 4 s before), at least 4 s in and 8 s before the end."""
    env = _envelope(path, step=0.1, lowpass=160)
    if len(env) < 140:
        return None
    best, best_t = 0.0, None
    for i in range(40, len(env) - 80):
        after = sum(env[i:i + 40]) / 40
        before = sum(env[i - 40:i]) / 40
        score = after / (before + 1e-4)
        if score > best:
            best, best_t = score, i * 0.1
    return round(best_t, 1) if best_t is not None else None


# --------------------------------------------------------------- schedule ---

def next_slot(post_time: str, taken: List[str], now: Optional[datetime.datetime] = None,
              tz: str = TZ) -> datetime.datetime:
    """The next day at `post_time` (local) with no Short already scheduled
    for it (taken = ISO dates "YYYY-MM-DD", local). At least 30 minutes away,
    since YouTube needs time to process the upload."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    now = (now or datetime.datetime.now(datetime.timezone.utc)).astimezone(zone)
    hh, mm = (int(x) for x in post_time.split(":"))
    day = now.date()
    for _ in range(400):
        slot = datetime.datetime.combine(day, datetime.time(hh, mm), tzinfo=zone)
        if slot - now >= datetime.timedelta(minutes=30) and day.isoformat() not in taken:
            return slot
        day += datetime.timedelta(days=1)
    raise RuntimeError("no free day in the next year")


# ---------------------------------------------------------------- render ---

def _overlay(text: str, watermark: str, wide_frame: bool, out: Path) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (VW, VH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    text = " ".join((text or "").upper().split())
    if text:
        size = 110
        while True:
            fnt = F("Anton", size)
            rows = wrap(fnt, text, VW - 140)
            if len(rows) <= 3 or size <= 64:
                break
            size -= 8
        y = 260 if not wide_frame else 300
        for i, row in enumerate(rows):
            d.text(((VW - text_w(fnt, row)) / 2, y), row, font=fnt, fill=YEL if i == len(rows) - 1 and len(rows) > 1 else WHITE,
                   stroke_width=9, stroke_fill=(0, 0, 0))
            y += int(size * 1.12)
    if watermark:
        wf = F("Inter-Black", 40)
        d.text(((VW - text_w(wf, watermark)) / 2, 1480), watermark, font=wf, fill=WHITE, stroke_width=4, stroke_fill=(0, 0, 0))
    img.save(out)
    return out


def render(source: Path, out: Path, start: float, end: float, goal: Optional[float], text: str,
           watermark: str, layout: str = "full", music: Optional[Path] = None, drop: Optional[float] = None,
           game_level: str = "medium") -> float:
    """One Short. start/end/goal are seconds in `source`; drop is seconds
    in `music`. Returns the length."""
    info = probe(source)
    length = round(min(MAX_SECONDS, max(1.0, end - start)), 3)
    fps = 60 if info["fps"] >= 50 else 30
    g = None if goal is None else goal - start
    if g is not None and not 0 <= g < length:
        g = None
    ov = _overlay(text, watermark, layout == "frame", out.with_name(f".{out.stem}_ov.png"))
    if layout == "frame":
        base = (f"[0:v]split[a][b];[a]scale={VW}:{VH}:force_original_aspect_ratio=increase,crop={VW}:{VH},"
                f"boxblur=24:2,eq=brightness=-0.3[bg];[b]scale={VW}:1400:force_original_aspect_ratio=decrease,"
                f"scale=trunc(iw/2)*2:trunc(ih/2)*2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    else:
        base = f"[0:v]scale={VW}:{VH}:force_original_aspect_ratio=increase,crop={VW}:{VH}"
    fx = ""
    if g is not None:
        zoom = (f"zoompan=z='if(between(it,{g:.3f},{g + ZOOM:.3f}),1.12-{0.12 / ZOOM:.3f}*(it-{g:.3f}),1)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={VW}x{VH}:fps={fps}")
        fx = (f",{zoom},drawbox=x=0:y=0:w=iw:h=ih:color=white@0.55:t=fill:"
              f"enable='between(t,{g:.3f},{g + FLASH:.3f})'")
    vf = (f"{base},fps={fps},setpts=PTS-STARTPTS{fx}[v1];[v1][1:v]overlay=0:0,format=yuv420p[v]")
    cmd = [_ffmpeg(), "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source),
           "-loop", "1", "-i", str(ov)]
    k = 2
    lufs = GAME_LEVELS.get(game_level, GAME_LEVELS["medium"])
    if info["audio"]:
        game = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS,"
                f"loudnorm=I={lufs}:TP=-2:LRA=11,aresample=48000,apad=whole_dur={length:.3f}[ga]")
    else:
        game = f"anullsrc=r=48000:cl=stereo,atrim=duration={length:.3f}[ga]"
    af = game
    if music is not None:
        cmd += ["-i", str(music)]
        # Line the drop up with the goal: start the track `drop - goal`
        # seconds in (or delay it when the goal comes before the drop).
        offset = (drop or 0.0) - (g if g is not None else 0.0)
        pre = f"atrim=start={offset:.3f}," if offset > 0 else ""
        delay = f"adelay={int(-offset * 1000)}|{int(-offset * 1000)}," if offset < 0 else ""
        af += (f";[{k}:a]{pre}asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
               f"loudnorm=I={MUSIC_LUFS}:TP=-1.5:LRA=11,aresample=48000,{delay}"
               f"afade=t=in:d=0.25,apad=whole_dur={length:.3f},atrim=duration={length:.3f},"
               f"afade=t=out:st={max(0.0, length - 1.0):.3f}:d=1.0[ma];"
               f"[ga][ma]amix=inputs=2:normalize=0:duration=first[a]")
    else:
        af += ";[ga]anull[a]"
    cmd += ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{length:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(fps),
            "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-600:]}")
    finally:
        ov.unlink(missing_ok=True)
    return length
