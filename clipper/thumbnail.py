"""Generate a downloadable Shorts thumbnail from an already-rendered clip:
a real frame from the clip with one bold line of caption text burned over
it -- the "screenshot + one line of text" style, not an AI-composited
image.

Burns the text via the same libass ffmpeg pass every other piece of
on-screen text in this app already goes through (see captions.py,
weekly_recap.py's intro/outro cards), rather than a new dependency like
Pillow -- it inherits the exact font/outline rendering already proven to
work in this container, for free.
"""
import subprocess
from pathlib import Path
from typing import List, Optional

from .captions import _escape_ass_text
from .loud_moments import find_loud_moments


def pick_thumbnail_frame_times(video_path: Path, duration: float, n: int = 4) -> List[float]:
    """Up to `n` candidate instants (seconds into the clip's OWN
    0..duration timeline, not the source video's) worth grabbing a
    thumbnail frame from, best first.

    Ranks by find_loud_moments' jump_db (how far a moment's peak volume
    sits above the surrounding baseline) as a proxy for "the exciting
    part" -- a reaction, a shout, a punchline landing. Pads out to `n`
    with frames spread evenly across the clip when there aren't enough
    (or any) loud moments, so a quiet clip still gets a full set of
    candidates instead of just one. Never raises -- loud-moment detection
    failing (an unparseable audio track) just means every candidate falls
    back to the spread."""
    # Clamp against the file's own real duration, not just the caller-
    # supplied metadata value -- the two can drift apart (concat rounding,
    # a stale duration field), and a candidate seeked past the actual end
    # of the file is a real, previously-hit failure mode: ffmpeg exits 0
    # but silently writes no frame at all (see render_thumbnail's own
    # existence check, which exists because of exactly this).
    try:
        real_duration = _probe_duration(video_path)
        if real_duration:
            duration = min(duration, real_duration)
    except Exception:
        pass

    try:
        moments = find_loud_moments(video_path, duration)
    except Exception:
        moments = []

    picked: List[float] = []
    for m in sorted(moments, key=lambda m: m.jump_db, reverse=True):
        t = max(0.0, min(duration, (m.start + m.end) / 2))
        if all(abs(t - p) > 1.0 for p in picked):
            picked.append(t)
        if len(picked) >= n:
            break

    i = 1
    while len(picked) < n:
        t = duration * i / (n + 1)
        if all(abs(t - p) > 1.0 for p in picked):
            picked.append(t)
        i += 1
        if i > n * 3:  # a very short clip can run out of room to spread into
            picked.append(duration / 2)

    return picked[:n]


def _probe_dimensions(video_path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", "-i", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)


def _probe_duration(video_path: Path) -> Optional[float]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "default=nk=1:nw=1", "-i", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def render_thumbnail(video_path: Path, frame_time: float, text: str, out_path: Path) -> None:
    """Grab the frame at `frame_time` and burn `text` across it as a
    single bold yellow line with a black outline and a slight tilt --
    matching the classic clip-thumbnail meme look, not a caption.

    Renders at the clip's own resolution (portrait for a normal Short,
    landscape for a recap clip) rather than forcing a fixed size -- the
    clip is already exactly the target aspect ratio, so there's nothing
    to crop or scale, only text to burn on top.

    Clamps `frame_time` to just inside the file's own real duration, and
    verifies the output actually got written -- both defend against the
    same failure mode: ffmpeg given a seek past the input's real end
    exits 0 but silently produces no frame at all, which without this
    would leave a broken-image thumbnail with no error anywhere."""
    out_w, out_h = _probe_dimensions(video_path)
    real_duration = _probe_duration(video_path)
    if real_duration:
        frame_time = max(0.0, min(frame_time, real_duration - 0.05))
    ass_path = out_path.with_suffix(".ass")
    fontsize = round(out_w * 0.105)
    outline = max(2, round(fontsize * 0.09))
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {out_w}\nPlayResY: {out_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Thumb,Arial Black,{fontsize},&H0000FFFF,&H0000FFFF,"
        f"&H00000000,&H00000000,-1,0,0,0,100,100,0,-8,1,{outline},0,5,40,40,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    dialogue = f"Dialogue: 0,0:00:00.00,0:00:05.00,Thumb,,0,0,0,,{_escape_ass_text(text)}"
    ass_path.write_text(header + dialogue + "\n", encoding="utf-8")
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{frame_time:.3f}", "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"ass='{ass_escaped}'",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    finally:
        ass_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed rendering thumbnail:\n{result.stderr[-2000:]}")
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg reported success but wrote no thumbnail (frame_time={frame_time:.3f}s):\n"
            f"{result.stderr[-2000:]}"
        )
