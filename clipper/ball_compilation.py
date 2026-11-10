"""The weekly compilation: the week's Ball Evolution videos in one long video (10 Nov).

From the channel research: Shorts pay very little (about $0.02-0.30 per
1,000 views) and the Partner Program's Shorts route needs 10M views in 90
days (20M for new applicants from 1 Feb 2027), far off; the other route is
4,000 hours of long-form watch time, which the big ball channels earn with
compilations and tournaments (SEEDPIT's 8-10 minute videos get 100k-967k
views, Earclacks' full season tournament). So every Sunday evening the
week's posted videos are joined into one regular (not Short) upload with a
chapter per video.

Cheap on purpose (Dean: Railway costs): the videos are joined with
ffmpeg's concat demuxer without re-encoding (every video here is the same
1080x1920 H.264 + AAC), a few seconds of CPU; only if that fails is it
re-encoded once (veryfast). Kept under 14 minutes: channels without phone
verification can't upload longer than 15.
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path
from typing import List

MAX_S = 14 * 60
MIN_VIDEOS = 3           # YouTube wants 3+ chapters of 10+ s each
MIN_S = 190              # a vertical video of 3 minutes or less is a Short: no long-form watch hours


def pick(items: List[dict], max_s: float = MAX_S) -> List[dict]:
    """items: {path, title, seconds, posted} oldest first. Keeps the newest
    that fit in max_s, back in posting order."""
    out, total = [], 0.0
    for it in sorted(items, key=lambda x: x["posted"], reverse=True):
        if total + it["seconds"] > max_s:
            continue
        out.append(it)
        total += it["seconds"]
    return sorted(out, key=lambda x: x["posted"])


def _stamp(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _clean(title: str) -> str:
    return " ".join(title.replace("#Shorts", "").split())


def text(items: List[dict], start: datetime.date, end: datetime.date) -> dict:
    """Title + description with a chapter per video (YouTube makes the
    chapters from the timestamps)."""
    span = f"{start.day} {start:%b}" + (f" - {end.day} {end:%b}" if end != start else "")
    title = f"Every ball video this week, all in one ({span})"
    lines, t = [], 0.0
    for it in items:
        lines.append(f"{_stamp(t)} {_clean(it['title'])[:80]}")
        t += it["seconds"]
    desc = ("All of this week's ball videos back to back: fill-up boxes, spike rings, the Ball League and "
            "the comment fights.\n\nWhich one was your favourite? 👇\n\n" + "\n".join(lines) +
            "\n\n#satisfying #physics #bouncingball #compilation")
    return {"title": title[:100], "description": desc[:4900]}


def build(ffmpeg: str, paths: List[Path], out: Path) -> None:
    """Join the videos into out: stream copy, else one re-encode."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    try:
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if r.returncode != 0 or not out.is_file() or out.stat().st_size < 1000:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)], check=True)
    finally:
        lst.unlink(missing_ok=True)
