"""Fetch a source video (YouTube URL or local file) plus captions if available."""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_cookiefile_cache: Optional[str] = None


def _cookiefile() -> Optional[str]:
    """Path to a Netscape-format cookies.txt for yt-dlp, or None.

    Cloud IPs (Railway included) regularly get YouTube's "sign in to
    confirm you're not a bot" wall, which only real session cookies get
    past. Set YTDLP_COOKIES_FILE to a path already on disk (e.g. a mounted
    volume), or YTDLP_COOKIES to the raw file contents (exported from a
    logged-in browser via an extension like "Get cookies.txt LOCALLY") and
    it's written to a temp file once and reused.
    """
    global _cookiefile_cache
    direct = os.environ.get("YTDLP_COOKIES_FILE")
    if direct:
        return direct
    if _cookiefile_cache:
        return _cookiefile_cache
    raw = os.environ.get("YTDLP_COOKIES")
    if not raw:
        return None
    fd, path = tempfile.mkstemp(prefix="yt_cookies_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(raw)
    _cookiefile_cache = path
    return path


@dataclass
class DownloadResult:
    video_path: Path
    duration: float
    title: str
    captions_path: Optional[Path]  # .vtt if YouTube provided one, else None


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source))


def download_video(source: str, out_dir: Path, lang: str = "en") -> DownloadResult:
    """Download `source` (a YouTube/URL) into out_dir, or wrap a local file path.

    Grabs YouTube's own auto-generated captions when available (fast, free,
    good enough for moment-selection and as a captions fallback). Falls back
    to local Whisper transcription later in the pipeline if none exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        duration = _ffprobe_duration(path)
        return DownloadResult(video_path=path, duration=duration, title=path.stem, captions_path=None)

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required to download from a URL. Install it with: pip install yt-dlp"
        ) from e

    outtmpl = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    cookiefile = _cookiefile()
    if cookiefile:
        ydl_opts["cookiefile"] = cookiefile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        video_id = info["id"]
        title = info.get("title", video_id)
        duration = float(info.get("duration") or 0.0)

    video_path = out_dir / f"{video_id}.mp4"
    if not video_path.exists():
        # yt-dlp may have kept the original container if merge wasn't needed
        candidates = list(out_dir.glob(f"{video_id}.*"))
        video_candidates = [p for p in candidates if p.suffix in {".mp4", ".mkv", ".webm"}]
        if not video_candidates:
            raise RuntimeError(f"Download finished but no video file found for {video_id}")
        video_path = video_candidates[0]

    if duration <= 0:
        duration = _ffprobe_duration(video_path)

    captions_path = None
    for p in out_dir.glob(f"{video_id}.{lang}*.vtt"):
        captions_path = p
        break

    return DownloadResult(video_path=video_path, duration=duration, title=title, captions_path=captions_path)


def _ffprobe_duration(video_path: Path) -> float:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())
