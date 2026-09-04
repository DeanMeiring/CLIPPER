"""Fetch a source video (YouTube URL or local file) plus captions if available."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_cookiefile_cache: Optional[str] = None


def _cookies_to_netscape(raw: str) -> str:
    """Accept either a Netscape cookies.txt already, or a JSON cookie export
    (e.g. the Cookie-Editor browser extension's {"url":..., "cookies":[...]}
    or a bare array of the same objects) and return Netscape format, which
    is what yt-dlp's cookiefile option expects."""
    raw = raw.strip()
    if not raw or raw[0] not in "{[":
        return raw  # already Netscape (or empty)

    data = json.loads(raw)
    cookies = data["cookies"] if isinstance(data, dict) else data

    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expiry = int(c["expirationDate"]) if c.get("expirationDate") else 0
        lines.append("\t".join([
            domain, flag, path, secure, str(expiry), c.get("name", ""), c.get("value", ""),
        ]))
    return "\n".join(lines) + "\n"


def _cookiefile() -> Optional[str]:
    """Path to a Netscape-format cookies.txt for yt-dlp, or None.

    Cloud IPs (Railway included) regularly get YouTube's "sign in to
    confirm you're not a bot" wall, which only real session cookies get
    past. Set YTDLP_COOKIES_FILE to a path already on disk (e.g. a mounted
    volume), or YTDLP_COOKIES to the exported cookies -- either a Netscape
    cookies.txt (e.g. from "Get cookies.txt LOCALLY") or a JSON export
    (e.g. from "Cookie-Editor") -- and it's converted if needed, written to
    a temp file once, and reused.
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
        f.write(_cookies_to_netscape(raw))
    _cookiefile_cache = path
    return path


@dataclass
class DownloadResult:
    video_path: Path
    duration: float
    title: str
    captions_path: Optional[Path]  # .vtt if YouTube provided one, else None


@dataclass
class VideoInfo:
    id: str
    duration: float
    title: str
    extractor: str          # e.g. "twitch:vod", "youtube"
    broadcaster_login: Optional[str]  # best-effort; None if not applicable/unavailable


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source))


def _base_ydl_opts() -> dict:
    opts = {"quiet": True, "no_warnings": False, "noplaylist": True}
    cookiefile = _cookiefile()
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


def probe_video(source: str) -> VideoInfo:
    """Fetch metadata (duration, title, platform) without downloading --
    used to decide whether a source needs the long-VOD highlight pipeline."""
    import yt_dlp

    with yt_dlp.YoutubeDL(_base_ydl_opts()) as ydl:
        info = ydl.extract_info(source, download=False)

    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    video_id = str(info.get("id", ""))
    if "twitch" in extractor and video_id.startswith("v") and video_id[1:].isdigit():
        # yt-dlp prefixes Twitch VOD ids with "v" as its own internal
        # convention; Twitch's actual API (e.g. a clip's video_id field)
        # uses the bare numeric id, so anything comparing against Twitch's
        # own data needs this stripped.
        video_id = video_id[1:]

    return VideoInfo(
        id=video_id,
        duration=float(info.get("duration") or 0.0),
        title=info.get("title", ""),
        extractor=extractor,
        broadcaster_login=info.get("uploader_id") or info.get("uploader") or None,
    )


def download_range(source: str, out_dir: Path, start: float, end: float, out_name: str) -> Path:
    """Download only [start, end] seconds of `source` as a standalone file --
    for pulling a short candidate window out of a long VOD without fetching
    the whole thing. Needs a server that supports HTTP range requests, which
    real video CDNs (YouTube, Twitch) do."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / f"{out_name}.%(ext)s")
    ydl_opts = {
        **_base_ydl_opts(),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(source, download=True)

    candidates = list(out_dir.glob(f"{out_name}.*"))
    video_candidates = [p for p in candidates if p.suffix in {".mp4", ".mkv", ".webm"}]
    if not video_candidates:
        raise RuntimeError(f"download_range: no output file found for {out_name}")
    return video_candidates[0]


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
        **_base_ydl_opts(),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
    }
    cookiefile = ydl_opts.get("cookiefile")
    if cookiefile:
        try:
            n_lines = sum(1 for line in open(cookiefile) if line.strip() and not line.startswith("#"))
        except OSError:
            n_lines = -1
        print(f"[clipper] cookiefile: {cookiefile} ({n_lines} cookie lines)", flush=True)
    else:
        print("[clipper] no cookiefile configured", flush=True)

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
