"""Fetch a source video (YouTube URL or local file) plus captions if available."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypeVar


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
    created_at: Optional[float] = None  # unix timestamp the VOD/stream started, if known


def is_url(source: str) -> bool:
    return bool(URL_RE.match(source))


def _base_ydl_opts() -> dict:
    opts = {
        "quiet": True, "no_warnings": False, "noplaylist": True,
        # A persistently failing source (e.g. a 403 on a restricted/expired
        # VOD) can otherwise have yt-dlp retry with growing backoff for a
        # very long time -- bound that so a bad source fails in minutes,
        # not indefinitely. _run_with_timeout below is the hard backstop
        # in case even this isn't enough (a genuinely stalled connection).
        "socket_timeout": 30,
        "retries": 3,
        "extractor_retries": 3,
        "fragment_retries": 3,
    }
    cookiefile = _cookiefile()
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts


_T = TypeVar("_T")


class DownloadTimeout(RuntimeError):
    pass


class CorruptDownload(RuntimeError):
    """A download "succeeded" but the file is too small to be real video --
    kept distinct from other failures so callers can specifically notice a
    run of these (a strong signal the source itself is being rate-limited/
    blocked, not just one flaky request) and stop early instead of grinding
    through every remaining candidate for nothing."""


def _run_with_timeout(
    fn: Callable[[], _T],
    timeout_seconds: float,
    on_late_completion: Optional[Callable[[], None]] = None,
) -> _T:
    """Run fn() in a daemon thread with a hard wall-clock ceiling. yt-dlp
    has no built-in way to bound how long a single extract_info call can
    take, and a call that hangs (a stalled connection, a source that keeps
    erroring through every retry) can't be interrupted by anything else in
    the pipeline -- including the emergency-stop signal, since it's one
    blocking call with no checkpoint inside it. This turns an indefinite
    hang into a clean, timed-out error instead.

    Python can't force-kill a thread, so on timeout the abandoned thread
    keeps running in the background and can still finish (and write its
    output file) well after the caller has moved on -- possibly after the
    job/candidate it belonged to has been deleted or superseded. Pass
    on_late_completion to have that stray result cleaned up (e.g. delete
    whatever file it eventually wrote) instead of left lying around."""
    result: list = []
    error: list = []
    finished = threading.Event()

    def _run() -> None:
        try:
            result.append(fn())
        except Exception as e:  # noqa: BLE001 - re-raised on the caller's thread below
            error.append(e)
        finally:
            finished.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        if on_late_completion is not None:
            def _cleanup_when_done() -> None:
                finished.wait()
                try:
                    on_late_completion()
                except Exception:
                    pass  # best-effort -- the job/dir it belonged to may be long gone

            threading.Thread(target=_cleanup_when_done, daemon=True).start()
        raise DownloadTimeout(
            f"Timed out after {timeout_seconds:.0f}s -- the source may be "
            "unavailable, restricted, or the connection stalled"
        )
    if error:
        raise error[0]
    return result[0]


def probe_video(source: str) -> VideoInfo:
    """Fetch metadata (duration, title, platform) without downloading --
    used to decide whether a source needs the long-VOD highlight pipeline."""
    import yt_dlp

    def _extract() -> dict:
        with yt_dlp.YoutubeDL(_base_ydl_opts()) as ydl:
            return ydl.extract_info(source, download=False)

    info = _run_with_timeout(_extract, timeout_seconds=90.0)

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
        created_at=float(info["timestamp"]) if info.get("timestamp") else None,
    )


# A real candidate window -- even a few seconds at the lowest quality
# Twitch/YouTube serve -- is comfortably above this. Below it means the
# server handed back an error page or empty placeholder instead of video
# (seen in practice: Twitch rate-limiting a source after many rapid
# range-requests in a row, degrading to a ~260-byte response that yt-dlp
# still reports as a "successful" 100% download). Catching that here turns
# a silently corrupt candidate -- which then fails transcription anyway,
# having burned the download time for nothing -- into a clean, fast-failing
# skip. Tuned specifically for the range-request candidate-window path
# (download_range) below, where every window is expected to be a real,
# multi-second slice -- NOT for a full-source download, where a
# legitimately tiny source (a several-second Short at low resolution)
# could plausibly land under this.
_MIN_VIDEO_BYTES = 50_000

# Used for full-source downloads (download_video): just needs to catch an
# empty/error-page response (typically well under a couple KB), not
# enforce a minimum content length -- mp4 muxing overhead alone puts any
# genuine video file, however short or low-quality, comfortably above this.
_MIN_FULL_VIDEO_BYTES = 2_000


def _check_not_corrupt(path: Path, label: str, min_bytes: int = _MIN_VIDEO_BYTES) -> None:
    size = path.stat().st_size
    if size < min_bytes:
        # The corrupt response is small enough to just show it -- rather
        # than guessing "rate-limited?" every time, surface what the
        # source actually said (a JSON error blob, an HTML "subscribers
        # only" or "video unavailable" page, a rate-limit message, etc.)
        # so a failure like this is self-diagnosing in the logs instead
        # of needing to be reproduced and inspected by hand.
        snippet = ""
        try:
            text = path.read_bytes()[:500].decode("utf-8", errors="replace").strip()
            if text:
                snippet = f" -- content: {text[:300]!r}"
        except OSError:
            pass
        raise CorruptDownload(
            f"{label}: downloaded file is only {size} bytes -- likely an error "
            f"response from the source rather than real video{snippet}"
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

    def _extract() -> None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(source, download=True)

    def _cleanup_late_files() -> None:
        for p in out_dir.glob(f"{out_name}.*"):
            p.unlink(missing_ok=True)

    _run_with_timeout(_extract, timeout_seconds=240.0, on_late_completion=_cleanup_late_files)

    candidates = list(out_dir.glob(f"{out_name}.*"))
    video_candidates = [p for p in candidates if p.suffix in {".mp4", ".mkv", ".webm"}]
    if not video_candidates:
        raise RuntimeError(f"download_range: no output file found for {out_name}")
    result = video_candidates[0]
    _check_not_corrupt(result, f"download_range({out_name})")
    return result


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

    def _extract() -> dict:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(source, download=True)

    # The output filename is only known once extract_info returns (it's
    # derived from the video id), so on a timeout-abandoned download that
    # completes late, clean up by diffing the directory instead: anything
    # that shows up after the fact that wasn't here when the download
    # started is this call's stray output.
    existing_before = set(out_dir.iterdir()) if out_dir.exists() else set()

    def _cleanup_late_files() -> None:
        for p in out_dir.iterdir():
            if p not in existing_before:
                p.unlink(missing_ok=True)

    info = _run_with_timeout(_extract, timeout_seconds=600.0, on_late_completion=_cleanup_late_files)
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

    _check_not_corrupt(video_path, f"download_video({video_id})", min_bytes=_MIN_FULL_VIDEO_BYTES)

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
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(result.stdout.strip())
