"""Read what's actually IN a competitor's top-performing short -- its
spoken transcript and where it gets loud -- instead of judging it by title
alone. Both are cheap for a Shorts-length video (this app already filters
to under 3 minutes elsewhere, see channel_insights.py): captions need no
video download at all, and loudness needs only the audio track, not the
full video.

Best-effort throughout: a private/missing caption track, no audio track,
or a transient yt-dlp error degrades to "no content signal for this video"
rather than blocking the AI overview that's asking for it.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .download import base_ydl_opts, run_with_timeout
from .loud_moments import LoudMoment, find_loud_moments
from .transcribe import words_from_vtt


@dataclass
class VideoContent:
    transcript_text: str
    loud_moments: List[LoudMoment]


def _fetch_captions_vtt(video_url: str, out_dir: Path, lang: str = "en") -> Optional[Path]:
    import yt_dlp

    ydl_opts = {
        **base_ydl_opts(),
        "outtmpl": str(out_dir / "captions.%(ext)s"),
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
    }

    def _extract() -> None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)

    run_with_timeout(_extract, timeout_seconds=60.0)
    matches = list(out_dir.glob("captions*.vtt"))
    return matches[0] if matches else None


def _fetch_audio(video_url: str, out_dir: Path) -> Optional[Path]:
    import yt_dlp

    ydl_opts = {
        **base_ydl_opts(),
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "format": "bestaudio/best",
    }

    def _extract() -> None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)

    run_with_timeout(_extract, timeout_seconds=90.0)
    matches = [p for p in out_dir.glob("audio.*") if p.is_file()]
    return matches[0] if matches else None


def analyze_video_content(video_url: str, duration: float, with_loudness: bool = True) -> Optional[VideoContent]:
    """Fetch a public video's transcript and (optionally) its loud-moment
    timestamps. Returns None if neither signal could be gathered -- a
    caller should treat that exactly like "this video wasn't analyzed",
    not as an error."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="content_"))
    try:
        transcript_text = ""
        try:
            vtt_path = _fetch_captions_vtt(video_url, tmp_dir)
            if vtt_path:
                words = words_from_vtt(vtt_path)
                transcript_text = " ".join(w.text for w in words)
        except Exception as e:
            print(f"[competitor_content] captions fetch failed for {video_url}: {e}", flush=True)

        loud_moments: List[LoudMoment] = []
        if with_loudness:
            try:
                audio_path = _fetch_audio(video_url, tmp_dir)
                if audio_path:
                    loud_moments = find_loud_moments(audio_path, duration)
            except Exception as e:
                print(f"[competitor_content] audio/loudness fetch failed for {video_url}: {e}", flush=True)

        if not transcript_text and not loud_moments:
            return None
        return VideoContent(transcript_text=transcript_text, loud_moments=loud_moments)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
