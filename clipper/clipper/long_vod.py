"""Pipeline for long-form Twitch VODs (streams, not short videos).

Downloading and transcribing an entire multi-hour VOD before asking Claude
to find the good parts doesn't work -- see highlights.py's docstring for
why. Instead:

  1. Find candidate highlight windows from chat activity (highlights.py) --
     a few dozen short windows out of the whole VOD, not the whole thing.
  2. Download and transcribe ONLY those windows (each is its own small
     file, via download.py's download_range).
  3. Ask Claude to pick the best ones from that much smaller candidate set
     (select_moments.py's select_from_candidate_windows).

The result is a list of (video_path, WindowPick) pairs, each a short
standalone file ready for the existing render pipeline (compute_layout,
build_ass, render_clip) -- unchanged, since every candidate window is just
"a video file" to those functions, the same as any other source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .download import VideoInfo, download_range
from .highlights import find_candidate_windows
from .select_moments import WindowPick, select_from_candidate_windows
from .transcribe import get_transcript

LONG_VOD_THRESHOLD_SECONDS = 90 * 60  # 90 minutes


def is_long_vod(info: VideoInfo) -> bool:
    return info.duration >= LONG_VOD_THRESHOLD_SECONDS and "twitch" in info.extractor


def gather_candidates(
    source: str,
    info: VideoInfo,
    raw_dir: Path,
    max_windows: int = 20,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Find, download, and transcribe candidate windows. Returns a list of
    dicts ready for select_from_candidate_windows, each also carrying its
    downloaded video_path for the render step."""

    def report(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        print(f"[long_vod] {msg}", flush=True)

    report("Scanning chat activity for highlight moments...")
    windows = find_candidate_windows(
        source, info.duration,
        broadcaster_login=info.broadcaster_login, vod_id=info.id, created_at=info.created_at,
        max_windows=max_windows,
    )
    if not windows:
        report("No candidate moments found from chat activity.")
        return []
    report(f"Found {len(windows)} candidate moment(s); downloading and transcribing each...")

    candidates = []
    for i, w in enumerate(windows):
        report(f"Candidate {i + 1}/{len(windows)}: {w.start:.0f}s-{w.end:.0f}s ({w.detail})")
        try:
            video_path = download_range(source, raw_dir, w.start, w.end, out_name=f"cand_{i:03d}")
        except Exception as e:
            report(f"  download failed, skipping: {e}")
            continue
        try:
            words = get_transcript(video_path, None, prefer_whisper=True)
        except Exception as e:
            report(f"  transcription failed, skipping: {e}")
            continue
        candidates.append({
            "index": i,
            "video_path": video_path,
            "duration": round(w.end - w.start, 2),
            "words": words,
            "signal": w.detail,
        })

    return candidates


def select_and_map(
    candidates: List[dict],
    n_clips: int,
    min_len: float,
    max_len: float,
    focus: Optional[str],
    api_key: Optional[str],
    source_title: str,
) -> List[Tuple[Path, WindowPick]]:
    """Run Claude selection over the candidates and pair each pick with its
    downloaded video file, ready for the existing render pipeline."""
    select_input = [
        {"index": c["index"], "duration": c["duration"], "words": c["words"], "signal": c["signal"]}
        for c in candidates
    ]
    picks = select_from_candidate_windows(
        select_input, n_clips=n_clips, min_len=min_len, max_len=max_len,
        focus=focus, api_key=api_key, source_title=source_title,
    )
    by_index = {c["index"]: c for c in candidates}
    return [(by_index[p.window_index]["video_path"], p) for p in picks if p.window_index in by_index]
