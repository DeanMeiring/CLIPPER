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

import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .download import CorruptDownload, VideoInfo, download_range
from .highlights import find_candidate_windows
from .select_moments import WindowPick, select_from_candidate_windows
from .transcribe import get_transcript

LONG_VOD_THRESHOLD_SECONDS = 90 * 60  # 90 minutes

# A run of corrupt downloads this long means the source itself is being
# rate-limited or blocked (seen in practice: every single candidate for a
# VOD failing identically) rather than one flaky request -- grinding
# through the rest of the candidate list at that point just burns ~30-90s
# per candidate for a guaranteed failure, so stop early instead.
MAX_CONSECUTIVE_CORRUPT_DOWNLOADS = 4

# How many times the upfront accessibility probe retries before concluding
# the source is genuinely inaccessible rather than just unlucky once.
_PROBE_ATTEMPTS = 2


def is_long_vod(info: VideoInfo) -> bool:
    return info.duration >= LONG_VOD_THRESHOLD_SECONDS and "twitch" in info.extractor


def probe_source_accessible(source: str, duration: float, raw_dir: Path) -> None:
    """Cheap upfront sanity check: pull one short window from partway
    through the VOD before committing to downloading/transcribing up to
    20 full candidate windows. A VOD that's subscriber-only, deleted-but-
    still-listed, or otherwise inaccessible returns the same tiny
    placeholder response for every range request (real content elsewhere
    in the VOD makes no difference -- confirmed in practice: 4 candidates
    spread across a whole VOD all failed identically). Catching that here
    on one quick probe fails fast with a clear, specific reason instead of
    only surfacing it after several wasted candidate downloads deep into
    the real list, each of which can take up to a minute.

    Retries a couple of times before concluding real inaccessibility, so
    one flaky/rate-limited request doesn't wrongly abort an otherwise-
    fine VOD -- a non-corrupt failure (timeout, network hiccup) isn't
    conclusive either way and is left for the real candidate loop to
    sort out."""
    probe_start = max(0.0, duration * 0.5 - 5.0)
    probe_end = probe_start + 8.0
    last_error: Optional[CorruptDownload] = None
    for attempt in range(_PROBE_ATTEMPTS):
        if attempt > 0:
            time.sleep(3)
        try:
            download_range(source, raw_dir, probe_start, probe_end, out_name="_probe")
            return  # got real video back -- source is accessible
        except CorruptDownload as e:
            last_error = e
        except Exception:
            return  # inconclusive (timeout, network hiccup) -- let the real loop judge it
        finally:
            for p in raw_dir.glob("_probe.*"):
                p.unlink(missing_ok=True)

    raise RuntimeError(
        "This VOD's video content isn't accessible right now -- most likely "
        "subscriber-only, restricted, or currently blocked by the source "
        f"(a probe near the middle of the VOD came back empty {_PROBE_ATTEMPTS} times in "
        "a row). Try a different VOD, or set YTDLP_COOKIES to an account with "
        "access to this one."
    ) from last_error


def gather_candidates(
    source: str,
    info: VideoInfo,
    raw_dir: Path,
    max_windows: int = 20,
    on_progress: Optional[Callable[[str], None]] = None,
    on_candidate_progress: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], None]] = None,
) -> List[dict]:
    """Find, download, and transcribe candidate windows. Returns a list of
    dicts ready for select_from_candidate_windows, each also carrying its
    downloaded video_path for the render step.

    `should_cancel`, if given, is called at each cancellable point and is
    expected to raise if the caller wants to abort the job."""

    def report(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        print(f"[long_vod] {msg}", flush=True)

    if should_cancel:
        should_cancel()

    report("Checking that the source video is actually accessible...")
    probe_source_accessible(source, info.duration, raw_dir)

    if should_cancel:
        should_cancel()

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
    consecutive_corrupt = 0
    for i, w in enumerate(windows):
        if should_cancel:
            should_cancel()
        if i > 0:
            # A pause between range-requests against the same VOD -- back-
            # to-back rapid-fire requests are the likely trigger for Twitch
            # rate-limiting the source mid-run (observed as downloads
            # degrading into tiny error-page-sized files partway through a
            # long candidate list).
            time.sleep(3)
        report(f"Candidate {i + 1}/{len(windows)}: {w.start:.0f}s-{w.end:.0f}s ({w.detail})")
        if on_candidate_progress:
            on_candidate_progress(i, len(windows))
        try:
            video_path = download_range(source, raw_dir, w.start, w.end, out_name=f"cand_{i:03d}")
        except CorruptDownload as e:
            consecutive_corrupt += 1
            report(f"  download failed, skipping: {e}")
            if consecutive_corrupt >= MAX_CONSECUTIVE_CORRUPT_DOWNLOADS:
                report(
                    f"  {consecutive_corrupt} downloads in a row came back corrupt -- "
                    "the source looks rate-limited or blocked right now. Stopping early "
                    "instead of repeating this for every remaining candidate; try again "
                    "later or with a different VOD."
                )
                break
            continue
        except Exception as e:
            consecutive_corrupt = 0
            report(f"  download failed, skipping: {e}")
            continue
        consecutive_corrupt = 0
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
    strategy_notes: Optional[str] = None,
) -> List[Tuple[Path, WindowPick]]:
    """Run Claude selection over the candidates and pair each pick with its
    downloaded video file, ready for the existing render pipeline."""
    select_input = [
        {"index": c["index"], "duration": c["duration"], "words": c["words"], "signal": c["signal"]}
        for c in candidates
    ]
    picks = select_from_candidate_windows(
        select_input, n_clips=n_clips, min_len=min_len, max_len=max_len,
        focus=focus, api_key=api_key, source_title=source_title, strategy_notes=strategy_notes,
    )
    by_index = {c["index"]: c for c in candidates}
    return [(by_index[p.window_index]["video_path"], p) for p in picks if p.window_index in by_index]
