"""Detect loud audio moments (shouting, screaming, "crashing out") in a
downloaded video's audio track -- one more signal for select_clips to weigh
alongside the transcript when picking clips, the same way the long-VOD
pipeline treats chat spikes as a signal rather than a standalone trigger.

Loud alone isn't a good enough filter on its own: a game explosion or a
music sting spikes just as hard as an actual reaction. So this only flags
audio that's loud *relative to that video's own recent baseline* and hands
the caller the timestamp plus how big the jump was -- select_clips still
has to judge from the transcript whether it's actually a moment worth
clipping, the same cross-check a human editor would do before cutting on
"it got loud here" alone.
"""
from __future__ import annotations

import array
import statistics
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from math import log10
from pathlib import Path
from typing import List


@dataclass
class LoudMoment:
    start: float
    end: float
    peak_db: float   # loudest point in the moment, dBFS (0 = full scale, more negative = quieter)
    jump_db: float    # how far the peak sits above the surrounding local baseline


_WINDOW_SECONDS = 0.5
_SAMPLE_RATE = 16000
_FULL_SCALE = 32768.0  # 16-bit PCM full-scale amplitude, the dBFS reference point
_SILENCE_FLOOR_DB = -60.0


def _extract_mono_wav(video_path: Path, duration: float) -> Path:
    import os

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "wav", tmp_path,
    ]
    # Audio-only extraction from an already-local file is fast (no network,
    # far less data than a full video pass) -- this is a generous ceiling
    # against a genuinely stuck ffmpeg process, not a realistic normal case.
    timeout = max(60.0, duration * 0.3)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg audio extraction timed out after {e.timeout:.0f}s") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr[-2000:]}")
    return Path(tmp_path)


def _window_db_levels(wav_path: Path) -> List[float]:
    """RMS loudness in dBFS for consecutive _WINDOW_SECONDS windows across
    the whole file."""
    levels: List[float] = []
    frames_per_window = int(_SAMPLE_RATE * _WINDOW_SECONDS)
    with wave.open(str(wav_path), "rb") as wf:
        while True:
            raw = wf.readframes(frames_per_window)
            if not raw:
                break
            samples = array.array("h")
            samples.frombytes(raw)
            if sys.byteorder == "big":
                samples.byteswap()  # WAV PCM is little-endian
            if not samples:
                break
            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5
            db = _SILENCE_FLOOR_DB if rms <= 1.0 else max(_SILENCE_FLOOR_DB, 20 * log10(rms / _FULL_SCALE))
            levels.append(db)
    return levels


def _spike_moments(
    levels: List[float],
    window_seconds: float = _WINDOW_SECONDS,
    baseline_window_buckets: int = 120,  # +/- 60s at 0.5s buckets
    spike_db: float = 10.0,               # how many dB above local baseline counts as a spike
    min_peak_db: float = -28.0,            # absolute floor -- must be genuinely loud, not just louder than near-silence
    pad_before: float = 3.0,
    pad_after: float = 4.0,
    merge_gap: float = 4.0,
    max_moments: int = 10,
) -> List[LoudMoment]:
    n = len(levels)
    if n == 0:
        return []

    overall_baseline = statistics.median(levels)

    flagged = []
    for i in range(n):
        lo, hi = max(0, i - baseline_window_buckets), min(n, i + baseline_window_buckets)
        local_baseline = max(statistics.median(levels[lo:hi]), overall_baseline)
        jump = levels[i] - local_baseline
        if jump >= spike_db and levels[i] >= min_peak_db:
            flagged.append((i, levels[i], jump))

    if not flagged:
        return []

    moments: List[LoudMoment] = []

    def flush(start_idx: int, end_idx: int, peak: float, jump: float) -> None:
        start = max(0.0, start_idx * window_seconds - pad_before)
        end = end_idx * window_seconds + window_seconds + pad_after
        moments.append(LoudMoment(start=round(start, 1), end=round(end, 1), peak_db=round(peak, 1), jump_db=round(jump, 1)))

    cur_start, cur_end = flagged[0][0], flagged[0][0]
    cur_peak, cur_jump = flagged[0][1], flagged[0][2]
    for idx, peak, jump in flagged[1:]:
        if (idx - cur_end) * window_seconds <= merge_gap:
            cur_end = idx
            cur_peak = max(cur_peak, peak)
            cur_jump = max(cur_jump, jump)
        else:
            flush(cur_start, cur_end, cur_peak, cur_jump)
            cur_start, cur_end, cur_peak, cur_jump = idx, idx, peak, jump
    flush(cur_start, cur_end, cur_peak, cur_jump)

    # Keep only the loudest handful -- this is a hint appended to the
    # selection prompt, not a candidate filter, so it shouldn't balloon the
    # prompt on a video with lots of ordinary loud moments (music, SFX).
    moments.sort(key=lambda m: m.peak_db, reverse=True)
    moments = moments[:max_moments]
    moments.sort(key=lambda m: m.start)
    return moments


def find_loud_moments(video_path: Path, duration: float) -> List[LoudMoment]:
    """Best-effort: any failure (ffmpeg missing an audio track, a corrupt
    file, an extraction timeout) degrades to "no loud-moment signal" instead
    of failing the job -- this is a hint for select_clips, not a required
    step."""
    wav_path = None
    try:
        wav_path = _extract_mono_wav(video_path, duration)
        levels = _window_db_levels(wav_path)
        return _spike_moments(levels)
    except Exception as e:
        print(f"[loud_moments] detection failed, continuing without it: {e}", flush=True)
        return []
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
