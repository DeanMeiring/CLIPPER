"""Measurable traits of a rendered clip -- how fast it gets to speech, how
dense the talking is up front, how long it goes without anyone speaking,
and whether it opens loud. Only measurements, no judgment calls: these are
the inputs clip_performance.py checks against the channel's own retention
data, so they have to be facts about the clip, not an opinion of it.

Transcript gaps are "no transcribed speech", not silence -- gameplay audio,
music, and wordless reactions all fall in a gap. That's the right thing to
measure for streamer clips (a stretch where nobody's talking is where
viewers drift), but it isn't the same as dead air.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import List

from .loud_moments import _WINDOW_SECONDS, _extract_mono_wav, _window_db_levels
from .transcribe import Word

_OPENING_SECONDS = 3.0
_OPENING_AUDIO_SECONDS = 2.0
# A pause this long between words reads as a lull on screen, not a breath.
_LULL_SECONDS = 0.7


def speech_features(clip_words: List[Word], start: float, end: float) -> dict:
    """Timing features from the clip's own transcript words (absolute
    source-video times, the same list build_ass gets)."""
    duration = max(0.0, end - start)
    words = sorted((w for w in clip_words if w.end > start and w.start < end), key=lambda w: w.start)
    if not words:
        return {
            "duration": round(duration, 2),
            "first_word_delay": None,
            "words_per_sec_opening": 0.0,
            "words_per_sec": 0.0,
            "longest_speech_gap": round(duration, 2),
            "lull_seconds": round(duration, 2),
        }

    first_word_delay = max(0.0, words[0].start - start)
    opening_span = max(min(_OPENING_SECONDS, duration), 0.1)
    opening_words = sum(1 for w in words if w.start < start + opening_span)
    gaps = [first_word_delay]
    gaps += [max(0.0, b.start - a.end) for a, b in zip(words, words[1:])]
    gaps.append(max(0.0, end - words[-1].end))
    return {
        "duration": round(duration, 2),
        "first_word_delay": round(first_word_delay, 2),
        "words_per_sec_opening": round(opening_words / opening_span, 2),
        "words_per_sec": round(len(words) / max(duration, 0.1), 2),
        "longest_speech_gap": round(max(gaps), 2),
        "lull_seconds": round(sum(g for g in gaps if g >= _LULL_SECONDS), 2),
    }


def audio_features(video_path: Path, duration: float) -> dict:
    """Loudness of the clip's opening against the rest of the clip, and
    where its loudest moment sits. Empty dict if the audio can't be read --
    a missing measurement, not a failed render."""
    wav_path = None
    try:
        wav_path = _extract_mono_wav(video_path, duration)
        levels = _window_db_levels(wav_path)
    except Exception as e:  # noqa: BLE001 - a feature measurement must never fail a render
        print(f"[clip_features] audio measurement failed for {video_path.name}: {e}", flush=True)
        return {}
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    if not levels:
        return {}

    opening = levels[: max(1, int(round(_OPENING_AUDIO_SECONDS / _WINDOW_SECONDS)))]
    median = statistics.median(levels)
    peak_index = max(range(len(levels)), key=lambda i: levels[i])
    return {
        "opening_vs_median_db": round(statistics.mean(opening) - median, 1),
        "peak_vs_median_db": round(levels[peak_index] - median, 1),
        "peak_position": round(peak_index / max(len(levels) - 1, 1), 2),
    }


def measure_clip(clip_words: List[Word], start: float, end: float, rendered_path: Path) -> dict:
    features = speech_features(clip_words, start, end)
    features.update(audio_features(rendered_path, end - start))
    return features


_HASHTAG_RE = re.compile(r"#\w+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _is_emoji(ch: str) -> bool:
    code = ord(ch)
    return 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF


def title_features(title: str) -> dict:
    """Traits of the title as actually posted. Hashtags are stripped first:
    every upload carries #Shorts, so counting it would only add noise."""
    text = _HASHTAG_RE.sub("", title or "").strip()
    words = _WORD_RE.findall(text)
    caps_words = [w for w in words if len(w) >= 2 and w.isupper()]
    return {
        "title_length": len(text),
        "title_is_question": "?" in text,
        "title_caps_words": len(caps_words),
        "title_has_emoji": any(_is_emoji(ch) for ch in text),
    }
