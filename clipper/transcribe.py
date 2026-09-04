"""Produce a flat, word-level transcript: [{text, start, end}, ...].

Two sources, in order of preference:
  1. YouTube's own captions (fast, free) -- word timing is *approximated* by
     spreading each caption cue evenly across its [start, end] window, which
     is good enough for on-screen captions and moment-selection.
  2. Local Whisper (faster-whisper) -- real word-level timestamps, slower and
     needs a one-time model download, but noticeably better for the
     karaoke-highlight caption style. Used automatically when there are no
     YouTube captions, or when the caller passes --whisper.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Word:
    text: str
    start: float
    end: float


_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)")


def _parse_ts(ts: str) -> float:
    m = _TIME_RE.search(ts)
    if not m:
        raise ValueError(f"Bad timestamp: {ts}")
    h, mi, s, ms = m.groups()
    ms = (ms + "000")[:3]
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0


def words_from_vtt(vtt_path: Path) -> List[Word]:
    """Parse a WebVTT file into an approximate flat word list."""
    text = vtt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\n+", text.strip())
    words: List[Word] = []
    last_end = -1.0

    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        cue_line = next((ln for ln in lines if "-->" in ln), None)
        if not cue_line:
            continue
        try:
            start_s, end_s = [t.strip().split(" ")[0] for t in cue_line.split("-->")]
            start, end = _parse_ts(start_s), _parse_ts(end_s)
        except ValueError:
            continue

        # skip near-duplicate cues auto-captions often emit (rolling text)
        if start <= last_end - 0.05:
            continue

        body_lines = lines[lines.index(cue_line) + 1:]
        body = " ".join(body_lines)
        body = html.unescape(body)  # &gt;&gt; -> >>, &amp; -> &, etc.
        body = re.sub(r"<[^>]+>", "", body)  # strip <c> timing tags
        body = re.sub(r"\[.*?\]", "", body)  # strip [Music] etc.
        body = re.sub(r">{1,2}", "", body)  # strip >> speaker-change markers
        toks = [t for t in body.split() if t.strip()]
        if not toks:
            continue

        span = max(end - start, 0.2)
        step = span / len(toks)
        for i, tok in enumerate(toks):
            w_start = start + i * step
            w_end = w_start + step
            words.append(Word(text=tok, start=round(w_start, 3), end=round(w_end, 3)))
        last_end = end

    return words


def whisper_transcribe(video_path: Path, model_size: str = "small", language: Optional[str] = None) -> List[Word]:
    """Transcribe locally with faster-whisper. Requires: pip install faster-whisper"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --whisper. Install it with: pip install faster-whisper"
        ) from e

    model = WhisperModel(model_size, compute_type="int8")
    segments, _info = model.transcribe(str(video_path), word_timestamps=True, language=language)

    words: List[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            token = w.word.strip()
            if token:
                words.append(Word(text=token, start=round(w.start, 3), end=round(w.end, 3)))
    return words


def get_transcript(
    video_path: Path,
    captions_path: Optional[Path],
    prefer_whisper: bool = False,
    whisper_model: str = "small",
) -> List[Word]:
    if not prefer_whisper and captions_path and captions_path.exists():
        words = words_from_vtt(captions_path)
        if words:
            return words
    return whisper_transcribe(video_path, model_size=whisper_model)
