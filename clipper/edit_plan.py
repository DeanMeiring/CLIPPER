"""Pacing edits for one clip: cut dead air, and optionally open on a flash
of the payoff. (Punch-in zooms on reactions exist below but are off.)

A clip is still picked and cut as one continuous stretch of the source
(select_moments / cut_points). This decides what happens inside it: which
pauses to drop, where the audio spikes deserve a zoom, and which moment to
tease up front. The result is an EditPlan -- an ordered list of pieces of
the clip, in clip-local seconds (0 = the clip's cut start) -- which
render.render_edited turns into the final video, and remap_words uses to
re-time the captions onto the edited timeline.

Everything here is measured, not guessed: pauses come from the word timings
and are only cut when the audio in them is quiet (a scream the transcript
missed is exactly what shouldn't be cut), and zooms land on audio peaks.
"""
from __future__ import annotations

import array
import statistics
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from math import log10
from pathlib import Path
from typing import List, Optional, Tuple

from .transcribe import Word

# Loudness is measured in windows this long -- fine enough to tell a quiet
# half-second pause from one with a shout in it.
WINDOW = 0.1
_SAMPLE_RATE = 16000
_SILENCE_DB = -60.0

# A pause between words at least this long is dead air worth cutting...
MIN_CUT_GAP = 0.6
# ...down to this much breathing room on each side of the cut, so words
# don't slam into each other.
KEEP_AFTER_WORD = 0.12
KEEP_BEFORE_WORD = 0.1
# A pause only counts as dead air if its audio stays this far under the
# clip's typical level; any window this far ABOVE it means something is
# happening (a reaction, a game moment) and the pause is kept whole.
QUIET_BELOW_MEDIAN_DB = 3.0
EVENT_ABOVE_MEDIAN_DB = 6.0

# Audio peaks this far above the clip's typical level get a punch-in zoom.
ZOOM_PEAK_DB = 8.0
ZOOM = 1.15
ZOOM_BEFORE = 0.25
ZOOM_AFTER = 1.25
# Off: on real clips the in-and-out punch every few seconds read as a
# glitch, not an edit. The machinery stays in case a subtler version is
# wanted later.
MAX_ZOOMS = 0
MIN_ZOOM_SPACING = 4.0
# The opening belongs to the hook text and the first words; don't zoom in
# before the viewer has even registered the shot.
NO_ZOOM_BEFORE = 1.0

# The teaser: this much of the loudest moment in the back part of the clip,
# played first.
TEASER_BEFORE = 0.5
TEASER_AFTER = 0.9
TEASER_SEARCH_FROM = 0.4  # fraction of the clip -- a payoff isn't in its opening
MIN_CLIP_FOR_TEASER = 12.0


@dataclass
class Piece:
    start: float
    end: float
    zoom: float = 1.0

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class EditPlan:
    pieces: List[Piece]
    # True when pieces[0] is a teaser replayed out of order.
    teaser: bool = False
    cut_seconds: float = 0.0
    zooms: int = 0
    # Clip-local times of the cuts, for the log line and the record.
    cuts: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(p.length for p in self.pieces)

    def is_trivial(self, clip_duration: float) -> bool:
        return (len(self.pieces) == 1 and not self.teaser and self.pieces[0].zoom == 1.0
                and self.pieces[0].start <= 0.001 and abs(self.pieces[0].end - clip_duration) <= 0.001)

    def to_dict(self) -> dict:
        return {
            "pieces": [asdict(p) for p in self.pieces], "teaser": self.teaser,
            "cut_seconds": round(self.cut_seconds, 2), "zooms": self.zooms,
            "cuts": [list(c) for c in self.cuts],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["EditPlan"]:
        if not data or not data.get("pieces"):
            return None
        return cls(
            pieces=[Piece(float(p["start"]), float(p["end"]), float(p.get("zoom", 1.0))) for p in data["pieces"]],
            teaser=bool(data.get("teaser")), cut_seconds=float(data.get("cut_seconds") or 0.0),
            zooms=int(data.get("zooms") or 0), cuts=[tuple(c) for c in data.get("cuts") or []],
        )


def clip_levels(video_path: Path, start: float, end: float) -> List[float]:
    """RMS loudness (dBFS) per WINDOW across [start, end] of the source.
    Empty on any failure -- the plan then just skips the audio-driven edits."""
    fd_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            fd_path = Path(tmp.name)
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{max(0.1, end - start):.3f}",
            "-vn", "-ac", "1", "-ar", str(_SAMPLE_RATE), "-f", "wav", str(fd_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        levels: List[float] = []
        per_window = int(_SAMPLE_RATE * WINDOW)
        with wave.open(str(fd_path), "rb") as wf:
            while True:
                raw = wf.readframes(per_window)
                if not raw:
                    break
                samples = array.array("h")
                samples.frombytes(raw)
                if sys.byteorder == "big":
                    samples.byteswap()
                if not samples:
                    break
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
                levels.append(_SILENCE_DB if rms <= 1.0 else max(_SILENCE_DB, 20 * log10(rms / 32768.0)))
        return levels
    except Exception as e:  # noqa: BLE001 - pacing edits are optional; the clip still renders
        print(f"[edit_plan] couldn't measure audio for {video_path.name} {start:.1f}-{end:.1f}s: {e}", flush=True)
        return []
    finally:
        if fd_path is not None:
            fd_path.unlink(missing_ok=True)


def _window_range(levels: List[float], a: float, b: float) -> List[float]:
    i, j = max(0, int(a / WINDOW)), min(len(levels), int(b / WINDOW) + 1)
    return levels[i:j]


def _dead_air(words: List[Word], clip_start: float, levels: List[float], median: float) -> List[Tuple[float, float]]:
    """Clip-local (from, to) spans to remove: quiet pauses between words."""
    cuts = []
    for prev, nxt in zip(words, words[1:]):
        gap_from, gap_to = prev.end - clip_start, nxt.start - clip_start
        if gap_to - gap_from < MIN_CUT_GAP:
            continue
        window = _window_range(levels, gap_from, gap_to)
        if not window:
            continue
        if statistics.mean(window) > median - QUIET_BELOW_MEDIAN_DB or max(window) >= median + EVENT_ABOVE_MEDIAN_DB:
            continue
        cut_from, cut_to = gap_from + KEEP_AFTER_WORD, gap_to - KEEP_BEFORE_WORD
        if cut_to - cut_from > 0.2:
            cuts.append((cut_from, cut_to))
    return cuts


def _peaks(levels: List[float], median: float, duration: float) -> List[float]:
    """Clip-local times of distinct audio spikes, loudest first."""
    loud = [i for i, db in enumerate(levels) if db >= median + ZOOM_PEAK_DB]
    groups: List[List[int]] = []
    for i in loud:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])
    peaks = [max(g, key=lambda i: levels[i]) for g in groups]
    peaks.sort(key=lambda i: levels[i], reverse=True)
    return [min(duration, (i + 0.5) * WINDOW) for i in peaks]


def _split_for_zooms(pieces: List[Piece], zooms: List[Tuple[float, float]]) -> List[Piece]:
    """Cut each piece at zoom boundaries and mark the zoomed stretches."""
    out: List[Piece] = []
    for p in pieces:
        points = sorted({p.start, p.end, *[t for z in zooms for t in z if p.start < t < p.end]})
        for a, b in zip(points, points[1:]):
            if b - a < 0.05:
                continue
            mid = (a + b) / 2
            zoomed = any(z0 <= mid < z1 for z0, z1 in zooms)
            out.append(Piece(round(a, 3), round(b, 3), ZOOM if zoomed else 1.0))
    return out


def plan_edit(
    clip_words: List[Word], clip_start: float, clip_end: float, levels: List[float],
    teaser: bool = False, max_len: Optional[float] = None,
) -> EditPlan:
    """The pacing edits for one clip. clip_words are the source-timed words
    inside [clip_start, clip_end]; levels come from clip_levels over the
    same span. With no audio levels, nothing is cut or zoomed."""
    duration = clip_end - clip_start
    if not levels or duration <= 0:
        return EditPlan([Piece(0.0, round(duration, 3))])
    median = statistics.median(levels)

    cuts = _dead_air(clip_words, clip_start, levels, median)
    pieces: List[Piece] = []
    cursor = 0.0
    for a, b in cuts:
        pieces.append(Piece(round(cursor, 3), round(a, 3)))
        cursor = b
    pieces.append(Piece(round(cursor, 3), round(duration, 3)))
    pieces = [p for p in pieces if p.length > 0.05]
    cut_seconds = sum(b - a for a, b in cuts)

    def kept(t: float) -> bool:
        return any(p.start <= t < p.end for p in pieces)

    zoom_windows: List[Tuple[float, float]] = []
    for t in _peaks(levels, median, duration):
        if len(zoom_windows) >= MAX_ZOOMS:
            break
        if t < NO_ZOOM_BEFORE or not kept(t) or any(abs(t - (z0 + ZOOM_BEFORE)) < MIN_ZOOM_SPACING for z0, _ in zoom_windows):
            continue
        zoom_windows.append((max(0.0, t - ZOOM_BEFORE), min(duration, t + ZOOM_AFTER)))
    pieces = _split_for_zooms(pieces, zoom_windows)

    plan = EditPlan(pieces, cut_seconds=cut_seconds, zooms=len(zoom_windows), cuts=[(round(a, 2), round(b, 2)) for a, b in cuts])

    if teaser and duration >= MIN_CLIP_FOR_TEASER:
        search_from = duration * TEASER_SEARCH_FROM
        candidates = [t for t in _peaks(levels, median, duration) if t >= search_from and kept(t)]
        if candidates:
            t = candidates[0]
            piece = Piece(round(max(0.0, t - TEASER_BEFORE), 3), round(min(duration, t + TEASER_AFTER), 3))
            if max_len is None or plan.duration + piece.length <= max_len:
                plan.pieces.insert(0, piece)
                plan.teaser = True
    return plan


def remap_words(clip_words: List[Word], clip_start: float, plan: EditPlan) -> List[Word]:
    """clip_words re-timed onto the edited video's timeline (0 = its first
    frame), ready for captions.build_ass with clip_start=0. A teaser's
    words appear twice -- once in the teaser, once in place -- just as the
    audio does."""
    out: List[Word] = []
    offset = 0.0
    for i, piece in enumerate(plan.pieces):
        for w in clip_words:
            ws, we = w.start - clip_start, w.end - clip_start
            if ws >= piece.start - 1e-6 and we <= piece.end + 1e-6:
                out.append(Word(w.text, round(offset + ws - piece.start, 3), round(offset + we - piece.start, 3)))
            elif ws < piece.end and we > piece.start and not (plan.teaser and i == 0):
                # A word straddling a zoom boundary: the next piece continues
                # it seamlessly, so place it by its start and keep its length.
                if piece.start <= ws < piece.end:
                    out.append(Word(w.text, round(offset + ws - piece.start, 3), round(offset + ws - piece.start + (we - ws), 3)))
        offset += piece.length
    out.sort(key=lambda w: w.start)
    return out
