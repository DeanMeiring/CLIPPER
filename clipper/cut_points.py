"""Turn the model's rough clip times into exact cuts on word boundaries.

The selection prompt only shows the model a [mm:ss] marker every 10-15
seconds, so its start/end numbers are interpolated guesses -- often a few
seconds off, which is the difference between a clip that opens on its hook
and one that opens mid-sentence or on dead air. The first second or two is
where Shorts viewers decide to stay or swipe, so the cut is placed from the
words instead: the model also quotes the first and last words it wants in
the clip, those are found in the word-level transcript near its rough
times, and the clip starts just before the first word and ends just after
the last one. When a quote can't be found, the rough time is snapped to the
nearest phrase boundary rather than used as-is.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .transcribe import Word

# Start this long before the first word so its onset isn't clipped...
LEAD_SECONDS = 0.12
# ...and end this long after the last word so the payoff isn't cut off
# mid-breath (never into the next word, though).
TAIL_SECONDS = 0.5
# How far from the model's rough time a quoted phrase may be found. The
# markers it interpolates between are 10-15s apart.
SEARCH_RADIUS = 20.0
# Minimum similarity for a quoted phrase to count as found -- loose enough
# for a changed word or two, strict enough not to latch onto a different line.
MIN_MATCH = 0.6
# Quotes longer than this are cut down: the first/last few words pin the
# spot just as well, and a long quote is likelier to drift from the text.
MAX_QUOTE_WORDS = 6
# A gap this long before a word (or sentence-ending punctuation) marks it as
# the start of a phrase, the preferred place to snap an unanchored cut.
PHRASE_GAP = 0.35
SNAP_RADIUS = 2.0

_SENTENCE_END = re.compile(r"[.!?…]['\"]?$")


@dataclass
class Cut:
    start: float
    end: float
    # False when the quoted words weren't found and the rough time was
    # snapped to a nearby phrase boundary instead.
    anchored_start: bool
    anchored_end: bool


def _norm(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _quote_tokens(quote: Optional[str]) -> List[str]:
    return [t for t in (_norm(p) for p in str(quote or "").split()) if t]


def _token_seq(words: List[Word]) -> List[Tuple[int, str]]:
    """(word index, normalized text), skipping words that are only punctuation."""
    return [(i, t) for i, t in ((i, _norm(w.text)) for i, w in enumerate(words)) if t]


def find_first_word(words: List[Word], quote: Optional[str], near: float) -> Optional[int]:
    """Index of the word where `quote` (the clip's first few words) begins,
    searching within SEARCH_RADIUS of `near`; the closest of equally good
    matches wins, so a phrase said twice resolves to the intended one."""
    q = _quote_tokens(quote)[:MAX_QUOTE_WORDS]
    if not q:
        return None
    seq = _token_seq(words)
    best = None
    for pos in range(len(seq) - len(q) + 1):
        i = seq[pos][0]
        distance = abs(words[i].start - near)
        if distance > SEARCH_RADIUS:
            continue
        score = SequenceMatcher(None, [t for _, t in seq[pos:pos + len(q)]], q).ratio()
        if score >= MIN_MATCH and (best is None or (score, -distance) > best[0]):
            best = ((score, -distance), i)
    return best[1] if best else None


def find_last_word(words: List[Word], quote: Optional[str], near: float, after: int = 0) -> Optional[int]:
    """Index of the word where `quote` (the clip's last few words) ends, at
    or after word index `after`, searching within SEARCH_RADIUS of `near`."""
    q = _quote_tokens(quote)[-MAX_QUOTE_WORDS:]
    if not q:
        return None
    seq = _token_seq(words)
    best = None
    for pos in range(len(q) - 1, len(seq)):
        j = seq[pos][0]
        if j < after:
            continue
        distance = abs(words[j].end - near)
        if distance > SEARCH_RADIUS:
            continue
        score = SequenceMatcher(None, [t for _, t in seq[pos - len(q) + 1:pos + 1]], q).ratio()
        if score >= MIN_MATCH and (best is None or (score, -distance) > best[0]):
            best = ((score, -distance), j)
    return best[1] if best else None


def _starts_phrase(words: List[Word], i: int) -> bool:
    if i == 0:
        return True
    prev = words[i - 1]
    return words[i].start - prev.end >= PHRASE_GAP or bool(_SENTENCE_END.search(prev.text))


def _ends_phrase(words: List[Word], j: int) -> bool:
    if j == len(words) - 1:
        return True
    return words[j + 1].start - words[j].end >= PHRASE_GAP or bool(_SENTENCE_END.search(words[j].text))


def snap_first_word(words: List[Word], rough: float) -> Optional[int]:
    """Best first word for a cut the model only placed roughly: the nearest
    phrase start within SNAP_RADIUS, else the word being said at `rough`
    (or the next one, if `rough` falls in a gap)."""
    phrase_starts = [
        i for i, w in enumerate(words)
        if abs(w.start - rough) <= SNAP_RADIUS and _starts_phrase(words, i)
    ]
    if phrase_starts:
        return min(phrase_starts, key=lambda i: abs(words[i].start - rough))
    return next((i for i, w in enumerate(words) if w.end > rough), None)


def snap_last_word(words: List[Word], rough: float, after: int = 0) -> Optional[int]:
    phrase_ends = [
        j for j in range(after, len(words))
        if abs(words[j].end - rough) <= SNAP_RADIUS and _ends_phrase(words, j)
    ]
    if phrase_ends:
        return min(phrase_ends, key=lambda j: abs(words[j].end - rough))
    before = [j for j in range(after, len(words)) if words[j].start < rough]
    return before[-1] if before else None


def start_before(words: List[Word], i: int) -> float:
    """Cut point just before word i, without taking in the end of word i-1."""
    first = words[i]
    prev_end = words[i - 1].end if i > 0 else 0.0
    return max(0.0, min(first.start, max(prev_end, first.start - LEAD_SECONDS)))


def end_after(words: List[Word], j: int, duration: float) -> float:
    """Cut point just after word j, stopping short of word j+1."""
    last = words[j]
    limit = words[j + 1].start - 0.05 if j + 1 < len(words) else duration
    return min(duration, max(last.end, min(limit, last.end + TAIL_SECONDS)))


def refine_cut(
    words: List[Word],
    rough_start: float,
    rough_end: float,
    start_words: Optional[str],
    end_words: Optional[str],
    duration: float,
    min_len: float,
    max_len: float,
) -> Optional[Cut]:
    """Exact cut points for one pick, or None when it can't make a clip of
    at least (roughly) min_len. max_len is enforced on a word boundary --
    see select_moments for why it's a hard ceiling."""
    if not words or rough_end <= rough_start:
        return None

    i = find_first_word(words, start_words, rough_start)
    anchored_start = i is not None
    if i is None:
        i = snap_first_word(words, rough_start)
    if i is None:
        return None

    j = find_last_word(words, end_words, rough_end, after=i)
    anchored_end = j is not None
    if j is None:
        j = snap_last_word(words, rough_end, after=i)
    if j is None:
        return None

    start = start_before(words, i)
    end = end_after(words, j, duration)
    if end - start > max_len:
        fitting = [k for k in range(i, j) if end_after(words, k, duration) - start <= max_len]
        end = end_after(words, fitting[-1], duration) if fitting else start + max_len
        anchored_end = False
    if end - start < min_len * 0.6:  # a little slack vs. exact min_len
        return None
    # Rounded outward: the render keeps only words fully inside [start, end],
    # so rounding start up past the first word's onset would drop its caption.
    start = math.floor(start * 100) / 100
    end = min(duration, math.ceil(end * 100) / 100)
    return Cut(start, end, anchored_start, anchored_end)
