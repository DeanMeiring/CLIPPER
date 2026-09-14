"""Build a burned-in, word-highlight ("karaoke") .ass subtitle file for one clip."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .transcribe import Word

# Every style value below (font size, outline, margins) was tuned by eye
# for a 1080x1920 portrait Short. build_ass() scales them by the target
# frame's height relative to this base, so a caption still reads as "the
# same size" proportionally on a landscape recap (see PLAY_RES below)
# instead of looking oversized/undersized just because the frame changed
# shape.
_BASE_PLAY_RES: Tuple[int, int] = (1080, 1920)
_BASE_FONTSIZE = 72
_BASE_OUTLINE = 6
_BASE_MARGIN_LR = 60
_BASE_MARGIN_V = 220

# The rank-badge overlay (see rank_badge_dialogue) is a small persistent
# top-left label, not a spoken caption -- tuned much smaller than the
# karaoke captions above so it reads as a corner badge, not competing
# text. Scaled by the same height ratio as the caption style so it stays
# proportionally sized on any output resolution.
_BASE_BADGE_FONTSIZE = 44
_BASE_BADGE_OUTLINE = 14  # box padding, via BorderStyle 3 below
_BASE_BADGE_MARGIN = 40


def _ass_header(play_res: Tuple[int, int] = _BASE_PLAY_RES) -> str:
    width, height = play_res
    scale = height / _BASE_PLAY_RES[1]
    fontsize = max(1, round(_BASE_FONTSIZE * scale))
    outline = max(1, round(_BASE_OUTLINE * scale))
    margin_lr = max(0, round(_BASE_MARGIN_LR * scale))
    margin_v = max(0, round(_BASE_MARGIN_V * scale))
    badge_fontsize = max(1, round(_BASE_BADGE_FONTSIZE * scale))
    badge_outline = max(1, round(_BASE_BADGE_OUTLINE * scale))
    badge_margin = max(0, round(_BASE_BADGE_MARGIN * scale))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,{fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,{margin_lr},{margin_lr},{margin_v},1
Style: RankBadge,Arial Black,{badge_fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,3,{badge_outline},0,7,{badge_margin},{badge_margin},{badge_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


# Word text is burned in as literal caption text between \k override tags.
# A literal "{" or "}" from the source transcript would otherwise
# prematurely close/open an override block -- breaking the \k word-highlight
# timing for the rest of that line -- and ASS has no escape sequence for a
# literal brace (or backslash, which can chain into the \N/\n/\h sequences
# ASS recognizes directly in dialogue text) in the Text field. Swap them for
# visually similar full-width characters instead of silently dropping them.
_ASS_UNSAFE_CHARS = {
    "\\": "＼",  # fullwidth reverse solidus
    "{": "｛",   # fullwidth left curly bracket
    "}": "｝",   # fullwidth right curly bracket
}


def _escape_ass_text(text: str) -> str:
    for bad, safe in _ASS_UNSAFE_CHARS.items():
        text = text.replace(bad, safe)
    return text


def rank_badge_dialogue(text: str, duration: float) -> str:
    """One static top-left-anchored Dialogue line spanning a clip's whole
    duration, using the RankBadge style from _ass_header. Meant to be
    appended into the same .ass file as the clip's own word-by-word
    captions (see build_ass) rather than written to a separate file, so
    both burn in during the same ffmpeg pass."""
    return f"Dialogue: 0,{_fmt_ts(0)},{_fmt_ts(duration)},RankBadge,,0,0,0,,{_escape_ass_text(text)}"


def _group_words(words: List[Word], max_words: int = 4, max_span: float = 2.2) -> List[List[Word]]:
    groups: List[List[Word]] = []
    cur: List[Word] = []
    cur_start = None
    for w in words:
        if cur_start is None:
            cur_start = w.start
        if cur and (len(cur) >= max_words or (w.end - cur_start) > max_span):
            groups.append(cur)
            cur = []
            cur_start = w.start
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(
    clip_words: List[Word], clip_start: float, output_path: Path,
    play_res: Tuple[int, int] = _BASE_PLAY_RES,
) -> Path:
    """clip_words are absolute-time Word objects that fall within the clip;
    clip_start is subtracted so the .ass timeline starts at 0 for this clip.

    play_res should match the actual output frame size (width, height) --
    it's both the coordinate system the Style/Dialogue margins below are
    written in AND (via _ass_header's scaling) what decides how big those
    margins and the font actually are. Defaults to the normal portrait
    Short's 1080x1920; pass the real target for anything else (e.g. a
    landscape recap) so captions land where the style values expect them
    rather than however libass happens to stretch a mismatched canvas."""
    lines = [_ass_header(play_res)]
    for group in _group_words(clip_words):
        g_start = group[0].start - clip_start
        g_end = group[-1].end - clip_start
        # \k tags highlight one word at a time (centiseconds per word)
        parts = []
        for w in group:
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            parts.append(f"{{\\k{dur_cs}}}{_escape_ass_text(w.text.upper())} ")
        text = "".join(parts).strip()
        lines.append(
            f"Dialogue: 0,{_fmt_ts(g_start)},{_fmt_ts(g_end)},Caption,,0,0,0,,{text}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
