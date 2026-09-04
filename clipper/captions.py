"""Build a burned-in, word-highlight ("karaoke") .ass subtitle file for one clip."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .transcribe import Word

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,72,&H00FFFFFF,&H0000D7FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


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


def build_ass(clip_words: List[Word], clip_start: float, output_path: Path) -> Path:
    """clip_words are absolute-time Word objects that fall within the clip;
    clip_start is subtracted so the .ass timeline starts at 0 for this clip."""
    lines = [ASS_HEADER]
    for group in _group_words(clip_words):
        g_start = group[0].start - clip_start
        g_end = group[-1].end - clip_start
        # \k tags highlight one word at a time (centiseconds per word)
        parts = []
        for w in group:
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            parts.append(f"{{\\k{dur_cs}}}{w.text.upper()} ")
        text = "".join(parts).strip()
        lines.append(
            f"Dialogue: 0,{_fmt_ts(g_start)},{_fmt_ts(g_end)},Caption,,0,0,0,,{text}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
