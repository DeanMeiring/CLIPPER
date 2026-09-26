"""Build a burned-in, word-highlight ("karaoke") .ass subtitle file for one clip."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

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

# The rank-badge overlay (see rank_badge_dialogue) is a persistent
# top-left label, not a spoken caption -- still noticeably smaller than
# the karaoke captions above so it reads as a corner badge, not competing
# text, but sized (and boxed -- see the near-opaque BackColour on the
# RankBadge style below) to actually read clearly over busy gameplay
# footage rather than blend into it. Scaled by the same height ratio as
# the caption style so it stays proportionally sized on any output
# resolution.
_BASE_BADGE_FONTSIZE = 62
_BASE_BADGE_OUTLINE = 22  # box padding, via BorderStyle 3 below
_BASE_BADGE_MARGIN = 40

# The flash-hook line (see hook_line_ass, clipper/hook_line.py) is shown
# dead-center and only for a fraction of a second, so it needs to read
# instantly -- bigger than the karaoke captions above, which have a full
# couple of seconds on screen per group to be read.
_BASE_HOOK_FONTSIZE = 92
_BASE_HOOK_OUTLINE = 10
_BASE_HOOK_MARGIN_LR = 90

# The hook text (see build_ass's hook_text) sits in a white box near the top
# for a clip's first few seconds -- where Shorts viewers decide whether to
# swipe -- clear of the bottom captions and of the Shorts UI along the
# bottom edge. Boxed rather than outlined so it stays readable over any
# gameplay, and a different look from the captions so it doesn't read as
# one more spoken line.
HOOK_TEXT_SECONDS = 3.0

# Short-form captions (build_ass's punchy mode): a few words at a time,
# bigger, each group popping in as it's spoken -- the fast-changing text
# Shorts viewers are used to, instead of calmer four-word lines.
_BASE_POP_FONTSIZE = 86
_BASE_POP_OUTLINE = 7
_POP_MAX_WORDS = 3
_POP_MAX_SPAN = 1.0
_POP_IN = "{\\fscx75\\fscy75\\t(0,90,\\fscx100\\fscy100)}"
_BASE_HOOK_TEXT_FONTSIZE = 64
_BASE_HOOK_TEXT_BOX = 18  # box padding, via BorderStyle 3 below
_BASE_HOOK_TEXT_MARGIN_LR = 80
_BASE_HOOK_TEXT_MARGIN_V = 250

# Channel branding (build_ass's brand): the channel's mascot -- the shocked
# face from its profile picture -- in the top-left corner for the whole
# clip, with the channel name stamped beside it while the hook text is up.
# Most Shorts views come from people who never open the channel page, so
# the only place they can learn to recognise the channel is inside the
# videos. Top-left, above the hook text box, is clear of the Shorts UI
# (buttons down the right, title and channel name along the bottom).
BRAND_NAME = os.environ.get("CLIPPER_BRAND_NAME", "Caught On Stream")
_BASE_MASCOT_SIZE = 132
_BASE_MASCOT_X = 44
_BASE_MASCOT_Y = 80
_BASE_STAMP_FONTSIZE = 40
_BASE_STAMP_BOX = 12
_STAMP_SECONDS = HOOK_TEXT_SECONDS
# Drawn until the video ends, however long the edited clip turns out to be.
_BRAND_END = 9 * 3600.0


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
    hook_fontsize = max(1, round(_BASE_HOOK_FONTSIZE * scale))
    hook_outline = max(1, round(_BASE_HOOK_OUTLINE * scale))
    hook_margin_lr = max(0, round(_BASE_HOOK_MARGIN_LR * scale))
    hook_text_fontsize = max(1, round(_BASE_HOOK_TEXT_FONTSIZE * scale))
    hook_text_box = max(1, round(_BASE_HOOK_TEXT_BOX * scale))
    hook_text_margin_lr = max(0, round(_BASE_HOOK_TEXT_MARGIN_LR * scale))
    hook_text_margin_v = max(0, round(_BASE_HOOK_TEXT_MARGIN_V * scale))
    pop_fontsize = max(1, round(_BASE_POP_FONTSIZE * scale))
    pop_outline = max(1, round(_BASE_POP_OUTLINE * scale))
    stamp_fontsize = max(1, round(_BASE_STAMP_FONTSIZE * scale))
    stamp_box = max(1, round(_BASE_STAMP_BOX * scale))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,{fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{outline},2,2,{margin_lr},{margin_lr},{margin_v},1
Style: RankBadge,Arial Black,{badge_fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H20000000,-1,0,0,0,100,100,0,0,3,{badge_outline},0,7,{badge_margin},{badge_margin},{badge_margin},1
Style: HookLine,Arial Black,{hook_fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{hook_outline},4,5,{hook_margin_lr},{hook_margin_lr},0,1
Style: CaptionPop,Arial Black,{pop_fontsize},&H00FFFFFF,&H0000D7FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{pop_outline},2,2,{margin_lr},{margin_lr},{margin_v},1
Style: HookText,Arial Black,{hook_text_fontsize},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,{hook_text_box},0,8,{hook_text_margin_lr},{hook_text_margin_lr},{hook_text_margin_v},1
Style: Stamp,Arial Black,{stamp_fontsize},&H00FFFFFF,&H00FFFFFF,&H001B12E3,&H001B12E3,-1,0,0,0,100,100,1,0,3,{stamp_box},0,4,0,0,0,1
Style: Mascot,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00111111,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

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


# Color emoji need a font the render container doesn't have, so they'd burn
# in as empty boxes.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+")


def clean_hook_text(text: Optional[str]) -> str:
    """The hook caption as it can actually be burned in: emoji removed,
    whitespace collapsed. Empty when nothing usable is left."""
    return " ".join(_EMOJI.sub(" ", text or "").split())


def rank_badge_dialogue(text: str, duration: float) -> str:
    """One static top-left-anchored Dialogue line spanning a clip's whole
    duration, using the RankBadge style from _ass_header. Meant to be
    appended into the same .ass file as the clip's own word-by-word
    captions (see build_ass) rather than written to a separate file, so
    both burn in during the same ffmpeg pass."""
    return f"Dialogue: 0,{_fmt_ts(0)},{_fmt_ts(duration)},RankBadge,,0,0,0,,{_escape_ass_text(text)}"


def hook_line_ass(
    text: str, flash_seconds: float, play_res: Tuple[int, int] = _BASE_PLAY_RES,
) -> str:
    """A standalone .ass file (header + one Dialogue) that flashes `text`
    dead-center on screen for the clip's first `flash_seconds`, then
    disappears -- the "spoil the payoff up front" hook line clip channels
    use (see clipper/hook_line.py). Written as its own small file and
    burned onto an ALREADY-RENDERED clip in a second ffmpeg pass, rather
    than appended into that clip's original build_ass output, so the hook
    line can be generated, previewed, and re-rendered on its own without
    re-running the original render."""
    lines = [_ass_header(play_res)]
    lines.append(
        f"Dialogue: 0,{_fmt_ts(0)},{_fmt_ts(flash_seconds)},HookLine,,0,0,0,,{_escape_ass_text(text.upper())}"
    )
    return "\n".join(lines)


# The mascot is drawn with ASS vector shapes (\p1) in a 100x100 design
# square, one Dialogue line per colour, so it burns in with the captions in
# every render path -- including facecam/IRL re-renders from the same file.
# Each shape starts with "m 0 0 m 100 100": libass positions a drawing by
# its bounding box, and pinning every layer's box to the full square keeps
# the layers lined up with each other.
_SQUARE = "m 0 0 m 100 100 "
_BEZIER_K = 0.5523  # control-point distance for a quarter circle


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> str:
    kx, ky = rx * _BEZIER_K, ry * _BEZIER_K
    quarters = [
        [(cx + kx, cy - ry), (cx + rx, cy - ky), (cx + rx, cy)],
        [(cx + rx, cy + ky), (cx + kx, cy + ry), (cx, cy + ry)],
        [(cx - kx, cy + ry), (cx - rx, cy + ky), (cx - rx, cy)],
        [(cx - rx, cy - ky), (cx - kx, cy - ry), (cx, cy - ry)],
    ]
    return f"m {_num(cx)} {_num(cy - ry)} " + " ".join(
        "b " + " ".join(f"{_num(x)} {_num(y)}" for x, y in q) for q in quarters
    )


def _polygon(points: List[Tuple[float, float]]) -> str:
    (x0, y0), rest = points[0], points[1:]
    return f"m {_num(x0)} {_num(y0)} l " + " ".join(f"{_num(x)} {_num(y)}" for x, y in rest)


def _brackets(inset: float = 17, arm: float = 15, thick: float = 5) -> str:
    """The four camera-viewfinder corners."""
    a, L, t = inset, arm, thick
    corner = [(a, a), (a + L, a), (a + L, a + t), (a + t, a + t), (a + t, a + L), (a, a + L)]
    shapes = []
    for fx in (False, True):
        for fy in (False, True):
            shapes.append(_polygon([(100 - x if fx else x, 100 - y if fy else y) for x, y in corner]))
    return " ".join(shapes)


def _brow(p0: Tuple[float, float], q: Tuple[float, float], p2: Tuple[float, float], thick: float = 3.6) -> str:
    """A raised eyebrow: a quadratic arc from p0 over q to p2, thick-wide."""
    h = thick / 2

    def cubic(a, c, b, dy):
        a, c, b = (a[0], a[1] + dy), (c[0], c[1] + dy), (b[0], b[1] + dy)
        c1 = (a[0] + 2 / 3 * (c[0] - a[0]), a[1] + 2 / 3 * (c[1] - a[1]))
        c2 = (b[0] + 2 / 3 * (c[0] - b[0]), b[1] + 2 / 3 * (c[1] - b[1]))
        return a, c1, c2, b

    a, c1, c2, b = cubic(p0, q, p2, -h)
    d, e1, e2, f = cubic(p2, q, p0, h)
    pts = lambda *ps: " ".join(f"{_num(x)} {_num(y)}" for x, y in ps)  # noqa: E731
    return f"m {pts(a)} b {pts(c1, c2, b)} l {pts(d)} b {pts(e1, e2, f)}"


# (layer offset, colour as ASS &HBBGGRR, extra tags, shape)
_MASCOT_LAYERS = [
    (0, "00CCFF", "\\bord1.4\\3c&H111111&\\shad1.2\\4c&H000000&\\4a&H60&", _ellipse(50, 50, 50, 50)),
    (1, "111111", "", " ".join([
        _brackets(),
        _brow((32.75, 30.6), (39, 25.6), (45, 28.9)),
        _brow((55, 28.9), (61, 25.6), (67.25, 30.6)),
        _ellipse(39, 41.25, 8.4, 9.9), _ellipse(61, 41.25, 8.4, 9.9),
        _ellipse(50, 68, 8.25, 11),
    ])),
    (2, "FFFFFF", "", " ".join([_ellipse(39, 41.25, 6.25, 7.75), _ellipse(61, 41.25, 6.25, 7.75)])),
    (3, "111111", "", " ".join([_ellipse(40, 42.9, 3.25, 3.25), _ellipse(60, 42.9, 3.25, 3.25)])),
    (3, "1B12E3", "", _ellipse(50, 69, 5.75, 8.25)),
    (4, "FFFFFF", "\\1a&H80&", _ellipse(48, 64.75, 1.75, 1.4)),
]


def brand_dialogues(play_res: Tuple[int, int] = _BASE_PLAY_RES, name: str = BRAND_NAME) -> List[str]:
    """The channel mascot (whole clip, top-left) and a name stamp beside it
    (first _STAMP_SECONDS), as Dialogue lines for the clip's .ass file."""
    scale = play_res[1] / _BASE_PLAY_RES[1]
    size = _BASE_MASCOT_SIZE * scale
    cx, cy = _BASE_MASCOT_X * scale + size / 2, _BASE_MASCOT_Y * scale + size / 2
    pct = size  # the design square is 100 units, so N% scale = N px
    # Pops in: small, a slight overshoot, then settles.
    pop = (
        f"\\fscx{_num(pct * 0.6)}\\fscy{_num(pct * 0.6)}"
        f"\\t(0,160,\\fscx{_num(pct * 1.12)}\\fscy{_num(pct * 1.12)})"
        f"\\t(160,260,\\fscx{_num(pct)}\\fscy{_num(pct)})"
    )
    lines = []
    for layer, colour, extra, shape in _MASCOT_LAYERS:
        lines.append(
            f"Dialogue: {20 + layer},{_fmt_ts(0)},{_fmt_ts(_BRAND_END)},Mascot,,0,0,0,,"
            f"{{\\an5\\pos({_num(cx)},{_num(cy)})\\bord0\\shad0{extra}\\1c&H{colour}&{pop}\\p1}}{_SQUARE}{shape}{{\\p0}}"
        )
    stamp = _escape_ass_text(clean_hook_text(name).upper())
    if stamp:
        x = cx + size / 2 + (_BASE_STAMP_BOX + 14) * scale
        lines.append(
            f"Dialogue: 20,{_fmt_ts(0.15)},{_fmt_ts(_STAMP_SECONDS)},Stamp,,0,0,0,,"
            f"{{\\an4\\pos({_num(x)},{_num(cy)})\\fad(120,250)}}{stamp}"
        )
    return lines


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
    hook_text: Optional[str] = None,
    punchy: bool = False,
    brand: bool = False,
) -> Path:
    """clip_words are absolute-time Word objects that fall within the clip;
    clip_start is subtracted so the .ass timeline starts at 0 for this clip.

    hook_text, when given, is shown boxed near the top for the clip's first
    HOOK_TEXT_SECONDS. It goes in this same file (not a second pass like
    hook_line_ass) so it burns in with the captions, and a later re-render
    from this file -- a manual facecam fix -- keeps it. brand adds the
    channel mascot and name stamp (brand_dialogues) the same way.

    play_res should match the actual output frame size (width, height) --
    it's both the coordinate system the Style/Dialogue margins below are
    written in AND (via _ass_header's scaling) what decides how big those
    margins and the font actually are. Defaults to the normal portrait
    Short's 1080x1920; pass the real target for anything else (e.g. a
    landscape recap) so captions land where the style values expect them
    rather than however libass happens to stretch a mismatched canvas."""
    lines = [_ass_header(play_res)]
    groups = _group_words(clip_words, _POP_MAX_WORDS, _POP_MAX_SPAN) if punchy else _group_words(clip_words)
    style, pop = ("CaptionPop", _POP_IN) if punchy else ("Caption", "")
    for group in groups:
        g_start = group[0].start - clip_start
        g_end = group[-1].end - clip_start
        # \k tags highlight one word at a time (centiseconds per word)
        parts = []
        for w in group:
            dur_cs = max(1, int(round((w.end - w.start) * 100)))
            parts.append(f"{{\\k{dur_cs}}}{_escape_ass_text(w.text.upper())} ")
        text = "".join(parts).strip()
        lines.append(
            f"Dialogue: 0,{_fmt_ts(g_start)},{_fmt_ts(g_end)},{style},,0,0,0,,{pop}{text}"
        )
    hook = clean_hook_text(hook_text)
    if hook:
        lines.append(
            f"Dialogue: 1,{_fmt_ts(0)},{_fmt_ts(HOOK_TEXT_SECONDS)},HookText,,0,0,0,,"
            f"{{\\fad(0,200)}}{_escape_ass_text(hook)}"
        )
    if brand:
        lines.extend(brand_dialogues(play_res))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
