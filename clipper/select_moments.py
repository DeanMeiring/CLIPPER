"""Ask Claude which segments of the transcript are worth clipping."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .transcribe import Word

# Check https://docs.claude.com/en/docs/about-claude/models for current model
# ids -- override anytime with the CLIPPER_MODEL env var without touching code.
DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")


@dataclass
class ClipPick:
    start: float
    end: float
    title: str
    hook_caption: str
    reason: str


def _chunk_transcript(words: List[Word], mark_every: float = 10.0) -> str:
    """Render the transcript as running text with periodic [mm:ss] markers,
    so the model can point back at real timestamps without us sending every
    single word's start/end (which would bloat the prompt on long videos)."""
    lines = []
    next_mark = 0.0
    buf: List[str] = []

    def flush():
        if buf:
            lines.append(" ".join(buf))
            buf.clear()

    for w in words:
        if w.start >= next_mark:
            flush()
            m, s = divmod(int(w.start), 60)
            lines.append(f"\n[{m:02d}:{s:02d}]")
            next_mark = w.start + mark_every
        buf.append(w.text)
    flush()
    return " ".join(lines)


def _mmss_to_seconds(text: str) -> Optional[float]:
    m = re.match(r"^(\d+):(\d\d)$", text.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def select_clips(
    words: List[Word],
    video_duration: float,
    n_clips: int = 5,
    min_len: float = 20.0,
    max_len: float = 90.0,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> List[ClipPick]:
    """Return up to n_clips non-overlapping ClipPicks, sorted by start time."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic is required for AI moment-selection. Install it with: pip install anthropic"
        ) from e

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY (get one at https://console.anthropic.com/) "
            "or pass --api-key."
        )

    transcript_text = _chunk_transcript(words)
    focus_line = f"\nThe creator specifically wants: {focus}\n" if focus else ""

    prompt = f"""You are picking highlight clips from a video transcript for short-form
content (YouTube Shorts / TikTok / Reels). The transcript below has [mm:ss]
timestamp markers every ~10 seconds -- use them to anchor your start/end times,
interpolating between markers for precision.

Video length: {video_duration:.0f} seconds.
{focus_line}
Pick up to {n_clips} clips. Each clip must:
- be between {min_len:.0f} and {max_len:.0f} seconds long
- work as a standalone moment (a hook, a punchline, a strong claim, a story
  beat with a payoff, a surprising fact) -- not a random mid-sentence cut
- not overlap with any other clip you pick
- start right at (or just before) the moment that hooks attention, not mid-thought

Respond with ONLY a JSON array, no other text, in this exact shape:
[
  {{
    "start": 12.5,
    "end": 58.0,
    "title": "short internal label, not shown on screen",
    "hook_caption": "punchy 4-8 word on-screen hook text for the first second of the clip",
    "reason": "one sentence on why this moment works as a clip"
  }}
]

Transcript:
{transcript_text}
"""

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON:\n{raw[:500]}") from e

    picks: List[ClipPick] = []
    for item in data:
        try:
            start = max(0.0, float(item["start"]))
            end = min(video_duration, float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < min_len * 0.6:  # allow a little slack vs. exact min_len
            continue
        if end - start > max_len * 1.2:
            end = start + max_len
        picks.append(ClipPick(
            start=round(start, 2),
            end=round(end, 2),
            title=str(item.get("title", "")).strip() or "Untitled clip",
            hook_caption=str(item.get("hook_caption", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
        ))

    picks.sort(key=lambda p: p.start)
    non_overlapping: List[ClipPick] = []
    last_end = -1.0
    for p in picks:
        if p.start >= last_end:
            non_overlapping.append(p)
            last_end = p.end
    return non_overlapping[:n_clips]
