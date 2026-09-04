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
    upload_title: str
    description: str
    reason: str


@dataclass
class WindowPick:
    """Like ClipPick, but start/end are local to one candidate window's own
    downloaded file (0 at the window's start), not the source VOD's absolute
    time -- used by select_from_candidate_windows for the long-VOD pipeline,
    where each candidate window is downloaded/transcribed as its own file."""
    window_index: int
    start: float
    end: float
    title: str
    hook_caption: str
    upload_title: str
    description: str
    reason: str


def _salvage_json_array(raw: str) -> Optional[list]:
    """Best-effort recovery when the model's JSON array doesn't parse
    outright -- most often the response got cut off mid-object (hit
    max_tokens partway through a long `description` field) or one field
    has a stray unescaped character. Walks the array decoding one
    complete object at a time and keeps whatever parsed cleanly before
    the break, so one bad/truncated pick doesn't throw away the whole
    batch of candidates."""
    if not raw.startswith("["):
        return None
    decoder = json.JSONDecoder()
    items: list = []
    idx = 1  # past the leading '['
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx] in " \t\n\r,":
            idx += 1
        if idx >= n or raw[idx] == "]":
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            break
        items.append(obj)
        idx = end
    return items or None


def _ask_claude_for_json(prompt: str, api_key: Optional[str], model: str, max_tokens: int = 4096) -> list:
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

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "max_tokens":
        print("[select_moments] response hit max_tokens -- may be truncated", flush=True)
    raw = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        salvaged = _salvage_json_array(raw)
        if salvaged:
            print(
                f"[select_moments] JSON parse failed, salvaged {len(salvaged)} "
                f"complete pick(s) out of the response: {e}", flush=True,
            )
            return salvaged
        raise RuntimeError(f"Model did not return valid JSON:\n{raw[:500]}") from e


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


def _with_shorts_tag(title: str) -> str:
    """Force #Shorts onto the upload title regardless of what the model
    produced -- YouTube's auto-detection of vertical clips as Shorts can be
    inconsistent above ~60s, and the hashtag makes the intent unambiguous
    no matter which upload path is used."""
    if re.search(r"#shorts\b", title, re.IGNORECASE):
        return title
    return f"{title} #Shorts"


def select_clips(
    words: List[Word],
    video_duration: float,
    n_clips: int = 5,
    min_len: float = 20.0,
    max_len: float = 90.0,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    source_title: Optional[str] = None,
) -> List[ClipPick]:
    """Return up to n_clips non-overlapping ClipPicks, sorted by start time."""
    transcript_text = _chunk_transcript(words)
    focus_line = f"\nThe creator specifically wants: {focus}\n" if focus else ""
    source_line = f"\nSource video title: {source_title}\n" if source_title else ""

    prompt = f"""You are picking highlight clips from a video transcript for short-form
content (YouTube Shorts / TikTok / Reels). The transcript below has [mm:ss]
timestamp markers every ~10 seconds -- use them to anchor your start/end times,
interpolating between markers for precision.

Video length: {video_duration:.0f} seconds.
{source_line}{focus_line}
Pick up to {n_clips} clips. Each clip must:
- be between {min_len:.0f} and {max_len:.0f} seconds long
- work as a standalone moment (a hook, a punchline, a strong claim, a story
  beat with a payoff, a surprising fact) -- not a random mid-sentence cut
- not overlap with any other clip you pick
- start right at (or just before) the moment that hooks attention, not mid-thought

Respond with ONLY a JSON array, no other text, in this exact shape (escape any
double-quote characters that appear inside a string value, e.g. \" ):
[
  {{
    "start": 12.5,
    "end": 58.0,
    "title": "short internal label, not shown on screen",
    "hook_caption": "punchy 4-8 word on-screen hook text for the first second of the clip",
    "upload_title": "the actual title to post the clip with on YouTube Shorts/Instagram Reels -- written like real clip-channel titles: attention-grabbing, often a question or a bold claim, can use ALL CAPS for emphasis on 1-2 key words, mention the creator/streamer by name if you can identify them from the transcript or source title for searchability and credit, no hashtags, under 90 characters",
    "description": "the actual post description to upload alongside the clip -- 1-3 short sentences giving context on what happens and why it's worth watching, credit the creator/streamer by name if identifiable, end with 3-6 relevant hashtags (e.g. #shorts, the game/topic, the creator's name), no links",
    "reason": "one sentence on why this moment works as a clip"
  }}
]

Transcript:
{transcript_text}
"""

    data = _ask_claude_for_json(prompt, api_key, model)

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
        title = str(item.get("title", "")).strip() or "Untitled clip"
        picks.append(ClipPick(
            start=round(start, 2),
            end=round(end, 2),
            title=title,
            hook_caption=str(item.get("hook_caption", "")).strip(),
            upload_title=_with_shorts_tag(str(item.get("upload_title", "")).strip() or title),
            description=str(item.get("description", "")).strip(),
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


def select_from_candidate_windows(
    windows: List[dict],
    n_clips: int = 5,
    min_len: float = 20.0,
    max_len: float = 90.0,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    source_title: Optional[str] = None,
) -> List[WindowPick]:
    """Pick the best clips from a set of pre-filtered candidate windows
    (long-VOD pipeline) instead of one continuous transcript.

    Each item in `windows` is a dict: {"index": int, "duration": float,
    "words": List[Word], "signal": str} -- words are timestamped local to
    that window (0 at its start), since each window is downloaded and
    transcribed as its own standalone file.
    """
    if not windows:
        return []

    focus_line = f"\nThe creator specifically wants: {focus}\n" if focus else ""
    source_line = f"\nSource VOD title: {source_title}\n" if source_title else ""

    blocks = []
    for w in windows:
        transcript_text = _chunk_transcript(w["words"], mark_every=15.0)
        blocks.append(
            f"--- Candidate window {w['index']} "
            f"(duration {w['duration']:.0f}s, signal: {w['signal']}) ---\n"
            f"{transcript_text}\n"
        )
    windows_text = "\n".join(blocks)

    prompt = f"""You are picking the best highlight clips from a set of CANDIDATE windows
that were already pre-filtered out of a much longer livestream VOD (chat
activity spikes and/or moments viewers already clipped). Each candidate
window below is its own short segment with its own transcript, timestamped
LOCALLY from 0 at the start of that window -- not the VOD's absolute time.
{source_line}{focus_line}
Not every candidate window is actually a good clip -- some chat spikes are
noise, reactions to something off-screen, or don't read well out of context.
Pick only the ones that would genuinely work as a standalone short-form clip.

Pick up to {n_clips} windows. For each one you pick, give a start/end IN
SECONDS LOCAL TO THAT WINDOW (0 to its duration) -- use the whole window or
trim it tighter around the actual moment. Each clip must:
- be between {min_len:.0f} and {max_len:.0f} seconds long
- work as a standalone moment, not a random mid-sentence cut
- start right at (or just before) the moment that hooks attention

Respond with ONLY a JSON array, no other text, in this exact shape (escape any
double-quote characters that appear inside a string value, e.g. \" ):
[
  {{
    "window_index": 3,
    "start": 4.0,
    "end": 52.0,
    "title": "short internal label, not shown on screen",
    "hook_caption": "punchy 4-8 word on-screen hook text for the first second of the clip",
    "upload_title": "the actual title to post the clip with on YouTube Shorts/Instagram Reels -- written like real clip-channel titles: attention-grabbing, often a question or a bold claim, can use ALL CAPS for emphasis on 1-2 key words, mention the creator/streamer by name if you can identify them for searchability and credit, no hashtags, under 90 characters",
    "description": "the actual post description to upload alongside the clip -- 1-3 short sentences giving context on what happens and why it's worth watching, credit the creator/streamer by name if identifiable, end with 3-6 relevant hashtags (e.g. #shorts, the game/topic, the creator's name), no links",
    "reason": "one sentence on why this moment works as a clip"
  }}
]

Candidate windows:
{windows_text}
"""

    data = _ask_claude_for_json(prompt, api_key, model)

    by_index = {w["index"]: w for w in windows}
    picks: List[WindowPick] = []
    seen_indices = set()
    for item in data:
        try:
            idx = int(item["window_index"])
            window = by_index[idx]
        except (KeyError, TypeError, ValueError):
            continue
        if idx in seen_indices:
            continue
        try:
            start = max(0.0, float(item["start"]))
            end = min(window["duration"], float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < min_len * 0.6:
            continue
        if end - start > max_len * 1.2:
            end = start + max_len
        title = str(item.get("title", "")).strip() or "Untitled clip"
        seen_indices.add(idx)
        picks.append(WindowPick(
            window_index=idx,
            start=round(start, 2),
            end=round(end, 2),
            title=title,
            hook_caption=str(item.get("hook_caption", "")).strip(),
            upload_title=_with_shorts_tag(str(item.get("upload_title", "")).strip() or title),
            description=str(item.get("description", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
        ))

    return picks[:n_clips]
