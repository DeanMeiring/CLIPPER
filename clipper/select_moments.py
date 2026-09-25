"""Ask Claude which segments of the transcript are worth clipping."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .cut_points import Cut, refine_cut
from .loud_moments import LoudMoment
from .transcribe import Word

# Picking and cutting the clips decides whether a Short holds viewers, so it
# has its own model setting rather than sharing CLIPPER_MODEL with the
# lighter tasks (facecam checks, hook lines, recaps) -- several of those send
# max_tokens=10 requests that a model with thinking on would spend entirely
# on thinking. Check https://docs.claude.com/en/docs/about-claude/models for
# current ids; override with CLIPPER_SELECT_MODEL without touching code.
DEFAULT_MODEL = os.environ.get("CLIPPER_SELECT_MODEL", "claude-opus-5")
# Used when DEFAULT_MODEL can't serve a pick at all (not enabled for this API
# key, rate limited, request rejected, declined even after the server-side
# fallback), so a job still gets its clips.
_BACKUP_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

# Model generations that take adaptive thinking; anything older (an old
# override, the backup model) gets the plain request it always got.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-opus-5", "claude-fable-5", "claude-sonnet-5",
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6",
)
# Models whose safety classifiers can decline a request. With server-side
# fallbacks the API reruns a declined request on Anthropic's recommended
# fallback model inside the same call instead of returning the refusal.
_SERVER_FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


@dataclass
class ClipPick:
    start: float
    end: float
    title: str
    hook_caption: str
    upload_title: str
    description: str
    reason: str
    # 1-10, Claude's own call on how well this clip will hold viewers,
    # relative to the other picks in the batch; None if it gave none.
    score: Optional[int] = None


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
    # 1-10, Claude's own call on how well this clip will hold viewers,
    # relative to the other picks in the batch; None if it gave none.
    score: Optional[int] = None


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


class ClaudeDeclined(RuntimeError):
    """Claude's safety classifiers declined the request (after any
    server-side fallback) -- sending the same prompt again won't change that."""


def _request_options(model: str) -> dict:
    options: dict = {"max_tokens": 8192}
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        # max_tokens caps thinking and the JSON answer together.
        options.update(max_tokens=32000, thinking={"type": "adaptive"})
    if model in _SERVER_FALLBACK_MODELS:
        options.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    return options


def _ask_claude_for_json_once(client, prompt: str, model: str) -> list:
    # Streamed because with thinking on, a pick over a long stream's
    # transcript can run for minutes: a non-streamed request has to finish
    # inside the SDK's 10-minute timeout, a streamed one only has to keep
    # sending events.
    with client.beta.messages.stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **_request_options(model),
    ) as stream:
        resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        category = getattr(getattr(resp, "stop_details", None), "category", None)
        raise ClaudeDeclined(
            f"Claude declined to pick clips from this transcript (category: {category or 'not given'})."
        )
    if any(getattr(entry, "type", None) == "fallback_message" for entry in (getattr(resp.usage, "iterations", None) or [])):
        print(f"[select_moments] {model} declined; served by fallback model {resp.model}", flush=True)
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
        # No usable JSON at all -- log what the response actually contained
        # so a recurrence is diagnosable from Railway logs instead of a
        # blank raw[:500]. In particular: if the model's entire max_tokens
        # budget went to a non-"text" block (e.g. thinking) before it ever
        # got to write the JSON, `raw` ends up empty even though real
        # tokens were spent -- this makes that visible.
        block_summary = [
            f"{getattr(b, 'type', 'unknown')}:{len(getattr(b, 'text', '') or '')}"
            for b in resp.content
        ]
        usage = getattr(resp, "usage", None)
        print(
            f"[select_moments] no usable JSON -- stop_reason={resp.stop_reason} "
            f"blocks={block_summary} usage={usage}", flush=True,
        )
        raise RuntimeError(f"Model did not return valid JSON:\n{raw[:500]}") from e


def _ask_with_one_retry(client, prompt: str, model: str) -> list:
    try:
        return _ask_claude_for_json_once(client, prompt, model)
    except ClaudeDeclined:
        raise
    except RuntimeError as e:
        # A response with no usable JSON and nothing for the salvage pass
        # to recover from is rare but confirmed to happen (seen in
        # practice: the model's output cut off right after the opening
        # ```json fence, before a single field). Rather than failing the
        # whole job over what's likely a one-off bad generation, retry
        # once with a fresh sample before giving up for real.
        print(f"[select_moments] first attempt failed ({e}), retrying once", flush=True)
        return _ask_claude_for_json_once(client, prompt, model)


def _ask_claude_for_json(prompt: str, api_key: Optional[str], model: str) -> list:
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
    try:
        return _ask_with_one_retry(client, prompt, model)
    except (
        ClaudeDeclined,
        anthropic.NotFoundError, anthropic.PermissionDeniedError, anthropic.BadRequestError,
        anthropic.RateLimitError, anthropic.OverloadedError, anthropic.InternalServerError,
    ) as e:
        if _BACKUP_MODEL == model:
            raise
        print(
            f"[select_moments] {model} couldn't serve this pick ({type(e).__name__}: {e}) "
            f"-- using {_BACKUP_MODEL} instead", flush=True,
        )
        return _ask_with_one_retry(client, prompt, _BACKUP_MODEL)


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


def _loud_moments_block(loud_moments: Optional[List[LoudMoment]]) -> str:
    if not loud_moments:
        return ""
    listing = "\n".join(
        f"- {m.start:.0f}s-{m.end:.0f}s (peak {m.peak_db:.0f} dBFS, {m.jump_db:.0f} dB above the surrounding baseline)"
        for m in loud_moments
    )
    return f"""
Audio analysis also flagged these moments as noticeably louder than the
surrounding audio -- shouting, a scream, a "crash out", or a similar burst
of volume. Treat this as a hint, not a verdict: game sound effects, music
stings, and other loud-but-unremarkable audio trigger it too, and plenty of
great clips aren't loud at all. Cross-check against what's actually being
said in the transcript at that timestamp before picking one of these as a
clip -- only pick it if the content itself earns it.
{listing}
"""


_GROUP_BANTER_NOTE = """
A transcript that reads messily -- overlapping speech, interruptions, cut-off
sentences, unclear who's talking -- is often just what it looks like in text
when MULTIPLE PEOPLE are reacting/bantering together, not a sign the moment
itself is weak. Group banter between multiple streamers (a collab/co-op
moment, everyone reacting to the same thing at once) is frequently the
funniest, highest-energy content on a stream precisely because of that
back-and-forth chaos. Don't undervalue or skip a candidate just because its
transcript is harder to read than a single person talking cleanly -- judge
it by whether the energy and content would land as a clip, not by how
cleanly it transcribes."""


def _strategy_notes_block(strategy_notes: Optional[str]) -> str:
    if not strategy_notes:
        return ""
    return f"""
A previous analysis of THIS channel's own real upload performance (actual
view counts, retention, and traffic data -- not a generic best-practices
list) found the following. Treat this as a genuine third factor in your
decision, on equal footing with how strong a moment reads in the
transcript and any audio/chat signal below -- not just a tiebreaker:
- SELECTION: when candidates are close, prefer the one whose topic, pacing,
  or hook most resembles what this data shows actually working for this
  specific audience (or actively avoid a pattern it shows failing).
- TITLE/DESCRIPTION: write upload_title and description to match the hook
  style, phrasing, and topic angle this data shows earning clicks for THIS
  channel specifically -- not a generic clip-title style. If it names a
  reach problem (good content, weak title) on past clips, that's a direct
  instruction to make the title stronger and more specific this time, not
  just descriptive.
Still judge each moment on its own merits from the transcript -- don't force
a pick that doesn't actually work as a clip just because it superficially
matches this analysis.
{strategy_notes}
"""


def _performance_block(performance_notes: Optional[str]) -> str:
    if not performance_notes:
        return ""
    return f"""
Measured results from THIS channel's own posted Shorts -- real retention
curves and view counts, compared across what the clips had in common. A
comparison marked TOO FEW has under 5 uploads behind it and can't support a
conclusion on its own. Where one with enough uploads shows a clear
difference (clip length, how fast the first words start, title style,
layout), lean your picks, cuts and titles toward what worked for this
audience, even over the general guidance below:
{performance_notes}
"""


def _clip_rules(min_len: float, max_len: float) -> str:
    return f"""What decides whether a Short gets watched: viewers choose within the first
second or two whether to keep watching or swipe away. So each clip must:
- open on its hook. The first words spoken ARE the hook: start on the line
  that makes a scrolling viewer stop -- the setup that creates tension, or
  the most surprising thing said -- never on a greeting, filler ("uh",
  "okay so", "chat"), dead air, or the tail of an unrelated sentence
- be the tightest cut that still holds the setup and the payoff. On
  streamer-clip channels most breakout Shorts run 15-35 seconds (median
  around 25). Past 35 seconds only for a story with several real beats
  that all pay off -- a single reaction never needs that long
- end right after the payoff lands (the punchline, the reaction, the
  result), with no trailing chatter -- a tight ending also loops cleanly
  back into the hook
- make sense to someone who has never watched this streamer and missed the
  rest of the stream
- be between {min_len:.0f} and {max_len:.0f} seconds long, and not overlap any other clip"""


# Field-by-field instructions shared by both prompts' JSON shape. A plain
# string (not an f-string), so its braces and quotes need no escaping.
_PICK_FIELDS = """    "start_words": "the exact first 3-6 words spoken in the clip, copied verbatim from the transcript -- the clip is cut on these words, so start/end only need to be roughly right",
    "end_words": "the exact last 3-6 words spoken in the clip, copied verbatim from the transcript",
    "title": "short internal label, not shown on screen",
    "hook_caption": "text burned onto the top of the video for its first 3 seconds: 3-7 words telling a scrolling viewer why to stay -- who, and the tension or setup of this specific moment. Not a generic 'wait for it', and not the punchline itself. Normal sentence case with at most one word in ALL CAPS, no emoji (they can't be rendered), no hashtags, and not a copy of upload_title",
    "upload_title": "the title to post the clip with. What the biggest recent Shorts on streamer-clip channels (Jynxzi, Stable Ronaldo and similar) have in common: the streamer's name comes first, it's about 6 words, and it says the specific thing that happens instead of teasing it; many end with one emoji (😭 🤣 😂 🤯 👀) and some put one word in ALL CAPS for emphasis. Questions and vague clickbait ('you won't believe...') are rare among them -- avoid both. Style examples, not to copy: 'Stable Ronaldo Chooses the Wrong ENDING In GTA V 🤯', 'Jynxzi *ATTEMPTS* to do MATH 🤣', 'Bodycam turns Jynxzi Evil'. Use the name viewers know the streamer by, from the transcript or source title; if you can't tell who it is, lead with the most specific thing that happens. No hashtags, under 60 characters",
    "description": "the post description: 1-2 short sentences on who's in it, what happens and why it's worth watching (credit the streamer by name), then 3-5 hashtags (#shorts, the streamer, the game or topic). No links",
    "reason": "one sentence on why this moment holds a viewer past the first few seconds, noting if this channel's own data factored in",
    "score": "whole number 1-10: how likely this clip is to hold scrolling viewers and pull views, compared with the other clips you picked -- spread the scores out (the best of the batch should clearly stand out) rather than giving everything a 7 or 8\""""


def _score(value) -> Optional[int]:
    try:
        return max(1, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def _log_cut(rough_start: float, rough_end: float, cut: Cut) -> None:
    def how(anchored: bool) -> str:
        return "on the quoted words" if anchored else "snapped to the nearest phrase"
    print(
        f"[select_moments] rough {rough_start:.1f}-{rough_end:.1f}s -> cut {cut.start:.2f}-{cut.end:.2f}s "
        f"(start {how(cut.anchored_start)}, end {how(cut.anchored_end)})", flush=True,
    )


def select_clips(
    words: List[Word],
    video_duration: float,
    n_clips: int = 5,
    min_len: float = 15.0,
    max_len: float = 60.0,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    source_title: Optional[str] = None,
    loud_moments: Optional[List[LoudMoment]] = None,
    strategy_notes: Optional[str] = None,
    performance_notes: Optional[str] = None,
) -> List[ClipPick]:
    """Return up to n_clips non-overlapping ClipPicks, sorted by start time."""
    transcript_text = _chunk_transcript(words)
    focus_line = f"\nThe creator specifically wants: {focus}\n" if focus else ""
    source_line = f"\nSource video title: {source_title}\n" if source_title else ""
    loud_line = _loud_moments_block(loud_moments)
    performance_line = _performance_block(performance_notes)
    strategy_line = _strategy_notes_block(strategy_notes)

    prompt = f"""You are picking highlight clips from a stream/video transcript for YouTube
Shorts (also posted to TikTok and Reels). The transcript below has [mm:ss]
timestamp markers every ~10 seconds.

Video length: {video_duration:.0f} seconds.
{source_line}{focus_line}{loud_line}{performance_line}{strategy_line}{_GROUP_BANTER_NOTE}

Pick up to {n_clips} clips.

{_clip_rules(min_len, max_len)}

Respond with ONLY a JSON array, no other text, in this exact shape (escape any
double-quote characters that appear inside a string value):
[
  {{
    "start": 12.5,
    "end": 38.0,
{_PICK_FIELDS}
  }}
]
"start" and "end" are seconds, read off the markers.

Transcript:
{transcript_text}
"""

    data = _ask_claude_for_json(prompt, api_key, model)

    picks: List[ClipPick] = []
    for item in data:
        try:
            rough_start, rough_end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        cut = refine_cut(
            words, rough_start, rough_end, item.get("start_words"), item.get("end_words"),
            video_duration, min_len, max_len,
        )
        if cut is None:
            continue
        _log_cut(rough_start, rough_end, cut)
        title = str(item.get("title", "")).strip() or "Untitled clip"
        picks.append(ClipPick(
            start=cut.start,
            end=cut.end,
            title=title,
            hook_caption=str(item.get("hook_caption", "")).strip(),
            upload_title=_with_shorts_tag(str(item.get("upload_title", "")).strip() or title),
            description=str(item.get("description", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
            score=_score(item.get("score")),
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
    min_len: float = 15.0,
    max_len: float = 60.0,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    source_title: Optional[str] = None,
    strategy_notes: Optional[str] = None,
    performance_notes: Optional[str] = None,
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
    performance_line = _performance_block(performance_notes)
    strategy_line = _strategy_notes_block(strategy_notes)

    blocks = []
    for w in windows:
        transcript_text = _chunk_transcript(w["words"], mark_every=15.0)
        blocks.append(
            f"--- Candidate window {w['index']} "
            f"(duration {w['duration']:.0f}s, signal: {w['signal']}) ---\n"
            f"{transcript_text}\n"
        )
    windows_text = "\n".join(blocks)

    prompt = f"""You are picking the best highlight clips for YouTube Shorts (also posted to
TikTok and Reels) from a set of CANDIDATE windows that were already
pre-filtered out of a much longer livestream VOD (chat activity spikes
and/or moments viewers already clipped). Each candidate window below is its
own short segment with its own transcript, timestamped LOCALLY from 0 at the
start of that window -- not the VOD's absolute time.
{source_line}{focus_line}{performance_line}{strategy_line}
Not every candidate window is actually a good clip -- some chat spikes are
noise, or a reaction to something off-screen that doesn't work without
context. Pick only the ones that would genuinely work as a standalone
short-form clip.
{_GROUP_BANTER_NOTE}

Pick up to {n_clips} windows, at most one clip per window. Use the whole
window or trim it tighter around the actual moment.

{_clip_rules(min_len, max_len)}

Respond with ONLY a JSON array, no other text, in this exact shape (escape any
double-quote characters that appear inside a string value):
[
  {{
    "window_index": 3,
    "start": 4.0,
    "end": 31.0,
{_PICK_FIELDS}
  }}
]
"start" and "end" are seconds LOCAL TO THAT WINDOW (0 to its duration), read
off its markers.

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
            rough_start, rough_end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        cut = refine_cut(
            window["words"], rough_start, rough_end, item.get("start_words"), item.get("end_words"),
            window["duration"], min_len, max_len,
        )
        if cut is None:
            continue
        _log_cut(rough_start, rough_end, cut)
        title = str(item.get("title", "")).strip() or "Untitled clip"
        seen_indices.add(idx)
        picks.append(WindowPick(
            window_index=idx,
            start=cut.start,
            end=cut.end,
            title=title,
            hook_caption=str(item.get("hook_caption", "")).strip(),
            upload_title=_with_shorts_tag(str(item.get("upload_title", "")).strip() or title),
            description=str(item.get("description", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
            score=_score(item.get("score")),
        ))

    return picks[:n_clips]
