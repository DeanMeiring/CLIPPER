"""Weekly cross-streamer recap: concatenate this week's best Twitch clips
across tracked streamers into one landscape, long-form compilation video
-- a normal upload, not a Short, for a second life outside the Shorts
feed. Counts down from the weakest of the picked clips to the biggest
hit, which plays last, with a "#N streamer: title" badge burned into the
top-left corner of every clip so the countdown is legible on its own.

Source material is Twitch's own "Clips" feature (the ones made from the
Clip button on a stream, by the creator or by viewers) -- these are
already curated highlight moments with a real Twitch view count,
available immediately via trending.get_top_twitch_clips(), with no
dependency on this app having already rendered and uploaded something
for that streamer first. View count alone is a noisy quality signal
though -- a clip can rack up views just for who's in it while being
mostly the streamer talking with no actual moment -- so
judge_clip_quality() screens each candidate's transcript before it's
counted as a pick, in view-count order, falling through to the next
candidate whenever one is skipped (see webapp/main.py's
_run_weekly_recap_job). Each kept clip still gets downloaded and
captioned (see webapp/main.py's _render_twitch_clip_for_recap) before
being concatenated, since a raw Twitch clip has no captions of its own
-- but NOT run through this app's vertical facecam-crop pipeline, since
a Twitch clip is already landscape (the streamer's own broadcast frame,
facecam included) and the recap stays landscape too.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

# The recap's fixed series title -- "Top 10 Twitch Clips of the Week
# #{episode}" -- rather than a fresh AI-guessed headline every time.
# A recurring series someone is posting on a regular cadence benefits
# from a stable, predictable name viewers (and YouTube's own recs) come
# to recognize, more than novelty each week.
SERIES_TITLE = "Top 10 Twitch Clips of the Week"

# How many clips the countdown aims for, if the tracked roster and the
# quality gate leave enough good candidates to reach it -- "top ten" as
# a target, not a guarantee; a quiet week or a lot of quality-skips can
# still end with fewer (down to the 2-clip floor _run_weekly_recap_job
# enforces).
TARGET_CLIP_COUNT = 10

# Caps how many of the week's picks can come from any one streamer, so a
# streamer having several viral clips this week can't crowd out every
# other tracked streamer -- this is meant to stay a CROSS-streamer recap.
MAX_CLIPS_PER_STREAMER = 3


def build_candidate_pool(clips: list) -> list:
    """Every fetched Twitch clip across all tracked streamers, ranked
    purely by view count -- the "top most-watched" order the selection
    loop in _run_weekly_recap_job walks down, applying the quality gate
    and the per-streamer cap as it goes. A clip with no streamer login is
    left out rather than attributed to a fake "unknown" streamer."""
    pool = [c for c in clips if (c.get("streamer_login") or "").strip()]
    pool.sort(key=lambda c: c.get("view_count") or 0, reverse=True)
    return pool


def build_recap_video(clip_paths: list, out_path: Path) -> None:
    """Concatenate clips back to back into one long-form video. Re-encodes
    rather than using the much faster stream-copy concat mode, because
    every input isn't guaranteed to share identical encoder parameters
    (e.g. a clip rendered before some past render.py change) -- concat
    demuxer's stream-copy mode fails hard, or silently produces broken
    output, the moment any input disagrees with the first one."""
    import subprocess

    if len(clip_paths) < 2:
        raise ValueError("need at least 2 clips to build a recap")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in clip_paths), encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        list_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s building the recap") from e
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building the recap:\n{result.stderr[-2000:]}")


def display_name(entry: dict) -> str:
    # Best-effort only -- Twitch's clips API gives us the broadcaster's
    # LOGIN (always lowercase), not their real display-name casing, so a
    # camelCase name like "TheBurntPeanut" comes back as "Theburntpeanut"
    # here. Fixing that would mean an extra Twitch users lookup per
    # streamer just for cosmetics; not worth it for a chapters list.
    login = entry.get("streamer_login") or ""
    return login.replace("_", " ").title()


def rank_badge_text(rank: int, entry: dict, max_len: int = 60) -> str:
    """'#7 Jynxzi: Insane 1v5 clutch' -- the on-screen top-left badge
    burned into each clip (see webapp/main.py's
    _render_twitch_clip_for_recap) so a countdown-ordered recap names
    which rank the viewer is watching instead of leaving it a silent,
    unlabeled countdown. Truncated with an ellipsis rather than wrapped
    or dropped, since the badge is one fixed-size line, not a caption."""
    label = f"#{rank} {display_name(entry)}: {entry.get('title', '')}"
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "…"


def _build_quality_prompt(candidate_title: str, streamer: str, view_count: int, transcript_text: str) -> str:
    return f"""You're screening one Twitch clip as a candidate for a "best clips of the
week" YouTube compilation. It ranked well by Twitch view count, but view count
alone doesn't catch a clip that's mostly someone talking with no real moment in
it -- that's your job here, before it goes in the video.

Streamer: {streamer}
Clip title: "{candidate_title}"
Twitch views: {view_count}
Spoken transcript of the clip: "{transcript_text[:2000]}"

Judge from the transcript: does something actually HAPPEN in this clip (a
clutch, a fail, a funny line, a sharp reaction, a callout, a joke landing)
that would hold a cold viewer's attention with zero context from the stream
-- or is it mostly filler talk, rambling, or setup with no real payoff? A
clip can still be great even if the transcript alone doesn't fully capture a
visual moment (a clutch play, a facial reaction) -- give it the benefit of
the doubt unless the transcript reads as clearly just chatter with nothing
happening.

Answer in exactly this format, nothing else:
VERDICT: KEEP or SKIP
REASON: one short sentence why"""


def judge_clip_quality(
    candidate_title: str, streamer: str, view_count: int, transcript_text: str,
    api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
) -> dict:
    """Best-effort AI gate on whether a Twitch clip is actually worth a
    spot in the recap, independent of its Twitch view count -- a highly-
    viewed clip can still be a streamer rambling for its whole length
    with no moment that plays for someone who wasn't already watching
    live. Returns {"keep": bool, "reason": str}; defaults to keep=True on
    any failure (no ANTHROPIC_API_KEY, a network error, an empty
    transcript, an unparseable response) so a broken quality check
    degrades to the old view-count-only behavior rather than blocking
    the recap or silently dropping every candidate."""
    import re

    default = {"keep": True, "reason": "quality check unavailable, defaulting to keep"}
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not transcript_text.strip():
        return default
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=200,
            messages=[{"role": "user", "content": _build_quality_prompt(candidate_title, streamer, view_count, transcript_text)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        verdict_match = re.search(r"VERDICT:\s*(KEEP|SKIP)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        if not verdict_match:
            return default
        return {
            "keep": verdict_match.group(1).upper() == "KEEP",
            "reason": reason_match.group(1).strip() if reason_match else "",
        }
    except Exception as e:
        print(f"[weekly_recap] quality check failed for {streamer!r}, defaulting to keep: {e}", flush=True)
        return default


def next_episode_number(path: Path) -> int:
    """Reads and increments a persisted counter for the recap's fixed
    series title ("Top 10 Twitch Clips of the Week #N", see SERIES_TITLE)
    -- starts at 1 the first time this is ever called, and survives
    restarts since `path` lives on the same persistent volume as
    everything else in BASE_DIR. Best-effort: a write failure just means
    the same episode number could repeat next time, which is a cosmetic
    problem, not a reason to fail an otherwise-good recap."""
    n = 1
    try:
        if path.exists():
            n = int(json.loads(path.read_text(encoding="utf-8")).get("next", 1))
    except (OSError, ValueError, TypeError):
        n = 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"next": n + 1}), encoding="utf-8")
    except OSError:
        pass
    return n


def _build_hook_prompt(lineup: list, week_label: str) -> str:
    lines = [
        f'- {display_name(u)}: "{u.get("title", "")}" ({u.get("view_count", 0)} views on Twitch)'
        for u in lineup
    ]
    clips_block = "\n".join(lines)
    return f"""You're writing the description opener for a YouTube compilation video
that stitches together this week's best Twitch clips from {len(lineup)} clip(s)
across multiple streamers this creator clips regularly. Every clip here is a real,
audience-tested highlight -- picked by Twitch view count AND a pass for whether it
actually has a moment in it, not a guess -- so you can lean on that instead of
generic hype. The video is a countdown that builds to its biggest hit, which plays
LAST, so the clips below are listed in the order they actually appear on screen
(weakest-viewed of the picks first):

Clips in this compilation, in on-screen order ({week_label}):
{clips_block}

Write 1-2 sentences for the very top of the video description that make someone
want to keep watching. Specific to what's actually in these clips (name a
streamer or a moment), not a generic "check out this week's craziest moments!".

Answer with just that text, nothing else -- no label, no markdown, no quotes
around it."""


def generate_recap_hook(lineup: list, week_label: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> Optional[str]:
    """A punchier, Claude-written opening line for the description than
    the plain deterministic one below -- best-effort: returns None on
    any failure (no ANTHROPIC_API_KEY, a network error, an empty
    response) so the recap can still ship with the deterministic
    fallback rather than blocking the whole build on this one call.
    Only the description's opening hook is AI-written -- the video's
    actual title is always the fixed, numbered SERIES_TITLE (see
    build_recap_metadata), since a series someone posts on a regular
    cadence needs a stable, predictable name, not a fresh guess every
    week."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not lineup:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=200,
            messages=[{"role": "user", "content": _build_hook_prompt(lineup, week_label)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return text or None
    except Exception as e:
        print(f"[weekly_recap] AI hook generation failed, using the deterministic fallback: {e}", flush=True)
        return None


def build_recap_metadata(
    lineup: list, week_label: str, episode: int, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
    intro_offset: float = 0.0,
) -> dict:
    """A fixed, numbered title (see SERIES_TITLE) + a chapter-formatted
    description (YouTube turns a description starting "0:00 ..." into
    clickable chapters). Tries a punchier Claude-written opening line
    first (generate_recap_hook); the chapters themselves are always the
    same plain arithmetic over durations already known from each clip's
    own render, Claude or not, since those need to stay accurate, not
    "captivating".

    `lineup` arrives in actual on-screen (countdown) order -- weakest of
    the picks first, biggest hit last -- since the chapters below have to
    match the real video. `intro_offset` shifts every chapter timestamp
    by the intro card's duration (see build_intro_clip) when one was
    prepended to the video, so the chapters still line up with what's
    actually on screen."""
    title = f"{SERIES_TITLE} #{episode}"[:100]

    hook = generate_recap_hook(lineup, week_label, api_key, model)
    if hook:
        intro = hook
    else:
        names = [display_name(u) for u in sorted(lineup, key=lambda u: u.get("view_count", 0), reverse=True)]
        seen: set = set()
        ordered_names = [n for n in names if not (n in seen or seen.add(n))]
        intro = f"This week's best clips from {', '.join(ordered_names)} ({week_label})."

    lines = [intro, ""]
    t = intro_offset
    if intro_offset:
        lines.append(f"0:00 Intro -- {title}")
    for u in lineup:
        minutes, seconds = divmod(int(t), 60)
        lines.append(f"{minutes}:{seconds:02d} {display_name(u)} -- {u.get('title', '')}")
        t += float(u.get("duration") or 0.0)
    description = "\n".join(lines)[:5000]
    return {"title": title, "description": description}


def build_outro_clip(out_path: Path, text: str, duration: float = 4.0, out_w: int = 1920, out_h: int = 1080) -> None:
    """A short end card (default: a plain black screen with centered
    text) appended after the last real clip -- built the same way every
    caption in this app is (a libass .ass file burned in via ffmpeg's
    `ass` filter), just onto a generated blank background instead of a
    downloaded video, so it reuses the exact font/rendering path already
    proven to work on every other clip rather than depending on ffmpeg's
    separate drawtext/fontconfig setup, which this app has never actually
    exercised and can't be assumed to be configured the same way.

    Needs a silent audio track (not just video) -- build_recap_video's
    concat step re-encodes assuming every input has the same stream
    layout as the real clips (video + audio); a video-only input breaks
    that assumption."""
    import subprocess

    from .captions import _ass_header, _escape_ass_text, _fmt_ts

    ass_path = out_path.with_suffix(".ass")
    # One static line for the outro's whole length -- build_ass's word-by-
    # word \k karaoke timing is for spoken captions synced to audio, not
    # applicable to one fixed line with nothing to time against. The
    # shared Caption style is bottom-anchored (right for a caption
    # overlaid on gameplay); \an5 overrides just this line to middle-
    # center, which reads as an actual end card instead of a caption
    # stuck on an empty background.
    dialogue = f"Dialogue: 0,{_fmt_ts(0)},{_fmt_ts(duration)},Caption,,0,0,0,,{{\\an5}}{_escape_ass_text(text)}"
    ass_path.write_text(_ass_header((out_w, out_h)) + dialogue + "\n", encoding="utf-8")
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={out_w}x{out_h}:d={duration}:r=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-shortest",
        "-vf", f"ass='{ass_escaped}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        ass_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s building the outro") from e
    ass_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building the outro:\n{result.stderr[-2000:]}")


def build_intro_clip(
    source_video: Path, out_path: Path, title_text: str,
    duration: float = 3.5, slow_factor: float = 1.8, out_w: int = 1920, out_h: int = 1080,
) -> None:
    """A short intro card prepended before the countdown starts: a
    dimmed, slow-motion peek at the first (weakest-ranked) clip with the
    series title burned in over it, so the video announces itself before
    diving straight into clip #10's own captions. Reuses the exact
    libass caption-rendering path build_outro_clip does for its text --
    same \\an5-centered override on the shared Caption style -- just
    layered over slowed/darkened footage instead of a blank background.

    `source_video` should already be this recap's own landscape
    (out_w x out_h) output -- this doesn't crop or reframe it, only
    trims, slows, darkens, and captions it. Slowing stretches
    `duration / slow_factor` seconds of real footage into `duration`
    seconds of intro, so even the shortest clip this app renders has
    comfortably enough source to draw from.

    Needs a silent audio track (not the source clip's own audio) for the
    same reason build_outro_clip does -- build_recap_video's concat step
    re-encodes assuming every input has a matching video+audio layout,
    and playing the source's real (sped-down, pitch-shifted) audio under
    a title card would also just sound wrong."""
    import subprocess

    from .captions import _ass_header, _escape_ass_text, _fmt_ts

    ass_path = out_path.with_suffix(".ass")
    dialogue = f"Dialogue: 0,{_fmt_ts(0)},{_fmt_ts(duration)},Caption,,0,0,0,,{{\\an5}}{_escape_ass_text(title_text)}"
    ass_path.write_text(_ass_header((out_w, out_h)) + dialogue + "\n", encoding="utf-8")
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")

    source_seconds = duration / slow_factor
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_video),
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{duration}",
        "-filter_complex",
        f"[0:v]trim=0:{source_seconds},setpts={slow_factor}*PTS,eq=brightness=-0.35,ass='{ass_escaped}'[v]",
        "-map", "[v]", "-map", "1:a", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        ass_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s building the intro") from e
    ass_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building the intro:\n{result.stderr[-2000:]}")
