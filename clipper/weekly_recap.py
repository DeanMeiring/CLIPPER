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

import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

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


def _build_hook_prompt(lineup: list, week_label: str) -> str:
    lines = [
        f'- {display_name(u)}: "{u.get("title", "")}" ({u.get("view_count", 0)} views on Twitch)'
        for u in lineup
    ]
    clips_block = "\n".join(lines)
    return f"""You're writing the title and description opener for a YouTube compilation
video that stitches together this week's best Twitch clips from {len(lineup)} clip(s)
across multiple streamers this creator clips regularly. Every clip here is a real,
audience-tested highlight -- picked by Twitch view count AND a pass for whether it
actually has a moment in it, not a guess -- so you can lean on that instead of
generic hype. The video is a countdown that builds to its biggest hit, which plays
LAST, so the clips below are listed in the order they actually appear on screen
(weakest-viewed of the picks first):

Clips in this compilation, in on-screen order ({week_label}):
{clips_block}

Write:
TITLE: a punchy, clickable YouTube title, under 100 characters. Naming the biggest
streamer(s) usually helps; use a hook (a number, a strong verb, "insane"/"wild" etc.)
only where it actually fits what's in the clips above -- never oversell something
the description doesn't back up.
HOOK: 1-2 sentences for the very top of the description that make someone want to
keep watching. Specific to what's actually in these clips (name a streamer or a
moment), not a generic "check out this week's craziest moments!".

Answer in exactly this format, nothing else, no markdown:
TITLE: <title>
HOOK: <hook>"""


def generate_recap_hook(lineup: list, week_label: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> Optional[dict]:
    """A punchier, Claude-written title + opening hook than the plain
    deterministic one below -- best-effort: returns None on any failure
    (no ANTHROPIC_API_KEY, a network error, an unparseable response) so
    the recap can still ship with the deterministic fallback rather than
    blocking the whole build on this one call."""
    import re

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not lineup:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": _build_hook_prompt(lineup, week_label)}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        title_match = re.search(r"TITLE:\s*(.+)", text)
        hook_match = re.search(r"HOOK:\s*(.+)", text, re.S)
        if not title_match or not hook_match:
            return None
        title = title_match.group(1).strip().splitlines()[0][:100]
        hook = hook_match.group(1).strip()
        if not title or not hook:
            return None
        return {"title": title, "hook": hook}
    except Exception as e:
        print(f"[weekly_recap] AI title/hook generation failed, using the deterministic fallback: {e}", flush=True)
        return None


def build_recap_metadata(
    lineup: list, week_label: str, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
) -> dict:
    """A title + a chapter-formatted description (YouTube turns a
    description starting "0:00 ..." into clickable chapters). Tries a
    punchier Claude-written title/hook first (generate_recap_hook); the
    chapters themselves are always the same plain arithmetic over
    durations already known from each clip's own render, Claude or not,
    since those need to stay accurate, not "captivating".

    `lineup` arrives in actual on-screen (countdown) order -- weakest of
    the picks first, biggest hit last -- since the chapters below have to
    match the real video. The deterministic fallback title should still
    lead with the biggest names though, so it re-ranks a copy by view
    count rather than naming whoever happens to be picked #10."""
    names = [display_name(u) for u in sorted(lineup, key=lambda u: u.get("view_count", 0), reverse=True)]
    seen: set = set()
    ordered_names = [n for n in names if not (n in seen or seen.add(n))]

    ai = generate_recap_hook(lineup, week_label, api_key, model)
    if ai:
        title = ai["title"]
        intro = ai["hook"]
    else:
        title = f"Best Clips of the Week: {', '.join(ordered_names[:4])}"
        if len(ordered_names) > 4:
            title += " & more"
        title = title[:100]
        intro = f"This week's best clips from {', '.join(ordered_names)} ({week_label})."

    lines = [intro, ""]
    t = 0.0
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
