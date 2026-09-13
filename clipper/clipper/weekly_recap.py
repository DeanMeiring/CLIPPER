"""Weekly cross-streamer recap: concatenate each tracked streamer's most-
viewed Twitch clip from the past week into one long-form compilation
video, so the content gets a second life outside the Shorts feed.

Source material is Twitch's own "Clips" feature (the ones made from the
Clip button on a stream, by the creator or by viewers) -- these are
already curated highlight moments with a real Twitch view count,
available immediately via trending.get_top_twitch_clips(), with no
dependency on this app having already rendered and uploaded something
for that streamer first. Each chosen clip still gets downloaded and run
through this app's own render pipeline (crop/facecam/captions -- see
webapp/main.py's _render_twitch_clip_for_recap) so the compilation looks
consistent with the rest of the channel, since a raw Twitch clip is
plain landscape footage with no captions of its own.
"""
from __future__ import annotations

from pathlib import Path

# Keeps the compilation's length sane as more streamers get tracked -- 2
# clips each stays a reasonable ~5-10 clip video up to 5 streamers; past
# that, 1 each keeps a 10+ streamer roster from turning into a 20+ clip,
# 20-minute video nobody asked for.
_MAX_STREAMERS_FOR_TWO_EACH = 5


def per_streamer_count(num_streamers: int) -> int:
    return 2 if num_streamers <= _MAX_STREAMERS_FOR_TWO_EACH else 1


def group_clips_by_streamer(clips: list) -> dict:
    """Groups Twitch clips by streamer, each group sorted by view count
    descending. Returns {login: [clip, clip, ...]} rather than an already
    top-N-truncated list, so a caller can walk each streamer's list in
    view-count order and fall through to the next one if a candidate
    fails to download or render -- one broken clip shouldn't silently
    drop that streamer from the recap entirely. A clip with no streamer
    login is left out rather than lumped into a fake "unknown" group."""
    by_streamer: dict = {}
    for c in clips:
        login = (c.get("streamer_login") or "").strip().lower()
        if not login:
            continue
        by_streamer.setdefault(login, []).append(c)
    for login in by_streamer:
        by_streamer[login].sort(key=lambda c: c.get("view_count") or 0, reverse=True)
    return by_streamer


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


def build_recap_metadata(lineup: list, week_label: str) -> dict:
    """A deterministic title + a chapter-formatted description (YouTube
    turns a description starting "0:00 ..." into clickable chapters) --
    no Claude call needed, this is just arithmetic over durations already
    known from each clip's own render."""
    names = [display_name(u) for u in lineup]
    seen: set = set()
    ordered_names = [n for n in names if not (n in seen or seen.add(n))]

    title = f"Best Clips of the Week: {', '.join(ordered_names[:4])}"
    if len(ordered_names) > 4:
        title += " & more"
    title = title[:100]

    lines = [f"This week's best clips from {', '.join(ordered_names)} ({week_label}).", ""]
    t = 0.0
    for u in lineup:
        minutes, seconds = divmod(int(t), 60)
        lines.append(f"{minutes}:{seconds:02d} {display_name(u)} -- {u.get('title', '')}")
        t += float(u.get("duration") or 0.0)
    description = "\n".join(lines)[:5000]
    return {"title": title, "description": description}
