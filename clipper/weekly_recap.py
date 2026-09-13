"""Weekly cross-streamer recap: concatenate each tracked streamer's best-
performing Short from the past week into one long-form compilation video,
so the content gets a second life outside the Shorts feed.

"Best-performing" can only be measured for a clip that's actually been
posted to YouTube -- there's no view-count signal for a clip this app only
ever rendered. So the recap draws from clips already uploaded through this
app's own "Upload to YouTube" button that week, not from every clip
generated. record_upload() is the write side of that log (called right
after a real upload succeeds); everything else here reads it back to build
the compilation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# Keeps the compilation's length sane as more streamers get tracked -- 2
# clips each stays a reasonable ~5-10 clip video up to 5 streamers; past
# that, 1 each keeps a 10+ streamer roster from turning into a 20+ clip,
# 20-minute video nobody asked for.
_MAX_STREAMERS_FOR_TWO_EACH = 5
_RECAP_WINDOW_DAYS = 7.0
_MAX_LOG_ENTRIES = 1000


def _load_raw(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def record_upload(path: Path, entry: dict) -> None:
    """Append one real upload to the log -- called right after a clip's
    own "Upload to YouTube" click succeeds, so the log always reflects
    what's actually live rather than what the pipeline merely rendered."""
    entries = _load_raw(path)
    entries.append(entry)
    entries = entries[-_MAX_LOG_ENTRIES:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")
    except OSError:
        pass  # best-effort -- losing the log entry shouldn't fail the upload that produced it


def uploads_in_window(path: Path, days: float = _RECAP_WINDOW_DAYS, now: Optional[float] = None) -> list:
    now = now if now is not None else time.time()
    cutoff = now - days * 86400
    return [e for e in _load_raw(path) if (e.get("uploaded_at") or 0) >= cutoff]


def pick_weekly_lineup(window_uploads: list, views_by_video_id: dict) -> list:
    """Group this week's uploads by streamer, rank each streamer's own
    clips by view count, and take the top N per streamer -- N scaled down
    as the number of streamers with an upload this week grows, so the
    compilation doesn't get longer just because more streamers are
    tracked. An upload with no known streamer (the source VOD's
    broadcaster login couldn't be determined) is left out entirely rather
    than lumped into one fake "unknown streamer" group."""
    by_streamer: dict = {}
    for u in window_uploads:
        login = (u.get("streamer_login") or "").strip().lower()
        if not login:
            continue
        by_streamer.setdefault(login, []).append(u)

    per_streamer = 2 if len(by_streamer) <= _MAX_STREAMERS_FOR_TWO_EACH else 1

    lineup = []
    for login, uploads in by_streamer.items():
        ranked = sorted(uploads, key=lambda u: views_by_video_id.get(u.get("video_id"), 0), reverse=True)
        for u in ranked[:per_streamer]:
            lineup.append({**u, "streamer_login": login, "views": views_by_video_id.get(u.get("video_id"), 0)})

    # Biggest hit first -- a compilation's opening clip is what decides
    # whether someone keeps watching, same as any other Short.
    lineup.sort(key=lambda u: u["views"], reverse=True)
    return lineup


def get_video_view_counts(video_ids: list, api_key: str) -> dict:
    """Current view count for each id, via the public Data API (no OAuth
    needed -- these are the creator's own already-public uploads). An id
    that fails to resolve (deleted, or the lookup errored) is simply
    absent from the result rather than defaulted to 0, so a lookup hiccup
    doesn't make a clip look like it flopped."""
    import requests

    result: dict = {}
    if not video_ids or not api_key:
        return result
    for i in range(0, len(video_ids), 50):  # videos.list caps at 50 ids/call
        batch = video_ids[i:i + 50]
        try:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={"part": "statistics", "id": ",".join(batch), "key": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            for v in resp.json().get("items") or []:
                result[v["id"]] = int(v.get("statistics", {}).get("viewCount", 0))
        except Exception as e:
            print(f"[weekly_recap] view-count lookup failed for a batch: {e}", flush=True)
    return result


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


def _display_name(entry: dict) -> str:
    # Best-effort only -- the log stores a streamer's Twitch LOGIN (always
    # lowercase), not their real display-name casing, so a camelCase name
    # like "TheBurntPeanut" comes back as "Theburntpeanut" here. Fixing
    # that would mean an extra Twitch API call per streamer just for
    # cosmetics; not worth it for a chapters list.
    login = entry.get("streamer_login") or ""
    return login.replace("_", " ").title()


def build_recap_metadata(lineup: list, week_label: str) -> dict:
    """A deterministic title + a chapter-formatted description (YouTube
    turns a description starting "0:00 ..." into clickable chapters) --
    no Claude call needed, this is just arithmetic over durations already
    known from each clip's own upload record."""
    names = [_display_name(u) for u in lineup]
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
        lines.append(f"{minutes}:{seconds:02d} {_display_name(u)} -- {u.get('title', '')}")
        t += float(u.get("duration") or 0.0)
    description = "\n".join(lines)[:5000]
    return {"title": title, "description": description}
