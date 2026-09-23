"""Every clip this app renders, kept on disk independently of the job that
made it -- deleting a job after downloading its clips (the normal flow)
would otherwise take the only record of what each clip looked like with
it, and there'd be nothing left to join the posted video's performance
against.

A clip gets linked to its YouTube video either directly (uploaded through
the app's button) or by title match (downloaded and posted by hand) --
see match_uploads.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

_lock = threading.Lock()

# How close a posted title has to be to the clip's generated upload_title
# to count as the same clip -- loose enough to survive a hand-edited word
# or two, tight enough that two clips from the same stream don't swap.
_TITLE_MATCH_THRESHOLD = 0.75
# A posted video can be a bit shorter than the render (trimmed before
# posting) but never meaningfully longer. Posted durations are whole
# seconds, so allow a little rounding either way.
_MAX_EXTRA_SECONDS = 2.0
_MAX_TRIMMED_SECONDS = 15.0


def clip_id_for(job_id: str, start: float, end: float, window_index: Optional[int] = None) -> str:
    """Stable id for one clip, derivable from what a job already stores --
    so a clip rendered before the registry existed gets the same id every
    time it's backfilled, and a facecam re-render (same start/end) updates
    the same record instead of making a new one."""
    key = f"{job_id}|{window_index}|{start:.2f}|{end:.2f}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: Path, records: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(records), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"[clip_registry] could not save {path.name}: {e}", flush=True)


def all_records(path: Path) -> List[dict]:
    with _lock:
        return list(_load(path).values())


def get_record(path: Path, clip_id: str) -> Optional[dict]:
    with _lock:
        return _load(path).get(clip_id)


def upsert(path: Path, record: dict) -> None:
    """Insert or update one clip record. A re-render keeps the existing
    link to a posted video -- only the fields passed in are replaced."""
    with _lock:
        records = _load(path)
        existing = records.get(record["clip_id"], {})
        merged = {**existing, **record}
        for key in ("video_id", "link_method", "linked_at", "posted_duration", "created_at"):
            if existing.get(key) is not None:
                merged[key] = existing[key]
        records[record["clip_id"]] = merged
        _save(path, records)


def update_fields(path: Path, clip_id: str, **fields) -> None:
    with _lock:
        records = _load(path)
        if clip_id in records:
            records[clip_id].update(fields)
            _save(path, records)


def link_video(
    path: Path, clip_id: str, video_id: str, method: str, posted_duration: Optional[float] = None,
) -> None:
    update_fields(
        path, clip_id,
        video_id=video_id, link_method=method, linked_at=time.time(), posted_duration=posted_duration,
    )


_HASHTAG_RE = re.compile(r"#\w+")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title: str) -> str:
    text = _HASHTAG_RE.sub(" ", (title or "").lower())
    text = _NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def match_uploads(records: List[dict], videos: List[dict]) -> List[tuple]:
    """Pair unlinked clip records with posted videos nothing is linked to
    yet, by title similarity with a duration and posting-time sanity check.
    Each video and each clip is used at most once, best match first.

    `videos` entries need id, title, duration_seconds, published_ts.
    Returns [(clip_id, video_id, posted_duration), ...]."""
    taken_videos = {r["video_id"] for r in records if r.get("video_id")}
    unlinked = [r for r in records if not r.get("video_id")]
    candidates = []
    for v in videos:
        if v["id"] in taken_videos:
            continue
        video_title = _normalize_title(v.get("title", ""))
        if not video_title:
            continue
        posted = v.get("duration_seconds")
        for r in unlinked:
            # Can't have been posted before it existed (small allowance for
            # clock skew between this server and YouTube).
            if v.get("published_ts") and r.get("created_at") and v["published_ts"] < r["created_at"] - 300:
                continue
            clip_duration = r.get("duration")
            if posted is not None and clip_duration is not None:
                if posted > clip_duration + _MAX_EXTRA_SECONDS or posted < clip_duration - _MAX_TRIMMED_SECONDS:
                    continue
            clip_title = _normalize_title(r.get("upload_title") or r.get("title") or "")
            if not clip_title:
                continue
            score = difflib.SequenceMatcher(None, video_title, clip_title).ratio()
            if score >= _TITLE_MATCH_THRESHOLD:
                candidates.append((score, v["id"], r["clip_id"], posted))

    candidates.sort(key=lambda c: c[0], reverse=True)
    used_videos, used_clips, matches = set(), set(), []
    for _score, video_id, clip_id, posted in candidates:
        if video_id in used_videos or clip_id in used_clips:
            continue
        used_videos.add(video_id)
        used_clips.add(clip_id)
        matches.append((clip_id, video_id, posted))
    return matches
