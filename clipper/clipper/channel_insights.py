"""No-auth channel snapshot + a rough best-day-to-post heuristic, built
entirely from the public YouTube Data API (the same YOUTUBE_API_KEY already
used for trending lookups) -- works immediately, no OAuth setup required.

Much noisier than the real Analytics-based version in youtube_analytics.py:
raw view counts are skewed by how long each video has been up, not just how
well it performed, so this normalizes by video age to make days comparable
at all -- still a heuristic, not a substitute for real Analytics data.
"""
from __future__ import annotations

import datetime
import os
import re
from typing import List, Optional

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Below this age a video's numbers say more about how long it's been up than
# how it performed -- short-form earns most of its views in the first day or
# two, so anything younger is reported but kept out of the performance
# comparison and the best-day averages rather than counted as a flop.
_MIN_AGE_DAYS_TO_JUDGE = 2.0

# YouTube's own eligibility ceiling for a video to actually be treated as a
# Short (feed placement, the Shorts shelf, etc.), not this app's guess --
# raised from the original 60s in 2024. Everything this app renders is
# comfortably under this, and every insight/comparison here is meant to be
# Shorts-vs-Shorts, so a channel's occasional long-form upload is excluded
# rather than silently diluting the "what's working" picture with a video
# in a different format nobody here is posting.
MAX_SHORT_SECONDS = 180

_ISO8601_DURATION_RE = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$")


def _parse_iso8601_duration(duration: str) -> Optional[float]:
    """YouTube's contentDetails.duration is ISO 8601 (e.g. "PT1M30S",
    "PT47S", "PT2H"). Returns None for anything that doesn't match rather
    than guessing -- a video whose length can't be read is left out of the
    Shorts/long-form split entirely instead of being miscounted as either."""
    if not duration:
        return None
    m = _ISO8601_DURATION_RE.match(duration.strip())
    if not m:
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return float(hours * 3600 + minutes * 60 + seconds)


def _channel_lookup_params(channel_id_or_handle: str) -> dict:
    if channel_id_or_handle.startswith("UC"):
        return {"id": channel_id_or_handle}
    if channel_id_or_handle.startswith("@"):
        return {"forHandle": channel_id_or_handle}
    return {"forUsername": channel_id_or_handle}


def get_channel_snapshot(channel_id_or_handle: str, sample_size: int = 25) -> Optional[dict]:
    import requests

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return None

    params = {"part": "snippet,statistics,contentDetails", "key": api_key}
    params.update(_channel_lookup_params(channel_id_or_handle))
    resp = requests.get("https://www.googleapis.com/youtube/v3/channels", params=params, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("items") or []
    if not items:
        return None
    channel = items[0]
    stats = channel.get("statistics", {})
    uploads_playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]

    pl_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/playlistItems",
        params={"part": "snippet", "playlistId": uploads_playlist, "maxResults": sample_size, "key": api_key},
        timeout=15,
    )
    pl_resp.raise_for_status()
    video_ids = [i["snippet"]["resourceId"]["videoId"] for i in pl_resp.json().get("items") or []]

    videos: List[dict] = []
    if video_ids:
        v_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "statistics,snippet,status,contentDetails", "id": ",".join(video_ids), "key": api_key},
            timeout=15,
        )
        v_resp.raise_for_status()
        now = datetime.datetime.now(datetime.timezone.utc)
        long_form_skipped = 0
        for v in v_resp.json().get("items") or []:
            duration_seconds = _parse_iso8601_duration((v.get("contentDetails") or {}).get("duration", ""))
            # Everything this app compares is meant to be Shorts-vs-Shorts --
            # an occasional long-form upload mixed into a channel's recent
            # videos would otherwise sit in the same ranking/best-day
            # average as actual Shorts despite competing in a completely
            # different format. A duration that couldn't be parsed is kept
            # rather than guessed at either way.
            if duration_seconds is not None and duration_seconds > MAX_SHORT_SECONDS:
                long_form_skipped += 1
                continue
            published_at = v["snippet"]["publishedAt"]
            published_dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            age_days = max(1.0, (now - published_dt).total_seconds() / 86400)
            views = int(v.get("statistics", {}).get("viewCount", 0))
            videos.append({
                "id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": published_at,
                "views": views,
                "views_per_day": round(views / age_days, 1),
                "age_days": round(age_days, 1),
                "duration_seconds": duration_seconds,
                # A just-posted video hasn't had time to earn its views yet,
                # and the max(1.0, ...) floor above actively understates it:
                # something posted 2 hours ago is scored as if a full day had
                # passed, so a clip pacing at 600 views/day reads as 50/day.
                # Left unmarked, those land in the "lower-performing" half and
                # get diagnosed as a reach or content failure purely for being
                # new -- on exactly the uploads a creator is most likely to be
                # asking about.
                "too_new_to_judge": age_days < _MIN_AGE_DAYS_TO_JUDGE,
                "weekday": published_dt.weekday(),
                "restriction": _restriction_note(v),
            })

    best_day, views_per_day_by_weekday = _best_day_heuristic(videos)

    hidden_subs = channel.get("statistics", {}).get("hiddenSubscriberCount")
    note = ("Heuristic from public view counts, normalized by video age -- "
            "noisy, especially with few videos. Connect your YouTube account "
            "for real Analytics-based day-of-week performance and retention.")
    if long_form_skipped:
        note += f" ({long_form_skipped} longer-than-Shorts upload(s) excluded from this list.)"
    return {
        "channel_title": channel["snippet"]["title"],
        "subscriber_count": None if hidden_subs else int(stats.get("subscriberCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "total_view_count": int(stats.get("viewCount", 0)),
        "recent_videos_sampled": len(videos),
        "recent_videos": videos,
        "avg_views_per_day_recent": round(sum(v["views_per_day"] for v in videos) / len(videos), 1) if videos else None,
        "best_day_heuristic": best_day,
        "views_per_day_by_weekday": views_per_day_by_weekday,
        "note": note,
    }


def _restriction_note(video: dict) -> Optional[str]:
    """A video stuck at near-zero views sometimes isn't a content or
    packaging problem at all -- it's actually unavailable to most viewers,
    which looks identical to "nobody wanted to watch it" from view count
    alone. Surfacing this from the Data API's own public status/
    contentDetails fields (no extra quota cost -- already fetching this
    video by id) lets the AI overview name the real cause instead of
    inventing a content explanation for a platform-level block."""
    status = video.get("status") or {}
    content_details = video.get("contentDetails") or {}
    notes = []

    privacy = status.get("privacyStatus")
    if privacy and privacy != "public":
        notes.append(f"not public ({privacy})")

    upload_status = status.get("uploadStatus")
    if upload_status and upload_status not in ("processed", "uploaded"):
        notes.append(f"upload status: {upload_status}")

    region = content_details.get("regionRestriction") or {}
    blocked = region.get("blocked")
    allowed = region.get("allowed")
    if blocked:
        sample = ", ".join(blocked[:5])
        notes.append(f"blocked in {len(blocked)} countries ({sample}{'...' if len(blocked) > 5 else ''})")
    elif allowed:
        notes.append(f"only viewable in {len(allowed)} countries")

    rating = content_details.get("contentRating") or {}
    if rating.get("ytRating") == "ytAgeRestricted":
        notes.append("age-restricted (18+, signed-out/limited reach)")

    return "; ".join(notes) if notes else None


def _best_day_heuristic(videos: List[dict]):
    """Rank days by average views-per-day-since-published, not raw views --
    a video posted 6 months ago always has more raw views than one posted
    last week regardless of which day performs better, so normalizing by
    age is what makes the days comparable at all."""
    totals_by_weekday = [0.0] * 7
    counts_by_weekday = [0] * 7
    for v in videos:
        if v.get("too_new_to_judge"):
            continue  # its score reflects its age, not the day it went up
        totals_by_weekday[v["weekday"]] += v["views_per_day"]
        counts_by_weekday[v["weekday"]] += 1
    avg_by_weekday = [
        round(totals_by_weekday[i] / counts_by_weekday[i], 1) if counts_by_weekday[i] else None
        for i in range(7)
    ]
    ranked = [i for i in range(7) if avg_by_weekday[i] is not None]
    best_day = DAY_NAMES[max(ranked, key=lambda i: avg_by_weekday[i])] if ranked else None
    return best_day, {DAY_NAMES[i]: avg_by_weekday[i] for i in range(7)}
