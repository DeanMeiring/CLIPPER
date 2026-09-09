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
from typing import List, Optional

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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
            params={"part": "statistics,snippet", "id": ",".join(video_ids), "key": api_key},
            timeout=15,
        )
        v_resp.raise_for_status()
        now = datetime.datetime.now(datetime.timezone.utc)
        for v in v_resp.json().get("items") or []:
            published_at = v["snippet"]["publishedAt"]
            published_dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            age_days = max(1.0, (now - published_dt).total_seconds() / 86400)
            views = int(v.get("statistics", {}).get("viewCount", 0))
            videos.append({
                "title": v["snippet"]["title"],
                "published_at": published_at,
                "views": views,
                "views_per_day": round(views / age_days, 1),
                "weekday": published_dt.weekday(),
            })

    best_day, views_per_day_by_weekday = _best_day_heuristic(videos)

    hidden_subs = channel.get("statistics", {}).get("hiddenSubscriberCount")
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
        "note": "Heuristic from public view counts, normalized by video age -- "
                "noisy, especially with few videos. Connect your YouTube account "
                "for real Analytics-based day-of-week performance and retention.",
    }


def _best_day_heuristic(videos: List[dict]):
    """Rank days by average views-per-day-since-published, not raw views --
    a video posted 6 months ago always has more raw views than one posted
    last week regardless of which day performs better, so normalizing by
    age is what makes the days comparable at all."""
    totals_by_weekday = [0.0] * 7
    counts_by_weekday = [0] * 7
    for v in videos:
        totals_by_weekday[v["weekday"]] += v["views_per_day"]
        counts_by_weekday[v["weekday"]] += 1
    avg_by_weekday = [
        round(totals_by_weekday[i] / counts_by_weekday[i], 1) if counts_by_weekday[i] else None
        for i in range(7)
    ]
    ranked = [i for i in range(7) if avg_by_weekday[i] is not None]
    best_day = DAY_NAMES[max(ranked, key=lambda i: avg_by_weekday[i])] if ranked else None
    return best_day, {DAY_NAMES[i]: avg_by_weekday[i] for i in range(7)}
