"""Discover YouTube channels actively posting clips of a given streamer, so
a creator can pick a handful as comparison points for the AI strategy
overview without already knowing who their competitors are.

Uses the public Data API (YOUTUBE_API_KEY, no OAuth) search.list -- the
most expensive call this app makes (100 quota units per call, versus ~1 for
most others, out of a default 10,000/day project budget). This only ever
runs when a person explicitly searches for a streamer, never automatically
or on a timer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CompetitorCandidate:
    channel_id: str
    channel_title: str
    thumbnail: Optional[str]
    subscriber_count: Optional[int]
    # The specific matching video that surfaced this channel, and its view
    # count -- both the reason it's ranked where it is and a concrete thing
    # to show in the picker instead of just a bare channel name.
    sample_video_title: str
    sample_video_views: int


def _looks_like_own_channel(channel_title: str, streamer_name: str) -> bool:
    """Skip a result that's just the streamer's own official channel --
    searching "jynxzi clips" surfaces jynxzi's own uploads too, and those
    aren't a competitor clipping channel in the sense this search is for."""
    a = re.sub(r"[^a-z0-9]", "", channel_title.lower())
    b = re.sub(r"[^a-z0-9]", "", streamer_name.lower())
    return bool(b) and (b in a or a in b)


def search_clipping_channels(streamer_name: str, max_channels: int = 8) -> List[CompetitorCandidate]:
    """Channels other than the streamer's own that are getting views
    clipping them, ranked by their single best-performing matching video.
    Empty (not an error) if YOUTUBE_API_KEY isn't set or nothing usable
    came back -- callers should show that as "no results", not fail."""
    streamer_name = streamer_name.strip()
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not streamer_name or not api_key:
        return []

    import requests

    search_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": f"{streamer_name} clips",
            "type": "video",
            "order": "viewCount",
            "maxResults": 25,
            "key": api_key,
        },
        timeout=15,
    )
    search_resp.raise_for_status()
    items = search_resp.json().get("items") or []
    video_ids = [i["id"]["videoId"] for i in items if i.get("id", {}).get("videoId")]
    if not video_ids:
        return []

    # search.list doesn't return view counts -- a second batched call
    # against the matched video ids is what actually ranks channels against
    # each other, not just against YouTube's own relevance ordering.
    videos_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "statistics,snippet", "id": ",".join(video_ids), "key": api_key},
        timeout=15,
    )
    videos_resp.raise_for_status()

    best_by_channel: dict = {}
    for v in videos_resp.json().get("items") or []:
        snippet = v.get("snippet") or {}
        channel_id = snippet.get("channelId")
        channel_title = snippet.get("channelTitle") or ""
        if not channel_id or _looks_like_own_channel(channel_title, streamer_name):
            continue
        views = int((v.get("statistics") or {}).get("viewCount", 0))
        existing = best_by_channel.get(channel_id)
        if existing is None or views > existing["views"]:
            best_by_channel[channel_id] = {
                "channel_title": channel_title,
                "views": views,
                "video_title": snippet.get("title", ""),
            }

    ranked = sorted(best_by_channel.items(), key=lambda kv: kv[1]["views"], reverse=True)[:max_channels]
    if not ranked:
        return []

    channels_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet,statistics", "id": ",".join(cid for cid, _ in ranked), "key": api_key},
        timeout=15,
    )
    channels_resp.raise_for_status()
    channel_info = {c["id"]: c for c in channels_resp.json().get("items") or []}

    candidates: List[CompetitorCandidate] = []
    for channel_id, best in ranked:
        info = channel_info.get(channel_id) or {}
        thumbs = (info.get("snippet") or {}).get("thumbnails") or {}
        thumbnail = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
        stats = info.get("statistics") or {}
        hidden = stats.get("hiddenSubscriberCount")
        candidates.append(CompetitorCandidate(
            channel_id=channel_id,
            channel_title=best["channel_title"],
            thumbnail=thumbnail,
            subscriber_count=None if hidden or "subscriberCount" not in stats else int(stats["subscriberCount"]),
            sample_video_title=best["video_title"],
            sample_video_views=best["views"],
        ))
    return candidates
