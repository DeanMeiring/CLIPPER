"""Real per-channel YouTube Analytics for the connected (OAuth'd) channel --
day-of-week performance, retention, and traffic-source breakdown from the
channel owner's own authenticated data, not the public Data API.

Note: the Analytics API has no "hour of day" dimension on regular reports
-- the audience-activity heatmap YouTube Studio shows isn't exposed via any
public API, official or otherwise. "Best time to post" here means best DAY
of the week, derived from real views/watch-time on the channel's own
videos -- the accurate ceiling of what's obtainable through official APIs,
not a substitute for the literal Studio graph.
"""
from __future__ import annotations

import datetime
from typing import Optional

ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get(url: str, access_token: str, params: dict) -> dict:
    import requests

    resp = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_own_channel(access_token: str) -> Optional[dict]:
    """The connected account's own channel id + display name."""
    data = _get(CHANNELS_URL, access_token, {"part": "id,snippet", "mine": "true"})
    items = data.get("items") or []
    if not items:
        return None
    return {"id": items[0]["id"], "title": items[0]["snippet"]["title"]}


def get_video_retention(access_token: str, channel_id: str, lookback_days: int = 90, max_videos: int = 50) -> dict:
    """Per-video views/retention over the last `lookback_days`, keyed by
    video id. This is what tells apart two different problems that both
    just look like "few views": nobody clicked it (a reach/hook problem --
    low views, retention doesn't matter yet) vs. people clicked but didn't
    stick around (a content/pacing problem -- decent views, weak
    retention). A video with strong retention but weak view count is a
    reach problem, not a quality one -- exactly the gap a creator's own
    gut sense of "this one was good" often can't see from view count
    alone."""
    import requests

    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)
    params = {
        "ids": f"channel=={channel_id}",
        "startDate": start.isoformat(), "endDate": end.isoformat(),
        "metrics": "views,averageViewDuration,averageViewPercentage,subscribersGained",
        "dimensions": "video",
        "sort": "-views",
        "maxResults": max_videos,
    }
    try:
        data = _get(ANALYTICS_URL, access_token, params)
    except requests.HTTPError as e:
        # Subscribers per video is a bonus; if the API ever refuses it,
        # keep the views and retention everything else depends on.
        if e.response is None or e.response.status_code != 400:
            raise
        data = _get(ANALYTICS_URL, access_token, {**params, "metrics": "views,averageViewDuration,averageViewPercentage"})
    result: dict = {}
    for row in data.get("rows") or []:
        video_id, views, avg_duration, avg_pct = row[0], row[1], row[2], row[3]
        result[video_id] = {
            "views": views,
            "average_view_duration_seconds": avg_duration,
            "average_view_percentage": avg_pct,
            "subscribers_gained": row[4] if len(row) > 4 else None,
        }
    return result


def get_video_engagement(access_token: str, channel_id: str, lookback_days: int = 365, max_videos: int = 200) -> dict:
    """Per-video engagedViews, likes, comments and shares, keyed by video id.

    Since March 2025 a Shorts "view" counts every start or replay, however
    short; engagedViews keeps the old count (watched past the first
    moments). engagedViews / views is the closest the API gets to Studio's
    "viewed vs swiped away" -- the signal the Shorts feed weighs most.
    Shares and comments are the next-strongest signals."""
    import requests

    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)
    params = {
        "ids": f"channel=={channel_id}",
        "startDate": start.isoformat(), "endDate": end.isoformat(),
        "metrics": "engagedViews,likes,comments,shares",
        "dimensions": "video",
        "sort": "-likes",
        "maxResults": max_videos,
    }
    names = ["engaged_views", "likes", "comments", "shares"]
    try:
        data = _get(ANALYTICS_URL, access_token, params)
    except requests.HTTPError as e:
        # An account or API version without engagedViews still has the rest.
        if e.response is None or e.response.status_code != 400:
            raise
        names = names[1:]
        data = _get(ANALYTICS_URL, access_token, {**params, "metrics": "likes,comments,shares"})
    result: dict = {}
    for row in data.get("rows") or []:
        result[row[0]] = {name: row[i + 1] for i, name in enumerate(names) if i + 1 < len(row)}
    return result


def get_daily_totals(access_token: str, channel_id: str, days: int = 63) -> list:
    """Channel-wide views and subscribers per day, oldest first:
    [{"date": "2026-09-01", "views": 1200, "subscribers_gained": 3,
    "subscribers_lost": 0}, ...]. YouTube's numbers run about two days
    behind, so the most recent days are simply absent."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    data = _get(ANALYTICS_URL, access_token, {
        "ids": f"channel=={channel_id}",
        "startDate": start.isoformat(), "endDate": end.isoformat(),
        "metrics": "views,subscribersGained,subscribersLost",
        "dimensions": "day",
        "sort": "day",
    })
    return [
        {"date": r[0], "views": r[1], "subscribers_gained": r[2], "subscribers_lost": r[3]}
        for r in data.get("rows") or []
    ]


def get_retention_curve(access_token: str, channel_id: str, video_id: str, start_date: str) -> list:
    """One video's audience-retention curve: 100 points across the video's
    length, each [elapsed fraction, audienceWatchRatio,
    relativeRetentionPerformance]. audienceWatchRatio is viewers watching
    at that point per view (a replayed Short can push it above 1);
    relativeRetentionPerformance ranks that point against YouTube videos
    of similar length (0.5 = typical). Empty list when YouTube doesn't have
    enough views on the video to report a curve yet. The third value is
    None if YouTube rejects the comparison metric -- the watch ratio alone
    still gives the curve."""
    import requests

    params = {
        "ids": f"channel=={channel_id}",
        "startDate": start_date,
        "endDate": datetime.date.today().isoformat(),
        "dimensions": "elapsedVideoTimeRatio",
        "filters": f"video=={video_id}",
    }
    try:
        data = _get(ANALYTICS_URL, access_token, {**params, "metrics": "audienceWatchRatio,relativeRetentionPerformance"})
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code != 400:
            raise
        data = _get(ANALYTICS_URL, access_token, {**params, "metrics": "audienceWatchRatio"})
    rows = [
        [float(r[0]), float(r[1]), float(r[2]) if len(r) > 2 and r[2] is not None else None]
        for r in (data.get("rows") or [])
    ]
    rows.sort(key=lambda r: r[0])
    return rows


def get_insights(access_token: str, channel_id: str, lookback_days: int = 90) -> dict:
    """Best day-of-week (by views), retention, and top traffic sources over
    the last `lookback_days`, from the channel's real Analytics data."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)
    start_date, end_date = start.isoformat(), end.isoformat()
    base_params = {"ids": f"channel=={channel_id}", "startDate": start_date, "endDate": end_date}

    by_day = _get(ANALYTICS_URL, access_token, {
        **base_params,
        "metrics": "views",
        "dimensions": "day",
        "sort": "day",
    })
    totals_by_weekday = [0.0] * 7
    for row in by_day.get("rows") or []:
        date_str, views = row[0], row[1]
        weekday = datetime.date.fromisoformat(date_str).weekday()
        totals_by_weekday[weekday] += views
    has_data = any(totals_by_weekday)
    best_day = DAY_NAMES[max(range(7), key=lambda i: totals_by_weekday[i])] if has_data else None
    views_by_day = {DAY_NAMES[i]: round(totals_by_weekday[i]) for i in range(7)}

    totals = _get(ANALYTICS_URL, access_token, {
        **base_params,
        "metrics": "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost",
    })
    totals_row = (totals.get("rows") or [[0, 0, 0, 0.0, 0, 0]])[0]

    traffic = _get(ANALYTICS_URL, access_token, {
        **base_params,
        "metrics": "views",
        "dimensions": "insightTrafficSourceType",
        "sort": "-views",
    })
    traffic_sources = [{"source": r[0], "views": r[1]} for r in (traffic.get("rows") or [])[:6]]

    return {
        "lookback_days": lookback_days,
        "best_day": best_day,
        "views_by_day": views_by_day,
        "total_views": totals_row[0],
        "estimated_minutes_watched": totals_row[1],
        "average_view_duration_seconds": totals_row[2],
        "average_view_percentage": totals_row[3],
        "subscribers_gained": totals_row[4],
        "subscribers_lost": totals_row[5],
        "traffic_sources": traffic_sources,
    }


def get_video_stats(access_token: str, channel_id: str, video_id: str, start_date: str) -> dict:
    """One video's totals since start_date: views, watch time, average view
    duration / percentage, subscribers, likes, comments, shares -- plus
    thumbnail impressions and click-through rate when YouTube reports
    them for the video (they're None otherwise)."""
    import requests

    base = {"ids": f"channel=={channel_id}", "startDate": start_date,
            "endDate": datetime.date.today().isoformat(), "filters": f"video=={video_id}"}
    names = ["views", "estimatedMinutesWatched", "averageViewDuration", "averageViewPercentage",
             "subscribersGained", "likes", "comments", "shares"]
    data = _get(ANALYTICS_URL, access_token, {**base, "metrics": ",".join(names)})
    row = (data.get("rows") or [[0] * len(names)])[0]
    out = dict(zip(names, row))
    out["impressions"] = out["click_rate"] = None
    try:
        imp = _get(ANALYTICS_URL, access_token, {**base, "metrics": "videoThumbnailImpressions,videoThumbnailImpressionsClickRate"})
        r = (imp.get("rows") or [None])[0]
        if r:
            out["impressions"], out["click_rate"] = r[0], r[1]
    except requests.HTTPError:
        pass  # not reported for this video or this account; the rest stands
    return out


def get_video_daily(access_token: str, channel_id: str, video_id: str, start_date: str) -> list:
    """One video's views per day since start_date, oldest first:
    [{"date", "views"}]. YouTube's numbers run about two days behind."""
    data = _get(ANALYTICS_URL, access_token, {
        "ids": f"channel=={channel_id}", "startDate": start_date, "endDate": datetime.date.today().isoformat(),
        "metrics": "views", "dimensions": "day", "sort": "day", "filters": f"video=={video_id}",
    })
    return [{"date": r[0], "views": r[1]} for r in data.get("rows") or []]


def get_video_traffic(access_token: str, channel_id: str, video_id: str, start_date: str) -> list:
    """Where one video's views came from: [{"source", "views"}], biggest
    first (YouTube's insightTrafficSourceType names)."""
    data = _get(ANALYTICS_URL, access_token, {
        "ids": f"channel=={channel_id}", "startDate": start_date, "endDate": datetime.date.today().isoformat(),
        "metrics": "views", "dimensions": "insightTrafficSourceType", "sort": "-views", "filters": f"video=={video_id}",
    })
    return [{"source": r[0], "views": r[1]} for r in data.get("rows") or []]
