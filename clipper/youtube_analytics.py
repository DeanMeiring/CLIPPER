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
