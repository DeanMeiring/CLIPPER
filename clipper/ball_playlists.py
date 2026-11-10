"""Playlists per series on the Ball Evolution channel (10 Nov).

From the channel research: the fast-growing ball channels run a few named
series, and YouTube's Shorts "Series" (episodes, "watch next") is built on
playlists, for channels in the Partner Program. So every upload goes into
its series' playlist, made on first use.

Playlists need the ``youtube.force-ssl`` scope (``youtube.upload`` can't
make one), which the Ball Evolution login asks for since this change: the
channel works without it, and ``can_use`` says when a reconnect is needed.
Each playlists.insert / playlistItems.insert costs 50 quota units.
"""

from __future__ import annotations

from typing import Optional

API = "https://www.googleapis.com/youtube/v3"
SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"

# kind -> series key; the rest go to "physics"
SERIES_OF = {"box": "box", "duel": "league", "crowd": "fights", "zen": "zen", "compilation": "weekly"}
SERIES = {
    "box": ("Fill-up boxes", "Every ball that escapes adds three more. How fast does it fill up?"),
    "league": ("Ball League", "The ball fights: champions, challengers and rematches, with records."),
    "fights": ("Comment fights", "Commenters' names as the fighters. Comment under a video to get in."),
    "zen": ("Spike rings", "Touch a spike and you stay. Which ball finds the way out?"),
    "physics": ("Ball physics", "Bouncing balls, rings and gaps: every other ball video."),
    "weekly": ("Every video, weekly", "All of the week's ball videos in one."),
}


def series_for(kind: str) -> str:
    return SERIES_OF.get(kind, "physics")


def can_use(token: Optional[dict]) -> bool:
    """Whether the stored token was granted the scope playlists need."""
    return bool(token) and SCOPE in (token.get("scope") or "").split()


def _post(access_token: str, path: str, part: str, body: dict) -> dict:
    import requests

    resp = requests.post(f"{API}/{path}", params={"part": part}, json=body,
                         headers={"Authorization": f"Bearer {access_token}"}, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"YouTube {path}: {resp.status_code} {resp.text[:200]}")
    return resp.json()


def create(access_token: str, series: str) -> str:
    title, desc = SERIES[series]
    data = _post(access_token, "playlists", "snippet,status",
                 {"snippet": {"title": title, "description": desc}, "status": {"privacyStatus": "public"}})
    return data["id"]


def add(access_token: str, playlist_id: str, video_id: str) -> None:
    _post(access_token, "playlistItems", "snippet",
          {"snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": video_id}}})


def file_video(access_token: str, ids: dict, kind: str, video_id: str) -> tuple:
    """Put video_id in its series' playlist, making the playlist when it's
    the series' first. Returns (series, ids) -- ids updated with any new
    playlist, for the caller to save."""
    series = series_for(kind)
    ids = dict(ids or {})
    if not ids.get(series):
        ids[series] = create(access_token, series)
    try:
        add(access_token, ids[series], video_id)
    except RuntimeError as e:
        if " 404 " not in str(e):
            raise
        ids[series] = create(access_token, series)      # deleted in Studio: make it again
        add(access_token, ids[series], video_id)
    return series, ids
