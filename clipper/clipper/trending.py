"""Look up creators' latest content (live Twitch streams, latest Twitch
VODs, latest YouTube uploads) so the web UI can offer one-click source
links instead of the user hunting down a URL themselves.

Configured via env vars, not hardcoded -- TRENDING_TWITCH_LOGINS and
TRENDING_YOUTUBE_CHANNELS (comma-separated Twitch logins / YouTube channel
IDs or @handles). Nothing is guessed here: an unconfigured platform just
returns no entries for it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CreatorEntry:
    platform: str          # "twitch" or "youtube"
    name: str               # login / handle, for display
    url: str
    title: str
    live: bool
    published_at: Optional[str]  # ISO timestamp, if known
    thumbnail: Optional[str]


_twitch_token: Optional[str] = None
_twitch_token_expires: float = 0.0


def _get_twitch_token() -> Optional[str]:
    global _twitch_token, _twitch_token_expires
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    if _twitch_token and time.time() < _twitch_token_expires:
        return _twitch_token

    import requests

    resp = requests.post(
        "https://id.twitch.tv/oauth2/token",
        data={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    _twitch_token = body["access_token"]
    _twitch_token_expires = time.time() + float(body.get("expires_in", 3600)) - 60
    return _twitch_token


def get_twitch_creators(logins: List[str]) -> List[CreatorEntry]:
    """One entry per login: the live stream if they're live, else their
    most recent VOD. Skips a login cleanly on any per-login failure so one
    bad name doesn't blank out the rest."""
    logins = [l.strip().lower() for l in logins if l.strip()]
    if not logins:
        return []
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    token = _get_twitch_token()
    if not client_id or not token:
        return []

    import requests

    headers = {"Client-Id": client_id, "Authorization": f"Bearer {token}"}
    entries: List[CreatorEntry] = []
    try:
        users_resp = requests.get(
            "https://api.twitch.tv/helix/users", params={"login": logins}, headers=headers, timeout=15,
        )
        users_resp.raise_for_status()
        users = {u["login"]: u for u in users_resp.json().get("data") or []}

        streams_resp = requests.get(
            "https://api.twitch.tv/helix/streams", params={"user_login": logins}, headers=headers, timeout=15,
        )
        streams_resp.raise_for_status()
        live_by_login = {s["user_login"].lower(): s for s in streams_resp.json().get("data") or []}
    except Exception as e:
        print(f"[trending] Twitch lookup failed: {e}", flush=True)
        return []

    for login in logins:
        user = users.get(login)
        if not user:
            print(f"[trending] Twitch login {login!r} not found -- skipping", flush=True)
            continue
        stream = live_by_login.get(login)
        try:
            if stream:
                entries.append(CreatorEntry(
                    platform="twitch", name=user.get("display_name", login),
                    url=f"https://www.twitch.tv/{login}",
                    title=stream.get("title", ""), live=True,
                    published_at=stream.get("started_at"),
                    thumbnail=(stream.get("thumbnail_url") or "").replace("{width}", "320").replace("{height}", "180"),
                ))
            else:
                videos_resp = requests.get(
                    "https://api.twitch.tv/helix/videos",
                    params={"user_id": user["id"], "type": "archive", "first": 1},
                    headers=headers, timeout=15,
                )
                videos_resp.raise_for_status()
                videos = videos_resp.json().get("data") or []
                if not videos:
                    continue
                v = videos[0]
                entries.append(CreatorEntry(
                    platform="twitch", name=user.get("display_name", login),
                    url=v["url"], title=v.get("title", ""), live=False,
                    published_at=v.get("published_at") or v.get("created_at"),
                    thumbnail=(v.get("thumbnail_url") or "").replace("%{width}", "320").replace("%{height}", "180"),
                ))
        except Exception as e:
            print(f"[trending] Twitch lookup for {login!r} failed: {e}", flush=True)
            continue

    return entries


def get_youtube_creators(channels: List[str]) -> List[CreatorEntry]:
    """One entry per channel: their most recent upload. `channels` entries
    may be a channel ID (UC...) or an @handle."""
    channels = [c.strip() for c in channels if c.strip()]
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not channels or not api_key:
        return []

    import requests

    entries: List[CreatorEntry] = []
    for channel in channels:
        try:
            params = {"part": "contentDetails,snippet", "key": api_key}
            if channel.startswith("@"):
                params["forHandle"] = channel
            elif channel.startswith("UC"):
                params["id"] = channel
            else:
                params["forHandle"] = f"@{channel}"

            chan_resp = requests.get(
                "https://www.googleapis.com/youtube/v3/channels", params=params, timeout=15,
            )
            chan_resp.raise_for_status()
            items = chan_resp.json().get("items") or []
            if not items:
                print(f"[trending] YouTube channel {channel!r} not found -- skipping", flush=True)
                continue
            item = items[0]
            uploads_playlist = item["contentDetails"]["relatedPlaylists"]["uploads"]
            display_name = item["snippet"]["title"]

            items_resp = requests.get(
                "https://www.googleapis.com/youtube/v3/playlistItems",
                params={"part": "snippet", "playlistId": uploads_playlist, "maxResults": 1, "key": api_key},
                timeout=15,
            )
            items_resp.raise_for_status()
            videos = items_resp.json().get("items") or []
            if not videos:
                continue
            snippet = videos[0]["snippet"]
            video_id = snippet["resourceId"]["videoId"]
            thumbs = snippet.get("thumbnails") or {}
            thumbnail = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")
            entries.append(CreatorEntry(
                platform="youtube", name=display_name,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=snippet.get("title", ""), live=False,
                published_at=snippet.get("publishedAt"),
                thumbnail=thumbnail,
            ))
        except Exception as e:
            print(f"[trending] YouTube lookup for {channel!r} failed: {e}", flush=True)
            continue

    return entries


def get_trending_creators() -> List[CreatorEntry]:
    twitch_logins = os.environ.get("TRENDING_TWITCH_LOGINS", "").split(",")
    youtube_channels = os.environ.get("TRENDING_YOUTUBE_CHANNELS", "").split(",")
    entries = get_twitch_creators(twitch_logins) + get_youtube_creators(youtube_channels)
    entries.sort(key=lambda e: (not e.live, e.published_at or ""), reverse=True)
    return entries
