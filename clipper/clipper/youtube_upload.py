"""Upload a rendered clip straight to the connected YouTube channel, via
the YouTube Data API v3's resumable upload protocol -- for the "Upload to
YouTube" button on a clip a creator has already hand-picked, not a bulk or
automatic publish.

Costs 1,600 YouTube Data API quota units per upload (out of a default
10,000/day project budget, shared with competitor search and every channel
snapshot this app pulls) -- expensive enough that this stays a deliberate,
one-clip-at-a-time action a person clicks, never something run in a loop
or on a schedule.
"""
from __future__ import annotations

from pathlib import Path

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

# Category 20 = "Gaming" -- a reasonable fixed default given every clip
# this app produces comes from clipping a gameplay stream. Not exposed as
# a setting; not worth the extra UI for how rarely it'd ever be anything
# else here.
_GAMING_CATEGORY_ID = "20"

_VALID_PRIVACY_STATUSES = ("public", "unlisted", "private")


class UploadError(RuntimeError):
    """Raised with a message that's already safe to show the creator --
    callers shouldn't need to inspect this further, just display str(e)."""


def upload_video(
    access_token: str,
    video_path: Path,
    title: str,
    description: str,
    privacy_status: str = "unlisted",
) -> str:
    """Upload video_path to the connected channel. Returns the new video's
    id on success."""
    import requests

    if privacy_status not in _VALID_PRIVACY_STATUSES:
        raise ValueError(f"invalid privacy_status: {privacy_status!r}")
    if not video_path.exists():
        raise UploadError(f"{video_path.name} no longer exists on the server")

    size = video_path.stat().st_size

    try:
        init_resp = requests.post(
            UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/mp4",
                "X-Upload-Content-Length": str(size),
            },
            json={
                # YouTube titles/descriptions have hard length caps -- a
                # generated title/description that happens to run long
                # would otherwise fail the whole upload on a 400 instead
                # of just being trimmed to fit.
                "snippet": {"title": title[:100], "description": description[:5000], "categoryId": _GAMING_CATEGORY_ID},
                "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
            },
            timeout=30,
        )
    except requests.RequestException as e:
        raise UploadError(f"Could not reach YouTube to start the upload: {e}") from e

    if init_resp.status_code == 401:
        raise UploadError("Your YouTube connection has expired -- disconnect and reconnect it on the analytics page.")
    if init_resp.status_code == 403:
        raise UploadError(
            "YouTube refused the upload -- either your connection was granted before upload access was "
            "added (disconnect and reconnect on the analytics page to re-grant it) or today's upload "
            "quota is used up."
        )
    if init_resp.status_code != 200:
        raise UploadError(f"Could not start the upload ({init_resp.status_code}): {init_resp.text[:500]}")

    upload_url = init_resp.headers.get("Location")
    if not upload_url:
        raise UploadError("YouTube accepted the request but didn't return an upload session URL")

    try:
        with video_path.open("rb") as f:
            put_resp = requests.put(
                upload_url,
                data=f,
                headers={"Content-Type": "video/mp4", "Content-Length": str(size)},
                # A clip is at most a couple hundred MB -- generous ceiling
                # for a slow upstream link, not a realistic normal case.
                timeout=900,
            )
    except requests.RequestException as e:
        raise UploadError(f"Upload to YouTube failed partway through: {e}") from e

    if put_resp.status_code not in (200, 201):
        raise UploadError(f"Upload failed ({put_resp.status_code}): {put_resp.text[:500]}")

    try:
        video_id = put_resp.json().get("id")
    except ValueError:
        video_id = None
    if not video_id:
        raise UploadError(f"Upload seemed to finish but YouTube didn't return a video id: {put_resp.text[:500]}")
    return video_id
