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

import re
from pathlib import Path
from typing import Optional

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


_YOUTUBE_DESCRIPTION_MAX = 5000


def _with_shorts_tag(description: str, max_length: int = _YOUTUBE_DESCRIPTION_MAX) -> str:
    """A vertical, <=3-minute video meets the technical bar for a Short,
    but YouTube's classifier still won't reliably surface an API-uploaded
    video in Shorts unless "#Shorts" actually appears in the title or
    description -- confirmed by Google's own upload guidance, and the
    reason a correctly-shaped clip uploaded through this button can still
    land as a plain regular video. Appended here rather than left to
    whatever Claude happened to write into the generated description.

    Truncates the ORIGINAL description first, leaving room for the tag, so
    a long description can never push the tag itself past max_length and
    lose the one thing this function exists to guarantee."""
    if re.search(r"#shorts\b", description, re.IGNORECASE):
        return description[:max_length]
    tag = "#Shorts"
    if not description:
        return tag
    suffix = f"\n\n{tag}"
    return description[: max_length - len(suffix)] + suffix


def _unauthorized_message(resp) -> str:
    """YouTube's 401 says why: most often the connected Google account has
    no YouTube channel (youtubeSignupRequired), not an expired login."""
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    reasons = {e.get("reason") for e in err.get("errors") or [] if isinstance(e, dict)}
    if "youtubeSignupRequired" in reasons:
        return ("The connected Google account has no YouTube channel. Reconnect and pick the account (or the "
                "channel's brand account) that owns the channel -- or create the channel first.")
    detail = f" (YouTube said: {err.get('message')})" if err.get("message") else ""
    return "YouTube didn't accept the connection -- reconnect the channel and try again." + detail


def upload_video(
    access_token: str,
    video_path: Path,
    title: str,
    description: str,
    privacy_status: str = "unlisted",
    is_short: bool = True,
    synthetic_media: bool = False,
    publish_at: Optional[str] = None,
    category_id: str = _GAMING_CATEGORY_ID,
) -> str:
    """Upload video_path to the connected channel. Returns the new video's
    id on success. ``category_id`` is YouTube's category (Gaming unless a
    caller says otherwise; Ball Evolution uses "24", Entertainment).

    `publish_at` (RFC 3339, UTC) schedules it: uploaded private, YouTube
    makes it public at that time on its own -- even while this app sleeps.

    `synthetic_media` ticks YouTube's "altered or synthetic content" box
    (status.containsSyntheticMedia): needed when an AI copy of a real
    person's voice says things they didn't record -- a friend's cloned
    voice narrating. Left off otherwise, so other uploads send exactly
    what they did before.

    `is_short` appends the "#Shorts" tag this app's clips need to be
    reliably classified as a Short (see _with_shorts_tag). Set it False
    for a long-form upload (e.g. the weekly cross-streamer recap) -- that
    tag on a multi-minute video would be actively misleading to viewers
    and to YouTube's own classifier."""
    import requests

    if privacy_status not in _VALID_PRIVACY_STATUSES:
        raise ValueError(f"invalid privacy_status: {privacy_status!r}")
    if not video_path.exists():
        raise UploadError(f"{video_path.name} no longer exists on the server")

    if is_short:
        description = _with_shorts_tag(description)
    else:
        description = description[:_YOUTUBE_DESCRIPTION_MAX]
    size = video_path.stat().st_size
    status = {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False}
    if synthetic_media:
        status["containsSyntheticMedia"] = True
    if publish_at:
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at

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
                "snippet": {"title": title[:100], "description": description, "categoryId": category_id},
                "status": status,
            },
            timeout=30,
        )
    except requests.RequestException as e:
        raise UploadError(f"Could not reach YouTube to start the upload: {e}") from e

    if init_resp.status_code == 401:
        raise UploadError(_unauthorized_message(init_resp))
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


def set_thumbnail(access_token: str, video_id: str, image_path: Path) -> None:
    """Set a video's custom thumbnail (JPEG, under 2 MB). YouTube only
    allows this on a channel verified with a phone number; a 403 says so."""
    import requests

    try:
        resp = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/thumbnails/set",
            params={"videoId": video_id, "uploadType": "media"},
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "image/jpeg"},
            data=image_path.read_bytes(),
            timeout=120,
        )
    except requests.RequestException as e:
        raise UploadError(f"Could not reach YouTube to set the thumbnail: {e}") from e
    if resp.status_code == 401:
        raise UploadError("Your YouTube connection has expired -- disconnect and reconnect it on the analytics page.")
    if resp.status_code == 403:
        raise UploadError(
            "YouTube didn't allow a custom thumbnail. That needs a verified channel: YouTube Studio -> Settings -> "
            "Channel -> Feature eligibility -> verify with your phone, then press Set on YouTube again."
        )
    if resp.status_code not in (200, 201):
        raise UploadError(f"Couldn't set the thumbnail ({resp.status_code}): {resp.text[:300]}")
