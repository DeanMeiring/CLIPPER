"""OAuth 2.0 login flow for the channel owner's own YouTube account, so the
web app can read real YouTube Analytics data (retention, traffic sources,
day-of-week performance) instead of only the public Data API's view counts.

This app is single-tenant (one channel owner, gated by APP_PASSWORD) --
there's one stored token for "the" connected channel, not per-visitor
accounts. The token lives on the same persistent volume job data already
lives on (CLIPPER_JOBS_DIR), so it survives restarts and redeploys.

Setup this requires (see README): a Google Cloud OAuth 2.0 Web application
client (client ID + secret) with the redirect URI pointed at this app's
/auth/youtube/callback, and YOUTUBE_OAUTH_CLIENT_ID / YOUTUBE_OAUTH_CLIENT_SECRET
set as environment variables.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Optional

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def is_configured() -> bool:
    return bool(os.environ.get("YOUTUBE_OAUTH_CLIENT_ID") and os.environ.get("YOUTUBE_OAUTH_CLIENT_SECRET"))


def build_authorize_url(redirect_uri: str, state: str) -> str:
    client_id = os.environ["YOUTUBE_OAUTH_CLIENT_ID"]
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        # Force Google to hand back a refresh_token every time, not just on
        # the very first consent -- reconnecting after a revoke/expiry
        # needs a new one, and without this it silently omits it.
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, redirect_uri: str) -> dict:
    import requests

    resp = requests.post(TOKEN_URL, data={
        "code": code,
        "client_id": os.environ["YOUTUBE_OAUTH_CLIENT_ID"],
        "client_secret": os.environ["YOUTUBE_OAUTH_CLIENT_SECRET"],
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _refresh(refresh_token: str) -> dict:
    import requests

    resp = requests.post(TOKEN_URL, data={
        "refresh_token": refresh_token,
        "client_id": os.environ["YOUTUBE_OAUTH_CLIENT_ID"],
        "client_secret": os.environ["YOUTUBE_OAUTH_CLIENT_SECRET"],
        "grant_type": "refresh_token",
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


class TokenStore:
    """Reads/writes the one stored token on disk, refreshing the access
    token transparently when it's expired or close to it."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, token: dict) -> None:
        data = dict(token)
        data["obtained_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def is_connected(self) -> bool:
        data = self.load()
        return bool(data and data.get("refresh_token"))

    def get_valid_access_token(self) -> Optional[str]:
        data = self.load()
        if not data or not data.get("refresh_token"):
            return None
        obtained_at = data.get("obtained_at", 0)
        expires_in = data.get("expires_in", 0)
        # Refresh a bit before actual expiry so a request never races a
        # token that expires mid-call.
        if time.time() < obtained_at + expires_in - 60:
            return data["access_token"]
        try:
            refreshed = _refresh(data["refresh_token"])
        except Exception:
            return None
        # A refresh response doesn't always include a new refresh_token --
        # keep the one already on file if this one didn't send one.
        refreshed.setdefault("refresh_token", data["refresh_token"])
        self.save(refreshed)
        return refreshed["access_token"]
