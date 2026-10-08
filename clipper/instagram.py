"""Post Reels to an Instagram professional account.

Two ways to connect, picked by INSTAGRAM_LOGIN (default "facebook"):

* "facebook" -- Instagram API with Facebook Login: the Instagram account is
  linked to a Facebook Page; Connect signs in with Facebook (the app's
  admin, so no tester invite) and keeps the Page's token for the linked
  Instagram account. A Page token made from a long-lived user token
  doesn't expire. Meta only offered this route on Dean's apps.
* "instagram" -- Instagram API with Instagram Login (no Page): code ->
  short-lived token -> 60-day token, refreshed when it's over a day old
  and has under 20 days left.

Setup: INSTAGRAM_APP_ID / INSTAGRAM_APP_SECRET (the Meta app's App ID and
secret for "facebook", the Instagram app ID and secret for "instagram"),
the redirect URL ``https://<RAILWAY_PUBLIC_DOMAIN>/auth/instagram/callback``
registered in the app, and for "facebook" optionally INSTAGRAM_FB_CONFIG_ID
(a Facebook Login for Business configuration; without it the login asks
for the permissions by name). Accounts with a role on the app need no App
Review.

Publishing a Reel: Instagram downloads the video itself, so it has to be at
a public URL; create a container (media_type=REELS), wait until its
status_code is FINISHED, then media_publish it. Limit: 100 API posts per
account per 24 hours.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

AUTHORIZE_URL = "https://www.instagram.com/oauth/authorize"
TOKEN_URL = "https://api.instagram.com/oauth/access_token"
GRAPH = "https://graph.instagram.com"
API_VERSION = "v25.0"
SCOPES = "instagram_business_basic,instagram_business_content_publish"
FB_DIALOG = "https://www.facebook.com/{v}/dialog/oauth"
FB_GRAPH = "https://graph.facebook.com"
FB_SCOPES = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement,business_management"
CAPTION_MAX = 2200


def login_mode() -> str:
    return "instagram" if os.environ.get("INSTAGRAM_LOGIN", "").strip().lower() == "instagram" else "facebook"


class InstagramError(Exception):
    pass


def is_configured() -> bool:
    return bool(os.environ.get("INSTAGRAM_APP_ID") and os.environ.get("INSTAGRAM_APP_SECRET"))


def authorize_url(redirect_uri: str, state: str) -> str:
    if login_mode() == "facebook":
        params = {"client_id": os.environ["INSTAGRAM_APP_ID"], "redirect_uri": redirect_uri,
                  "response_type": "code", "state": state}
        config = os.environ.get("INSTAGRAM_FB_CONFIG_ID", "").strip()
        if config:
            params["config_id"] = config        # Facebook Login for Business configuration
        else:
            params["scope"] = FB_SCOPES
        return FB_DIALOG.format(v=API_VERSION) + "?" + urlencode(params)
    return AUTHORIZE_URL + "?" + urlencode({
        "client_id": os.environ["INSTAGRAM_APP_ID"], "redirect_uri": redirect_uri,
        "response_type": "code", "scope": SCOPES, "state": state,
        # always ask which account, so the right one gets connected
        "force_reauth": "true",
    })


def _check(resp) -> dict:
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400 or "error" in data or "error_message" in data:
        err = data.get("error") if isinstance(data.get("error"), dict) else {}
        msg = err.get("error_user_msg") or err.get("message") or data.get("error_message") \
            or resp.text[:300] or f"HTTP {resp.status_code}"
        if err.get("code") == 190:
            msg = "The Instagram connection has expired or was removed -- connect it again. (" + msg + ")"
        raise InstagramError(msg)
    return data


def _exchange_code_facebook(code: str, redirect_uri: str) -> dict:
    """Facebook Login: code -> user token -> long-lived user token -> the
    Page token of the first Page with a linked Instagram account."""
    import requests

    app = {"client_id": os.environ["INSTAGRAM_APP_ID"], "client_secret": os.environ["INSTAGRAM_APP_SECRET"]}
    base = f"{FB_GRAPH}/{API_VERSION}"
    short = _check(requests.get(f"{base}/oauth/access_token", params={
        **app, "redirect_uri": redirect_uri, "code": code}, timeout=30))["access_token"]
    long = _check(requests.get(f"{base}/oauth/access_token", params={
        **app, "grant_type": "fb_exchange_token", "fb_exchange_token": short}, timeout=30))["access_token"]
    pages = _check(requests.get(f"{base}/me/accounts", params={
        "fields": "id,name,access_token,instagram_business_account{id,username}",
        "access_token": long, "limit": 100}, timeout=30)).get("data") or []
    linked = [p for p in pages if p.get("instagram_business_account")]
    if not linked:
        names = ", ".join(p.get("name", "?") for p in pages) or "none"
        raise InstagramError("None of your Facebook Pages has an Instagram professional account linked "
                             f"(Pages the app can see: {names}). Link the Instagram account to a Page "
                             "(Page settings -> Linked accounts -> Instagram) and connect again.")
    page = linked[0]
    ig = page["instagram_business_account"]
    now = time.time()
    return {"mode": "facebook", "graph": FB_GRAPH, "access_token": page["access_token"],
            "obtained_at": now, "expires_at": now + 3650 * 86400,     # Page tokens don't expire
            "ig_id": str(ig["id"]), "username": ig.get("username"), "page": page.get("name")}


def exchange_code(code: str, redirect_uri: str) -> dict:
    """The login's code -> a stored token plus the Instagram account's id
    and username."""
    if login_mode() == "facebook":
        return _exchange_code_facebook(code, redirect_uri)
    import requests

    r = requests.post(TOKEN_URL, data={
        "client_id": os.environ["INSTAGRAM_APP_ID"], "client_secret": os.environ["INSTAGRAM_APP_SECRET"],
        "grant_type": "authorization_code", "redirect_uri": redirect_uri, "code": code}, timeout=30)
    data = _check(r)
    entry = data["data"][0] if isinstance(data.get("data"), list) and data["data"] else data
    short = entry["access_token"]
    r = requests.get(f"{GRAPH}/access_token", params={
        "grant_type": "ig_exchange_token", "client_secret": os.environ["INSTAGRAM_APP_SECRET"],
        "access_token": short}, timeout=30)
    long = _check(r)
    me = _check(requests.get(f"{GRAPH}/{API_VERSION}/me", params={
        "fields": "user_id,username", "access_token": long["access_token"]}, timeout=30))
    now = time.time()
    return {"access_token": long["access_token"], "obtained_at": now,
            "expires_at": now + float(long.get("expires_in") or 60 * 86400),
            "ig_id": str(me.get("user_id") or entry.get("user_id")), "username": me.get("username")}


class TokenStore:
    def __init__(self, path: Path):
        self.path = path

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    def status(self) -> dict:
        data = self.load()
        if not data or data.get("expires_at", 0) < time.time():
            return {"connected": False, "expired": bool(data)}
        return {"connected": True, "username": data.get("username"),
                "expires_at": data.get("expires_at")}

    def get_valid(self) -> Optional[dict]:
        """The stored token (refreshed first when it's due), or None."""
        import requests

        data = self.load()
        now = time.time()
        if not data or data.get("expires_at", 0) < now:
            return None
        if data.get("mode") == "facebook":
            return data             # Page tokens don't need refreshing
        if now - data.get("obtained_at", now) > 86400 and data["expires_at"] - now < 20 * 86400:
            try:
                r = _check(requests.get(f"{GRAPH}/refresh_access_token", params={
                    "grant_type": "ig_refresh_token", "access_token": data["access_token"]}, timeout=30))
                data.update(access_token=r["access_token"], obtained_at=now,
                            expires_at=now + float(r.get("expires_in") or 60 * 86400))
                self.save(data)
            except Exception:
                pass        # the old token still works until it expires
        return data


def caption_for(title: str, description: str) -> str:
    """YouTube's title + description as an Instagram caption, without the
    YouTube-only #Shorts tag."""
    text = "\n\n".join(p for p in ((title or "").strip(), (description or "").strip()) if p)
    return re.sub(r"(?i)[ \t]*#shorts\b", "", text).strip()[:CAPTION_MAX]


def publish_reel(token: dict, video_url: str, caption: str, log=None, poll: float = 10.0,
                 timeout: float = 600.0) -> dict:
    """Make a Reel from a public video URL and publish it. Returns
    {"media_id", "permalink"}. Blocks while Instagram processes the video
    (usually under a minute or two)."""
    import requests

    log = log or (lambda *_: None)
    auth = {"access_token": token["access_token"]}
    ig = token["ig_id"]
    host = token.get("graph") or GRAPH      # graph.facebook.com for Facebook Login
    r = requests.post(f"{host}/{API_VERSION}/{ig}/media", data={
        "media_type": "REELS", "video_url": video_url, "caption": caption, "share_to_feed": "true", **auth},
        timeout=60)
    container = _check(r)["id"]
    log("container", container)
    deadline = time.time() + timeout
    while True:
        st = _check(requests.get(f"{host}/{API_VERSION}/{container}",
                                 params={"fields": "status_code,status", **auth}, timeout=30))
        code = st.get("status_code")
        log("status", code)
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise InstagramError(f"Instagram couldn't process the video: {st.get('status') or code}")
        if time.time() > deadline:
            raise InstagramError("Instagram took too long to process the video -- try again later.")
        time.sleep(poll)
    media_id = _check(requests.post(f"{host}/{API_VERSION}/{ig}/media_publish",
                                    data={"creation_id": container, **auth}, timeout=60))["id"]
    permalink = None
    try:
        permalink = _check(requests.get(f"{host}/{API_VERSION}/{media_id}",
                                        params={"fields": "permalink", **auth}, timeout=30)).get("permalink")
    except InstagramError:
        pass
    return {"media_id": media_id, "permalink": permalink}
