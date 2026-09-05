"""Notify the user via Telegram when a job reaches a final state.

Needs the bot token from @BotFather, read from CLIPPER_BOT_API (or
TELEGRAM_BOT_TOKEN, either name works). If TELEGRAM_CHAT_ID isn't set,
the chat to message is auto-discovered via the Bot API's getUpdates --
the most recent chat that has messaged the bot -- so no manual chat-id
lookup is required, just message the bot once first.

That discovery only looks at Telegram's backlog of not-yet-fetched
updates, which isn't kept forever -- so a chat_id found once is cached
to disk (on the same persistent volume job state lives on) and reused
from then on, rather than re-discovering it via getUpdates on every
send and silently failing once that backlog has aged out or already
been consumed.

Best-effort and silent on any failure (missing token, nobody has
messaged the bot yet, a network error) -- a notification problem
should never take down the actual render job.
"""
from __future__ import annotations

import os
from pathlib import Path


def _get_token() -> str | None:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("CLIPPER_BOT_API")


def _chat_id_cache_path() -> Path:
    base = Path(os.environ.get("CLIPPER_JOBS_DIR", "/tmp/clipper_jobs"))
    return base / "telegram_chat_id.txt"


def _load_cached_chat_id() -> str | None:
    try:
        path = _chat_id_cache_path()
        if path.exists():
            chat_id = path.read_text(encoding="utf-8").strip()
            return chat_id or None
    except OSError:
        pass
    return None


def _save_chat_id(chat_id: str) -> None:
    try:
        path = _chat_id_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(chat_id, encoding="utf-8")
    except OSError:
        pass  # best-effort -- worst case it just re-discovers next time


def _discover_chat_id(token: str) -> str | None:
    import requests

    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15)
        resp.raise_for_status()
        results = resp.json().get("result") or []
        for update in reversed(results):  # most recent first
            message = update.get("message") or update.get("channel_post")
            if message and message.get("chat", {}).get("id") is not None:
                return str(message["chat"]["id"])
    except Exception as e:
        print(f"[notify] Telegram chat-id discovery failed: {e}", flush=True)
    return None


def send_telegram(text: str) -> bool:
    token = _get_token()
    if not token:
        return False

    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _load_cached_chat_id()
    if not chat_id:
        chat_id = _discover_chat_id(token)
        if chat_id:
            _save_chat_id(chat_id)
    if not chat_id:
        print("[notify] no Telegram chat found -- message the bot once first", flush=True)
        return False

    import requests

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[notify] Telegram send failed: {e}", flush=True)
        return False
