"""Notify the user via Telegram when a job reaches a final state.

Needs the bot token from @BotFather, read from CLIPPER_BOT_API (or
TELEGRAM_BOT_TOKEN, either name works). If TELEGRAM_CHAT_ID isn't set,
the chat to message is auto-discovered via the Bot API's getUpdates --
the most recent chat that has messaged the bot -- so no manual chat-id
lookup is required, just message the bot once first. Best-effort and
silent on any failure (missing token, nobody has messaged the bot yet,
a network error) -- a notification problem should never take down the
actual render job.
"""
from __future__ import annotations

import os


def _get_token() -> str | None:
    return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("CLIPPER_BOT_API")


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

    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _discover_chat_id(token)
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
