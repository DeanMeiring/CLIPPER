"""Notify the user via Telegram when a job reaches a final state.

Needs TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID (the
numeric id of the chat to message -- get it by messaging the bot once,
then hitting https://api.telegram.org/bot<TOKEN>/getUpdates). Best-effort
and silent if either is unset or the request fails -- a notification
problem should never take down the actual render job.
"""
from __future__ import annotations

import os


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
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
