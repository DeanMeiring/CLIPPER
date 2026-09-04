"""Find candidate highlight windows in a long Twitch VOD without transcribing
the whole thing.

Two signals, combined:
  1. Chat activity -- message-rate spikes and reaction-keyword density
     ("lol", "wtf", "no way", ...), pulled from the VOD's chat replay via
     the chat-downloader library. No Twitch auth needed for this.
  2. Existing viewer-made clips of the VOD (if TWITCH_CLIENT_ID and
     TWITCH_CLIENT_SECRET are set) -- a clip someone already cut from this
     exact VOD is about as strong a "this moment is good" signal as exists.
     Optional: skipped cleanly if those env vars aren't set.

Either signal alone would work; together they cover streams with active
chat and streams where the community clips more than it types.
"""
from __future__ import annotations

import datetime
import os
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional


REACTION_KEYWORDS = [
    "lol", "lmao", "lmfao", "haha", "hahaha", "hahahaha", "wtf", "omg",
    "no way", "noway", "insane", "clip that", "clip it", "clipped",
    "poggers", "pog", "pogchamp", "omegalul", "kekw",
    "what happened", "what just happened", "wait what", "holy",
    "bruh", "nooo", "no no no", "let's go", "lets go", "ggs",
]

_KEYWORD_RE = re.compile(
    r"(" + "|".join(re.escape(k) for k in REACTION_KEYWORDS) + r")",
    re.IGNORECASE,
)


@dataclass
class CandidateWindow:
    start: float
    end: float
    score: float
    source: str            # "chat" or "clip" (or "chat+clip" if merged)
    detail: str             # short human-readable reason, for logging/debugging


def _fetch_chat_messages(video_url: str, timeout_seconds: float = 300.0) -> List[dict]:
    """Pull the full chat replay. Bounded by a hard wall-clock timeout,
    run in a worker thread so it applies even if the fetch blocks entirely
    (e.g. a library edge case treating the VOD as still live and waiting
    for new messages that never come) -- this degrades to "no chat signal"
    rather than hanging the whole job indefinitely."""
    try:
        from chat_downloader import ChatDownloader
    except ImportError as e:
        raise RuntimeError(
            "chat-downloader is required for chat-spike detection. Install it with: pip install chat-downloader"
        ) from e

    import threading
    import time
    import traceback

    messages: List[dict] = []
    error_info: List[str] = []
    deadline = time.monotonic() + timeout_seconds

    def _run():
        try:
            # interruptible_retry defaults to True, which polls stdin for a
            # "press a key to retry now" prompt -- crashes here since this
            # runs as a headless background service with no real stdin.
            # max_attempts capped low so a real rejection (rate limit, GQL
            # schema drift) fails fast instead of retrying 15 times.
            chat = ChatDownloader().get_chat(video_url, interruptible_retry=False, max_attempts=3)
            for msg in chat:
                messages.append(msg)
                if time.monotonic() > deadline:
                    break
        except Exception:  # chat disabled, VOD deleted, library hiccup, etc.
            error_info.append(traceback.format_exc())

    # A daemon thread: if the fetch truly hangs (e.g. a library edge case
    # treating the VOD as still live), .join()'s timeout below still
    # returns control to the caller instead of blocking the job forever --
    # the thread is simply abandoned rather than force-killed, since
    # Python can't do that safely.
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"[highlights] chat fetch hit the {timeout_seconds:.0f}s timeout "
              f"with {len(messages)} messages collected so far; continuing with those", flush=True)
    elif error_info:
        print(f"[highlights] chat replay fetch failed for {video_url}:\n{error_info[0]}", flush=True)

    return messages


def _bucket_messages(messages: List[dict], duration: float, bucket_seconds: float):
    n_buckets = max(1, int(duration // bucket_seconds) + 1)
    counts = [0] * n_buckets
    keyword_counts = [0] * n_buckets
    for msg in messages:
        t = msg.get("time_in_seconds")
        text = msg.get("message") or ""
        if t is None or t < 0:
            continue
        idx = min(int(t // bucket_seconds), n_buckets - 1)
        counts[idx] += 1
        if _KEYWORD_RE.search(text):
            keyword_counts[idx] += 1
    return counts, keyword_counts


def _chat_spike_windows(
    messages: List[dict],
    duration: float,
    bucket_seconds: float = 10.0,
    baseline_window_buckets: int = 30,   # +/- ~5 min at 10s buckets
    spike_multiplier: float = 3.0,
    min_keyword_hits: int = 2,
    keyword_weight: float = 4.0,
    pad_before: float = 15.0,
    pad_after: float = 30.0,
    merge_gap: float = 20.0,
) -> List[CandidateWindow]:
    counts, keyword_counts = _bucket_messages(messages, duration, bucket_seconds)
    n = len(counts)
    if n == 0:
        return []

    overall_baseline = max(statistics.median(counts), 1.0)

    flagged = []
    for i in range(n):
        lo, hi = max(0, i - baseline_window_buckets), min(n, i + baseline_window_buckets)
        local_baseline = max(statistics.median(counts[lo:hi]), overall_baseline * 0.3, 1.0)
        ratio = counts[i] / local_baseline
        kw = keyword_counts[i]
        if ratio >= spike_multiplier or kw >= min_keyword_hits:
            flagged.append((i, ratio, kw))

    if not flagged:
        return []

    windows: List[CandidateWindow] = []

    def flush(start_idx, end_idx, ratio_sum, kw_sum, n_merged):
        start = max(0.0, start_idx * bucket_seconds - pad_before)
        end = min(duration, (end_idx + 1) * bucket_seconds + pad_after)
        avg_ratio = ratio_sum / max(1, n_merged)
        score = avg_ratio + kw_sum * keyword_weight
        windows.append(CandidateWindow(
            start=round(start, 1), end=round(end, 1), score=round(score, 2),
            source="chat",
            detail=f"chat spike {avg_ratio:.1f}x baseline, {kw_sum} reaction keyword(s)",
        ))

    cur_start, cur_end = flagged[0][0], flagged[0][0]
    ratio_sum, kw_sum, n_merged = flagged[0][1], flagged[0][2], 1
    for idx, ratio, kw in flagged[1:]:
        if (idx - cur_end) * bucket_seconds <= merge_gap:
            cur_end = idx
            ratio_sum += ratio
            kw_sum += kw
            n_merged += 1
        else:
            flush(cur_start, cur_end, ratio_sum, kw_sum, n_merged)
            cur_start, cur_end, ratio_sum, kw_sum, n_merged = idx, idx, ratio, kw, 1
    flush(cur_start, cur_end, ratio_sum, kw_sum, n_merged)

    return windows


def _existing_clip_windows(
    broadcaster_login: str,
    vod_id: str,
    created_at: Optional[float] = None,
    duration: Optional[float] = None,
    pad_before: float = 20.0,
    pad_after: float = 40.0,
) -> List[CandidateWindow]:
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    if not client_id or not client_secret:
        return []

    import requests

    try:
        token_resp = requests.post(
            "https://id.twitch.tv/oauth2/token",
            data={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
            timeout=15,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]
        headers = {"Client-Id": client_id, "Authorization": f"Bearer {access_token}"}

        user_resp = requests.get(
            "https://api.twitch.tv/helix/users",
            params={"login": broadcaster_login}, headers=headers, timeout=15,
        )
        user_resp.raise_for_status()
        user_data = user_resp.json().get("data") or []
        if not user_data:
            print(f"[highlights] Twitch user lookup for login={broadcaster_login!r} returned no results "
                  f"-- likely not the account's real Twitch login (a display name with different casing/"
                  f"spacing won't match)", flush=True)
            return []
        broadcaster_id = user_data[0]["id"]

        clip_params: dict = {"broadcaster_id": broadcaster_id, "first": 100}
        if created_at is not None:
            # Without a time window, /helix/clips paginates a popular
            # streamer's ENTIRE clip history -- an old VOD's clips can sit
            # far past any reasonable page cap. Narrow to the broadcast's
            # own window (generous slack for timestamp imprecision) so we
            # actually reach the clips that matter.
            slack = 7200.0
            start_dt = datetime.datetime.fromtimestamp(created_at - slack, tz=datetime.timezone.utc)
            end_dt = datetime.datetime.fromtimestamp(created_at + (duration or 0.0) + slack, tz=datetime.timezone.utc)
            clip_params["started_at"] = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            clip_params["ended_at"] = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        clips = []
        cursor = None
        for _ in range(10):  # cap pagination -- with the time window above, plenty for one VOD
            params = dict(clip_params)
            if cursor:
                params["after"] = cursor
            resp = requests.get("https://api.twitch.tv/helix/clips", params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            clips.extend(body.get("data") or [])
            cursor = (body.get("pagination") or {}).get("cursor")
            if not cursor:
                break
    except Exception as e:
        print(f"[highlights] Twitch clips lookup failed: {e}", flush=True)
        return []

    vod_clips = [c for c in clips if c.get("video_id") == str(vod_id) and c.get("vod_offset") is not None]
    print(f"[highlights] Twitch clips: {len(clips)} total for broadcaster_id={broadcaster_id}, "
          f"{len(vod_clips)} matched vod_id={vod_id}", flush=True)
    if not vod_clips:
        return []

    max_views = max((c.get("view_count") or 0) for c in vod_clips) or 1
    windows = []
    for c in vod_clips:
        offset = float(c["vod_offset"])
        views = c.get("view_count") or 0
        score = 5.0 + 10.0 * (views / max_views)  # crowd-clipped moments score high by default
        windows.append(CandidateWindow(
            start=max(0.0, offset - pad_before),
            end=offset + pad_after,
            score=round(score, 2),
            source="clip",
            detail=f"already clipped by a viewer ({views} views on that clip)",
        ))
    return windows


def _merge_overlapping(windows: List[CandidateWindow]) -> List[CandidateWindow]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w.start)
    merged = [windows[0]]
    for w in windows[1:]:
        last = merged[-1]
        if w.start <= last.end:
            merged[-1] = CandidateWindow(
                start=last.start,
                end=max(last.end, w.end),
                score=max(last.score, w.score) + min(last.score, w.score) * 0.25,
                source=last.source if last.source == w.source else f"{last.source}+{w.source}",
                detail=last.detail if last.score >= w.score else w.detail,
            )
        else:
            merged.append(w)
    return merged


def find_candidate_windows(
    video_url: str,
    duration: float,
    broadcaster_login: Optional[str] = None,
    vod_id: Optional[str] = None,
    created_at: Optional[float] = None,
    max_windows: int = 20,
) -> List[CandidateWindow]:
    """Return up to max_windows candidate highlight windows, best first."""
    windows: List[CandidateWindow] = []

    messages = _fetch_chat_messages(video_url)
    if messages:
        windows.extend(_chat_spike_windows(messages, duration))
    else:
        print("[highlights] no chat messages found (chat disabled, or fetch failed) -- relying on other signals", flush=True)

    if broadcaster_login and vod_id:
        print(f"[highlights] checking Twitch clips for broadcaster_login={broadcaster_login!r} vod_id={vod_id!r}", flush=True)
        windows.extend(_existing_clip_windows(broadcaster_login, vod_id, created_at=created_at, duration=duration))
    else:
        print(f"[highlights] skipping Twitch clips lookup (broadcaster_login={broadcaster_login!r}, "
              f"vod_id={vod_id!r}, TWITCH_CLIENT_ID set={bool(os.environ.get('TWITCH_CLIENT_ID'))})", flush=True)

    windows = _merge_overlapping(windows)
    windows.sort(key=lambda w: w.score, reverse=True)
    return windows[:max_windows]
