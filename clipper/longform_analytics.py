"""How a posted "Story Of" episode is doing, for the Analytics page: the
YouTube numbers for the video, its retention curve laid over the episode's
own scenes and chapters (so a drop points at the scene people left in),
where its views came from, and how its cliffhanger Shorts did.

build_episode() is pure: webapp/main.py fetches the numbers
(youtube_analytics.get_video_stats / get_retention_curve / ...) and passes
them in.
"""
from __future__ import annotations

from typing import List, Optional

# YouTube's traffic source names, in words.
TRAFFIC_LABELS = {
    "YT_SEARCH": "YouTube search",
    "RELATED_VIDEO": "Suggested videos",
    "SUBSCRIBER": "Home page & subscriptions",
    "BROWSE": "Home page & subscriptions",
    "YT_CHANNEL": "Your channel page",
    "SHORTS": "Shorts feed",
    "NOTIFICATION": "Notifications",
    "EXT_URL": "Other websites & apps",
    "NO_LINK_OTHER": "Direct or unknown",
    "NO_LINK_EMBEDDED": "Embedded players",
    "PLAYLIST": "Playlists",
    "YT_PLAYLIST_PAGE": "Playlists",
    "END_SCREEN": "End screens",
    "HASHTAGS": "Hashtag pages",
    "YT_OTHER_PAGE": "Other YouTube pages",
    "ADVERTISING": "Ads",
    "CAMPAIGN_CARD": "Campaign cards",
}

# A drop smaller than this (share of viewers, over ~3% of the video)
# isn't worth pointing at.
MIN_DROP = 0.02


def _scene_at(t: float, starts: List[float]) -> int:
    idx = 0
    for i, s in enumerate(starts):
        if s <= t:
            idx = i
        else:
            break
    return idx


def _scene_label(scene: dict) -> str:
    kind = scene.get("kind", "narrate")
    if kind == "title":
        return f"chapter card “{scene.get('title', '')}”"
    if kind == "moment":
        words = (scene.get("caption") or "").strip()
        return "clip moment" + (f" “{words[:60]}”" if words else "")
    all_words = (scene.get("narration") or "").split()
    words = " ".join(all_words[:9])
    if len(all_words) > 9:
        words = words.rstrip(".,;:!?") + "…"
    return f"narration “{words}”" if words else "narration"


def chapters(scenes: List[dict], starts: List[float]) -> List[dict]:
    return [{"t": starts[i], "title": sc.get("title", "")}
            for i, sc in enumerate(scenes) if sc.get("kind") == "title" and i < len(starts)]


def biggest_drops(curve: list, duration: float, scenes: List[dict], starts: List[float], top: int = 3) -> List[dict]:
    """The steepest falls in the retention curve after the first 30 s (the
    opening always falls; it's reported on its own), each tied to the
    scene playing at that moment."""
    if len(curve) < 6 or not duration:
        return []
    drops = []
    for i in range(len(curve) - 3):
        t0 = curve[i][0] * duration
        if t0 < 30:
            continue
        fall = curve[i][1] - curve[i + 3][1]
        if fall >= MIN_DROP:
            # Pin it to the steepest single step inside the window, so it
            # lands in the scene people actually left during.
            j = max(range(i, i + 3), key=lambda k: curve[k][1] - curve[k + 1][1])
            drops.append((fall, (curve[j][0] + curve[j + 1][0]) / 2 * duration))
    drops.sort(reverse=True)
    picked: List[dict] = []
    for fall, t in drops:
        if any(abs(t - p["t"]) < max(20.0, duration * 0.05) for p in picked):
            continue
        k = _scene_at(t, starts)
        sc = scenes[k] if k < len(scenes) else {}
        picked.append({"t": round(t, 1), "fall": round(fall, 3), "scene": k + 1, "what": _scene_label(sc) if sc else ""})
        if len(picked) >= top:
            break
    return sorted(picked, key=lambda p: p["t"])


def _watch_at(curve: list, frac: float) -> Optional[float]:
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p[0] - frac))[1]


def build_episode(project: dict, stats: Optional[dict], curve: list, daily: list, traffic: list, shorts: List[dict]) -> dict:
    render = project.get("render") or {}
    duration = float(render.get("duration") or 0)
    starts = [float(s) for s in render.get("starts") or []]
    scenes = (project.get("scenes") or [])[:len(starts)] if starts else []
    total_traffic = sum(t["views"] for t in traffic) or 0
    rel = [p[2] for p in curve if len(p) > 2 and p[2] is not None]
    at30 = _watch_at(curve, 30 / duration) if duration and curve else None
    return {
        "id": project["id"],
        "title": project.get("title") or "Untitled",
        "video_id": project.get("youtube_video_id"),
        "url": project.get("youtube_url"),
        "privacy": project.get("uploaded_privacy"),
        "duration": duration,
        "stats": stats,
        "curve": [[round(p[0], 4), round(p[1], 4)] for p in curve],
        "watch_30s": at30,
        "watch_half": _watch_at(curve, 0.5),
        "watch_end": _watch_at(curve, 0.95),
        "vs_similar": round(sum(rel) / len(rel), 2) if rel else None,
        "chapters": chapters(scenes, starts),
        "drops": biggest_drops(curve, duration, scenes, starts),
        "daily": daily,
        "traffic": [{"source": TRAFFIC_LABELS.get(t["source"], t["source"].replace("_", " ").title()), "views": t["views"],
                     "share": round(t["views"] / total_traffic, 3) if total_traffic else 0} for t in traffic[:6]],
        "shorts": shorts,
    }
