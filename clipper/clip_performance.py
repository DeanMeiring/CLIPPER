"""Turn the channel's posted Shorts -- real retention curves and view
counts -- plus what this app measured about each clip (clip_features.py)
into plain comparisons: "20-30s clips: 58% still watching at 3s (n=11)
vs 45-60s: 41% (n=8)".

Deterministic on purpose, no model involved: the Analyze page shows these
numbers as-is, and the AI overview gets them as a table it has to cite
rather than forming its own impression. Groups too small to trust are
reported with their size and flagged, never hidden -- a thin result shown
honestly is more useful than a confident-sounding one built on 3 videos.
"""
from __future__ import annotations

import statistics
from typing import Callable, List, Optional

from .clip_features import title_features

# Below this many uploads in a group, a difference is as likely to be two
# lucky or unlucky clips as a real pattern.
MIN_GROUP_SIZE = 5


def curve_metrics(curve: list, duration: Optional[float]) -> Optional[dict]:
    """Retention-curve summary for one video. `curve` rows are
    [elapsed fraction, audienceWatchRatio, relativeRetentionPerformance]
    (see youtube_analytics.get_retention_curve); watch values are viewers
    at that point per view, so 0.52 at 3s means 52% still watching."""
    if not curve or not duration or duration <= 0:
        return None

    def watch_at(seconds: float) -> float:
        target = min(1.0, max(0.0, seconds / duration))
        return min(curve, key=lambda p: abs(p[0] - target))[1]

    drops = [(curve[i - 1][1] - curve[i][1], curve[i][0]) for i in range(1, len(curve))]
    steepest = max(drops, key=lambda d: d[0]) if drops else None
    relative = [p for p in curve if p[2] is not None]
    opening = [p[2] for p in relative if p[0] * duration <= 3.0] or [p[2] for p in relative[:1]]
    return {
        "watch_1s": round(watch_at(1.0), 3),
        "watch_3s": round(watch_at(3.0), 3),
        "watch_mid": round(watch_at(duration * 0.5), 3),
        "watch_end": round(watch_at(duration * 0.95), 3),
        "steepest_drop_at": round(steepest[1] * duration, 1) if steepest and steepest[0] > 0 else None,
        "opening_vs_similar": round(statistics.mean(opening), 3) if opening else None,
        "overall_vs_similar": round(statistics.mean(p[2] for p in relative), 3) if relative else None,
    }


def build_rows(videos: List[dict], metrics_by_id: dict, curves_by_id: dict, records: List[dict]) -> List[dict]:
    """One row per posted Short: its numbers, its title traits, and -- when
    it was made in this app and matched -- the clip's measured features."""
    by_video = {r["video_id"]: r for r in records if r.get("video_id")}
    rows = []
    for v in videos:
        m = metrics_by_id.get(v["id"]) or {}
        record = by_video.get(v["id"])
        row = {
            "id": v["id"],
            "title": v["title"],
            "published_at": v.get("published_at"),
            "age_days": v.get("age_days"),
            "too_new_to_judge": bool(v.get("too_new_to_judge")),
            "duration": v.get("duration_seconds"),
            "views": m.get("views", v.get("views")),
            "avg_view_pct": m.get("average_view_percentage"),
            "retention": curve_metrics(curves_by_id.get(v["id"]) or [], v.get("duration_seconds")),
            "made_here": record is not None,
            "clip": None,
        }
        row.update(title_features(v["title"]))
        if record is not None:
            row["clip"] = {
                **(record.get("features") or {}),
                "layout": record.get("layout"),
                "link_method": record.get("link_method"),
                "hook_caption": record.get("hook_caption"),
            }
        rows.append(row)
    return rows


def _length(row: dict) -> Optional[str]:
    d = row.get("duration")
    if d is None:
        return None
    if d < 20:
        return "under 20s"
    if d < 30:
        return "20-30s"
    if d < 45:
        return "30-45s"
    return "45-60s"


def _title_length(row: dict) -> str:
    n = row["title_length"]
    return "under 40 chars" if n < 40 else "40-70 chars" if n < 70 else "70+ chars"


def _first_words(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or "first_word_delay" not in clip:
        return None
    delay = clip["first_word_delay"]
    if delay is None:
        return "no talking"
    return "within 0.3s" if delay < 0.3 else "0.3-1s in" if delay < 1.0 else "after 1s"


def _opening_pace(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or "words_per_sec_opening" not in clip:
        return None
    wps = clip["words_per_sec_opening"]
    return "fast (2.5+ words/s)" if wps >= 2.5 else "some (1-2.5 words/s)" if wps >= 1 else "little or none"


def _longest_gap(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or "longest_speech_gap" not in clip:
        return None
    gap = clip["longest_speech_gap"]
    return "under 1s" if gap < 1 else "1-2.5s" if gap < 2.5 else "2.5s+"


def _opening_loudness(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or "opening_vs_median_db" not in clip:
        return None
    db = clip["opening_vs_median_db"]
    return "louder than the rest" if db >= 3 else "quieter than the rest" if db <= -3 else "about the same"


_LAYOUT_LABELS = {
    "crop": "single crop",
    "split": "gameplay + facecam",
    "multicam": "gameplay + several cams",
    "letterbox": "full frame (letterboxed)",
}


def _layout(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or not clip.get("layout"):
        return None
    return _LAYOUT_LABELS.get(clip["layout"], clip["layout"])


GROUPS: List[tuple] = [
    ("length", "Clip length", _length, ["under 20s", "20-30s", "30-45s", "45-60s"]),
    ("title_question", "Title asks a question", lambda r: "yes" if r["title_is_question"] else "no", ["yes", "no"]),
    ("title_caps", "Title has ALL-CAPS words", lambda r: "yes" if r["title_caps_words"] else "no", ["yes", "no"]),
    ("title_length", "Title length", _title_length, ["under 40 chars", "40-70 chars", "70+ chars"]),
    ("first_words", "First words start", _first_words, ["within 0.3s", "0.3-1s in", "after 1s", "no talking"]),
    ("opening_pace", "Talking in the first 3s", _opening_pace, ["fast (2.5+ words/s)", "some (1-2.5 words/s)", "little or none"]),
    ("longest_gap", "Longest stretch with no talking", _longest_gap, ["under 1s", "1-2.5s", "2.5s+"]),
    ("opening_loudness", "Opening loudness vs the rest of the clip", _opening_loudness,
     ["louder than the rest", "about the same", "quieter than the rest"]),
    ("layout", "Layout", _layout, list(_LAYOUT_LABELS.values())),
]

# Groups built from this app's own measurements -- only clips made here and
# matched to a posted video can fill these.
MADE_HERE_GROUPS = {"first_words", "opening_pace", "longest_gap", "opening_loudness", "layout"}


def _mean(values: list) -> Optional[float]:
    return round(statistics.mean(values), 3) if values else None


def _median(values: list) -> Optional[float]:
    return statistics.median(values) if values else None


def _bucket_stats(label: str, rows: List[dict]) -> dict:
    views = [r["views"] for r in rows if r.get("views") is not None]
    pcts = [r["avg_view_pct"] for r in rows if r.get("avg_view_pct") is not None]
    watch_3s = [r["retention"]["watch_3s"] for r in rows if r.get("retention")]
    median_views = _median(views)
    return {
        "label": label,
        "n": len(rows),
        "median_views": round(median_views) if median_views is not None else None,
        "avg_view_pct": round(statistics.mean(pcts), 1) if pcts else None,
        "watch_3s": _mean(watch_3s),
        "n_retention": len(watch_3s),
        "enough": len(rows) >= MIN_GROUP_SIZE,
    }


def _group(rows: List[dict], key: str, name: str, fn: Callable, order: List[str]) -> Optional[dict]:
    buckets: dict = {}
    for r in rows:
        label = fn(r)
        if label is not None:
            buckets.setdefault(label, []).append(r)
    if len(buckets) < 2:
        return None  # nothing to compare against
    labels = [lbl for lbl in order if lbl in buckets] + sorted(lbl for lbl in buckets if lbl not in order)
    return {
        "key": key,
        "name": name,
        "made_here_only": key in MADE_HERE_GROUPS,
        "buckets": [_bucket_stats(lbl, buckets[lbl]) for lbl in labels],
    }


def build_stats(videos: List[dict], metrics_by_id: dict, curves_by_id: dict, records: List[dict]) -> dict:
    rows = build_rows(videos, metrics_by_id, curves_by_id, records)
    settled = [r for r in rows if not r["too_new_to_judge"]]
    with_curve = [r for r in settled if r.get("retention")]
    views = [r["views"] for r in settled if r.get("views") is not None]
    pcts = [r["avg_view_pct"] for r in settled if r.get("avg_view_pct") is not None]

    def curve_values(key: str) -> list:
        return [r["retention"][key] for r in with_curve if r["retention"][key] is not None]

    drops = curve_values("steepest_drop_at")
    median_views = _median(views)
    summary = {
        "shorts": len(settled),
        "too_new": len(rows) - len(settled),
        "with_retention": len(with_curve),
        "made_here": sum(1 for r in settled if r["made_here"]),
        "median_views": round(median_views) if median_views is not None else None,
        "avg_view_pct": round(statistics.mean(pcts), 1) if pcts else None,
        "watch_1s": _mean(curve_values("watch_1s")),
        "watch_3s": _mean(curve_values("watch_3s")),
        "watch_mid": _mean(curve_values("watch_mid")),
        "watch_end": _mean(curve_values("watch_end")),
        "typical_steepest_drop_at": round(statistics.median(drops), 1) if drops else None,
        "opening_vs_similar": _mean(curve_values("opening_vs_similar")),
        "overall_vs_similar": _mean(curve_values("overall_vs_similar")),
    }
    groups = [g for g in (_group(settled, *spec) for spec in GROUPS) if g]
    rows.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return {"summary": summary, "groups": groups, "videos": rows}


def _pct(fraction: Optional[float]) -> Optional[str]:
    return f"{fraction * 100:.0f}%" if fraction is not None else None


def _bucket_line(b: dict) -> str:
    parts = [f"{b['label']} n={b['n']}" + ("" if b["enough"] else " TOO FEW")]
    details = []
    if b["median_views"] is not None:
        details.append(f"median {b['median_views']:,} views")
    if b["avg_view_pct"] is not None:
        details.append(f"{b['avg_view_pct']:.0f}% of the video watched")
    if b["watch_3s"] is not None:
        details.append(f"{_pct(b['watch_3s'])} still watching at 3s (from {b['n_retention']} curves)")
    return parts[0] + (": " + ", ".join(details) if details else "")


def render_prompt_text(stats: dict) -> str:
    """The stats as compact plain text for the AI overview prompt."""
    s = stats["summary"]
    lines = [
        f"{s['shorts']} settled Shorts ({s['too_new']} more too new to judge), "
        f"{s['with_retention']} with a retention curve, {s['made_here']} matched to clips made in this app "
        "(only those have the talking/loudness/layout measurements).",
    ]
    overall = []
    if s["median_views"] is not None:
        overall.append(f"median {s['median_views']:,} views")
    if s["avg_view_pct"] is not None:
        overall.append(f"{s['avg_view_pct']:.0f}% of the video watched on average")
    if s["watch_1s"] is not None:
        overall.append(
            f"per view, {_pct(s['watch_1s'])} still watching at 1s, {_pct(s['watch_3s'])} at 3s, "
            f"{_pct(s['watch_mid'])} at the halfway point, {_pct(s['watch_end'])} near the end"
        )
    if s["typical_steepest_drop_at"] is not None:
        overall.append(f"the single biggest drop typically lands at {s['typical_steepest_drop_at']}s")
    if overall:
        lines.append("Overall: " + "; ".join(overall) + ".")
    if s["opening_vs_similar"] is not None and s["overall_vs_similar"] is not None:
        lines.append(
            f"First 3 seconds vs similar-length YouTube videos: {s['opening_vs_similar']:.2f} "
            f"(0.5 = typical, below 0.5 = losing more viewers than similar videos); "
            f"whole video: {s['overall_vs_similar']:.2f}."
        )
    lines.append(
        f"Comparisons (n = uploads in the group; groups under {MIN_GROUP_SIZE} are marked TOO FEW "
        "and cannot support a conclusion on their own):"
    )
    for g in stats["groups"]:
        lines.append(f"- {g['name']}: " + "; ".join(_bucket_line(b) for b in g["buckets"]))
    return "\n".join(lines)
