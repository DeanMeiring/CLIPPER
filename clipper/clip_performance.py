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

import datetime
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
            "subs_gained": m.get("subscribers_gained"),
            "engaged_views": m.get("engaged_views"),
            "likes": m.get("likes"),
            "comments": m.get("comments"),
            "shares": m.get("shares"),
            "retention": curve_metrics(curves_by_id.get(v["id"]) or [], v.get("duration_seconds")),
            "made_here": record is not None,
            "clip": None,
        }
        row.update(title_features(v["title"]))
        views = row["views"] or 0
        # Engaged views / views: the share of plays that didn't swipe away
        # in the first moments (see youtube_analytics.get_video_engagement).
        row["stayed"] = round(min(1.0, row["engaged_views"] / views), 3) if row["engaged_views"] is not None and views else None
        row["shares_per_1k"] = round(row["shares"] / views * 1000, 2) if row["shares"] is not None and views else None
        row["comments_per_1k"] = round(row["comments"] / views * 1000, 2) if row["comments"] is not None and views else None
        if record is not None:
            row["source_title"] = record.get("source_title")
            row["clip"] = {
                **(record.get("features") or {}),
                "layout": record.get("layout"),
                "link_method": record.get("link_method"),
                "hook_caption": record.get("hook_caption"),
                "hook_text": record.get("hook_text"),
                "branding": record.get("branding"),
                "score": record.get("score"),
                "moment_type": record.get("moment_type"),
                "subscores": record.get("subscores"),
                "edits": record.get("edits"),
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


def _hook_text(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip:
        return None
    # Clips made before hook text existed have no field -- they had none.
    return "yes" if clip.get("hook_text") else "no"


def _branding(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip:
        return None
    # Clips made before branding existed had none.
    return "yes" if clip.get("branding") else "no"


def _paced(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip:
        return None
    # Clips made before pacing edits existed had none.
    return "yes" if (clip.get("edits") or {}).get("cut_seconds", 0) > 0 else "no"


def _teaser(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip:
        return None
    return "yes" if (clip.get("edits") or {}).get("teaser") else "no"


def _picker(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip:
        return "not matched to a clip made here"
    # Moment types and sub-scores arrived with the current way of picking
    # and cutting clips (hook-first cuts, strong-stance ranking, pacing,
    # branding), so they mark which clips it made.
    if clip.get("moment_type") or clip.get("subscores"):
        return "made here, current picker"
    return "made here, earlier version"


def _posted(row: dict) -> Optional[str]:
    age = row.get("age_days")
    if age is None:
        return None
    return "last 7 days" if age <= 7 else "8-28 days ago" if age <= 28 else "older"


def _band(value) -> Optional[str]:
    if value is None:
        return None
    return "8-10" if value >= 8 else "5-7" if value >= 5 else "1-4"


def _score(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or clip.get("score") is None:
        return None
    return "8-10" if clip["score"] >= 8 else "5-7" if clip["score"] >= 5 else "1-4"


def _layout(row: dict) -> Optional[str]:
    clip = row.get("clip")
    if not clip or not clip.get("layout"):
        return None
    return _LAYOUT_LABELS.get(clip["layout"], clip["layout"])


GROUPS: List[tuple] = [
    ("posted", "When it was posted", _posted, ["last 7 days", "8-28 days ago", "older"]),
    ("picker", "Which version of this app made it", _picker,
     ["made here, current picker", "made here, earlier version", "not matched to a clip made here"]),
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
    ("hook_text", "Hook text on screen for the first 3s", _hook_text, ["yes", "no"]),
    ("branding", "Channel mascot + name on screen", _branding, ["yes", "no"]),
    ("paced", "Pauses cut out", _paced, ["yes", "no"]),
    ("teaser", "Opens on a payoff teaser", _teaser, ["yes", "no"]),
    ("score", "Claude's score when it picked the clip", _score, ["8-10", "5-7", "1-4"]),
    ("hook_score", "Claude's hook score", lambda r: _band(((r.get("clip") or {}).get("subscores") or {}).get("hook")),
     ["8-10", "5-7", "1-4"]),
    ("controversy_score", "Claude's controversy score",
     lambda r: _band(((r.get("clip") or {}).get("subscores") or {}).get("controversy")), ["8-10", "5-7", "1-4"]),
    ("moment_type", "Kind of moment", lambda r: (r.get("clip") or {}).get("moment_type"),
     ["controversy", "drama", "fail", "rage", "funny", "skill", "wholesome", "other"]),
]

# Groups built from this app's own measurements -- only clips made here and
# matched to a posted video can fill these.
MADE_HERE_GROUPS = {"first_words", "opening_pace", "longest_gap", "opening_loudness", "layout", "hook_text", "branding", "paced", "teaser", "score", "moment_type", "hook_score", "controversy_score"}


def _mean(values: list) -> Optional[float]:
    return round(statistics.mean(values), 3) if values else None


def _median(values: list) -> Optional[float]:
    return statistics.median(values) if values else None


def _subs_per_1k(rows: List[dict]) -> Optional[float]:
    """New subscribers per 1,000 views, pooled over the rows that have both
    numbers -- how well a kind of clip turns viewers into subscribers."""
    pairs = [(r["subs_gained"], r["views"]) for r in rows if r.get("subs_gained") is not None and r.get("views")]
    views = sum(v for _, v in pairs)
    return round(sum(s for s, _ in pairs) / views * 1000, 2) if views else None


def _bucket_stats(label: str, rows: List[dict]) -> dict:
    views = [r["views"] for r in rows if r.get("views") is not None]
    pcts = [r["avg_view_pct"] for r in rows if r.get("avg_view_pct") is not None]
    watch_3s = [r["retention"]["watch_3s"] for r in rows if r.get("retention")]
    stayed = [r["stayed"] for r in rows if r.get("stayed") is not None]
    median_views = _median(views)
    return {
        "label": label,
        "n": len(rows),
        "median_views": round(median_views) if median_views is not None else None,
        "stayed": _mean(stayed),
        "subs_per_1k": _subs_per_1k(rows),
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


def build_stats(videos: List[dict], metrics_by_id: dict, curves_by_id: dict, records: List[dict],
                streamers: Optional[List[str]] = None) -> dict:
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
    summary["subs_per_1k"] = _subs_per_1k(settled)
    summary["stayed"] = _mean([r["stayed"] for r in settled if r.get("stayed") is not None])
    for r in rows:
        r["streamer"] = _streamer(r, streamers or [])
    groups = [g for g in (_group(settled, *spec) for spec in GROUPS) if g]
    rows.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return {"summary": summary, "groups": groups, "videos": rows, "learning": build_learning(settled)}


# ---------------------------------------------------------- learning ---
# Group averages alone taught the picker little (Dean, Oct 2026: views
# declining). These give it concrete cases from the channel itself: the
# best and worst recent Shorts with what they were, which streamers'
# clips are rising or wearing out, and whether the last two weeks are up
# or down -- the closest thing to watching the channel's own analytics.

RECENT_DAYS = 60


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _streamer(row: dict, streamers: List[str]) -> Optional[str]:
    """Which tracked streamer a Short is about: the one whose name appears
    first in its title (titles lead with the streamer's name), else in the
    source stream's title. None when no tracked name appears."""
    names = [(n, _norm(n)) for n in streamers if len(_norm(n)) >= 3]
    for text in (row.get("title"), row.get("source_title")):
        hay = _norm(text)
        found = [(hay.find(key), name) for name, key in names if key and key in hay]
        if found:
            return min(found)[1]
    return None


def _example(r: dict) -> dict:
    clip = r.get("clip") or {}
    return {
        "id": r["id"], "title": r["title"], "views": r.get("views"), "duration": r.get("duration"),
        "age_days": r.get("age_days"), "streamer": r.get("streamer"),
        "stayed": r.get("stayed"), "watch_3s": (r.get("retention") or {}).get("watch_3s"),
        "avg_view_pct": r.get("avg_view_pct"), "shares_per_1k": r.get("shares_per_1k"),
        "comments_per_1k": r.get("comments_per_1k"), "subs_gained": r.get("subs_gained"),
        "moment_type": clip.get("moment_type"), "hook_caption": clip.get("hook_caption"),
        "score": clip.get("score"), "first_word_delay": clip.get("first_word_delay"),
    }


def _window(rows: List[dict], lo: int, hi: int) -> dict:
    part = [r for r in rows if r.get("age_days") is not None and lo <= r["age_days"] <= hi]
    views = [r["views"] for r in part if r.get("views") is not None]
    med = _median(views)
    return {
        "n": len(part),
        "median_views": round(med) if med is not None else None,
        "stayed": _mean([r["stayed"] for r in part if r.get("stayed") is not None]),
        "watch_3s": _mean([r["retention"]["watch_3s"] for r in part if r.get("retention")]),
    }


def build_learning(settled: List[dict]) -> dict:
    recent = [r for r in settled if r.get("age_days") is not None and r["age_days"] <= RECENT_DAYS and r.get("views") is not None]
    ranked = sorted(recent, key=lambda r: r["views"], reverse=True)
    k = min(6, len(ranked) // 3)
    best = [_example(r) for r in ranked[:k]] if k >= 2 else []
    worst = [_example(r) for r in ranked[-k:][::-1]] if k >= 2 else []
    by: dict = {}
    for r in recent:
        if r.get("streamer"):
            by.setdefault(r["streamer"], []).append(r)
    streamers = []
    for name, rs in by.items():
        if len(rs) < 2:
            continue
        newer = [r["views"] for r in rs if r["age_days"] <= 21]
        older = [r["views"] for r in rs if r["age_days"] > 21]
        med = _median([r["views"] for r in rs])
        streamers.append({
            "name": name, "n": len(rs), "median_views": round(med) if med is not None else None,
            "stayed": _mean([r["stayed"] for r in rs if r.get("stayed") is not None]),
            "recent_median": round(_median(newer)) if newer else None, "recent_n": len(newer),
            "earlier_median": round(_median(older)) if older else None, "earlier_n": len(older),
        })
    streamers.sort(key=lambda x: (-x["n"], -(x["median_views"] or 0)))
    return {
        "best": best, "worst": worst, "streamers": streamers[:12],
        "last_14": _window(settled, 0, 14), "prev_14": _window(settled, 15, 28),
    }


TREND_WEEKS = 8


def _day(value: Optional[str]) -> Optional[datetime.date]:
    try:
        return datetime.date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _change_pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    return round((new - old) / old * 100) if new is not None and old else None


def build_trend(daily: List[dict], rows: List[dict], weeks: int = TREND_WEEKS) -> Optional[dict]:
    """Week-by-week growth: channel views and subscribers (from
    youtube_analytics.get_daily_totals) next to the Shorts posted that week
    and how many views they got. Weeks are the 7 days ending on the latest
    day YouTube has numbers for, not the calendar week -- today and
    yesterday aren't in the Analytics data yet, and counting them as zero
    would make every week look like a drop."""
    by_day = {d: row for row in daily if (d := _day(row.get("date")))}
    if not by_day:
        return None
    first, last = min(by_day), max(by_day)
    out = []
    for i in range(weeks - 1, -1, -1):
        end = last - datetime.timedelta(days=7 * i)
        start = end - datetime.timedelta(days=6)
        if start < first:
            continue  # only part of this week is in the data -- it would read as a dip
        days = [by_day[d] for d in by_day if start <= d <= end]
        posted = [r for r in rows if (p := _day(r.get("published_at"))) and start <= p <= end]
        post_views = [r["views"] for r in posted if r.get("views") is not None]
        gained = sum(d.get("subscribers_gained") or 0 for d in days)
        lost = sum(d.get("subscribers_lost") or 0 for d in days)
        views = sum(d.get("views") or 0 for d in days)
        out.append({
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": f"{start.day} {start:%b} - {end.day} {end:%b}",
            "views": round(views),
            "subscribers_gained": gained,
            "subscribers_net": gained - lost,
            "subs_per_1k": round(gained / views * 1000, 2) if views else None,
            "shorts_posted": len(posted),
            "made_here_posted": sum(1 for r in posted if r.get("made_here")),
            "median_views_per_short": round(_median(post_views)) if post_views else None,
        })
    if not out:
        return None
    this, prev = out[-1], (out[-2] if len(out) > 1 else None)
    return {
        "weeks": out,
        "data_through": last.isoformat(),
        "last_week": this,
        "views_change_pct": _change_pct(this["views"], prev["views"] if prev else None),
        "subs_change": this["subscribers_net"] - prev["subscribers_net"] if prev else None,
        "median_views_change_pct": _change_pct(
            this["median_views_per_short"], prev["median_views_per_short"] if prev else None,
        ),
    }


def _pct(fraction: Optional[float]) -> Optional[str]:
    return f"{fraction * 100:.0f}%" if fraction is not None else None


def _bucket_line(b: dict) -> str:
    parts = [f"{b['label']} n={b['n']}" + ("" if b["enough"] else " TOO FEW")]
    details = []
    if b["median_views"] is not None:
        details.append(f"median {b['median_views']:,} views")
    if b.get("subs_per_1k") is not None:
        details.append(f"{b['subs_per_1k']:.1f} new subscribers per 1,000 views")
    if b["avg_view_pct"] is not None:
        details.append(f"{b['avg_view_pct']:.0f}% of the video watched")
    if b["watch_3s"] is not None:
        details.append(f"{_pct(b['watch_3s'])} still watching at 3s (from {b['n_retention']} curves)")
    if b.get("stayed") is not None:
        details.append(f"{_pct(b['stayed'])} stayed to watch")
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
    if s.get("stayed") is not None:
        overall.append(f"{_pct(s['stayed'])} of plays stayed to watch rather than swiping away (engaged views / views)")
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
    trend = stats.get("trend")
    if trend:
        lines.append(
            f"Week by week (7-day weeks up to {trend['data_through']}, oldest first; channel-wide views and "
            "subscribers, and the Shorts posted that week):"
        )
        for w in trend["weeks"]:
            median = w["median_views_per_short"]
            lines.append(
                f"- {w['label']}: {w['views']:,} views, {w['subscribers_net']:+d} subscribers, "
                f"{w['shorts_posted']} Shorts posted ({w['made_here_posted']} made here)"
                + (f", median {median:,} views each" if median is not None else "")
            )
        if trend["views_change_pct"] is not None:
            lines.append(f"Latest week vs the week before: views {trend['views_change_pct']:+d}%.")
    lines += _learning_lines(stats.get("learning") or {})
    return "\n".join(lines)


def _example_line(e: dict) -> str:
    bits = [f"{e['views']:,} views" if e.get("views") is not None else None,
            f"{_pct(e['stayed'])} stayed to watch" if e.get("stayed") is not None else None,
            f"{_pct(e['watch_3s'])} at 3s" if e.get("watch_3s") is not None else None,
            f"{e['avg_view_pct']:.0f}% watched" if e.get("avg_view_pct") is not None else None,
            f"{e['shares_per_1k']:.1f} shares/1k" if e.get("shares_per_1k") is not None else None,
            f"{e['comments_per_1k']:.1f} comments/1k" if e.get("comments_per_1k") is not None else None,
            f"{e['duration']:.0f}s long" if e.get("duration") is not None else None,
            f"a {e['moment_type']} moment" if e.get("moment_type") else None,
            f"hook text \"{e['hook_caption']}\"" if e.get("hook_caption") else None,
            f"first words after {e['first_word_delay']:.1f}s" if e.get("first_word_delay") is not None else None,
            f"you scored it {e['score']}/10 when picking" if e.get("score") is not None else None]
    return f"- \"{e['title']}\": " + ", ".join(b for b in bits if b)


def _learning_lines(learning: dict) -> List[str]:
    out = []
    a, b = learning.get("last_14") or {}, learning.get("prev_14") or {}
    if a.get("n") and b.get("n"):
        def part(w: dict) -> str:
            bits = [f"{w['n']} Shorts", f"median {w['median_views']:,} views" if w.get("median_views") is not None else None,
                    f"{_pct(w['stayed'])} stayed to watch" if w.get("stayed") is not None else None,
                    f"{_pct(w['watch_3s'])} at 3s" if w.get("watch_3s") is not None else None]
            return ", ".join(x for x in bits if x)
        out.append(f"Momentum: Shorts posted in the last 2 weeks: {part(a)}. The 2 weeks before: {part(b)}.")
    if learning.get("best") and learning.get("worst"):
        out.append(
            f"This channel's BEST and WORST Shorts of the last {RECENT_DAYS} days, by views. Before picking, work out "
            "what the best share that the worst lack -- kind of moment, how fast it gets going, the hook, length, "
            "which streamer -- and pick and cut toward the best. Where your own score was high on a flop, that "
            "kind of moment is weaker for this audience than it looks in a transcript:")
        out.append("BEST:")
        out += [_example_line(e) for e in learning["best"]]
        out.append("WORST:")
        out += [_example_line(e) for e in learning["worst"]]
    if learning.get("streamers"):
        out.append(f"By streamer (last {RECENT_DAYS} days; 'recent' = last 3 weeks). A streamer whose recent clips "
                   "do clearly worse than before is wearing out with this audience: only pick them for an "
                   "outstanding moment. One whose recent clips do better is gaining: favour them.")
        for x in learning["streamers"]:
            bits = [f"{x['n']} Shorts", f"median {x['median_views']:,} views" if x.get("median_views") is not None else None,
                    f"{_pct(x['stayed'])} stayed" if x.get("stayed") is not None else None]
            if x.get("recent_median") is not None and x.get("earlier_median") is not None:
                bits.append(f"recent median {x['recent_median']:,} ({x['recent_n']}) vs earlier {x['earlier_median']:,} ({x['earlier_n']})")
            out.append(f"- {x['name']}: " + ", ".join(b for b in bits if b))
    return out
