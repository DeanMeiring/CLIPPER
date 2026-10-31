"""Ball Evolution learns from its own numbers (Dean, 29 Oct: "it sees which
video gets a lot of views ... and then it decides which videos to make").

Once a day: the channel's YouTube Analytics per video (views, engaged views
-- the ones that didn't swipe away --, likes, shares, average % watched),
matched to the kind each video was (its recipe: format, and the duel's /
tournament's / royale's look), summed up per kind, then Claude reads that
table and sets the mix for the next videos: a share per kind, the looks of
the fights, and a few plain notes for Dean. Guardrails in code, not in the
prompt: every kind in the rotation keeps a small share (so a kind that
hasn't had its lucky video yet still gets tried), no kind takes more than
a third, Evolve stays out (Dean dropped it). Without a Claude answer the mix
comes from the same table by a fixed formula.
"""
from __future__ import annotations

import datetime
import json
import math
import random
import statistics
from typing import Callable, Dict, List, Optional

ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"
MIN_AGE_H = 48            # a video's numbers count once it's had two days (the feed tests it first)
LOOKBACK_DAYS = 60
FLOOR = 2.0               # % every rotation kind keeps
CAP = 34.0                # % no kind goes over
LOOK_FLOOR = 10.0         # % every look of a fight keeps
LOOKS = {"duel": ("colours", "flags", "teams"), "tourney": ("colours", "flags"), "royale": ("colours", "flags")}
NEVER = ("evolve",)       # Dean dropped Evolve from the autopilot (11 Oct)


def fetch_stats(access_token: str, channel_id: str, days: int = LOOKBACK_DAYS, http=None) -> Dict[str, dict]:
    """Per-video numbers from the channel owner's own Analytics, keyed by
    video id. A "video" report must be sorted by views (sorting by likes is
    what made the main channel's report answer 400)."""
    import requests
    http = http or requests
    # yesterday (UTC): a date that's still tomorrow on YouTube's Pacific
    # clock gave an empty report (31 Oct, 05:07 UTC), and today has no
    # numbers yet anyway
    end = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    params = {"ids": f"channel=={channel_id}", "startDate": (end - datetime.timedelta(days=days)).isoformat(),
              "endDate": end.isoformat(), "dimensions": "video", "sort": "-views", "maxResults": 200}
    names = ["views", "engaged_views", "likes", "comments", "shares", "avg_pct"]
    metrics = "views,engagedViews,likes,comments,shares,averageViewPercentage"
    r = http.get(ANALYTICS_URL, headers={"Authorization": f"Bearer {access_token}"},
                 params={**params, "metrics": metrics}, timeout=30)
    if r.status_code == 400:            # no engagedViews on this account / API version
        names.remove("engaged_views")
        r = http.get(ANALYTICS_URL, headers={"Authorization": f"Bearer {access_token}"},
                     params={**params, "metrics": metrics.replace("engagedViews,", "")}, timeout=30)
    r.raise_for_status()
    out = {}
    for row in r.json().get("rows") or []:
        out[row[0]] = {n: float(row[i + 1] or 0) for i, n in enumerate(names)}
    return out


def kind_of(recipe: dict) -> Optional[str]:
    fmt = (recipe or {}).get("format")
    if not fmt:
        return None
    if fmt in LOOKS:
        look = recipe.get("look") or "colours"
        return f"{fmt}:{look}"
    return fmt


def table(videos: List[dict], stats: Dict[str, dict], now: Optional[float] = None) -> dict:
    """Per kind: how many videos, middle views, likes per 100 views, the share
    that kept watching, average % watched -- plus the channel's own middle
    and each kind's best and worst video (title + views)."""
    now = now or datetime.datetime.now(datetime.timezone.utc).timestamp()
    rows: Dict[str, list] = {}
    for v in videos:
        yt = v.get("youtube") or {}
        vid = yt.get("video_id")
        st = stats.get(vid) if vid else None
        kind = kind_of(v.get("recipe") or {})
        posted = yt.get("publish_at_ts") or yt.get("uploaded_at") or 0
        if not st or not kind or now - posted < MIN_AGE_H * 3600 or st.get("views", 0) <= 0:
            continue
        rows.setdefault(kind, []).append({"title": v.get("title") or (v.get("recipe") or {}).get("hook") or "",
                                          "posted": posted, **st})
    every = [r for rs in rows.values() for r in rs]

    def summary(rs):
        views = [r["views"] for r in rs]
        out = {"videos": len(rs), "median_views": round(statistics.median(views)),
               "likes_per_100": round(100 * sum(r["likes"] for r in rs) / max(1, sum(views)), 2),
               "shares_per_1000": round(1000 * sum(r.get("shares", 0) for r in rs) / max(1, sum(views)), 2),
               "avg_pct": round(statistics.mean(r.get("avg_pct", 0) for r in rs), 1)}
        if all("engaged_views" in r for r in rs):
            out["stayed_pct"] = round(100 * sum(r["engaged_views"] for r in rs) / max(1, sum(views)), 1)
        best = max(rs, key=lambda r: r["views"])
        worst = min(rs, key=lambda r: r["views"])
        out["best"] = {"title": best["title"][:80], "views": round(best["views"])}
        out["worst"] = {"title": worst["title"][:80], "views": round(worst["views"])}
        return out

    return {"kinds": {k: summary(rs) for k, rs in sorted(rows.items())},
            "channel": summary(every) if every else None, "videos": len(every)}


def _score(row: dict, channel: dict) -> float:
    """How a kind does against the channel's own middle (0 = average)."""
    def ratio(a, b):
        return math.log2(max(a, 1e-6) / max(b, 1e-6))
    s = 0.5 * ratio(row["median_views"], channel["median_views"])
    s += 0.25 * ratio(row["likes_per_100"] + 0.1, channel["likes_per_100"] + 0.1)
    if "stayed_pct" in row and "stayed_pct" in channel:
        s += 0.25 * ratio(row["stayed_pct"], channel["stayed_pct"])
    # few videos: don't trust it much yet
    return s * min(1.0, row["videos"] / 4)


def formula_mix(tab: dict, rotation: List[str]) -> dict:
    """The fallback mix: 2^(1.5 x score) per kind, the looks the same way."""
    ch = tab.get("channel")
    raw = {}
    for k in rotation:
        rows = [(kk, r) for kk, r in tab["kinds"].items() if kk.split(":")[0] == k]
        if not rows or not ch:
            raw[k] = 1.0
            continue
        best = max(_score(r, ch) for _kk, r in rows)
        raw[k] = 2 ** (1.5 * best)
    looks = {}
    for fmt, opts in LOOKS.items():
        w = {}
        for o in opts:
            r = tab["kinds"].get(f"{fmt}:{o}")
            w[o] = 2 ** (1.5 * _score(r, ch)) if (r and ch) else 1.0
        looks[fmt] = w
    return {"weights": raw, "looks": looks, "notes": []}


def _spread(w: Dict[str, float], floor: Dict[str, float], cap: float) -> Dict[str, float]:
    """Scale w to 100%, with each share held between its floor and the cap:
    the ones that hit a limit are fixed there and the rest share what's left
    in proportion."""
    fixed: Dict[str, float] = {}
    scaled: Dict[str, float] = {}
    for _ in range(len(w) + 1):
        free = {k: v for k, v in w.items() if k not in fixed}
        if not free:
            break
        left = 100.0 - sum(fixed.values())
        total = sum(free.values())
        scaled = {k: (left * v / total if total > 0 else left / len(free)) for k, v in free.items()}
        out = [k for k, v in scaled.items() if v < floor.get(k, 0) - 1e-9 or v > cap + 1e-9]
        if not out:
            break
        for k in out:
            fixed[k] = floor.get(k, 0) if scaled[k] < floor.get(k, 0) else cap
        scaled = {}
    return {**fixed, **scaled}


def clamp(plan: dict, rotation: List[str], allowed: List[str]) -> dict:
    """Keep Claude's (or the formula's) mix inside the guardrails."""
    w = {}
    for k, v in (plan.get("weights") or {}).items():
        if k in allowed and k not in NEVER and isinstance(v, (int, float)) and not isinstance(v, bool):
            w[k] = max(0.0, float(v))
    for k in rotation:
        w.setdefault(k, 0.0)
    w = _spread(w, {k: FLOOR for k in rotation}, CAP)
    w = {k: round(v, 1) for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v >= 0.05}
    looks = {}
    for fmt, opts in LOOKS.items():
        given = (plan.get("looks") or {}).get(fmt) or {}
        lw = {}
        for o in opts:
            try:
                lw[o] = max(0.0, float(given.get(o, 0) or 0))
            except (TypeError, ValueError):
                lw[o] = 0.0
        if not sum(lw.values()):
            lw = {o: 1.0 for o in opts}
        looks[fmt] = {o: round(v, 1) for o, v in _spread(lw, {o: LOOK_FLOOR for o in opts}, 100.0).items()}
    notes = [str(n).strip()[:200] for n in (plan.get("notes") or []) if str(n).strip()][:5]
    return {"weights": w, "looks": looks, "notes": notes}


_PROMPT = """You decide what a small YouTube Shorts channel makes next. The channel posts \
2 physics-simulation Shorts a day (bouncing balls, made by code), picking each video's \
"kind" from a mix you set. Here are the channel's own numbers from the last {days} days, \
per kind (only videos at least 2 days old):

{table}

Channel middle: {channel}

What the columns mean: median_views = the middle video's views; stayed_pct = share of \
views that kept watching instead of swiping away (the Shorts feed weighs this most); \
likes_per_100 and shares_per_1000 = how many reacted; avg_pct = average % of the video \
watched. Fight kinds are split by look: colours (two coloured balls), flags (countries), \
teams (2 VS 2).

Kinds you can use (with what they are): {kinds}
Kinds with no numbers yet: {untested}

Set the mix for the next two weeks:
- More of what wins, less of what doesn't -- but a single lucky video is luck: a kind with \
1-2 videos is still being tested. Shorts views are very uneven; weigh stayed_pct and \
likes_per_100 (steadier) together with views.
- Give every kind in the rotation at least 2%, nothing over 34%.
- For the fights, the share of each look.
- Notes: 2-4 short plain sentences for the channel owner (no jargon, no hype): what's \
working, what you changed and why.

Answer with JSON only:
{{"weights": {{"<kind>": <percent>, ...}}, "looks": {{"duel": {{"colours": n, "flags": n, \
"teams": n}}, "tourney": {{"colours": n, "flags": n}}, "royale": {{"colours": n, "flags": n}}}}, \
"notes": ["..."]}}"""


def plan(tab: dict, rotation: List[str], allowed: List[str], labels: Dict[str, str], api_key: Optional[str],
         ask: Optional[Callable] = None, log: Callable[[str], None] = print) -> dict:
    """The mix for the next videos: Claude's, kept inside the guardrails, or
    the formula's when there's no key or no usable answer."""
    source = "formula"
    raw = None
    if api_key and tab.get("channel"):
        try:
            from clipper import ball_packs
            ask = ask or (lambda text: ball_packs._claude_json(text, api_key, max_tokens=6000))
            untested = [k for k in rotation if not any(kk.split(":")[0] == k for kk in tab["kinds"])]
            raw = ask(_PROMPT.format(
                days=LOOKBACK_DAYS,
                table=json.dumps(tab["kinds"], indent=1),
                channel=json.dumps({k: v for k, v in tab["channel"].items() if k not in ("best", "worst")}),
                kinds=", ".join(f"{k} ({labels.get(k, k)})" for k in allowed),
                untested=", ".join(untested) or "none"))
            source = "claude"
        except Exception as e:          # no answer: the formula decides, the autopilot carries on
            log(f"[balls-learn] Claude's plan failed, using the formula: {e}")
            raw = None
    if raw is None:
        raw = formula_mix(tab, rotation) if tab.get("channel") else {"weights": {k: 1 for k in rotation}}
    out = clamp(raw, rotation, allowed)
    out["source"] = source
    return out


def pick(weights: Dict[str, float], seed: str, avoid: Optional[str] = None) -> Optional[str]:
    """A kind by the mix (the same answer for the same seed), not the one
    the slot before had unless it's the only one."""
    items = [(k, w) for k, w in weights.items() if w > 0]
    if not items:
        return None
    choice = random.Random(seed).choices([k for k, _ in items], [w for _, w in items])[0]
    if choice == avoid and len(items) > 1:
        rest = [(k, w) for k, w in items if k != avoid]
        choice = random.Random(seed + "-again").choices([k for k, _ in rest], [w for _, w in rest])[0]
    return choice


# ---- the daily report: everything above, per video, for Claude Code to read --
# Dean (29 Oct): "let it pull a report ... available to you so that you can
# read it. Then you can edit it as you see fit." Each learning run writes
# report.json and prints it to the app's logs as "[balls-report]" lines (one
# per video, so no line gets long), which a Claude Code session reads with
# Railway's get-logs (filter "balls-report") to decide what to change in the
# videos themselves.
REPORT_TAG = "[balls-report]"
CURVES = 3                # retention curves for the best and the worst this many


def _sound(recipe: dict) -> str:
    song = recipe.get("song") or {}
    if song.get("name"):
        return f"song: {song['name']}"
    return recipe.get("bounce_sound") or recipe.get("bed") or ""


def _pictures(recipe: dict) -> str:
    """What the balls were: a fight's contestants, else the picture pack
    or the emoji theme. (A fight's recipe still carries a theme it never
    shows.)"""
    players = recipe.get("players")
    if recipe.get("format") == "crowd":          # commenters (ball_crowd)
        cast = (recipe.get("crowd") or {}).get("cast") or []
        real = [m["name"] for m in cast if not m.get("bot")]
        return " VS ".join(real) if len(cast) <= 2 else f"{len(real)} commenters + {len(cast) - len(real)} bots"
    if recipe.get("format") in LOOKS or recipe.get("format") in ("race", "survivor"):
        if recipe.get("team_names"):
            return " VS ".join(recipe["team_names"])
        if not players:
            return ""
        names = [" ".join(x for x in p[:2] if x) for p in players]
        return " VS ".join(names) if len(names) <= 4 else f"{len(names)} {'countries' if recipe.get('look') == 'flags' else 'colours'}"
    return (recipe.get("pack") or {}).get("title") or recipe.get("theme") or ""


def video_rows(videos: List[dict], stats: Dict[str, dict], now: Optional[float] = None) -> List[dict]:
    """One row per posted video with numbers: what it was (kind, look, hook,
    pictures, sound, length) next to how it did. Newest first."""
    now = now or datetime.datetime.now(datetime.timezone.utc).timestamp()
    rows = []
    for v in videos:
        yt = v.get("youtube") or {}
        st = stats.get(yt.get("video_id") or "")
        rec = v.get("recipe") or {}
        if not st or not rec.get("format"):
            continue
        posted = yt.get("publish_at_ts") or yt.get("uploaded_at") or 0
        views = st.get("views", 0)
        row = {"yt": yt["video_id"], "posted": datetime.datetime.fromtimestamp(posted, datetime.timezone.utc)
               .strftime("%Y-%m-%d %H:%M"), "age_h": round((now - posted) / 3600),
               "kind": kind_of(rec), "title": (v.get("title") or "")[:90], "hook": (rec.get("hook") or "")[:80],
               "pictures": _pictures(rec)[:60],
               "sound": _sound(rec)[:60], "art": rec.get("art") or "", "seconds": round(v.get("seconds") or 0, 1),
               "views": round(views), "avg_pct": round(st.get("avg_pct", 0), 1),
               "likes": round(st.get("likes", 0)), "comments": round(st.get("comments", 0)),
               "shares": round(st.get("shares", 0))}
        if "engaged_views" in st and views:
            row["stayed_pct"] = round(100 * st["engaged_views"] / views, 1)
        rows.append(row)
    rows.sort(key=lambda r: r["posted"], reverse=True)
    return rows


def curve_picks(rows: List[dict], n: int = CURVES) -> List[dict]:
    """The best and worst n videos by views (at least MIN_AGE_H old)."""
    old = sorted((r for r in rows if r["age_h"] >= MIN_AGE_H), key=lambda r: -r["views"])
    if len(old) <= 2 * n:
        return old
    return old[:n] + old[-n:]


def short_curve(points: list, n: int = 20) -> List[float]:
    """A retention curve (ElapsedVideoTimeRatio, watch ratio, ...) cut to n
    points: the share still watching at 5%, 10% ... 100% of the video."""
    if not points:
        return []
    out = []
    for i in range(1, n + 1):
        at = i / n
        p = min(points, key=lambda q: abs(q[0] - at))
        out.append(round(float(p[1]), 2))
    return out


def log_report(report: dict, log: Callable[[str], None] = print) -> None:
    """The report as log lines: a summary line, the mix, each kind, each
    video, each curve -- every line "[balls-report] <json>" with the run's
    time, so one run's lines can be picked out of the logs."""
    run = datetime.datetime.fromtimestamp(report["at"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    def line(kind, data):
        log(f"{REPORT_TAG} " + json.dumps({"run": run, "type": kind, **data}, ensure_ascii=False,
                                          separators=(",", ":")))
    line("summary", {"videos": len(report["videos"]), "source": report.get("source"),
                     "channel": report.get("channel"), "notes": report.get("notes") or []})
    line("mix", {"weights": report.get("weights") or {}, "looks": report.get("looks") or {}})
    for k, r in (report.get("kinds") or {}).items():
        line("kind", {"kind": k, **r})
    for r in report["videos"]:
        line("video", r)
    for yt, c in (report.get("curves") or {}).items():
        line("curve", {"yt": yt, "watching": c})
    line("end", {"lines": 3 + len(report.get("kinds") or {}) + len(report["videos"]) + len(report.get("curves") or {})})
