"""Ask Claude for a plain-language content-strategy overview from a
channel's own stats -- when to post, what kind of content performs best,
and what format/pattern to lean into. Synthesizes the same heuristic/
Analytics data channel_insights.py and youtube_analytics.py already
gather; this doesn't pull any new data of its own, it just asks Claude to
read what's there and give a direct, specific take instead of leaving the
creator to eyeball a table of numbers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# How many past analyses to keep on disk. Only the single most recent one
# actually gets fed into clip selection (see load_latest_overview) -- the
# rest are kept only in case they're useful to look back on later, not to
# accumulate into future prompts, since a channel's own patterns shift
# over time and stale advice from months ago shouldn't outvote what the
# most recent, freshest analysis found.
_MAX_HISTORY = 10


def _load_history(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_overview(path: Path, overview: str, channel_title: Optional[str] = None) -> None:
    """Append this overview to the channel's saved-analysis history on
    disk, so a later clip-selection run can be informed by what's already
    been learned about what performs, without re-running the analysis."""
    entries = _load_history(path)
    entries.append({
        "timestamp": time.time(),
        "overview": overview,
        "channel_title": channel_title,
    })
    entries = entries[-_MAX_HISTORY:]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")
    except OSError:
        pass  # best-effort -- losing the save shouldn't break the request that produced it


def load_latest_overview(path: Path) -> Optional[dict]:
    """The most recently saved analysis, or None if nothing's been saved
    yet. Returns the whole entry (overview text + timestamp + channel
    title) so a caller can show its age, not just use it blind."""
    entries = _load_history(path)
    return entries[-1] if entries else None


def clear_history(path: Path) -> None:
    """Wipe all saved analyses, so a stale or noisy history (e.g. from
    early testing, or the channel's content strategy having genuinely
    changed) stops influencing new clip selection -- the next overview
    generated starts a fresh history rather than building on old notes."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _build_prompt(snapshot: dict, analytics: Optional[dict], focus: Optional[str]) -> str:
    lines = [
        f"Channel: {snapshot.get('channel_title', 'unknown')}",
        f"Subscribers: {snapshot.get('subscriber_count')}",
        f"Total videos on the channel: {snapshot.get('video_count')}",
    ]

    videos = snapshot.get("recent_videos") or []
    if videos:
        # Sorted and split into two explicitly labeled groups instead of
        # one flat list -- doing the comparison ourselves means Claude
        # answers "why do winners win" by actually contrasting two
        # concrete groups, not by eyeballing a table and hoping a pattern
        # jumps out.
        # A video posted hours ago hasn't had time to earn its views, and
        # its views/day score actively understates it (see
        # channel_insights). Ranking it against settled uploads would file
        # it under "lower-performing" for being new and invite a reach- or
        # content-problem diagnosis it hasn't earned -- so hold those out
        # of the comparison and report them separately as not-yet-judgeable.
        too_new = [v for v in videos if v.get("too_new_to_judge")]
        settled = [v for v in videos if not v.get("too_new_to_judge")]
        ranked = sorted(settled, key=lambda v: v["views_per_day"], reverse=True)

        def _fmt(v: dict) -> str:
            weekday_name = _WEEKDAY_NAMES[v["weekday"]]
            line = f'- "{v["title"]}" -- {v["views"]} views, {v["views_per_day"]}/day, posted {weekday_name}'
            # Per-video retention (real Analytics data, only present when
            # connected) is what separates "nobody clicked this" from
            # "people clicked but didn't stay" -- two different problems
            # that look identical from view count alone.
            duration = v.get("average_view_duration_seconds")
            pct = v.get("average_view_percentage")
            if duration is not None and pct is not None:
                line += f", retention: {pct:.0f}% watched ({duration:.0f}s avg)"
            # A video stuck at near-zero views because it's literally not
            # public/available to most viewers is a THIRD explanation, on
            # top of reach and content -- and the one Claude can't guess at
            # from view count alone, so it's flagged explicitly here rather
            # than left for the model to invent a content-quality reason.
            restriction = v.get("restriction")
            if restriction:
                line += f" -- ⚠ PLATFORM RESTRICTED: {restriction}"
            return line

        if len(ranked) >= 2:
            half = max(1, len(ranked) // 2)
            lines.append("\nHigher-performing uploads (top half by views/day since posted):")
            lines.extend(_fmt(v) for v in ranked[:half])
            lines.append("\nLower-performing uploads (bottom half by views/day since posted):")
            lines.extend(_fmt(v) for v in ranked[half:])
        elif ranked:
            lines.append("\nRecent uploads (title -- views, views/day since posted, weekday posted):")
            lines.extend(_fmt(v) for v in ranked)

        if too_new:
            lines.append(
                "\nPosted too recently to judge (up less than 2 days -- their view "
                "counts mostly reflect how little time has passed, NOT how they "
                "performed). Do NOT treat these as underperforming and do not "
                "diagnose them as a reach or content problem:"
            )
            lines.extend(
                f'- "{v["title"]}" -- {v["views"]} views, up {v.get("age_days", 0):.1f} days'
                + (f' -- ⚠ PLATFORM RESTRICTED: {v["restriction"]}' if v.get("restriction") else "")
                for v in too_new
            )

    if analytics:
        lines.append(f"\nReal YouTube Analytics (last {analytics['lookback_days']} days, the channel's own authenticated data):")
        lines.append(f"Views by day of week: {analytics['views_by_day']}")
        lines.append(
            f"Average view duration: {analytics['average_view_duration_seconds']:.0f}s "
            f"({analytics['average_view_percentage']:.0f}% of video watched on average)"
        )
        lines.append(f"Traffic sources (views by source): {analytics['traffic_sources']}")
        lines.append(f"Subscribers gained: {analytics['subscribers_gained']}, lost: {analytics['subscribers_lost']}")
    else:
        lines.append("\n(No connected Analytics account -- only public view counts above, no retention/traffic data.)")

    focus_line = f"\nThe creator specifically wants advice on: {focus}\n" if focus else ""
    data_block = "\n".join(lines)

    return f"""You're a short-form YouTube strategy advisor looking at one creator's own
channel data below. Give concrete, specific advice grounded in what's
actually there -- don't give generic "post consistently" filler advice
that isn't backed by this data.
{focus_line}
{data_block}

A common frustration this creator has: a clip they personally thought was
weak takes off, while one they were proud of gets almost nothing. There
are THREE different problems that can all just look like "few views" --
check them in this order:
- PLATFORM RESTRICTED (flagged explicitly above on a video's line, if
  present): the video is age-restricted, region-blocked, unlisted/private,
  or otherwise not actually available to most viewers. This is NOT a
  content or packaging problem -- do not invent a creative explanation for
  it. Name the restriction plainly and say what to fix (privacy setting,
  appeal/re-upload, avoid whatever triggered an age restriction).
- REACH problem: low views + no retention data, or nobody watched long =
  the clip is fine, but the title/hook/thumbnail-adjacent framing didn't
  earn the click, or it wasn't discoverable (see traffic sources). Low
  views but retention is STRONG is also a reach problem, and the clearest
  kind: the content is good, it's just not getting shown/clicked enough --
  point this out explicitly if you see it, it's the single most actionable
  insight this creator can get, since it means fix the title/packaging,
  not the clip.
- CONTENT problem: views came in but retention is weak (people bailed
  early) -- the hook, pacing, or payoff itself isn't landing once someone's
  actually watching.

Do not give generic, boilerplate advice ("post more consistently",
"engage with your audience", "use eye-catching thumbnails") unless you can
tie it to a specific number or title in the data above -- if a claim
doesn't cite something concrete from this data, cut it instead of padding
the answer with it.

Based on this data, answer in exactly 4 short sections (plain text, no
markdown headers or bullet symbols, just a label then 1-2 sentences --
stay terse, this is a quick read not a report):

BEST TIME TO POST: which day(s) actually look strongest, and how
confident that is given how much data there is.

WHY SOME CLIPS WIN: name one specific higher-performing and one specific
lower-performing video (prefer a lower performer flagged PLATFORM
RESTRICTED above, if any -- that's the clearest, most useful answer this
creator can get), and diagnose the gap as platform-restricted, a reach
problem, or a content problem per the framework above (or say which if you
can't tell without retention data).

CONTENT THAT WORKS: what type of clip and title/hook should they make
more of, and what should they stop clipping?

FORMAT NOTES: one concrete format or editing change from the
retention/traffic signals (or general best practice if none given).

Hard limit: under 220 words total, and every section must be a complete
thought -- if you're running long, cut detail, not sentences."""


def get_ai_overview(
    snapshot: dict,
    analytics: Optional[dict] = None,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = _build_prompt(snapshot, analytics, focus)
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        # A ~220-word target is comfortably under 500 tokens in the normal
        # case, but this has visibly truncated in practice even at 1500 --
        # Claude doesn't reliably hit a word target exactly. This call is
        # infrequent (a button click, not per-clip), so there's no real
        # cost reason to keep the budget tight; give it generous headroom
        # instead of chasing the exact number that happens to be enough.
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
    if resp.stop_reason == "max_tokens":
        print(f"[channel_strategy] response hit max_tokens -- likely truncated ({len(text)} chars)", flush=True)
    return text
