"""Ask Claude for a plain-language content-strategy overview from a
channel's own stats -- when to post, what kind of content performs best,
and what format/pattern to lean into. Synthesizes the same heuristic/
Analytics data channel_insights.py and youtube_analytics.py already
gather; this doesn't pull any new data of its own, it just asks Claude to
read what's there and give a direct, specific take instead of leaving the
creator to eyeball a table of numbers.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


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
        ranked = sorted(videos, key=lambda v: v["views_per_day"], reverse=True)

        def _fmt(v: dict) -> str:
            weekday_name = _WEEKDAY_NAMES[v["weekday"]]
            return f'- "{v["title"]}" -- {v["views"]} views, {v["views_per_day"]}/day, posted {weekday_name}'

        if len(ranked) >= 2:
            half = max(1, len(ranked) // 2)
            lines.append("\nHigher-performing uploads (top half by views/day since posted):")
            lines.extend(_fmt(v) for v in ranked[:half])
            lines.append("\nLower-performing uploads (bottom half by views/day since posted):")
            lines.extend(_fmt(v) for v in ranked[half:])
        else:
            lines.append("\nRecent uploads (title -- views, views/day since posted, weekday posted):")
            lines.extend(_fmt(v) for v in ranked)

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
weak takes off, while one they were proud of gets almost nothing. Your job
in the WHY section below is specifically to explain that gap using
audience-facing signals -- the things that decide whether a clip gets
shown/clicked/watched at all, which usually have nothing to do with how
good the moment actually is: title/hook wording (does it work as a
stand-alone hook out of context, or does it need setup the title doesn't
give?), how discoverable it is (Shorts feed vs. search vs. suggested --
see traffic sources if given), whether the topic/subject is something
people search for or recognize, and how it performs in the first few
seconds (retention, if given). A "good" moment with a flat or vague title
can lose to a "worse" moment with a punchy one, because most viewers never
get past the first second either way.

Based on this data, answer in exactly 4 short sections (plain text, no
markdown headers or bullet symbols, just a label then 1-3 sentences):

BEST TIME TO POST: which day(s) actually look strongest here, and how
confident that read is given how much data there is -- be honest if it's
too thin to trust yet rather than overstating it.

WHY SOME CLIPS WIN: directly contrast the higher-performing uploads
against the lower-performing ones above -- name at least one specific
video from each group and explain the gap using the audience-facing
signals described above, not a vague "funny moments do well" observation
that could apply to any channel.

CONTENT THAT WORKS: given that contrast, what specific type of clip (and
specifically what kind of title/hook) should they make more of, and what
should they stop clipping or deprioritize?

FORMAT NOTES: from retention/traffic-source signals if available (or
general best practice for this kind of short-form channel if not), what
format or editing change would most likely help -- hook strength,
pacing, length, captions, etc.

Keep the whole response under 280 words total. Be direct and specific,
not hedging analyst-speak. Finish every section you start -- do not cut
a sentence off partway."""


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
        # A ~280-word target is comfortably under 800 tokens in the normal
        # case, but Claude doesn't always hit a word target exactly --
        # this got visibly cut off mid-sentence in practice at 800. Give
        # it real headroom rather than trimming the prompt's ambition to
        # fit a tight budget.
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
    if resp.stop_reason == "max_tokens":
        print(f"[channel_strategy] response hit max_tokens -- likely truncated ({len(text)} chars)", flush=True)
    return text
