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


def load_competitors(path: Path) -> list:
    """The creator's saved list of competitor channels ({channel_id,
    channel_title}) to compare against in the AI overview -- picked via
    competitor_discovery's streamer search, not hand-typed, since a
    creator clipping someone else's stream often has no idea who else is
    clipping the same person. Empty list if nothing's been saved yet."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_competitors(path: Path, channels: list) -> None:
    """Overwrite the saved competitor list -- the frontend always sends the
    full desired list (added/removed client-side first), so a plain
    replace is simpler and race-free compared to incremental add/remove
    calls against the same file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(channels), encoding="utf-8")
    except OSError:
        pass


def _format_competitor_block(competitors: list) -> str:
    """One labeled section per competitor channel, each listing its
    highest-performing recent titles the same way the creator's own
    top-half is shown -- so Claude is comparing two concrete lists of
    titles against each other, not summarizing a vague impression of
    "what other clip channels do"."""
    blocks = []
    for c in competitors:
        title = c.get("channel_title") or "unknown channel"
        videos = c.get("recent_videos") or []
        settled = [v for v in videos if not v.get("too_new_to_judge")]
        ranked = sorted(settled, key=lambda v: v["views_per_day"], reverse=True)[:6]
        if not ranked:
            continue
        lines = [f'-- {title} --']
        for v in ranked:
            weekday_name = _WEEKDAY_NAMES[v["weekday"]]
            lines.append(f'"{v["title"]}" -- {v["views"]} views, {v["views_per_day"]}/day, posted {weekday_name}')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_content_analysis_block(content_analyses: list) -> str:
    """One block per analyzed competitor video: its actual transcript and
    where it gets loud, not just its title -- this is what lets Claude
    describe WHAT KIND of moment won (a clutch, a fail, a funny exchange)
    instead of only noticing a title pattern. Each entry is
    {channel_title, video_title, transcript_text, loud_moments}."""
    blocks = []
    for a in content_analyses:
        header = f'-- {a.get("channel_title", "unknown channel")}: "{a.get("video_title", "")}" --'
        parts = [header]
        transcript = (a.get("transcript_text") or "").strip()
        if transcript:
            # A Shorts-length transcript is naturally small (well under a
            # a minute of speech); this cap is a backstop against an
            # unusually dense/long one, not something normal clips hit.
            parts.append(f"Transcript: {transcript[:1500]}")
        loud = a.get("loud_moments") or []
        if loud:
            spots = ", ".join(
                f'{m["start"]:.0f}-{m["end"]:.0f}s (peak {m["peak_db"]:.0f}dB, +{m["jump_db"]:.0f}dB over baseline)'
                for m in loud[:5]
            )
            parts.append(f"Loud/high-energy moments: {spots}")
        if len(parts) > 1:
            blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _build_prompt(
    snapshot: dict,
    analytics: Optional[dict],
    focus: Optional[str],
    competitors: Optional[list] = None,
    content_analyses: Optional[list] = None,
) -> str:
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

    competitor_block = _format_competitor_block(competitors) if competitors else ""
    content_block = _format_content_analysis_block(content_analyses) if content_analyses else ""

    focus_line = f"\nThe creator specifically wants advice on: {focus}\n" if focus else ""
    data_block = "\n".join(lines)
    competitor_section = (
        f"\n\nOther channels clipping similar streamers/content, for comparison -- NOT this "
        f"creator's own channel, do not mix these into the above analysis, only use them for "
        f"the COMPETITOR PATTERNS section below:\n\n{competitor_block}"
        if competitor_block else ""
    )
    content_section = (
        f"\n\nActual content of some of those competitor videos -- their real transcript "
        f"and where the audio gets loud, not just the title -- for the WHAT KIND OF MOMENT "
        f"WINS section below:\n\n{content_block}"
        if content_block else ""
    )
    competitor_patterns_section = (
        "\nCOMPETITOR PATTERNS: compare this creator's own top-performing titles above\n"
        "against the competitor channels' top-performing titles. Name one concrete\n"
        "title/hook/wording pattern (e.g. a phrasing style, emoji use, question\n"
        "hooks, ALL CAPS words, naming the streamer in the title, numbers) that\n"
        "shows up repeatedly in the competitors' best performers but not in this\n"
        "creator's own -- something they could actually try, not a vague \"be more\n"
        "engaging\". If nothing clearly differs, say that plainly instead of\n"
        "inventing a pattern.\n"
        if competitor_block else ""
    )
    content_patterns_section = (
        "\nWHAT KIND OF MOMENT WINS: from the actual transcript(s) and loud-moment\n"
        "timestamps above, describe the TYPE of moment that's winning for competitors --\n"
        "a clutch/comeback, a fail, a funny exchange, a shocked reaction, an argument,\n"
        "a jumpscare -- and whether it lines up with a loud/high-energy spot or is quiet\n"
        "(a punchline, a deadpan line). Say what kind of moment this creator should be\n"
        "watching their own footage for, specifically. Don't just restate the transcript.\n"
        if content_block else ""
    )

    return f"""You're a short-form YouTube strategy advisor looking at one creator's own
channel data below. Give concrete, specific advice grounded in what's
actually there -- don't give generic "post consistently" filler advice
that isn't backed by this data.
{focus_line}
{data_block}{competitor_section}{content_section}

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

Based on this data, answer in exactly {4 + bool(competitor_block) + bool(content_block)} short sections (plain text, no
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
{competitor_patterns_section}{content_patterns_section}
Hard limit: under {220 + (40 if competitor_block else 0) + (40 if content_block else 0)} words total, and every section must be a complete
thought -- if you're running long, cut detail, not sentences."""


def _build_recommend_prompt(candidates: list, notes: Optional[str]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        age_days = max(0.0, (time.time() - c["_published_ts"]) / 86400) if c.get("_published_ts") else None
        age = f"{age_days:.1f}d old" if age_days is not None else "age unknown"
        views = f'{c.get("view_count")} views' if c.get("view_count") is not None else "views unknown"
        duration = c.get("duration") or "duration unknown"
        lines.append(f'[{i}] {c.get("name", "unknown streamer")} -- "{c.get("title", "")}" -- {views}, {duration}, {age}')
    candidates_block = "\n".join(lines)

    notes_block = f"\n\nWhat's worked on this creator's own channel so far (their saved AI overview):\n{notes}\n" if notes else ""

    return f"""You're helping a YouTube Shorts clipper decide which Twitch VOD, out of a
short list they're already tracking, is worth downloading and clipping
today. Each candidate is a VOD from a streamer they clip regularly --
this is not a discovery task, just triage of a small list.

Candidates:
{candidates_block}
{notes_block}
Pick the ONE candidate most worth clipping right now. View count is the
strongest signal you have (a VOD already earning more views than that
streamer's other recent ones had a moment worth watching), but also weigh
duration (a 6+ hour VOD has more chances at a highlight than a 45-minute
one) and, if the saved overview above names a type of moment or streamer
pattern that's worked before, factor that in too. Don't just always pick
the single highest view count without reasoning -- say why.

Answer in exactly this format, nothing else:
PICK: <index number>
WHY: <one or two sentences, grounded in this candidate's actual numbers above -- no generic filler>
RUNNER_UP: <index number, or NONE if there's only one reasonable option>
RUNNER_UP_WHY: <one sentence, or omit this line if RUNNER_UP is NONE>"""


def recommend_vod(
    candidates: list,
    notes: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Ask Claude to pick which of a short list of tracked streamers' recent
    VODs is most worth downloading and clipping today. `candidates` is a
    list of dicts (from trending.CreatorEntry, each needs a "_published_ts"
    float added by the caller for age calculation). Returns
    {"pick_index", "why", "runner_up_index", "runner_up_why"} -- indices
    are None if Claude's response couldn't be parsed or it named an
    out-of-range index, so the caller can fall back to the highest view
    count instead of trusting a bad parse."""
    import re

    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if not candidates:
        raise ValueError("no candidates to recommend from")

    prompt = _build_recommend_prompt(candidates, notes)
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()

    def _index(pattern: str) -> Optional[int]:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return None
        try:
            idx = int(m.group(1))
        except ValueError:
            return None
        return idx if 0 <= idx < len(candidates) else None

    def _text_after(label: str) -> Optional[str]:
        m = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return {
        "pick_index": _index(r"PICK:\s*(\d+)"),
        "why": _text_after("WHY"),
        "runner_up_index": _index(r"RUNNER_UP:\s*(\d+)"),
        "runner_up_why": _text_after("RUNNER_UP_WHY"),
        "raw": text,
    }


def get_ai_overview(
    snapshot: dict,
    analytics: Optional[dict] = None,
    focus: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    competitors: Optional[list] = None,
    content_analyses: Optional[list] = None,
) -> str:
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    prompt = _build_prompt(snapshot, analytics, focus, competitors, content_analyses)
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
