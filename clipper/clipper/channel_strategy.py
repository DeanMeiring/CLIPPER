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


def _format_long_form_block(snapshot: dict) -> str:
    """Recent long-form (non-Shorts) uploads -- e.g. the weekly cross-
    streamer recap -- reported the same way Shorts are (two ranked
    groups plus retention where available) but kept as their own
    section rather than mixed into the Shorts comparison above, since
    the two formats compete for completely different things (a Short's
    whole draw is its first few seconds; a long-form video's is holding
    a session and earning the click via its thumbnail in the first
    place). See channel_insights.get_channel_snapshot for how the split
    is made."""
    videos = snapshot.get("recent_long_form_videos") or []
    if not videos:
        return ""
    too_new = [v for v in videos if v.get("too_new_to_judge")]
    settled = [v for v in videos if not v.get("too_new_to_judge")]
    ranked = sorted(settled, key=lambda v: v["views_per_day"], reverse=True)

    def _fmt(v: dict) -> str:
        weekday_name = _WEEKDAY_NAMES[v["weekday"]]
        minutes = (v.get("duration_seconds") or 0) / 60
        line = f'- "{v["title"]}" -- {minutes:.0f}min, {v["views"]} views, {v["views_per_day"]}/day, posted {weekday_name}'
        duration = v.get("average_view_duration_seconds")
        pct = v.get("average_view_percentage")
        if duration is not None and pct is not None:
            line += f", retention: {pct:.0f}% watched ({duration / 60:.1f}min avg)"
        restriction = v.get("restriction")
        if restriction:
            line += f" -- ⚠ PLATFORM RESTRICTED: {restriction}"
        return line

    lines = []
    if len(ranked) >= 2:
        half = max(1, len(ranked) // 2)
        lines.append("Higher-performing long-form uploads (top half by views/day since posted):")
        lines.extend(_fmt(v) for v in ranked[:half])
        lines.append("\nLower-performing long-form uploads (bottom half by views/day since posted):")
        lines.extend(_fmt(v) for v in ranked[half:])
    elif ranked:
        lines.append("Recent long-form uploads (title -- length, views, views/day since posted, weekday posted):")
        lines.extend(_fmt(v) for v in ranked)
    if too_new:
        lines.append("\nPosted too recently to judge (do not diagnose these as underperforming):")
        lines.extend(f'- "{v["title"]}" -- {v["views"]} views, up {v.get("age_days", 0):.1f} days' for v in too_new)
    return "\n".join(lines)


def _select_thumbnail_images(snapshot: dict, max_images: int = 4) -> list:
    """The top- and bottom-performing long-form videos with a usable
    thumbnail URL -- few enough to keep the vision call cheap and fast,
    but a real top-vs-bottom pair so Claude can compare a thumbnail that
    worked against one that didn't instead of judging one in isolation.
    Empty (no images, no vision call) whenever there's nothing settled
    enough to rank yet."""
    videos = snapshot.get("recent_long_form_videos") or []
    settled = [v for v in videos if not v.get("too_new_to_judge") and v.get("thumbnail_url")]
    ranked = sorted(settled, key=lambda v: v["views_per_day"], reverse=True)
    if not ranked:
        return []
    half = max(1, max_images // 2)
    chosen = ranked[:half] + ranked[-half:]
    seen_ids: set = set()
    result = []
    for v in chosen:
        if v["id"] in seen_ids:
            continue
        seen_ids.add(v["id"])
        result.append(v)
    return result[:max_images]


def _build_prompt(
    snapshot: dict,
    analytics: Optional[dict],
    focus: Optional[str],
    competitors: Optional[list] = None,
    content_analyses: Optional[list] = None,
    thumbnail_videos: Optional[list] = None,
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

    long_form_block = _format_long_form_block(snapshot)
    if long_form_block:
        lines.append("\n-- Long-form uploads (NOT Shorts -- a separate format, judge separately) --")
        lines.append(long_form_block)

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

    thumbnail_videos = thumbnail_videos or []
    thumbnails_note = ""
    if thumbnail_videos:
        labels = ", ".join(f'"{v["title"]}"' for v in thumbnail_videos)
        thumbnails_note = (
            f"\n\n{len(thumbnail_videos)} actual thumbnail image(s) are attached after this "
            f"message (each preceded by a text label naming which video it's for): {labels}. "
            f"Actually look at them for the LONG-FORM & THUMBNAILS section below -- don't guess "
            f"at what they look like from the titles alone."
        )
    long_form_section = (
        "\nLONG-FORM & THUMBNAILS: how are the long-form uploads doing relative to EACH\n"
        "OTHER (not vs. Shorts -- different format, not a fair comparison)? Use the same\n"
        "platform-restricted / reach / content framework above. If thumbnail images are\n"
        "attached, actually compare the higher- and lower-performing ones: is there a real,\n"
        "visible difference (a clear focal face/expression vs. a busy or dark frame, readable\n"
        "text vs. none or too much, a strong contrast/color vs. blending into YouTube's own\n"
        "UI)? Name the specific difference you see, not a generic \"make it more eye-catching\" --\n"
        "if the thumbnails genuinely look about the same, say so instead of inventing a\n"
        "difference.\n"
        if (long_form_block or thumbnail_videos) else ""
    )

    return f"""You're a short-form YouTube strategy advisor looking at one creator's own
channel data below. Give concrete, specific advice grounded in what's
actually there -- don't give generic "post consistently" filler advice
that isn't backed by this data.
{focus_line}
{data_block}{competitor_section}{content_section}{thumbnails_note}

SHORTS RETENTION GROUND TRUTH (use these as the actual bar when judging a
retention number above, not vague intuition -- these are the real signals
that separate a Short that holds its audience from one that doesn't):
- The hook is won or lost in the first 1-3 seconds -- roughly half of all
  drop-off happens there. The open needs to already BE the moment (or
  promise one immediately), never a slow lead-in or setup before anything
  happens.
- A visual change (a cut, zoom, camera angle, or text-overlay swap) roughly
  every 1.5-4 seconds keeps attention; long static stretches with nothing
  changing on screen are where retention bleeds out mid-clip.
- Length vs. retention is a real tradeoff, not just "shorter is better":
  15-30s clips typically hold 70-90%+ retention on a strong moment; 30-60s
  needs tighter pacing to hold up; past ~90s retention drops off hard
  unless the moment is genuinely exceptional. A weak retention number on a
  clip pushing toward this app's 180s cap is itself a signal the clip may
  simply be too long for its own content, not just weakly edited.
- Burned-in captions measurably help retention (many viewers watch muted) --
  worth flagging as a fix if a low-retention clip doesn't already have them.
- As a rough retention bar for THIS content type: under ~40% average
  view percentage is a real content problem worth digging into, 40-60% is
  middling, 60%+ is strong and shouldn't be reframed as a problem just
  because it's not higher.

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
  early) -- diagnose WHICH lever from the ground truth above is most
  likely the cause (a slow/no hook in the first few seconds, too little
  visual change to hold attention, the clip simply running long for what
  the moment can sustain, or missing captions) rather than a generic "the
  hook, pacing, or payoff isn't landing."

Do not give generic, boilerplate advice ("post more consistently",
"engage with your audience", "use eye-catching thumbnails") unless you can
tie it to a specific number or title in the data above -- if a claim
doesn't cite something concrete from this data, cut it instead of padding
the answer with it.

Based on this data, answer in exactly {4 + bool(competitor_block) + bool(content_block) + bool(long_form_section)} short sections (plain text, no
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

FORMAT NOTES: one concrete format or editing change grounded in the
retention/traffic signals and the ground truth above (e.g. tighten the
hook, cut clip length down, add captions, increase cut frequency) --
whichever one the actual numbers point to, not a generic tip.
{competitor_patterns_section}{content_patterns_section}{long_form_section}
Hard limit: under {220 + (40 if competitor_block else 0) + (40 if content_block else 0) + (40 if long_form_section else 0)} words total, and every section must be a complete
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
Rank ALL {len(candidates)} candidates from most to least worth clipping today --
list every index exactly once. View count is the strongest signal you have
(a VOD already earning more views than that streamer's other recent ones
had a moment worth watching), but also weigh duration (a 6+ hour VOD has
more chances at a highlight than a 45-minute one) and, if the saved
overview above names a type of moment or streamer pattern that's worked
before, factor that in too. Don't just always rank by view count alone --
say why the top two are ranked where they are.

Answer in exactly this format, nothing else:
RANKING: <comma-separated index numbers, best to worst, every index 0-{len(candidates) - 1} exactly once>
WHY_BEST: <one or two sentences on your #1 pick, grounded in its actual numbers above -- no generic filler>
WHY_SECOND: <one sentence on your #2 pick>"""


def recommend_vod(
    candidates: list,
    notes: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Ask Claude to rank a short list of tracked streamers' recent VODs by
    how worth downloading and clipping each is today. `candidates` is a
    list of dicts (from trending.CreatorEntry, each needs a "_published_ts"
    float added by the caller for age calculation). Returns
    {"ranking": [index, ...], "why_best", "why_second"} -- `ranking` always
    contains every valid index exactly once (any index Claude's response
    didn't parse or omitted is appended in its original order), so a caller
    that needs to skip an unreachable VOD always has a next-best option to
    fall back to instead of surfacing nothing."""
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

    def _text_after(label: str) -> Optional[str]:
        m = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    ranking: list = []
    seen: set = set()
    raw_ranking = _text_after("RANKING") or ""
    for part in raw_ranking.split(","):
        part = part.strip()
        if not part.isdigit():
            continue
        idx = int(part)
        if 0 <= idx < len(candidates) and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    # Anything Claude's ranking missed or garbled still needs a place in
    # line -- append the leftovers (by view count, since that's the
    # fallback signal used when nothing parses at all) rather than losing
    # them, so a skipped-for-being-inaccessible top pick always has
    # somewhere to fall back to.
    leftovers = sorted(
        (i for i in range(len(candidates)) if i not in seen),
        key=lambda i: candidates[i].get("view_count") or 0,
        reverse=True,
    )
    ranking.extend(leftovers)

    return {
        "ranking": ranking,
        "why_best": _text_after("WHY_BEST"),
        "why_second": _text_after("WHY_SECOND"),
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

    # A top-vs-bottom pair of long-form thumbnails, if there's enough
    # settled data to rank them -- passed to Claude as actual images (via
    # the API's url-source image blocks, no download/base64 needed on our
    # side) so the LONG-FORM & THUMBNAILS section can compare what a
    # thumbnail actually looks like, not just its video's title.
    thumbnail_videos = _select_thumbnail_images(snapshot)
    prompt = _build_prompt(snapshot, analytics, focus, competitors, content_analyses, thumbnail_videos)

    content: list = [{"type": "text", "text": prompt}]
    for v in thumbnail_videos:
        content.append({"type": "text", "text": f'Thumbnail for "{v["title"]}" ({v["views_per_day"]}/day):'})
        content.append({"type": "image", "source": {"type": "url", "url": v["thumbnail_url"]}})

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
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()
    if resp.stop_reason == "max_tokens":
        print(f"[channel_strategy] response hit max_tokens -- likely truncated ({len(text)} chars)", flush=True)
    return text
