"""Weekly "best clips of the week" recap for ONE GAME, across every Twitch
streamer playing it -- not just the streamers tracked in
TRENDING_TWITCH_LOGINS. E.g. "Best Watch Dogs: Legion Twitch Clips This
Week".

Discovery is the only genuinely new part (see
trending.get_top_clips_for_game); everything downstream -- the AI quality
gate, the intro/outro cards, AUDIO_ENCODE_ARGS, the concat/render pipeline,
_render_twitch_clip_for_recap -- is weekly_recap.py's / webapp/main.py's,
reused as-is (see webapp/main.py's _run_game_recap_job). This module only
holds the pieces that are genuinely specific to being a per-game recap
instead of a per-streamer one: how many clips it aims for, and its title.

Kept as its own module, job pipeline ("game_recap", not "weekly_recap"),
endpoint, and page rather than folded into the streamer-based recap's flow
-- the two are meant to stay reachable and manageable independently,
picking a game being a different creator decision than picking a week.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import weekly_recap

# A game recap casts a much wider net than the streamer recap (every
# streamer playing the game this week, not just a handful of tracked
# logins), so more candidates clear the quality gate in a typical week for
# a popular game -- 20 instead of the streamer recap's 10.
TARGET_CLIP_COUNT = 20

SERIES_TITLE_TEMPLATE = "Best {game} Twitch Clips"


def _episode_key(game_name: str) -> str:
    return game_name.strip().lower()


def next_episode_number(path: Path, game_name: str) -> int:
    """Same "persisted, best-effort counter" pattern as
    weekly_recap.next_episode_number, but keyed per game -- one shared JSON
    file holding {game_key: next_n} instead of one counter per file -- so
    "Best Watch Dogs: Legion Twitch Clips #3" and "Best Fortnite Twitch
    Clips #1" number independently instead of sharing a single counter
    neither title's number would make sense against."""
    key = _episode_key(game_name)
    counters: dict = {}
    try:
        if path.exists():
            counters = json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, TypeError):
        counters = {}
    n = int(counters.get(key, 1))
    counters[key] = n + 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(counters), encoding="utf-8")
    except OSError:
        pass  # cosmetic-only, same as weekly_recap.next_episode_number
    return n


def build_recap_metadata(
    lineup: list, week_label: str, game_name: str, episode: int,
    api_key: Optional[str] = None, model: str = weekly_recap.DEFAULT_MODEL,
    intro_offset: float = 0.0,
) -> dict:
    """Same chapter-formatted description as
    weekly_recap.build_recap_metadata (reuses its AI hook writer and
    display_name as-is), just with this feature's own numbered title --
    "Best {game} Twitch Clips #{episode}" -- instead of the streamer
    recap's fixed SERIES_TITLE, since the two are independent series with
    independent episode counts. See weekly_recap.build_recap_metadata for
    why `lineup` must already be in on-screen (countdown) order and what
    `intro_offset` is for."""
    title = f"{SERIES_TITLE_TEMPLATE.format(game=game_name)} #{episode}"[:100]

    hook = weekly_recap.generate_recap_hook(lineup, week_label, api_key, model)
    if hook:
        intro = hook
    else:
        names = [weekly_recap.display_name(u) for u in sorted(lineup, key=lambda u: u.get("view_count", 0), reverse=True)]
        seen: set = set()
        ordered_names = [n for n in names if not (n in seen or seen.add(n))]
        intro = f"This week's best {game_name} clips from {', '.join(ordered_names)} ({week_label})."

    lines = [intro, ""]
    t = intro_offset
    if intro_offset:
        lines.append(f"0:00 Intro -- {title}")
    for u in lineup:
        minutes, seconds = divmod(int(t), 60)
        lines.append(f"{minutes}:{seconds:02d} {weekly_recap.display_name(u)} -- {u.get('title', '')}")
        t += float(u.get("duration") or 0.0)
    description = "\n".join(lines)[:5000]
    return {"title": title, "description": description}
