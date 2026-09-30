"""What "The Story Of" has learned: short editing lessons from YouTube's
reviews of past episodes and Dean's own notes. They go into the story and
visuals prompts (documentary.write_script / plan_visuals), so each new
episode is written and cut with them in mind -- the long-form counterpart
of the Shorts' "learns from YouTube stats" notes.

Stored in BASE_DIR/_longform/_lessons.json:
  {"lessons": [{"text", "source", "at"}], "reviews": [{"text", "source", "at"}]}
Pasted feedback is kept as-is in "reviews"; Claude distills it into the
lessons list, merged with what's already there.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List

MAX_LESSONS = 15

_FIRST_REVIEW = "YouTube's review of The Story of StableRonaldo (Sept 2026)"

# From YouTube's editing review of the first episode: the opening laughter
# ran long, a moment opened on a static desktop, a loud clip cut straight
# into serious news, and the end card sat still for 11 seconds.
SEED_LESSONS = [
    "Open fast: the cold-open moment is 5 to 10 seconds and ends right after the key line or reaction, so the narration starts by about second 11.",
    "Cut every moment tight: start it within a second of when something is said or happens, and end it within about 1.5 seconds of the last word or reaction. No silent setup, no long trailing laughter.",
    "If a clip opens on a static screen, a menu or silence, start the moment later, where the action begins.",
    "When a loud or chaotic moment is followed by serious news, put a short calm narration line or a chapter card between them rather than cutting straight from one to the other.",
    "End on the question to viewers; say nothing after it.",
]


class LessonStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data.get("lessons"), list):
                return data
        except (OSError, ValueError):
            pass
        now = time.time()
        return {"lessons": [{"text": t, "source": _FIRST_REVIEW, "at": now} for t in SEED_LESSONS], "reviews": []}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)

    def texts(self) -> List[str]:
        return [x["text"] for x in self.load()["lessons"]]

    def remove(self, index: int) -> dict:
        with self.lock:
            data = self.load()
            if not 0 <= index < len(data["lessons"]):
                raise IndexError(index)
            data["lessons"].pop(index)
            self._save(data)
            return data

    def learn(self, feedback: str, source: str) -> dict:
        """Distill new feedback into the lessons (Claude), keeping the
        feedback itself too."""
        feedback = feedback.strip()[:12000]
        current = self.load()
        merged = distill([x["text"] for x in current["lessons"]], feedback)
        with self.lock:
            data = self.load()
            old = {x["text"]: x for x in data["lessons"]}
            now = time.time()
            data["lessons"] = [old.get(t) or {"text": t, "source": source, "at": now} for t in merged]
            data["reviews"] = (data.get("reviews") or []) + [{"text": feedback, "source": source, "at": now}]
            self._save(data)
            return data


def distill(current: List[str], feedback: str) -> List[str]:
    from .select_moments import DEFAULT_MODEL, _ask_claude_for_json

    listed = "\n".join(f"- {t}" for t in current) or "(none yet)"
    prompt = f"""You keep a short list of editing lessons for a YouTube documentary series
about Twitch streamers ("The Story Of"). Each episode is written by a model
from research and the streamer's clips: narrated scenes, clip "moments" that
play with their own sound, chapter cards and on-screen visuals. The lessons
are given to that model every time it writes and cuts a new episode.

The lessons so far:
{listed}

New feedback on an episode (a YouTube review or the creator's own notes):
\"\"\"
{feedback}
\"\"\"

Update the list:
- Add what this feedback teaches, as instructions the writer can act on
  when choosing the story, the clip moments (where they start and end),
  pacing, visuals and the ending. Say what to do, briefly.
- Merge anything that says the same thing; drop a lesson only if the new
  feedback clearly replaces it.
- Leave out timestamps and anything specific to only that one video, and
  anything the writer can't control.
- At most {MAX_LESSONS} lessons, each under 40 words. Plain words.

Respond with ONLY a JSON array of strings: ["...", "..."]
"""
    data = _ask_claude_for_json(prompt, None, DEFAULT_MODEL)
    out: List[str] = []
    for t in data if isinstance(data, list) else []:
        t = " ".join(str(t).split())
        if t and len(t) <= 400 and t not in out:
            out.append(t)
    if not out:
        raise RuntimeError("Claude didn't return any lessons -- try again.")
    return out[:MAX_LESSONS]


def prompt_block(lessons: List[str]) -> str:
    """The lessons as a section of a prompt ("" when there are none)."""
    if not lessons:
        return ""
    return ("LESSONS FROM EARLIER EPISODES (from YouTube's reviews and the creator's notes -- follow them):\n"
            + "\n".join(f"- {t}" for t in lessons) + "\n\n")
