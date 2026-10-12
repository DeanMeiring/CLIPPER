"""Claude looks at a finished "merge" video before it goes out.

Four frames (the start, two along the way, the ending with the final form
big on screen) go to Claude with the pictures' names. It answers whether
the ending works and nothing looks broken (``looks``), and whether the
pictures are fine to post (``pictures``). The web app decides what to do
with a "no": make it again, or swap the pictures.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from . import ball_packs
from .ball_evolution import _ffmpeg

PROBLEMS = ("none", "looks", "pictures")

KINDS = {
    "merge": 'a "merge into final form" video: pictures drop into a ring, two of the same merge '
             'into the next one, and the last one is the final form',
    "escape": "a ring video: balls with pictures bounce in a spinning ring; each one that gets out "
              "through the gap makes two more inside, and new ones come out as the next picture, "
              "up to the final one",
    "touch": "a ring video: balls with pictures bounce in a closed ring; two that touch make a new "
             "one, and new ones come out as the next picture, up to the final one",
    "breakin": "a brick-breaker video: a ball with a picture bounces in a ring at the bottom, gets "
               "through a corridor into a room of bricks, and every broken brick makes another "
               "ball; new ones come out as the next picture, up to the final one",
    "evolve": "a ball-drop video: pictures fall through pegs into a jar, two of the same merge "
              "into the next one, up to the final one",
    "center": "a 'hold the center' video: the pictures bounce in a ring around a growing black hole; "
              "one that falls in is out (crossed out at the top), and the last one left is the winner "
              "shown big at the end",
    "spikes": "a 'spikes will grow' video: one ball with a picture bounces in an octagon, a spike grows "
              "where it hits a wall, and the ball changes into the next picture every few bounces, up "
              "to the final one",
    "shrink": "a 'the ball grows with every bounce' video: one ball with a picture gets bigger on every "
              "bounce, leaving a rainbow trail that stays, changing pictures on the way, until it fills the ring",
    "jaws": "a '4 seconds to escape' video: balls with pictures bounce in a turning broken ring, each has "
            "4 seconds (the arc around it) to get out or it pops, and every escape makes two more; new "
            "ones come out as the next picture, up to the final one",
    "spiral": "a 'will the ball multiply?' video: balls with pictures break a spiral wall piece by piece, "
              "every broken piece makes another ball (the counter is the multiplier), and new balls come "
              "out as the next picture, up to the final one",
    "rays": "a 'rays in a parabola' video: rounds of rainbow lines fall into a parabola and all bounce "
            "through its focus, where each round's picture appears; the last round shows the final one",
    "box": "a '+3 balls after each escape' video: balls with pictures fall out of a hole in a box's floor "
           "and each one adds three more at the top; new ones come out as the next picture",
    "zen": "a 'reminds me of something' video: a ring of spikes with one gap; each picture drops in, and "
           "one that touches the spikes turns grey and stays there; the last picture gets out the gap",
    "trail": "a 'with every bounce, the ball evolves' video: one ball with a picture grows on every bounce "
             "and leaves a permanent rainbow trail that fills the ring; it changes pictures up to the final one",
}

_PROMPT = """These are 4 frames from a vertical YouTube Short, {kind}. \
Its title is: {title!r}. The pictures, in order (the last is the \
final form): {names}.

Frame 1 is the start, frames 2 and 3 are along the way, frame 4 is the ending (the final \
form big in the middle with the ending text).

Check, as the channel's last look before it's posted:
1. The ending works: the final form is clearly visible, recognisable and a fun payoff \
after the earlier pictures.
2. Nothing looks broken: text cut off or overlapping, blank or black boxes, stretched or \
glitched pictures, a leftover background around a sticker that looks ugly.
3. The pictures are fine to post: no nudity, gore, real people's faces, hateful or \
offensive text or symbols.

The ending text and the big final form are drawn over the pile of pictures on purpose, and pictures overlap while they bounce: that's normal. Small taste issues are fine; only say no when a viewer would notice something wrong.
Answer with JSON only:
{{"ok": true or false, "problem": "none" or "looks" (point 1 or 2) or "pictures" \
(point 3), "notes": "one short plain sentence on what you saw"}}"""


def frame_times(info: dict) -> List[float]:
    secs = float(info.get("seconds") or 30)
    end = info.get("done_at")
    end = float(end) + 1.2 if end else secs - 0.5
    return [min(1.5, secs / 4), secs * 0.4, secs * 0.7, min(end, secs - 0.2)]


def grab_frames(video: Path, times: List[float], width: int = 540) -> List[bytes]:
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, t in enumerate(times):
            jpg = Path(tmp) / f"{k}.jpg"
            subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}",
                            "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2",
                            "-q:v", "4", str(jpg)], check=True, timeout=60)
            out.append(jpg.read_bytes())
    return out


def review(video: Path, info: dict, title: str, api_key: str,
           log: Optional[Callable[[str], None]] = None) -> dict:
    """{"ok", "problem", "notes"}. Raises when Claude or ffmpeg fails (the
    caller lets the video through then: a failed check never blocks it)."""
    pack = info.get("pack") or {}
    names = [i["name"] for i in pack.get("items") or []] or list((info.get("unlocks") or {}).keys())
    frames = grab_frames(Path(video), frame_times(info))
    content = []
    for k, jpg in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {k}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(jpg).decode()}})
    content.append({"type": "text", "text": _PROMPT.format(kind=KINDS.get(info.get("format"), KINDS["merge"]), title=title or "",
                                                         names=" > ".join(names))})
    raw = ball_packs._claude_text(content, api_key, max_tokens=4000, log=log)
    return parse(raw)


def parse(raw: str) -> dict:
    """Claude's answer, read leniently; anything unreadable raises."""
    ok = re.search(r'"ok"\s*:\s*(true|false)', raw or "", re.I)
    if not ok:
        raise ValueError(f"couldn't read the check: {(raw or '')[:200]!r}")
    problem = re.search(r'"problem"\s*:\s*"(\w+)"', raw)
    notes = re.search(r'"notes"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    good = ok.group(1).lower() == "true"
    p = (problem.group(1).lower() if problem else "none")
    if p not in PROBLEMS:
        p = "looks"
    if good:
        p = "none"
    elif p == "none":
        p = "looks"
    return {"ok": good, "problem": p, "notes": (notes.group(1) if notes else "").replace('\\"', '"')[:300]}


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
