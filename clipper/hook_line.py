"""Generate a spoiler-style "flash hook" line for an ALREADY-RENDERED clip --
a single line of text shown dead-center for well under a second right as
the clip starts, then gone, giving away the payoff before it happens on
screen (e.g. "MARLON ALMOST KNOCKS OUT JASON"). The curiosity-gap trick
common on streamer-clip channels.

Deliberately kept separate from the main clip-selection/render pipeline
(clipper/select_moments.py also writes a milder `hook_caption` per clip, which
the main render boxes at the top of the clip's first 3 seconds unless the job
turned hook text off -- see captions.build_ass) -- see webapp/main.py's
/hook-line page, which lets the user pick an existing
finished clip, generate/edit this line, and render it as its own pass via
clipper.render.overlay_hook_line. Nothing here touches the original render.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from .transcribe import Word

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

# Long enough to register as a real line of text, short enough that it's
# gone before the clip's own captions would start competing with it.
FLASH_SECONDS = 0.8


def generate_hook_line(
    clip_words: List[Word], clip_title: str,
    api_key: Optional[str] = None, model: Optional[str] = None,
) -> str:
    """Ask Claude for ONE spoiler-style hook line naming who's involved and
    the SPECIFIC outcome of this clip -- not a vague topic, since the whole
    trick only works if the line actually gives away the payoff."""
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to generate a hook line.")

    transcript = " ".join(w.text for w in clip_words).strip()
    if not transcript:
        raise RuntimeError("This clip has no transcript to write a hook line from.")

    prompt = f"""You write flash-hook captions for streamer clip channels -- a single
line of text shown on screen for well under a second right as a clip starts,
then gone, before the actual moment plays out. The whole trick is spoiling
the payoff up front so people keep watching to see it happen -- e.g.
"MARLON ALMOST KNOCKS OUT JASON" or "STREAMER RAGE QUITS MID-BOSS FIGHT".

Clip title (internal label, not shown on screen): {clip_title}

Transcript of this clip:
{transcript}

Write ONE hook line for this specific clip:
- name the actual people/streamer involved if identifiable from the transcript or title, and the SPECIFIC thing that happens in this clip -- not a vague topic
- under 10 words
- ALL CAPS
- no hashtags, no emoji, no trailing punctuation

Respond with ONLY the hook line, nothing else."""

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model or DEFAULT_MODEL, max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    raw = raw.strip().strip('"').strip()
    if not raw:
        raise RuntimeError("Claude returned an empty hook line.")
    return raw


def clip_dimensions(video_path: Path) -> Tuple[int, int]:
    """The rendered clip's actual (width, height), so the flash-hook .ass
    is built at the right PlayRes -- falls back to the normal portrait
    Short's 1080x1920 if the file can't be probed (still renders fine,
    just not perfectly scaled on an unusual output size)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (1080, 1920)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return (w, h) if w > 0 and h > 0 else (1080, 1920)
    except Exception:
        return (1080, 1920)
