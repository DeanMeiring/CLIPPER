"""Ask Claude (vision) to identify real facecam/webcam overlay regions
directly from sample frames, instead of relying purely on OpenCV's Haar
cascade for that call.

Confirmed in practice: the Haar-based heuristic in reframe.py both missed
genuine small/angled facecams and, when loosened enough to catch those,
started mistaking game UI textures for faces -- a blue diamond-pattern
background tile ended up rendered as a "facecam." A classical cascade
trained on frontal faces has no way to tell "real person's camera feed"
from "graphic that happens to have face-like blobs," but a vision model
can just look at the picture. This is strictly a fallback-guarded
addition: when it's unavailable (no API key) or a call fails for any
reason, callers get None back and are expected to fall back to the
Haar-based heuristic unchanged -- clip rendering should never fail or
even slow down meaningfully just because this optional check didn't work
this time.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

# Three frames (start/middle/end of the clip) is enough to tell a real,
# consistently-present overlay from a one-off visual coincidence in a
# single frame, without making the request slow or expensive.
_FRAMES = 3
_MAX_FRAME_WIDTH = 960


def _extract_frames_b64(video_path: Path, start: float, end: float) -> List[str]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    duration = max(end - start, 0.1)
    out: List[str] = []
    for i in range(_FRAMES):
        t = start + duration * (i + 0.5) / _FRAMES
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > _MAX_FRAME_WIDTH:
            scale = _MAX_FRAME_WIDTH / w
            frame = cv2.resize(frame, (_MAX_FRAME_WIDTH, int(h * scale)))
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return out


_PROMPT = """These are frames sampled across one short video clip from a livestream.

Identify every distinct FACECAM/WEBCAM overlay window showing a real person's
live video feed, composited on top of gameplay or other screen content -- the
kind of small window a streamer's camera feed appears in.

Do NOT include: game UI elements, question marks, icons, logos, text boxes,
player-name bubbles, spinners, or any other graphic that isn't an actual live
camera feed of a person -- even if it's roughly face-shaped, face-colored, or
positioned where a facecam might be. If a candidate box's content is a static
graphic or texture rather than a real person, or it doesn't appear
consistently across the frames, leave it out entirely.

Respond with ONLY a JSON array (no other text), one entry per distinct
facecam overlay found, in this exact shape:
[{"x": 0.0, "y": 0.62, "w": 0.18, "h": 0.20}]

x/y/w/h are fractions of the frame's width/height (0 to 1), covering the
visible facecam window as tightly as reasonable. Return an empty array []
if there's no real facecam overlay visible in these frames at all."""


def detect_facecams(
    video_path: Path,
    start: float,
    end: float,
    src_w: int,
    src_h: int,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Pixel-space (x, y, w, h) boxes for each real facecam/webcam overlay
    found -- or None if vision detection wasn't usable this time (no API
    key, extraction failed, the call errored, or the response didn't
    parse), in which case the caller should fall back to the Haar-based
    heuristic. An empty list (as opposed to None) means vision actually
    ran and confidently found no facecam overlay here."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    frames_b64 = _extract_frames_b64(video_path, start, end)
    if not frames_b64:
        return None

    content: List[dict] = [{"type": "text", "text": _PROMPT}]
    for b64 in frames_b64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"[facecam_vision] detection failed, falling back to heuristic: {e}", flush=True)
        return None

    if not isinstance(data, list):
        return None

    boxes: List[Tuple[int, int, int, int]] = []
    for item in data:
        try:
            x = float(item["x"]) * src_w
            y = float(item["y"]) * src_h
            w = float(item["w"]) * src_w
            h = float(item["h"]) * src_h
        except (KeyError, TypeError, ValueError):
            continue
        if w <= 1 or h <= 1:
            continue
        x = max(0.0, min(x, src_w - 1))
        y = max(0.0, min(y, src_h - 1))
        w = min(w, src_w - x)
        h = min(h, src_h - y)
        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes
