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
import re
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_MODEL = os.environ.get("CLIPPER_MODEL", "claude-sonnet-4-5")

# Three frames (start/middle/end of the clip) is enough to tell a real,
# consistently-present overlay from a one-off visual coincidence in a
# single frame, without making the request slow or expensive.
_FRAMES = 3
_MAX_FRAME_WIDTH = 960

# Whole-word only (\b...\b) -- a plain substring check on these would
# false-positive on real descriptions ("stat" inside "stationary", "ui"
# inside "quiet", "text" inside "textured/context"). youtube/webpage/
# thumbnail/browser catch the "face inside a video being watched" failure
# mode (a paused YouTube video with a real face in it, mistaken for a
# facecam) -- low collision risk, since none of these would plausibly
# appear in a genuine "person's own camera feed" description.
_RED_FLAG_RE = re.compile(
    r"\b(card|graphic|icon|logo|score|stat|stats|report|level|ui|text|screenshot|screen shot"
    r"|youtube|webpage|website|browser|thumbnail)\b"
)


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

A real facecam overlay is small relative to the whole frame -- roughly a
third of the frame's width/height at most, usually less, and positioned in a
corner or along one edge. If a face or video takes up most or all of the
frame, it is NOT a facecam overlay, no matter how real the person in it
looks -- it's either the main content itself, or a video/webpage/photo the
streamer is watching or browsing (e.g. a YouTube video, a movie clip, a
paused video with playback controls visible, a thumbnail, someone's profile
picture). A real person's face appearing INSIDE content being watched or
displayed on screen is not the streamer's own camera feed -- leave it out
even though it's a real face.

Do NOT include, even if it's roughly face-shaped, face-colored, or positioned
where a facecam might be:
- game UI elements, icons, logos, question marks, spinners, player-name
  bubbles
- score cards, stat trackers, results/report screens ("LEVEL 12", "TIME
  3:36", grade letters, etc.), leaderboards, level-complete screens
- a face or video that's part of content being watched/browsed on screen
  (a YouTube/video player, a website, a photo, a video thumbnail) rather
  than a small overlay window composited on top of everything else
- any other static graphic or texture that isn't an actual live camera feed
  of a person

If you're not confident a candidate box is a real person's live camera feed
specifically -- a small, corner-positioned overlay window, not the main
content on screen -- leave it out. A missed facecam is a much smaller
problem than treating the wrong thing as someone's face.

Each real person should appear as exactly ONE box, even if their camera feed
is highlighted, spotlighted, or duplicated elsewhere on screen (e.g. an
"active speaker" indicator showing the same person again) -- pick whichever
single box best represents their main camera window and skip the rest. Never
return two overlapping or near-identical boxes for the same person.

Respond with ONLY a JSON array (no other text), one entry per distinct
facecam overlay found, in this exact shape:
[{"x": 0.0, "y": 0.62, "w": 0.18, "h": 0.20, "what": "person wearing headphones, real camera feed"}]

x/y/w/h are fractions of the frame's width/height (0 to 1), covering the
visible facecam window as tightly as reasonable. "what" is a short (under 10
words) description of what's actually in the box -- answering it forces you
to look closely before committing to a box, and it must describe an actual
person's live video, not a graphic. Return an empty array [] if there's no
real facecam overlay visible in these frames at all."""


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedupe_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Drop boxes that substantially overlap an already-kept one, keeping
    the larger of the two. A safety net for when the model returns two
    boxes for what's really the same person's camera feed (seen in
    practice on a source where multiple closely-packed webcam tiles made
    the boundary between them ambiguous) -- despite the prompt asking for
    exactly one box per person, this doesn't rely on the model getting
    that right every time."""
    kept: List[Tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if not any(_iou(box, k) >= iou_threshold for k in kept):
            kept.append(box)
    return kept


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
            # A 4-5 person co-stream needs a box + "what" description per
            # person -- enough JSON that 500 tokens could clip it mid-
            # response (seen in logs as a bare JSON parse failure), which
            # discards every detected box and falls back to the much less
            # reliable Haar heuristic for the whole clip. More headroom
            # only costs anything if it's actually used.
            max_tokens=1024,
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
        # Hard geometric backstop, independent of anything the model says:
        # a real facecam overlay is a small window in a corner, never
        # close to the whole frame. Confirmed in practice: a paused
        # YouTube video filling almost the entire frame (playback
        # controls, like/share buttons visible) got flagged as a
        # "facecam" because there was a real human face in the video's
        # own thumbnail/content -- the model correctly saw a real face,
        # just not correctly reasoning that it belonged to something
        # being watched, not the streamer's own camera. No legitimate
        # corner overlay is anywhere near this large, so reject by
        # geometry rather than depending on the model's own judgment call
        # every time.
        if h > 0.4 * src_h or w > 0.6 * src_w:
            print(f"[facecam_vision] rejected box ({int(w)}x{int(h)} of {src_w}x{src_h}) -- too large to be a corner overlay", flush=True)
            continue
        what = str(item.get("what", "")).strip()
        # Defensive backstop, not the primary defense (that's the prompt
        # itself and the model's own "what" reasoning) -- catches the
        # model flagging something as a game-UI/graphic in its own
        # description while still handing back a box for it. Whole-word
        # matching only -- a plain substring check would false-positive
        # on real descriptions like "sitting stationary" (contains "stat")
        # or "textured background" (contains "text").
        if what and _RED_FLAG_RE.search(what.lower()):
            print(f"[facecam_vision] rejected box described as {what!r} (looks like UI, not a person)", flush=True)
            continue
        x = max(0.0, min(x, src_w - 1))
        y = max(0.0, min(y, src_h - 1))
        w = min(w, src_w - x)
        h = min(h, src_h - y)
        boxes.append((int(x), int(y), int(w), int(h)))
        print(f"[facecam_vision] kept box ({int(x)},{int(y)},{int(w)},{int(h)}): {what!r}", flush=True)

    return _dedupe_boxes(boxes)


def _extract_frame_b64(video_path: Path, t_frac: float = 0.5) -> Optional[str]:
    """One frame from partway through an already-rendered (short) clip."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
    cap.set(cv2.CAP_PROP_POS_MSEC, duration_ms * t_frac)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > _MAX_FRAME_WIDTH:
        scale = _MAX_FRAME_WIDTH / w
        frame = cv2.resize(frame, (_MAX_FRAME_WIDTH, int(h * scale)))
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok2:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


_VERIFY_PROMPT = """This is a frame from an already-rendered vertical short-form clip. The
bottom portion of the frame is supposed to show one or more clean webcam/
facecam windows of real people -- their actual camera feed, filling that
part of the frame.

IMPORTANT, two things that are expected and correct, not defects:
- This clip has burned-in captions, and the caption text (white text,
  usually with a colored highlight word, near the bottom of the frame)
  is drawn OVER the video on every clip, whether or not the caption box
  happens to overlap the facecam band underneath it. Do NOT count
  caption text, its outline, or its drop shadow as a defect.
- When there's more than one facecam side by side, each one is fit into
  its own tile without stretching -- so a tile can have plain black bars
  on its left/right or top/bottom around the actual facecam footage if
  the person's camera feed doesn't perfectly match the tile's shape.
  Do NOT count plain black letterboxing bars next to/around an otherwise
  clean facecam as a defect -- only flag it if the visible facecam
  content itself (not the black bars) is wrong.

Judge only the actual video content showing through/around the caption
and outside any letterbox bars.

Look specifically at the bottom portion and answer: does the video content
there (ignoring caption text) show clean, correctly-cropped facecam(s) of
real people? Answer NO if you see any of:
- part of the surrounding gameplay/game footage bleeding into the facecam
  area (not just the person's own camera feed)
- a game UI element, score card, stats/results screen, or other graphic
  instead of (or alongside) a real person
- a video player, webpage, browser, or other on-screen content the
  streamer is watching/browsing -- even one with a real person's face
  visible in it (e.g. a YouTube video) -- since that's content being
  viewed, not the streamer's own camera feed
- the same person's face duplicated/repeated in more than one spot
- no facecam at all where one should be, or something clearly wrong or
  broken-looking about the video crop itself (not the caption overlay)

Respond with ONLY one word: YES or NO."""


# One frame is too easy to catch at a bad instant -- a caption transition,
# a motion blur frame, a brief occlusion -- and a single unlucky NO used to
# be enough to discard an entire correctly-detected facecam. Sample a few
# points across the clip and go with the majority instead, so one noisy
# frame can't overrule the rest.
_VERIFY_FRACS = (0.25, 0.5, 0.75)


def verify_rendered_facecam(
    output_path: Path, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
) -> Optional[bool]:
    """True if the ALREADY-RENDERED clip's facecam band looks clean, False
    if it looks visibly wrong, or None if this check wasn't usable at all
    (no API key, no frame could be extracted/answered) -- callers should
    treat None as "can't tell, don't block on it," not as either true or
    false.

    This is a final catch-all after pre-render detection: it looks at what
    actually got burned into the output, not just what compute_layout
    predicted from the source frames, so it can catch any failure mode
    (a wrong box, a duplicated face, a misidentified graphic, padding that
    grabbed surrounding gameplay) in one check instead of needing a
    bespoke fix for each new way this can go wrong. Samples multiple
    frames and requires a clear majority of NOs before rejecting -- ties
    (e.g. only one frame gave a usable answer) favor keeping the render,
    since discarding a real facecam is a worse outcome than occasionally
    keeping one with a minor blemish."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    votes: List[bool] = []
    for t_frac in _VERIFY_FRACS:
        frame_b64 = _extract_frame_b64(output_path, t_frac=t_frac)
        if not frame_b64:
            continue
        content: List[dict] = [
            {"type": "text", "text": _VERIFY_PROMPT},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}},
        ]
        try:
            resp = client.messages.create(model=model, max_tokens=10, messages=[{"role": "user", "content": content}])
            raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip().upper()
        except Exception as e:
            print(f"[facecam_vision] post-render verification call failed, skipping frame: {e}", flush=True)
            continue
        if raw.startswith("Y"):
            votes.append(True)
        elif raw.startswith("N"):
            votes.append(False)
        else:
            print(f"[facecam_vision] post-render verification gave an unclear answer ({raw!r}), skipping frame", flush=True)

        # Once two frames agree, a third can't change the majority --
        # stop early rather than spend another API call per clip on it.
        if votes.count(True) >= 2 or votes.count(False) >= 2:
            break

    if len(votes) < 2:
        # A single usable frame is exactly the failure mode this function
        # exists to avoid trusting -- not enough to call it either way.
        print(f"[facecam_vision] post-render verification only got {len(votes)} usable frame(s), skipping", flush=True)
        return None

    no_votes = votes.count(False)
    yes_votes = len(votes) - no_votes
    result = no_votes <= yes_votes
    print(f"[facecam_vision] post-render verification votes: {yes_votes} YES / {no_votes} NO -> {'keep' if result else 'reject'}", flush=True)
    return result
