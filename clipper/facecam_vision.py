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


def _sent_size(w: int, h: int) -> Tuple[int, int]:
    """The size a frame is actually sent to the model at -- what any pixel
    coordinates the model gives back are measured in."""
    if w > _MAX_FRAME_WIDTH:
        return _MAX_FRAME_WIDTH, int(h * (_MAX_FRAME_WIDTH / w))
    return w, h


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
        sent_w, sent_h = _sent_size(w, h)
        if (sent_w, sent_h) != (w, h):
            frame = cv2.resize(frame, (sent_w, sent_h))
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
- a real person who is simply part of the same continuous real-world shot as
  everything else on screen -- e.g. two or more people filmed together in
  one room/scene (an interview, a conversation, people at a table or
  walking together), even if one of them happens to sit toward a corner or
  edge of the frame. A genuine facecam overlay is a separately COMPOSITED
  window: it has its own distinct border/edge, and its lighting/background
  don't match the rest of the frame because it's a different camera feed
  layered on top. If a person shares the same background, lighting, and
  physical space as the rest of the shot, they are part of the main scene,
  not an overlay -- no matter where in the frame they're standing
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

Each frame is {width}x{height} pixels. Respond with ONLY a JSON array (no
other text), one entry per distinct facecam overlay found, in this exact
shape:
[{{"x": 0, "y": 335, "w": 173, "h": 108, "what": "person wearing headphones, real camera feed"}}]

x/y are the pixel coordinates of the window's top-left corner and w/h its
size in pixels, in that {width}x{height} image, covering the visible facecam
window as tightly as reasonable. "what" is a short (under 10
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


_CROP_REFINE_PROMPT = """Each image is a region cut out of a livestream frame, taken from around a
spot where a streamer's facecam/webcam window was detected. The detected
bounds are frequently the wrong SIZE -- too large or too small, or offset
-- so in any given image the webcam window may not fill the frame. It
might sit off to one side, or be surrounded by gameplay, room decor, a
wall poster, a monitor, or a neighbouring streamer's separate webcam.

For each image IN ORDER, locate the streamer's own webcam window -- the
rectangular live camera feed of a real person -- and give its TIGHT
bounds as fractions (0-1) of THAT IMAGE's own width and height.

Fit the box to the webcam window itself: include the whole of that
person's camera frame, and exclude gameplay, UI, wall/room background
outside the window, and any separate webcam belonging to someone else.

Return a JSON array in the same order as the images, each entry either:
  {"x":0.10,"y":0.22,"w":0.55,"h":0.63}   the window's bounds in that image
  null                                    no real person's camera feed here

Respond with ONLY the JSON array. Nothing else."""

# How much context to include around a detected box when asking for a
# refinement. The detected bounds are unreliable in size, so the crop has
# to be roomy enough that a too-small box still contains the whole window,
# while staying tight enough that the window is large in frame -- which is
# the entire reason this second pass localizes better than the first.
_REFINE_PAD = 1.9


def _extract_frame_bgr(video_path: Path, t: float):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _padded_region(box, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
    """The region to show the model when refining `box` -- the box grown
    about its own centre by _REFINE_PAD, clipped to the frame."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    rw, rh = min(w * _REFINE_PAD, src_w), min(h * _REFINE_PAD, src_h)
    rx = int(round(max(0, min(cx - rw / 2, src_w - rw))))
    ry = int(round(max(0, min(cy - rh / 2, src_h - rh))))
    return rx, ry, int(round(rw)), int(round(rh))


def _refine_box_crops(
    video_path: Path, start: float, end: float,
    boxes: List[Tuple[int, int, int, int]], client, model: str,
) -> List[Tuple[int, int, int, int]]:
    """Re-locate each detected facecam window by looking at it up close.

    Whole-frame localization is the weakest thing asked of the model, and
    the failure is one of SIZE rather than position: a three-way collab
    came back with all three windows found in roughly the right places but
    bounded wrongly, so the crops landed on a wall poster and a monitor
    beside two of the streamers instead of on the streamers. All three
    were real people; only one was framed correctly.

    Rejecting the odd-looking ones would throw away two real facecams, so
    this corrects them instead. Each candidate is re-shown as a roomy crop
    around itself, where the window occupies much more of the image and is
    correspondingly easier to bound, and the model returns tight bounds
    within that crop, which are mapped back to source coordinates. A null
    verdict (genuinely nothing there) still drops the box, which is what
    catches a face that belongs to on-screen content rather than a camera.

    Fails open at every step: an error, an unusable answer, a mismatched
    count, or an implausible refinement all keep the original box, so a
    bad pass can never silently turn facecams off."""
    if not boxes:
        return boxes

    frame = _extract_frame_bgr(video_path, start + max(end - start, 0.1) / 2)
    if frame is None:
        return boxes

    import cv2

    src_h, src_w = frame.shape[:2]
    regions = [_padded_region(b, src_w, src_h) for b in boxes]
    content: List[dict] = [{"type": "text", "text": _CROP_REFINE_PROMPT}]
    for (rx, ry, rw, rh) in regions:
        crop = frame[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            return boxes
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return boxes
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.b64encode(buf.tobytes()).decode("ascii")},
        })

    try:
        resp = client.messages.create(
            model=model, max_tokens=600, messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        answers = json.loads(raw)
    except Exception as e:
        print(f"[facecam_vision] box refinement failed, keeping boxes as detected: {e}", flush=True)
        return boxes

    if not isinstance(answers, list) or len(answers) != len(boxes):
        got = len(answers) if isinstance(answers, list) else "?"
        print(
            f"[facecam_vision] box refinement returned {got} answer(s) for {len(boxes)} box(es), "
            "keeping boxes as detected", flush=True,
        )
        return boxes

    refined: List[Tuple[int, int, int, int]] = []
    for box, region, answer in zip(boxes, regions, answers):
        if answer is None:
            print(f"[facecam_vision] refinement found no camera feed around {box} -- dropping it", flush=True)
            continue
        rx, ry, rw, rh = region
        try:
            nx = rx + float(answer["x"]) * rw
            ny = ry + float(answer["y"]) * rh
            nw = float(answer["w"]) * rw
            nh = float(answer["h"]) * rh
        except (KeyError, TypeError, ValueError):
            print(f"[facecam_vision] refinement gave an unusable box for {box}, keeping it as detected", flush=True)
            refined.append(box)
            continue
        # A refinement should tighten a box, not invent a new one somewhere
        # else -- anything degenerate or larger than the region it came
        # from means the model lost track of the frame it was given.
        if nw < 8 or nh < 8 or nw > rw or nh > rh:
            print(f"[facecam_vision] refinement for {box} was implausible ({int(nw)}x{int(nh)}), keeping it as detected", flush=True)
            refined.append(box)
            continue
        nx = max(0.0, min(nx, src_w - 1))
        ny = max(0.0, min(ny, src_h - 1))
        nw, nh = min(nw, src_w - nx), min(nh, src_h - ny)
        new_box = (int(nx), int(ny), int(nw), int(nh))
        print(f"[facecam_vision] refined box {box} -> {new_box}", flush=True)
        refined.append(new_box)

    print(f"[facecam_vision] refinement kept {len(refined)}/{len(boxes)} box(es)", flush=True)
    return refined


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

    # Boxes come back in pixels of the image the model actually saw, not as
    # fractions of it. Asked for fractions, it put every facecam's y at
    # ~9/16 of where it really was (x was fine) -- job after job, the box
    # drawn by hand afterwards sat at 1.73-1.77x the detected y, i.e. y
    # measured against the frame's width instead of its height. The crop
    # then took the gameplay above each cam, and the post-render check
    # rightly rejected every one of those renders.
    sent_w, sent_h = _sent_size(src_w, src_h)
    content: List[dict] = [{"type": "text", "text": _PROMPT.format(width=sent_w, height=sent_h)}]
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
            x = float(item["x"]) * src_w / sent_w
            y = float(item["y"]) * src_h / sent_h
            w = float(item["w"]) * src_w / sent_w
            h = float(item["h"]) * src_h / sent_h
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

    # _refine_box_crops is deliberately NOT called. Re-localizing each box
    # in a zoomed crop was supposed to fix its size; in production it moved
    # boxes around instead -- a cam at y=172 refined down to y=340 while the
    # one at y=302 refined up to y=251, the two crossing straight past each
    # other, and one came back as a 577x137 strip that no webcam has ever
    # been shaped like. The raw detections, by contrast, land on the same
    # three positions clip after clip (bottom-left ~(0,650,580,430) and two
    # right-edge cams at y~170 and y~302), so they are the more trustworthy
    # signal and are used as-is.
    return _dedupe_boxes(boxes)


_WIDE_SCENE_PROMPT = """These are frames sampled from a video clip already confirmed to have NO
facecam/webcam overlay window composited on it (no small corner picture-in-
picture of a streamer's own camera feed).

Decide which of these better describes the shot:

WIDE -- a genuine multi-person or wide IRL scene filmed as one continuous
shot: more than one person visible and interacting (an interview, a duo/
group conversation, people at a table or walking around), or a wide
establishing shot where no single person should be zoomed into while
cutting the others out of frame.

SOLO -- a single person's shot (a webcam-style explainer, one streamer
facing their own camera with nobody else consistently in frame) where
zooming/cropping to that one person is correct, even if they move around
some.

Respond with ONLY one word: WIDE or SOLO."""


def detect_wide_scene(
    video_path: Path, start: float, end: float,
    api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
) -> Optional[bool]:
    """True if this clip (already confirmed to have no facecam overlay) is
    a genuine multi-person/wide IRL scene that should show the WHOLE frame
    rather than being cropped/zoomed into one person -- see
    reframe.compute_layout's landscape branch, which otherwise anchors on
    whichever single face recurred most and crops everyone else out. False
    for a solo talking-head shot, where the existing face-anchored crop is
    already correct. None if vision wasn't usable (no API key, frame
    extraction/the call failed, or an unclear answer) -- the caller should
    fall back to the existing crop/anchor logic unchanged, the same
    fails-open behavior as detect_facecams."""
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

    content: List[dict] = [{"type": "text", "text": _WIDE_SCENE_PROMPT}]
    for b64 in frames_b64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(model=model, max_tokens=10, messages=[{"role": "user", "content": content}])
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip().upper()
    except Exception as e:
        print(f"[facecam_vision] wide-scene classification failed, keeping existing crop logic: {e}", flush=True)
        return None

    if raw.startswith("WIDE"):
        return True
    if raw.startswith("SOLO"):
        return False
    print(f"[facecam_vision] wide-scene classification gave an unclear answer ({raw!r}), keeping existing crop logic", flush=True)
    return None


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
