"""Work out a 9:16 (or other target ratio) layout for a clip.

Samples a handful of frames across the clip and runs OpenCV's built-in Haar
cascade face detector on each (ships with opencv-python, no extra download).

Three outcomes:
  - No face, an unreliable/one-off detection (chaotic handheld footage --
    fast panning, motion blur, crowds -- where nothing was detected
    consistently enough to trust), or a large/roughly-centered face
    (talking head, interview, explainer video): a single CropWindow
    centered on the face, or a plain center crop if there's no face worth
    anchoring on.
  - Exactly one small, corner-positioned, recurring face (a streamer's
    webcam overlay sitting on top of gameplay footage): a SplitLayout with
    gameplay on top and a zoomed-in facecam crop on the bottom, stacked to
    fill the vertical frame -- the standard layout real clip channels use,
    instead of zooming into just the tiny facecam box and losing all the
    game context.
  - Two or more such overlays (a duo/co-stream layout with multiple
    facecams over the same gameplay): a MultiCamSplitLayout with gameplay
    on top and all the facecams (up to MAX_COCAM_TILES) tiled side-by-side
    across the bottom, instead of picking just one or falling back to a
    plain crop that shows none of them.

Before falling back to this Haar-based heuristic, compute_layout first
tries asking Claude (vision) to identify the real facecam overlays
directly from a few sample frames -- see facecam_vision.py. A classical
cascade can't reliably tell a real person's camera feed from a game
texture that just happens to look vaguely face-like, or catch a small/
angled facecam a human would obviously recognize; a vision model can
just look at the picture. The Haar path below only runs when vision is
unavailable (no API key) or a call fails, so it stays as a real fallback,
not dead code.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import facecam_vision


@dataclass
class CropWindow:
    x: int
    y: int
    w: int
    h: int


@dataclass
class SplitLayout:
    top: CropWindow       # gameplay region, source-video pixel coordinates
    bottom: CropWindow    # facecam region, source-video pixel coordinates
    top_out_h: int        # output pixel height the top region scales to
    bottom_out_h: int     # output pixel height the bottom region scales to


@dataclass
class MultiCamSplitLayout:
    """Like SplitLayout, but for a duo/trio co-stream: 2-3 facecam overlays
    tiled side-by-side across the bottom band instead of picking (or
    guessing at) just one. bottom_cams is left-to-right in on-screen
    reading order. bottom_cam_out_widths gives each tile's output width --
    an equal split, kept as explicit values so render_clip tiles them
    across out_w in that order without recomputing the division."""
    top: CropWindow
    bottom_cams: List[CropWindow]
    top_out_h: int
    bottom_out_h: int
    bottom_cam_out_widths: List[int]


Layout = Union[CropWindow, SplitLayout, MultiCamSplitLayout]

# Detected overlay clusters beyond this are almost always detector noise
# (Haar false-positives), not a real 4+-way co-stream -- and even a genuine
# one would make each tile too small to be worth showing. Cap at 3 and,
# when there are more candidates than that, keep the ones detected most
# consistently (see compute_layout).
MAX_COCAM_TILES = 3


def _even(value: float) -> int:
    """Round to the nearest even integer.

    Every output dimension has to be even: the encoder is yuv420p, whose
    chroma planes are half-resolution, so libx264 rejects an odd width or
    height outright ("height not divisible by 2"). This bit the split
    layouts specifically -- each half is scaled separately and only then
    stacked, so an odd top and an odd bottom each get rounded to even on
    their own and the stacked result misses target_h by a pixel (seen in
    production as a 1080x1919 encoder failure that killed the whole job).
    Back when the band was a fixed fraction of the frame it was always
    even by luck; deriving it from measured aspect ratios lands on odd
    roughly half the time, so it has to be forced."""
    return int(round(value / 2)) * 2


def _detect_faces(video_path: Path, start: float, end: float, samples: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if src_w <= 0 or src_h <= 0:
        cap.release()
        raise RuntimeError(
            f"Could not read video dimensions (got {src_w}x{src_h}): {video_path}"
        )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Keep every face found in each sampled frame, not just the biggest --
    # a duo/co-stream layout with two facecam overlays needs both to show
    # up in the detections for compute_layout to tell "two static
    # overlays" apart from "one face wandering around" (picking only the
    # biggest per frame means "biggest" can flip between the two overlays
    # from sample to sample, which otherwise looks just like one face
    # jumping between two unrelated positions).
    all_faces: List[Tuple[int, int, int, int]] = []
    duration = max(end - start, 0.1)
    for i in range(samples):
        t = start + duration * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tuned from the original scaleFactor=1.15/minNeighbors=5/
        # minSize=(60,60): that missed a facecam confirmed genuinely on
        # screen the whole clip (only 1-2 of 9 samples hit). A first pass
        # at 1.08/4/(35,35) fixed the miss but went too far the other way
        # -- 21 detected clusters on one clip where only 3 were real,
        # noticeably "off" on clips that don't actually have a co-stream
        # overlay. minNeighbors is the main per-frame false-positive
        # control (back to the original 5, not loosened), scaleFactor is
        # only slightly finer than stock, and minSize is smaller than
        # stock but not as small as the first attempt -- still catches a
        # smaller/angled facecam, without flooding every clip with noise.
        # The recurrence-based occurrence_frac check downstream is the
        # real defense against any single false positive being trusted;
        # this tuning is about not burying real detections in noise.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
        for f in faces:
            all_faces.append(tuple(int(v) for v in f))

    cap.release()
    return all_faces, src_w, src_h


def _cluster_faces(
    boxes: List[Tuple[int, int, int, int]], src_w: int, src_h: int, samples: int, tolerance: float = 0.08
) -> List[dict]:
    """Group detected face boxes by on-screen position. A stream with two
    facecam overlays produces two separate, recurring clusters; a single
    face that moves around the frame (walking, turning) produces several
    small, one-off clusters instead of one that recurs. Simple greedy
    clustering by normalized center distance is plenty for a handful of
    samples with at most a couple of faces each -- no need for a real
    clustering library here."""
    clusters: List[dict] = []
    for box in boxes:
        x, y, w, h = box
        cx, cy = (x + w / 2) / src_w, (y + h / 2) / src_h
        match = None
        for c in clusters:
            if ((cx - c["cx"]) ** 2 + (cy - c["cy"]) ** 2) ** 0.5 < tolerance:
                match = c
                break
        if match is None:
            clusters.append({"boxes": [box], "cx": cx, "cy": cy})
        else:
            match["boxes"].append(box)
            n = len(match["boxes"])
            match["cx"] += (cx - match["cx"]) / n
            match["cy"] += (cy - match["cy"]) / n

    results = []
    for c in clusters:
        cb = c["boxes"]
        med_box = (
            statistics.median(b[0] for b in cb),
            statistics.median(b[1] for b in cb),
            statistics.median(b[2] for b in cb),
            statistics.median(b[3] for b in cb),
        )
        # What fraction of sampled frames this position showed up in -- a
        # real composited overlay is on screen essentially the whole time,
        # unlike a one-off detection blip or a face passing through this
        # spot briefly.
        results.append({"box": med_box, "occurrence_frac": len(cb) / samples})
    return results


def center_crop_layout(video_path: Path, target_w: int = 1080, target_h: int = 1920) -> CropWindow:
    """A plain center crop of the whole frame -- no face or facecam
    detection at all. Used as a safe fallback re-render when a facecam
    layout fails post-render verification: a plain crop can't have any of
    the specific failure modes (wrong box, a duplicated face, a game
    graphic mistaken for a person) a facecam split can, so it's always a
    safe thing to fall back to."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if src_w <= 0 or src_h <= 0:
        raise RuntimeError(f"Could not read video dimensions (got {src_w}x{src_h}): {video_path}")
    return _center_crop(src_w, src_h, target_w, target_h)


def _center_crop(src_w: int, src_h: int, target_w_ratio: int, target_h_ratio: int) -> CropWindow:
    if src_w / src_h > target_w_ratio / target_h_ratio:
        crop_h = src_h
        crop_w = int(crop_h * target_w_ratio / target_h_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w * target_h_ratio / target_w_ratio)
    x = max(0, (src_w - crop_w) // 2)
    y = max(0, (src_h - crop_h) // 2)
    return CropWindow(x=x, y=y, w=crop_w, h=crop_h)


def _top_crop_excluding_overlays(
    src_w: int, src_h: int, boxes: List[Tuple[int, int, int, int]], target_w: int, target_h: int,
) -> CropWindow:
    """The "gameplay" region's crop for a split layout, taken from the
    widest strip of the source that no facecam box sits in -- looking
    across the frame as well as down it.

    Only vertical bands were considered at first, on the assumption that
    overlays sit along one edge. A three-way collab breaks that: cams in
    the top-right AND bottom-left corners leave no clear band above or
    below, so the widest gap found was the 170px sliver above the topmost
    cam, and that got magnified to fill the whole frame -- a rendered
    clip whose gameplay half was a blown-up strip of near-empty
    background. The free space in that layout is the column BETWEEN the
    corners, which this now finds by looking for gaps in both axes.

    A plain _center_crop doesn't know where the facecams are, and for a
    typical 16:9 source cropped to a taller target ratio it keeps the
    FULL source height anyway (confirmed: 1920x1080 cropped to a
    1080x1152 target keeps all 1080px of height) -- so the "gameplay"
    region silently re-shows the exact same facecam pixels natively,
    right above the fresh, zoomed-in tiles of those same faces rendered
    below it. That's what a duplicated/nested-looking camera grid in the
    output actually was: not a rendering bug, the top crop was simply
    never excluding the region the bottom band was also showing.

    Falls back to the plain center crop if the box-free band is too
    thin to be worth cropping to (rather than producing a nonsensically
    tiny/over-zoomed result)."""
    full = _center_crop(src_w, src_h, target_w, target_h)
    if not boxes:
        return full

    def _gaps(intervals: List[Tuple[int, int]], limit: int) -> List[Tuple[int, int]]:
        """Stretches of 0..limit not covered by any interval."""
        merged: List[List[int]] = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        free, cursor = [], 0
        for lo, hi in merged:
            if lo > cursor:
                free.append((cursor, lo - cursor))
            cursor = max(cursor, hi)
        if cursor < limit:
            free.append((cursor, limit - cursor))
        return free

    candidates = [
        (gx, 0, gw, src_h) for gx, gw in _gaps([(b[0], b[0] + b[2]) for b in boxes], src_w)
    ] + [
        (0, gy, src_w, gh) for gy, gh in _gaps([(b[1], b[1] + b[3]) for b in boxes], src_h)
    ]

    best = None
    for bx, by, bw, bh in candidates:
        if bw <= 0 or bh <= 0:
            continue
        crop = _center_crop(bw, bh, target_w, target_h)
        area = crop.w * crop.h
        if best is None or area > best[0]:
            best = (area, bx, by, crop)

    # Nothing box-free is big enough to be worth the zoom it would force.
    # A plain centre crop re-shows the source's own facecams above the
    # tiles rendered from them, which looks like a duplicated camera grid
    # -- but that still beats magnifying a sliver of background to fill
    # the frame.
    if best is None or best[0] < full.w * full.h * 0.25:
        return full
    _, bx, by, crop = best
    return CropWindow(x=bx + crop.x, y=by + crop.y, w=crop.w, h=crop.h)


def _face_centered_crop(src_w: int, src_h: int, center_x: float, target_w_ratio: int, target_h_ratio: int) -> CropWindow:
    base = _center_crop(src_w, src_h, target_w_ratio, target_h_ratio)
    x = int(round(center_x - base.w / 2))
    x = max(0, min(x, src_w - base.w))
    return CropWindow(x=x, y=base.y, w=base.w, h=base.h)


def _facecam_crop(src_w: int, src_h: int, box, out_w: int, out_h: int, pad: float = 1.7) -> CropWindow:
    """Crop around a padded facecam box, expanded about its centre to
    exactly the output tile's aspect ratio so it scales in with no
    distortion and no black bars.

    This only stays close to the box itself while the tile is shaped
    roughly like the box. It is _build_overlay_layout's job to keep it
    that way by sizing the band to the cams -- give this a tall tile and
    a wide window and it will reach far outside the window to fill it,
    which is where the gameplay-bleeding renders came from."""
    fx, fy, fw, fh = box
    cx, cy = fx + fw / 2, fy + fh / 2

    pad_w, pad_h = fw * pad, fh * pad
    target_aspect = out_w / out_h
    if pad_w / pad_h > target_aspect:
        crop_w = pad_w
        crop_h = crop_w / target_aspect
    else:
        crop_h = pad_h
        crop_w = crop_h * target_aspect

    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    x = int(round(cx - crop_w / 2))
    y = int(round(cy - crop_h / 2))
    x = max(0, min(x, src_w - int(crop_w)))
    y = max(0, min(y, src_h - int(crop_h)))
    return CropWindow(x=x, y=y, w=int(crop_w), h=int(crop_h))



def _even_tile_widths(total_w: int, n: int) -> List[int]:
    """Split total_w into n even widths summing exactly to total_w --
    even because libx264 rejects odd dimensions on a yuv420p encode."""
    base = _even(total_w / n)
    widths = [base] * n
    widths[-1] = total_w - base * (n - 1)
    return widths


def _build_overlay_layout(
    boxes: List[Tuple[int, int, int, int]],
    src_w: int, src_h: int, target_w: int, target_h: int, facecam_height_frac: float,
    pad: float = 1.7,
) -> Optional[Layout]:
    """Build a SplitLayout/MultiCamSplitLayout from a list of pixel-space
    facecam boxes already decided to be the real overlays and already
    capped to at most MAX_COCAM_TILES -- however that decision was made
    (vision or the Haar heuristic). Sorts left-to-right for on-screen
    reading order. None (meaning: no overlay here) if boxes is empty --
    the caller should fall through to the no-overlay case (anchor or
    center crop).

    `pad` should match the box shape: Haar returns a tight face and needs
    generous padding to reach the webcam window around it, while a
    vision-detected box already covers the whole window and needs very
    little."""
    if not boxes:
        return None
    boxes = sorted(boxes, key=lambda b: b[0])

    # Tiles are EQUAL width. They were briefly sized per-cam from each
    # window's own aspect, which fills each one perfectly but makes a row
    # of visibly mismatched tiles -- and uneven tiles are what read as
    # wrong on the first multi-cam render that actually shipped, not the
    # content, which was three real people throughout.
    #
    # The band height is then set so that an equal tile matches the shape
    # these cams actually are: at 3 across, 280x180 windows want a ~230px
    # band, not the fixed 768 a flat 40%-of-frame gives. Getting that
    # wrong is what every earlier attempt was really fighting -- too tall
    # a band left tiles mostly black, and cropping wider to fill them
    # dragged in gameplay and the neighbouring cam. Sizing it to the cams
    # means each tile is filled by the person, tiles stay uniform, and
    # whatever variation remains between cams is a thin letterbox that
    # render_clip already handles.
    tile_out_widths = _even_tile_widths(target_w, len(boxes))
    aspects = sorted(bw / bh for (_, _, bw, bh) in boxes)
    median_aspect = aspects[len(aspects) // 2]
    natural_h = min(tile_out_widths) / median_aspect
    bottom_out_h = _even(min(max(natural_h, target_h * 0.10), target_h * facecam_height_frac))
    top_out_h = target_h - bottom_out_h
    top = _top_crop_excluding_overlays(src_w, src_h, boxes, target_w, top_out_h)
    if len(boxes) == 1:
        bottom = _facecam_crop(src_w, src_h, boxes[0], target_w, bottom_out_h, pad=pad)
        return SplitLayout(top=top, bottom=bottom, top_out_h=top_out_h, bottom_out_h=bottom_out_h)
    bottom_cams = [
        _facecam_crop(src_w, src_h, b, tw, bottom_out_h, pad=pad)
        for b, tw in zip(boxes, tile_out_widths)
    ]
    # Split target_w in proportion to the aspect ratios, rather than as
    return MultiCamSplitLayout(
        top=top, bottom_cams=bottom_cams, top_out_h=top_out_h, bottom_out_h=bottom_out_h,
        bottom_cam_out_widths=tile_out_widths,
    )


def _snap_boxes_to_faces(
    vision_boxes: List[Tuple[int, int, int, int]], clusters: List[dict],
    src_w: int, src_h: int, min_occurrence: float,
) -> List[Tuple[int, int, int, int]]:
    """Re-centre each vision box on a real detected face near it.

    Vision reads a facecam's SIZE well but places it unreliably: a cam
    actually sitting at y~650 came back at y=345 on one clip, and the
    resulting tiles cropped the space above each person with only the top
    of a head showing at the bottom edge. It's a vertical offset, not a
    misidentification -- the right window, in the wrong place.

    The Haar detector already ran over this clip and its clusters are
    exactly what's needed to fix that: it is poor at judging whether a
    face-like patch is a real camera feed (which is why vision decides
    that) but precise about where a face actually is. So each box keeps
    vision's size and takes Haar's position, when a recurring face is
    found near enough to be the same one.

    A box with no recurring face nearby is left exactly as detected --
    Haar misses real faces often enough (poor lighting, an angled head)
    that treating a miss as evidence of absence would drop real cams."""
    snapped: List[Tuple[int, int, int, int]] = []
    for box in vision_boxes:
        vx, vy, vw, vh = box
        vcx, vcy = vx + vw / 2, vy + vh / 2
        best = None
        for c in clusters:
            if c["occurrence_frac"] < min_occurrence:
                continue  # one-off detector noise, not a face to trust
            fx, fy, fw, fh = c["box"]
            fcx, fcy = fx + fw / 2, fy + fh / 2
            # Within roughly this box's own span, so a cam never snaps
            # onto the neighbouring streamer's face.
            if abs(fcx - vcx) > vw or abs(fcy - vcy) > vh:
                continue
            dist = (fcx - vcx) ** 2 + (fcy - vcy) ** 2
            if best is None or dist < best[0]:
                best = (dist, fcx, fcy)
        if best is None:
            snapped.append(box)
            continue
        _, fcx, fcy = best
        nx = int(round(max(0, min(fcx - vw / 2, src_w - vw))))
        ny = int(round(max(0, min(fcy - vh / 2, src_h - vh))))
        if (nx, ny) != (vx, vy):
            print(f"[reframe] snapped facecam box ({vx},{vy}) -> ({nx},{ny}) onto a detected face", flush=True)
        snapped.append((nx, ny, vw, vh))
    return snapped


def _log_layout(start: float, end: float, clusters: List[dict], outcome: str) -> None:
    """One line per clip on which layout it got and why -- there's no
    other way to tell, after the fact, whether a clip that came out with
    no facecam was a deliberate "nothing reliable enough" call or a
    detection near-miss, short of re-running compute_layout by hand."""
    occs = ", ".join(f"{c['occurrence_frac']:.2f}" for c in clusters)
    print(f"[reframe] clip {start:.1f}-{end:.1f}s: {len(clusters)} cluster(s) [{occs}] -> {outcome}", flush=True)


def compute_layout(
    video_path: Path,
    start: float,
    end: float,
    target_w: int = 1080,
    target_h: int = 1920,
    # Bumped from 9: with per-frame detection tuned down to avoid noise
    # (see _detect_faces), a marginal-but-real facecam clears the ~1/3
    # occurrence bar less reliably on any single run of samples -- more
    # temporal trials narrows that gap without loosening detection itself.
    # This differentially helps: a facecam that's genuinely on screen the
    # whole clip recurs at roughly the same rate no matter how many times
    # it's sampled, so more samples mostly just reduces the odds bad luck
    # (a brief occlusion, a bad angle) pushes it under the bar; one-off
    # detector noise, by contrast, doesn't gain nearly as much of a boost
    # from being sampled more since it isn't actually recurring.
    samples: int = 15,
    facecam_height_frac: float = 0.40,
) -> Layout:
    boxes, src_w, src_h = _detect_faces(video_path, start, end, samples)

    if not boxes:
        _log_layout(start, end, [], "center crop (no faces detected)")
        return _center_crop(src_w, src_h, target_w, target_h)

    clusters = _cluster_faces(boxes, src_w, src_h, samples)

    # Anchor point for a single-crop layout: whichever detected face
    # recurred across the most sampled frames (ties broken by size), not
    # simply the single largest box seen. Chaotic, fast-moving footage --
    # handheld/IRL streams, motion blur, panning, crowds in the background
    # -- makes face detection noisy; picking by raw size alone lets one
    # spurious, oversized false-positive from a single frame hijack the
    # crop for the *entire* clip (compute_layout runs once per clip and
    # commits to one static crop window). A detection that only showed up
    # once out of several samples isn't reliable enough to anchor on --
    # a plain center crop is the safer default there.
    most_consistent = max(clusters, key=lambda c: (c["occurrence_frac"], c["box"][2] * c["box"][3]))
    min_confident_occurrence = 1.0 / 3  # detected in at least ~1/3 of sampled frames
    has_confident_anchor = most_consistent["occurrence_frac"] >= min_confident_occurrence
    anchor_x = most_consistent["box"][0] + most_consistent["box"][2] / 2

    # A source that's already portrait/vertical (a phone-held IRL stream,
    # anything already shot roughly 9:16) is one continuous scene -- not a
    # landscape gameplay feed with facecam window(s) composited on top of
    # it. The overlay handling below exists specifically to catch that
    # pattern, and misfires here: an IRL streamer's face is naturally
    # small and/or off-center plenty of the time just from them moving
    # around, and splitting the frame on that would zoom into a chunk of
    # it as a fake "facecam" while throwing away the rest of the actual
    # scene. Always give a portrait source a single crop of the whole
    # frame instead, anchored on the most reliably-detected face -- or
    # centered, if nothing was detected reliably enough to trust.
    if src_h >= src_w:
        if has_confident_anchor:
            _log_layout(start, end, clusters, "portrait source: face-anchored crop")
            return _face_centered_crop(src_w, src_h, anchor_x, target_w, target_h)
        _log_layout(start, end, clusters, "portrait source: center crop (no confident anchor)")
        return _center_crop(src_w, src_h, target_w, target_h)

    # Ask Claude to identify the real facecam overlay(s) directly from a
    # few sample frames -- far more reliable than the Haar heuristic below
    # at telling an actual person's camera feed apart from a game texture
    # that just happens to look vaguely face-like, and at catching a
    # small/angled facecam a human would obviously recognize. None means
    # vision wasn't usable this call (no API key, a transient error) --
    # fall through to the Haar-based heuristic unchanged in that case. An
    # empty list means vision actually ran and found no overlay here.
    try:
        vision_boxes = facecam_vision.detect_facecams(video_path, start, end, src_w, src_h)
    except Exception as e:
        print(f"[reframe] vision facecam detection errored, falling back to heuristic: {e}", flush=True)
        vision_boxes = None

    if vision_boxes is not None:
        # A small pad here (not the 1.7x the Haar path below needs) --
        # vision is asked for the whole facecam window already, not just a
        # face, so padding it again by 1.7x overshoots past the window
        # into surrounding gameplay. Just enough margin to keep a
        # razor-tight crop off the window's own edge.
        snapped_boxes = _snap_boxes_to_faces(
            vision_boxes[:MAX_COCAM_TILES], clusters, src_w, src_h, min_confident_occurrence,
        )
        layout = _build_overlay_layout(
            snapped_boxes, src_w, src_h, target_w, target_h, facecam_height_frac,
            pad=1.08,
        )
        if layout is not None:
            n = len(vision_boxes[:MAX_COCAM_TILES])
            _log_layout(start, end, clusters, f"{n}-cam split (vision)")
            return layout
        # Vision confidently found no facecam overlay -- fall through to
        # the anchor/center-crop case below (vision only rules out an
        # *overlay* pattern here, not a talking-head-style crop, so the
        # Haar-based anchor logic still gets a say).
    else:
        def is_overlay(c: dict) -> bool:
            med_x, med_y, med_w, med_h = c["box"]
            is_small = (med_h / src_h) < 0.30
            cx, cy = (med_x + med_w / 2) / src_w, (med_y + med_h / 2) / src_h
            is_off_center = cx < 0.30 or cx > 0.70 or cy < 0.30 or cy > 0.70
            # A real composited overlay is on screen essentially the whole
            # time, but "on screen" and "face detected" aren't the same
            # thing -- a streamer looking down, turning to their other
            # monitor, or just being poorly lit drops out of individual
            # detections even though their camera box never moves.
            # Requiring *most* samples to hit (the old 0.5 bar) meant a
            # co-stream with 2-3 overlays only classified correctly when
            # every single one of them happened to be well-detected in the
            # same clip -- in practice, one weak detector out of three was
            # enough to silently drop that person's tile and change the
            # whole layout for that clip. Use the same "at least ~1/3 of
            # samples" bar as the single-face anchor above instead: still
            # well above one-off noise, but not so strict that ordinary
            # looking-away moments defeat it.
            is_recurring = c["occurrence_frac"] >= min_confident_occurrence
            return is_small and is_off_center and is_recurring

        overlay_clusters = [c for c in clusters if is_overlay(c)]
        cams = overlay_clusters
        if len(cams) > MAX_COCAM_TILES:
            # More than this is almost always detector noise rather than a
            # real 4+-way co-stream -- keep whichever recurred most
            # consistently across the sampled frames.
            cams = sorted(cams, key=lambda c: c["occurrence_frac"], reverse=True)[:MAX_COCAM_TILES]
        layout = _build_overlay_layout(
            [c["box"] for c in cams], src_w, src_h, target_w, target_h, facecam_height_frac
        )
        if layout is not None:
            _log_layout(start, end, clusters, f"{len(cams)}-cam split (heuristic)")
            return layout

    # No stable small/off-center overlay -- a large and/or roughly centered
    # face (talking head, interview, explainer video), a face that moves
    # around the frame (a real scene, not an overlay), or noisy/unreliable
    # detections (fast-moving handheld footage). Anchor on the most
    # consistently-detected face if there's one worth trusting; otherwise
    # a plain center crop beats guessing from a one-off detection.
    if has_confident_anchor:
        _log_layout(start, end, clusters, "face-anchored crop (no overlay pattern)")
        return _face_centered_crop(src_w, src_h, anchor_x, target_w, target_h)
    _log_layout(start, end, clusters, "center crop (no confident anchor)")
    return _center_crop(src_w, src_h, target_w, target_h)
