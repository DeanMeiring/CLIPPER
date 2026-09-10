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
    reading order. bottom_cam_out_widths gives each tile's own output
    width (a "justified row" split sized from each cam's own aspect
    ratio -- see _build_overlay_layout -- not an equal split), and
    render_clip tiles them across out_w in that order using those
    widths."""
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
    """The "gameplay" region's crop for a split layout, restricted to
    whichever vertical band (above or below all the facecam boxes) has
    more room -- overlays are virtually always along one edge, so this
    covers the common cases without needing to know which edge.

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
    if not boxes:
        return _center_crop(src_w, src_h, target_w, target_h)
    min_y = min(b[1] for b in boxes)
    max_y = max(b[1] + b[3] for b in boxes)
    room_above, room_below = min_y, src_h - max_y
    band_y, band_h = (0, room_above) if room_above >= room_below else (max_y, room_below)
    if band_h < src_h * 0.15:
        return _center_crop(src_w, src_h, target_w, target_h)
    crop = _center_crop(src_w, band_h, target_w, target_h)
    return CropWindow(x=crop.x, y=band_y + crop.y, w=crop.w, h=crop.h)


def _face_centered_crop(src_w: int, src_h: int, center_x: float, target_w_ratio: int, target_h_ratio: int) -> CropWindow:
    base = _center_crop(src_w, src_h, target_w_ratio, target_h_ratio)
    x = int(round(center_x - base.w / 2))
    x = max(0, min(x, src_w - base.w))
    return CropWindow(x=x, y=base.y, w=base.w, h=base.h)


def _facecam_crop(src_w: int, src_h: int, box, pad: float = 1.7) -> CropWindow:
    """Crop around a padded face box at its OWN natural aspect ratio --
    not forced to match the output tile's aspect.

    This used to expand the crop to exactly match the tile's aspect
    ratio so ffmpeg's scale wouldn't distort it. That works fine for a
    single full-width tile (its aspect is already close to a real
    webcam window's), but a 2-3 way tile is much narrower against the
    same band height -- forcing a real (typically landscape-ish)
    facecam box to that portrait tile aspect meant either ballooning
    the crop far past the box into surrounding gameplay, or (once that
    expansion was capped) squashing the image with a severe non-uniform
    stretch -- measured on a real rejected render: crop 248x204 scaled
    to a 360x768 tile stretched height 2.6x more than width, visibly
    warped. Neither is fixable by tuning this crop alone.

    render_clip now scales this crop to fit *within* its output tile
    (preserving aspect) and pads any leftover space with black bars
    instead of stretching to fill it exactly -- so this just needs to
    return a clean, undistorted crop around the actual facecam window,
    and letterboxing handles the rest."""
    fx, fy, fw, fh = box
    cx, cy = fx + fw / 2, fy + fh / 2

    crop_w = min(fw * pad, src_w)
    crop_h = min(fh * pad, src_h)

    x = int(round(cx - crop_w / 2))
    y = int(round(cy - crop_h / 2))
    x = max(0, min(x, src_w - int(crop_w)))
    y = max(0, min(y, src_h - int(crop_h)))
    return CropWindow(x=x, y=y, w=int(crop_w), h=int(crop_h))


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

    `pad` should match what kind of box this is: the Haar heuristic only
    ever detects a face, tightly, so it needs generous padding (the
    default, 1.7x) to reach the edges of the actual webcam window around
    it. A vision-detected box is already asked to cover the *whole*
    visible facecam window, not just the face -- padding that again by
    1.7x overshoots well past the window into the surrounding gameplay,
    which is exactly what "it's grabbing facecam and gameplay in the same
    crop, not a clean facecam" turned out to be. Callers with
    vision-sourced boxes should pass a much tighter pad instead."""
    if not boxes:
        return None
    boxes = sorted(boxes, key=lambda b: b[0])
    aspects = [bw / bh for (_, _, bw, bh) in boxes]

    # "Justified row" layout -- the same technique photo grids (Google
    # Photos, Flickr) use to tile images of different aspect ratios into
    # one row with no cropping and no padding: pick the one row height at
    # which each image's own aspect ratio, laid out at full width for
    # that height, sums to exactly target_w. Different people's facecam
    # windows almost never share an aspect ratio, and forcing them into
    # equal-width tiles at a shared height (the previous approach, no
    # matter how that height was chosen) always left at least one tile
    # with a mismatched aspect -- fixed via letterboxing/cropping/
    # stretching, all of which failed post-render checks in practice.
    # This solves it exactly instead of approximating it: every tile
    # comes out fully filled by construction, whatever the mix of
    # aspects, because the widths are derived FROM those aspects rather
    # than assumed equal.
    # Floor low enough that an ordinary 3-way collab still fits exactly:
    # three 16:9 cams want a ~202px band, and a floor above that forces
    # letterboxing on the single most common multi-cam setup there is.
    # At 1080 wide that still leaves each of three tiles ~360x200, which
    # is a perfectly legible face thumbnail.
    min_bottom_h = round(target_h * 0.10)
    max_bottom_h = round(target_h * facecam_height_frac)
    natural_bottom_h = target_w / sum(aspects)
    bottom_out_h = int(round(min(max(natural_bottom_h, min_bottom_h), max_bottom_h)))
    top_out_h = target_h - bottom_out_h
    top = _top_crop_excluding_overlays(src_w, src_h, boxes, target_w, top_out_h)
    if len(boxes) == 1:
        bottom = _facecam_crop(src_w, src_h, boxes[0], pad=pad)
        return SplitLayout(top=top, bottom=bottom, top_out_h=top_out_h, bottom_out_h=bottom_out_h)
    bottom_cams = [_facecam_crop(src_w, src_h, b, pad=pad) for b in boxes]
    # Split target_w in proportion to the aspect ratios, rather than as
    # bottom_out_h * aspect. The two are identical whenever bottom_out_h
    # is its natural (unclamped) value, but deriving from target_w keeps
    # the widths summing to it even when the height hit a clamp: three
    # 16:9 cams -- an entirely ordinary 3-way collab -- push the natural
    # height under the floor, and the old form then left the last tile
    # absorbing a badly wrong remainder (36% off its aspect, and outright
    # NEGATIVE for ultrawide cams, which would fail the ffmpeg render and
    # error the whole job). This form is proportional and always
    # positive; a clamped height just means each tile letterboxes a
    # little, which render_clip already handles.
    raw_widths = [target_w * a / sum(aspects) for a in aspects]
    tile_out_widths = [max(1, int(round(w))) for w in raw_widths[:-1]]
    tile_out_widths.append(target_w - sum(tile_out_widths))  # remainder absorbed by the last tile
    return MultiCamSplitLayout(
        top=top, bottom_cams=bottom_cams, top_out_h=top_out_h, bottom_out_h=bottom_out_h,
        bottom_cam_out_widths=tile_out_widths,
    )


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
        # vision was asked for the whole facecam window already, not just
        # a face, so padding it again by 1.7x was overshooting well past
        # the window into the surrounding gameplay ("taking a snippet of
        # their facecam AND gameplay" instead of a clean facecam crop).
        # Just enough margin to avoid a razor-tight crop cutting into the
        # window's own edge/border.
        layout = _build_overlay_layout(
            vision_boxes[:MAX_COCAM_TILES], src_w, src_h, target_w, target_h, facecam_height_frac,
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
