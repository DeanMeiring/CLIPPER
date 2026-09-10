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
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


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
    reading order; render_clip tiles them across out_w in that order using
    the same width split (see _tile_widths)."""
    top: CropWindow
    bottom_cams: List[CropWindow]
    top_out_h: int
    bottom_out_h: int


Layout = Union[CropWindow, SplitLayout, MultiCamSplitLayout]

# Detected overlay clusters beyond this are almost always detector noise
# (Haar false-positives), not a real 4+-way co-stream -- and even a genuine
# one would make each tile too small to be worth showing. Cap at 3 and,
# when there are more candidates than that, keep the ones detected most
# consistently (see compute_layout).
MAX_COCAM_TILES = 3


def _tile_widths(total_w: int, n: int) -> List[int]:
    """Split total_w into n integer widths that sum exactly to total_w --
    used to size both the source-side facecam crops (so each one scales
    cleanly to its output tile with no distortion) and the output-side
    scale/hstack in render_clip, which must agree on the same split."""
    base = total_w // n
    widths = [base] * n
    widths[-1] = total_w - base * (n - 1)  # remainder absorbed by the last tile
    return widths


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
        # Loosened from the original scaleFactor=1.15/minNeighbors=5/
        # minSize=(60,60): confirmed in practice that a facecam genuinely
        # on screen the whole clip was still only detected in 1-2 of 9
        # samples with those settings -- likely a smaller/angled/lower-
        # contrast facecam (a co-stream tile is often smaller than a
        # single streamer's main cam, and people often aren't looking
        # square at their webcam). Finer scale steps and a smaller minSize
        # catch more of those; the recurrence-based occurrence_frac check
        # downstream (not raw per-frame detection) is what actually guards
        # against one-off false positives hijacking a crop, so loosening
        # the per-frame detector doesn't reopen that -- it just gives the
        # recurrence check more real detections to work with.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(35, 35))
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


def _face_centered_crop(src_w: int, src_h: int, center_x: float, target_w_ratio: int, target_h_ratio: int) -> CropWindow:
    base = _center_crop(src_w, src_h, target_w_ratio, target_h_ratio)
    x = int(round(center_x - base.w / 2))
    x = max(0, min(x, src_w - base.w))
    return CropWindow(x=x, y=base.y, w=base.w, h=base.h)


def _facecam_crop(src_w: int, src_h: int, box, out_w: int, out_h: int, pad: float = 1.7) -> CropWindow:
    """Crop around a padded face box at the exact aspect ratio needed to
    scale cleanly to (out_w, out_h) with no distortion."""
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
    samples: int = 9,
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

    def is_overlay(c: dict) -> bool:
        med_x, med_y, med_w, med_h = c["box"]
        is_small = (med_h / src_h) < 0.30
        cx, cy = (med_x + med_w / 2) / src_w, (med_y + med_h / 2) / src_h
        is_off_center = cx < 0.30 or cx > 0.70 or cy < 0.30 or cy > 0.70
        # A real composited overlay is on screen essentially the whole
        # time, but "on screen" and "face detected" aren't the same thing
        # -- a streamer looking down, turning to their other monitor, or
        # just being poorly lit drops out of individual detections even
        # though their camera box never moves. Requiring *most* samples
        # to hit (the old 0.5 bar) meant a co-stream with 2-3 overlays
        # only classified correctly when every single one of them
        # happened to be well-detected in the same clip -- in practice,
        # one weak detector out of three was enough to silently drop that
        # person's tile and change the whole layout for that clip. Use
        # the same "at least ~1/3 of samples" bar as the single-face
        # anchor above instead: still well above one-off noise, but not
        # so strict that ordinary looking-away moments defeat it.
        is_recurring = c["occurrence_frac"] >= min_confident_occurrence
        return is_small and is_off_center and is_recurring

    overlay_clusters = [c for c in clusters if is_overlay(c)]

    if len(overlay_clusters) >= 2:
        # Two or more genuine facecam overlays -- a duo/co-stream layout.
        # There's no principled way to pick which one "matters" (and no
        # per-speaker audio to infer who's actually talking from), so show
        # all of them, tiled left-to-right across the bottom band, instead
        # of guessing at one or giving up and showing none.
        cams = overlay_clusters
        if len(cams) > MAX_COCAM_TILES:
            # More than this is almost always detector noise rather than a
            # real 4+-way co-stream -- keep whichever recurred most
            # consistently across the sampled frames.
            cams = sorted(cams, key=lambda c: c["occurrence_frac"], reverse=True)[:MAX_COCAM_TILES]
        cams.sort(key=lambda c: c["box"][0])  # left-to-right on-screen order

        bottom_out_h = round(target_h * facecam_height_frac)
        top_out_h = target_h - bottom_out_h
        tile_widths = _tile_widths(target_w, len(cams))
        bottom_cams = [
            _facecam_crop(src_w, src_h, c["box"], w, bottom_out_h)
            for c, w in zip(cams, tile_widths)
        ]
        top = _center_crop(src_w, src_h, target_w, top_out_h)
        _log_layout(start, end, clusters, f"{len(cams)}-cam co-stream split")
        return MultiCamSplitLayout(top=top, bottom_cams=bottom_cams, top_out_h=top_out_h, bottom_out_h=bottom_out_h)

    if len(overlay_clusters) == 1:
        # Exactly one small, corner-positioned, stationary face -- a
        # facecam overlay on top of gameplay/screen content. Build a
        # stacked split layout.
        bottom_out_h = round(target_h * facecam_height_frac)
        top_out_h = target_h - bottom_out_h
        bottom = _facecam_crop(src_w, src_h, overlay_clusters[0]["box"], target_w, bottom_out_h)
        top = _center_crop(src_w, src_h, target_w, top_out_h)
        _log_layout(start, end, clusters, "1-cam facecam split")
        return SplitLayout(top=top, bottom=bottom, top_out_h=top_out_h, bottom_out_h=bottom_out_h)

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
