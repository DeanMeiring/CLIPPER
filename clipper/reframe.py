"""Work out a 9:16 (or other target ratio) layout for a clip.

Samples a handful of frames across the clip and runs OpenCV's built-in Haar
cascade face detector on each (ships with opencv-python, no extra download).

Two outcomes:
  - No face, or a large/roughly-centered face (talking head, interview,
    explainer video): a single CropWindow centered on the face (or a plain
    center crop if no face was found at all).
  - A small, corner-positioned face (a streamer's webcam overlay sitting on
    top of gameplay footage): a SplitLayout with gameplay on top and a
    zoomed-in facecam crop on the bottom, stacked to fill the vertical frame
    -- the standard layout real clip channels use, instead of zooming into
    just the tiny facecam box and losing all the game context.
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


Layout = Union[CropWindow, SplitLayout]


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

    boxes: List[Tuple[int, int, int, int]] = []
    duration = max(end - start, 0.1)
    for i in range(samples):
        t = start + duration * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            continue
        biggest = max(faces, key=lambda f: f[2] * f[3])
        boxes.append(tuple(int(v) for v in biggest))

    cap.release()
    return boxes, src_w, src_h


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


def compute_layout(
    video_path: Path,
    start: float,
    end: float,
    target_w: int = 1080,
    target_h: int = 1920,
    samples: int = 6,
    facecam_height_frac: float = 0.40,
) -> Layout:
    boxes, src_w, src_h = _detect_faces(video_path, start, end, samples)

    if not boxes:
        return _center_crop(src_w, src_h, target_w, target_h)

    med_x = statistics.median(b[0] for b in boxes)
    med_y = statistics.median(b[1] for b in boxes)
    med_w = statistics.median(b[2] for b in boxes)
    med_h = statistics.median(b[3] for b in boxes)
    med_box = (med_x, med_y, med_w, med_h)

    face_center_x = med_x + med_w / 2
    face_center_y = med_y + med_h / 2

    # A source that's already portrait/vertical (a phone-held IRL stream,
    # anything already shot roughly 9:16) is one continuous scene -- not a
    # landscape gameplay feed with a small facecam window composited on
    # top of it. The gameplay+facecam split below exists specifically to
    # catch that overlay pattern, and misfires here: an IRL streamer's
    # face is naturally small and/or off-center plenty of the time just
    # from them moving around, and splitting the frame on that would zoom
    # into a chunk of it as a fake "facecam" while throwing away the rest
    # of the actual scene. Always give a portrait source a single crop of
    # the whole frame instead.
    if src_h >= src_w:
        return _face_centered_crop(src_w, src_h, face_center_x, target_w, target_h)

    is_small = (med_h / src_h) < 0.30
    x_frac, y_frac = face_center_x / src_w, face_center_y / src_h
    is_off_center = x_frac < 0.30 or x_frac > 0.70 or y_frac < 0.30 or y_frac > 0.70

    # A real composited facecam overlay sits at a fixed screen position
    # for the whole stream; a real face in one continuous scene (e.g. a
    # landscape IRL stream where the streamer just happens to be off to
    # one side) moves around across the sampled frames. Only treat this
    # as an overlay -- worth splitting out from the rest of the frame --
    # if the face barely moves across the samples; otherwise this is the
    # actual scene, not a gameplay+facecam composite.
    centers_x = [(b[0] + b[2] / 2) / src_w for b in boxes]
    centers_y = [(b[1] + b[3] / 2) / src_h for b in boxes]
    position_is_stable = (
        statistics.pstdev(centers_x) < 0.03 and statistics.pstdev(centers_y) < 0.03
    )

    if not (is_small and is_off_center and position_is_stable):
        # Large face, roughly centered, and/or moving around the frame --
        # talking head, interview, explainer video, or a real scene the
        # person just isn't dead-center in. Single-crop behavior handles
        # all of these well.
        return _face_centered_crop(src_w, src_h, face_center_x, target_w, target_h)

    # Small, corner-positioned, stationary face -- a facecam overlay on top
    # of gameplay/screen content. Build a stacked split layout.
    bottom_out_h = round(target_h * facecam_height_frac)
    top_out_h = target_h - bottom_out_h

    bottom = _facecam_crop(src_w, src_h, med_box, target_w, bottom_out_h)
    top = _center_crop(src_w, src_h, target_w, top_out_h)

    return SplitLayout(top=top, bottom=bottom, top_out_h=top_out_h, bottom_out_h=bottom_out_h)
