"""Work out a 9:16 (or other target ratio) crop window for a clip.

Samples a handful of frames across the clip, runs OpenCV's built-in Haar
cascade face detector on each (ships with opencv-python, no extra download),
and centers the crop on the median face position across samples so the crop
doesn't jitter clip-to-clip. Falls back to a plain center crop when no faces
are found (title cards, screen-share footage, b-roll, etc).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class CropWindow:
    x: int
    y: int
    w: int
    h: int


def compute_crop_window(
    video_path: Path,
    start: float,
    end: float,
    target_ratio: Tuple[int, int] = (9, 16),
    samples: int = 6,
) -> CropWindow:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w_ratio, target_h_ratio = target_ratio
    # Crop full height, narrow the width to hit the target ratio (typical
    # landscape source -> vertical clip case). If the source is already
    # narrower than the target ratio, crop height instead.
    if src_w / src_h > target_w_ratio / target_h_ratio:
        crop_h = src_h
        crop_w = int(crop_h * target_w_ratio / target_h_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w * target_h_ratio / target_w_ratio)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    centers_x: List[float] = []
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
        # weight by face area so the biggest (closest) face dominates
        biggest = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = biggest
        centers_x.append(fx + fw / 2)

    cap.release()

    if centers_x:
        center_x = statistics.median(centers_x)
    else:
        center_x = src_w / 2  # fallback: plain center crop

    x = int(round(center_x - crop_w / 2))
    x = max(0, min(x, src_w - crop_w))
    y = max(0, (src_h - crop_h) // 2)

    return CropWindow(x=x, y=y, w=crop_w, h=crop_h)
