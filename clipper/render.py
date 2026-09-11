"""Cut, crop, and burn captions into a single output clip via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .reframe import CropWindow, Layout, MultiCamSplitLayout, SplitLayout


def _escape_for_filter(path: Path) -> str:
    """ffmpeg filtergraph args treat : and \\ specially -- this keeps
    Windows paths (C:\\Users\\...) working inside -vf/-filter_complex."""
    s = str(path).replace("\\", "/")
    s = s.replace(":", "\\:")
    return s


def _letterbox_scale_pad(w: int, h: int) -> str:
    """Scale a facecam crop to fit within w x h without distorting it,
    padding any leftover space with black bars instead of stretching to
    fill it exactly.

    A narrow multi-cam tile's aspect ratio (out_w split 2-3 ways against
    a fixed band height) doesn't match any real facecam window's -- the
    crop side of this used to force an exact-aspect match, which either
    pulled surrounding gameplay into the tile or squashed the person's
    face with a severe non-uniform stretch (measured on a real rejected
    render: height stretched 2.6x more than width). A small letterboxed
    thumbnail reads far better than either."""
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"


def render_clip(
    source_video: Path,
    start: float,
    end: float,
    layout: Layout,
    ass_path: Path,
    output_path: Path,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    ass = _escape_for_filter(ass_path)

    if isinstance(layout, MultiCamSplitLayout):
        top = layout.top
        parts = [f"[0:v]crop={top.w}:{top.h}:{top.x}:{top.y},scale={out_w}:{layout.top_out_h}[top];"]
        tile_labels = []
        for i, (cam, tw) in enumerate(zip(layout.bottom_cams, layout.bottom_cam_out_widths)):
            label = f"cam{i}"
            parts.append(
                f"[0:v]crop={cam.w}:{cam.h}:{cam.x}:{cam.y},"
                f"{_letterbox_scale_pad(tw, layout.bottom_out_h)}[{label}];"
            )
            tile_labels.append(f"[{label}]")
        parts.append(f"{''.join(tile_labels)}hstack=inputs={len(tile_labels)}[bottom];")
        parts.append("[top][bottom]vstack=inputs=2[stacked];")
        parts.append(f"[stacked]ass='{ass}'[outv]")
        filter_complex = "".join(parts)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    elif isinstance(layout, SplitLayout):
        top, bottom = layout.top, layout.bottom
        filter_complex = (
            f"[0:v]crop={top.w}:{top.h}:{top.x}:{top.y},scale={out_w}:{layout.top_out_h}[top];"
            f"[0:v]crop={bottom.w}:{bottom.h}:{bottom.x}:{bottom.y},"
            f"{_letterbox_scale_pad(out_w, layout.bottom_out_h)}[bottom];"
            f"[top][bottom]vstack=inputs=2[stacked];"
            f"[stacked]ass='{ass}'[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        crop: CropWindow = layout
        vf = (
            f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
            f"scale={out_w}:{out_h},"
            f"ass='{ass}'"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_video),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
            str(output_path),
        ]

    # A hard wall-clock ceiling: this runs on the web app's single sequential
    # worker thread, and the cancel signal is only checked between pipeline
    # steps, not inside a blocking subprocess call -- an ffmpeg hang (a
    # malformed/truncated source, an unusual codec, a filter stall) would
    # otherwise wedge that thread, and every future queued job, forever.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"ffmpeg timed out after {e.timeout:.0f}s rendering {output_path.name} "
            "-- source video may be corrupt or an unusual codec"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path.name}:\n{result.stderr[-2000:]}")
    return output_path


def trim_clip(
    source_path: Path, output_path: Path, trim_start: float, trim_end: float, duration: float,
) -> Path:
    """Cut trim_start seconds off the front and trim_end seconds off the
    back of an ALREADY-RENDERED clip -- for trimming a beat or two before
    posting it, not part of the original render.

    Captions are burned directly into the pixels at render time (there's
    no separate subtitle track), so a plain cut of the finished video
    carries its captions along automatically -- nothing needs re-syncing
    the way it would if this touched the source video and layout instead."""
    new_duration = duration - trim_start - trim_end
    if trim_start < 0 or trim_end < 0:
        raise ValueError("Trim amounts can't be negative")
    if new_duration < 1.0:
        raise ValueError("That trim would leave less than a second of clip")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{trim_start:.3f}",
        "-i", str(source_path),
        "-t", f"{new_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {e.timeout:.0f}s trimming {source_path.name}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed trimming {source_path.name}:\n{result.stderr[-2000:]}")
    return output_path
